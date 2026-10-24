"""Dependency-aware execution of small declarative OmniCLI workflows."""

from __future__ import annotations

import concurrent.futures
import re
import threading
from typing import Any, Callable, Dict, List, Set


_TASK_FIELDS = {
    "id", "prompt", "depends_on", "writes", "provider", "model", "max_tokens",
    "ground", "files",
}
_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def validate_workflow(workflow: Any) -> List[Dict[str, Any]]:
    """Validate a workflow DAG and reject unordered writes before execution."""
    if not isinstance(workflow, dict) or not isinstance(workflow.get("tasks"), list):
        raise ValueError("Workflow must be an object containing a tasks array.")
    tasks = workflow["tasks"]
    if not tasks:
        raise ValueError("Workflow must contain at least one task.")

    by_id: Dict[str, Dict[str, Any]] = {}
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Workflow task at index {index} must be an object.")
        unknown = set(task) - _TASK_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown field(s) in task at index {index}: {', '.join(sorted(unknown))}"
            )
        task_id = task.get("id")
        if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
            raise ValueError(f"Workflow task at index {index} needs a valid id.")
        if task_id in by_id:
            raise ValueError(f"Duplicate workflow task id: {task_id}")
        prompt = task.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Workflow task {task_id} needs a non-empty prompt.")
        depends_on = task.get("depends_on", [])
        writes = task.get("writes", [])
        if (
            not isinstance(depends_on, list)
            or any(not isinstance(dep, str) or not dep for dep in depends_on)
            or len(set(depends_on)) != len(depends_on)
        ):
            raise ValueError(f"Workflow task {task_id} has invalid depends_on.")
        if (
            not isinstance(writes, list)
            or any(not isinstance(write, str) or not write.strip() for write in writes)
            or len(set(writes)) != len(writes)
        ):
            raise ValueError(f"Workflow task {task_id} has invalid writes.")
        max_tokens = task.get("max_tokens", 1024)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(f"Workflow task {task_id} max_tokens must be a positive integer.")
        for field in ("provider", "model"):
            value = task.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Workflow task {task_id} has an invalid {field}.")
        ground = task.get("ground", True)
        if not isinstance(ground, bool):
            raise ValueError(f"Workflow task {task_id} ground must be a boolean.")
        files = task.get("files", [])
        if (
            not isinstance(files, list)
            or any(not isinstance(item, str) or not item.strip() for item in files)
            or len(set(files)) != len(files)
        ):
            raise ValueError(f"Workflow task {task_id} has invalid files.")
        by_id[task_id] = {
            **task,
            "depends_on": depends_on,
            "writes": writes,
            "max_tokens": max_tokens,
            "ground": ground,
            "files": files,
        }

    for task_id, task in by_id.items():
        for dependency in task["depends_on"]:
            if dependency not in by_id:
                raise ValueError(
                    f"Workflow task {task_id} depends on unknown task {dependency}."
                )
            if dependency == task_id:
                raise ValueError(f"Workflow task {task_id} cannot depend on itself.")

    ancestors: Dict[str, Set[str]] = {}

    def find_ancestors(task_id: str, visiting: Set[str]) -> Set[str]:
        if task_id in visiting:
            raise ValueError("Workflow dependencies contain a cycle.")
        if task_id in ancestors:
            return ancestors[task_id]
        visiting.add(task_id)
        result: Set[str] = set()
        for dependency in by_id[task_id]["depends_on"]:
            result.add(dependency)
            result.update(find_ancestors(dependency, visiting))
        visiting.remove(task_id)
        ancestors[task_id] = result
        return result

    for task_id in by_id:
        find_ancestors(task_id, set())

    ids = list(by_id)
    for index, first_id in enumerate(ids):
        first_writes = set(by_id[first_id]["writes"])
        if not first_writes:
            continue
        for second_id in ids[index + 1:]:
            conflict = first_writes & set(by_id[second_id]["writes"])
            if (
                conflict
                and first_id not in ancestors[second_id]
                and second_id not in ancestors[first_id]
            ):
                raise ValueError(
                    f"Unordered workflow tasks {first_id} and {second_id} both write: "
                    f"{', '.join(sorted(conflict))}."
                )
    return [by_id[task["id"]] for task in tasks]


def run_workflow(
    workflow: Any,
    execute: Callable[[Dict[str, Any], Dict[str, str]], str],
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Run ready DAG nodes concurrently; failed prerequisites skip descendants."""
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer.")
    tasks = validate_workflow(workflow)
    by_id = {task["id"]: task for task in tasks}
    pending = set(by_id)
    states: Dict[str, str] = {}
    results: Dict[str, Dict[str, Any]] = {}
    events: List[Dict[str, Any]] = []
    event_lock = threading.Lock()

    def event(name: str, task_id: str, **details: Any) -> None:
        with event_lock:
            events.append({"sequence": len(events) + 1, "event": name, "task_id": task_id, **details})

    def run_one(task_id: str, task: Dict[str, Any], dependencies: Dict[str, str]) -> Dict[str, Any]:
        event("task_started", task_id)
        try:
            output = execute(task, dependencies)
            if not isinstance(output, str):
                raise TypeError("Workflow task executor must return text.")
            return {"status": "succeeded", "output": output}
        except Exception as exc:  # task failures are isolated to descendants
            return {"status": "failed", "error": str(exc)[:240]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        while pending:
            ready = sorted(
                task_id for task_id in pending
                if all(dependency in states for dependency in by_id[task_id]["depends_on"])
            )
            if not ready:
                raise RuntimeError("Workflow made no progress after DAG validation.")

            runnable = []
            for task_id in ready:
                task = by_id[task_id]
                failed_dependencies = [
                    dependency for dependency in task["depends_on"]
                    if states[dependency] != "succeeded"
                ]
                if failed_dependencies:
                    states[task_id] = "skipped"
                    results[task_id] = {
                        "status": "skipped",
                        "error": "Failed prerequisite(s): " + ", ".join(failed_dependencies),
                    }
                    pending.remove(task_id)
                    event("task_skipped", task_id, failed_dependencies=failed_dependencies)
                else:
                    runnable.append(task_id)

            futures = {}
            for task_id in runnable:
                task = by_id[task_id]
                dependency_outputs = {
                    dependency: results[dependency]["output"]
                    for dependency in task["depends_on"]
                }
                futures[pool.submit(run_one, task_id, task, dependency_outputs)] = task_id

            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                result = future.result()
                states[task_id] = result["status"]
                results[task_id] = result
                pending.remove(task_id)
                event(
                    "task_succeeded" if result["status"] == "succeeded" else "task_failed",
                    task_id,
                    **({"error": result["error"]} if "error" in result else {}),
                )

    succeeded = sum(state == "succeeded" for state in states.values())
    failed = sum(state == "failed" for state in states.values())
    skipped = sum(state == "skipped" for state in states.values())
    status = "completed" if failed == 0 and skipped == 0 else ("failed" if succeeded == 0 else "partial")
    return {
        "status": status,
        "summary": {"succeeded": succeeded, "failed": failed, "skipped": skipped},
        "tasks": {task["id"]: results[task["id"]] for task in tasks},
        "events": events,
    }
