"""
OmniCLI heartbeat.

Two concerns, kept separate:
  - Built-in health checks (status): per-provider connectivity, latency, model
    count, with an optional canary prompt. This is the DEFAULT behavior.
  - A user task file (increment D): user-defined tasks run on a tick. Health
    checks are NOT part of that file.

Results can be written to a status file so other tools can read current health.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_heartbeat(
    cfg: Any,
    client: Any,
    canary: bool = False,
    providers: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Per-provider health record. Reuses client.ping(); adds a canary when asked."""
    ids = providers or list(cfg.data.get("providers", {}).keys())
    records: List[Dict[str, Any]] = []
    for pid in ids:
        rec = dict(client.ping(pid))
        rec["provider"] = pid
        if canary and rec.get("status") == "healthy":
            rec["canary"] = _canary(cfg, client, pid)
        records.append(rec)
    return records


def _canary(cfg: Any, client: Any, provider: str) -> Dict[str, Any]:
    """Send a tiny prompt to confirm the model actually responds, not just the endpoint."""
    model = cfg.get_default_model(provider)
    t0 = time.perf_counter()
    try:
        res = client.chat_complete(
            provider_id=provider,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            model=model,
            max_tokens=5,
        )
        content = ((res.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "") or ""
        return {
            "ok": "ok" in content.lower(),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "reply": content.strip()[:40],
        }
    except Exception as e:  # noqa: BLE001 - a canary failure is a result, not a crash
        return {"ok": False, "error": str(e)[:120]}


def write_status(
    cfg: Any,
    records: List[Dict[str, Any]],
    tasks: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Write heartbeat records (+ optional task results, + timestamp) to the status file."""
    path_s = cfg.get_heartbeat_config().get("status_file")
    if not path_s:
        return None
    path = Path(str(path_s)).expanduser()
    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers": records,
    }
    if tasks is not None:
        payload["tasks"] = tasks
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)
    except Exception as e:  # noqa: BLE001
        print(f"[heartbeat] WARNING: could not write status {path}: {e}")
        return None


# ── Task runner (increment D) ──────────────────────────────────────────────────
#
# Health checks above are the default. The task file is USER-POPULATED: a JSON
# list (or {"tasks": [...]}) of {"type": "ask|rescan|agent", ...params}. v1 is a
# minimal, extensible dispatcher — unknown types are recorded as errors, never
# fatal. Task shapes may change in a follow-up spec.

def load_tasks(cfg: Any) -> List[Dict[str, Any]]:
    """Read the configured heartbeat tasks file. Missing/invalid → []."""
    path_s = cfg.get_heartbeat_config().get("tasks_file")
    if not path_s:
        return []
    path = Path(str(path_s)).expanduser()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[heartbeat] WARNING: could not parse tasks file {path}: {e}")
        return []
    if isinstance(data, dict):
        data = data.get("tasks", [])
    return [t for t in data if isinstance(t, dict)]


def run_tasks(cfg: Any, client: Any, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dispatch each task by type; collect results. One bad task never aborts the rest."""
    results: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks):
        ttype = str(task.get("type", "")).lower()
        try:
            if ttype == "ask":
                results.append(_task_ask(cfg, client, task))
            elif ttype == "rescan":
                results.append(_task_rescan(cfg, task))
            elif ttype == "agent":
                results.append(_task_agent(cfg, task))
            else:
                results.append({"task": i, "type": ttype, "ok": False,
                                "error": f"unknown task type '{ttype}'"})
        except Exception as e:  # noqa: BLE001
            results.append({"task": i, "type": ttype, "ok": False, "error": str(e)[:160]})
    return results


def _task_ask(cfg: Any, client: Any, task: Dict[str, Any]) -> Dict[str, Any]:
    """Send a prompt to a provider; optional 'expect' substring sets ok/fail."""
    provider = str(task.get("provider") or cfg.get_default_provider()).lower()
    prompt = str(task.get("prompt", ""))
    model = task.get("model") or cfg.get_default_model(provider)
    expect = task.get("expect")
    res = client.chat_complete(
        provider_id=provider,
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=int(task.get("max_tokens", 256)),
    )
    content = ((res.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "") or ""
    ok = True if expect is None else (str(expect).lower() in content.lower())
    return {"type": "ask", "provider": provider, "ok": ok, "reply": content.strip()[:200]}


def _task_rescan(cfg: Any, task: Dict[str, Any]) -> Dict[str, Any]:
    """Re-scan one provider or all providers to refresh rosters."""
    from omnicli.scanner import scan_all_providers, scan_provider
    provider = task.get("provider")
    if provider:
        r = scan_provider(str(provider).lower(), cfg=cfg)
        return {"type": "rescan", "provider": provider, "ok": r.get("ok", False),
                "models": r.get("model_count", 0)}
    results = scan_all_providers(cfg=cfg)
    return {"type": "rescan", "ok": all(r.get("ok") for r in results), "providers": len(results)}


def _task_agent(cfg: Any, task: Dict[str, Any]) -> Dict[str, Any]:
    """Run an autonomous agent goal (heavyweight). Shell stays off unless set."""
    from omnicli.agent import AgentRunner
    runner = AgentRunner(
        provider=task.get("provider"),
        model=task.get("model"),
        max_turns=int(task.get("max_turns", 5)),
        redteam=bool(task.get("redteam", False)),
        rotate_families=bool(task.get("rotate_families", False)),
        allow_shell=bool(task.get("allow_shell", False)),
        allow_local=bool(task.get("allow_local", False)),
    )
    answer = runner.run(str(task.get("goal", "")))
    return {"type": "agent", "ok": True, "answer": (answer or "")[:300]}
