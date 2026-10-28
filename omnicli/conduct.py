"""Oracle pipeline conductor for OmniCLI.

OmniCLI does not re-implement Oracle's scope, mindsets, or product gates.
It runs the same canonical command chain an operator would run:

    prescan --sync  ->  scan -N 1 --download  ->  queue --execute

and then verifies every product bundle the queue produced. The conductor is
successful only when the chain succeeds AND at least one bundle is both
integrity-clean and submission-ready; a chain that runs but mints no product
is ``no-product``, and mixed bundles are ``partial``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PIPELINE_SCHEMA = "omnicli.oracle.pipeline/v1"
Runner = Callable[..., subprocess.CompletedProcess]


class ConductorError(RuntimeError):
    """The conductor cannot start or parse a stage result."""


_SECRET_RE = re.compile(
    r"(?i)(authorization|cookie|x-api-key|api[_-]?token)\s*[:=]\s*\S+"
)


def _redact(text: str, limit: int = 20000) -> str:
    text = _SECRET_RE.sub(r"\1: [REDACTED]", str(text or ""))
    if len(text) > limit:
        return text[-limit:]
    return text


def _newest(root: Path, pattern: str, since: float) -> Optional[Path]:
    if not root.is_dir():
        return None
    found = []
    for path in root.glob(pattern):
        try:
            if path.is_file() and path.stat().st_mtime >= since:
                found.append(path)
        except OSError:
            continue
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def _load_state(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _bundle_for_workspace(workspace: Path, since: float) -> Optional[Path]:
    reports = workspace / "reports"
    if not reports.is_dir():
        return None
    found = []
    for path in reports.glob("product_*.zip"):
        try:
            if path.is_file() and path.stat().st_mtime >= since:
                found.append(path)
        except OSError:
            continue
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def _discover_bundles(
    oracle_root: Path, state_path: Optional[Path], since: float
) -> List[Dict[str, str]]:
    if state_path is not None:
        state = _load_state(state_path)
        items = state.get("items")
        if isinstance(items, dict):
            out = []
            for record in items.values():
                if not isinstance(record, dict):
                    continue
                workspace = record.get("workspace")
                raw_path = (
                    workspace.get("path") if isinstance(workspace, dict) else None
                )
                if not raw_path:
                    result = record.get("result")
                    raw_path = (
                        result.get("workspace") if isinstance(result, dict) else None
                    )
                if not raw_path:
                    continue
                ws = Path(str(raw_path)).expanduser()
                zip_path = _bundle_for_workspace(ws, since)
                if zip_path is not None:
                    out.append({"workspace": str(ws), "bundle": str(zip_path)})
            if out:
                return out
    # Fallback only when the queue state is absent/unreadable: scan target
    # workspaces for bundles created during this conductor run.
    out = []
    targets = oracle_root / "targets"
    if targets.is_dir():
        for zip_path in targets.glob("*/*/reports/product_*.zip"):
            try:
                if zip_path.is_file() and zip_path.stat().st_mtime >= since:
                    out.append(
                        {
                            "workspace": str(zip_path.parents[1]),
                            "bundle": str(zip_path),
                        }
                    )
            except OSError:
                continue
    return out


def _parse_verify(stdout: str) -> Dict[str, Any]:
    try:
        data = json.loads(stdout)
    except ValueError as exc:
        raise ConductorError(f"verify-download did not return JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConductorError("verify-download JSON is not an object")
    return data


def run_pipeline(
    oracle_root: Path,
    *,
    limit: int = 50,
    count: int = 1,
    timeout: int = 3600,
    runner: Optional[Runner] = None,
) -> Dict[str, Any]:
    """Run the canonical Oracle chain and verify the product bundles."""
    root = Path(oracle_root).expanduser().resolve()
    entry = root / "oracle_hunter.py"
    if not entry.is_file():
        raise ConductorError(f"oracle_hunter.py not found under {root}")
    if limit < 1:
        raise ConductorError(f"limit must be >= 1, got {limit!r}")
    if count < 1:
        raise ConductorError(f"count must be >= 1, got {count!r}")
    if timeout < 1:
        raise ConductorError(f"timeout must be >= 1, got {timeout!r}")
    run = runner or subprocess.run
    started = time.time()
    base = [sys.executable, str(entry)]
    stages_spec = [
        ("prescan", ["prescan", "--sync", "--limit", str(limit)]),
        ("scan", ["scan", "-N", str(count), "--download"]),
        ("queue", ["queue", "--execute"]),
    ]
    stages: List[Dict[str, Any]] = []
    for name, stage_args in stages_spec:
        cmd = base + stage_args
        started_stage = time.monotonic()
        record: Dict[str, Any] = {
            "name": name,
            "argv": stage_args,
            "status": "failed",
        }
        try:
            cp = run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            record["status"] = "timeout"
            record["duration_s"] = round(time.monotonic() - started_stage, 3)
            record["stdout_tail"] = _redact(
                exc.stdout if isinstance(exc.stdout, str) else ""
            )
            record["stderr_tail"] = _redact(
                exc.stderr if isinstance(exc.stderr, str) else ""
            )
            stages.append(record)
            return {
                "_schema": PIPELINE_SCHEMA,
                "status": "failed",
                "ready_for_submission": False,
                "failed_stage": name,
                "error": f"{name} timed out after {timeout}s",
                "stages": stages,
                "bundles": [],
                "verified_bundles": [],
            }
        except OSError as exc:
            record["duration_s"] = round(time.monotonic() - started_stage, 3)
            record["error"] = f"{type(exc).__name__}: {exc}"
            stages.append(record)
            return {
                "_schema": PIPELINE_SCHEMA,
                "status": "failed",
                "ready_for_submission": False,
                "failed_stage": name,
                "error": record["error"],
                "stages": stages,
                "bundles": [],
                "verified_bundles": [],
            }
        record["exit_code"] = int(cp.returncode)
        record["duration_s"] = round(time.monotonic() - started_stage, 3)
        record["stdout_tail"] = _redact(getattr(cp, "stdout", ""))
        record["stderr_tail"] = _redact(getattr(cp, "stderr", ""))
        if cp.returncode != 0:
            stages.append(record)
            return {
                "_schema": PIPELINE_SCHEMA,
                "status": "failed",
                "ready_for_submission": False,
                "failed_stage": name,
                "error": f"{name} exited {cp.returncode}",
                "stages": stages,
                "bundles": [],
                "verified_bundles": [],
            }
        record["status"] = "ok"
        stages.append(record)

    state_path = _newest(root / "reports", "queue_state_*.json", started - 2)
    bundles = _discover_bundles(root, state_path, started - 2)
    verified: List[Dict[str, Any]] = []
    for bundle in bundles:
        cmd = base + ["verify-download", bundle["bundle"], "--json"]
        try:
            cp = run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result = _parse_verify(getattr(cp, "stdout", ""))
            result["_exit_code"] = int(cp.returncode)
        except (ConductorError, OSError, subprocess.TimeoutExpired) as exc:
            result = {
                "integrity_ok": False,
                "ready_for_submission": False,
                "status": "verify-error",
                "errors": [str(exc)],
                "reasons": [],
            }
        # Ready means all three gates agree: Oracle's JSON claims readiness,
        # the bundle is integrity-clean, and verify-download exited 0. A
        # ready claim from a failed or integrity-broken verify is not ready.
        verify_exit = result.get("_exit_code")
        integrity_ok = bool(result.get("integrity_ok"))
        errors = list(result.get("errors") or [])
        if verify_exit not in (0, None):
            errors.append(f"verify-download exited {verify_exit}")
        verified.append(
            {
                "workspace": bundle["workspace"],
                "bundle": bundle["bundle"],
                "integrity_ok": integrity_ok,
                "ready_for_submission": bool(
                    result.get("ready_for_submission")
                    and integrity_ok
                    and verify_exit == 0
                ),
                "status": result.get("status"),
                "errors": errors,
                "reasons": list(result.get("reasons") or []),
            }
        )
    ready = [b for b in verified if b["ready_for_submission"]]
    not_ready = [b for b in verified if not b["ready_for_submission"]]
    if not verified:
        status = "no-product"
        error = "queue completed but produced no product bundle"
    elif not ready:
        status = "no-product"
        error = "product bundles exist but none are submission-ready"
    elif not_ready:
        status = "partial"
        error = "some product bundles are not submission-ready"
    else:
        status = "complete"
        error = ""
    return {
        "_schema": PIPELINE_SCHEMA,
        "status": status,
        "ready_for_submission": status == "complete",
        "failed_stage": None,
        "error": error,
        "stages": stages,
        "state": str(state_path) if state_path else None,
        "bundles": bundles,
        "verified_bundles": verified,
    }
