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


def write_status(cfg: Any, records: List[Dict[str, Any]]) -> Optional[str]:
    """Write the heartbeat records (+ timestamp) to the configured status file."""
    path_s = cfg.get_heartbeat_config().get("status_file")
    if not path_s:
        return None
    path = Path(str(path_s)).expanduser()
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers": records,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)
    except Exception as e:  # noqa: BLE001
        print(f"[heartbeat] WARNING: could not write status {path}: {e}")
        return None
