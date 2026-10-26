"""
Per-provider memory for OmniCLI.

Read-only context injection: you edit memory files, OmniCLI composes them and
prepends the result to the model's system prompt. Each provider can point at
several files (relative to the configured memory dir, or absolute), so you can
layer shared + per-API context. A configured-but-missing file warns loudly and
is skipped — memory is additive and must not crash a session.

Persistent per-provider history (append/load transcripts) lives here too; it is
opt-in via config and used by the chat REPL.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

def _default_dir(cfg: Any) -> Path:
    """Memory lives under the manager's own config dir (standalone law)."""
    config_dir = getattr(cfg, "config_dir", None)
    if config_dir:
        return Path(config_dir) / "memory"
    from omnicli.config import _default_config_dir

    return _default_config_dir() / "memory"


def _base_dir(cfg: Any) -> Path:
    mem = cfg.get_memory_config()
    return Path(str(mem.get("dir") or _default_dir(cfg))).expanduser()


def _resolved_files(cfg: Any, provider: str) -> List[Path]:
    """Resolve the configured memory file names for a provider to absolute paths."""
    base = _base_dir(cfg)
    out: List[Path] = []
    for name in cfg.get_memory_files(provider):
        p = Path(str(name)).expanduser()
        if not p.is_absolute():
            p = base / p
        out.append(p)
    return out


def memory_file_count(cfg: Any, provider: str) -> int:
    """Number of configured memory files that actually exist (for UI display)."""
    if not cfg.get_memory_config().get("enabled"):
        return 0
    return sum(1 for p in _resolved_files(cfg, provider) if p.is_file())


def load_memory(cfg: Any, provider: str) -> str:
    """Return the composed memory text to prepend to the system prompt.

    Empty string when memory is disabled or no files resolve. Missing/unreadable
    files warn to stderr and are skipped.
    """
    if not cfg.get_memory_config().get("enabled"):
        return ""
    chunks: List[str] = []
    for p in _resolved_files(cfg, provider):
        if not p.is_file():
            print(f"[memory] WARNING: configured memory file missing: {p}", file=sys.stderr)
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:  # noqa: BLE001 - report and skip, don't crash the session
            print(f"[memory] WARNING: could not read {p}: {e}", file=sys.stderr)
            continue
        if text:
            chunks.append(f"# Memory: {p.name}\n{text}")
    return "\n\n".join(chunks)


# ── Persistent per-provider history (increment B) ──────────────────────────────

def history_path(cfg: Any, provider: str) -> Path:
    """Path to a provider's history file (memory.dir/history/<provider>.jsonl)."""
    return _base_dir(cfg) / "history" / f"{provider.lower()}.jsonl"


def _history_cfg(cfg: Any) -> Dict[str, Any]:
    return dict(cfg.get_memory_config().get("history") or {})


def append_history(cfg: Any, provider: str, role: str, content: str) -> None:
    """Append one message to a provider's history (no-op unless history enabled)."""
    if not _history_cfg(cfg).get("enabled"):
        return
    path = history_path(cfg, provider)
    try:
        # History can contain sensitive transcript content — keep it owner-only.
        os.makedirs(path.parent, mode=0o700, exist_ok=True)
        try:
            os.chmod(path.parent, 0o700)  # tighten even if the dir pre-existed
        except OSError:
            pass
        fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(json.dumps({"role": role, "content": content}) + "\n")
    except Exception as e:  # noqa: BLE001 - history is best-effort, never fatal
        print(f"[memory] WARNING: could not write history {path}: {e}", file=sys.stderr)


def load_history(cfg: Any, provider: str, max_turns: int = 0) -> List[Dict[str, str]]:
    """Load a provider's history, newest-capped to max_turns (0 = use config)."""
    hcfg = _history_cfg(cfg)
    if not hcfg.get("enabled"):
        return []
    cap = max_turns or int(hcfg.get("max_turns") or 40)
    path = history_path(cfg, provider)
    if not path.is_file():
        return []
    out: List[Dict[str, str]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and "role" in rec and "content" in rec:
                out.append({"role": str(rec["role"]), "content": str(rec["content"])})
    except Exception as e:  # noqa: BLE001
        print(f"[memory] WARNING: could not read history {path}: {e}", file=sys.stderr)
        return []
    return out[-cap:] if cap > 0 else out
