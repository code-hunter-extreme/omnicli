"""Adjudication memory for the OmniCLI red-team judge.

Per-system + global scopes, JSONL-backed, with an in-memory mode for tests.
Writers must never crash the caller: file IO is guarded and falls back to
in-memory operation with a stderr warning. No network. Original code.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional

_STR_LIT_RE = re.compile(r"'[^']*'")
_NUM_RE = re.compile(r"\b\d+\b")

UPHOLD = "UPHOLD"
KILL = "KILL"
_VERDICTS = (UPHOLD, KILL)


def pattern_signature(candidate: Mapping) -> str:
    """Stable pattern signature for a candidate: category | normalized sink |
    guard kinds. Same pattern -> same signature regardless of literal values.
    """
    category = str(candidate.get("category", "")).lower()
    quote = str(candidate.get("quote", ""))
    normalized_sink = _NUM_RE.sub("<N>", _STR_LIT_RE.sub("<STR>", quote)).lower()
    guards = candidate.get("guards")
    if guards:
        guard_pattern = ",".join(sorted(str(g) for g in guards))
    else:
        guard_pattern = "none"
    joined = "|".join([category, normalized_sink, guard_pattern])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def default_store_path() -> Path:
    """Default on-disk location for adjudication records (config-rooted)."""
    from omnicli.config import _default_config_dir

    return _default_config_dir() / "adjudications.jsonl"


class AdjudicationStore:
    """JSONL-backed adjudication memory (in-memory when path is None).

    Records carry system_type (a specific engine/system, or "global"), the
    pattern signature, category, verdict (UPHOLD/KILL), reasoning, and a UTC
    ISO timestamp.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path is not None else None
        self._records: List[Dict] = []
        self._memory_only = self._path is None
        if self._path is not None:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                if self._path.exists():
                    with open(self._path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                            except json.JSONDecodeError:
                                continue  # skip malformed lines
                            if isinstance(rec, dict):
                                self._records.append(rec)
            except Exception as exc:  # never crash the caller on load
                print(
                    f"warning: adjudication store {self._path} unreadable "
                    f"({exc}); using in-memory mode",
                    file=sys.stderr,
                )
                self._memory_only = True

    def _persist(self, record: Dict) -> None:
        if self._memory_only or self._path is None:
            return
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:  # never crash the caller on write
            print(
                f"warning: could not persist adjudication to {self._path} "
                f"({exc}); falling back to in-memory mode",
                file=sys.stderr,
            )
            self._memory_only = True

    def record(
        self,
        system_type: str,
        candidate: Mapping,
        verdict: str,
        reasoning: str,
    ) -> Dict:
        """Append one adjudication; verdict must be UPHOLD or KILL."""
        if verdict not in _VERDICTS:
            raise ValueError(f"verdict must be one of {_VERDICTS}, got {verdict!r}")
        entry = {
            "system_type": system_type,
            "pattern_signature": pattern_signature(candidate),
            "category": str(candidate.get("category", "")).lower(),
            "verdict": verdict,
            "reasoning": reasoning,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._records.append(entry)
        self._persist(entry)
        return entry

    def find_precedents(self, candidate: Mapping, system_type: str) -> List[Dict]:
        """Precedents for this pattern in this system or global scope.

        Newest first.
        """
        sig = pattern_signature(candidate)
        matches = [
            r
            for r in self._records
            if r.get("pattern_signature") == sig
            and r.get("system_type") in (system_type, "global")
        ]
        matches.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return matches

    def check_auto_kill(
        self, candidate: Mapping, system_type: str
    ) -> Optional[Dict]:
        """Auto-kill rule: >=2 KILL records on this signature (system or
        global scope) -> the candidate is a repeat false positive.
        """
        kills = [
            p
            for p in self.find_precedents(candidate, system_type)
            if p.get("verdict") == KILL
        ]
        if len(kills) < 2:
            return None
        summary = re.sub(r"\s+", " ", str(kills[0].get("reasoning", ""))).strip()[:160]
        return {
            "auto_kill": True,
            "reason": (
                f"Repeat false positive: killed {len(kills)} times before "
                f"({summary})"
            ),
            "precedents": kills,
        }
