"""Per-system deterministic engines for the red-team judge.

Each engine is an independent verifier behind one verdict interface:
analyze(sources, candidates) -> {candidate_id: EngineVote}. Engines vote
SUPPORTED / NOT_SUPPORTED / NOT_COVERED; a council judge aggregates the
votes. Original code for this repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Vote constants.
# ---------------------------------------------------------------------------

SUPPORTED = "supported"
NOT_SUPPORTED = "not_supported"
NOT_COVERED = "not_covered"

VOTES: Tuple[str, str, str] = (SUPPORTED, NOT_SUPPORTED, NOT_COVERED)


@dataclass
class EngineFact:
    """One deterministic claim made by an engine about a candidate."""

    id: str                      # stable fact id, e.g. "sink_present"
    kind: str                    # "sink"|"source"|"guard"|"route"|"hook"|"absence"|"meta"
    claim: str                   # deterministic claim text
    line: Optional[int] = None   # source line the fact refers to, if any
    code: str = ""               # verbatim code, or "" for absences
    supports_finding: bool = True


@dataclass
class EngineVote:
    """One engine's verdict on one candidate finding.

    Engines are VOTERS, not executioners: a vote feeds the ballot
    aggregation (weighted judge over unit-weight engine ballots) and may
    never unilaterally KILL a candidate.
    """

    candidate_id: str
    engine: str
    engine_version: str
    vote: str
    facts: List[EngineFact] = field(default_factory=list)
    reason: str = ""
    reasons: List[str] = field(default_factory=list)  # human-readable reasons


class Engine:
    """Independent deterministic verifier of candidate findings."""

    name: str = "base"
    version: str = "0.0.0"
    rule_set_date: str = "1970-01-01"
    covered_categories: Tuple[str, ...] = ()
    specificity: int = 0

    def analyze(
        self,
        sources: Dict[str, str],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, EngineVote]:
        """Return {candidate_id: EngineVote} for every candidate."""
        raise NotImplementedError


ENGINE_REGISTRY: Dict[str, Engine] = {}


def register_engine(cls):
    """Class decorator: register an Engine subclass by its name."""
    instance = cls()
    if not instance.name or instance.name in ENGINE_REGISTRY:
        raise ValueError(
            f"Engine name {instance.name!r} is missing or already registered."
        )
    ENGINE_REGISTRY[instance.name] = instance
    return cls


__all__ = [
    "SUPPORTED",
    "NOT_SUPPORTED",
    "NOT_COVERED",
    "VOTES",
    "Engine",
    "EngineFact",
    "EngineVote",
    "ENGINE_REGISTRY",
    "register_engine",
]

# Import built-in engines so they self-register on package import. These come
# last to avoid circular imports (engine modules import the names above).
from omnicli.engines import python_general  # noqa: E402,F401
from omnicli.engines import python_web  # noqa: E402,F401
