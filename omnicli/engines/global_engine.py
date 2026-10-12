"""Global deterministic aggregation engine for the OmniCLI red-team judge.

Merges per-engine votes into a single verdict with no model calls and no
randomness: identical inputs always yield identical outputs. When covering
engines conflict, the most specific covering engine decides. Original code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

try:  # shared types live in omnicli/engines/__init__.py (sibling agent)
    from omnicli.engines import (
        NOT_COVERED,
        NOT_SUPPORTED,
        SUPPORTED,
        EngineFact,
        EngineVote,
    )
except ImportError:
    # Local fallback with the same names and shapes, used only when the
    # package __init__.py has not been created yet. Prefer the real import.
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    NOT_COVERED = "not_covered"

    @dataclass
    class EngineFact:
        id: str
        kind: str
        claim: str
        line: Optional[int] = None
        code: str = ""
        supports_finding: bool = True

    @dataclass
    class EngineVote:
        candidate_id: str
        engine: str
        engine_version: str
        vote: str
        facts: List[EngineFact] = field(default_factory=list)
        reason: str = ""


@dataclass
class GlobalVerdict:
    """The council's single verdict for one candidate."""

    candidate_id: str
    verdict: str  # one of "supported" | "not_supported" | "not_covered"
    facts: List[EngineFact]  # merged, deduplicated by fact id
    reasoning: str
    engine_votes: Dict[str, str]  # engine name -> vote


def _dedupe_votes(votes: List[EngineVote], candidate_id: str) -> List[EngineVote]:
    """Votes for this candidate, sorted by engine name; first wins on dupes."""
    ours = sorted(
        (v for v in votes if v.candidate_id == candidate_id),
        key=lambda v: v.engine,
    )
    by_engine: Dict[str, EngineVote] = {}
    for v in ours:
        by_engine.setdefault(v.engine, v)
    return list(by_engine.values())


def _specificity(vote: EngineVote, specificities: Mapping[str, int]) -> int:
    try:
        return int(specificities.get(vote.engine, 0))
    except (TypeError, ValueError):
        return 0


def _fact_source_engines(votes: List[EngineVote]) -> Dict[str, str]:
    """First-wins mapping of fact id -> engine name (votes already sorted)."""
    sources: Dict[str, str] = {}
    for v in votes:
        for f in v.facts or []:
            sources.setdefault(f.id, v.engine)
    return sources


def _merge_facts(
    votes: List[EngineVote],
) -> Tuple[List[EngineFact], List[Tuple[str, List[str]]]]:
    """Merge facts across engines; first engine wins a fact-id collision.

    Returns (facts, collisions) where each collision is (fact_id, engines).
    """
    facts: List[EngineFact] = []
    collisions: Dict[str, List[str]] = {}
    for v in votes:
        for f in v.facts or []:
            if f.id in collisions:
                collisions[f.id].append(v.engine)
            elif any(existing.id == f.id for existing in facts):
                collisions[f.id] = [v.engine]
            else:
                facts.append(f)
    return facts, [(fid, engines) for fid, engines in collisions.items()]


def _collision_note(collisions: List[Tuple[str, List[str]]], sources: Dict[str, str]) -> str:
    if not collisions:
        return ""
    parts = [
        f"{fid} (kept {sources.get(fid, '?')}, dropped {', '.join(drop)})"
        for fid, drop in sorted(collisions)
    ]
    return "; fact-id collisions resolved first-engine-wins: " + "; ".join(parts)


def aggregate(
    candidate: Mapping,
    votes: List[EngineVote],
    specificities: Mapping[str, int],
) -> GlobalVerdict:
    """Aggregate engine votes into one deterministic verdict.

    - Only votes for this candidate count.
    - No covering engine (vote != NOT_COVERED) -> NOT_COVERED.
    - All covering agree -> that verdict.
    - Conflict -> the most specific covering engine(s) decide: any of them
      voting NOT_SUPPORTED kills the candidate; otherwise any SUPPORTED
      covering vote carries (dissent noted); else NOT_SUPPORTED.
    """
    candidate_id = str(candidate.get("id", ""))
    ours = _dedupe_votes(votes, candidate_id)
    engine_votes = {v.engine: v.vote for v in ours}
    covering = [v for v in ours if v.vote != NOT_COVERED]

    sources = _fact_source_engines(covering)
    facts, collisions = _merge_facts(covering)
    collision_note = _collision_note(collisions, sources)

    if not covering:
        return GlobalVerdict(
            candidate_id=candidate_id,
            verdict=NOT_COVERED,
            facts=[],
            reasoning="no engine covers this candidate's category" + collision_note,
            engine_votes=engine_votes,
        )

    distinct = sorted({v.vote for v in covering})
    if len(distinct) == 1:
        verdict = distinct[0]
        reasoning = f"all {len(covering)} covering engine(s) agree: {verdict}"
        return GlobalVerdict(
            candidate_id=candidate_id,
            verdict=verdict,
            facts=facts,
            reasoning=reasoning + collision_note,
            engine_votes=engine_votes,
        )

    # Conflict: the most specific covering engine(s) decide.
    top_spec = max(_specificity(v, specificities) for v in covering)
    top = [v for v in covering if _specificity(v, specificities) == top_spec]
    killers = sorted((v for v in top if v.vote == NOT_SUPPORTED), key=lambda v: v.engine)
    if killers:
        lead = killers[0]
        key_fact = lead.facts[0].claim if lead.facts else "no route reaches this sink"
        overridden = ", ".join(
            f"{v.engine} {v.vote.upper()}"
            for v in sorted(covering, key=lambda v: v.engine)
            if v.engine != lead.engine
        )
        names = ", ".join(v.engine for v in killers)
        reasoning = (
            f"{names} (specificity {top_spec}) reports {key_fact}; "
            f"overrides {overridden}"
        )
        return GlobalVerdict(
            candidate_id=candidate_id,
            verdict=NOT_SUPPORTED,
            facts=facts,
            reasoning=reasoning + collision_note,
            engine_votes=engine_votes,
        )

    supporters = sorted(v.engine for v in covering if v.vote == SUPPORTED)
    if supporters:
        dissent = sorted(v.engine for v in covering if v.vote == NOT_SUPPORTED)
        reasoning = (
            f"{', '.join(supporters)} vote(s) SUPPORTED; "
            f"dissent from {', '.join(dissent)} (NOT_SUPPORTED) noted but "
            f"no more-specific covering engine contradicts"
        )
        return GlobalVerdict(
            candidate_id=candidate_id,
            verdict=SUPPORTED,
            facts=facts,
            reasoning=reasoning + collision_note,
            engine_votes=engine_votes,
        )

    # Only NOT_SUPPORTED among covering (defensive: unreachable via the
    # branches above, kept so no conflict path can fall through silently).
    top_names = ", ".join(sorted(v.engine for v in top))
    reasoning = (
        f"no covering engine supports this candidate; most specific covering "
        f"engine(s) ({top_names}, specificity {top_spec}) do not claim support"
    )
    return GlobalVerdict(
        candidate_id=candidate_id,
        verdict=NOT_SUPPORTED,
        facts=facts,
        reasoning=reasoning + collision_note,
        engine_votes=engine_votes,
    )


def _fact_dict(fact: EngineFact, sources: Dict[str, str]) -> Dict:
    return {
        "id": fact.id,
        "kind": fact.kind,
        "claim": fact.claim,
        "line": fact.line,
        "code": fact.code,
        "supports_finding": fact.supports_finding,
        "engine": sources.get(fact.id, ""),
    }


def build_case_file(
    candidate: Mapping,
    votes: List[EngineVote],
    global_verdict: GlobalVerdict,
    precedents: List[Dict],
) -> Dict:
    """Build the case record the LLM judge must rule on.

    Returns {"candidate_id", "candidate", "engine_votes", "global_verdict",
    "facts", "precedents"}.
    """
    candidate_id = str(candidate.get("id", ""))
    ours = _dedupe_votes(votes, candidate_id)
    sources = _fact_source_engines(ours)
    return {
        "candidate_id": candidate_id,
        "candidate": dict(candidate),
        "engine_votes": {
            v.engine: {
                "vote": v.vote,
                "reason": v.reason,
                "version": v.engine_version,
            }
            for v in ours
        },
        "global_verdict": {
            "verdict": global_verdict.verdict,
            "reasoning": global_verdict.reasoning,
        },
        "facts": [_fact_dict(f, sources) for f in global_verdict.facts],
        "precedents": list(precedents),
    }
