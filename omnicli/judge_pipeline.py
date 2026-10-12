"""Typed, evidence-fed adjudication for candidates from the legacy pipeline."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from omnicli.engines import ENGINE_REGISTRY
from omnicli.engines.global_engine import (
    aggregate as aggregate_engine_votes,
    build_case_file,
)
from omnicli.evidence import anchor_auth_candidate_to_mutation
from omnicli.redteam import (
    JUDGE_BALLOT_WEIGHT,
    Ballot,
    aggregate_ballots,
    ballot_margin,
    build_evidence_section,
    contradiction_flip,
    deterministic_cross_check,
    engine_vote_to_ballot,
    judge_calibration_prompt,
    judge_system_prompt,
    parse_judgment,
)
from omnicli.redteam_graph import (
    AGGREGATE,
    DETECT,
    JUDGE,
    PROBE,
    StageNode,
    build_redteam_graph,
)

ModelCall = Callable[
    [str, str, List[Dict[str, str]], int], Dict[str, Any]
]


def _source_excerpt(
    sources: Mapping[str, str],
    candidate: Mapping[str, Any],
    radius: int = 8,
) -> str:
    path = str(candidate.get("file", ""))
    source = next(
        (
            text
            for source_path, text in sources.items()
            if str(source_path).replace("\\", "/") == path.replace("\\", "/")
        ),
        None,
    )
    if not isinstance(source, str):
        return "(cited source file is unavailable)"
    try:
        line = int(candidate.get("start_line") or 1)
    except (TypeError, ValueError):
        line = 1
    lines = source.splitlines()
    start = max(0, line - radius - 1)
    stop = min(len(lines), line + radius)
    return "\n".join(
        f"{index + 1:05d} | {text}" for index, text in enumerate(lines[start:stop], start)
    )


def run_candidate_judge_graph(
    candidates: Sequence[Mapping[str, Any]],
    sources: Mapping[str, str],
    source_facts: List[Mapping[str, Any]],
    swarm: Sequence[Tuple[str, str]],
    model_call: ModelCall,
    max_tokens: int,
    designated_judge: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Judge existing candidates without rerunning probe or challenge stages."""
    normalized: List[Dict[str, Any]] = []
    used_ids = set()
    for index, item in enumerate(candidates, 1):
        candidate_id = str(item.get("candidate_id") or f"candidate-{index}")
        if candidate_id in used_ids:
            candidate_id = f"{candidate_id}-{index}"
        used_ids.add(candidate_id)
        candidate_evidence = dict(item)
        candidate_evidence["grounded"] = bool(item.get("quote_grounded"))
        candidate_evidence = anchor_auth_candidate_to_mutation(
            candidate_evidence, sources, source_facts
        )
        normalized.append({
            "id": candidate_id,
            "finding": str(item.get("title", "Unspecified candidate")),
            "category": str(item.get("category", "unknown")),
            "file": str(candidate_evidence.get("file", "")),
            "start_line": candidate_evidence.get("start_line"),
            "end_line": candidate_evidence.get("end_line"),
            "quote": str(candidate_evidence.get("quote", "")),
            "grounded": bool(candidate_evidence.get("grounded")),
            "citation_reanchored": candidate_evidence.get("citation_reanchored"),
            "attack_path": str(item.get("attack_path", "")),
            "preconditions": str(item.get("preconditions", "")),
            "unverified": str(item.get("unverified", "")),
            "original_status": str(item.get("status", "unverified_candidate")),
            "challenge_reviews": list(item.get("challenge_reviews", [])),
        })

    call_records: List[Dict[str, Any]] = []

    def probe_node(state: Dict[str, Any], _context: Any) -> Dict[str, Any]:
        return {"reprobe_count": 0}

    def detect_node(state: Dict[str, Any], _context: Any) -> Dict[str, Any]:
        entries = []
        engine_ballots: Dict[str, List[Ballot]] = {}
        engines = list(ENGINE_REGISTRY.values())
        specificities = {engine.name: engine.specificity for engine in engines}
        engine_votes: Dict[str, Dict[str, Any]] = {}
        if state.get("candidates"):
            for engine in engines:
                engine_votes[engine.name] = engine.analyze(
                    dict(sources), list(state["candidates"])
                )
        for candidate in state.get("candidates", []):
            candidate_id = str(candidate["id"])
            votes = [
                engine_votes[name][candidate_id]
                for name in engine_votes
                if candidate_id in engine_votes[name]
            ]
            global_verdict = aggregate_engine_votes(
                candidate, votes, specificities
            )
            entry: Dict[str, Any] = {
                "candidate": candidate,
                "reported_by": "existing scenario/adaptive/challenge pipeline",
                "cross_check": deterministic_cross_check(candidate, source_facts),
                "engine_votes": {vote.engine: vote.vote for vote in votes},
                "global_verdict": {
                    "verdict": global_verdict.verdict,
                    "reasoning": global_verdict.reasoning,
                },
                "case_file": build_case_file(
                    candidate, votes, global_verdict, []
                ),
            }
            entries.append(entry)
            engine_ballots[candidate_id] = [
                ballot
                for vote in votes
                if (ballot := engine_vote_to_ballot(
                    vote.engine, {"vote": vote.vote, "reason": vote.reason}
                )) is not None
            ]
        return {
            "candidates": entries,
            "engine_ballots": engine_ballots,
            "judge_ballots": {},
            "flip_notes": {},
        }

    def judge_node(state: Dict[str, Any], _context: Any) -> Dict[str, Any]:
        pairs = [designated_judge] if designated_judge else list(swarm)
        if not pairs and state.get("candidates"):
            raise ValueError("No configured model is available for candidate judging.")
        judgment_results = []
        judge_ballots: Dict[str, List[Ballot]] = {}
        flip_notes: Dict[str, List[str]] = {}
        all_parseable = bool(state.get("candidates"))
        for entry in state.get("candidates", []):
            candidate = entry["candidate"]
            review_pairs = {
                (str(review.get("provider", "")), str(review.get("model", "")))
                for review in candidate.get("challenge_reviews", [])
                if isinstance(review, Mapping)
                and review.get("provider")
                and review.get("model")
            }
            independent_pairs = [pair for pair in pairs if pair not in review_pairs]
            roster = list(independent_pairs) + [
                pair for pair in pairs if pair not in independent_pairs
            ]
            ordered = list(dict.fromkeys(roster))
            max_attempts = min(3, len(ordered))
            prompt = (
                "### CANDIDATE FINDING UNDER REVIEW\n"
                f"FINDING: {candidate['finding']}\n"
                f"CATEGORY: {candidate['category']}\n"
                f"FILE: {candidate['file']}\n"
                f"LINE: {candidate.get('start_line')}\n"
                f"QUOTE: `{candidate['quote']}`\n"
                f"Quote grounded at cited location: {candidate['grounded']}\n\n"
                f"{build_evidence_section(entry)}\n\n"
                "### AUTHORITATIVE SOURCE EXCERPT\n"
                f"```text\n{_source_excerpt(sources, candidate)}\n```\n\n"
                f"{judge_calibration_prompt()}\n\n"
                "Judge the supplied candidate, not the source as a whole. "
                "Return one JUDGMENT: UPHOLD or JUDGMENT: KILL block and "
                "concise REASONING grounded in the supplied code."
            )
            provider, model = ordered[0]
            parsed: Dict[str, Any] = {"judgment": "UNPARSEABLE"}
            call_record: Dict[str, Any] = {}
            attempts = 0
            for attempt_pair in ordered:
                if attempts >= max_attempts:
                    break
                provider, model = attempt_pair
                call = model_call(
                    provider,
                    model,
                    [
                        {"role": "system", "content": judge_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens,
                )
                parsed = parse_judgment(str(call.get("answer", "")))
                call_record = {
                    **call,
                    "stage": "typed_candidate_judge",
                    "focus": candidate["id"],
                    "role": f"JUDGE — {provider}:{model}",
                    "parsed_judgment": parsed,
                }
                call_records.append(call_record)
                attempts += 1
                if not call.get("error") and parsed.get("judgment") in {
                    "UPHOLD",
                    "KILL",
                }:
                    break
            judgment_results.append(call_record)
            vote = parsed["judgment"]
            notes: List[str] = []
            if attempts > 1:
                notes.append(
                    f"Judge failed over after {attempts - 1} unsuccessful "
                    f"attempt(s); final judge {provider}:{model}."
                )
            ballots: List[Ballot] = []
            all_parseable = all_parseable and vote in {"UPHOLD", "KILL"}
            if vote in {"UPHOLD", "KILL"}:
                adjusted, note = contradiction_flip(
                    vote, entry, str(parsed.get("reasoning", ""))
                )
                if note:
                    notes.append(note)
                ballots.append(Ballot(
                    voter=f"judge:{provider}:{model}",
                    verdict=adjusted,
                    weight=JUDGE_BALLOT_WEIGHT,
                    reason=note or f"judge voted {vote}",
                ))
            judge_ballots[candidate["id"]] = ballots
            flip_notes[candidate["id"]] = notes
        combined_margins = []
        for entry in state.get("candidates", []):
            candidate_id = entry["candidate"]["id"]
            combined = list(state.get("engine_ballots", {}).get(candidate_id, []))
            combined.extend(judge_ballots.get(candidate_id, []))
            combined_margins.append(ballot_margin(combined))
        return {
            "judgment_results": judgment_results,
            "judge_ballots": judge_ballots,
            "flip_notes": flip_notes,
            "judge_agreement_margin": min(combined_margins, default=1.0),
            "judge_round_unparseable": not all_parseable,
        }

    def aggregate_node(state: Dict[str, Any], _context: Any) -> Dict[str, Any]:
        by_id = {
            result["focus"]: result
            for result in state.get("judgment_results", [])
        }
        verdicts = []
        for entry in state.get("candidates", []):
            candidate_id = entry["candidate"]["id"]
            judgment = by_id.get(candidate_id, {})
            parsed = judgment.get("parsed_judgment", {})
            judge_ballots = state.get("judge_ballots", {}).get(candidate_id, [])
            ballots = list(
                state.get("engine_ballots", {}).get(candidate_id, [])
            ) + list(judge_ballots)
            if judgment.get("error"):
                verdict = "UNJUDGED"
            elif parsed.get("judgment") == "UNPARSEABLE":
                verdict = "UNJUDGED"
            else:
                verdict = aggregate_ballots(ballots)
            verdicts.append({
                "candidate": entry["candidate"],
                "original_status": entry["candidate"]["original_status"],
                "cross_check": entry["cross_check"],
                "engine_votes": entry["engine_votes"],
                "global_verdict": entry["global_verdict"],
                "judgment": parsed,
                "provider": judgment.get("provider"),
                "model": judgment.get("model"),
                "provider_error": judgment.get("error"),
                "verdict": verdict,
                "ballots": [
                    {
                        "voter": ballot.voter,
                        "verdict": ballot.verdict,
                        "weight": ballot.weight,
                        "reason": ballot.reason,
                    }
                    for ballot in ballots
                ],
                "flip_notes": state.get("flip_notes", {}).get(candidate_id, []),
                "margin": ballot_margin(ballots),
            })
        return {"verdicts": verdicts}

    nodes = {
        PROBE: StageNode(
            name=PROBE,
            fn=probe_node,
            produces=frozenset({"reprobe_count"}),
            description="Pass through existing candidates; do not repeat probes.",
        ),
        DETECT: StageNode(
            name=DETECT,
            fn=detect_node,
            requires=frozenset({"candidates"}),
            produces=frozenset({
                "engine_ballots", "judge_ballots", "flip_notes",
            }),
            description="Attach independent deterministic evidence and ballots.",
        ),
        JUDGE: StageNode(
            name=JUDGE,
            fn=judge_node,
            requires=frozenset({"candidates", "engine_ballots"}),
            produces=frozenset({
                "judgment_results", "judge_ballots", "flip_notes",
                "judge_agreement_margin", "judge_round_unparseable",
            }),
            description="Judge each existing candidate with its evidence case file.",
        ),
        AGGREGATE: StageNode(
            name=AGGREGATE,
            fn=aggregate_node,
            requires=frozenset({"candidates", "judge_ballots"}),
            produces=frozenset({"verdicts"}),
            description="Combine judge and engine ballots without altering triage.",
        ),
    }
    graph = build_redteam_graph(nodes, max_reprobes=0)
    state = graph.run(
        {"candidates": normalized},
        initial_keys=frozenset({"candidates"}),
    )
    return {
        "verdicts": state.get("verdicts", []),
        "audit_trail": state.get("audit_trail", []),
        "calls": call_records,
        "failures": [
            (
                f"{result.get('provider')}:{result.get('model')} "
                f"(candidate judge/{result.get('focus')}): "
                + (
                    str(result["error"])
                    if result.get("error")
                    else "response did not contain a parseable UPHOLD/KILL judgment"
                )
            )
            for result in call_records
            if result.get("error")
            or result.get("parsed_judgment", {}).get("judgment") == "UNPARSEABLE"
        ],
        "reviewer_independence": (
            "challenge-reviewer-avoiding when another configured member exists; "
            "not proof of independence from initial attack authors"
        ),
        "graph_status": "completed",
    }
