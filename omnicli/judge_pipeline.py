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


MAX_JUDGE_SOURCE_CHARS = 60_000


def _number_source(path: str, source: str, max_chars: int) -> str:
    """Render one source file with stable line numbers, bounded by max_chars."""
    header = f"FILE: {path}\n"
    budget = max(0, max_chars - len(header))
    lines: List[str] = []
    used = 0
    truncated = False
    for index, text in enumerate(source.splitlines(), start=1):
        rendered = f"{index:05d} | {text}"
        if used + len(rendered) + 1 > budget:
            truncated = True
            break
        lines.append(rendered)
        used += len(rendered) + 1
    if truncated:
        lines.append("... [source truncated to fit judge context budget]")
    return header + "\n".join(lines)


def _source_context(
    sources: Mapping[str, str],
    candidate: Mapping[str, Any],
    max_chars: int = MAX_JUDGE_SOURCE_CHARS,
) -> str:
    """Return full supplied source context, citing file first.

    The typed judge makes the final UPHOLD/KILL call, so an eight-line excerpt
    is not enough: guards, middleware, decorators, and callers often live far
    from the cited sink. For normal-sized reviews this includes every supplied
    file; large reviews preserve the cited file first and label omitted files.
    """
    cited_path = str(candidate.get("file", "")).replace("\\", "/")
    ordered = sorted(
        sources.items(),
        key=lambda item: (
            0 if str(item[0]).replace("\\", "/") == cited_path else 1,
            str(item[0]),
        ),
    )
    sections: List[str] = []
    omitted: List[str] = []
    remaining = max_chars
    cited_seen = False
    for path, source in ordered:
        normalized = str(path).replace("\\", "/")
        is_cited = normalized == cited_path
        if is_cited:
            cited_seen = True
        if not isinstance(source, str) or remaining <= 0:
            omitted.append(normalized)
            continue
        rendered = _number_source(normalized, source, remaining)
        sections.append(rendered)
        remaining -= len(rendered) + 2
    if not cited_seen:
        sections.insert(0, f"(cited source file is unavailable: {cited_path})")
    if omitted:
        sections.append(
            "OMITTED FILES (judge context budget): " + ", ".join(omitted)
        )
    return "\n\n".join(sections) if sections else "(no supplied source available)"


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
            "reported_by": [
                (str(pair[0]), str(pair[1]))
                for pair in (item.get("reported_by") or [])
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            ] + [
                tuple(pair.split(":", 1))
                for pair in (item.get("reported_by") or [])
                if isinstance(pair, str) and ":" in pair
            ],
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
                "reported_by": list(candidate.get("reported_by") or []),
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
        # A designated judge is a preference, not a blank cheque to judge its
        # own candidate. The pool keeps the full roster available for
        # failover when the designated member authored or challenged a
        # candidate.
        preferred = [designated_judge] if designated_judge else []
        pool = list(dict.fromkeys(
            pair for pair in [*preferred, *swarm] if pair
        ))
        if not pool and state.get("candidates"):
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
            author_pairs = set(candidate.get("reported_by") or [])
            independent_pairs = [
                pair for pair in pool
                if pair not in author_pairs and pair not in review_pairs
            ]
            non_author_pairs = [
                pair for pair in pool if pair not in author_pairs
            ]
            non_reviewer_pairs = [
                pair for pair in pool if pair not in review_pairs
            ]
            ordered = list(dict.fromkeys(
                independent_pairs
                + non_author_pairs
                + non_reviewer_pairs
                + pool
            ))
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
                "### AUTHORITATIVE SOURCE CONTEXT\n"
                "Use this full supplied source to trace reachability and guards; "
                "the cited file is shown first.\n"
                f"```text\n{_source_context(sources, candidate)}\n```\n\n"
                f"{judge_calibration_prompt()}\n\n"
                "Judge the supplied candidate against the source context. "
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
            selected_pair = (provider, model)
            if selected_pair in author_pairs:
                judge_independence = "author"
            elif selected_pair in review_pairs:
                judge_independence = "challenge-reviewer"
            else:
                judge_independence = "independent"
            if call_record:
                call_record["judge_independence"] = judge_independence
            judgment_results.append(call_record)
            vote = parsed["judgment"]
            notes: List[str] = []
            if attempts > 1:
                notes.append(
                    f"Judge failed over after {attempts - 1} unsuccessful "
                    f"attempt(s); final judge {provider}:{model}."
                )
            if judge_independence == "author":
                notes.append(
                    "No non-author judge was available; the judging model "
                    "authored this candidate."
                )
            elif judge_independence == "challenge-reviewer":
                notes.append(
                    "Selected judge also challenged this candidate; the "
                    "verdict is not independent."
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
                "judge_independence": judgment.get("judge_independence"),
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
            "judge selection avoids candidate authors and challenge reviewers "
            "when another configured member exists; fallback is labeled"
        ),
        "graph_status": "completed",
    }
