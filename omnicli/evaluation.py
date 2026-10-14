"""Repeatable, labeled evaluation for security-audit model selection."""

from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_REQUIRED_FINDING_FIELDS = {"category", "file", "line", "evidence"}


def load_security_dataset(path: str) -> List[Dict[str, Any]]:
    """Load versioned JSONL cases and reject incomplete or ambiguous labels."""
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise ValueError(f"Evaluation dataset does not exist: {dataset_path}")

    cases: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    paired_cases: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for line_number, line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(case, dict):
            raise ValueError(f"Dataset line {line_number} must be a JSON object.")
        case_id = case.get("id")
        prompt = case.get("prompt")
        source = case.get("source")
        findings = case.get("expected_findings")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Dataset line {line_number} needs a non-empty id.")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case id: {case_id}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Dataset case {case_id} needs a non-empty prompt.")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Dataset case {case_id} needs non-empty source.")
        if not isinstance(findings, list):
            raise ValueError(f"Dataset case {case_id} needs expected_findings as a list.")
        pair_id = case.get("pair_id")
        pair_role = case.get("pair_role")
        pair_category = case.get("pair_category")
        if (pair_id is None) != (pair_role is None):
            raise ValueError(
                f"Dataset case {case_id} must provide pair_id and pair_role together."
            )
        if pair_id is not None:
            if not isinstance(pair_id, str) or not pair_id.strip():
                raise ValueError(f"Dataset case {case_id} has an invalid pair_id.")
            if pair_role not in {"vulnerable", "fixed"}:
                raise ValueError(
                    f"Dataset case {case_id} pair_role must be vulnerable or fixed."
                )
            if not isinstance(pair_category, str) or not pair_category.strip():
                raise ValueError(
                    f"Paired dataset case {case_id} needs a shared pair_category."
                )
            roles = paired_cases.setdefault(pair_id, {})
            if pair_role in roles:
                raise ValueError(
                    f"Paired dataset group {pair_id} has duplicate {pair_role} cases."
                )
            roles[pair_role] = case
            if pair_role == "vulnerable" and len(findings) != 1:
                raise ValueError(
                    f"Vulnerable paired case {case_id} must have exactly one expected finding."
                )
            if pair_role == "fixed" and findings:
                raise ValueError(
                    f"Fixed paired case {case_id} must have no expected findings."
                )
        split = case.get("split")
        if split is not None and (
            not isinstance(split, str) or split not in {"development", "held_out"}
        ):
            raise ValueError(
                f"Dataset case {case_id} split must be 'development' or 'held_out'."
            )
        if split is not None:
            if not isinstance(case.get("category"), str) or not case["category"].strip():
                raise ValueError(f"Dataset case {case_id} needs a category.")
            adjudication_status = case.get("adjudication_status")
            reviewer_count = case.get("reviewer_count")
            if adjudication_status is not None and adjudication_status not in {
                "provisional", "independently_reviewed",
            }:
                raise ValueError(
                    f"Dataset case {case_id} has an invalid adjudication_status."
                )
            if reviewer_count is not None and (
                not isinstance(reviewer_count, int)
                or isinstance(reviewer_count, bool)
                or reviewer_count < 1
            ):
                raise ValueError(f"Dataset case {case_id} has invalid reviewer_count.")
            if adjudication_status == "independently_reviewed" and reviewer_count != 2:
                raise ValueError(
                    f"Independently reviewed case {case_id} must record reviewer_count 2."
                )
            if findings:
                if not isinstance(case.get("severity"), str) or not case["severity"].strip():
                    raise ValueError(f"Positive dataset case {case_id} needs a severity.")
                for finding in findings:
                    if (
                        not isinstance(finding, dict)
                        or not isinstance(finding.get("preconditions"), str)
                        or not finding["preconditions"].strip()
                    ):
                        raise ValueError(
                            f"Positive finding in {case_id} needs explicit preconditions."
                        )
            elif not isinstance(case.get("benign_rationale"), str) or not case["benign_rationale"].strip():
                raise ValueError(f"Clean dataset case {case_id} needs a benign rationale.")

        seen_ids.add(case_id)
        gold_keys = set()
        for finding in findings:
            if not isinstance(finding, dict) or not _REQUIRED_FINDING_FIELDS.issubset(finding):
                raise ValueError(
                    f"Each expected finding in {case_id} needs category, file, line, and evidence."
                )
            if (
                not isinstance(finding["category"], str)
                or not finding["category"].strip()
                or not isinstance(finding["file"], str)
                or not finding["file"].strip()
                or not isinstance(finding["line"], int)
                or isinstance(finding["line"], bool)
                or finding["line"] < 1
                or not isinstance(finding["evidence"], str)
                or not finding["evidence"].strip()
            ):
                raise ValueError(f"Invalid expected finding in dataset case {case_id}.")
            source_lines = source.splitlines()
            evidence_lines = finding["evidence"].splitlines()
            evidence_start = finding["line"] - 1
            source_excerpt = "\n".join(
                source_lines[evidence_start:evidence_start + len(evidence_lines)]
            )
            if (
                evidence_start >= len(source_lines)
                or _normalize(finding["evidence"]) != _normalize(source_excerpt)
            ):
                raise ValueError(
                    f"Expected evidence in case {case_id} does not match its labeled source line."
                )
            key = _finding_key(finding)
            if key in gold_keys:
                raise ValueError(f"Duplicate expected finding in dataset case {case_id}.")
            gold_keys.add(key)
        cases.append(case)

    if not cases:
        raise ValueError("Evaluation dataset contains no cases.")
    if not any(case["expected_findings"] for case in cases):
        raise ValueError("Evaluation dataset needs at least one case with a labeled finding.")
    if not any(not case["expected_findings"] for case in cases):
        raise ValueError("Evaluation dataset needs at least one explicitly clean case.")
    for pair_id, roles in paired_cases.items():
        if set(roles) != {"vulnerable", "fixed"}:
            raise ValueError(
                f"Paired dataset group {pair_id} must contain one vulnerable and one fixed case."
            )
        vulnerable, fixed = roles["vulnerable"], roles["fixed"]
        if vulnerable["pair_category"].strip().lower() != fixed["pair_category"].strip().lower():
            raise ValueError(
                f"Paired dataset group {pair_id} must use the same pair_category for both cases."
            )
    return cases


def create_blinded_review_package(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a neutral source-only package for independent held-out review."""
    if not cases or any(case.get("split") != "held_out" for case in cases):
        raise ValueError("Independent review packages require only held_out cases.")

    review_cases = []
    for number, case in enumerate(cases, start=1):
        review_cases.append({
            "review_id": f"review-{number:03d}",
            "file": f"source_{number:03d}.py",
            "task": (
                "Review the supplied source for security findings. For each finding, "
                "record the category, exact source file and line, quoted evidence, "
                "attacker preconditions, and impact. If no finding is supported, "
                "explain why the source is intentionally safe."
            ),
            "source": case["source"],
        })
    return {
        "schema_version": 1,
        "purpose": "independent_held_out_security_adjudication",
        "review_instructions": [
            "Work independently before comparing results with any existing labels.",
            "Do not infer expected outcomes from filenames, identifiers, or prior reports.",
            "Record findings or an explicit benign rationale for every review_id.",
        ],
        "cases": review_cases,
    }


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _finding_key(finding: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        finding["category"].strip().lower(),
        finding["file"].replace("\\", "/").strip(),
        finding["line"],
    )


def _citation_grounded(case: Dict[str, Any], finding: Dict[str, Any]) -> bool:
    if (
        not finding["evidence"].strip()
        or finding["file"].replace("\\", "/").strip()
        != case.get("file", "source.py").replace("\\", "/").strip()
    ):
        return False
    from omnicli.evidence import strip_display_prefixes

    source_lines = case["source"].splitlines()
    start = finding["line"] - 1
    cleaned_lines, _ = strip_display_prefixes(
        finding["evidence"].splitlines(), finding["line"]
    )
    evidence_lines = cleaned_lines
    if not _normalize("\n".join(evidence_lines)):
        return False
    if start < 0 or start >= len(source_lines):
        return False
    cited_excerpt = "\n".join(source_lines[start:start + max(1, len(evidence_lines))])
    return _normalize("\n".join(evidence_lines)) in _normalize(cited_excerpt)


def _parse_findings(content: Any) -> List[Dict[str, Any]]:
    if not isinstance(content, str):
        raise ValueError("Model response content is not text.")
    response = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", response, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        response = fenced.group(1)
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("findings"), list):
        raise ValueError("Model JSON must contain a findings array.")
    findings = parsed["findings"]
    for finding in findings:
        if (
            not isinstance(finding, dict)
            or not _REQUIRED_FINDING_FIELDS.issubset(finding)
            or not isinstance(finding.get("category"), str)
            or not finding["category"].strip()
            or not isinstance(finding["file"], str)
            or not isinstance(finding["line"], int)
            or isinstance(finding["line"], bool)
            or not isinstance(finding["evidence"], str)
        ):
            raise ValueError("A model finding did not match the required JSON schema.")
        from omnicli.evidence import canonical_finding_category
        finding["category"] = canonical_finding_category(finding["category"])
    return findings


def _evaluate_case(case: Dict[str, Any], predicted: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected = case["expected_findings"]
    unmatched = set(range(len(expected)))
    true_positives = 0
    false_positives = 0
    false_positive_categories: List[str] = []
    category_mismatches: List[Dict[str, Any]] = []
    grounded_citations = 0
    matched: List[Dict[str, Any]] = []
    for candidate in predicted:
        if _citation_grounded(case, candidate):
            grounded_citations += 1
        candidate_key = _finding_key(candidate)
        evidence = _normalize(candidate["evidence"])
        match_index = next(
            (
                index for index in sorted(unmatched)
                if _finding_key(expected[index]) == candidate_key
                and evidence
                and evidence == _normalize(expected[index]["evidence"])
            ),
            None,
        )
        if match_index is None:
            aligned_index = next(
                (
                    index for index in sorted(unmatched)
                    if expected[index]["file"].replace("\\", "/").strip()
                    == candidate["file"].replace("\\", "/").strip()
                    and expected[index]["line"] == candidate["line"]
                    and evidence
                    and evidence == _normalize(expected[index]["evidence"])
                ),
                None,
            )
            if aligned_index is not None:
                category_mismatches.append({
                    "predicted_category": candidate["category"].strip().lower(),
                    "expected_category": expected[aligned_index]["category"].strip().lower(),
                    "file": candidate["file"],
                    "line": candidate["line"],
                    "evidence": evidence,
                })
            false_positives += 1
            false_positive_categories.append(candidate["category"].strip().lower())
            continue
        unmatched.remove(match_index)
        true_positives += 1
        matched.append({"category": candidate["category"], "file": candidate["file"], "line": candidate["line"]})
    return {
        "case_id": case["id"],
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": len(unmatched),
        "citations": len(predicted),
        "grounded_citations": grounded_citations,
        "ungrounded_citations": len(predicted) - grounded_citations,
        "predicted_findings": predicted,
        "matched": matched,
        "category_mismatches": category_mismatches,
        "evidence_aligned_findings": len(matched) + len(category_mismatches),
        "false_positive_categories": false_positive_categories,
        "missed": [
            expected[index]
            for index in sorted(unmatched)
        ],
    }


def _score(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _wilson_interval(successes: int, total: int) -> Optional[List[float]]:
    if total == 0:
        return None
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def _bootstrap_ci(
    successes: int,
    total: int,
    resamples: int = 2000,
    seed: int = 0,
) -> Optional[List[float]]:
    """Bootstrap 95% CI for a binomial proportion via resampling with replacement.

    Uses a fixed seed so gate reports are reproducible. Returns None when
    there are no trials.
    """
    if total == 0:
        return None
    rng = random.Random(seed)
    outcomes = [1] * successes + [0] * (total - successes)
    estimates = []
    for _ in range(resamples):
        sample = [rng.choice(outcomes) for _ in range(total)]
        estimates.append(sum(sample) / total)
    estimates.sort()
    lower = estimates[int(0.025 * resamples)]
    upper = estimates[min(resamples - 1, int(0.975 * resamples))]
    return [round(lower, 4), round(upper, 4)]


def _word_ngrams(text: str, n: int = 8) -> set:
    words = re.findall(r"[a-z0-9_]+", (text or "").lower())
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def ngram_overlap_ratio(case_source: str, reference_texts: List[str], n: int = 8) -> float:
    """Max n-gram overlap of a case source against any reference text.

    Scenario-leakage check: a high ratio suggests the case may have been seen
    during training, so the gate would measure memorization, not capability.
    """
    case_grams = _word_ngrams(case_source, n)
    if not case_grams:
        return 0.0
    best = 0.0
    for reference in reference_texts or []:
        ref_grams = _word_ngrams(reference, n)
        if not ref_grams:
            continue
        overlap = len(case_grams & ref_grams) / len(case_grams)
        best = max(best, overlap)
    return round(best, 4)


def flag_leaked_cases(
    cases: List[Dict[str, Any]],
    reference_corpus: List[str],
    threshold: float = 0.5,
    n: int = 8,
) -> Dict[str, Any]:
    """Flag cases whose source overlaps a reference corpus above threshold."""
    flagged = []
    for case in cases:
        ratio = ngram_overlap_ratio(case.get("source", ""), reference_corpus, n)
        if ratio >= threshold:
            flagged.append({"case_id": case.get("id"), "overlap_ratio": ratio})
    clean_ids = {
        case.get("id") for case in cases
    } - {item["case_id"] for item in flagged}
    return {
        "threshold": threshold,
        "ngram_n": n,
        "leaked_cases": flagged,
        "leaked_count": len(flagged),
        "clean_case_ids": sorted(clean_ids),
        "clean_count": len(clean_ids),
    }


def _paired_change_metrics(
    cases: List[Dict[str, Any]],
    case_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score whether vulnerable/fixed variants receive the correct distinct outcomes."""
    pairs: Dict[str, Dict[str, str]] = {}
    for case in cases:
        pair_id = case.get("pair_id")
        role = case.get("pair_role")
        if isinstance(pair_id, str) and role in {"vulnerable", "fixed"}:
            pairs.setdefault(pair_id, {})[role] = case["id"]

    results_by_trial = {
        (result.get("case_id"), result.get("repetition", 1)): result
        for result in case_results
    }
    repetitions = sorted({
        repetition for _, repetition in results_by_trial
        if isinstance(repetition, int)
    }) or [1]
    vulnerable_detected = fixed_not_flagged = pair_correct = pair_trials = 0
    for pair in pairs.values():
        vulnerable_id = pair.get("vulnerable")
        fixed_id = pair.get("fixed")
        if not vulnerable_id or not fixed_id:
            continue
        for repetition in repetitions:
            vulnerable = results_by_trial.get((vulnerable_id, repetition))
            fixed = results_by_trial.get((fixed_id, repetition))
            if vulnerable is None or fixed is None:
                continue
            pair_trials += 1
            detected = (
                not vulnerable.get("error")
                and vulnerable.get("true_positives") == 1
                and vulnerable.get("false_negatives") == 0
                and vulnerable.get("false_positives") == 0
            )
            clean = (
                not fixed.get("error")
                and fixed.get("false_positives") == 0
            )
            vulnerable_detected += int(detected)
            fixed_not_flagged += int(clean)
            pair_correct += int(detected and clean)
    return {
        "unique_pairs": len(pairs),
        "pair_trials": pair_trials,
        "vulnerable_detected": vulnerable_detected,
        "fixed_not_flagged": fixed_not_flagged,
        "both_correct": pair_correct,
        "paired_accuracy": round(pair_correct / pair_trials, 4) if pair_trials else None,
    }


def evaluate_security_models(
    client: Any,
    cases: List[Dict[str, Any]],
    models: List[Tuple[str, str]],
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
    repetitions: int = 1,
) -> Dict[str, Any]:
    """Run models against labeled cases and compute exact-quote precision/recall."""
    if not models:
        raise ValueError("At least one provider:model pair is required.")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("Repetitions must be a positive integer.")
    pricing = pricing or {}
    for model_key, rates in pricing.items():
        if not isinstance(model_key, str) or not isinstance(rates, dict):
            raise ValueError("Pricing entries must map model names to rate objects.")
        for rate_name in ("input_per_million", "output_per_million"):
            rate = rates.get(rate_name)
            if (
                not isinstance(rate, (int, float))
                or isinstance(rate, bool)
                or not math.isfinite(rate)
                or rate < 0
            ):
                raise ValueError(
                    f"Invalid {rate_name} for {model_key}; expected a finite non-negative number."
                )
    model_results: List[Dict[str, Any]] = []
    for provider, model in models:
        case_results: List[Dict[str, Any]] = []
        tp = fp = fn = errors = 0
        citations = grounded_citations = 0
        true_negative_cases = false_positive_clean_cases = 0
        failed_clean_trials = 0
        category_counts: Dict[str, Dict[str, int]] = {}
        latencies: List[float] = []
        estimated_cost = 0.0
        input_tokens_total = output_tokens_total = 0
        token_usage_known = True
        pricing_key = f"{provider}:{model}"
        model_pricing = pricing.get(pricing_key, pricing.get(model))
        cost_known = model_pricing is not None
        for repetition, case in (
            (repetition, case)
            for repetition in range(1, repetitions + 1)
            for case in cases
        ):
            prompt = (
                "Perform a conservative security review of the source below. "
                "Report only findings supported by the supplied source. Return "
                "JSON only with this shape: "
                '{"findings":[{"category":"one of: sql_injection, command_injection, path_traversal, unsafe_deserialization, authentication_authorization, sensitive_data_exposure",'
                '"file":"path from source context",'
                '"line":1,"evidence":"exact source quote"}]}. '
                "Cite the line where the finding materializes: the execute, query, "
                "eval, open, or loads call for injection; the unguarded privileged "
                "operation for authorization; the exposure statement or secret "
                "definition for sensitive data. "
                "Use an empty findings array when no finding is supported. Do not "
                "include explanations outside JSON.\n\n"
                f"### Task\n{case['prompt']}\n\n"
                f"### File\n{case.get('file', 'source.py')}\n\n"
                "### Source (line numbers are shown as N:; do not include them in evidence quotes)\n"
                "```text\n"
                + "\n".join(
                    f"{line_number}: {line}"
                    for line_number, line in enumerate(case["source"].splitlines(), start=1)
                )
                + "\n```"
            )
            started = time.perf_counter()
            raw_output = None
            try:
                response = client.chat_complete(
                    provider_id=provider,
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.0,
                    max_tokens=case.get("max_tokens", 1024),
                )
                latency_ms = round((time.perf_counter() - started) * 1000, 1)
                latencies.append(latency_ms)
                content = (
                    ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
                )
                raw_output = content if isinstance(content, str) else None
                predictions = _parse_findings(content)
                result = _evaluate_case(case, predictions)
                result["raw_output"] = raw_output
                if not case["expected_findings"]:
                    if predictions:
                        false_positive_clean_cases += 1
                    else:
                        true_negative_cases += 1
                usage = response.get("usage") if isinstance(response, dict) else None
                input_tokens = (
                    usage.get("prompt_tokens", usage.get("input_tokens"))
                    if isinstance(usage, dict) else None
                )
                output_tokens = (
                    usage.get("completion_tokens", usage.get("output_tokens"))
                    if isinstance(usage, dict) else None
                )
                valid_usage = all(
                    isinstance(token_count, int)
                    and not isinstance(token_count, bool)
                    and token_count >= 0
                    for token_count in (input_tokens, output_tokens)
                )
                if valid_usage:
                    input_tokens_total += input_tokens
                    output_tokens_total += output_tokens
                else:
                    token_usage_known = False
                if cost_known and valid_usage:
                    rates = model_pricing
                    estimated_cost += (
                        input_tokens * rates["input_per_million"]
                        + output_tokens * rates["output_per_million"]
                    ) / 1_000_000
                else:
                    cost_known = False
            except Exception as exc:
                latency_ms = round((time.perf_counter() - started) * 1000, 1)
                latencies.append(latency_ms)
                errors += 1
                token_usage_known = False
                if not case["expected_findings"]:
                    failed_clean_trials += 1
                result = {
                    "case_id": case["id"],
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": len(case["expected_findings"]),
                    "citations": 0,
                    "grounded_citations": 0,
                    "ungrounded_citations": 0,
                    "predicted_findings": [],
                    "missed": case["expected_findings"],
                    "false_positive_categories": [],
                    "raw_output": raw_output,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:240],
                }
            result["repetition"] = repetition
            for matched in result.get("matched", []):
                stats = category_counts.setdefault(
                    matched["category"].strip().lower(),
                    {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
                )
                stats["true_positives"] += 1
            for missed in result.get("missed", []):
                stats = category_counts.setdefault(
                    missed["category"].strip().lower(),
                    {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
                )
                stats["false_negatives"] += 1
            for category in result.get("false_positive_categories", []):
                stats = category_counts.setdefault(
                    category,
                    {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
                )
                stats["false_positives"] += 1
            tp += result["true_positives"]
            fp += result["false_positives"]
            fn += result["false_negatives"]
            citations += result.get("citations", 0)
            grounded_citations += result.get("grounded_citations", 0)
            case_results.append(result)
        ordered_latencies = sorted(latencies)
        p50_index = (len(ordered_latencies) - 1) // 2
        model_results.append({
            "provider": provider,
            "model": model,
            "method": "single_pass_model",
            "cases": len(cases) * repetitions,
            "unique_cases": len(cases),
            "repetitions": repetitions,
            "errors": errors,
            "failed_clean_trials": failed_clean_trials,
            "category_mismatches": sum(
                len(result.get("category_mismatches", []))
                for result in case_results
            ),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "citations": citations,
            "grounded_citations": grounded_citations,
            "ungrounded_citations": citations - grounded_citations,
            "citation_accuracy": (
                round(grounded_citations / citations, 4) if citations else None
            ),
            "citation_accuracy_ci95": _wilson_interval(grounded_citations, citations),
            **_score(tp, fp, fn),
            "recall_ci95": _wilson_interval(tp, tp + fn),
            "precision_ci95": _wilson_interval(tp, tp + fp),
            "true_negative_cases": true_negative_cases,
            "false_positive_clean_cases": false_positive_clean_cases,
            "specificity": (
                round(
                    true_negative_cases
                    / (true_negative_cases + false_positive_clean_cases),
                    4,
                )
                if true_negative_cases + false_positive_clean_cases
                else None
            ),
            "specificity_ci95": _wilson_interval(
                true_negative_cases,
                true_negative_cases + false_positive_clean_cases,
            ),
            "per_category": {
                category: {
                    **counts,
                    **_score(
                        counts["true_positives"],
                        counts["false_positives"],
                        counts["false_negatives"],
                    ),
                }
                for category, counts in sorted(category_counts.items())
            },
            "p50_latency_ms": ordered_latencies[p50_index] if ordered_latencies else None,
            "input_tokens": input_tokens_total if token_usage_known else None,
            "output_tokens": output_tokens_total if token_usage_known else None,
            "estimated_cost_usd": round(estimated_cost, 8) if cost_known else None,
            "paired_change": _paired_change_metrics(cases, case_results),
            "case_results": case_results,
        })
    return {
        "schema_version": 1,
        "metric": "exact_category_file_line_and_quoted_gold_evidence",
        "dataset_cases": len(cases),
        "repetitions": repetitions,
        "models": model_results,
        "limitations": [
            "This benchmark measures only the labels and cases supplied by the dataset author.",
            "A match is not proof of exploitability; labels must be independently reviewed.",
            "A missing result or malformed model response counts as missed gold findings and is reported as an error.",
            "Confidence intervals are 95% Wilson intervals and do not correct for correlated benchmark cases or repeated prompts.",
            "Estimated cost is shown only when pricing and token usage are available for every evaluated case.",
            "Do not route production audits from a small or unrepresentative dataset.",
        ],
    }


def _deterministic_findings(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate conservative source-pattern candidates for the local baseline."""
    import ast

    from omnicli.evidence import analyze_python_security_facts

    file_name = case.get("file", "source.py").replace("\\", "/")
    source = case["source"]
    lines = source.splitlines()
    report = analyze_python_security_facts({file_name: source})[0]
    findings: List[Dict[str, Any]] = []

    def emit(category: str, line: int) -> None:
        if 1 <= line <= len(lines):
            findings.append({
                "category": category,
                "file": file_name,
                "line": line,
                "evidence": lines[line - 1].strip(),
            })

    for function in report.get("functions", []):
        for sink in function.get("security_sinks", []):
            request_derived = bool(sink.get("request_derived_arguments"))
            if (
                sink.get("category") == "database_execution"
                and sink.get("dynamic_argument")
                and request_derived
            ):
                emit("sql_injection", sink["line"])
            elif (
                sink.get("category") == "command_execution"
                and sink.get("shell_enabled")
                and request_derived
            ):
                emit("command_injection", sink["line"])
            elif (
                sink.get("category") == "unsafe_deserialization"
                and request_derived
            ):
                emit("unsafe_deserialization", sink["line"])
            elif (
                sink.get("category") == "filesystem_access"
                and request_derived
            ):
                emit("path_traversal", sink["line"])

    for route in report.get("routes", []):
        if (
            route.get("mutating_calls")
            and not route.get("local_auth_guards")
        ):
            mutating_names = set(route["mutating_calls"])
            call_site = next(
                (
                    item for item in route.get("call_sites", [])
                    if item.get("call") in mutating_names
                ),
                None,
            )
            if call_site:
                emit("authentication_authorization", call_site["line"])

    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        sensitive_names = {
            "password", "secret", "token", "credential", "ssn", "private_key",
            "api_key",
        }
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            assignments = [
                node for node in ast.walk(function)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
            ]
            secret_variables = set()
            dependencies: Dict[str, set[str]] = {}
            for assignment in assignments:
                value = assignment.value
                if value is None:
                    continue
                targets = (
                    assignment.targets if isinstance(assignment, ast.Assign)
                    else [assignment.target]
                )
                target_names = {
                    target.id for target in targets if isinstance(target, ast.Name)
                }
                value_names = {
                    node.id for node in ast.walk(value)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                }
                dependencies.update({
                    name: value_names for name in target_names
                })
                reads_environment = any(
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "os"
                    and node.attr == "environ"
                    for node in ast.walk(value)
                ) or any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "getenv"
                    for node in ast.walk(value)
                )
                names_sensitive_value = any(
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and any(marker in node.value.lower() for marker in sensitive_names)
                    for node in ast.walk(value)
                ) or any(
                    any(marker in name.lower() for marker in sensitive_names)
                    for name in target_names
                )
                if reads_environment and names_sensitive_value:
                    secret_variables.update(target_names)
            for _ in range(len(dependencies)):
                for target_name, value_names in dependencies.items():
                    if value_names.intersection(secret_variables):
                        secret_variables.add(target_name)
            for node in ast.walk(function):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue
                returned_variables = {
                    child.id for child in ast.walk(node.value)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                }
                direct_secret_return = any(
                    isinstance(child, ast.Attribute)
                    and isinstance(child.value, ast.Name)
                    and child.value.id == "os"
                    and child.attr == "environ"
                    for child in ast.walk(node.value)
                )
                direct_secret_key = any(
                    isinstance(child, ast.Constant)
                    and isinstance(child.value, str)
                    and any(marker in child.value.lower() for marker in sensitive_names)
                    for child in ast.walk(node.value)
                )
                if (
                    direct_secret_return and direct_secret_key
                    or isinstance(node.value, ast.Name)
                    and bool(returned_variables.intersection(secret_variables))
                ):
                    emit("sensitive_data_exposure", node.lineno)

    return findings


def evaluate_deterministic_security(
    cases: List[Dict[str, Any]],
    repetitions: int = 1,
) -> Dict[str, Any]:
    """Score a bounded local-pattern baseline against the same labeled cases."""
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("Repetitions must be a positive integer.")
    true_positives = false_positives = false_negatives = 0
    citations = grounded_citations = 0
    true_negative_cases = false_positive_clean_cases = 0
    case_results = []
    category_counts: Dict[str, Dict[str, int]] = {}
    for repetition in range(1, repetitions + 1):
        for case in cases:
            predictions = _deterministic_findings(case)
            result = _evaluate_case(case, predictions)
            result["repetition"] = repetition
            if not case["expected_findings"]:
                if predictions:
                    false_positive_clean_cases += 1
                else:
                    true_negative_cases += 1
            for matched in result["matched"]:
                stats = category_counts.setdefault(
                    matched["category"].strip().lower(),
                    {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
                )
                stats["true_positives"] += 1
            for missed in result["missed"]:
                stats = category_counts.setdefault(
                    missed["category"].strip().lower(),
                    {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
                )
                stats["false_negatives"] += 1
            for category in result["false_positive_categories"]:
                stats = category_counts.setdefault(
                    category,
                    {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
                )
                stats["false_positives"] += 1
            true_positives += result["true_positives"]
            false_positives += result["false_positives"]
            false_negatives += result["false_negatives"]
            citations += result["citations"]
            grounded_citations += result["grounded_citations"]
            case_results.append(result)

    specificity_total = true_negative_cases + false_positive_clean_cases
    return {
        "provider": "local",
        "model": "python-static-v1",
        "method": "deterministic_static_baseline",
        "cases": len(cases) * repetitions,
        "unique_cases": len(cases),
        "repetitions": repetitions,
        "errors": 0,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "citations": citations,
        "grounded_citations": grounded_citations,
        "ungrounded_citations": citations - grounded_citations,
        "citation_accuracy": (
            round(grounded_citations / citations, 4) if citations else None
        ),
        "citation_accuracy_ci95": _wilson_interval(grounded_citations, citations),
        **_score(true_positives, false_positives, false_negatives),
        "recall_ci95": _wilson_interval(
            true_positives, true_positives + false_negatives
        ),
        "precision_ci95": _wilson_interval(
            true_positives, true_positives + false_positives
        ),
        "true_negative_cases": true_negative_cases,
        "false_positive_clean_cases": false_positive_clean_cases,
        "specificity": (
            round(true_negative_cases / specificity_total, 4)
            if specificity_total else None
        ),
        "specificity_ci95": _wilson_interval(
            true_negative_cases, specificity_total
        ),
        "per_category": {
            category: {
                **counts,
                **_score(
                    counts["true_positives"],
                    counts["false_positives"],
                    counts["false_negatives"],
                ),
            }
            for category, counts in sorted(category_counts.items())
        },
        "p50_latency_ms": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "paired_change": _paired_change_metrics(cases, case_results),
        "case_results": case_results,
    }


def staged_trial_failed(
    gate_status: str,
    raw_outputs: list,
    verdicts: Optional[Iterable[object]] = (),
) -> bool:
    """A staged trial fails when the gate fails or zero members survived.

    One throttled member must not invalidate survivors' grounded output;
    member errors are recorded on the trial, but only total failure —
    gate FAIL or every member errored — invalidates it. Staged predictions
    require judge verdicts, so emitted verdicts with zero adjudicated
    (UPHOLD/KILL/CONTESTED) outcomes also invalidate the trial:
    adjudication was attempted but produced nothing scorable. A CONTESTED
    deadlock counts as adjudicated — the council completed with no
    consensus, which scores on the merits (zero predictions), unlike an
    outage (UNJUDGED), which invalidates.
    """
    if gate_status != "PASS_GROUNDING_ONLY":
        return True
    judged = [v for v in (verdicts or []) if isinstance(v, dict)]
    if judged and not any(
        v.get("verdict") in ("UPHOLD", "KILL", "CONTESTED") for v in judged
    ):
        return True
    return not any(isinstance(m, dict) and not m.get("error") for m in raw_outputs)


def evaluate_staged_security(
    cfg: Any,
    client: Any,
    cases: List[Dict[str, Any]],
    models: List[Tuple[str, str]],
    repetitions: int = 1,
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Evaluate the staged red-team pipeline on the same labeled cases.

    Each model runs the staged pipeline solo (per-model isolation for
    benchmarking); predictions are the typed judge's UPHOLD verdicts, not
    raw attack claims. Invalid trials (gate FAIL, total member failure)
    still score their predictions on the merits — true positives are never
    zeroed — but the trial is flagged (trial_invalid) and counted in
    errors/invalid_trials, which fail the quality gate. The penalty for an
    invalid trial is gate failure, not erased detections.
    """
    from omnicli.orchestrate import swarm_redteam

    if not models:
        raise ValueError("At least one provider:model pair is required.")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("Repetitions must be a positive integer.")
    pricing = pricing or {}
    for model_key, rates in pricing.items():
        if not isinstance(model_key, str) or not isinstance(rates, dict):
            raise ValueError("Pricing entries must map model names to rate objects.")
        for rate_name in ("input_per_million", "output_per_million"):
            rate = rates.get(rate_name)
            if (
                not isinstance(rate, (int, float))
                or isinstance(rate, bool)
                or not math.isfinite(rate)
                or rate < 0
            ):
                raise ValueError(
                    f"Invalid {rate_name} for {model_key}; expected a finite non-negative number."
                )
    model_results = []
    for provider, model in models:
        tp = fp = fn = errors = gate_failures = citations = grounded_citations = 0
        clean_true_negatives = clean_false_positives = 0
        failed_clean_trials = 0
        invalid_trials = 0
        category_counts: Dict[str, Dict[str, int]] = {}
        case_results = []
        latencies: List[float] = []
        input_tokens_total = output_tokens_total = 0
        tokens_known = True
        costs: List[float] = []
        cost_known = True
        member_calls_total = 0
        model_rates = pricing.get(f"{provider}:{model}", pricing.get(model))

        for repetition in range(1, repetitions + 1):
            for case in cases:
                case_id = case["id"]
                path = case.get("file", "source.py").replace("\\", "/")
                raw_outputs: List[Dict[str, Any]] = []
                try:
                    review = swarm_redteam(
                        cfg,
                        client,
                        case["prompt"],
                        max_tokens=case.get("max_tokens", 1024),
                        synth=False,
                        swarm=[(provider, model)],
                        source_files={path: case["source"]},
                        temperature=0.0,
                    )
                    raw_outputs = [
                        {
                            "provider": member.get("provider"),
                            "model": member.get("model"),
                            "answer": member.get("answer"),
                            "error": member.get("error"),
                            "attack_validations": member.get("attack_validations", []),
                        }
                        for member in review.get("members", [])
                    ]
                    metrics = review.get("metrics", {})
                    member_calls_total += metrics.get("member_calls", 0)
                    reported_cost = metrics.get("estimated_cost_usd")
                    if isinstance(reported_cost, (int, float)) and not isinstance(reported_cost, bool):
                        costs.append(float(reported_cost))
                    else:
                        cost_known = False
                    latency = metrics.get("elapsed_ms")
                    if isinstance(latency, (int, float)):
                        latencies.append(float(latency))
                    in_tokens = metrics.get("input_tokens_reported")
                    out_tokens = metrics.get("output_tokens_reported")
                    if (
                        isinstance(in_tokens, int) and not isinstance(in_tokens, bool)
                        and isinstance(out_tokens, int) and not isinstance(out_tokens, bool)
                    ):
                        input_tokens_total += in_tokens
                        output_tokens_total += out_tokens
                    else:
                        tokens_known = False

                    case_citations = case_grounded = 0
                    for validation in review.get("attack_validations", []):
                        if not isinstance(validation, dict):
                            continue
                        for finding in validation.get("findings_checked", []):
                            if not isinstance(finding, dict):
                                continue
                            case_citations += 1
                            case_grounded += bool(finding.get("grounded"))
                    typed_judgment = review.get("typed_judgment") or {}
                    predictions = []
                    seen_predictions = set()
                    for adjudication in typed_judgment.get("verdicts", []):
                        if (
                            not isinstance(adjudication, dict)
                            or adjudication.get("verdict") != "UPHOLD"
                        ):
                            continue
                        candidate = adjudication.get("candidate") or {}
                        prediction = {
                            "category": candidate.get("category", "unknown"),
                            "file": candidate.get("file", ""),
                            "line": candidate.get("start_line", 0),
                            "evidence": candidate.get("quote", ""),
                        }
                        identity = (
                            prediction["category"],
                            prediction["file"],
                            prediction["line"],
                            prediction["evidence"],
                        )
                        if identity not in seen_predictions:
                            seen_predictions.add(identity)
                            predictions.append(prediction)
                    result = _evaluate_case(case, predictions)
                    result.update({
                        "citations": case_citations,
                        "grounded_citations": case_grounded,
                        "ungrounded_citations": case_citations - case_grounded,
                        "raw_outputs": raw_outputs,
                        "evidence_gate": review.get("evidence_gate"),
                    })
                    evidence_gate = review.get("evidence_gate") or {}
                    member_errors = [item.get("error") for item in raw_outputs if item.get("error")]
                    gate_failed = evidence_gate.get("status") == "FAIL"
                    trial_failed = staged_trial_failed(
                        evidence_gate.get("status", ""),
                        raw_outputs,
                        verdicts=typed_judgment.get("verdicts", []),
                    )
                    if gate_failed:
                        gate_failures += 1
                    result["trial_invalid"] = bool(trial_failed)
                    if trial_failed:
                        errors += 1
                        invalid_trials += 1
                        result["trial_invalid_reason"] = (
                            "; ".join(member_errors)[:240]
                            if member_errors
                            else "Staged review evidence gate failed."
                        )
                        result["invalid_trial_observations"] = {
                            "predicted_findings": predictions,
                            "false_positives": result["false_positives"],
                            "category_mismatches": result.get(
                                "category_mismatches", []
                            ),
                            "citations": case_citations,
                            "grounded_citations": case_grounded,
                        }
                        result["error"] = result["trial_invalid_reason"]
                        # Scored on the merits above: no zeroing. The trial
                        # is penalized via errors/invalid_trials (which fail
                        # the quality gate), not by erasing detections.
                    citations += case_citations
                    grounded_citations += case_grounded
                    if not case["expected_findings"]:
                        if trial_failed:
                            failed_clean_trials += 1
                        elif predictions:
                            clean_false_positives += 1
                        else:
                            clean_true_negatives += 1
                except Exception as exc:
                    errors += 1
                    gate_failures += 1
                    invalid_trials += 1
                    tokens_known = False
                    result = {
                        "case_id": case_id,
                        "true_positives": 0,
                        "false_positives": 0,
                        "false_negatives": len(case["expected_findings"]),
                        "citations": 0,
                        "grounded_citations": 0,
                        "ungrounded_citations": 0,
                        "predicted_findings": [],
                        "matched": [],
                        "missed": case["expected_findings"],
                        "false_positive_categories": [],
                        "raw_outputs": raw_outputs,
                        "error": str(exc)[:240],
                        "trial_invalid": True,
                        "trial_invalid_reason": f"Crashed trial: {str(exc)[:200]}",
                    }
                    if not case["expected_findings"]:
                        failed_clean_trials += 1
                result["repetition"] = repetition
                for matched in result.get("matched", []):
                    counts = category_counts.setdefault(
                        matched["category"].strip().lower(),
                        {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
                    )
                    counts["true_positives"] += 1
                for missed in result.get("missed", []):
                    counts = category_counts.setdefault(
                        missed["category"].strip().lower(),
                        {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
                    )
                    counts["false_negatives"] += 1
                for category in result.get("false_positive_categories", []):
                    counts = category_counts.setdefault(
                        category,
                        {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
                    )
                    counts["false_positives"] += 1
                tp += result["true_positives"]
                fp += result["false_positives"]
                fn += result["false_negatives"]
                case_results.append(result)

        ordered_latencies = sorted(latencies)
        total_clean = clean_true_negatives + clean_false_positives
        model_results.append({
            "provider": provider,
            "model": model,
            "method": "staged_redteam",
            "cases": len(cases) * repetitions,
            "unique_cases": len(cases),
            "repetitions": repetitions,
            "errors": errors,
            "failed_clean_trials": failed_clean_trials,
            "invalid_trials": invalid_trials,
            "invalid_trial_rate": (
                round(invalid_trials / (len(cases) * repetitions), 4)
                if cases and repetitions else None
            ),
            "category_mismatches": sum(
                len(result.get("category_mismatches", []))
                for result in case_results
            ),
            "evidence_gate_failures": gate_failures,
            "member_calls": member_calls_total,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "citations": citations,
            "grounded_citations": grounded_citations,
            "ungrounded_citations": citations - grounded_citations,
            "citation_accuracy": (
                round(grounded_citations / citations, 4) if citations else None
            ),
            "citation_accuracy_ci95": _wilson_interval(grounded_citations, citations),
            **_score(tp, fp, fn),
            "recall_ci95": _wilson_interval(tp, tp + fn),
            "precision_ci95": _wilson_interval(tp, tp + fp),
            "true_negative_cases": clean_true_negatives,
            "false_positive_clean_cases": clean_false_positives,
            "specificity": (
                round(clean_true_negatives / total_clean, 4) if total_clean else None
            ),
            "specificity_ci95": _wilson_interval(clean_true_negatives, total_clean),
            "per_category": {
                category: {
                    **counts,
                    **_score(
                        counts["true_positives"],
                        counts["false_positives"],
                        counts["false_negatives"],
                    ),
                }
                for category, counts in sorted(category_counts.items())
            },
            "p50_latency_ms": (
                ordered_latencies[(len(ordered_latencies) - 1) // 2]
                if ordered_latencies else None
            ),
            "input_tokens": input_tokens_total if tokens_known else None,
            "output_tokens": output_tokens_total if tokens_known else None,
            "estimated_cost_usd": (
                round(
                    (
                        input_tokens_total * model_rates["input_per_million"]
                        + output_tokens_total * model_rates["output_per_million"]
                    ) / 1_000_000,
                    8,
                )
                if model_rates is not None and tokens_known
                else round(sum(costs), 8) if cost_known and costs else None
            ),
            "paired_change": _paired_change_metrics(cases, case_results),
            "case_results": case_results,
        })
    return {
        "schema_version": 1,
        "metric": "exact_category_file_line_and_quoted_gold_evidence",
        "dataset_cases": len(cases),
        "repetitions": repetitions,
        "models": model_results,
        "limitations": [
            "Staged reviews are serialized per case and issue four focus calls plus a challenge call.",
            "Category accuracy depends on the model-provided CATEGORY label; unknown or incorrect categories score as false positives.",
            "Repeated trials on a fixed case set are correlated; Wilson intervals do not model this dependence.",
            "Quote grounding and the fail-closed evidence gate are not proof of exploitability or completeness.",
        ],
    }


def evaluate_staged_council(
    cfg: Any,
    client: Any,
    cases: List[Dict[str, Any]],
    models: List[Tuple[str, str]],
    repetitions: int = 1,
) -> Dict[str, Any]:
    """Evaluate the actual multi-member red-team council on labeled cases.

    Predictions are the typed judge's UPHOLD verdicts over the full council
    run. Invalid trials score on the merits (true positives are never
    zeroed) and are flagged via trial_invalid; errors/invalid_trials still
    fail the quality gate.
    """
    from omnicli.orchestrate import swarm_redteam

    if len(models) < 2:
        raise ValueError("A staged council evaluation requires at least two models.")
    if len(set(models)) != len(models):
        raise ValueError("A staged council cannot contain duplicate provider/model pairs.")
    if not cases:
        raise ValueError("At least one evaluation case is required.")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("Repetitions must be a positive integer.")

    case_results: List[Dict[str, Any]] = []
    tp = fp = fn = errors = gate_failures = citations = grounded_citations = 0
    clean_true_negatives = clean_false_positives = failed_clean_trials = 0
    invalid_trials = 0
    member_calls = input_tokens = output_tokens = 0
    usage_known = True
    latencies: List[float] = []
    for repetition in range(1, repetitions + 1):
        for case in cases:
            raw_outputs: List[Dict[str, Any]] = []
            try:
                review = swarm_redteam(
                    cfg,
                    client,
                    case["prompt"],
                    max_tokens=case.get("max_tokens", 1024),
                    synth=False,
                    swarm=models,
                    source_files={
                        case.get("file", "source.py").replace("\\", "/"): case["source"]
                    },
                    temperature=0.0,
                )
                raw_outputs = [
                    {
                        key: member.get(key)
                        for key in (
                            "provider", "model", "family", "answer", "error",
                            "stage_evidence_status", "usage",
                        )
                    }
                    for member in review.get("members", [])
                ]
                metrics = review.get("metrics", {})
                member_calls += metrics.get("member_calls", 0)
                if isinstance(metrics.get("elapsed_ms"), (int, float)):
                    latencies.append(float(metrics["elapsed_ms"]))
                reported_input = metrics.get("input_tokens_reported")
                reported_output = metrics.get("output_tokens_reported")
                if (
                    isinstance(reported_input, int) and not isinstance(reported_input, bool)
                    and isinstance(reported_output, int) and not isinstance(reported_output, bool)
                ):
                    input_tokens += reported_input
                    output_tokens += reported_output
                else:
                    usage_known = False

                predictions: List[Dict[str, Any]] = []
                seen = set()
                case_citations = case_grounded = 0
                for validation in review.get("attack_validations", []):
                    if not isinstance(validation, dict):
                        continue
                    for finding in validation.get("findings_checked", []):
                        if not isinstance(finding, dict):
                            continue
                        case_citations += 1
                        case_grounded += int(bool(finding.get("grounded")))
                typed_judgment = review.get("typed_judgment") or {}
                for adjudication in typed_judgment.get("verdicts", []):
                    if not isinstance(adjudication, dict) or adjudication.get("verdict") != "UPHOLD":
                        continue
                    candidate = adjudication.get("candidate") or {}
                    prediction = {
                        "category": candidate.get("category", "unknown"),
                        "file": candidate.get("file", ""),
                        "line": candidate.get("start_line", 0),
                        "evidence": candidate.get("quote", ""),
                    }
                    identity = (
                        prediction["category"], prediction["file"],
                        prediction["line"], prediction["evidence"],
                    )
                    if identity not in seen:
                        seen.add(identity)
                        predictions.append(prediction)
                result = _evaluate_case(case, predictions)
                gate = review.get("evidence_gate") or {}
                member_errors = [
                    member.get("error") for member in raw_outputs if member.get("error")
                ]
                trial_failed = staged_trial_failed(
                    gate.get("status", ""),
                    raw_outputs,
                    verdicts=typed_judgment.get("verdicts", []),
                )
                result.update({
                    "citations": case_citations,
                    "grounded_citations": case_grounded,
                    "ungrounded_citations": case_citations - case_grounded,
                    "evidence_gate": gate,
                    "raw_outputs": raw_outputs,
                    "review_diagnostics": {
                        "stage_calls": review.get("stage_calls", []),
                        "attack_validations": review.get("attack_validations", []),
                        "challenge_validations": review.get("challenge_validations", []),
                        "candidate_triage": review.get("candidate_triage", []),
                        "typed_judgment": typed_judgment,
                        "scenario_coverage": review.get("scenario_coverage", []),
                        "evidence_gap_cues": review.get("evidence_gap_cues", []),
                    },
                })
                result["trial_invalid"] = bool(trial_failed)
                if trial_failed:
                    errors += 1
                    invalid_trials += 1
                    gate_failures += int(gate.get("status") != "PASS_GROUNDING_ONLY")
                    result["trial_invalid_reason"] = (
                        "; ".join(str(error) for error in member_errors)[:240]
                        if member_errors else "Council evidence gate failed."
                    )
                    result["invalid_trial_observations"] = {
                        "predicted_findings": predictions,
                        "false_positives": result["false_positives"],
                        "category_mismatches": result.get("category_mismatches", []),
                    }
                    result["error"] = result["trial_invalid_reason"]
                    # Scored on the merits above: no zeroing. Penalty is
                    # errors/invalid_trials (quality-gate failure), not
                    # erased detections.
                citations += case_citations
                grounded_citations += case_grounded
                if not case["expected_findings"]:
                    if trial_failed:
                        failed_clean_trials += 1
                    elif predictions:
                        clean_false_positives += 1
                    else:
                        clean_true_negatives += 1
            except Exception as exc:
                errors += 1
                gate_failures += 1
                invalid_trials += 1
                usage_known = False
                result = {
                    "case_id": case["id"],
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": len(case["expected_findings"]),
                    "citations": 0,
                    "grounded_citations": 0,
                    "ungrounded_citations": 0,
                    "predicted_findings": [],
                    "matched": [],
                    "missed": case["expected_findings"],
                    "false_positive_categories": [],
                    "raw_outputs": raw_outputs,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:240],
                    "trial_invalid": True,
                    "trial_invalid_reason": f"Crashed trial: {str(exc)[:200]}",
                }
                if not case["expected_findings"]:
                    failed_clean_trials += 1
            result["repetition"] = repetition
            tp += result["true_positives"]
            fp += result["false_positives"]
            fn += result["false_negatives"]
            case_results.append(result)

    ordered_latencies = sorted(latencies)
    known_clean_trials = clean_true_negatives + clean_false_positives
    return {
        "schema_version": 1,
        "method": "staged_redteam_council",
        "models": [{"provider": provider, "model": model} for provider, model in models],
        "dataset_cases": len(cases),
        "repetitions": repetitions,
        "member_calls": member_calls,
        "errors": errors,
        "invalid_trials": invalid_trials,
        "invalid_trial_rate": (
            round(invalid_trials / (len(cases) * repetitions), 4)
            if cases and repetitions else None
        ),
        "evidence_gate_failures": gate_failures,
        "category_mismatches": sum(
            len(result.get("category_mismatches", [])) for result in case_results
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "citations": citations,
        "grounded_citations": grounded_citations,
        "ungrounded_citations": citations - grounded_citations,
        "citation_accuracy": (
            round(grounded_citations / citations, 4) if citations else None
        ),
        **_score(tp, fp, fn),
        "true_negative_cases": clean_true_negatives,
        "false_positive_clean_cases": clean_false_positives,
        "failed_clean_trials": failed_clean_trials,
        "specificity": (
            round(clean_true_negatives / known_clean_trials, 4)
            if known_clean_trials else None
        ),
        "p50_latency_ms": (
            ordered_latencies[(len(ordered_latencies) - 1) // 2]
            if ordered_latencies else None
        ),
        "input_tokens": input_tokens if usage_known else None,
        "output_tokens": output_tokens if usage_known else None,
        "estimated_cost_usd": None,
        "case_results": case_results,
        "limitations": [
            "Council detections are the deduplicated union of member citations, not exploitability proof.",
            "Failed council trials are reported separately and excluded from ordinary finding metrics.",
            "Cost is unavailable unless provider pricing and complete usage are supplied.",
        ],
    }


def assess_redteam_quality_gate(
    cases: List[Dict[str, Any]],
    single_pass: List[Dict[str, Any]],
    staged: List[Dict[str, Any]],
    minimum_per_class: int = 30,
    minimum_families: int = 3,
    minimum_repetitions: int = 2,
) -> Dict[str, Any]:
    """Assess the explicit held-out transition gate without conflating modes."""
    from omnicli.scanner import classify_family

    failures = []
    if not cases or any(case.get("split") != "held_out" for case in cases):
        failures.append("Quality-gate evaluation requires only held_out cases.")
    if any(
        case.get("adjudication_status") != "independently_reviewed"
        or case.get("reviewer_count") != 2
        for case in cases
    ):
        failures.append(
            "Every held-out case must be independently adjudicated by two reviewers."
        )
    positive_cases = sum(bool(case.get("expected_findings")) for case in cases)
    benign_cases = len(cases) - positive_cases
    if positive_cases < minimum_per_class:
        failures.append(
            f"Need at least {minimum_per_class} unique held-out positive cases; found {positive_cases}."
        )
    if benign_cases < minimum_per_class:
        failures.append(
            f"Need at least {minimum_per_class} unique held-out benign cases; found {benign_cases}."
        )

    direct_by_model = {
        (item.get("provider"), item.get("model")): item for item in single_pass
    }
    staged_by_model = {
        (item.get("provider"), item.get("model")): item for item in staged
    }
    direct_keys = [(item.get("provider"), item.get("model")) for item in single_pass]
    staged_keys = [(item.get("provider"), item.get("model")) for item in staged]
    if len(set(direct_keys)) != len(direct_keys):
        failures.append("Single-pass results contain duplicate provider/model rows.")
    if len(set(staged_keys)) != len(staged_keys):
        failures.append("Staged results contain duplicate provider/model rows.")
    all_models = set(direct_by_model) | set(staged_by_model)
    paired = sorted(set(direct_by_model).intersection(staged_by_model))
    for pair in sorted(all_models - set(paired)):
        failures.append(
            f"{pair[0]}:{pair[1]} is missing a paired single-pass or staged result."
        )
    failed_pairs = [
        pair for pair in paired
        if (
            direct_by_model[pair].get("errors", 0) != 0
            or staged_by_model[pair].get("errors", 0) != 0
            or staged_by_model[pair].get("evidence_gate_failures", 0) != 0
        )
    ]
    for pair in failed_pairs:
        failures.append(
            f"{pair[0]}:{pair[1]} has provider or evidence-gate failures; "
            "failed trials cannot be excluded from the quality gate."
        )
    healthy_pairs = [
        pair for pair in paired
        if (
            direct_by_model[pair].get("errors", 0) == 0
            and staged_by_model[pair].get("errors", 0) == 0
            and staged_by_model[pair].get("evidence_gate_failures", 0) == 0
        )
    ]
    healthy_families = sorted({
        classify_family(model)
        for _provider, model in healthy_pairs
        if classify_family(model) != "other"
    })
    if len(healthy_families) < minimum_families:
        failures.append(
            f"Need {minimum_families} healthy model families with both modes; "
            f"found {len(healthy_families)}."
        )
    if len(healthy_pairs) < minimum_families:
        failures.append("Each qualifying model family needs paired single-pass and staged results.")
    for pair in paired:
        for method, result in (
            ("single-pass", direct_by_model[pair]),
            ("staged", staged_by_model[pair]),
        ):
            if result.get("repetitions", 0) < minimum_repetitions:
                failures.append(
                    f"{pair[0]}:{pair[1]} {method} used fewer than "
                    f"{minimum_repetitions} repetitions."
                )

    def pooled(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        tp = sum(row.get("true_positives", 0) for row in rows)
        fp = sum(row.get("false_positives", 0) for row in rows)
        fn = sum(row.get("false_negatives", 0) for row in rows)
        tn = sum(row.get("true_negative_cases", 0) for row in rows)
        clean_fp = sum(row.get("false_positive_clean_cases", 0) for row in rows)
        grounded = sum(row.get("grounded_citations", 0) for row in rows)
        citations = sum(row.get("citations", 0) for row in rows)
        scores = {
            **_score(tp, fp, fn),
            "specificity": (
                round(tn / (tn + clean_fp), 4) if tn + clean_fp else None
            ),
            "citation_accuracy": (
                round(grounded / citations, 4) if citations else None
            ),
            "recall_ci95": _wilson_interval(tp, tp + fn),
            "specificity_ci95": _wilson_interval(tn, tn + clean_fp),
            "citation_accuracy_ci95": _wilson_interval(grounded, citations),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negative_cases": tn,
            "false_positive_clean_cases": clean_fp,
            "grounded_citations": grounded,
            "citations": citations,
        }
        category_totals: Dict[str, Dict[str, int]] = {}
        for row in rows:
            for category, values in row.get("per_category", {}).items():
                totals = category_totals.setdefault(
                    category,
                    {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
                )
                for key in totals:
                    totals[key] += values.get(key, 0)
        scores["per_category"] = {
            category: {
                **counts,
                **_score(
                    counts["true_positives"],
                    counts["false_positives"],
                    counts["false_negatives"],
                ),
            }
            for category, counts in sorted(category_totals.items())
        }
        return scores

    def by_family(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        families: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            family = classify_family(row["model"])
            if family != "other":
                families.setdefault(family, []).append(row)
        return families

    direct_rows = [direct_by_model[pair] for pair in healthy_pairs]
    staged_rows = [staged_by_model[pair] for pair in healthy_pairs]
    direct_families = by_family(direct_rows)
    staged_families = by_family(staged_rows)
    direct_scores = pooled(direct_rows)
    staged_scores = pooled(staged_rows)
    for method, scores in (("single-pass", direct_scores), ("staged", staged_scores)):
        if scores["recall"] < 0.85:
            failures.append(f"{method} pooled recall is below 85%.")
        if scores["specificity"] is None or scores["specificity"] < 0.90:
            failures.append(f"{method} pooled specificity is below 90%.")
        if scores["citation_accuracy"] is None or scores["citation_accuracy"] < 0.95:
            failures.append(f"{method} exact citation accuracy is below 95%.")
    if staged_scores["recall"] < direct_scores["recall"]:
        failures.append("Staged mode recall regresses against the single-pass baseline.")

    per_family: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for family in sorted(set(direct_families) | set(staged_families)):
        family_direct = pooled(direct_families.get(family, []))
        family_staged = pooled(staged_families.get(family, []))
        per_family[family] = {
            "single_pass": family_direct,
            "staged": family_staged,
        }
        for method, scores in (
            ("single-pass", family_direct),
            ("staged", family_staged),
        ):
            if scores["recall"] < 0.85:
                failures.append(
                    f"{family} {method} recall is below 85%."
                )
            if scores["specificity"] is None or scores["specificity"] < 0.90:
                failures.append(
                    f"{family} {method} specificity is below 90%."
                )
            if scores["citation_accuracy"] is None or scores["citation_accuracy"] < 0.95:
                failures.append(
                    f"{family} {method} exact citation accuracy is below 95%."
                )
        if family_staged["recall"] < family_direct["recall"]:
            failures.append(
                f"{family} staged mode recall regresses against the single-pass baseline."
            )

    return {
        "status": "FAIL" if failures else "PASS_BOUNDED_GATE",
        "failures": failures,
        "held_out_cases": len(cases),
        "positive_cases": positive_cases,
        "benign_cases": benign_cases,
        "healthy_model_families": healthy_families,
        "paired_models": [
            {"provider": provider, "model": model} for provider, model in healthy_pairs
        ],
        "failed_models": [
            {"provider": provider, "model": model} for provider, model in failed_pairs
        ],
        "single_pass": direct_scores,
        "staged": staged_scores,
        "per_family": per_family,
        "limitations": [
            "Passing qualifies only for this reviewed held-out set, roster, and measured configuration.",
            "Wilson intervals are descriptive and do not account for correlations between repeated trials.",
            "A pass is not proof of universal coverage, exploitability, or bounty eligibility.",
        ],
    }


# Binding Transition-Gate bars (docs/REDTEAM_QUALITY_PLAN.md phase 5, as
# enforced by assess_redteam_quality_gate). These are stricter than the
# looser numbers in the roadmap's Task-3 checklist; the plan and the code
# are authoritative. If assess_redteam_quality_gate ever changes its bars,
# this validator must be updated in the same edit — silent drift between
# the gate and its artifact check would let a stale artifact pass.
_TRANSITION_BAR_RECALL = 0.85
_TRANSITION_BAR_SPECIFICITY = 0.90
_TRANSITION_BAR_CITATION = 0.95
_TRANSITION_MINIMUM_PER_CLASS = 30
_TRANSITION_MINIMUM_FAMILIES = 3
_TRANSITION_MINIMUM_REPETITIONS = 2


def validate_qualifying_run(report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a claimed Transition-Gate qualifying-run artifact.

    Takes the parsed JSON written by ``omni evaluate-security ... --split
    held_out --staged-redteam --quality-gate --output <artifact>`` and
    re-derives the gate anatomy independently instead of trusting the
    artifact's own summary strings: model families are recomputed from the
    paired rows, pooled and per-family bars are re-checked, and every
    paired model must show both a single-pass and a staged row with enough
    repetitions and zero excluded failures.

    Returns ``{"valid": bool, "failures": [...], "summary": {...}}``. A
    hand-edited or partial artifact fails with explicit reasons; only a
    genuine PASS_BOUNDED_GATE report validates.
    """
    from omnicli.scanner import classify_family

    failures: List[str] = []
    if not isinstance(report, dict):
        return {
            "valid": False,
            "failures": ["Qualifying-run artifact must be a JSON object."],
            "summary": {},
        }
    gate = report.get("quality_gate")
    if not isinstance(gate, dict):
        failures.append("Artifact is missing its quality_gate section.")
        gate = {}
    elif gate.get("status") != "PASS_BOUNDED_GATE":
        failures.append(
            "quality_gate.status is "
            f"{gate.get('status')!r}, not PASS_BOUNDED_GATE."
        )
    if isinstance(gate, dict) and gate.get("failures"):
        failures.append(
            "quality_gate.failures is not empty; a qualifying run has none."
        )
    positive_cases = gate.get("positive_cases", 0) if isinstance(gate, dict) else 0
    benign_cases = gate.get("benign_cases", 0) if isinstance(gate, dict) else 0
    if not isinstance(positive_cases, int) or positive_cases < _TRANSITION_MINIMUM_PER_CLASS:
        failures.append(
            f"Need at least {_TRANSITION_MINIMUM_PER_CLASS} held-out positive "
            f"cases; artifact claims {positive_cases}."
        )
    if not isinstance(benign_cases, int) or benign_cases < _TRANSITION_MINIMUM_PER_CLASS:
        failures.append(
            f"Need at least {_TRANSITION_MINIMUM_PER_CLASS} held-out benign "
            f"cases; artifact claims {benign_cases}."
        )
    if isinstance(gate, dict) and gate.get("failed_models"):
        failures.append("Artifact lists failed models; a qualifying run has none.")

    paired = gate.get("paired_models", []) if isinstance(gate, dict) else []
    if not isinstance(paired, list) or not paired:
        failures.append("Artifact lists no paired single-pass/staged models.")
        paired = []
    rederived_families = set()
    for entry in paired:
        if not isinstance(entry, dict) or not entry.get("provider") or not entry.get("model"):
            failures.append(f"Paired-model entry is malformed: {entry!r}.")
            continue
        family = classify_family(str(entry["model"]))
        if family != "other":
            rederived_families.add(family)
    if len(rederived_families) < _TRANSITION_MINIMUM_FAMILIES:
        failures.append(
            f"Need {_TRANSITION_MINIMUM_FAMILIES} distinct model families "
            f"re-derived from paired rows; found {len(rederived_families)}."
        )
    claimed_families = gate.get("healthy_model_families") if isinstance(gate, dict) else None
    if not isinstance(claimed_families, list) or set(claimed_families) != rederived_families:
        failures.append(
            "Artifact's healthy_model_families does not match the families "
            "re-derived from its paired rows."
        )

    rows = report.get("models", [])
    rows = rows if isinstance(rows, list) else []
    for entry in paired:
        if not isinstance(entry, dict):
            continue
        pair = (entry.get("provider"), entry.get("model"))
        direct = next(
            (
                row
                for row in rows
                if isinstance(row, dict)
                and (row.get("provider"), row.get("model")) == pair
                and row.get("method") == "single_pass_model"
            ),
            None,
        )
        staged = next(
            (
                row
                for row in rows
                if isinstance(row, dict)
                and (row.get("provider"), row.get("model")) == pair
                and row.get("method") == "staged_redteam"
            ),
            None,
        )
        if direct is None or staged is None:
            failures.append(
                f"{pair[0]}:{pair[1]} is missing a paired single-pass or "
                "staged row in the artifact."
            )
            continue
        for method, row in (("single-pass", direct), ("staged", staged)):
            repetitions = row.get("repetitions", 0)
            if (
                not isinstance(repetitions, int)
                or isinstance(repetitions, bool)
                or repetitions < _TRANSITION_MINIMUM_REPETITIONS
            ):
                failures.append(
                    f"{pair[0]}:{pair[1]} {method} shows fewer than "
                    f"{_TRANSITION_MINIMUM_REPETITIONS} repetitions."
                )
            if row.get("errors", 0) != 0:
                failures.append(
                    f"{pair[0]}:{pair[1]} {method} shows provider errors; "
                    "failed trials cannot be excluded from the quality gate."
                )
        if staged.get("evidence_gate_failures", 0) != 0:
            failures.append(
                f"{pair[0]}:{pair[1]} staged shows evidence-gate failures."
            )

    def _check_bars(scores: Any, label: str) -> None:
        if not isinstance(scores, dict):
            failures.append(f"{label} scores are missing from the artifact.")
            return
        recall = scores.get("recall")
        specificity = scores.get("specificity")
        citation = scores.get("citation_accuracy")
        if not isinstance(recall, (int, float)) or recall < _TRANSITION_BAR_RECALL:
            failures.append(f"{label} recall is below 85%.")
        if (
            not isinstance(specificity, (int, float))
            or specificity < _TRANSITION_BAR_SPECIFICITY
        ):
            failures.append(f"{label} specificity is below 90%.")
        if (
            not isinstance(citation, (int, float))
            or citation < _TRANSITION_BAR_CITATION
        ):
            failures.append(f"{label} exact citation accuracy is below 95%.")

    direct_scores = gate.get("single_pass") if isinstance(gate, dict) else None
    staged_scores = gate.get("staged") if isinstance(gate, dict) else None
    _check_bars(direct_scores, "single-pass pooled")
    _check_bars(staged_scores, "staged pooled")
    if (
        isinstance(direct_scores, dict)
        and isinstance(staged_scores, dict)
        and isinstance(staged_scores.get("recall"), (int, float))
        and isinstance(direct_scores.get("recall"), (int, float))
        and staged_scores["recall"] < direct_scores["recall"]
    ):
        failures.append(
            "Staged mode recall regresses against the single-pass baseline."
        )
    per_family = gate.get("per_family") if isinstance(gate, dict) else None
    if not isinstance(per_family, dict) or not per_family:
        failures.append("Artifact is missing per-family scores.")
    else:
        for family in sorted(per_family):
            scores = per_family[family]
            if not isinstance(scores, dict):
                failures.append(f"{family} family scores are malformed.")
                continue
            _check_bars(scores.get("single_pass"), f"{family} single-pass")
            _check_bars(scores.get("staged"), f"{family} staged")
            family_direct = scores.get("single_pass")
            family_staged = scores.get("staged")
            if (
                isinstance(family_direct, dict)
                and isinstance(family_staged, dict)
                and isinstance(family_staged.get("recall"), (int, float))
                and isinstance(family_direct.get("recall"), (int, float))
                and family_staged["recall"] < family_direct["recall"]
            ):
                failures.append(
                    f"{family} staged mode recall regresses against the "
                    "single-pass baseline."
                )

    limitations = gate.get("limitations") if isinstance(gate, dict) else None
    if not isinstance(limitations, list) or not limitations:
        failures.append(
            "Artifact omits the bounded-gate limitations; the qualified claim "
            "must travel with the artifact."
        )
    council = report.get("staged_council")
    if not isinstance(council, dict):
        failures.append("Artifact is missing its staged_council section.")
    else:
        if council.get("method") != "staged_redteam_council":
            failures.append("staged_council section is malformed.")
        if not isinstance(council.get("member_calls"), int) or council["member_calls"] <= 0:
            failures.append("staged_council shows no member calls.")
        if council.get("errors", 0) != 0:
            failures.append("staged_council shows invalid case trials.")

    summary = {
        "positive_cases": positive_cases,
        "benign_cases": benign_cases,
        "families": sorted(rederived_families),
        "paired_models": len(paired),
    }
    return {"valid": not failures, "failures": failures, "summary": summary}


def recommend_model(report: Dict[str, Any], minimum_cases: int = 10) -> Optional[Dict[str, Any]]:
    """Recommend only among models evaluated on the minimum number of cases."""
    eligible = [
        model for model in report.get("models", [])
        if (
            model.get("method", "single_pass_model") == "single_pass_model"
            and model.get("cases", 0) >= minimum_cases
            and model.get("errors", 0) == 0
            and model.get("true_positives", 0) > 0
        )
    ]
    if not eligible:
        return None
    all_cost_known = all(model.get("estimated_cost_usd") is not None for model in eligible)
    if all_cost_known:
        eligible.sort(
            key=lambda model: (
                -model["f1"],
                model["estimated_cost_usd"],
                model["p50_latency_ms"] if model["p50_latency_ms"] is not None else float("inf"),
            )
        )
    else:
        eligible.sort(
            key=lambda model: (
                -model["f1"],
                model["p50_latency_ms"] if model["p50_latency_ms"] is not None else float("inf"),
            )
        )
    selected = eligible[0]
    return {
        "provider": selected["provider"],
        "model": selected["model"],
        "f1": selected["f1"],
        "tie_breaker": "estimated_cost_then_latency" if all_cost_known else "latency",
        "minimum_cases": minimum_cases,
    }
