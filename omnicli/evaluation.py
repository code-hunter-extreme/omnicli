"""Repeatable, labeled evaluation for security-audit model selection."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_REQUIRED_FINDING_FIELDS = {"category", "file", "line", "evidence"}


def load_security_dataset(path: str) -> List[Dict[str, Any]]:
    """Load versioned JSONL cases and reject incomplete or ambiguous labels."""
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise ValueError(f"Evaluation dataset does not exist: {dataset_path}")

    cases: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
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
            if _normalize(finding["evidence"]) not in _normalize(source):
                raise ValueError(
                    f"Expected evidence in case {case_id} is not present in its source."
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
    return cases


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _finding_key(finding: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        finding["category"].strip().lower(),
        finding["file"].replace("\\", "/").strip(),
        finding["line"],
    )


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
            or not isinstance(finding["category"], str)
            or not isinstance(finding["file"], str)
            or not isinstance(finding["line"], int)
            or isinstance(finding["line"], bool)
            or not isinstance(finding["evidence"], str)
        ):
            raise ValueError("A model finding did not match the required JSON schema.")
    return findings


def _evaluate_case(case: Dict[str, Any], predicted: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected = case["expected_findings"]
    unmatched = set(range(len(expected)))
    true_positives = 0
    false_positives = 0
    matched: List[Dict[str, Any]] = []
    for candidate in predicted:
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
            false_positives += 1
            continue
        unmatched.remove(match_index)
        true_positives += 1
        matched.append({"category": candidate["category"], "file": candidate["file"], "line": candidate["line"]})
    return {
        "case_id": case["id"],
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": len(unmatched),
        "matched": matched,
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


def evaluate_security_models(
    client: Any,
    cases: List[Dict[str, Any]],
    models: List[Tuple[str, str]],
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Run models against labeled cases and compute exact-quote precision/recall."""
    if not models:
        raise ValueError("At least one provider:model pair is required.")
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
        latencies: List[float] = []
        estimated_cost = 0.0
        pricing_key = f"{provider}:{model}"
        model_pricing = pricing.get(pricing_key, pricing.get(model))
        cost_known = model_pricing is not None
        for case in cases:
            prompt = (
                "Perform a conservative security review of the source below. "
                "Report only findings supported by the supplied source. Return "
                "JSON only with this shape: "
                '{"findings":[{"category":"short_snake_case","file":"path from source context",'
                '"line":1,"evidence":"exact source quote"}]}. '
                "Use an empty findings array when no finding is supported. Do not "
                "include explanations outside JSON.\n\n"
                f"### Case id\n{case['id']}\n\n"
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
                predictions = _parse_findings(content)
                result = _evaluate_case(case, predictions)
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
                result = {
                    "case_id": case["id"],
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": len(case["expected_findings"]),
                    "error": str(exc)[:240],
                }
            tp += result["true_positives"]
            fp += result["false_positives"]
            fn += result["false_negatives"]
            case_results.append(result)
        ordered_latencies = sorted(latencies)
        p50_index = (len(ordered_latencies) - 1) // 2
        model_results.append({
            "provider": provider,
            "model": model,
            "cases": len(cases),
            "errors": errors,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            **_score(tp, fp, fn),
            "p50_latency_ms": ordered_latencies[p50_index] if ordered_latencies else None,
            "estimated_cost_usd": round(estimated_cost, 8) if cost_known else None,
            "case_results": case_results,
        })
    return {
        "schema_version": 1,
        "metric": "exact_category_file_line_and_quoted_gold_evidence",
        "dataset_cases": len(cases),
        "models": model_results,
        "limitations": [
            "This benchmark measures only the labels and cases supplied by the dataset author.",
            "A match is not proof of exploitability; labels must be independently reviewed.",
            "A missing result or malformed model response counts as missed gold findings and is reported as an error.",
            "Estimated cost is shown only when pricing and token usage are available for every evaluated case.",
            "Do not route production audits from a small or unrepresentative dataset.",
        ],
    }


def recommend_model(report: Dict[str, Any], minimum_cases: int = 10) -> Optional[Dict[str, Any]]:
    """Recommend only among models evaluated on the minimum number of cases."""
    eligible = [
        model for model in report.get("models", [])
        if (
            model.get("cases", 0) >= minimum_cases
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
