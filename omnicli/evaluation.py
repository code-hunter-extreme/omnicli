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
    return cases


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
    source_lines = case["source"].splitlines()
    start = finding["line"] - 1
    evidence_lines = finding["evidence"].splitlines()
    if start < 0 or start >= len(source_lines):
        return False
    cited_excerpt = "\n".join(source_lines[start:start + max(1, len(evidence_lines))])
    return _normalize(finding["evidence"]) in _normalize(cited_excerpt)


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
    return findings


def _evaluate_case(case: Dict[str, Any], predicted: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected = case["expected_findings"]
    unmatched = set(range(len(expected)))
    true_positives = 0
    false_positives = 0
    false_positive_categories: List[str] = []
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
                    false_positive_clean_cases += 1
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
            and not route.get("same_module_request_hooks")
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
        "case_results": case_results,
    }


def evaluate_staged_security(
    cfg: Any,
    client: Any,
    cases: List[Dict[str, Any]],
    models: List[Tuple[str, str]],
    repetitions: int = 1,
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Evaluate the staged red-team pipeline on the same labeled cases."""
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

                    predictions = []
                    seen_predictions = set()
                    case_citations = case_grounded = 0
                    for validation in review.get("attack_validations", []):
                        if not isinstance(validation, dict):
                            continue
                        for finding in validation.get("findings_checked", []):
                            if not isinstance(finding, dict):
                                continue
                            case_citations += 1
                            case_grounded += bool(finding.get("grounded"))
                            prediction = {
                                "category": finding.get("category", "unknown"),
                                "file": finding.get("file", ""),
                                "line": finding.get("start_line", 0),
                                "evidence": finding.get("quote", ""),
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
                    citations += case_citations
                    grounded_citations += case_grounded
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
                    trial_failed = gate_failed or bool(member_errors)
                    if gate_failed:
                        gate_failures += 1
                    if trial_failed:
                        errors += 1
                        result["error"] = (
                            "; ".join(member_errors)[:240]
                            if member_errors
                            else "Staged review evidence gate failed."
                        )
                        result["true_positives"] = 0
                        result["false_negatives"] = len(case["expected_findings"])
                        result["matched"] = []
                        result["missed"] = case["expected_findings"]
                    if not case["expected_findings"]:
                        if predictions or result.get("error"):
                            clean_false_positives += 1
                        else:
                            clean_true_negatives += 1
                except Exception as exc:
                    errors += 1
                    gate_failures += 1
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
                    }
                    if not case["expected_findings"]:
                        clean_false_positives += 1
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
    paired = sorted(set(direct_by_model).intersection(staged_by_model))
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

    direct_rows = [direct_by_model[pair] for pair in healthy_pairs]
    staged_rows = [staged_by_model[pair] for pair in healthy_pairs]
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
        "single_pass": direct_scores,
        "staged": staged_scores,
        "limitations": [
            "Passing qualifies only for this reviewed held-out set, roster, and measured configuration.",
            "Wilson intervals are descriptive and do not account for correlations between repeated trials.",
            "A pass is not proof of universal coverage, exploitability, or bounty eligibility.",
        ],
    }


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
