"""Deterministic grounding checks for model-quoted source evidence."""

from __future__ import annotations

import ast
import re
from typing import Dict, Mapping, Optional


_FINDING_HEADER = re.compile(
    r"(?is)(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*\*)?FINDING:\s*(?P<title>.*?)"
    r"\s+FILE:\s*(?P<file>.*?)"
    r"\s+LINE:\s*(?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?"
    r"\s+QUOTE:\s*"
)
_CODE_QUOTE = re.compile(r"```(?:[^\n`]*)\n?(.*?)```|`([^`\n]+)`", re.DOTALL)
_CLAIM = re.compile(
    r"\b(finding|findings|vulnerabilit(?:y|ies)|security issue|exploitable|"
    r"defect|bug|issue)\b",
    re.IGNORECASE,
)
_SOURCE_BLOCK = re.compile(
    r"(?ims)^###\s*Source file:\s*(?P<path>[^\n]+)\n"
    r"```[^\n]*\n(?P<source>.*?)\n```"
)


def extract_sources_from_prompt(prompt: str) -> Dict[str, str]:
    """Extract only explicitly labeled source-file blocks from prompt text."""
    return {
        match.group("path").strip().replace("\\", "/"): match.group("source")
        for match in _SOURCE_BLOCK.finditer(prompt)
    }


def analyze_python_security_facts(
    sources: Mapping[str, str],
) -> list[Dict[str, object]]:
    """Collect conservative Python route/input/sink facts without claiming safety."""
    auth_markers = (
        "login_required", "auth_required", "requires_auth", "jwt_required",
        "permission_required", "current_user", "authorize", "check_auth",
        "verify_token", "require_role",
    )
    mutation_markers = (
        "delete", "remove", "update", "transfer", "insert", "execute",
        "write", "send", "grant", "revoke", "create",
    )
    input_markers = {
        "args", "form", "json", "values", "get_json", "query_params",
        "headers", "cookies", "data", "body",
    }
    sensitive_markers = {
        "password", "secret", "token", "credential", "ssn", "private_key",
        "api_key", "email", "phone",
    }

    def dotted_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    def literal_only(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return all(literal_only(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return all(
                (key is None or literal_only(key)) and literal_only(value)
                for key, value in zip(node.keys, node.values)
            )
        return False

    reports: list[Dict[str, object]] = []
    for path, source in sources.items():
        if not path.lower().endswith(".py"):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            reports.append({
                "file": path,
                "parse_error": f"Python syntax error at line {exc.lineno}",
                "routes": [],
            })
            continue

        routes: list[Dict[str, object]] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                route_kind = dotted_name(decorator.func).split(".")[-1].lower()
                if route_kind not in {"route", "get", "post", "put", "patch", "delete"}:
                    continue
                route_path = next(
                    (
                        arg.value for arg in decorator.args
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    ),
                    None,
                )
                if route_path is None:
                    continue
                methods = [route_kind.upper()] if route_kind != "route" else []
                for keyword in decorator.keywords:
                    if keyword.arg == "methods" and isinstance(keyword.value, (ast.List, ast.Tuple, ast.Set)):
                        methods = [
                            item.value.upper()
                            for item in keyword.value.elts
                            if isinstance(item, ast.Constant) and isinstance(item.value, str)
                        ]

                calls = [
                    dotted_name(item.func)
                    for item in ast.walk(node)
                    if isinstance(item, ast.Call)
                ]
                decorators = [dotted_name(item) for item in node.decorator_list]
                names = [dotted_name(item) for item in ast.walk(node)]
                request_inputs = sorted({
                    name for name in names
                    if name.split(".")[0] in {"request", "req"}
                    and any(part.lower() in input_markers for part in name.split(".")[1:])
                })
                local_auth = sorted({
                    name for name in [*calls, *decorators]
                    if any(marker in name.lower() for marker in auth_markers)
                })
                mutating_calls = sorted({
                    name for name in calls
                    if any(marker in name.rsplit(".", 1)[-1].lower() for marker in mutation_markers)
                })
                output_keys = sorted({
                    item.value.lower()
                    for item in ast.walk(node)
                    if isinstance(item, ast.Constant)
                    and isinstance(item.value, str)
                    and any(marker in item.value.lower() for marker in sensitive_markers)
                })
                returns = [item for item in ast.walk(node) if isinstance(item, ast.Return)]
                literal_returns = bool(returns) and all(
                    item.value is not None and literal_only(item.value)
                    for item in returns
                )
                routes.append({
                    "path": route_path,
                    "methods": methods,
                    "handler": node.name,
                    "handler_line": node.lineno,
                    "start_line": min(
                        [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
                    ),
                    "end_line": max(
                        [node.end_lineno or node.lineno]
                        + [
                            decorator.end_lineno or decorator.lineno
                            for decorator in node.decorator_list
                        ]
                    ),
                    "local_auth_guards": local_auth,
                    "request_input_accesses": request_inputs,
                    "mutating_calls": mutating_calls,
                    "sensitive_output_markers": output_keys,
                    "returns_only_literals": literal_returns,
                })
        reports.append({"file": path, "routes": routes})
    return reports


def triage_grounded_findings(
    validations: list[Mapping[str, object]],
    source_facts: list[Mapping[str, object]],
) -> list[Dict[str, object]]:
    """Deduplicate citations and screen claims against conservative AST route facts."""
    routes_by_file: Dict[str, list[Mapping[str, object]]] = {}
    for report in source_facts:
        path = report.get("file")
        routes = report.get("routes")
        if isinstance(path, str) and isinstance(routes, list):
            routes_by_file[path.replace("\\", "/")] = [
                route for route in routes if isinstance(route, Mapping)
            ]

    findings: Dict[tuple[object, ...], Dict[str, object]] = {}
    auth_claim = re.compile(
        r"\b(unauth|unauthor|authentic|authorization|access control|privilege)\w*\b",
        re.I,
    )
    input_claim = re.compile(r"\b(input|injection|validation|unvalidated|saniti[sz])\b", re.I)
    for validation in validations:
        items = validation.get("findings_checked", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("finding", "Unspecified candidate"))
            path = str(item.get("file", "")).replace("\\", "/")
            start = item.get("start_line")
            end = item.get("end_line")
            key = (title.strip().lower(), path, start, end)
            grounded = bool(item.get("grounded"))
            candidate: Dict[str, object] = {
                "title": title,
                "file": path,
                "start_line": start,
                "end_line": end,
                "quote_grounded": grounded,
                "status": "unverified_candidate",
                "reason": "Static source patterns do not establish the complete security behavior.",
            }
            if not grounded:
                candidate.update({
                    "status": "unsupported_citation",
                    "reason": str(item.get("reason", "Citation did not match supplied source.")),
                })
            else:
                route_candidates = routes_by_file.get(path, [])
                if isinstance(start, int) and isinstance(end, int):
                    matched = [
                        route for route in route_candidates
                        if isinstance(route.get("start_line"), int)
                        and isinstance(route.get("end_line"), int)
                        and route["start_line"] <= end
                        and route["end_line"] >= start
                    ]
                    if len(matched) == 1:
                        route = matched[0]
                        candidate["route"] = route.get("path")
                        candidate["http_methods"] = route.get("methods", [])
                        if auth_claim.search(title):
                            mutating = bool(route.get("mutating_calls"))
                            sensitive_output = bool(route.get("sensitive_output_markers"))
                            if not mutating and not sensitive_output and route.get("returns_only_literals"):
                                candidate.update({
                                    "status": "not_actionable_for_handler",
                                    "reason": (
                                        "Cited route has no detected mutating call or sensitive-output "
                                        "marker and returns only literals; a missing local auth guard "
                                        "alone does not establish a vulnerability."
                                    ),
                                })
                            elif not route.get("local_auth_guards"):
                                candidate.update({
                                    "status": "unverified_authorization_gap",
                                    "reason": (
                                        "No local auth guard pattern was detected before a sensitive "
                                        "operation; middleware, decorators outside this handler, and "
                                        "effective route reachability still require verification."
                                    ),
                                })
                        elif input_claim.search(title):
                            if not route.get("request_input_accesses") or not route.get("mutating_calls"):
                                candidate.update({
                                    "status": "not_actionable_for_handler",
                                    "reason": (
                                        "The cited handler has no detected request-data access followed "
                                        "by a mutating operation; an input/injection claim lacks a "
                                        "source-supported sink."
                                    ),
                                })
            findings.setdefault(key, candidate)
    return list(findings.values())


def assess_evidence_gate(
    sources: Mapping[str, str],
    validations: list[Mapping[str, object]],
    provider_errors: list[str],
    candidate_triage: list[Mapping[str, object]],
) -> Dict[str, object]:
    """Fail closed when red-team claims cannot be grounded to supplied evidence."""
    failures: list[str] = []
    if not sources:
        failures.append("No source files were supplied; claims cannot be grounded.")
    elif not any(isinstance(text, str) and text.strip() for text in sources.values()):
        failures.append("Supplied source files are empty; claims cannot be grounded.")
    if provider_errors:
        failures.extend(f"Review stage failed: {error}" for error in provider_errors)

    bad_validation_statuses = {
        "no_source_provided",
        "unsupported_citations",
        "unstructured_claims",
    }
    for index, validation in enumerate(validations, start=1):
        status = validation.get("status")
        if status in bad_validation_statuses:
            failures.append(f"Review output {index} failed grounding: {status}.")

    for candidate in candidate_triage:
        if candidate.get("status") == "unsupported_citation":
            failures.append(
                "Candidate citation is not grounded: "
                f"{candidate.get('title', 'unnamed candidate')} "
                f"({candidate.get('file', 'unknown file')}:{candidate.get('start_line', '?')})."
            )

    return {
        "status": "FAIL" if failures else "PASS_GROUNDING_ONLY",
        "failed_checks": failures,
        "source_files_checked": len(sources),
        "review_outputs_checked": len(validations),
        "candidate_count": len(candidate_triage),
        "security_conclusion": "NOT_ASSESSED",
        "meaning": (
            "PASS_GROUNDING_ONLY means the supplied review outputs did not fail "
            "the source-grounding checks. It does not prove exploitability, "
            "completeness, or safety."
        ),
    }


def validate_quoted_findings(
    answer: str,
    sources: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    """Check structured file/line/quote claims against supplied source text.

    A grounded quote establishes only that the quoted text exists at the cited
    location. It does not establish control flow, exploitability, or impact.
    """
    source_map = {
        str(path).strip().replace("\\", "/"): text
        for path, text in (sources or {}).items()
        if isinstance(path, str) and isinstance(text, str)
    }
    answer_text = answer or ""
    findings = list(_FINDING_HEADER.finditer(answer_text))
    checked = []
    covered_claim_spans = []
    for index, match in enumerate(findings):
        path = match.group("file").strip().replace("\\", "/")
        source = source_map.get(path)
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        quote_end = findings[index + 1].start() if index + 1 < len(findings) else len(answer_text)
        quote_tail = answer_text[match.end():quote_end].lstrip()
        code_quote = _CODE_QUOTE.search(quote_tail)
        if code_quote:
            quote = (
                code_quote.group(1)
                if code_quote.group(1) is not None
                else code_quote.group(2)
            )
            covered_claim_spans.append((match.start(), match.end() + code_quote.end()))
        else:
            quote = next(
                (line.strip() for line in quote_tail.splitlines() if line.strip()),
                "",
            )
            covered_claim_spans.append((match.start(), match.end() + len(quote)))
        quote = " ".join(quote.split())
        reason = None
        if source is None:
            reason = "cited file was not among the supplied source files"
        elif end < start or start < 1 or end > len(source.splitlines()):
            reason = "cited line range is outside the supplied file"
        elif not quote:
            reason = "citation did not include a source quote"
        else:
            source_lines = source.splitlines()[start - 1:end]
            cited_text = " ".join(
                " ".join(" ".join(line.split()) for line in source_lines).split()
            )
            if quote not in cited_text:
                reason = "quoted lines do not exactly match the cited source range"
        checked.append({
            "finding": match.group("title").strip(),
            "file": path,
            "start_line": start,
            "end_line": end,
            "quote": quote,
            "grounded": reason is None,
            **({"reason": reason} if reason else {}),
        })

    unstructured_parts = []
    previous = 0
    for start, end in covered_claim_spans:
        unstructured_parts.append(answer_text[previous:start])
        previous = end
    unstructured_parts.append(answer_text[previous:])
    unstructured_text = "\n".join(unstructured_parts)
    unstructured_claim = bool(_CLAIM.search(unstructured_text))
    if not source_map:
        status = "no_source_provided"
    elif findings and (
        unstructured_claim or not all(item["grounded"] for item in checked)
    ):
        status = "unsupported_citations"
    elif unstructured_claim:
        status = "unstructured_claims"
    elif findings:
        status = "quotes_grounded"
    else:
        status = "no_structured_findings"
    return {
        "status": status,
        "source_files_checked": len(source_map),
        "findings_checked": checked,
        "candidate_claims_unstructured": unstructured_claim,
        "security_conclusion": "not_assessed",
    }
