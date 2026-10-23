"""Deterministic grounding checks for model-quoted source evidence."""

from __future__ import annotations

import ast
import hashlib
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


_FINDING_HEADER = re.compile(
    r"(?is)(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*\*)?FINDING:\s*(?P<title>.*?)"
    r"(?:\s+CATEGORY:\s*(?P<category>[a-z][a-z0-9_]*))?"
    r"\s+FILE:\s*(?P<file>.*?)"
    r"\s+LINE:\s*(?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?"
    r"\s+QUOTE:\s*"
)
_QUOTE_DETAIL = re.compile(
    r"(?im)^\s*(?:ATTACK PATH|PRECONDITIONS|UNVERIFIED):"
)
_CLAIM = re.compile(
    r"\b(finding|findings|vulnerabilit(?:y|ies)|security issue|exploitable|"
    r"defect|bug|issue)\b",
    re.IGNORECASE,
)
_SOURCE_BLOCK = re.compile(
    r"(?ims)^###\s*Source file:\s*(?P<path>[^\n]+)\n"
    r"```[^\n]*\n(?P<source>.*?)\n```"
)


_CANONICAL_CATEGORIES = {
    "sql_injection",
    "command_injection",
    "path_traversal",
    "unsafe_deserialization",
    "authentication_authorization",
    "sensitive_data_exposure",
}

_CATEGORY_ALIASES = {
    "missing_authorization": "authentication_authorization",
    "broken_access_control": "authentication_authorization",
    "authorization": "authentication_authorization",
    "authentication": "authentication_authorization",
    "authz": "authentication_authorization",
    "authn": "authentication_authorization",
    "access_control": "authentication_authorization",
    "idor": "authentication_authorization",
    "insecure_direct_object_reference": "authentication_authorization",
    "privilege_escalation": "authentication_authorization",
    "unauthorized_access": "authentication_authorization",
    "broken_authentication": "authentication_authorization",
    "sqli": "sql_injection",
    "sql": "sql_injection",
    "os_command_injection": "command_injection",
    "cmd_injection": "command_injection",
    "shell_injection": "command_injection",
    "directory_traversal": "path_traversal",
    "traversal": "path_traversal",
    "lfi": "path_traversal",
    "information_disclosure": "sensitive_data_exposure",
    "info_disclosure": "sensitive_data_exposure",
    "data_exposure": "sensitive_data_exposure",
    "secret_exposure": "sensitive_data_exposure",
    "hardcoded_secret": "sensitive_data_exposure",
    "insecure_deserialization": "unsafe_deserialization",
    "deserialization": "unsafe_deserialization",
}


def canonical_finding_category(category: str) -> str:
    """Normalize finding categories to the pinned 6-class benchmark taxonomy."""
    cleaned = (category or "").strip().lower()
    return _CATEGORY_ALIASES.get(cleaned, cleaned)


def extract_sources_from_prompt(prompt: str) -> Dict[str, str]:
    """Extract only explicitly labeled source-file blocks from prompt text."""
    return {
        match.group("path").strip().replace("\\", "/"): match.group("source")
        for match in _SOURCE_BLOCK.finditer(prompt)
    }


def strip_source_blocks_from_prompt(prompt: str) -> str:
    """Remove labeled source blocks while preserving the surrounding task text.

    Red-team stages render their own line-numbered source map from the parsed
    source files. Keeping the original blocks in the goal as well would send
    every file to the model twice.
    """
    return _SOURCE_BLOCK.sub("", prompt or "").strip()


def candidate_reference_id(
    path: str,
    start_line: object,
    end_line: object,
    category: str = "unknown",
    quote: str = "",
) -> str:
    """Identify one cited evidence unit independently of model title or category."""
    identity = "\0".join((
        path.strip().replace("\\", "/"),
        str(start_line),
        str(end_line),
        " ".join(quote.split()),
    ))
    return "CAND-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12].upper()


def parse_challenge_triage(answer: str) -> list[Dict[str, str]]:
    """Extract explicitly structured challenge dispositions without treating them as proof."""
    parsed = []
    text = (answer or "").replace("**", "").replace("__", "")
    candidate_pattern = re.compile(r"CANDIDATE:\s*(CAND-[A-F0-9]{12})", re.I)
    status_values = r"SUPPORTED|REFUTED|UNRESOLVED|UNVERIFIED"
    status_pattern = re.compile(
        rf"(?im)^\s*TRIAGE\s*:?\s*({status_values})\s*$"
    )
    direct_status_pattern = re.compile(
        rf"(?im)^\s*(CAND-[A-F0-9]{{12}})\s*:\s*({status_values})\b",
        re.I,
    )

    def normalize_status(status: str) -> str:
        normalized = status.lower()
        return "unresolved" if normalized == "unverified" else normalized

    statuses = list(status_pattern.finditer(text))
    for index, status_match in enumerate(statuses):
        next_status = statuses[index + 1].start() if index + 1 < len(statuses) else len(text)
        following = text[status_match.end():next_status]
        candidate = candidate_pattern.search(following)
        if not candidate:
            previous_status_end = statuses[index - 1].end() if index else 0
            preceding = text[previous_status_end:status_match.start()]
            previous_ids = list(re.finditer(r"\b(CAND-[A-F0-9]{12})\b", preceding, re.I))
            if not previous_ids:
                continue
            candidate_id = previous_ids[-1].group(1).upper()
            candidate_start = previous_status_end + previous_ids[-1].start()
        else:
            candidate_id = candidate.group(1).upper()
            candidate_start = status_match.end() + candidate.start()
        detail = text[candidate_start:next_status]
        reason = re.search(r"(?im)^\s*REASON:\s*(.+?)\s*$", detail)
        safe_test = re.search(r"(?im)^\s*SAFE TEST:\s*(.+?)\s*$", detail)
        parsed.append({
            "candidate_id": candidate_id,
            "status": normalize_status(status_match.group(1)),
            "reason": " ".join(reason.group(1).split()) if reason else "",
            "safe_test": " ".join(safe_test.group(1).split()) if safe_test else "",
        })
    direct_statuses = list(direct_status_pattern.finditer(text))
    for index, match in enumerate(direct_statuses):
        end = direct_statuses[index + 1].start() if index + 1 < len(direct_statuses) else len(text)
        block = text[match.start():end]
        reason = re.search(r"(?im)^\s*REASON:\s*(.+?)\s*$", block)
        safe_test = re.search(r"(?im)^\s*SAFE TEST:\s*(.+?)\s*$", block)
        parsed.append({
            "candidate_id": match.group(1).upper(),
            "status": normalize_status(match.group(2)),
            "reason": " ".join(reason.group(1).split()) if reason else "",
            "safe_test": " ".join(safe_test.group(1).split()) if safe_test else "",
        })
    unique = {item["candidate_id"]: item for item in parsed}
    return sorted(unique.values(), key=lambda item: text.find(item["candidate_id"]))


# Guard functions that reject/redirect unauthenticated callers when *called*
# (as a statement in a before_request body, not merely referenced).
_AUTH_GUARD_CALLS = frozenset({
    "login_required", "jwt_required", "require_auth", "requires_auth",
    "authenticate", "verify_jwt", "verify_jwt_in_request", "check_auth",
    "verify_token", "require_role",
})
# Exception types whose raise denies the request.
_AUTH_RAISE_NAMES = frozenset({"unauthorized", "forbidden"})
# Identity/login-state references. On their own these prove nothing (logging
# current_user is not enforcement), but a redirect *guarded by* one of these is
# the standard Flask-Login pattern of bouncing anonymous users to a login page.
_AUTH_IDENTITY_NAMES = frozenset({
    "current_user", "is_authenticated", "is_anonymous", "logged_in", "authenticated",
})


def _call_name(call: ast.Call) -> Optional[str]:
    """Bare name of a call target: ``abort`` for ``flask.abort(...)``."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _has_rejecting_status(node: ast.AST) -> bool:
    """True if a 401/403 literal appears anywhere in this subtree."""
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Constant)
            and isinstance(sub.value, int)
            and not isinstance(sub.value, bool)
            and sub.value in (401, 403)
        ):
            return True
    return False


def _hook_enforces_auth(fn: ast.AST) -> bool:
    """Best-effort: does this request-hook body actually *reject* callers?

    Requires a concrete rejecting action — ``abort(401/403)``, a raised
    Unauthorized/Forbidden (or any raise carrying a 401/403), a ``return`` that
    carries a 401/403 status, a call to a known auth guard, or a ``redirect(...)``
    guarded by a login-state reference (the Flask-Login bounce-to-login pattern).
    A bare reference to ``current_user`` or a 401 literal in a log line does NOT
    count: logging is not enforcement. When enforcement is unproven this returns
    False, and callers must not then claim auth is *absent* on this record alone.
    """
    has_redirect = False
    has_identity_ref = False
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            cn = _call_name(n)
            if cn == "abort" and any(
                isinstance(a, ast.Constant) and a.value in (401, 403) for a in n.args
            ):
                return True
            if cn and cn.lower() in _AUTH_GUARD_CALLS:
                return True
            if cn == "redirect":
                has_redirect = True
        elif isinstance(n, ast.Raise) and n.exc is not None:
            exc = n.exc
            if isinstance(exc, ast.Call):
                if _has_rejecting_status(exc):
                    return True
                rn = _call_name(exc)
            elif isinstance(exc, ast.Name):
                rn = exc.id
            elif isinstance(exc, ast.Attribute):
                rn = exc.attr
            else:
                rn = None
            if rn and rn.lower() in _AUTH_RAISE_NAMES:
                return True
        elif isinstance(n, ast.Return) and n.value is not None:
            if _has_rejecting_status(n.value):
                return True
        ref = None
        if isinstance(n, ast.Name):
            ref = n.id
        elif isinstance(n, ast.Attribute):
            ref = n.attr
        if ref and ref.lower() in _AUTH_IDENTITY_NAMES:
            has_identity_ref = True
    # A redirect gated on login state (e.g. `if not current_user.is_authenticated:
    # return redirect(url_for("login"))`) rejects anonymous callers. A redirect
    # without any identity check, or an identity reference with no rejecting
    # action, does not qualify.
    return has_redirect and has_identity_ref


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
        "args", "form", "json", "values", "get_json", "get_data", "query_params",
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

    def request_input_expression(node: ast.AST) -> bool:
        def access_name(value: ast.AST) -> str:
            while isinstance(value, ast.Subscript):
                value = value.value
            return dotted_name(value)

        return any(
            isinstance(item, (ast.Attribute, ast.Subscript))
            and access_name(item).split(".")[0] in {"request", "req"}
            and any(
                part.lower() in input_markers
                for part in access_name(item).split(".")[1:]
            )
            for item in ast.walk(node)
        )

    def security_sink(call: ast.Call) -> Optional[str]:
        name = dotted_name(call.func).lower()
        tail = name.rsplit(".", 1)[-1]
        if tail in {"execute", "executemany"}:
            return "database_execution"
        if name.startswith("subprocess.") or name in {"os.system", "os.popen"}:
            return "command_execution"
        if name in {"pickle.loads", "yaml.load", "marshal.loads"}:
            return "unsafe_deserialization"
        if tail == "open" or name == "open":
            return "filesystem_access"
        return None

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
        request_hooks = []
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = [dotted_name(item) for item in function.decorator_list]
            hook_decorators = [
                decorator for decorator in decorators
                if decorator.rsplit(".", 1)[-1].lower()
                in {"before_request", "before_app_request"}
            ]
            if hook_decorators:
                request_hooks.append({
                    "handler": function.name,
                    "line": function.lineno,
                    "decorators": hook_decorators,
                    "enforces_auth": _hook_enforces_auth(function),
                })
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
                assignments = [
                    item for item in ast.walk(node)
                    if isinstance(item, (ast.Assign, ast.AnnAssign))
                ]
                dependencies: Dict[str, set[str]] = {}
                request_derived = set()
                dynamic_variables = set()
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
                        item.id for item in ast.walk(value)
                        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                    }
                    for target_name in target_names:
                        dependencies.setdefault(target_name, set()).update(value_names)
                        if request_input_expression(value):
                            request_derived.add(target_name)
                        if any(
                            isinstance(item, (ast.JoinedStr, ast.BinOp))
                            for item in ast.walk(value)
                        ):
                            dynamic_variables.add(target_name)
                for _ in range(len(dependencies)):
                    for target_name, value_names in dependencies.items():
                        if value_names.intersection(request_derived):
                            request_derived.add(target_name)
                        if value_names.intersection(dynamic_variables):
                            dynamic_variables.add(target_name)
                call_sites = []
                for item in ast.walk(node):
                    if not isinstance(item, ast.Call):
                        continue
                    argument_names = sorted({
                        child.id for argument in [
                            *item.args,
                            *(keyword.value for keyword in item.keywords),
                        ]
                        for child in ast.walk(argument)
                        if isinstance(child, ast.Name)
                        and isinstance(child.ctx, ast.Load)
                    })
                    call_sites.append({
                        "call": dotted_name(item.func),
                        "line": item.lineno,
                        "argument_names": argument_names,
                        "request_derived_arguments": sorted(
                            set(argument_names).intersection(request_derived)
                        ),
                    })
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
                sinks = []
                for item in ast.walk(node):
                    if not isinstance(item, ast.Call):
                        continue
                    category = security_sink(item)
                    if category:
                        argument_names = sorted({
                            child.id for argument in [*item.args, *(
                                keyword.value for keyword in item.keywords
                            )]
                            for child in ast.walk(argument)
                            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                        })
                        sinks.append({
                            "category": category,
                            "call": dotted_name(item.func),
                            "line": item.lineno,
                            "argument_names": argument_names,
                            "dynamic_argument": any(
                                isinstance(child, (ast.JoinedStr, ast.BinOp))
                                for argument in item.args
                                for child in ast.walk(argument)
                            ) or bool(set(argument_names).intersection(dynamic_variables)),
                            "shell_enabled": dotted_name(item.func) in {"os.system", "os.popen"} or any(
                                keyword.arg == "shell"
                                and isinstance(keyword.value, ast.Constant)
                                and keyword.value.value is True
                                for keyword in item.keywords
                            ),
                            "request_derived_arguments": sorted(
                                set(argument_names).intersection(request_derived)
                            ),
                        })
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
                    "same_module_request_hooks": request_hooks,
                    "request_input_accesses": request_inputs,
                    "request_derived_variables": sorted(request_derived),
                    "dynamic_variables": sorted(dynamic_variables),
                    "security_sinks": sinks,
                    "request_derived_sinks": [
                        sink for sink in sinks
                        if sink["request_derived_arguments"]
                    ],
                    "mutating_calls": mutating_calls,
                    "call_sites": call_sites,
                    "sensitive_output_markers": output_keys,
                    "returns_only_literals": literal_returns,
                })
        functions = []
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls_in_function = [
                item for item in ast.walk(function) if isinstance(item, ast.Call)
            ]
            assignment_nodes = [
                item for item in ast.walk(function)
                if isinstance(item, (ast.Assign, ast.AnnAssign))
            ]
            dependencies: Dict[str, set[str]] = {}
            request_derived = set()
            dynamic_variables = set()
            for assignment in assignment_nodes:
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
                    item.id for item in ast.walk(value)
                    if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                }
                for target_name in target_names:
                    dependencies.setdefault(target_name, set()).update(value_names)
                    if request_input_expression(value):
                        request_derived.add(target_name)
                    if any(
                        isinstance(item, (ast.JoinedStr, ast.BinOp))
                        for item in ast.walk(value)
                    ):
                        dynamic_variables.add(target_name)
            for _ in range(len(dependencies)):
                for target_name, value_names in dependencies.items():
                    if value_names.intersection(request_derived):
                        request_derived.add(target_name)
                    if value_names.intersection(dynamic_variables):
                        dynamic_variables.add(target_name)

            sinks = []
            for call in calls_in_function:
                category = security_sink(call)
                if not category:
                    continue
                argument_names = sorted({
                    child.id for argument in [*call.args, *(
                        keyword.value for keyword in call.keywords
                    )]
                    for child in ast.walk(argument)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                })
                sinks.append({
                    "category": category,
                    "call": dotted_name(call.func),
                    "line": call.lineno,
                    "argument_names": argument_names,
                    "dynamic_argument": any(
                        isinstance(child, (ast.JoinedStr, ast.BinOp))
                        for argument in call.args
                        for child in ast.walk(argument)
                    ) or bool(set(argument_names).intersection(dynamic_variables)),
                    "shell_enabled": dotted_name(call.func) in {"os.system", "os.popen"} or any(
                        keyword.arg == "shell"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                        for keyword in call.keywords
                    ),
                    "request_derived_arguments": sorted(
                        set(argument_names).intersection(request_derived)
                    ),
                })
            function_calls = [
                {"call": dotted_name(call.func), "line": call.lineno}
                for call in calls_in_function
            ]
            decorators = [dotted_name(item) for item in function.decorator_list]
            function_names = [*decorators, *(item["call"] for item in function_calls)]
            functions.append({
                "name": function.name,
                "start_line": function.lineno,
                "end_line": function.end_lineno or function.lineno,
                "local_auth_guards": sorted({
                    name for name in function_names
                    if any(marker in name.lower() for marker in auth_markers)
                }),
                "mutating_calls": sorted({
                    name for name in (item["call"] for item in function_calls)
                    if any(marker in name.rsplit(".", 1)[-1].lower() for marker in mutation_markers)
                }),
                "call_sites": function_calls,
                "request_derived_variables": sorted(request_derived),
                "dynamic_variables": sorted(dynamic_variables),
                "security_sinks": sinks,
            })
        reports.append({
            "file": path,
            "routes": routes,
            "functions": functions,
            "request_hooks": request_hooks,
        })
    functions_by_name: Dict[str, list[tuple[str, Mapping[str, object]]]] = {}
    project_hooks = []
    for report in reports:
        path = str(report.get("file", ""))
        for hook in report.get("request_hooks", []):
            if isinstance(hook, Mapping):
                project_hooks.append({"file": path, **hook})
        for function in report.get("functions", []):
            if isinstance(function, Mapping) and isinstance(function.get("name"), str):
                functions_by_name.setdefault(function["name"], []).append((path, function))

    for report in reports:
        path = str(report.get("file", ""))
        for route in report.get("routes", []):
            if not isinstance(route, dict):
                continue
            helper_context = []
            for call_site in route.get("call_sites", []):
                if not isinstance(call_site, Mapping):
                    continue
                call_name = str(call_site.get("call", ""))
                function_name = call_name.rsplit(".", 1)[-1]
                matches = functions_by_name.get(function_name, [])
                for helper_path, helper in matches:
                    if helper_path == path:
                        continue
                    helper_context.append({
                        "call": call_name,
                        "call_line": call_site.get("line"),
                        "file": helper_path,
                        "function": function_name,
                        "start_line": helper.get("start_line"),
                        "end_line": helper.get("end_line"),
                        "local_auth_guards": helper.get("local_auth_guards", [])[:8],
                        "mutating_calls": helper.get("mutating_calls", [])[:8],
                        "request_derived_variables": helper.get(
                            "request_derived_variables", []
                        )[:8],
                        "security_sinks": helper.get("security_sinks", [])[:8],
                    })
                    if len(helper_context) >= 8:
                        break
                if len(helper_context) >= 8:
                    break
            route["cross_file_helpers"] = helper_context
            route["project_request_hooks"] = [
                hook for hook in project_hooks if hook.get("file") != path
            ][:8]

    return reports


def anchor_auth_candidate_to_mutation(
    candidate: Mapping[str, Any],
    sources: Mapping[str, str],
    source_facts: list[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Sharpen an existing auth candidate to one unique route mutation call."""
    anchored = dict(candidate)
    if str(candidate.get("category", "")).strip().lower() != "authentication_authorization":
        return anchored

    path = str(candidate.get("file", "")).strip().replace("\\", "/")
    source = next(
        (
            text for source_path, text in sources.items()
            if str(source_path).replace("\\", "/").strip() == path
        ),
        None,
    )
    if not isinstance(source, str):
        return anchored
    try:
        candidate_start = int(candidate.get("start_line", 0) or 0)
        candidate_end = int(
            candidate.get("end_line", candidate_start) or candidate_start
        )
    except (TypeError, ValueError):
        return anchored
    if candidate_start < 1 or candidate_end < candidate_start:
        return anchored

    report = next(
        (
            item for item in source_facts
            if str(item.get("file", "")).replace("\\", "/").strip() == path
        ),
        None,
    )
    if report is None:
        return anchored
    matching_routes = []
    for route in report.get("routes", []):
        if not isinstance(route, Mapping):
            continue
        try:
            route_start = int(route.get("start_line", 0) or 0)
            route_end = int(route.get("end_line", route_start) or route_start)
        except (TypeError, ValueError):
            continue
        if candidate_start <= route_end and candidate_end >= route_start:
            matching_routes.append(route)
    if len(matching_routes) != 1:
        return anchored

    route = matching_routes[0]
    if route.get("local_auth_guards") or any(
        isinstance(hook, Mapping) and hook.get("enforces_auth")
        for hook in route.get("same_module_request_hooks", [])
    ):
        return anchored
    mutating_calls = set(route.get("mutating_calls") or [])
    sites = {
        int(site["line"]): str(site["call"])
        for site in route.get("call_sites", [])
        if isinstance(site, Mapping)
        and site.get("call") in mutating_calls
        and isinstance(site.get("line"), int)
        and not isinstance(site.get("line"), bool)
    }
    if not sites:
        return anchored
    in_range = {
        line: call for line, call in sites.items()
        if candidate_start <= line <= candidate_end
    }
    if len(in_range) == 1:
        line, call = next(iter(in_range.items()))
    elif len(sites) == 1:
        line, call = next(iter(sites.items()))
    else:
        return anchored

    source_lines = source.splitlines()
    if line < 1 or line > len(source_lines):
        return anchored
    exact_quote = source_lines[line - 1].strip()
    if not exact_quote:
        return anchored
    anchored["citation_reanchored"] = {
        "method": "unique_route_mutation_call",
        "from": {
            "start_line": candidate.get("start_line"),
            "end_line": candidate.get("end_line"),
            "quote": candidate.get("quote"),
            "grounded": candidate.get("grounded"),
        },
        "call": call,
    }
    anchored.update({
        "start_line": line,
        "end_line": line,
        "quote": exact_quote,
        "grounded": True,
    })
    anchored.pop("reason", None)
    return anchored


def _reported_by_set(value: object) -> set[str]:
    """Normalize a candidate's reporter list to ``provider:model`` strings."""
    result: set[str] = set()
    if not isinstance(value, (list, tuple)):
        return result
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            result.add(f"{item[0]}:{item[1]}")
        elif isinstance(item, str) and item:
            result.add(item)
    return result


def _review_is_self_review(
    candidate: Mapping[str, object],
    review: Mapping[str, object],
) -> bool:
    """Prefer the explicit flag, but infer it for older/foreign outcomes."""
    if "self_review" in review:
        return bool(review.get("self_review"))
    reporter = f"{review.get('provider')}:{review.get('model')}"
    return reporter in _reported_by_set(candidate.get("reported_by"))


def triage_grounded_findings(
    validations: list[Mapping[str, object]],
    source_facts: list[Mapping[str, object]],
    challenge_outcomes: Optional[list[Mapping[str, object]]] = None,
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
    challenge_by_id: Dict[str, list[Dict[str, object]]] = {}
    for outcome in challenge_outcomes or []:
        candidate_id = outcome.get("candidate_id")
        status = outcome.get("status")
        if (
            isinstance(candidate_id, str)
            and status in {"supported", "refuted", "unresolved"}
        ):
            challenge_by_id.setdefault(candidate_id, []).append({
                "status": str(status),
                "reason": str(outcome.get("reason", "")),
                "safe_test": str(outcome.get("safe_test", "")),
                **(
                    {"provider": str(outcome["provider"])}
                    if outcome.get("provider") else {}
                ),
                **(
                    {"model": str(outcome["model"])}
                    if outcome.get("model") else {}
                ),
                **(
                    {"self_review": True}
                    if outcome.get("self_review") else {}
                ),
            })
    auth_claim = re.compile(
        r"\b(unauth|unauthor|authentic|authorization|access control|privilege)\w*\b",
        re.I,
    )
    input_claim = re.compile(r"\b(input|injection|validation|unvalidated|saniti[sz])\b", re.I)
    for validation in validations:
        items = validation.get("findings_checked", [])
        if not isinstance(items, list):
            continue
        reporter: Optional[List[str]] = None
        provider = validation.get("provider")
        model = validation.get("model")
        if (
            isinstance(provider, str) and provider
            and isinstance(model, str) and model
        ):
            reporter = [provider, model]
        for item in items:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("finding", "Unspecified candidate"))
            path = str(item.get("file", "")).replace("\\", "/")
            start = item.get("start_line")
            end = item.get("end_line")
            category = str(item.get("category", "unknown")).strip().lower()
            quote = " ".join(str(item.get("quote", "")).split())
            key = (path, start, end, quote)
            grounded = bool(item.get("grounded"))
            candidate_id = candidate_reference_id(
                path, start, end, category, quote
            )
            candidate: Dict[str, object] = {
                "title": title,
                "file": path,
                "start_line": start,
                "end_line": end,
                "category": category,
                "quote": quote,
                **{
                    field: str(item[field])
                    for field in ("attack_path", "preconditions", "unverified")
                    if item.get(field)
                },
                "quote_grounded": grounded,
                "status": "unverified_candidate",
                "reason": "Static source patterns do not establish the complete security behavior.",
                "reported_categories": [category],
                "reported_by": [reporter] if reporter else [],
                "candidate_id": candidate_id,
            }
            reviews = challenge_by_id.get(candidate_id, [])
            if reviews:
                challenge_statuses = {review["status"] for review in reviews}
                candidate["challenge_status"] = (
                    next(iter(challenge_statuses))
                    if len(challenge_statuses) == 1
                    else "mixed"
                )
                self_reviews = sum(
                    _review_is_self_review(candidate, review)
                    for review in reviews
                )
                candidate["challenge_independence"] = (
                    "self_only" if self_reviews == len(reviews)
                    else "mixed" if self_reviews
                    else "independent"
                )
                candidate["challenge_reviews"] = [
                    {
                        **review,
                        "self_review": _review_is_self_review(candidate, review),
                    }
                    for review in reviews
                ]
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
                            if (
                                not mutating
                                and not sensitive_output
                                and route.get("returns_only_literals")
                                and not route.get("same_module_request_hooks")
                            ):
                                candidate.update({
                                    "status": "not_actionable_for_handler",
                                    "reason": (
                                        "Cited route has no detected mutating call or sensitive-output "
                                        "marker and returns only literals; a missing local auth guard "
                                        "alone does not establish a vulnerability."
                                    ),
                                })
                            elif not route.get("local_auth_guards"):
                                hook_names = [
                                    hook.get("handler", "unknown")
                                    for hook in route.get("same_module_request_hooks", [])
                                    if isinstance(hook, Mapping)
                                ]
                                hook_context = (
                                    " Same-module request hooks were observed "
                                    f"({', '.join(hook_names)}); inspect their behavior."
                                    if hook_names else ""
                                )
                                candidate.update({
                                    "status": "unverified_authorization_gap",
                                    "reason": (
                                        "No local auth guard pattern was detected before a sensitive "
                                        "operation; middleware, decorators outside this handler, and "
                                        "effective route reachability still require verification."
                                        + hook_context
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
            existing = findings.get(key)
            if existing is None:
                findings[key] = candidate
            else:
                categories = existing["reported_categories"]
                if category not in categories:
                    categories.append(category)
                reporters = existing.setdefault("reported_by", [])
                for pair in candidate.get("reported_by", []):
                    if pair not in reporters:
                        reporters.append(pair)
                titles = existing.setdefault("alternate_titles", [])
                if title != existing["title"] and title not in titles:
                    titles.append(title)
                if not existing.get("challenge_reviews") and candidate.get("challenge_reviews"):
                    existing["challenge_reviews"] = candidate["challenge_reviews"]
                    existing["challenge_status"] = candidate.get("challenge_status")
                    existing["challenge_independence"] = candidate.get(
                        "challenge_independence"
                    )
    return list(findings.values())


def _verdict_evidence_entry(verdict: Mapping[str, Any]) -> Dict[str, object]:
    """Extract the cited evidence for one judged verdict record.

    Tolerates hand-built verdict shapes: missing fields stay None rather
    than raising, so a malformed record degrades to an explicit unjudged
    entry instead of crashing the gate.
    """
    candidate = verdict.get("candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    provider = verdict.get("provider")
    model = verdict.get("model")
    entry: Dict[str, object] = {
        "verdict": str(verdict.get("verdict", "UNJUDGED")).upper(),
        "category": candidate.get("category"),
        "file": candidate.get("file"),
        "line": candidate.get("start_line"),
        "quote": candidate.get("quote"),
        "grounded": bool(candidate.get("grounded")),
        "judge": (
            f"{provider}:{model}"
            if provider is not None or model is not None
            else None
        ),
        "margin": verdict.get("margin"),
    }
    provider_error = verdict.get("provider_error")
    if provider_error:
        entry["provider_error"] = provider_error
    return entry


def _assess_council_verdict(
    verdicts: Optional[Sequence[Mapping[str, Any]]],
    gate_status: str,
) -> Dict[str, object]:
    """Derive the council's security conclusion from its judged verdicts.

    Staged scoring counts exactly the UPHOLD verdicts as predictions, so the
    gate conclusion is drawn from the same verdict set: UPHELD_FINDINGS with
    cited evidence when any candidate is upheld, NO_UPHELD_FINDINGS when the
    council judged candidates and killed every one, CONTESTED when verdicts
    exist but none were upheld and some deadlocked, UNJUDGED when verdicts
    exist but none were adjudicated (judge outage or unparseable judgment).
    NOT_ASSESSED is returned only when there are zero verdict records —
    nothing was judged, so no conclusion is claimed.
    """
    judged = [v for v in (verdicts or []) if isinstance(v, Mapping)]
    if not judged:
        return {
            "security_conclusion": "NOT_ASSESSED",
            "security_evidence": [],
            "verdict_counts": {
                "uphold": 0,
                "kill": 0,
                "contested": 0,
                "unjudged": 0,
            },
            "meaning": (
                "PASS_GROUNDING_ONLY means at least one member's review outputs "
                "passed the source-grounding checks. It does not prove "
                "exploitability, completeness, or safety."
            ),
        }
    upheld = [v for v in judged if str(v.get("verdict", "")).upper() == "UPHOLD"]
    contested = [v for v in judged if str(v.get("verdict", "")).upper() == "CONTESTED"]
    kills = sum(1 for v in judged if str(v.get("verdict", "")).upper() == "KILL")
    unjudged = [
        v
        for v in judged
        if str(v.get("verdict", "")).upper() not in ("UPHOLD", "KILL", "CONTESTED")
    ]
    counts = {
        "uphold": len(upheld),
        "kill": kills,
        "contested": len(contested),
        "unjudged": len(unjudged),
    }
    trial_note = (
        ""
        if gate_status == "PASS_GROUNDING_ONLY"
        else " The evidence gate FAILED, so this trial is invalid for scoring; "
        "treat these verdicts as diagnostic leads, not a qualified assessment."
    )
    if upheld:
        return {
            "security_conclusion": "UPHELD_FINDINGS",
            "security_evidence": [_verdict_evidence_entry(v) for v in upheld],
            "verdict_counts": counts,
            "meaning": (
                f"The council judged {len(judged)} candidate(s) and UPHELD "
                f"{len(upheld)} with the cited source evidence below. Upheld "
                "findings are evidence-backed leads, not exploitability proof, "
                "completeness, or safety." + trial_note
            ),
        }
    if contested:
        return {
            "security_conclusion": "CONTESTED",
            "security_evidence": [_verdict_evidence_entry(v) for v in contested],
            "verdict_counts": counts,
            "meaning": (
                f"The council judged {len(judged)} candidate(s), upheld none, "
                f"and deadlocked on {len(contested)}: no consensus, so no "
                "finding is established and no clean bill is given. Human or "
                "stronger-judge review is required." + trial_note
            ),
        }
    if unjudged:
        return {
            "security_conclusion": "UNJUDGED",
            "security_evidence": [_verdict_evidence_entry(v) for v in unjudged],
            "verdict_counts": counts,
            "meaning": (
                f"The council judged {len(judged)} candidate(s), upheld none, "
                f"and left {len(unjudged)} unadjudicated (judge outage or "
                "unparseable judgment). No finding is established and no clean "
                "bill is given." + trial_note
            ),
        }
    return {
        "security_conclusion": "NO_UPHELD_FINDINGS",
        "security_evidence": [],
        "verdict_counts": counts,
        "meaning": (
            f"The council judged {len(judged)} candidate(s) and upheld none: "
            "every judged candidate was KILLed. This is an assessed negative "
            "over the judged candidates only, not proof of safety." + trial_note
        ),
    }


def assess_evidence_gate(
    sources: Mapping[str, str],
    validations: list[Mapping[str, object]],
    provider_errors: list[str],
    candidate_triage: list[Mapping[str, object]],
    verdicts: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, object]:
    """Fail closed when red-team claims cannot be grounded to supplied evidence.

    The council is redundant: any fully-clean member carries the trial.
    Provider errors fail the gate only when they leave zero usable review
    outputs. Surviving members are judged on grounding merit; members with
    failures are demoted to warnings (never silently dropped) while at
    least one member passes. Unsupported triage citations are scoped to
    their reporting members via reported_by — one bad citation fails its
    reporters, never the clean members. Sources, vacuous-input, and
    unattributed failures always fail closed.

    When the typed-judge verdicts are supplied, the gate also reports the
    council's assessed security conclusion drawn from those verdicts (the
    same UPHOLD set that staged scoring counts as predictions). Without
    verdicts no conclusion is claimed and the conclusion stays NOT_ASSESSED.
    """
    failures: list = []  # (member-key-or-None, message) pairs
    warnings: list[str] = []
    if not sources:
        failures.append(
            (None, "No source files were supplied; claims cannot be grounded.")
        )
    elif not any(isinstance(text, str) and text.strip() for text in sources.values()):
        failures.append(
            (None, "Supplied source files are empty; claims cannot be grounded.")
        )
    # Per-member all-or-nothing: a member with zero usable output is fully
    # throttled — its errors degrade to warnings and its vacuous validations
    # are skipped. A member with partial output stays under full judgment:
    # its errors fail and every validation is looped. Unparseable errors
    # fail closed.
    member_usable: dict = {}
    for entry in validations:
        if isinstance(entry, dict) and (
            entry.get("review_complete") or entry.get("explicit_no_findings")
        ):
            member_usable[(entry.get("provider"), entry.get("model"))] = True

    def _error_member(message: object):
        match = re.match(r"(\S+):(\S+)\s", str(message or ""))
        return (match.group(1), match.group(2)) if match else None

    judged = []
    for entry in validations:
        if not isinstance(entry, dict):
            continue
        key = (entry.get("provider"), entry.get("model"))
        if key == (None, None) or key in member_usable:
            judged.append(entry)
    if not judged:
        failures.append(
            (
                None,
                "No usable review outputs to judge — "
                "the gate cannot pass on vacuous input.",
            )
        )
    for error in provider_errors:
        member = _error_member(error)
        if member is not None and member not in member_usable:
            warnings.append(f"Review degraded, survivors judged: {error}")
        else:
            failures.append((member, f"Review stage failed: {error}"))

    bad_validation_statuses = {
        "no_source_provided",
        "unsupported_citations",
        "unstructured_claims",
    }
    for index, validation in enumerate(judged, start=1):
        key = (validation.get("provider"), validation.get("model"))
        owner = key if key != (None, None) else None
        status = validation.get("status")
        if status in bad_validation_statuses:
            failures.append(
                (owner, f"Review output {index} failed grounding: {status}.")
            )
        if validation.get("review_complete") is False:
            failures.append(
                (
                    owner,
                    f"Review output {index} did not complete its assigned review stage.",
                )
            )
        elif (
            status == "no_structured_findings"
            and not validation.get("review_complete")
            and not validation.get("explicit_no_findings")
        ):
            failures.append(
                (
                    owner,
                    f"Review output {index} returned no findings without an explicit "
                    "no-finding result.",
                )
            )

    for candidate in candidate_triage:
        if candidate.get("status") == "unsupported_citation":
            message = (
                "Candidate citation is not grounded: "
                f"{candidate.get('title', 'unnamed candidate')} "
                f"({candidate.get('file', 'unknown file')}:"
                f"{candidate.get('start_line', '?')})."
            )
            # Scope the veto to the owning members: one hallucinated quote
            # fails its reporters, never the clean members. Ownership
            # unknown (hand-built triage without reported_by) fails closed
            # as a global failure.
            owners = set()
            reported = candidate.get("reported_by") or []
            if isinstance(reported, list):
                for pair in reported:
                    if (
                        isinstance(pair, (list, tuple))
                        and len(pair) == 2
                        and pair[0]
                        and pair[1]
                    ):
                        owners.add((pair[0], pair[1]))
            if not owners:
                failures.append((None, message))
            else:
                for owner in sorted(owners, key=str):
                    failures.append((owner, message))

    global_failures = [message for owner, message in failures if owner is None]
    member_failures: Dict[Tuple[object, object], List[str]] = {}
    for owner, message in failures:
        if owner is not None:
            member_failures.setdefault(owner, []).append(message)
    passing_members = sorted(
        member for member in member_usable if member not in member_failures
    )
    if global_failures or not passing_members:
        failed_checks = [message for _, message in failures]
        status = "FAIL"
    else:
        for owner in sorted(member_failures, key=str):
            provider, model = owner
            warnings.extend(
                f"Non-passing member {provider}:{model}: {message}"
                for message in member_failures[owner]
            )
        failed_checks = []
        status = "PASS_GROUNDING_ONLY"

    conclusion = _assess_council_verdict(verdicts, status)
    return {
        "status": status,
        "failed_checks": failed_checks,
        "provider_warnings": warnings,
        "passing_members": [
            f"{provider}:{model}" for provider, model in passing_members
        ],
        "source_files_checked": len(sources),
        "review_outputs_checked": len(validations),
        "candidate_count": len(candidate_triage),
        "security_conclusion": conclusion["security_conclusion"],
        "security_evidence": conclusion["security_evidence"],
        "verdict_counts": conclusion["verdict_counts"],
        "meaning": conclusion["meaning"],
    }


_NUMBERED_DISPLAY_PREFIX = re.compile(r"^(?P<line>\d+) \| (?P<code>.*)$")
_BARE_PIPE_PREFIX = re.compile(r"^\| (?P<code>.*)$")


def strip_display_prefixes(
    quote_lines: List[str], start: int
) -> Tuple[List[str], bool]:
    """Remove echoed source-display line prefixes from a quoted excerpt.

    Supplied source is rendered with strictly `%05d | <code>` display
    prefixes and models are told not to copy them, but echoes come back in
    model-normalized forms (`00003 | code`, `3 | code`, `| code`) — rendering
    artifacts, not forged evidence. Assurance rests on the sequential and
    verbatim checks below, not on exact artifact provenance. Numbered
    prefixes strip only when every line carries one AND the numbers run
    sequentially from the cited start (which positively validates the
    range); digit runs over 10 never strip (no real file needs them, and
    unbounded int() can throw). A stray bare `| ` on any line strips
    unconditionally since it is never valid leading code in a quoted excerpt
    and the remainder must still match source verbatim. Returns cleaned
    lines plus whether anything was removed.
    """
    if not quote_lines:
        return quote_lines, False
    numbered = [_NUMBERED_DISPLAY_PREFIX.fullmatch(line) for line in quote_lines]
    if all(
        match is not None
        and len(match.group("line")) <= 10
        and int(match.group("line")) == start + index
        for index, match in enumerate(numbered)
    ):
        return (
            [match.group("code") for match in numbered if match is not None],
            True,
        )
    cleaned = [_BARE_PIPE_PREFIX.sub(r"\g<code>", line) for line in quote_lines]
    return cleaned, cleaned != quote_lines


def _extract_quoted_source(quote_tail: str) -> tuple[str, int]:
    """Extract a source excerpt and its answer-text span before review details."""
    detail = _QUOTE_DETAIL.search(quote_tail)
    excerpt = quote_tail[:detail.start()] if detail else quote_tail
    excerpt = excerpt.strip()
    offset = len(quote_tail) - len(quote_tail.lstrip())

    fenced = re.match(r"```[^\n`]*\n?(.*?)```", excerpt, re.DOTALL)
    if fenced:
        quote = fenced.group(1)
        consumed = offset + fenced.end()
    elif excerpt.startswith("`"):
        quote_lines = excerpt.splitlines()
        if len(quote_lines) > 1 and all(
            line.strip().startswith("`") and line.strip().endswith("`")
            for line in quote_lines if line.strip()
        ):
            quote = "\n".join(
                line.strip()[1:-1] if line.strip() else ""
                for line in quote_lines
            )
            consumed = offset + len(excerpt)
        else:
            wrapped = re.match(
                r"`(?P<quote>.*)`[ \t]*(?:\r?\n)?\Z",
                excerpt,
                re.DOTALL,
            )
            if wrapped:
                quote = wrapped.group("quote")
                consumed = offset + wrapped.end()
            else:
                quote = excerpt
                consumed = offset + len(excerpt)
    else:
        if detail:
            quote = excerpt
            consumed = offset + len(excerpt)
        else:
            quote = next((line.strip() for line in excerpt.splitlines() if line.strip()), "")
            consumed = offset + len(quote)

    # Some providers serialize a multiline citation using visible escapes.
    decoded = quote.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    if decoded != quote:
        quote = decoded
    return quote, consumed


def _structured_detail_span(detail_tail: str) -> int:
    """Cover only explicitly labeled review fields and their list continuations."""
    consumed = 0
    active_field = False
    for line in detail_tail.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or re.match(
            r"(?i)^\s*(?:ATTACK PATH|PRECONDITIONS|UNVERIFIED):", line
        ):
            active_field = True if stripped else active_field
            consumed += len(line)
            continue
        if re.match(r"^\s*(?:[-*]\s+|```)", line) or (
            active_field and line[:1].isspace()
        ):
            consumed += len(line)
            continue
        break
    return consumed


def _has_unstructured_claim(text: str) -> bool:
    for match in _CLAIM.finditer(text):
        sentence_start = max(
            text.rfind("\n", 0, match.start()),
            text.rfind(".", 0, match.start()),
            text.rfind("!", 0, match.start()),
            text.rfind("?", 0, match.start()),
        )
        prefix = text[sentence_start + 1:match.start()]
        if re.search(
            r"\b(?:no|not|never|without|none|neither|cannot|can't|"
            r"doesn't|isn't|does\s+not|is\s+not)\b"
            r"(?:\W+\w+){0,5}\W*$",
            prefix,
            re.IGNORECASE,
        ):
            continue
        return True
    return False


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
        quote, quote_span = _extract_quoted_source(quote_tail)
        detail_tail = quote_tail[quote_span:]
        detail_span = _structured_detail_span(detail_tail)
        covered_claim_spans.append((
            match.start(),
            match.end() + quote_span + detail_span,
        ))
        quote_lines = quote.strip("\r\n").splitlines()
        quote_lines, display_line_prefix_removed = strip_display_prefixes(
            quote_lines, start
        )
        quote_is_multiline = len(quote_lines) > 1
        quote_first_line = " ".join(quote_lines[0].split()) if quote_lines else ""
        quote = " ".join(quote.split())
        if display_line_prefix_removed:
            quote = " ".join(" ".join(quote_lines).split())
        reason = None
        if source is None:
            reason = "cited file was not among the supplied source files"
        elif end < start or start < 1 or end > len(source.splitlines()):
            reason = "cited line range is outside the supplied file"
        elif end - start > 7:
            reason = "cited line range is too broad; cite at most eight lines"
        elif not quote:
            reason = "citation did not include a source quote"
        else:
            source_lines = source.splitlines()[start - 1:end]
            first_line = " ".join(source_lines[0].split())
            cited_text = " ".join(
                " ".join(" ".join(line.split()) for line in source_lines).split()
            )
            normalized_quote = " ".join(quote.split())
            if quote_is_multiline:
                grounded_quote = (
                    len(quote_lines) == end - start + 1
                    and normalized_quote in cited_text
                    and quote_first_line in first_line
                )
            else:
                grounded_quote = end == start and normalized_quote in first_line
            if not grounded_quote:
                reason = "quoted lines do not exactly match the cited source range"
        raw_cat = (match.group("category") or "unknown").strip().lower()
        checked.append({
            "finding": match.group("title").strip(),
            "category": canonical_finding_category(raw_cat),
            "raw_category": raw_cat,
            "file": path,
            "start_line": start,
            "end_line": end,
            "quote": quote,
            "display_line_prefix_removed": display_line_prefix_removed,
            **{
                field: " ".join(found.group(1).split())
                for field, label in (
                    ("attack_path", "ATTACK PATH"),
                    ("preconditions", "PRECONDITIONS"),
                    ("unverified", "UNVERIFIED"),
                )
                if (found := re.search(
                    rf"(?im)^\s*{label}:\s*(.+?)\s*$", detail_tail
                ))
            },
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
    unstructured_claim = _has_unstructured_claim(unstructured_text)
    explicit_no_findings = bool(re.search(
        r"\bNO ACTIONABLE CANDIDATE\b|\bNO FINDINGS?\b",
        answer_text,
        re.IGNORECASE,
    ))
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
        "explicit_no_findings": explicit_no_findings,
        "review_complete": bool(findings) or explicit_no_findings,
        "security_conclusion": "not_assessed",
    }
