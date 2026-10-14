"""General Python injection/deserialization engine for the red-team judge.

Votes on candidates in sql_injection, command_injection, path_traversal and
unsafe_deserialization by AST-checking the cited file for a real sink within
+/-3 lines of the cited line, untrusted (request-derived or parameter) input
reaching the sink's arguments, and the absence of a category-specific guard.
Original code; sink-name patterns follow omnicli.deterministic /
omnicli.evidence conventions but the analysis is engine-shaped.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Set

from . import (
    Engine,
    EngineFact,
    EngineVote,
    NOT_COVERED,
    NOT_SUPPORTED,
    SUPPORTED,
    register_engine,
)


def _norm(path: str) -> str:
    return path.replace("\\", "/")


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


# ---------------------------------------------------------------------------
# Sink matchers: name predicates keyed by category.
# ---------------------------------------------------------------------------

_SQL_TAILS = {"execute", "executemany", "executescript", "query", "raw", "raw_query"}
_DESER_NAMES = {"pickle.loads", "pickle.load", "yaml.load", "jsonpickle.decode",
                "marshal.loads"}


def _sink_matches(category: str, name: str) -> bool:
    tail = name.rsplit(".", 1)[-1].lower()
    if category == "sql_injection":
        return tail in _SQL_TAILS
    if category == "command_injection":
        lowered = name.lower()
        return lowered in {"os.system", "os.popen", "eval", "exec"} or lowered.startswith(
            "subprocess."
        )
    if category == "unsafe_deserialization":
        return name.lower() in _DESER_NAMES
    if category == "path_traversal":
        lowered = name.lower()
        return lowered in {"open", "os.path.join", "pathlib.path"}
    return False


def _dynamic_argument(call: ast.Call, tainted: Set[str]) -> bool:
    for argument in call.args:
        for child in ast.walk(argument):
            if isinstance(child, (ast.JoinedStr, ast.BinOp, ast.Call, ast.Subscript)):
                return True
            if isinstance(child, ast.Name) and child.id in tainted:
                return True
    return False


_INPUT_MARKERS = {
    "args", "form", "json", "values", "get_json", "get_data", "query_params",
    "headers", "cookies", "data", "body",
}


def _request_derived_names(function: ast.AST) -> Set[str]:
    """Local names in the function assigned from request-derived input."""
    derived: Set[str] = set()
    for item in ast.walk(function):
        if isinstance(item, ast.Assign):
            targets: List[ast.AST] = list(item.targets)
        elif isinstance(item, ast.AnnAssign):
            targets = [item.target]
        else:
            continue
        value = item.value
        if value is None or not _request_derived(value):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                derived.add(target.id)
    return derived


def _tainted_names(function: ast.AST, param_names: Set[str]) -> Set[str]:
    """Names transitively assigned from request input or function params."""
    tainted: Set[str] = set(param_names)
    dependencies: Dict[str, Set[str]] = {}
    for item in ast.walk(function):
        if isinstance(item, ast.Assign):
            targets: List[ast.AST] = list(item.targets)
        elif isinstance(item, ast.AnnAssign):
            targets = [item.target]
        else:
            continue
        value = item.value
        if value is None:
            continue
        value_names = {
            child.id for child in ast.walk(value)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }
        for target in targets:
            if isinstance(target, ast.Name):
                dependencies.setdefault(target.id, set()).update(value_names)
                if _request_derived(value):
                    tainted.add(target.id)
    for _ in range(len(dependencies)):
        for target_name, value_names in dependencies.items():
            if value_names & tainted:
                tainted.add(target_name)
    return tainted


def _request_derived(node: ast.AST) -> bool:
    """True if any request.* input access appears under node."""
    for item in ast.walk(node):
        if isinstance(item, (ast.Attribute, ast.Subscript)):
            inner = item
            while isinstance(inner, ast.Subscript):
                inner = inner.value
            dotted = _dotted(inner)
            head, _, rest = dotted.partition(".")
            if head in {"request", "req"} and any(
                part.lower() in _INPUT_MARKERS for part in rest.split(".")
            ):
                return True
    return False


def _arg_names(call: ast.Call) -> Set[str]:
    return {
        child.id
        for argument in [*call.args, *(keyword.value for keyword in call.keywords)]
        for child in ast.walk(argument)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _constant_arg(call: ast.Call, index: int = 0) -> bool:
    if len(call.args) <= index:
        return False
    return isinstance(call.args[index], ast.Constant)


def _enclosing_function(tree: ast.Module, line: int) -> Optional[ast.AST]:
    best: Optional[ast.AST] = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno or node.lineno
            if node.lineno <= line <= end and (
                best is None or node.lineno > best.lineno
            ):
                best = node
    return best


def _enclosing_class_dotted(tree: ast.Module, function: ast.AST) -> str:
    """Dotted path to the class containing the function, or ""."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in ast.walk(node):
                if child is function:
                    return node.name
    return ""


_META_MARKERS = {"detect", "analyz", "security_sink", "scanner"}


def _parent(tree: ast.Module, target: ast.AST) -> Optional[ast.AST]:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            if child is target:
                return node
    return None


def _is_meta_code(tree: ast.Module, call: ast.Call, sink_line: int,
                  untrusted_reaches: bool) -> bool:
    """Is this call the analyzer's own bookkeeping rather than target code?

    Triggers when the call sits inside a detect/analyz/security_sink/scanner
    context and the call itself is the machinery: a bookkeeping-named call,
    the assignment of a bookkeeping result (e.g. ``sink = security_sink(...)``),
    or a sink call with no untrusted input reaching it (the analyzer
    demonstrating/scanning a sink pattern rather than a live vulnerable path).
    """
    function = _enclosing_function(tree, sink_line)
    if function is None:
        return False
    dotted = (function.name or "").lower()
    cls = _enclosing_class_dotted(tree, function).lower()
    full = f"{cls}.{dotted}" if cls else dotted
    if not any(marker in full for marker in _META_MARKERS):
        return False
    call_name = _dotted(call.func).lower()
    if any(marker in call_name for marker in ("security_sink", "detect", "scan",
                                              "analyz", "collect", "find_sink")):
        return True
    parent = _parent(tree, call)
    if isinstance(parent, (ast.Assign, ast.AnnAssign)):
        targets = (
            list(parent.targets) if isinstance(parent, ast.Assign) else [parent.target]
        )
        for target in targets:
            if isinstance(target, ast.Name) and any(
                marker in target.id.lower()
                for marker in ("sink", "finding", "fact", "detection")
            ):
                return True
    return not untrusted_reaches


def _guard_sql(call: ast.Call) -> Optional[str]:
    """Parameterized-query guard: placeholder query with separate params."""
    if len(call.args) < 2:
        return None
    query = call.args[0]
    if not isinstance(query, ast.Constant) or not isinstance(query.value, str):
        return None
    text = query.value
    # Placeholder styles: qmark (?), format (%s), and named (:name).
    if (
        "?" not in text
        and "%s" not in text
        and not re.search(r":[A-Za-z_]\w*", text)
    ):
        return None
    # The query string must be a plain literal: no f-string / %-format / + concat.
    for child in ast.walk(query):
        if isinstance(child, (ast.JoinedStr, ast.FormattedValue)):
            return None
    return "guard_parameterized"


def _guard_command(call: ast.Call, source_line_text: str) -> Optional[str]:
    for keyword in call.keywords:
        if keyword.arg == "shell":
            if isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                if call.args and isinstance(call.args[0], (ast.List, ast.Tuple)):
                    return "guard_no_shell"
    if "shlex.quote" in source_line_text:
        return "guard_no_shell"
    return None


def _guard_path(call: ast.Call, call_src: str, enclosing: Optional[ast.AST]) -> Optional[str]:
    has_canonical = ".resolve()" in call_src or "abspath(" in call_src
    if not has_canonical:
        return None
    text = ""
    if enclosing is not None:
        text = ast.dump(enclosing)
    if "startswith" in text or "commonprefix" in text or "is_relative_to" in text:
        return "guard_canonicalization"
    return None


@register_engine
class PythonGeneralEngine(Engine):
    """Deterministic verifier for classic Python injection sinks."""

    name = "python_general"
    version = "1.0.0"
    rule_set_date = "2026-10-07"
    covered_categories = (
        "sql_injection",
        "command_injection",
        "path_traversal",
        "unsafe_deserialization",
    )
    specificity = 1

    def analyze(
        self,
        sources: Dict[str, str],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, EngineVote]:
        normalized = {_norm(path): source for path, source in sources.items()}
        votes: Dict[str, EngineVote] = {}
        for candidate in candidates:
            candidate_id = str(candidate.get("id", ""))
            votes[candidate_id] = self._analyze_one(normalized, candidate_id, candidate)
        return votes

    # ------------------------------------------------------------------
    def _vote(self, candidate_id: str, vote: str,
              facts: List[EngineFact], reason: str) -> EngineVote:
        reasons = [reason] if reason else []
        reasons.extend(f.claim for f in facts[:3])
        return EngineVote(
            candidate_id=candidate_id,
            engine=self.name,
            engine_version=self.version,
            vote=vote,
            facts=facts,
            reason=reason,
            reasons=reasons,
        )

    def _out_of_domain(self, candidate_id: str, category: str) -> EngineVote:
        return self._vote(
            candidate_id, NOT_COVERED, [],
            f"category {category!r} is outside engine domain",
        )

    def _analyze_one(self, normalized: Dict[str, str], candidate_id: str,
                     candidate: Dict[str, Any]) -> EngineVote:
        category = str(candidate.get("category", ""))
        if category not in self.covered_categories:
            return self._out_of_domain(candidate_id, category)

        path = _norm(str(candidate.get("file", "")))
        source = normalized.get(path)
        if source is None:
            return self._vote(
                candidate_id, NOT_COVERED, [],
                f"cited file {path!r} not among provided sources",
            )
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return self._vote(
                candidate_id, NOT_COVERED, [],
                f"cited file {path!r} failed to parse at line {exc.lineno}",
            )

        try:
            start_line = int(candidate.get("start_line", 0) or 0)
        except (TypeError, ValueError):
            start_line = 0
        lines = source.splitlines()

        # --- Sink detection within +/-3 lines of the cited line ----------
        sink_call: Optional[ast.Call] = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and abs(node.lineno - start_line) <= 3:
                if _sink_matches(category, _dotted(node.func)):
                    sink_call = node
                    break
        if sink_call is None:
            return self._vote(
                candidate_id, NOT_SUPPORTED,
                [EngineFact(
                    id="sink_at_cited_location_absent",
                    kind="absence",
                    claim=f"no {category} sink call within ±3 lines of line {start_line}",
                    line=start_line,
                    code="",
                    supports_finding=False,
                )],
                f"no {category} sink at cited location",
            )

        sink_line = sink_call.lineno
        sink_code = lines[sink_line - 1].strip() if 0 < sink_line <= len(lines) else ""
        facts = [EngineFact(
            id="sink_present",
            kind="sink",
            claim=f"{category} sink call at line {sink_line}",
            line=sink_line,
            code=sink_code,
            supports_finding=True,
        )]

        # --- Source check: untrusted input reaches the sink's args -------
        enclosing = _enclosing_function(tree, sink_line)
        param_names: Set[str] = set()
        tainted: Set[str] = set()
        if enclosing is not None:
            func_args = getattr(enclosing, "args", None)
            param_names = {
                arg.arg for arg in (func_args.args if func_args else [])
            }
            tainted = _tainted_names(enclosing, param_names)
        arg_names = _arg_names(sink_call)
        reaches = (
            _request_derived(sink_call)
            or bool(arg_names & tainted)
        )

        # --- Meta-code: the analyzer analyzing itself --------------------
        if _is_meta_code(tree, sink_call, sink_line, reaches):
            facts.append(EngineFact(
                id="meta_code_detected",
                kind="meta",
                claim="cited call is the analyzer's own bookkeeping, not target code",
                line=sink_line,
                code=sink_code,
                supports_finding=False,
            ))
            return self._vote(
                candidate_id, NOT_SUPPORTED, facts,
                "cited call belongs to the analyzer's own bookkeeping",
            )

        if reaches:
            facts.append(EngineFact(
                id="untrusted_source_reaches_sink",
                kind="source",
                claim="request-derived or function-parameter value reaches the sink call",
                line=sink_line,
                code=sink_code,
                supports_finding=True,
            ))
        else:
            facts.append(EngineFact(
                id="source_at_sink_absent",
                kind="absence",
                claim="no request-derived input reaches this call",
                line=sink_line,
                code="",
                supports_finding=False,
            ))
            return self._vote(
                candidate_id, NOT_SUPPORTED, facts,
                "sink present but no untrusted input reaches it",
            )

        # --- Guard check by category --------------------------------------
        guard_id: Optional[str] = None
        if category == "sql_injection":
            guard_id = _guard_sql(sink_call)
        elif category == "command_injection":
            guard_id = _guard_command(sink_call, sink_code)
        elif category == "path_traversal":
            if not _dynamic_argument(sink_call, tainted):
                return self._vote(
                    candidate_id, NOT_SUPPORTED, facts + [EngineFact(
                        id="path_argument_constant",
                        kind="absence",
                        claim="path argument is constant; no dynamic input to traverse with",
                        line=sink_line,
                        code=sink_code,
                        supports_finding=False,
                    )],
                    "path sink called with a constant argument",
                )
            func_text = ""
            if enclosing is not None:
                end = enclosing.end_lineno or enclosing.lineno
                func_text = "\n".join(
                    lines[enclosing.lineno - 1:end]
                )
            guard_id = _guard_path(sink_call, func_text, enclosing)
        # unsafe_deserialization: any untrusted pickle.loads is bad -- no guard.

        if guard_id is not None:
            facts.append(EngineFact(
                id=guard_id,
                kind="guard",
                claim=f"sink is guarded ({guard_id})",
                line=sink_line,
                code=sink_code,
                supports_finding=False,
            ))
            return self._vote(
                candidate_id, NOT_SUPPORTED, facts,
                f"sink is guarded by {guard_id}",
            )

        return self._vote(
            candidate_id, SUPPORTED, facts,
            "real sink, untrusted input reaches it, no guard",
        )


__all__ = ["PythonGeneralEngine"]
