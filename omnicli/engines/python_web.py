"""Web-framework route engine for the red-team judge.

Votes on candidates in authentication_authorization and
sensitive_data_exposure by consulting omnicli.evidence's
analyze_python_security_facts route/hook analysis for the cited file.
Original engine-shaped code; route and hook facts come from omnicli.evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from omnicli.evidence import analyze_python_security_facts

from . import (
    Engine,
    EngineFact,
    EngineVote,
    NOT_COVERED,
    NOT_SUPPORTED,
    SUPPORTED,
    register_engine,
)

_ROUTE_TOLERANCE = 5


def _norm(path: str) -> str:
    return path.replace("\\", "/")


def _fact(fact_id: str, kind: str, claim: str, line: int,
          code: str, supports: bool) -> EngineFact:
    return EngineFact(
        id=fact_id,
        kind=kind,
        claim=claim,
        line=line,
        code=code,
        supports_finding=supports,
    )


def _route_span(route: Mapping[str, Any]) -> tuple:
    start = route.get("start_line")
    end = route.get("end_line")
    return (
        int(start) if isinstance(start, int) else None,
        int(end) if isinstance(end, int) else None,
    )


@register_engine
class PythonWebEngine(Engine):
    """Deterministic verifier for route-level auth and exposure claims."""

    name = "python_web"
    version = "1.0.0"
    rule_set_date = "2026-10-07"
    covered_categories = (
        "authentication_authorization",
        "sensitive_data_exposure",
    )
    specificity = 2

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

    def _analyze_one(self, normalized: Dict[str, str], candidate_id: str,
                     candidate: Dict[str, Any]) -> EngineVote:
        category = str(candidate.get("category", ""))
        if category not in self.covered_categories:
            return self._vote(
                candidate_id, NOT_COVERED, [],
                f"category {category!r} is outside engine domain",
            )

        path = _norm(str(candidate.get("file", "")))
        source = normalized.get(path)
        if source is None:
            return self._vote(
                candidate_id, NOT_COVERED, [],
                f"cited file {path!r} not among provided sources",
            )
        try:
            start_line = int(candidate.get("start_line", 0) or 0)
        except (TypeError, ValueError):
            start_line = 0

        reports = analyze_python_security_facts({path: source})
        report = next(
            (entry for entry in reports
             if isinstance(entry, Mapping) and entry.get("file") == path),
            None,
        )
        routes = (
            list(report.get("routes") or []) if isinstance(report, Mapping) else []
        )
        if not routes:
            return self._vote(
                candidate_id, NOT_COVERED, [],
                f"cited file {path!r} has no web routes; engine does not "
                "freelance on non-web code",
            )

        route = self._find_route(routes, start_line)
        if route is None:
            return self._vote(
                candidate_id, NOT_SUPPORTED,
                [_fact(
                    "route_at_location_absent", "absence",
                    f"no route covers line {start_line}; file has routes elsewhere",
                    start_line, "", False,
                )],
                f"no route at cited line {start_line}",
            )

        if category == "authentication_authorization":
            return self._vote_auth(candidate_id, route, start_line)
        return self._vote_exposure(candidate_id, route, start_line)

    @staticmethod
    def _find_route(routes: List[Mapping[str, Any]],
                    line: int) -> Mapping[str, Any] | None:
        # Prefer exact containment; tolerance matches are a fallback so a
        # cited line near (but outside) a route doesn't steal a later route
        # that actually contains it.
        for route in routes:
            start, end = _route_span(route)
            if start is not None and end is not None and start <= line <= end:
                return route
        for route in routes:
            start, end = _route_span(route)
            if start is not None and end is not None and (
                start - _ROUTE_TOLERANCE <= line <= end + _ROUTE_TOLERANCE
            ):
                return route
        return None

    def _vote_auth(self, candidate_id: str, route: Mapping[str, Any],
                   start_line: int) -> EngineVote:
        route_path = str(route.get("path", ""))
        start, _ = _route_span(route)
        line = start if start is not None else start_line
        facts = [_fact(
            "route_path", "route",
            f"route {route_path!r} covers cited line {start_line}",
            line, "", True,
        )]

        mutating = list(route.get("mutating_calls") or [])
        facts.append(_fact(
            "route_mutating" if mutating else "route_not_mutating", "route",
            f"route makes mutating calls: {mutating}" if mutating
            else "route makes no mutating calls",
            line, "", bool(mutating),
        ))
        request_inputs = list(route.get("request_input_accesses") or [])
        for index, site in enumerate(route.get("call_sites") or []):
            if not isinstance(site, Mapping):
                continue
            call = str(site.get("call", ""))
            derived_arguments = list(site.get("request_derived_arguments") or [])
            if call not in mutating or not derived_arguments:
                continue
            call_line = site.get("line")
            if not isinstance(call_line, int) or isinstance(call_line, bool):
                continue
            facts.append(_fact(
                f"request_derived_mutation_{index}",
                "source",
                f"mutating call {call} at line {call_line} receives "
                f"request-derived argument(s) {derived_arguments}; "
                f"route reads request input via {request_inputs}",
                call_line,
                "",
                True,
            ))
        if not mutating:
            return self._vote(
                candidate_id, NOT_SUPPORTED, facts,
                f"route {route_path!r} at cited line performs no state-changing calls",
            )

        guards = list(route.get("local_auth_guards") or [])
        facts.append(_fact(
            "auth_guard_present" if guards else "auth_guard_absent", "guard",
            f"route has auth guards: {guards}" if guards
            else "route has no local auth guards",
            line, "", False if guards else True,
        ))

        hooks = list(route.get("same_module_request_hooks") or [])
        enforcing = [
            hook for hook in hooks
            if isinstance(hook, Mapping) and hook.get("enforces_auth")
        ]
        if enforcing:
            handlers = [str(hook.get("handler", "?")) for hook in enforcing]
            facts.append(_fact(
                "hook_enforces_auth", "hook",
                f"module request hooks enforce auth: {handlers}",
                line, "", False,
            ))
        else:
            facts.append(_fact(
                "hook_enforces_auth_absent", "absence",
                "no same-module request hook enforces auth",
                line, "", True,
            ))

        if guards or enforcing:
            return self._vote(
                candidate_id, NOT_SUPPORTED, facts,
                "auth is enforced by a route guard or request hook",
            )
        return self._vote(
            candidate_id, SUPPORTED, facts,
            f"mutating route {route_path!r} has no auth guard or auth-enforcing hook",
        )

    def _vote_exposure(self, candidate_id: str, route: Mapping[str, Any],
                       start_line: int) -> EngineVote:
        route_path = str(route.get("path", ""))
        start, _ = _route_span(route)
        line = start if start is not None else start_line
        markers = list(route.get("sensitive_output_markers") or [])
        facts = [_fact(
            "route_path", "route",
            f"route {route_path!r} covers cited line {start_line}",
            line, "", True,
        )]
        if markers:
            facts.append(_fact(
                "sensitive_output_present", "source",
                f"sensitive output markers in route: {markers}",
                line, "", True,
            ))
            return self._vote(
                candidate_id, SUPPORTED, facts,
                f"route {route_path!r} emits sensitive markers {markers}",
            )
        facts.append(_fact(
            "sensitive_output_absent", "absence",
            "no sensitive output markers in route at cited line",
            line, "", False,
        ))
        return self._vote(
            candidate_id, NOT_SUPPORTED, facts,
            "no sensitive data markers at cited location",
        )


__all__ = ["PythonWebEngine"]
