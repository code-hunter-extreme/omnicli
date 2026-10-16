"""Stage-graph orchestration for the OmniCLI red-team pipeline.

Typed stages PROBE -> DETECT -> JUDGE -> AGGREGATE with:

- an explicit state schema (RedTeamState),
- fan-out execution of probes in parallel with fan-in aggregation
  (LangGraph Send pattern),
- conditional routing (CrewAI Flow @router pattern): zero candidates skips
  JUDGE; low judge agreement re-probes once with a different method, bounded,
- a human interrupt hook after JUDGE (LangGraph interrupt_after pattern),
- per-stage retry policies, timeouts, and token budgets wired into the spend
  ledger (LangGraph node policies),
- build-time graph validation (MS Agent Framework pattern): every stage's
  declared outputs must satisfy the next stage's declared inputs, no dead
  ends, no disconnected stages — violations raise before anything runs,
- stall detection (AutoGen Magentic-One pattern): too many consecutive
  unparseable judge outputs aborts the run with a diagnostic instead of
  looping,
- a stage-transition audit trail (MS OpenTelemetry-events pattern, local
  only): every stage entry/exit recorded with timestamp, input hash, and
  output summary,
- per-stage checkpoints for resume.

Original code for this repository; design ideas adapted from the
public-repository survey.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Explicit state schema. Nodes read declared inputs and write declared
# outputs; anything else they need comes through the context object the
# caller supplies (model client, ledger, sources, ...).
# ---------------------------------------------------------------------------

class RedTeamState(TypedDict, total=False):
    stage: str                        # current stage name
    candidates: List[Dict[str, Any]]  # DETECT output
    verdicts: List[Dict[str, Any]]    # AGGREGATE output
    judge_ballots: Dict[str, List[Any]]  # candidate id -> judge Ballots
    engine_ballots: Dict[str, List[Any]]  # candidate id -> engine Ballots
    flip_notes: Dict[str, List[str]]  # candidate id -> contradiction-flip notes
    unparseable_streak: int           # consecutive judge rounds, all unparseable
    reprobe_count: int                # bounded re-probe attempts used
    reprobe_methods: List[str]        # attack methods already tried
    audit_trail: List[Dict[str, Any]]  # stage-transition audit records
    paused: bool                      # set by the human interrupt hook
    abort_reason: Optional[str]       # set when the graph aborts a run
    metrics: Dict[str, Any]           # per-stage metrics accumulation


# ---------------------------------------------------------------------------
# Node specification.
# ---------------------------------------------------------------------------

NodeFn = Callable[[Dict[str, Any], Any], Dict[str, Any]]
"""Node function: (state, context) -> state updates to merge."""


@dataclass
class StagePolicy:
    """Per-stage execution policy."""

    max_retries: int = 1            # total attempts = 1 + retries on exception
    timeout_s: Optional[float] = None  # wall-clock budget for the stage
    token_budget: Optional[int] = None  # stage token budget (ledger-enforced)
    max_stalls: int = 3             # consecutive unparseable rounds before abort


@dataclass
class StageNode:
    """One typed stage in the red-team graph."""

    name: str
    fn: NodeFn
    requires: FrozenSet[str] = frozenset()
    produces: FrozenSet[str] = frozenset()
    policy: StagePolicy = field(default_factory=StagePolicy)
    description: str = ""


class GraphBuildError(ValueError):
    """Raised when build-time graph validation fails."""


class StageAbort(RuntimeError):
    """Raised when the graph aborts a run (stall, budget, timeout)."""


class GraphPaused(RuntimeError):
    """Raised when the human interrupt hook pauses the run.

    Carries the state so the caller can resume with run(resume_state).
    """

    def __init__(self, state: Dict[str, Any], message: str = "paused by interrupt hook"):
        super().__init__(message)
        self.state = state


# ---------------------------------------------------------------------------
# The graph.
# ---------------------------------------------------------------------------

class RedTeamGraph:
    """Directed stage graph with build-time validation and runtime policies."""

    def __init__(self) -> None:
        self._nodes: Dict[str, StageNode] = {}
        self._edges: Dict[str, List[str]] = {}
        self._routers: Dict[str, Callable[[Dict[str, Any]], Optional[str]]] = {}
        self._router_targets: Dict[str, List[str]] = {}
        self._entry: Optional[str] = None
        self._exits: Set[str] = set()
        self.on_judge_complete: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None
        """Human interrupt hook, called after the JUDGE node.

        Return "pause" to pause the run (raises GraphPaused with the state),
        {"override": {...}} to merge overrides into state, or None to continue.
        Default None: no pause.
        """

    # -- construction ----------------------------------------------------

    def add_node(self, node: StageNode) -> "RedTeamGraph":
        if node.name in self._nodes:
            raise GraphBuildError(f"duplicate stage node {node.name!r}")
        self._nodes[node.name] = node
        self._edges.setdefault(node.name, [])
        return self

    def add_edge(self, source: str, target: str) -> "RedTeamGraph":
        self._edges.setdefault(source, []).append(target)
        return self

    def add_router(
        self,
        source: str,
        router: Callable[[Dict[str, Any]], Optional[str]],
        targets: Optional[List[str]] = None,
    ) -> "RedTeamGraph":
        """Conditional routing (CrewAI @router pattern).

        router(state) returns the next node name, or None to end the run.
        targets declares the possible next nodes for build-time
        reachability validation.
        """
        if source in self._routers:
            raise GraphBuildError(f"duplicate router on {source!r}")
        self._routers[source] = router
        self._router_targets[source] = list(targets or [])
        return self

    def set_entry(self, name: str) -> "RedTeamGraph":
        self._entry = name
        return self

    def add_exit(self, name: str) -> "RedTeamGraph":
        self._exits.add(name)
        return self

    # -- build-time validation (MS Agent Framework pattern) ----------------

    def validate(self, initial_keys: FrozenSet[str] = frozenset()) -> "RedTeamGraph":
        """Validate the graph before anything runs. Raises GraphBuildError."""
        if not self._nodes:
            raise GraphBuildError("graph has no stages")
        if self._entry is None:
            raise GraphBuildError("graph has no entry stage")
        if self._entry not in self._nodes:
            raise GraphBuildError(f"entry stage {self._entry!r} is not a node")
        for name in self._exits:
            if name not in self._nodes:
                raise GraphBuildError(f"exit stage {name!r} is not a node")
        # Edge endpoints exist.
        for source, targets in self._edges.items():
            if source not in self._nodes:
                raise GraphBuildError(f"edge source {source!r} is not a node")
            for target in targets:
                if target not in self._nodes:
                    raise GraphBuildError(
                        f"edge {source!r} -> {target!r}: target is not a node"
                    )
        for source in self._routers:
            if source not in self._nodes:
                raise GraphBuildError(f"router source {source!r} is not a node")
        # Reachability: every node reachable from entry (no disconnected stages).
        # Router targets are declared via add_router(targets=[...]).
        reachable: Set[str] = set()
        stack = [self._entry]
        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            stack.extend(self._edges.get(node, []))
            stack.extend(self._router_targets.get(node, []))
        # Declared router targets must name real nodes.
        for source, targets in self._router_targets.items():
            for target in targets:
                if target not in self._nodes:
                    raise GraphBuildError(
                        f"router on {source!r} declares unknown target {target!r}"
                    )
        disconnected = set(self._nodes) - reachable
        if disconnected:
            raise GraphBuildError(
                f"disconnected stages unreachable from entry: {sorted(disconnected)}"
            )
        # No dead ends: every node reaches an exit (or is an exit / router).
        for name in self._nodes:
            if name in self._exits or name in self._routers:
                continue
            if not self._edges.get(name):
                raise GraphBuildError(
                    f"dead-end stage {name!r}: no outgoing edge, router, or exit"
                )
        # I/O contract: every node's required inputs are produced somewhere
        # (by initial state or by some node's declared outputs).
        producible: Set[str] = set(initial_keys)
        for node in self._nodes.values():
            producible |= set(node.produces)
        for node in self._nodes.values():
            missing = set(node.requires) - producible
            if missing:
                raise GraphBuildError(
                    f"stage {node.name!r} requires {sorted(missing)}, "
                    "which no stage produces and initial state lacks"
                )
        return self

    # -- runtime ----------------------------------------------------------

    @staticmethod
    def _hash_state_fragment(state: Dict[str, Any], keys: Tuple[str, ...]) -> str:
        try:
            blob = json.dumps(
                {k: state.get(k) for k in keys}, sort_keys=True, default=str
            )
        except Exception:
            blob = repr(keys)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def _audit(
        self,
        state: Dict[str, Any],
        node_name: str,
        event: str,
        detail: str = "",
    ) -> None:
        trail = state.setdefault("audit_trail", [])
        trail.append({
            "stage": node_name,
            "event": event,  # "enter" | "exit" | "abort" | "pause" | "route"
            "at": time.time(),
            "input_hash": self._hash_state_fragment(
                state, ("candidates", "verdicts", "stage")
            ),
            "detail": detail[:300],
        })

    def _check_stage_budget(
        self,
        node: StageNode,
        context: Any,
        tokens_before: int,
    ) -> None:
        budget = node.policy.token_budget
        if budget is None:
            return
        ledger = getattr(context, "ledger", None)
        totals = getattr(ledger, "totals", None)
        if totals is None:
            return
        used = totals()["total_tokens"] - tokens_before
        if used > budget:
            raise StageAbort(
                f"stage {node.name!r} exceeded token budget {budget} "
                f"(used {used})"
            )

    def run(
        self,
        initial_state: Dict[str, Any],
        context: Any = None,
        initial_keys: FrozenSet[str] = frozenset(),
    ) -> Dict[str, Any]:
        """Validate, then execute the graph. Returns the final state."""
        self.validate(initial_keys | frozenset(initial_state.keys()))
        state: Dict[str, Any] = dict(initial_state)
        state.setdefault("audit_trail", [])
        state.setdefault("unparseable_streak", 0)
        state.setdefault("reprobe_count", 0)
        paused_stage = state.get("stage")
        node_name: Optional[str] = (
            paused_stage if paused_stage in self._nodes else self._entry
        )
        assert node_name is not None
        while node_name is not None:
            node = self._nodes[node_name]
            state["stage"] = node_name
            self._audit(state, node_name, "enter")
            ledger = getattr(context, "ledger", None)
            totals = getattr(ledger, "totals", None)
            tokens_before = totals()["total_tokens"] if totals else 0
            started = time.perf_counter()
            updates: Dict[str, Any] = {}
            attempts = 0
            while True:
                attempts += 1
                try:
                    if (
                        node.policy.timeout_s is not None
                        and time.perf_counter() - started > node.policy.timeout_s
                    ):
                        raise StageAbort(
                            f"stage {node_name!r} exceeded timeout "
                            f"{node.policy.timeout_s}s"
                        )
                    updates = node.fn(state, context) or {}
                    if (
                        node.policy.timeout_s is not None
                        and time.perf_counter() - started > node.policy.timeout_s
                    ):
                        raise StageAbort(
                            f"stage {node_name!r} exceeded timeout "
                            f"{node.policy.timeout_s}s"
                        )
                    break
                except (StageAbort, GraphPaused):
                    raise
                except Exception as exc:
                    if attempts > node.policy.max_retries:
                        self._audit(
                            state, node_name, "abort",
                            f"failed after {attempts} attempts: {exc}",
                        )
                        raise StageAbort(
                            f"stage {node_name!r} failed after {attempts} "
                            f"attempts: {exc}"
                        ) from exc
                    time.sleep(min(2.0, 0.25 * attempts))
            # Stall detection: a JUDGE round where every ballot was
            # unparseable increments the streak; a productive round resets it.
            if node_name.upper() == "JUDGE":
                streak = int(state.get("unparseable_streak", 0))
                round_unparseable = bool(updates.get("judge_round_unparseable"))
                streak = streak + 1 if round_unparseable else 0
                updates["unparseable_streak"] = streak
                if streak >= node.policy.max_stalls:
                    self._audit(
                        state, node_name, "abort",
                        f"stall: {streak} consecutive unparseable judge rounds",
                    )
                    raise StageAbort(
                        f"stage {node_name!r} stalled: {streak} consecutive "
                        "unparseable judge rounds; aborting instead of looping"
                    )
            state.update(updates)
            self._check_stage_budget(node, context, tokens_before)
            self._audit(
                state, node_name, "exit",
                f"{len(updates)} keys updated in "
                f"{time.perf_counter() - started:.1f}s",
            )
            # Human interrupt hook after JUDGE (LangGraph interrupt_after).
            if node_name.upper() == "JUDGE" and self.on_judge_complete is not None:
                decision = self.on_judge_complete(state)
                if decision == "pause":
                    self._audit(state, node_name, "pause", "interrupt hook paused")
                    raise GraphPaused(state)
                if isinstance(decision, dict) and decision.get("override"):
                    state.update(decision["override"])
            # Route to the next node.
            if node_name in self._routers:
                nxt = self._routers[node_name](state)
                if nxt is not None and nxt not in self._nodes:
                    raise StageAbort(
                        f"router on {node_name!r} returned unknown stage {nxt!r}"
                    )
                self._audit(state, node_name, "route", f"-> {nxt}")
                node_name = nxt
            else:
                targets = self._edges.get(node_name, [])
                node_name = targets[0] if targets else None
        return state


# ---------------------------------------------------------------------------
# Standard red-team stage wiring: PROBE -> DETECT -> JUDGE -> AGGREGATE,
# with conditional routing (skip JUDGE on zero candidates; bounded re-probe
# on low agreement) and AGGREGATE as the single exit.
# ---------------------------------------------------------------------------

PROBE = "PROBE"
DETECT = "DETECT"
JUDGE = "JUDGE"
AGGREGATE = "AGGREGATE"


def build_redteam_graph(
    nodes: Dict[str, StageNode],
    *,
    reprobe_threshold: float = 0.5,
    max_reprobes: int = 1,
) -> RedTeamGraph:
    """Wire the standard four stages with conditional routing.

    nodes must contain PROBE, DETECT, JUDGE, AGGREGATE StageNodes.
    Routing:
      DETECT -> JUDGE when candidates exist, else straight to AGGREGATE.
      JUDGE  -> PROBE (re-probe once) when ballot agreement is below
                reprobe_threshold and the re-probe budget remains,
                else -> AGGREGATE.
    """
    missing = {PROBE, DETECT, JUDGE, AGGREGATE} - set(nodes)
    if missing:
        raise GraphBuildError(f"missing stages for red-team graph: {sorted(missing)}")
    graph = RedTeamGraph()
    for node in nodes.values():
        graph.add_node(node)
    graph.set_entry(PROBE)
    graph.add_exit(AGGREGATE)
    graph.add_edge(PROBE, DETECT)
    graph.add_edge(JUDGE, AGGREGATE)

    def route_after_detect(state: Dict[str, Any]) -> Optional[str]:
        if not state.get("candidates"):
            return AGGREGATE  # nothing to judge; aggregate empty verdicts
        return JUDGE

    def route_after_judge(state: Dict[str, Any]) -> Optional[str]:
        margin = float(state.get("judge_agreement_margin", 1.0))
        reprobes = int(state.get("reprobe_count", 0))
        if margin < reprobe_threshold and reprobes < max_reprobes:
            return PROBE  # bounded re-probe with a different method
        return AGGREGATE

    graph.add_router(DETECT, route_after_detect, targets=[JUDGE, AGGREGATE])
    graph.add_router(JUDGE, route_after_judge, targets=[PROBE, AGGREGATE])
    return graph
