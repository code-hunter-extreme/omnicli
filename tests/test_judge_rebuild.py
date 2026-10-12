"""Tests for the rebuilt judge stage (voters + contradiction flips) and the
stage-graph orchestration (validation, routing, stall detection).

The veto architecture is gone: engines vote, the judge votes, majority
rules, and only genuine claim-vs-fact contradiction flips a ballot.
"""

import unittest

from omnicli.redteam import (
    ENGINE_BALLOT_WEIGHT,
    JUDGE_BALLOT_WEIGHT,
    Ballot,
    aggregate_ballots,
    ballot_margin,
    build_evidence_section,
    contradiction_flip,
    engine_vote_to_ballot,
    judge_calibration_prompt,
    judge_system_prompt,
)
from omnicli.redteam_graph import (
    AGGREGATE,
    DETECT,
    JUDGE,
    PROBE,
    GraphBuildError,
    GraphPaused,
    RedTeamGraph,
    StageAbort,
    StageNode,
    StagePolicy,
    build_redteam_graph,
)


def _entry(**overrides):
    candidate = {
        "id": "candidate-1",
        "finding": "t",
        "category": "sql_injection",
        "file": "a.py",
        "start_line": 4,
        "quote": "cursor.execute(q)",
        "grounded": True,
    }
    candidate.update(overrides.pop("candidate", {}))
    entry = {
        "candidate": candidate,
        "reported_by": "p:m",
        "cross_check": {"signal": "supported", "reason": "sink"},
        "case_file": {"facts": []},
        "engine_votes": {},
        "global_verdict": {},
        "precedents": [],
    }
    entry.update(overrides)
    return entry


def _fact(fid, kind="sink", supports=True, line=4, code="cursor.execute(q)"):
    return {
        "id": fid, "kind": kind, "claim": "c", "line": line, "code": code,
        "supports_finding": supports, "engine": "python_general",
    }


class TestVoterAggregation(unittest.TestCase):
    def test_majority_uphold(self):
        ballots = [
            Ballot("e1", "UPHOLD"), Ballot("judge:s1", "UPHOLD"),
            Ballot("e2", "KILL"),
        ]
        self.assertEqual(aggregate_ballots(ballots), "UPHOLD")

    def test_majority_kill(self):
        ballots = [
            Ballot("e1", "KILL"), Ballot("judge:s1", "KILL"),
            Ballot("e2", "UPHOLD"),
        ]
        self.assertEqual(aggregate_ballots(ballots), "KILL")

    def test_tie_is_contested(self):
        ballots = [Ballot("e1", "UPHOLD"), Ballot("judge:s1", "KILL")]
        self.assertEqual(aggregate_ballots(ballots), "CONTESTED")

    def test_empty_ballot_box_is_contested(self):
        self.assertEqual(aggregate_ballots([]), "CONTESTED")

    def test_weighted_majority(self):
        ballots = [
            Ballot("memory", "KILL", weight=2.0),
            Ballot("judge:s1", "UPHOLD"),
            Ballot("e1", "UPHOLD"),
        ]
        # 2.0 KILL vs 2.0 UPHOLD -> tie -> CONTESTED
        self.assertEqual(aggregate_ballots(ballots), "CONTESTED")
        ballots.append(Ballot("e2", "KILL"))
        self.assertEqual(aggregate_ballots(ballots), "KILL")

    def test_weighted_judge_decides_against_single_engine(self):
        # The judge counts double: alone with one opposing engine it
        # carries the verdict instead of deadlocking.
        self.assertEqual(JUDGE_BALLOT_WEIGHT, 2.0 * ENGINE_BALLOT_WEIGHT)
        ballots = [
            Ballot("judge:p:m", "UPHOLD", weight=JUDGE_BALLOT_WEIGHT),
            Ballot("e1", "KILL", weight=ENGINE_BALLOT_WEIGHT),
        ]
        self.assertEqual(aggregate_ballots(ballots), "UPHOLD")
        ballots = [
            Ballot("judge:p:m", "KILL", weight=JUDGE_BALLOT_WEIGHT),
            Ballot("e1", "UPHOLD", weight=ENGINE_BALLOT_WEIGHT),
        ]
        self.assertEqual(aggregate_ballots(ballots), "KILL")

    def test_weighted_judge_deadlocks_only_against_unanimous_engines(self):
        ballots = [
            Ballot("judge:p:m", "UPHOLD", weight=JUDGE_BALLOT_WEIGHT),
            Ballot("e1", "KILL", weight=ENGINE_BALLOT_WEIGHT),
            Ballot("e2", "KILL", weight=ENGINE_BALLOT_WEIGHT),
        ]
        # 2.0 vs 2.0: the single deadlock — human review required.
        self.assertEqual(aggregate_ballots(ballots), "CONTESTED")
        # Any one engine siding with the judge breaks the deadlock.
        ballots[2] = Ballot("e2", "UPHOLD", weight=ENGINE_BALLOT_WEIGHT)
        self.assertEqual(aggregate_ballots(ballots), "UPHOLD")

    def test_engine_ballots_carry_unit_weight(self):
        ballot = engine_vote_to_ballot("e", {"vote": "supported"})
        self.assertEqual(ballot.weight, ENGINE_BALLOT_WEIGHT)
        ballot = engine_vote_to_ballot("e", {"vote": "not_supported"})
        self.assertEqual(ballot.weight, ENGINE_BALLOT_WEIGHT)

    def test_engine_vote_to_ballot(self):
        self.assertEqual(
            engine_vote_to_ballot("e", {"vote": "supported"}) .verdict, "UPHOLD")
        self.assertEqual(
            engine_vote_to_ballot("e", {"vote": "not_supported"}).verdict, "KILL")
        self.assertIsNone(engine_vote_to_ballot("e", {"vote": "not_covered"}))
        self.assertIsNone(engine_vote_to_ballot("e", {"vote": "bogus"}))

    def test_ballot_margin(self):
        self.assertAlmostEqual(
            ballot_margin([Ballot("a", "UPHOLD"), Ballot("b", "UPHOLD")]), 1.0)
        self.assertAlmostEqual(
            ballot_margin([Ballot("a", "UPHOLD"), Ballot("b", "KILL")]), 0.0)
        self.assertAlmostEqual(ballot_margin([]), 0.0)


class TestContradictionFlip(unittest.TestCase):
    def test_uphold_flipped_when_nothing_supports_it(self):
        entry = _entry(
            candidate={"grounded": False},
            cross_check={"signal": "not_supported", "reason": "no sink"},
        )
        vote, reason = contradiction_flip("UPHOLD", entry)
        self.assertEqual(vote, "KILL")
        self.assertIn("contradiction", reason)

    def test_uphold_stands_when_grounded_and_supported(self):
        entry = _entry(
            case_file={"facts": [_fact("sink_present")]},
        )
        vote, reason = contradiction_flip("UPHOLD", entry)
        self.assertEqual(vote, "UPHOLD")
        self.assertEqual(reason, "")

    def test_uphold_stands_when_grounded_despite_not_supported(self):
        # Grounded quote + no engine facts + not_supported: the judge may
        # know something the engines missed; no genuine contradiction.
        entry = _entry(
            cross_check={"signal": "not_supported", "reason": "no sink"},
        )
        vote, _ = contradiction_flip("UPHOLD", entry)
        self.assertEqual(vote, "UPHOLD")

    def test_uphold_flipped_on_category_mismatch(self):
        # Judge claims authentication_authorization but reasons about SQL
        # injection: miscategorized claim contradicts the reasoning.
        entry = _entry(
            candidate={"category": "authentication_authorization"},
            cross_check={"signal": "not_covered", "reason": "no routes"},
        )
        reasoning = (
            "The query flows into cursor.execute. No parameterization, "
            "allowing SQL injection."
        )
        vote, reason = contradiction_flip("UPHOLD", entry, reasoning)
        self.assertEqual(vote, "KILL")
        self.assertIn("contradiction", reason)

    def test_uphold_stands_when_reasoning_matches_category(self):
        entry = _entry(
            candidate={"category": "sensitive_data_exposure"},
            cross_check={"signal": "not_covered", "reason": "no routes"},
        )
        reasoning = (
            "The secret token is returned in the response, exposing it to "
            "the client."
        )
        vote, _ = contradiction_flip("UPHOLD", entry, reasoning)
        self.assertEqual(vote, "UPHOLD")

    def test_kill_flipped_when_evidence_converges(self):
        entry = _entry(
            case_file={"facts": [_fact("sink_present"), _fact("taint", kind="source")]},
        )
        vote, reason = contradiction_flip("KILL", entry)
        self.assertEqual(vote, "UPHOLD")
        self.assertIn("contradiction", reason)

    def test_kill_not_flipped_for_ungrounded_candidate(self):
        entry = _entry(
            candidate={"grounded": False},
            case_file={"facts": [_fact("sink_present"), _fact("taint", kind="source")]},
        )
        vote, reason = contradiction_flip("KILL", entry)
        self.assertEqual(vote, "KILL")
        self.assertEqual(reason, "")

    def test_kill_stands_when_evidence_disagrees(self):
        entry = _entry(
            cross_check={"signal": "not_supported", "reason": "no sink"},
        )
        vote, _ = contradiction_flip("KILL", entry)
        self.assertEqual(vote, "KILL")

    def test_kill_stands_when_contradicting_fact_exists(self):
        entry = _entry(
            case_file={"facts": [
                _fact("sink_present"),
                _fact("guard_found", kind="guard", supports=False,
                       code="guard_parameterized"),
            ]},
        )
        vote, _ = contradiction_flip("KILL", entry)
        self.assertEqual(vote, "KILL")

    def test_unparseable_never_flipped(self):
        entry = _entry()
        vote, reason = contradiction_flip("UNPARSEABLE", entry)
        self.assertEqual(vote, "UNPARSEABLE")
        self.assertEqual(reason, "")

    def test_flip_never_fires_on_format(self):
        # A judge that upholds without mentioning facts is NOT flipped for
        # format: only claim-vs-fact contradiction flips.
        entry = _entry(
            case_file={"facts": [_fact("sink_present")]},
            candidate={"grounded": True},
        )
        vote, reason = contradiction_flip("UPHOLD", entry)
        self.assertEqual(vote, "UPHOLD")
        self.assertEqual(reason, "")


class TestEvidenceSection(unittest.TestCase):
    def test_evidence_section_renders_facts(self):
        entry = _entry(
            case_file={"facts": [_fact("sink_present", code="cursor.execute(q)")]},
            engine_votes={"python_general": "supported"},
            global_verdict={"verdict": "supported", "reasoning": "r"},
        )
        section = build_evidence_section(entry)
        self.assertIn("### DETERMINISTIC EVIDENCE", section)
        self.assertIn("sink_present", section)
        self.assertIn("cursor.execute(q)", section)
        self.assertIn("python_general: supported", section)

    def test_calibration_examples_stable(self):
        first = judge_calibration_prompt()
        second = judge_calibration_prompt()
        self.assertEqual(first, second)
        self.assertIn("Example 1", first)
        self.assertIn("JUDGMENT: UPHOLD", first)
        self.assertIn("JUDGMENT: KILL", first)

    def test_system_prompt_has_no_fact_ruling_demand(self):
        system = judge_system_prompt()
        self.assertNotIn("FACT <fact-id>", system)
        self.assertNotIn("CASE-FILE RULINGS", system)
        # ... but keeps the reasoning-chain demands and evidence language.
        self.assertIn("SINK REACHABILITY", system)
        self.assertIn("GUARD ABSENCE", system)
        self.assertIn("NOT_SUPPORTED", system)
        self.assertIn("CONTRADICTION RULES", system)
        self.assertIn("verbatim", system)


def _trivial_node(name, updates=None, requires=frozenset(), produces=frozenset()):
    return StageNode(
        name=name,
        fn=lambda state, ctx: dict(updates or {}),
        requires=requires,
        produces=produces,
    )


class TestGraphValidation(unittest.TestCase):
    def test_valid_linear_graph(self):
        graph = RedTeamGraph()
        graph.add_node(_trivial_node("a", updates={"x": 1},
                                     produces=frozenset({"x"})))
        graph.add_node(_trivial_node("b", updates={"y": 2},
                                     requires=frozenset({"x"}),
                                     produces=frozenset({"y"})))
        graph.add_edge("a", "b")
        graph.set_entry("a")
        graph.add_exit("b")
        graph.validate()
        final = graph.run({})
        self.assertEqual(final["x"], 1)
        self.assertEqual(final["y"], 2)

    def test_disconnected_stage_rejected(self):
        graph = RedTeamGraph()
        graph.add_node(_trivial_node("a"))
        graph.add_node(_trivial_node("lonely"))
        graph.set_entry("a")
        graph.add_exit("a")
        with self.assertRaises(GraphBuildError):
            graph.validate()

    def test_dead_end_rejected(self):
        graph = RedTeamGraph()
        graph.add_node(_trivial_node("a"))
        graph.add_node(_trivial_node("b"))
        graph.add_edge("a", "b")
        graph.set_entry("a")
        graph.add_exit("a")
        with self.assertRaises(GraphBuildError):
            graph.validate()

    def test_missing_input_rejected(self):
        graph = RedTeamGraph()
        graph.add_node(_trivial_node("a", requires=frozenset({"nope"})))
        graph.set_entry("a")
        graph.add_exit("a")
        with self.assertRaises(GraphBuildError):
            graph.validate()

    def test_unknown_edge_target_rejected(self):
        graph = RedTeamGraph()
        graph.add_node(_trivial_node("a"))
        graph.add_edge("a", "ghost")
        graph.set_entry("a")
        graph.add_exit("a")
        with self.assertRaises(GraphBuildError):
            graph.validate()

    def test_standard_redteam_graph_validates(self):
        nodes = {
            PROBE: _trivial_node(PROBE, produces=frozenset({"reprobe_count"})),
            DETECT: _trivial_node(
                DETECT,
                produces=frozenset({"candidates", "engine_ballots",
                                    "judge_ballots", "flip_notes",
                                    "adjudications"})),
            JUDGE: _trivial_node(
                JUDGE,
                requires=frozenset({"candidates", "engine_ballots"}),
                produces=frozenset({"judgment_results", "judge_ballots",
                                    "flip_notes", "judge_panel_size",
                                    "judge_agreement_margin",
                                    "judge_round_unparseable"})),
            AGGREGATE: _trivial_node(
                AGGREGATE,
                requires=frozenset({"candidates", "engine_ballots",
                                    "judge_ballots"}),
                produces=frozenset({"verdicts"})),
        }
        graph = build_redteam_graph(nodes)
        graph.validate()  # must not raise


class TestConditionalRouting(unittest.TestCase):
    def _graph_with_judge_flag(self, candidates):
        seen = []

        def detect_fn(state, ctx):
            return {"candidates": candidates,
                    "engine_ballots": {}, "judge_ballots": {},
                    "flip_notes": {}, "adjudications": None}

        def judge_fn(state, ctx):
            seen.append("judge")
            return {"judgment_results": [], "judge_ballots": {},
                    "flip_notes": {}, "judge_panel_size": 1,
                    "judge_agreement_margin": 1.0,
                    "judge_round_unparseable": False}

        def aggregate_fn(state, ctx):
            return {"verdicts": []}

        nodes = {
            PROBE: StageNode(name=PROBE, fn=lambda s, c: {}, produces=frozenset({"reprobe_count"})),
            DETECT: StageNode(name=DETECT, fn=detect_fn,
                              produces=frozenset({"candidates", "engine_ballots",
                                                  "judge_ballots", "flip_notes",
                                                  "adjudications"})),
            JUDGE: StageNode(name=JUDGE, fn=judge_fn,
                             requires=frozenset({"candidates", "engine_ballots"}),
                             produces=frozenset({"judgment_results", "judge_ballots",
                                                 "flip_notes", "judge_panel_size",
                                                 "judge_agreement_margin",
                                                 "judge_round_unparseable"})),
            AGGREGATE: StageNode(name=AGGREGATE, fn=aggregate_fn,
                                 requires=frozenset({"candidates", "engine_ballots",
                                                     "judge_ballots"}),
                                 produces=frozenset({"verdicts"})),
        }
        return build_redteam_graph(nodes), seen

    def test_zero_candidates_skips_judge(self):
        graph, seen = self._graph_with_judge_flag([])
        final = graph.run({})
        self.assertEqual(seen, [])
        self.assertEqual(final["verdicts"], [])

    def test_candidates_reach_judge(self):
        graph, seen = self._graph_with_judge_flag([{"candidate": {"id": "c1"}}])
        graph.run({})
        self.assertEqual(seen, ["judge"])

    def test_low_margin_reprobes_once_then_aggregates(self):
        calls = []

        def judge_fn(state, ctx):
            calls.append("judge")
            return {"judgment_results": [], "judge_ballots": {},
                    "flip_notes": {}, "judge_panel_size": 1,
                    "judge_agreement_margin": 0.0,  # always low
                    "judge_round_unparseable": False}

        def probe_fn(state, ctx):
            calls.append("probe")
            # Only a re-probe (candidates already exist) consumes the budget.
            if state.get("candidates"):
                return {"reprobe_count": state.get("reprobe_count", 0) + 1}
            return {}

        nodes = {
            PROBE: StageNode(name=PROBE, fn=probe_fn,
                             produces=frozenset({"reprobe_count"})),
            DETECT: StageNode(
                name=DETECT,
                fn=lambda s, c: {"candidates": [{"candidate": {"id": "c1"}}],
                                 "engine_ballots": {}, "judge_ballots": {},
                                 "flip_notes": {}, "adjudications": None},
                produces=frozenset({"candidates", "engine_ballots",
                                    "judge_ballots", "flip_notes",
                                    "adjudications"})),
            JUDGE: StageNode(name=JUDGE, fn=judge_fn,
                             requires=frozenset({"candidates", "engine_ballots"}),
                             produces=frozenset({"judgment_results", "judge_ballots",
                                                 "flip_notes", "judge_panel_size",
                                                 "judge_agreement_margin",
                                                 "judge_round_unparseable"})),
            AGGREGATE: StageNode(name=AGGREGATE, fn=lambda s, c: {"verdicts": []},
                                 requires=frozenset({"candidates", "engine_ballots",
                                                     "judge_ballots"}),
                                 produces=frozenset({"verdicts"})),
        }
        graph = build_redteam_graph(nodes, reprobe_threshold=0.5, max_reprobes=1)
        graph.run({})
        # probe, detect, judge, probe (re-probe), detect, judge, aggregate
        self.assertEqual(calls.count("probe"), 2)
        self.assertEqual(calls.count("judge"), 2)


class TestStallDetection(unittest.TestCase):
    def _stall_nodes(self, judge_fn, max_stalls=2):
        def probe_fn(state, ctx):
            if state.get("candidates"):
                return {"reprobe_count": state.get("reprobe_count", 0) + 1}
            return {}

        return {
            PROBE: StageNode(name=PROBE, fn=probe_fn,
                             produces=frozenset({"reprobe_count"})),
            DETECT: StageNode(
                name=DETECT,
                fn=lambda s, c: {"candidates": [{"candidate": {"id": "c1"}}],
                                 "engine_ballots": {}, "judge_ballots": {},
                                 "flip_notes": {}, "adjudications": None},
                produces=frozenset({"candidates", "engine_ballots",
                                    "judge_ballots", "flip_notes",
                                    "adjudications"})),
            JUDGE: StageNode(name=JUDGE, fn=judge_fn,
                             requires=frozenset({"candidates", "engine_ballots"}),
                             produces=frozenset({"judgment_results", "judge_ballots",
                                                 "flip_notes", "judge_panel_size",
                                                 "judge_agreement_margin",
                                                 "judge_round_unparseable"}),
                             policy=StagePolicy(max_stalls=max_stalls)),
            AGGREGATE: StageNode(name=AGGREGATE, fn=lambda s, c: {"verdicts": []},
                                 requires=frozenset({"candidates", "engine_ballots",
                                                     "judge_ballots"}),
                                 produces=frozenset({"verdicts"})),
        }

    def test_consecutive_unparseable_aborts(self):
        def judge_fn(state, ctx):
            # Low margin forces a re-probe so JUDGE runs again; every round
            # is unparseable, so the streak hits max_stalls and aborts.
            return {"judgment_results": [], "judge_ballots": {},
                    "flip_notes": {}, "judge_panel_size": 1,
                    "judge_agreement_margin": 0.0,
                    "judge_round_unparseable": True}

        graph = build_redteam_graph(self._stall_nodes(judge_fn))
        with self.assertRaises(StageAbort) as cm:
            graph.run({})
        self.assertIn("stalled", str(cm.exception))

    def test_productive_round_resets_stall_streak(self):
        rounds = iter([True, False])

        def judge_fn(state, ctx):
            return {"judgment_results": [], "judge_ballots": {},
                    "flip_notes": {}, "judge_panel_size": 1,
                    "judge_agreement_margin": 0.0,
                    "judge_round_unparseable": next(rounds, False)}

        graph = build_redteam_graph(self._stall_nodes(judge_fn))
        # One unparseable round then a productive one: no abort.
        final = graph.run({})
        self.assertEqual(final["unparseable_streak"], 0)


class TestInterruptHook(unittest.TestCase):
    def test_pause_hook_raises_with_state(self):
        graph = RedTeamGraph()
        graph.add_node(_trivial_node("JUDGE", updates={"x": 1},
                                     produces=frozenset({"x"})))
        graph.set_entry("JUDGE")
        graph.add_exit("JUDGE")
        graph.on_judge_complete = lambda state: "pause"
        with self.assertRaises(GraphPaused) as cm:
            graph.run({})
        self.assertTrue(cm.exception.state["x"] == 1)

    def test_override_hook_merges(self):
        graph = RedTeamGraph()
        graph.add_node(_trivial_node("JUDGE", updates={"x": 1},
                                     produces=frozenset({"x"})))
        graph.set_entry("JUDGE")
        graph.add_exit("JUDGE")
        graph.on_judge_complete = lambda state: {"override": {"y": 2}}
        final = graph.run({})
        self.assertEqual(final["y"], 2)


class TestAuditTrail(unittest.TestCase):
    def test_stage_transitions_recorded(self):
        graph = RedTeamGraph()
        graph.add_node(_trivial_node("a", produces=frozenset({"x"})))
        graph.add_node(_trivial_node("b", requires=frozenset({"x"})))
        graph.add_edge("a", "b")
        graph.set_entry("a")
        graph.add_exit("b")
        final = graph.run({})
        trail = final["audit_trail"]
        events = [(t["stage"], t["event"]) for t in trail]
        self.assertIn(("a", "enter"), events)
        self.assertIn(("a", "exit"), events)
        self.assertIn(("b", "enter"), events)
        self.assertIn(("b", "exit"), events)
        self.assertTrue(all("at" in t and "input_hash" in t for t in trail))


if __name__ == "__main__":
    unittest.main()
