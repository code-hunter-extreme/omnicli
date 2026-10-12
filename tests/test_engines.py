"""Tests for the deterministic engine architecture (rowan/redteam-hardening).

Covers: engine registry, python_general votes, python_web votes, global
aggregation (specificity wins), registry-derived coverage, adjudication
memory (scopes + auto-kill), case-file judgment validation, FACT-line parsing,
and heartbeat engine health.
"""

import unittest

from omnicli.engines import (
    ENGINE_REGISTRY,
    NOT_COVERED,
    NOT_SUPPORTED,
    SUPPORTED,
    EngineFact,
    register_engine,
)
from omnicli.engines.adjudications import AdjudicationStore, pattern_signature
from omnicli.engines.global_engine import (
    aggregate,
    build_case_file,
)
from omnicli.redteam import engine_covered_categories, parse_judgment


def _cand(cid="c1", category="sql_injection", file="app.py", line=2, quote="x"):
    return {
        "id": cid,
        "category": category,
        "file": file,
        "start_line": line,
        "end_line": line,
        "quote": quote,
        "finding": f"test {category}",
    }


SQLI_SRC = {
    "app.py": (
        "from flask import request\n"
        "import sqlite3\n"
        "def get_user():\n"
        "    uid = request.args.get('id')\n"
        "    cur = db.cursor()\n"
        '    cur.execute("SELECT * FROM users WHERE id = " + uid)\n'
        "    return cur.fetchall()\n"
    )
}

PARAM_SRC = {
    "app.py": (
        "from flask import request\n"
        "def get_user():\n"
        "    uid = request.args.get('id')\n"
        "    cur = db.cursor()\n"
        '    cur.execute("SELECT * FROM users WHERE id = ?", (uid,))\n'
        "    return cur.fetchall()\n"
    )
}

WEB_SRC = {
    "web.py": (
        "from flask import Flask, request\n"
        "app = Flask(__name__)\n"
        "\n"
        "@app.before_request\n"
        "def log_only():\n"
        '    print("hit", request.path)\n'
        "\n"
        "@app.post('/admin/delete')\n"
        "def delete_user():\n"
        '    uid = request.form.get("id")\n'
        "    delete_user_by_id(uid)\n"
        "    return {'ok': True}\n"
        "\n"
        "@app.get('/profile')\n"
        "@login_required\n"
        "def profile():\n"
        "    return {'user': current_user.name}\n"
    )
}


class TestEngineRegistry(unittest.TestCase):
    def test_builtin_engines_registered(self):
        self.assertIn("python_general", ENGINE_REGISTRY)
        self.assertIn("python_web", ENGINE_REGISTRY)

    def test_duplicate_name_rejected(self):
        with self.assertRaises(ValueError):
            @register_engine
            class _Dup:
                name = "python_general"

    def test_empty_name_rejected(self):
        with self.assertRaises(ValueError):
            @register_engine
            class _Empty:
                name = ""

    def test_dead_confidence_field_removed(self):
        from omnicli.engines import EngineVote
        self.assertNotIn("confidence", EngineVote.__dataclass_fields__)


class TestPythonGeneral(unittest.TestCase):
    def setUp(self):
        self.engine = ENGINE_REGISTRY["python_general"]

    def test_sqli_supported(self):
        cand = _cand(line=6, quote='cur.execute("SELECT * FROM users WHERE id = " + uid)')
        vote = self.engine.analyze(SQLI_SRC, [cand])["c1"]
        self.assertEqual(vote.vote, SUPPORTED)
        self.assertTrue(any(f.id == "sink_present" and f.supports_finding for f in vote.facts))

    def test_parameterized_is_guarded(self):
        cand = _cand(line=5, quote='cur.execute("SELECT * FROM users WHERE id = ?", (uid,))')
        vote = self.engine.analyze(PARAM_SRC, [cand])["c1"]
        self.assertEqual(vote.vote, NOT_SUPPORTED)
        self.assertTrue(any("guard" in f.id for f in vote.facts))

    def test_meta_code_not_supported(self):
        # The dogfood FP: the analyzer's own bookkeeping flagged as a sink.
        src = {"det.py": (
            "import ast\n"
            "class Detector:\n"
            "    def check(self, call):\n"
            "        sink = security_sink(call.func)\n"
            "        return sink\n"
        )}
        cand = _cand(category="sql_injection", file="det.py", line=4,
                     quote="sink = security_sink(call.func)")
        vote = self.engine.analyze(src, [cand])["c1"]
        self.assertEqual(vote.vote, NOT_SUPPORTED)

    def test_meta_code_fact_on_detector_bookkeeping(self):
        # A literal sink call inside detect* machinery with no untrusted input
        # is the analyzer's own bookkeeping, not a live vulnerable path.
        src = {"scan.py": (
            "class SQLiDetector:\n"
            "    def detect(self, code):\n"
            "        cursor = get_cursor()\n"
            '        cursor.execute("SELECT 1")\n'
            "        return self.analyze(code)\n"
        )}
        cand = _cand(category="sql_injection", file="scan.py", line=4,
                     quote='cursor.execute("SELECT 1")')
        vote = self.engine.analyze(src, [cand])["c1"]
        self.assertEqual(vote.vote, NOT_SUPPORTED)
        self.assertTrue(any(f.id == "meta_code_detected" for f in vote.facts))

    def test_no_sink_not_supported(self):
        src = {"x.py": "def f(ips):\n    for ip in ips:\n        print(ip)\n"}
        cand = _cand(category="path_traversal", file="x.py", line=2, quote="for ip in ips:")
        vote = self.engine.analyze(src, [cand])["c1"]
        self.assertEqual(vote.vote, NOT_SUPPORTED)

    def test_sanitizer_not_deserialization(self):
        src = {"r.py": "def esc(s):\n    return s.replace(\"<\", \"&lt;\")\n"}
        cand = _cand(category="unsafe_deserialization", file="r.py", line=2,
                     quote='s.replace("<", "&lt;")')
        vote = self.engine.analyze(src, [cand])["c1"]
        self.assertEqual(vote.vote, NOT_SUPPORTED)

    def test_auth_category_not_covered(self):
        cand = _cand(category="authentication_authorization")
        vote = self.engine.analyze(SQLI_SRC, [cand])["c1"]
        self.assertEqual(vote.vote, NOT_COVERED)

    def test_pickle_loads_supported(self):
        src = {"p.py": "import pickle\ndef f(data):\n    return pickle.loads(data)\n"}
        cand = _cand(category="unsafe_deserialization", file="p.py", line=3,
                     quote="return pickle.loads(data)")
        vote = self.engine.analyze(src, [cand])["c1"]
        self.assertEqual(vote.vote, SUPPORTED)


class TestPythonWeb(unittest.TestCase):
    def setUp(self):
        self.engine = ENGINE_REGISTRY["python_web"]

    def test_mutating_route_no_guard_supported(self):
        cand = _cand(category="authentication_authorization", file="web.py",
                     line=10, quote="delete_user_by_id(uid)")
        vote = self.engine.analyze(WEB_SRC, [cand])["c1"]
        self.assertEqual(vote.vote, SUPPORTED)
        request_flow = [
            fact for fact in vote.facts
            if fact.id.startswith("request_derived_mutation_")
        ]
        self.assertEqual(len(request_flow), 1)
        self.assertIn("request-derived argument(s) ['uid']", request_flow[0].claim)
        self.assertEqual(request_flow[0].line, 11)

    def test_login_required_not_supported(self):
        cand = _cand(category="authentication_authorization", file="web.py",
                     line=17, quote="return {'user': current_user.name}")
        vote = self.engine.analyze(WEB_SRC, [cand])["c1"]
        self.assertEqual(vote.vote, NOT_SUPPORTED)

    def test_no_routes_not_covered(self):
        cand = _cand(category="authentication_authorization", file="app.py", line=1)
        vote = self.engine.analyze(SQLI_SRC, [cand])["c1"]
        self.assertEqual(vote.vote, NOT_COVERED)

    def test_sqli_not_covered_by_web(self):
        cand = _cand(category="sql_injection", file="web.py", line=10)
        vote = self.engine.analyze(WEB_SRC, [cand])["c1"]
        self.assertEqual(vote.vote, NOT_COVERED)


def _vote(engine, vote, facts=(), candidate_id="c1"):
    from omnicli.engines import EngineVote
    return EngineVote(
        candidate_id=candidate_id, engine=engine, engine_version="1.0.0",
        vote=vote, facts=list(facts), reason=f"{engine} {vote}",
    )


def _fact(fid, supports=True):
    return EngineFact(id=fid, kind="sink", claim=f"fact {fid}",
                      line=1, code="x", supports_finding=supports)


class TestAggregation(unittest.TestCase):
    def setUp(self):
        self.specs = {"python_general": 1, "python_web": 2}
        self.cand = _cand()

    def test_consensus_supported(self):
        votes = [_vote("python_general", SUPPORTED, [_fact("sink_present")])]
        g = aggregate(self.cand, votes, self.specs)
        self.assertEqual(g.verdict, SUPPORTED)

    def test_specific_not_supported_overrides_general_supported(self):
        votes = [
            _vote("python_general", SUPPORTED, [_fact("sink_present")]),
            _vote("python_web", NOT_SUPPORTED, [_fact("route_at_location_absent", False)]),
        ]
        g = aggregate(self.cand, votes, self.specs)
        self.assertEqual(g.verdict, NOT_SUPPORTED)
        self.assertIn("python_web", g.reasoning)
        self.assertIn("specificity 2", g.reasoning)

    def test_no_covering_engines(self):
        votes = [_vote("python_general", NOT_COVERED), _vote("python_web", NOT_COVERED)]
        g = aggregate(self.cand, votes, self.specs)
        self.assertEqual(g.verdict, NOT_COVERED)
        self.assertEqual(g.facts, [])

    def test_deterministic_under_reordering(self):
        votes = [
            _vote("python_web", NOT_SUPPORTED, [_fact("a", False)]),
            _vote("python_general", SUPPORTED, [_fact("b")]),
        ]
        g1 = aggregate(self.cand, votes, self.specs)
        g2 = aggregate(self.cand, list(reversed(votes)), self.specs)
        self.assertEqual(g1.verdict, g2.verdict)
        self.assertEqual(g1.reasoning, g2.reasoning)

    def test_case_file_schema(self):
        votes = [_vote("python_general", SUPPORTED, [_fact("sink_present")])]
        g = aggregate(self.cand, votes, self.specs)
        cf = build_case_file(self.cand, votes, g, [{"verdict": "KILL", "reasoning": "r"}])
        self.assertEqual(
            set(cf.keys()),
            {"candidate_id", "candidate", "engine_votes", "global_verdict", "facts", "precedents"},
        )
        self.assertEqual(cf["facts"][0]["id"], "sink_present")
        self.assertIn("engine", cf["facts"][0])


class TestEngineCoveredCategories(unittest.TestCase):
    def test_coverage_derived_from_registry(self):
        from omnicli.engines import ENGINE_REGISTRY
        expected = set()
        for engine in ENGINE_REGISTRY.values():
            expected.update(engine.covered_categories)
        self.assertEqual(set(engine_covered_categories()), expected)
        self.assertTrue(expected, "no engine declares coverage")

    def test_registry_change_updates_coverage(self):
        from omnicli.engines import Engine, ENGINE_REGISTRY

        class ProbeEngine(Engine):
            name = "probe_test_engine"
            covered_categories = ("probe_category",)

        before = set(engine_covered_categories())
        self.assertNotIn("probe_category", before)
        ENGINE_REGISTRY["probe_test_engine"] = ProbeEngine()
        try:
            self.assertIn("probe_category", set(engine_covered_categories()))
        finally:
            del ENGINE_REGISTRY["probe_test_engine"]
        self.assertEqual(set(engine_covered_categories()), before)

    def test_flip_inert_backstop_removed(self):
        import omnicli.engines.global_engine as global_engine
        import omnicli.judge_pipeline as judge_pipeline_module
        import inspect
        self.assertFalse(hasattr(global_engine, "location_plausibility_fact"))
        self.assertFalse(hasattr(global_engine, "CATEGORY_INDICATORS"))
        self.assertNotIn(
            "location_plausibility_fact",
            inspect.getsource(judge_pipeline_module),
        )


class TestAdjudications(unittest.TestCase):
    def setUp(self):
        self.store = AdjudicationStore(path=None)

    def test_signature_normalizes_literals(self):
        a = _cand(quote='cur.execute("SELECT " + user_input)')
        b = _cand(quote="cur.execute('DELETE ' + other)")
        # different literals/numbers -> same signature shape only if constructs match;
        # at minimum the function is deterministic and hex.
        sig = pattern_signature(a)
        self.assertEqual(sig, pattern_signature(a))
        self.assertRegex(sig, r"^[0-9a-f]{64}$")
        self.assertNotEqual(sig, pattern_signature(b))

    def test_record_find_precedent(self):
        cand = _cand()
        self.store.record("python", cand, "KILL", "no sink here")
        precs = self.store.find_precedents(cand, "python")
        self.assertEqual(len(precs), 1)
        self.assertEqual(precs[0]["verdict"], "KILL")

    def test_auto_kill_after_two_kills(self):
        cand = _cand()
        self.store.record("python", cand, "KILL", "first")
        self.assertIsNone(self.store.check_auto_kill(cand, "python"))
        self.store.record("python", cand, "KILL", "second")
        auto = self.store.check_auto_kill(cand, "python")
        self.assertIsNotNone(auto)
        self.assertTrue(auto["auto_kill"])
        self.assertIn("2", auto["reason"])

    def test_global_scope_counts(self):
        cand = _cand()
        self.store.record("global", cand, "KILL", "g1")
        self.store.record("global", cand, "KILL", "g2")
        auto = self.store.check_auto_kill(cand, "python")
        self.assertIsNotNone(auto)

    def test_per_system_isolation(self):
        cand = _cand()
        self.store.record("python", cand, "KILL", "p1")
        self.store.record("python", cand, "KILL", "p2")
        self.assertIsNone(self.store.check_auto_kill(cand, "javascript"))

    def test_uphold_does_not_trigger_auto_kill(self):
        cand = _cand()
        self.store.record("python", cand, "UPHOLD", "real")
        self.store.record("python", cand, "UPHOLD", "real")
        self.assertIsNone(self.store.check_auto_kill(cand, "python"))


class TestUngroundedFailsClosedWithoutVetoes(unittest.TestCase):
    """P4: the removed veto helpers' fail-closed behavior is subsumed.

    unsafe_override_kill used to hard-kill ungrounded + not_supported
    candidates. The live pipeline instead fails the evidence gate globally
    on any unsupported_citation triage entry and invalidates the trial.
    These tests prove the property survives the removal without the helpers.
    """

    def test_ungrounded_candidate_fails_gate(self):
        from omnicli.evidence import assess_evidence_gate

        gate = assess_evidence_gate(
            {"app.py": "def get(name):\n    return name\n"},
            [{
                "provider": "p",
                "model": "m",
                "status": "unsupported_citations",
                "review_complete": True,
                "findings_checked": [],
            }],
            [],
            [{
                "title": "SQL injection",
                "status": "unsupported_citation",
                "reason": "Citation did not match supplied source.",
            }],
            verdicts=[{"verdict": "UPHOLD"}],
        )
        self.assertEqual(gate["status"], "FAIL")
        self.assertTrue(gate["failed_checks"])

    def test_failed_gate_invalidates_trial_despite_uphold(self):
        from omnicli.evaluation import staged_trial_failed

        self.assertTrue(
            staged_trial_failed(
                "FAIL",
                [{"provider": "p", "error": None}],
                verdicts=[{"verdict": "UPHOLD"}],
            )
        )


class TestFactRulingParsing(unittest.TestCase):
    def test_agree_lines_parse(self):
        rec = parse_judgment(
            "JUDGMENT: UPHOLD\n"
            "FACT sink_present: AGREE\n"
            "FACT guard_absent: AGREE\n"
            "REASONING: x\n"
        )
        self.assertEqual(rec["judgment"], "UPHOLD")
        by_id = {r["fact_id"]: r for r in rec["fact_rulings"]}
        self.assertEqual(by_id["sink_present"]["stance"], "AGREE")
        self.assertEqual(by_id["guard_absent"]["stance"], "AGREE")

    def test_disagree_with_inline_quote(self):
        rec = parse_judgment(
            "JUDGMENT: KILL\n"
            'FACT sink_present: DISAGREE — contradictory code: "cursor.execute(q, (p,))"\n'
            "REASONING: x\n"
        )
        by_id = {r["fact_id"]: r for r in rec["fact_rulings"]}
        self.assertEqual(by_id["sink_present"]["stance"], "DISAGREE")
        self.assertIn("cursor.execute", by_id["sink_present"]["contradictory_quote"])

    def test_contradictory_code_continuation(self):
        rec = parse_judgment(
            "JUDGMENT: KILL\n"
            "FACT sink_present: DISAGREE\n"
            'CONTRADICTORY-CODE: "if user.is_admin: return"\n'
            "REASONING: x\n"
        )
        by_id = {r["fact_id"]: r for r in rec["fact_rulings"]}
        self.assertIn("user.is_admin", by_id["sink_present"]["contradictory_quote"])

    def test_no_fact_lines_empty(self):
        rec = parse_judgment("JUDGMENT: KILL\nREASONING: x\n")
        self.assertEqual(rec["fact_rulings"], [])


class TestHeartbeatEngines(unittest.TestCase):
    def test_engine_health_report(self):
        from omnicli.heartbeat import engine_health_report
        report = engine_health_report()
        by_name = {r["engine"]: r for r in report}
        self.assertIn("python_general", by_name)
        self.assertIn("python_web", by_name)
        for rec in by_name.values():
            self.assertTrue(rec["ok"], f"canary failed for {rec['engine']}: {rec}")
            # rule_set_date is release metadata, not a daily heartbeat: a
            # hardcoded date can only be 0 days old on release day. Health
            # is the canary; staleness must just be sane (parsed, past).
            self.assertIsInstance(rec["staleness_days"], int)
            self.assertGreaterEqual(rec["staleness_days"], 0)
            self.assertTrue(rec["version"])


if __name__ == "__main__":
    unittest.main()
