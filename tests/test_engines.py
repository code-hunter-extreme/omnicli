"""Tests for the deterministic engine architecture (rowan/redteam-hardening).

Covers: engine registry, python_general votes, python_web votes, global
aggregation (specificity wins), category-indicator plausibility, adjudication
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
    location_plausibility_fact,
)
from omnicli.redteam import parse_judgment, validate_judgment_against_casefile


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


class TestPlausibility(unittest.TestCase):
    def test_bare_def_line_implausible_for_auth(self):
        src = {"m.py": "def helper(x):\n    return x + 1\n"}
        cand = _cand(category="authentication_authorization", file="m.py", line=1,
                     quote="def helper(x):")
        fact = location_plausibility_fact(cand, src)
        self.assertIsNotNone(fact)
        self.assertEqual(fact.id, "no_category_indicators")
        self.assertFalse(fact.supports_finding)

    def test_request_handler_plausible_for_auth(self):
        src = {"m.py": "def delete():\n    uid = request.form['id']\n    remove(uid)\n"}
        cand = _cand(category="authentication_authorization", file="m.py", line=1,
                     quote="def delete():")
        self.assertIsNone(location_plausibility_fact(cand, src))

    def test_unknown_category_none(self):
        cand = _cand(category="weird_new_thing")
        self.assertIsNone(location_plausibility_fact(cand, {"m.py": "x = 1"}))


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


def _case_file(*facts):
    return {
        "candidate_id": "c1",
        "facts": [
            {"id": fid, "kind": "sink", "claim": f"claim {fid}", "line": 1,
             "code": "x", "supports_finding": supports, "engine": "python_general"}
            for fid, supports in facts
        ],
    }


class TestJudgmentValidation(unittest.TestCase):
    SOURCES = {"app.py": 'def get(name):\n    cursor.execute("SELECT " + name)\n'}

    def test_unaddressed_fact_fails(self):
        judgment = {"judgment": "UPHOLD", "fact_rulings": []}
        ok, reason = validate_judgment_against_casefile(
            judgment, _case_file(("sink_present", True)), self.SOURCES)
        self.assertFalse(ok)
        self.assertIn("sink_present", reason)

    def test_disagree_without_quote_fails(self):
        judgment = {"judgment": "UPHOLD", "fact_rulings": [
            {"fact_id": "sink_present", "stance": "DISAGREE", "contradictory_quote": ""}]}
        ok, reason = validate_judgment_against_casefile(
            judgment, _case_file(("sink_present", True)), self.SOURCES)
        self.assertFalse(ok)
        self.assertIn("without", reason)

    def test_disagree_with_invented_quote_fails(self):
        judgment = {"judgment": "UPHOLD", "fact_rulings": [
            {"fact_id": "sink_present", "stance": "DISAGREE",
             "contradictory_quote": "cursor.execute('SELECT 1')"}]}
        ok, _ = validate_judgment_against_casefile(
            judgment, _case_file(("sink_present", True)), self.SOURCES)
        self.assertFalse(ok)

    def test_disagree_with_verbatim_quote_passes(self):
        judgment = {"judgment": "UPHOLD", "fact_rulings": [
            {"fact_id": "sink_present", "stance": "DISAGREE",
             "contradictory_quote": 'cursor.execute("SELECT " + name)'}]}
        ok, _ = validate_judgment_against_casefile(
            judgment, _case_file(("sink_present", True)), self.SOURCES)
        self.assertTrue(ok)

    def test_full_agreement_passes(self):
        judgment = {"judgment": "UPHOLD", "fact_rulings": [
            {"fact_id": "sink_present", "stance": "AGREE", "contradictory_quote": ""},
            {"fact_id": "guard_absent", "stance": "AGREE", "contradictory_quote": ""}]}
        ok, _ = validate_judgment_against_casefile(
            judgment, _case_file(("sink_present", True), ("guard_absent", False)),
            self.SOURCES)
        self.assertTrue(ok)

    def test_empty_facts_passes(self):
        judgment = {"judgment": "UPHOLD", "fact_rulings": []}
        ok, _ = validate_judgment_against_casefile(judgment, _case_file(), self.SOURCES)
        self.assertTrue(ok)


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
