"""Council degrades gracefully under provider throttling.

One 429'd member must not zero survivors' grounded output: the gate
judges survivors on merit (errors ride as warnings), trials fail only
when zero members survive, and throttled providers shed pending calls.
"""

from omnicli.client import ProviderError
from omnicli.evaluation import staged_trial_failed
from omnicli.evidence import assess_evidence_gate
from omnicli.orchestrate import (
    _is_throttle_error,
    challenge_review_complete,
    partition_judge_failures,
    swarm_redteam,
)

SOURCE = {
    "vuln.py": 'def f(request):\n    q = request.args["q"]\n'
    '    return db.execute("SELECT * FROM t WHERE a = " + q)\n'
}
BLOCK = (
    "FINDING: SQL injection in f\n"
    "CATEGORY: sql_injection\n"
    "FILE: vuln.py\n"
    "LINE: 3\n"
    'QUOTE: `return db.execute("SELECT * FROM t WHERE a = " + q)`\n'
    "ATTACK PATH: request.args q -> concat -> db.execute\n"
    "PRECONDITIONS: caller controls q\n"
    "UNVERIFIED: live execution\n"
)


class _ProviderFake:
    def __init__(self, behaviors):
        self.behaviors = behaviors
        self.calls = []

    def chat_complete(self, provider_id, messages, model=None, **kw):
        import re as _re

        self.calls.append(provider_id)
        behavior = self.behaviors[provider_id]
        if behavior == "429":
            raise ProviderError(provider_id, 429, "Rate limit reached")
        prompt = " ".join(m.get("content", "") for m in messages)
        if "CANDIDATES ASSIGNED TO YOU" in prompt:
            ids = _re.findall(r'"candidate_id":\s*"([^"]+)"', prompt)
            if not ids:
                return {
                    "choices": [{"message": {"content": "NO CANDIDATES TO CHALLENGE"}}]
                }
            blocks = "".join(
                f"TRIAGE: SUPPORTED\nCANDIDATE: {cid}\n"
                f"REASON: sink matches source\n"
                f"EVIDENCE: vuln.py / 3\n"
                f"SAFE TEST: read the file\n"
                for cid in ids
            )
            return {"choices": [{"message": {"content": blocks}}]}
        return {"choices": [{"message": {"content": behavior}}]}


def _complete_validation():
    return {
        "status": "ok",
        "review_complete": True,
        "findings_checked": [
            {
                "category": "sql_injection",
                "file": "vuln.py",
                "start_line": 3,
                "quote": "return db.execute",
                "grounded": True,
            }
        ],
    }


def test_gate_judges_survivors_and_warns():
    gate = assess_evidence_gate(
        SOURCE, [_complete_validation()], ["slow:429 Rate limit"], []
    )
    assert gate["status"] == "PASS_GROUNDING_ONLY"
    assert len(gate["provider_warnings"]) == 1
    assert gate["failed_checks"] == []


def test_gate_fails_when_nothing_survives():
    gate = assess_evidence_gate(SOURCE, [], ["slow:429 Rate limit"], [])
    assert gate["status"] == "FAIL"
    assert gate["failed_checks"]


def test_trial_failed_needs_total_failure():
    ok_member = {"provider": "a", "error": None}
    bad_member = {"provider": "b", "error": "429"}
    assert staged_trial_failed("PASS_GROUNDING_ONLY", [ok_member, bad_member]) is False
    assert staged_trial_failed("PASS_GROUNDING_ONLY", [bad_member]) is True
    assert staged_trial_failed("FAIL", [ok_member]) is True
    assert staged_trial_failed("PASS_GROUNDING_ONLY", []) is True


def test_trial_failed_without_adjudication():
    ok_member = {"provider": "a", "error": None}
    unjudged = [{"verdict": "UNJUDGED"}, {"verdict": "UNJUDGED"}]
    upheld = [{"verdict": "UPHOLD"}, {"verdict": "UNJUDGED"}]
    killed = [{"verdict": "KILL"}]
    contested = [{"verdict": "CONTESTED"}]
    assert (
        staged_trial_failed("PASS_GROUNDING_ONLY", [ok_member], verdicts=unjudged)
        is True
    )
    assert (
        staged_trial_failed("PASS_GROUNDING_ONLY", [ok_member], verdicts=upheld)
        is False
    )
    assert (
        staged_trial_failed("PASS_GROUNDING_ONLY", [ok_member], verdicts=killed)
        is False
    )
    assert (
        staged_trial_failed("PASS_GROUNDING_ONLY", [ok_member], verdicts=contested)
        is False
    )
    assert staged_trial_failed("PASS_GROUNDING_ONLY", [ok_member], verdicts=[]) is False


def test_throttle_detector():
    assert _is_throttle_error("[GROQ] Error 429: rate limit")
    assert _is_throttle_error("quota exceeded for today")
    assert not _is_throttle_error("timeout after 30s")
    assert not _is_throttle_error(None)
    assert not _is_throttle_error("")


def test_council_survives_throttled_member():
    client = _ProviderFake({"slow": "429", "fast": BLOCK})
    review = swarm_redteam(
        None,
        client,
        "Review vuln.py for sql injection.",
        swarm=[("slow", "m1"), ("fast", "m2")],
        source_files=dict(SOURCE),
        temperature=0.0,
    )
    assert review["evidence_gate"]["status"] == "PASS_GROUNDING_ONLY"
    assert review["evidence_gate"]["provider_warnings"]
    assert any(
        f.get("grounded")
        for v in review["attack_validations"]
        for f in v.get("findings_checked", [])
    )


class _JudgeThrottleFake(_ProviderFake):
    """Attack/challenge succeed; only the supplemental judge is throttled."""

    def chat_complete(self, provider_id, messages, model=None, **kw):
        prompt = " ".join(m.get("content", "") for m in messages)
        if "red-team finding judge" in prompt:
            raise ProviderError(provider_id, 429, "Rate limit reached")
        return super().chat_complete(provider_id, messages, model=model, **kw)


def test_partition_judge_failures_marks_warnings():
    warnings = partition_judge_failures(["boom"])
    assert len(warnings) == 1
    assert "supplemental" in warnings[0]
    assert "boom" in warnings[0]
    assert partition_judge_failures([]) == []
    assert partition_judge_failures(None) == []


def test_judge_throttle_warns_without_failing_gate():
    client = _JudgeThrottleFake({"fast": BLOCK})
    review = swarm_redteam(
        None,
        client,
        "Review vuln.py for sql injection.",
        swarm=[("fast", "m2")],
        source_files=dict(SOURCE),
        temperature=0.0,
    )
    judge_calls = review["typed_judgment"]["calls"]
    assert judge_calls, "judge stage never ran; test would pass vacuously"
    assert any(c.get("error") for c in judge_calls)
    gate = review["evidence_gate"]
    assert gate["status"] == "PASS_GROUNDING_ONLY"
    assert not [f for f in gate["failed_checks"] if "judge" in f.lower()]
    assert any(
        "Judge supplemental verdict degraded" in w for w in gate["provider_warnings"]
    )


def test_challenge_review_complete_vacuous_on_empty_assignment():
    assert challenge_review_complete(set(), set()) is True
    assert challenge_review_complete([], []) is True
    assert challenge_review_complete(None, None) is True
    assert challenge_review_complete({"a", "b"}, {"a", "b"}) is True
    assert challenge_review_complete({"a", "b"}, {"a"}) is False
    assert challenge_review_complete({"a"}, set()) is False


class _CleanBillFake(_ProviderFake):
    """Canonical clean-bill attacks; challenge answers carry no magic phrase."""

    def chat_complete(self, provider_id, messages, model=None, **kw):
        prompt = " ".join(m.get("content", "") for m in messages)
        if "CANDIDATES ASSIGNED TO YOU" in prompt:
            self.calls.append(provider_id)
            return {"choices": [{"message": {"content": "I have nothing to add."}}]}
        self.calls.append(provider_id)
        return {"choices": [{"message": {"content": "NO ACTIONABLE CANDIDATE"}}]}


def _attack_validation(provider, complete=True, grounded=True):
    return {
        "provider": provider,
        "model": "m",
        "stage": "independent_attack",
        "status": "quotes_grounded" if grounded else "unsupported_citations",
        "review_complete": complete,
        "findings_checked": [
            {"grounded": grounded, "file": "vuln.py", "start_line": 3}
        ],
    }


def _challenge_validation(provider, complete=True):
    return {
        "provider": provider,
        "model": "m",
        "stage": "cross_examination",
        "status": "challenge_opinions",
        "review_complete": complete,
        "assigned_candidate_ids": ["CAND-1"] if complete else ["CAND-1"],
        "reviewed_candidate_ids": ["CAND-1"] if complete else [],
    }


def test_clean_member_carries_partial_member():
    gate = assess_evidence_gate(
        SOURCE,
        [
            _attack_validation("clean"),
            _challenge_validation("clean"),
            _attack_validation("partial"),
            _challenge_validation("partial", complete=False),
        ],
        ["partial:m (independent_attack/focus): 429 Rate limit"],
        [],
    )
    assert gate["status"] == "PASS_GROUNDING_ONLY"
    assert gate["failed_checks"] == []
    assert gate["passing_members"] == ["clean:m"]
    assert any("Non-passing member partial:m" in w for w in gate["provider_warnings"])
    assert any("did not complete" in w for w in gate["provider_warnings"])


def test_all_failing_members_still_fail():
    gate = assess_evidence_gate(
        SOURCE,
        [
            _attack_validation("a"),
            _challenge_validation("a", complete=False),
            _attack_validation("b"),
            _challenge_validation("b", complete=False),
        ],
        [],
        [],
    )
    assert gate["status"] == "FAIL"
    assert gate["failed_checks"]
    assert gate["passing_members"] == []


def test_empty_challenge_assignment_needs_no_magic_phrase():
    client = _CleanBillFake({"fast": "unused"})
    review = swarm_redteam(
        None,
        client,
        "Review vuln.py for sql injection.",
        swarm=[("fast", "m2")],
        source_files=dict(SOURCE),
        temperature=0.0,
    )
    challenges = review["challenge_validations"]
    assert challenges, "challenge stage never ran; test would pass vacuously"
    assert challenges[0]["assigned_candidate_ids"] == []
    assert challenges[0]["review_complete"] is True
    assert review["evidence_gate"]["status"] == "PASS_GROUNDING_ONLY"
