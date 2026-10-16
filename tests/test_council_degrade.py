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


def test_redteam_without_members_returns_failed_evidence_gate():
    review = swarm_redteam(
        None,
        _ProviderFake({}),
        "Review vuln.py for sql injection.",
        swarm=[],
        source_files=dict(SOURCE),
    )
    assert review["evidence_gate"]["status"] == "FAIL"
    assert any(
        "No red-team members are configured." in failure
        for failure in review["evidence_gate"]["failed_checks"]
    )


def test_minimum_families_rejects_single_family_roster_before_calls():
    client = _ProviderFake({"a": BLOCK, "b": BLOCK})
    review = swarm_redteam(
        None,
        client,
        "Review vuln.py for sql injection.",
        swarm=[("a", "openai/gpt-4o"), ("b", "openai/o3-mini")],
        source_files=dict(SOURCE),
        minimum_families=2,
    )
    assert review["evidence_gate"]["status"] == "FAIL"
    assert review["metrics"]["minimum_families"] == 2
    assert client.calls == []


def test_minimum_families_requires_two_passing_families():
    client = _ProviderFake({"slow": "429", "fast": BLOCK})
    review = swarm_redteam(
        None,
        client,
        "Review vuln.py for sql injection.",
        swarm=[("slow", "meta/llama-3.3-70b"), ("fast", "openai/gpt-4o")],
        source_files=dict(SOURCE),
        minimum_families=2,
    )
    assert review["evidence_gate"]["status"] == "FAIL"
    assert review["evidence_gate"]["selected_families"] == ["meta", "openai"]
    assert review["evidence_gate"]["passing_families"] == ["openai"]
    assert any(
        "required 2, passed 1" in failure
        for failure in review["evidence_gate"]["failed_checks"]
    )


def test_dispatcher_forwards_minimum_families_to_redteam():
    from unittest.mock import patch

    from omnicli.orchestrate import run_orchestration

    result = {"members": [], "final": "stub"}
    with patch("omnicli.orchestrate.swarm_redteam", return_value=result) as redteam:
        assert run_orchestration(
            "redteam",
            None,
            None,
            "review source",
            swarm=[("a", "openai/gpt-4o"), ("b", "meta/llama-3.3-70b")],
            source_files=dict(SOURCE),
            minimum_families=2,
        ) is result
    assert redteam.call_args.kwargs["minimum_families"] == 2


def test_dispatcher_forwards_minimum_families_to_collaboration():
    from unittest.mock import patch

    from omnicli.orchestrate import run_orchestration

    result = {"members": [], "final": "stub"}
    with patch(
        "omnicli.orchestrate.swarm_collaborate", return_value=result
    ) as collaborate:
        assert run_orchestration(
            "collaborate",
            None,
            None,
            "compare analysis",
            swarm=[("a", "openai/gpt-4o"), ("b", "meta/llama-3.3-70b")],
            minimum_families=2,
        ) is result
    assert collaborate.call_args.kwargs["minimum_families"] == 2


def test_collaboration_requires_distinct_selected_families():
    client = _ProviderFake({"a": BLOCK, "b": BLOCK})
    from omnicli.orchestrate import swarm_collaborate

    try:
        swarm_collaborate(
            None,
            client,
            "Compare the candidate.",
            synth=False,
            swarm=[("a", "openai/gpt-4o"), ("b", "openai/o3-mini")],
            minimum_families=2,
        )
    except RuntimeError as exc:
        assert "required 2, found 1" in str(exc)
    else:
        raise AssertionError("single-family collaboration unexpectedly passed")
    assert client.calls == []


def test_collaboration_requires_distinct_successful_families():
    from omnicli.orchestrate import swarm_collaborate

    class OneFamilySucceeds:
        def chat_complete(self, provider_id, **_kwargs):
            if provider_id == "slow":
                raise RuntimeError("provider unavailable")
            return {"choices": [{"message": {"content": "analysis"} }]}

    try:
        swarm_collaborate(
            None,
            OneFamilySucceeds(),
            "Compare the candidate.",
            synth=False,
            swarm=[
                ("fast", "openai/gpt-4o"),
                ("slow", "meta/llama-3.3-70b"),
            ],
            minimum_families=2,
        )
    except RuntimeError as exc:
        assert "required 2, completed 1" in str(exc)
    else:
        raise AssertionError("one-family collaboration unexpectedly passed")


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


def _unsupported_triage(title, reporters):
    return {
        "title": title,
        "file": "vuln.py",
        "start_line": 3,
        "status": "unsupported_citation",
        "reported_by": [list(pair) for pair in reporters],
    }


def test_scoped_veto_clean_member_carries_trial():
    from omnicli.evidence import assess_evidence_gate
    gate = assess_evidence_gate(
        SOURCE,
        [
            _attack_validation("clean"),
            _challenge_validation("clean"),
            _attack_validation("bad", grounded=False),
            _challenge_validation("bad"),
        ],
        [],
        [_unsupported_triage("Hallucinated quote", [("bad", "m")])],
    )
    assert gate["status"] == "PASS_GROUNDING_ONLY"
    assert gate["failed_checks"] == []
    assert gate["passing_members"] == ["clean:m"]
    assert any("Non-passing member bad:m" in w for w in gate["provider_warnings"])
    assert any("not grounded" in w for w in gate["provider_warnings"])


def test_scoped_veto_shared_bad_citation_fails_all_reporters():
    from omnicli.evidence import assess_evidence_gate
    gate = assess_evidence_gate(
        SOURCE,
        [
            _attack_validation("a", grounded=False),
            _challenge_validation("a"),
            _attack_validation("b", grounded=False),
            _challenge_validation("b"),
        ],
        [],
        [_unsupported_triage("Shared hallucination", [("a", "m"), ("b", "m")])],
    )
    assert gate["status"] == "FAIL"
    assert gate["passing_members"] == []
    assert any("not grounded" in f for f in gate["failed_checks"])


def test_scoped_veto_unknown_owner_fails_closed():
    from omnicli.evidence import assess_evidence_gate
    triage = _unsupported_triage("Orphan citation", [])
    del triage["reported_by"]
    gate = assess_evidence_gate(
        SOURCE,
        [
            _attack_validation("clean"),
            _challenge_validation("clean"),
        ],
        [],
        [triage],
    )
    assert gate["status"] == "FAIL"
    assert any("not grounded" in f for f in gate["failed_checks"])


def test_triage_records_reporters_and_merges_on_dedup():
    from omnicli.evidence import triage_grounded_findings

    def validation(provider, quote, grounded):
        return {
            "provider": provider,
            "model": "m",
            "findings_checked": [{
                "finding": "Candidate",
                "category": "sql_injection",
                "file": "vuln.py",
                "start_line": 3,
                "end_line": 3,
                "quote": quote,
                "grounded": grounded,
            }],
        }

    triaged = triage_grounded_findings(
        [
            validation("a", "same excerpt", False),
            validation("b", "same excerpt", False),
            validation("a", "other excerpt", True),
        ],
        [],
    )
    by_quote = {item["quote"]: item for item in triaged}
    assert by_quote["same excerpt"]["status"] == "unsupported_citation"
    assert sorted(tuple(pair) for pair in by_quote["same excerpt"]["reported_by"]) == [
        ("a", "m"), ("b", "m"),
    ]
    assert by_quote["other excerpt"]["reported_by"] == [["a", "m"]]


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
