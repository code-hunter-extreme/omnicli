"""P4 rulings: real council verdict, Transition-Gate anatomy, dead-path resolution.

(a) The evidence gate must produce an assessed security conclusion from the
judged verdicts — NOT_ASSESSED is allowed only when zero verdicts exist.
(b) The Transition-Gate qualifying-run anatomy is pinned by a validator, and
the artifact test below is SKIPPED (marked human_gate) until a genuine
qualifying run exists, so CI stays able to signal real regressions.
(c) Removed dead paths stay removed; the deliberate+checkpoint refusal stays
a tested fail-fast.
"""

import inspect
import json
from pathlib import Path

import pytest

import omnicli.redteam as redteam_module
from omnicli.evaluation import assess_redteam_quality_gate, validate_qualifying_run
from omnicli.evidence import assess_evidence_gate

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "reports" / "redteam_qualifying_run.json"

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
JUDGE_UPHOLD = (
    "JUDGMENT: UPHOLD\n"
    "SINK-REACHABLE: yes\n"
    "GUARD-ABSENT: yes\n"
    "REASONING: the sql query string concatenates attacker q into the "
    'statement passed to `return db.execute("SELECT * FROM t WHERE a = " + q)` '
    "with no parameterization."
)


def _passing_validation():
    return {
        "provider": "fast",
        "model": "m2",
        "status": "quotes_grounded",
        "review_complete": True,
        "explicit_no_findings": False,
        "findings_checked": [
            {
                "category": "sql_injection",
                "file": "vuln.py",
                "start_line": 3,
                "end_line": 3,
                "quote": 'return db.execute("SELECT * FROM t WHERE a = " + q)',
                "grounded": True,
            }
        ],
    }


def _verdict(verdict, **overrides):
    record = {
        "verdict": verdict,
        "candidate": {
            "id": "candidate-1",
            "category": "sql_injection",
            "file": "vuln.py",
            "start_line": 3,
            "quote": 'return db.execute("SELECT * FROM t WHERE a = " + q)',
            "grounded": True,
        },
        "provider": "fast",
        "model": "m2",
        "provider_error": None,
        "margin": 1.0,
    }
    record.update(overrides)
    return record


# ── (a) assessed council verdict ──────────────────────────────────────────


def test_gate_uphold_yields_assessed_positive_with_cited_evidence():
    gate = assess_evidence_gate(
        SOURCE, [_passing_validation()], [], [], verdicts=[_verdict("UPHOLD")]
    )
    assert gate["status"] == "PASS_GROUNDING_ONLY"
    assert gate["security_conclusion"] == "UPHELD_FINDINGS"
    assert gate["verdict_counts"] == {
        "uphold": 1,
        "kill": 0,
        "contested": 0,
        "unjudged": 0,
    }
    (entry,) = gate["security_evidence"]
    assert entry["verdict"] == "UPHOLD"
    assert entry["category"] == "sql_injection"
    assert entry["file"] == "vuln.py"
    assert entry["line"] == 3
    assert "db.execute" in entry["quote"]
    assert entry["grounded"] is True
    assert entry["judge"] == "fast:m2"
    assert "not exploitability proof" in gate["meaning"]


def test_gate_all_kill_yields_assessed_negative():
    gate = assess_evidence_gate(
        SOURCE,
        [_passing_validation()],
        [],
        [],
        verdicts=[_verdict("KILL"), _verdict("KILL")],
    )
    assert gate["security_conclusion"] == "NO_UPHELD_FINDINGS"
    assert gate["security_evidence"] == []
    assert gate["verdict_counts"]["kill"] == 2
    assert "upheld none" in gate["meaning"]
    assert "not proof of safety" in gate["meaning"]


def test_gate_contested_is_explicit():
    gate = assess_evidence_gate(
        SOURCE,
        [_passing_validation()],
        [],
        [],
        verdicts=[_verdict("KILL"), _verdict("CONTESTED")],
    )
    assert gate["security_conclusion"] == "CONTESTED"
    assert gate["verdict_counts"]["contested"] == 1
    (entry,) = gate["security_evidence"]
    assert entry["verdict"] == "CONTESTED"
    assert entry["file"] == "vuln.py"
    assert "no clean bill is given" in gate["meaning"]


def test_gate_unjudged_is_explicit():
    gate = assess_evidence_gate(
        SOURCE,
        [_passing_validation()],
        [],
        [],
        verdicts=[_verdict("UNJUDGED", provider_error="429 Rate limit")],
    )
    assert gate["security_conclusion"] == "UNJUDGED"
    assert gate["verdict_counts"]["unjudged"] == 1
    (entry,) = gate["security_evidence"]
    assert entry["provider_error"] == "429 Rate limit"
    assert "no clean bill is given" in gate["meaning"]


def test_gate_without_verdicts_stays_not_assessed():
    for verdicts in (None, []):
        gate = assess_evidence_gate(
            SOURCE, [_passing_validation()], [], [], verdicts=verdicts
        )
        assert gate["security_conclusion"] == "NOT_ASSESSED"
        assert gate["security_evidence"] == []
        assert gate["verdict_counts"] == {
            "uphold": 0,
            "kill": 0,
            "contested": 0,
            "unjudged": 0,
        }
        assert "does not prove" in gate["meaning"]


def test_gate_fail_with_verdicts_assesses_but_marks_trial_invalid():
    gate = assess_evidence_gate(
        SOURCE,
        [_passing_validation()],
        [],
        [
            {
                "title": "bad quote",
                "status": "unsupported_citation",
                "reason": "Citation did not match supplied source.",
            }
        ],
        verdicts=[_verdict("UPHOLD")],
    )
    assert gate["status"] == "FAIL"
    assert gate["security_conclusion"] == "UPHELD_FINDINGS"
    assert "invalid for scoring" in gate["meaning"]


def test_judged_paths_never_emit_not_assessed():
    combos = [
        ["UPHOLD"],
        ["KILL"],
        ["CONTESTED"],
        ["UNJUDGED"],
        ["uphold"],
        ["kill", "CONTESTED"],
        ["BOGUS"],
        [{"not_a_verdict": True}],
    ]
    for combo in combos:
        verdicts = [_verdict(v) if isinstance(v, str) else v for v in combo]
        gate = assess_evidence_gate(
            SOURCE, [_passing_validation()], [], [], verdicts=verdicts
        )
        assert gate["security_conclusion"] != "NOT_ASSESSED", combo


class _E2EClient:
    def chat_complete(self, provider_id, messages, model=None, **kw):
        import re as _re

        prompt = " ".join(m.get("content", "") for m in messages)
        if "red-team finding judge" in prompt:
            return {"choices": [{"message": {"content": JUDGE_UPHOLD}}]}
        if "CANDIDATES ASSIGNED TO YOU" in prompt:
            ids = _re.findall(r'"candidate_id":\s*"([^"]+)"', prompt)
            if not ids:
                return {
                    "choices": [{"message": {"content": "NO CANDIDATES TO CHALLENGE"}}]
                }
            return {
                "choices": [
                    {
                        "message": {
                            "content": "".join(
                                f"TRIAGE: SUPPORTED\nCANDIDATE: {cid}\n"
                                f"REASON: sink matches source\n"
                                f"EVIDENCE: vuln.py / 3\n"
                                f"SAFE TEST: read the file\n"
                                for cid in ids
                            )
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"content": BLOCK}}]}


def test_swarm_redteam_end_to_end_conclusion_matches_verdicts():
    from omnicli.orchestrate import swarm_redteam

    review = swarm_redteam(
        None,
        _E2EClient(),
        "Review vuln.py for sql injection.",
        swarm=[("fast", "m2")],
        source_files=dict(SOURCE),
        temperature=0.0,
    )
    gate = review["evidence_gate"]
    verdicts = review["typed_judgment"]["verdicts"]
    assert verdicts, "judge stage never ran; test would pass vacuously"
    assert gate["security_conclusion"] != "NOT_ASSESSED"
    counts = gate["verdict_counts"]
    assert sum(counts.values()) == len(verdicts)
    upheld = [v for v in verdicts if v["verdict"] == "UPHOLD"]
    if gate["security_conclusion"] == "UPHELD_FINDINGS":
        assert upheld
        assert len(gate["security_evidence"]) == len(upheld)
        for entry in gate["security_evidence"]:
            assert entry["file"] == "vuln.py"
            assert entry["line"] == 3
            assert "db.execute" in entry["quote"]
    assert gate["security_conclusion"] == "UPHELD_FINDINGS"


def test_report_renders_council_verdict_with_citations(tmp_path, monkeypatch):
    from omnicli.orchestrate import generate_redteam_report

    monkeypatch.chdir(tmp_path)
    report = generate_redteam_report(
        task="audit vuln.py",
        final_findings="triage text",
        members=[{"family": "meta", "provider": "fast", "model": "m2"}],
        evidence_gate=assess_evidence_gate(
            SOURCE, [_passing_validation()], [], [], verdicts=[_verdict("UPHOLD")]
        ),
    )
    content = report.read_text(encoding="utf-8")
    assert "UPHELD_FINDINGS" in content
    assert "Upheld 1 / killed 0 / contested 0 / unjudged 0" in content
    assert "vuln.py" in content and "db.execute" in content


# ── (b) Transition-Gate qualifying-run anatomy ────────────────────────────

_ROSTER = [
    ("cloudflare", "meta-llama/llama-3.3-70b-instruct", "meta"),
    ("xkiro", "google/gemini-2.5-flash", "google"),
    ("apinex", "deepseek/deepseek-chat-v3-0324", "deepseek"),
]


def _row(provider, model, method, tp, fn, tn, clean_fp, grounded, citations):
    return {
        "provider": provider,
        "model": model,
        "method": method,
        "cases": 120,
        "repetitions": 2,
        "errors": 0,
        "evidence_gate_failures": 0,
        "true_positives": tp,
        "false_positives": 0,
        "false_negatives": fn,
        "true_negative_cases": tn,
        "false_positive_clean_cases": clean_fp,
        "grounded_citations": grounded,
        "citations": citations,
        "per_category": {
            "sql_injection": {
                "true_positives": tp,
                "false_positives": 0,
                "false_negatives": fn,
            }
        },
    }


def _genuine_pass_report():
    cases = []
    for i in range(60):
        cases.append(
            {
                "id": f"hc-{i:03d}",
                "split": "held_out",
                "adjudication_status": "independently_reviewed",
                "reviewer_count": 2,
                "expected_findings": (
                    [{"category": "sql_injection"}] if i < 30 else []
                ),
            }
        )
    direct, staged = [], []
    for provider, model, _family in _ROSTER:
        direct.append(
            _row(provider, model, "single_pass_model", 56, 4, 58, 2, 114, 118)
        )
        staged.append(_row(provider, model, "staged_redteam", 57, 3, 59, 1, 116, 118))
    gate = assess_redteam_quality_gate(cases, direct, staged)
    assert gate["status"] == "PASS_BOUNDED_GATE", gate["failures"]
    return {
        "models": direct + staged,
        "staged_council": {
            "method": "staged_redteam_council",
            "models": [{"provider": p, "model": m} for p, m, _f in _ROSTER],
            "member_calls": 360,
            "errors": 0,
        },
        "quality_gate": gate,
    }


def test_validator_accepts_genuine_pass_round_trip():
    result = validate_qualifying_run(_genuine_pass_report())
    assert result["valid"], result["failures"]
    assert result["failures"] == []
    assert result["summary"]["families"] == ["deepseek", "google", "meta"]
    assert result["summary"]["paired_models"] == 3


def test_validator_rejects_edited_or_partial_artifacts():
    base = _genuine_pass_report()

    def check(mutator, needle):
        import copy

        report = copy.deepcopy(base)
        mutator(report)
        result = validate_qualifying_run(report)
        assert result["valid"] is False
        assert any(needle in f for f in result["failures"]), result["failures"]

    check(
        lambda r: r["quality_gate"].update(status="FAIL"),
        "not PASS_BOUNDED_GATE",
    )
    check(
        lambda r: r["quality_gate"].update(failures=["edited"]),
        "failures is not empty",
    )
    check(
        lambda r: r["quality_gate"].update(positive_cases=12),
        "positive",
    )
    check(
        lambda r: r["quality_gate"].update(
            paired_models=r["quality_gate"]["paired_models"][:2],
            healthy_model_families=["meta", "google"],
        ),
        "distinct model families",
    )
    check(
        lambda r: r["quality_gate"].update(healthy_model_families=["meta"]),
        "does not match",
    )

    def drop_staged_rows(report):
        report["models"] = [
            m for m in report["models"] if m["method"] != "staged_redteam"
        ]

    check(drop_staged_rows, "missing a paired single-pass or staged row")
    check(
        lambda r: r["models"][0].update(repetitions=1),
        "fewer than 2 repetitions",
    )
    check(
        lambda r: r["models"][0].update(errors=1),
        "provider errors",
    )
    check(
        lambda r: r["quality_gate"]["staged"].update(recall=0.10),
        "staged pooled recall is below 85%",
    )

    def force_regression(report):
        report["quality_gate"]["staged"]["recall"] = 0.90
        report["quality_gate"]["single_pass"]["recall"] = 0.99

    check(force_regression, "regresses against the single-pass baseline")
    check(
        lambda r: r["quality_gate"].update(limitations=[]),
        "limitations",
    )
    check(lambda r: r.pop("staged_council"), "staged_council")
    assert validate_qualifying_run({"nope": True})["valid"] is False
    assert validate_qualifying_run([])["valid"] is False  # type: ignore[arg-type]


_HUMAN_STEPS = """\
No qualifying Transition-Gate run exists yet (expected artifact: {artifact}).
What remains for a human reviewer (see docs/TRANSITION_GATE_STATUS.md):
1. A second independent reviewer completes
   reports/redteam_blinded_review_packet_20261007.json following
   reports/redteam_review_packet_guide_20261006.md (blind labels, no peeking).
2. Compare both label sets, resolve disagreements in writing, then record
   adjudication_status=independently_reviewed with reviewer_count=2 on every
   held-out case (tests/fixtures/redteam/cases.held_out.provisional.jsonl).
3. Run the full qualifying command (3 model families x 2 repetitions):
   omni evaluate-security --dataset tests/fixtures/redteam/cases.held_out.provisional.jsonl \\
     --split held_out --model cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast \\
     --model xkiro:google/gemini-2.5-flash:free --model apinex:free/deepseek-v4.1-flash \\
     --repetitions 2 --staged-redteam --quality-gate \\
     --output reports/redteam_qualifying_run.json
4. Re-run this test: it passes only on a genuine PASS_BOUNDED_GATE artifact.
"""


@pytest.mark.human_gate
def test_qualifying_run_artifact_exists_and_validates():
    if not ARTIFACT.is_file():
        pytest.skip(_HUMAN_STEPS.format(artifact=ARTIFACT))
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    result = validate_qualifying_run(report)
    assert result["valid"], (
        "reports/redteam_qualifying_run.json exists but does not validate:\n"
        + "\n".join(f"- {f}" for f in result["failures"])
        + "\n"
        + _HUMAN_STEPS.format(artifact=ARTIFACT)
    )


# ── (c) dead-path resolution ──────────────────────────────────────────────


def test_contract_helpers_removed():
    for name in (
        "RedTeamContract",
        "RedTeamContractError",
        "require_contract",
        "evaluation_contract",
        "local_analysis_contract",
    ):
        assert not hasattr(redteam_module, name), name


def test_veto_helpers_removed():
    for name in (
        "unsafe_override_kill",
        "ungrounded_not_covered_cap",
        "decide_council_verdict",
        "validate_judgment_against_casefile",
        "_quote_in_sources",
    ):
        assert not hasattr(redteam_module, name), name


def test_staged_runs_take_no_contract_and_stay_offline():
    from omnicli.orchestrate import swarm_redteam

    assert "contract" not in inspect.signature(swarm_redteam).parameters
    calls = []

    class RecordingClient(_E2EClient):
        def chat_complete(self, provider_id, messages, model=None, **kw):
            calls.append((provider_id, model))
            return super().chat_complete(provider_id, messages, model=model, **kw)

    review = swarm_redteam(
        None,
        RecordingClient(),
        "Review vuln.py for sql injection.",
        swarm=[("fast", "m2")],
        source_files=dict(SOURCE),
        temperature=0.0,
    )
    assert calls, "staged run made no model calls; test would pass vacuously"
    assert review["members"]
    assert review["evidence_gate"]["status"] == "PASS_GROUNDING_ONLY"


def test_deliberate_checkpoint_refusal_creates_no_file(tmp_path):
    from omnicli.orchestrate import run_orchestration
    from unittest.mock import MagicMock

    checkpoint = tmp_path / "deliberate.sqlite3"
    try:
        run_orchestration(
            "deliberate",
            MagicMock(),
            MagicMock(),
            "task",
            checkpoint_path=str(checkpoint),
        )
    except ValueError as exc:
        assert "not yet supported" in str(exc)
    else:  # pragma: no cover - refusal must raise
        raise AssertionError("deliberate+checkpoint refusal did not raise")
    assert not checkpoint.exists()
