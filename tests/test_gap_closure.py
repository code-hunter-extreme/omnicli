"""Regression tests for the orchestration and red-team gap-closure pass."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from omnicli.evidence import candidate_reference_id, triage_grounded_findings
from omnicli.judge_pipeline import run_candidate_judge_graph
from omnicli.orchestrate import (
    build_swarm,
    merge_claims,
    run_orchestration,
    swarm_redteam,
)
from omnicli.redteam import contradiction_flip, parse_judgment
from omnicli.tools import scan_directory_sources


class _RecordingClient:
    """Offline chat client that records prompts and returns canned answers."""

    def __init__(self, answer: str = "NO ACTIONABLE CANDIDATE"):
        self.answer = answer
        self.calls: List[Dict[str, Any]] = []

    def chat_complete(self, provider_id, messages, model=None, **kwargs):
        self.calls.append({
            "provider": provider_id,
            "model": model,
            "messages": messages,
        })
        return {"choices": [{"message": {"content": self.answer}}]}


class _TwoProviderClient:
    """Report the same candidate from two members, then triage it."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def chat_complete(self, provider_id, messages, model=None, **kwargs):
        prompt = messages[-1]["content"]
        system = messages[0]["content"]
        self.calls.append({
            "provider": provider_id,
            "model": model,
            "messages": messages,
            "role": (
                "challenge" if "CANDIDATES ASSIGNED TO YOU" in prompt
                else "judge" if "red-team finding judge" in system
                else "attack"
            ),
        })
        if "CANDIDATES ASSIGNED TO YOU" in prompt:
            match = re.search(r"\bCAND-[A-F0-9]{12}\b", prompt)
            if not match:
                answer = "NO CANDIDATES TO CHALLENGE"
            else:
                answer = (
                    "TRIAGE: UNRESOLVED\n"
                    f"CANDIDATE: {match.group(0)}\n"
                    "REASON: The route and helper need runtime context.\n"
                    "SAFE TEST: Inspect middleware and route registration."
                )
        elif "red-team finding judge" in system:
            answer = (
                "JUDGMENT: KILL\n"
                "SINK-REACHABLE: unverifiable\n"
                "GUARD-ABSENT: unverifiable\n"
                "REASONING: The source needs external context.\n"
            )
        else:
            answer = (
                "FINDING: Missing authorization on deletion\n"
                "CATEGORY: authentication_authorization\n"
                "FILE: app.py\nLINE: 5\n"
                "QUOTE: `delete_user_by_id(1)`\n"
                "ATTACK PATH: An unauthenticated request reaches deletion.\n"
                "PRECONDITIONS: The route is reachable without authentication.\n"
                "UNVERIFIED: Check middleware and the helper."
            )
        return {"choices": [{"message": {"content": answer}}]}


def _judge_candidate(**overrides):
    candidate = {
        "candidate_id": "CAND-123456789ABC",
        "title": "Unsupported test-category behavior",
        "category": "unknown_test_category",
        "file": "app.py",
        "start_line": 2,
        "end_line": 2,
        "quote": "dangerous_call()",
        "quote_grounded": True,
        "status": "unverified_candidate",
    }
    candidate.update(overrides)
    return candidate


def _flip_entry(**overrides):
    candidate = {
        "grounded": True,
        "start_line": 4,
        "category": "sql_injection",
    }
    candidate.update(overrides.pop("candidate", {}))
    entry = {
        "candidate": candidate,
        "reported_by": [["p", "m"]],
        "cross_check": {"signal": "supported", "reason": "sink"},
        "case_file": {
            "facts": [
                {
                    "id": "sink_present",
                    "kind": "sink",
                    "claim": "sink",
                    "line": 4,
                    "code": "cursor.execute(q)",
                    "supports_finding": True,
                    "engine": "python_general",
                },
                {
                    "id": "taint",
                    "kind": "source",
                    "claim": "taint",
                    "line": 4,
                    "code": "q = request.args['q']",
                    "supports_finding": True,
                    "engine": "python_general",
                },
            ]
        },
        "engine_votes": {},
        "global_verdict": {},
        "precedents": [],
    }
    entry.update(overrides)
    return entry


class TestSourceInjectionOnce(unittest.TestCase):
    def test_redteam_goal_does_not_duplicate_explicit_source(self):
        sentinel = "SOURCE_SENTINEL_DO_NOT_DUPLICATE"
        client = _RecordingClient()
        result = run_orchestration(
            "redteam",
            object(),
            client,
            "review the supplied source",
            synth=False,
            swarm=[("p", "meta/llama-3.3-70b")],
            source_files={"app.py": f"{sentinel} = 1\n"},
        )
        self.assertEqual(result["members"][0]["error"], None)
        self.assertEqual(len(client.calls), 5)  # 4 attack passes + 1 challenge
        for call in client.calls:
            user = call["messages"][-1]["content"]
            self.assertEqual(
                user.count(sentinel), 1,
                "source content must appear exactly once in every stage prompt",
            )
            self.assertNotIn("### Source file:", user)

    def test_redteam_strips_source_blocks_embedded_in_the_task_prompt(self):
        sentinel = "PROMPT_SOURCE_SENTINEL_ONCE"
        prompt = (
            "review the supplied source\n\n"
            "### Source file: app.py\n"
            f"```text\n{sentinel} = 1\n```\n"
        )
        client = _RecordingClient()
        run_orchestration(
            "redteam",
            object(),
            client,
            prompt,
            synth=False,
            swarm=[("p", "meta/llama-3.3-70b")],
        )
        self.assertEqual(len(client.calls), 5)
        for call in client.calls:
            user = call["messages"][-1]["content"]
            self.assertEqual(user.count(sentinel), 1)
            self.assertNotIn("### Source file:", user)

    def test_checkpoint_detects_changed_redteam_source(self):
        with tempfile.TemporaryDirectory() as td:
            checkpoint = str(Path(td) / "run.sqlite3")
            first = _RecordingClient()
            run_orchestration(
                "redteam",
                object(),
                first,
                "review the supplied source",
                synth=False,
                checkpoint_path=checkpoint,
                swarm=[("p", "meta/llama-3.3-70b")],
                source_files={"app.py": "FIRST_SOURCE = 1\n"},
            )
            with self.assertRaisesRegex(ValueError, "must match"):
                run_orchestration(
                    "redteam",
                    object(),
                    _RecordingClient(),
                    "review the supplied source",
                    synth=False,
                    checkpoint_path=checkpoint,
                    resume=True,
                    swarm=[("p", "meta/llama-3.3-70b")],
                    source_files={"app.py": "SECOND_SOURCE = 2\n"},
                )


class TestJudgeParsingAndContext(unittest.TestCase):
    def test_multiline_reasoning_is_not_truncated(self):
        parsed = parse_judgment(
            "JUDGMENT: KILL\n"
            "SINK-REACHABLE: no\n"
            "GUARD-ABSENT: no\n"
            "REASONING: the helper is parameterized,\n"
            "so the quoted interpolation is not injectable.\n"
            "FACT sink_present: AGREE\n"
            "CONTRADICTORY-CODE: \"cursor.execute(q, (p,))\"\n"
        )
        self.assertEqual(parsed["judgment"], "KILL")
        self.assertEqual(parsed["sink_reachable"], "no")
        self.assertIn("parameterized", parsed["reasoning"])
        self.assertIn("not injectable", parsed["reasoning"])
        self.assertNotIn("FACT", parsed["reasoning"])
        self.assertNotIn("CONTRADICTORY", parsed["reasoning"])

    def test_judge_receives_full_source_not_a_short_excerpt(self):
        source = "line 1\nline 2 dangerous_call()\n" + "\n".join(
            f"line {index}" for index in range(3, 41)
        )
        prompts: List[str] = []

        def model_call(provider, model, messages, max_tokens):
            prompts.append(messages[-1]["content"])
            return {
                "provider": provider,
                "model": model,
                "answer": "JUDGMENT: UPHOLD\nREASONING: code supports it.",
                "error": None,
            }

        result = run_candidate_judge_graph(
            [_judge_candidate()],
            {"app.py": source},
            [],
            [("p", "model-x")],
            model_call,
            256,
        )

        self.assertEqual(result["verdicts"][0]["verdict"], "UPHOLD")
        self.assertEqual(len(prompts), 1)
        self.assertIn("line 1", prompts[0])
        self.assertIn("line 40", prompts[0])
        self.assertIn("AUTHORITATIVE SOURCE CONTEXT", prompts[0])

    def test_judge_prefers_a_non_author_when_designated_judge_is_author(self):
        calls: List[tuple] = []

        def model_call(provider, model, messages, max_tokens):
            calls.append((provider, model))
            return {
                "provider": provider,
                "model": model,
                "answer": "JUDGMENT: UPHOLD\nREASONING: code supports it.",
                "error": None,
            }

        result = run_candidate_judge_graph(
            [_judge_candidate(reported_by=[["p1", "author"]] )],
            {"app.py": "def endpoint():\n    dangerous_call()\n"},
            [],
            [("p1", "author"), ("p2", "independent")],
            model_call,
            256,
            designated_judge=("p1", "author"),
        )

        self.assertEqual(calls, [("p2", "independent")])
        verdict = result["verdicts"][0]
        self.assertEqual(verdict["judge_independence"], "independent")
        self.assertFalse(any("authored this candidate" in n for n in verdict["flip_notes"]))

    def test_judge_falls_back_to_author_only_when_no_alternative_exists(self):
        def model_call(provider, model, messages, max_tokens):
            return {
                "provider": provider,
                "model": model,
                "answer": "JUDGMENT: KILL\nREASONING: code is safe.",
                "error": None,
            }

        result = run_candidate_judge_graph(
            [_judge_candidate(reported_by=[["p1", "author"]])],
            {"app.py": "def endpoint():\n    dangerous_call()\n"},
            [],
            [("p1", "author")],
            model_call,
            256,
            designated_judge=("p1", "author"),
        )
        verdict = result["verdicts"][0]
        self.assertEqual(verdict["judge_independence"], "author")
        self.assertTrue(any("authored this candidate" in n for n in verdict["flip_notes"]))


class TestContradictionAndIndependence(unittest.TestCase):
    def test_kill_is_preserved_when_judge_names_a_mitigating_control(self):
        vote, reason = contradiction_flip(
            "KILL",
            _flip_entry(),
            "The query uses a parameterized placeholder and a tuple, so no injection.",
        )
        self.assertEqual(vote, "KILL")
        self.assertIn("withheld", reason)
        self.assertIn("control", reason)

    def test_kill_still_flips_when_judge_gives_no_mitigation(self):
        vote, reason = contradiction_flip("KILL", _flip_entry(), "")
        self.assertEqual(vote, "UPHOLD")
        self.assertIn("converging evidence", reason)

    def test_kill_still_flips_when_judge_says_the_control_is_absent(self):
        for reasoning in (
            "The endpoint has no validation and no guard.",
            "The query is not parameterized; the guard is missing.",
            "The input is not sanitized.",
            "The route is not unreachable and has no authentication.",
        ):
            with self.subTest(reasoning=reasoning):
                vote, reason = contradiction_flip("KILL", _flip_entry(), reasoning)
                self.assertEqual(vote, "UPHOLD")
                self.assertIn("converging evidence", reason)

    def test_kill_is_preserved_for_a_positive_control_after_because(self):
        vote, reason = contradiction_flip(
            "KILL",
            _flip_entry(),
            "No SQL injection is possible because the query uses parameterized placeholders.",
        )
        self.assertEqual(vote, "KILL")
        self.assertIn("withheld", reason)

    def test_kill_is_preserved_for_an_unreachable_path(self):
        for reasoning in (
            "The endpoint is unreachable from the router.",
            "The dangerous function is never called.",
            "This is dead code.",
            "The input is not attacker-controlled.",
            "The value cannot reach the sink.",
        ):
            with self.subTest(reasoning=reasoning):
                vote, reason = contradiction_flip("KILL", _flip_entry(), reasoning)
                self.assertEqual(vote, "KILL")
                self.assertIn("withheld", reason)

    def test_kill_still_flips_for_a_negated_unreachability_claim(self):
        vote, reason = contradiction_flip(
            "KILL", _flip_entry(), "The endpoint is not unreachable; it is live."
        )
        self.assertEqual(vote, "UPHOLD")
        self.assertIn("converging evidence", reason)

    def test_kill_still_flips_when_no_route_means_no_authentication(self):
        vote, reason = contradiction_flip(
            "KILL",
            _flip_entry(),
            "There is no route-level authentication on this delete handler.",
        )
        self.assertEqual(vote, "UPHOLD")
        self.assertIn("converging evidence", reason)

    def test_duplicate_reports_are_labeled_self_review_not_independent(self):
        source = (
            "from flask import Flask\n"
            "app = Flask(__name__)\n"
            "@app.post('/admin/delete')\n"
            "def delete_user():\n"
            "    delete_user_by_id(1)\n"
            "    return {'ok': True}\n"
        )
        result = swarm_redteam(
            object(),
            _TwoProviderClient(),
            "Review the supplied route.",
            synth=False,
            swarm=[("p1", "meta/llama-3.3-70b"), ("p2", "qwen/qwen2.5-72b")],
            source_files={"app.py": source},
        )

        self.assertEqual(
            result["metrics"]["challenge_independence"],
            "self-critique only; not independent corroboration",
        )
        self.assertEqual(
            result["metrics"]["challenge_independence_detail"]["self_review_assignments"],
            2,
        )
        self.assertEqual(
            result["metrics"]["challenge_independence_detail"]["independent_assignments"],
            0,
        )
        for validation in result["challenge_validations"]:
            self.assertEqual(len(validation["self_review_candidate_ids"]), 1)
            self.assertEqual(validation["independent_candidate_ids"], [])
        candidate = result["candidate_triage"][0]
        self.assertEqual(candidate["challenge_independence"], "self_only")
        self.assertTrue(all(
            review.get("self_review") for review in candidate["challenge_reviews"]
        ))
        challenge_calls = [
            call for call in result["stage_calls"]
            if call.get("stage") == "cross_examination"
        ]
        self.assertEqual(len(challenge_calls), 2)
        for call in challenge_calls:
            self.assertIn("includes self-review", call["role"])
            self.assertNotIn("INDEPENDENT CHALLENGER", call["role"])

    def test_triage_infers_self_review_for_outcomes_without_the_flag(self):
        candidate_id = candidate_reference_id(
            "app.py", 1, 1, "sensitive_data_exposure", "secret_token"
        )
        validations = [{
            "provider": "p1",
            "model": "m1",
            "findings_checked": [{
                "finding": "token exposed",
                "file": "app.py",
                "start_line": 1,
                "end_line": 1,
                "category": "sensitive_data_exposure",
                "quote": "secret_token",
                "grounded": True,
            }],
        }]
        outcomes = [{
            "candidate_id": candidate_id,
            "status": "supported",
            "reason": "the response returns the token",
            "provider": "p1",
            "model": "m1",
        }]
        candidates = triage_grounded_findings(validations, [], outcomes)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["challenge_independence"], "self_only")
        self.assertTrue(candidates[0]["challenge_reviews"][0]["self_review"])


class TestMergeAndSwarmDiversity(unittest.TestCase):
    def test_claim_merge_ignores_punctuation_and_emphasis_only(self):
        merged = merge_claims([
            {"provider": "p1", "model": "m1", "answer": "Shared point.", "error": None},
            {"provider": "p2", "model": "m2", "answer": "**Shared point**", "error": None},
        ])
        self.assertEqual(len(merged["supported"]), 1)
        self.assertEqual(merged["supported"][0]["supporters"], ["p1:m1", "p2:m2"])
        self.assertEqual(merged["single_source"], [])

    def test_build_swarm_pass_two_prefers_unused_family(self):
        class Cfg:
            data = {"providers": {"pa": {}}}

            def get_api_key(self, provider):
                return "key"

            def get_default_model(self, provider):
                return "openai/gpt-4o" if provider == "pa" else ""

            def get_fallback_models(self, provider):
                return ["openai/gpt-4o-mini", "meta/llama-3.3-70b"]

        self.assertEqual(
            build_swarm(Cfg(), size=2),
            [("pa", "openai/gpt-4o"), ("pa", "meta/llama-3.3-70b")],
        )


class TestDirectoryScanSafety(unittest.TestCase):
    def test_scan_skips_credentials_and_symlink_escapes(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
            root = Path(td)
            (root / "app.py").write_text("safe()\n", encoding="utf-8")
            (root / "notes.md").write_text("# notes\n", encoding="utf-8")
            (root / "credentials.json").write_text('{"token": "x"}\n', encoding="utf-8")
            (root / "config").mkdir()
            (root / "config" / "secrets.yaml").write_text("password: x\n", encoding="utf-8")
            outside = Path(outside_td) / "outside.py"
            outside.write_text("EXFILTRATE = 1\n", encoding="utf-8")
            (root / "outside.py").symlink_to(outside)

            sources = scan_directory_sources(root)
            self.assertIn("app.py", sources)
            self.assertIn("notes.md", sources)
            self.assertNotIn("credentials.json", sources)
            self.assertNotIn("config/secrets.yaml", sources)
            self.assertNotIn("outside.py", sources)

            included = scan_directory_sources(root, include_sensitive=True)
            self.assertIn("credentials.json", included)
            self.assertIn("config/secrets.yaml", included)


if __name__ == "__main__":
    unittest.main()
