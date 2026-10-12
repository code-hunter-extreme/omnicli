import unittest

from omnicli.judge_pipeline import run_candidate_judge_graph


class CandidateJudgeGraphTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {
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
        self.sources = {"app.py": "def endpoint():\n    dangerous_call()\n"}

    def test_judges_existing_candidate_through_typed_stages(self):
        calls = []

        def model_call(provider, model, messages, max_tokens):
            calls.append((provider, model, max_tokens, messages))
            return {
                "provider": provider,
                "model": model,
                "answer": (
                    "JUDGMENT: UPHOLD\n"
                    "REASONING: the cited code supports the candidate."
                ),
                "error": None,
            }

        result = run_candidate_judge_graph(
            [self.candidate],
            self.sources,
            [],
            [("provider", "model-x")],
            model_call,
            256,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(result["verdicts"][0]["verdict"], "UPHOLD")
        judge_ballots = [
            ballot for ballot in result["verdicts"][0]["ballots"]
            if ballot["voter"].startswith("judge:")
        ]
        self.assertEqual(len(judge_ballots), 1)
        self.assertEqual(judge_ballots[0]["weight"], 2.0)
        self.assertEqual(
            result["verdicts"][0]["original_status"],
            "unverified_candidate",
        )
        self.assertEqual(
            [event["stage"] for event in result["audit_trail"] if event["event"] == "enter"],
            ["PROBE", "DETECT", "JUDGE", "AGGREGATE"],
        )
        self.assertIn("dangerous_call()", calls[0][3][1]["content"])
        self.assertEqual(self.candidate["status"], "unverified_candidate")

    def test_provider_error_is_visible_and_never_becomes_a_verdict(self):
        result = run_candidate_judge_graph(
            [self.candidate],
            self.sources,
            [],
            [("provider", "model-x")],
            lambda provider, model, _messages, _tokens: {
                "provider": provider,
                "model": model,
                "answer": "",
                "error": "rate limited",
            },
            256,
        )

        verdict = result["verdicts"][0]
        self.assertEqual(verdict["verdict"], "UNJUDGED")
        self.assertEqual(verdict["provider_error"], "rate limited")
        self.assertEqual(result["calls"][0]["parsed_judgment"]["judgment"], "UNPARSEABLE")

    def test_judge_fails_over_to_surviving_member(self):
        calls = []

        def model_call(provider, model, _messages, _tokens):
            calls.append(provider)
            if provider == "dead":
                return {
                    "provider": provider,
                    "model": model,
                    "answer": "",
                    "error": "429 Rate limit reached",
                }
            return {
                "provider": provider,
                "model": model,
                "answer": (
                    "JUDGMENT: UPHOLD\n"
                    "REASONING: the cited code supports the candidate."
                ),
                "error": None,
            }

        result = run_candidate_judge_graph(
            [self.candidate],
            self.sources,
            [],
            [("dead", "m1"), ("alive", "m2")],
            model_call,
            256,
        )

        self.assertEqual(calls, ["dead", "alive"])
        self.assertEqual(len(result["calls"]), 2)
        verdict = result["verdicts"][0]
        self.assertEqual(verdict["verdict"], "UPHOLD")
        self.assertEqual(verdict["provider"], "alive")
        self.assertTrue(any("failed over" in note for note in verdict["flip_notes"]))
        self.assertEqual(len(result["failures"]), 1)
        self.assertIn("dead", result["failures"][0])

    def test_judge_failover_preserves_unjudged_when_all_fail(self):
        result = run_candidate_judge_graph(
            [self.candidate],
            self.sources,
            [],
            [("dead-a", "m1"), ("dead-b", "m2")],
            lambda provider, model, _messages, _tokens: {
                "provider": provider,
                "model": model,
                "answer": "",
                "error": "429 Rate limit reached",
            },
            256,
        )

        verdict = result["verdicts"][0]
        self.assertEqual(verdict["verdict"], "UNJUDGED")
        self.assertEqual(len(result["calls"]), 2)
        self.assertEqual(len(result["failures"]), 2)

    def test_judge_fails_over_on_unparseable_answer(self):
        calls = []

        def model_call(provider, model, _messages, _tokens):
            calls.append(provider)
            if provider == "vague":
                return {
                    "provider": provider,
                    "model": model,
                    "answer": "looks bad, not sure",
                    "error": None,
                }
            return {
                "provider": provider,
                "model": model,
                "answer": "JUDGMENT: KILL\nREASONING: quote is benign.",
                "error": None,
            }

        result = run_candidate_judge_graph(
            [self.candidate],
            self.sources,
            [],
            [("vague", "m1"), ("crisp", "m2")],
            model_call,
            256,
        )

        self.assertEqual(calls, ["vague", "crisp"])
        self.assertEqual(result["calls"][1]["parsed_judgment"]["judgment"], "KILL")
        self.assertNotEqual(result["verdicts"][0]["verdict"], "UNJUDGED")

    def test_empty_candidate_set_skips_judge_and_preserves_graph_audit(self):
        result = run_candidate_judge_graph(
            [],
            self.sources,
            [],
            [],
            lambda *_args: self.fail("empty candidate set must not call a model"),
            256,
        )

        self.assertEqual(result["verdicts"], [])
        self.assertEqual(result["calls"], [])
        self.assertEqual(
            [event["stage"] for event in result["audit_trail"] if event["event"] == "enter"],
            ["PROBE", "DETECT", "AGGREGATE"],
        )


if __name__ == "__main__":
    unittest.main()
