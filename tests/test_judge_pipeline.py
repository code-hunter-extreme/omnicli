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
