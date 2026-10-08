"""Regression tests for blind source-security evaluation fixtures."""

import json
import re
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner
from omnicli.cli import main
from omnicli.evaluation import (
    create_blinded_review_package,
    evaluate_security_models,
    load_security_dataset,
)


class _RecordingClient:
    def __init__(self):
        self.prompts = []

    def chat_complete(self, *, messages, **_kwargs):
        self.prompts.append(messages[-1]["content"])
        return {
            "choices": [{"message": {"content": '{"findings": []}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class TestRedTeamBenchmarkIntegrity(unittest.TestCase):
    def test_held_out_fixture_uses_neutral_identity_metadata(self):
        path = (
            Path(__file__).parent
            / "fixtures"
            / "redteam"
            / "cases.held_out.provisional.jsonl"
        )
        cases = load_security_dataset(str(path))

        self.assertEqual(
            [case["id"] for case in cases],
            [f"hc-{number:03d}" for number in range(1, 61)],
        )
        for number, case in enumerate(cases, start=1):
            expected_file = f"source_{number:03d}.py"
            self.assertEqual(case["file"], expected_file)
            self.assertTrue(re.fullmatch(r"source_\d{3}\.py", case["file"]))
            self.assertNotRegex(case["id"], r"(positive|benign|vulnerable|safe)")
            self.assertNotRegex(case["file"], r"(positive|benign|vulnerable|safe)")
            for finding in case["expected_findings"]:
                self.assertEqual(finding["file"], expected_file)

    def test_direct_review_prompt_excludes_case_identifier(self):
        client = _RecordingClient()
        case = {
            "id": "internal-label-must-not-reach-the-model",
            "prompt": "Review the supplied source.",
            "file": "source_001.py",
            "source": "def health():\n    return 'ok'\n",
            "expected_findings": [],
        }

        evaluate_security_models(
            client=client,
            cases=[case],
            models=[("test", "test-model")],
        )

        self.assertEqual(len(client.prompts), 1)
        self.assertNotIn(case["id"], client.prompts[0])
        self.assertIn(case["file"], client.prompts[0])

    def test_blinded_review_package_omits_gold_labels_and_identity_metadata(self):
        cases = [{
            "id": "heldout-positive-sql",
            "split": "held_out",
            "category": "sql_injection",
            "severity": "high",
            "prompt": "Find the known injection.",
            "file": "heldout_positive.py",
            "source": "def query(value):\n    return execute(value)\n",
            "expected_findings": [{
                "category": "sql_injection",
                "file": "heldout_positive.py",
                "line": 2,
                "evidence": "return execute(value)",
                "preconditions": "The caller controls value.",
            }],
            "benign_rationale": None,
            "adjudication_status": "provisional",
            "reviewer_count": 1,
        }]

        package = create_blinded_review_package(cases)
        rendered = json.dumps(package)

        self.assertEqual(package["cases"][0]["review_id"], "review-001")
        self.assertEqual(package["cases"][0]["file"], "source_001.py")
        self.assertEqual(
            set(package["cases"][0]), {"review_id", "file", "task", "source"}
        )
        for concealed in (
            "heldout-positive-sql",
            "heldout_positive.py",
            "sql_injection",
            '"severity"',
            '"expected_findings"',
            '"benign_rationale"',
            '"preconditions"',
        ):
            self.assertNotIn(concealed, rendered)

    def test_blinded_review_package_rejects_non_held_out_cases(self):
        with self.assertRaisesRegex(ValueError, "held_out"):
            create_blinded_review_package([{
                "id": "development",
                "split": "development",
                "source": "x = 1\n",
            }])

    def test_cli_writes_blinded_review_package_without_overwriting(self):
        dataset = (
            Path(__file__).parent
            / "fixtures"
            / "redteam"
            / "cases.held_out.provisional.jsonl"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "independent-review.json"
            result = CliRunner().invoke(
                main,
                [
                    "prepare-security-review",
                    "--dataset", str(dataset),
                    "--output", str(output),
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            package = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(package["cases"]), 60)
            self.assertEqual(
                set(package["cases"][0]), {"review_id", "file", "task", "source"}
            )
            self.assertEqual(package["cases"][0]["review_id"], "review-001")
            self.assertEqual(package["cases"][0]["file"], "source_001.py")

            result = CliRunner().invoke(
                main,
                [
                    "prepare-security-review",
                    "--dataset", str(dataset),
                    "--output", str(output),
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Refusing to overwrite", result.output)


if __name__ == "__main__":
    unittest.main()
