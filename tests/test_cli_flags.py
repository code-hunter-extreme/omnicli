"""Offline CLI contract tests for swarm/chat flags and numeric option ranges."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from omnicli.cli import main


def _passing_redteam_payload():
    return {
        "members": [],
        "final": None,
        "evidence_gate": {
            "status": "PASS_GROUNDING_ONLY",
            "security_conclusion": "NOT_ASSESSED",
            "failed_checks": [],
        },
    }


class TestSwarmFlagValidation(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_help_no_longer_advertises_removed_builder_mode(self):
        result = self.runner.invoke(main, ["swarm", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("builder", result.output.lower())
        self.assertIn("--source-file", result.output)
        self.assertIn("--scan-cwd", result.output)

    def test_conflicting_mode_flags_are_rejected(self):
        result = self.runner.invoke(
            main, ["swarm", "--collaborate", "--redteam", "audit", "this"]
        )
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("Choose one mode", result.output)

    def test_default_swarm_mode_is_collaborate(self):
        with (
            patch("omnicli.orchestrate.build_swarm", return_value=[("p", "openai/test")]),
            patch(
                "omnicli.orchestrate.run_orchestration",
                return_value={"members": [], "final": "ok", "metrics": {}},
            ) as run,
        ):
            result = self.runner.invoke(
                main, ["swarm", "--raw", "--provider", "p", "audit", "this"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(run.call_args.args[0], "collaborate")

    def test_chat_conflicting_mode_flags_are_rejected(self):
        result = self.runner.invoke(main, ["chat", "--redteam", "--single"])
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("Choose one mode", result.output)

    def test_scan_cwd_is_redteam_only(self):
        result = self.runner.invoke(
            main, ["swarm", "--scan-cwd", "audit", "this"]
        )
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("only supported in red-team mode", result.output)

    def test_report_path_is_redteam_only(self):
        result = self.runner.invoke(
            main, ["swarm", "--report-path", "out.md", "audit", "this"]
        )
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("only supported in red-team mode", result.output)

    def test_allowed_hosts_requires_member_tools(self):
        result = self.runner.invoke(
            main, ["swarm", "--allowed-hosts", "example.test", "audit", "this"]
        )
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("--allowed-hosts requires --member-tools", result.output)

    def test_size_must_be_positive(self):
        result = self.runner.invoke(main, ["swarm", "--size", "0", "audit", "this"])
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("Invalid value", result.output)

    def test_redteam_default_report_path_is_forwarded(self):
        with (
            patch("omnicli.orchestrate.build_swarm", return_value=[("p", "openai/test")]),
            patch(
                "omnicli.orchestrate.run_orchestration",
                return_value=_passing_redteam_payload(),
            ) as run,
        ):
            result = self.runner.invoke(
                main, ["swarm", "--redteam", "--raw", "--provider", "p", "audit", "this"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(run.call_args.kwargs["report_path"], "reports")
        self.assertEqual(run.call_args.kwargs["source_files"], {})

    def test_redteam_explicit_report_path_is_forwarded(self):
        payload = _passing_redteam_payload()
        with (
            patch("omnicli.orchestrate.build_swarm", return_value=[("p", "openai/test")]),
            patch("omnicli.orchestrate.run_orchestration", return_value=payload) as run,
        ):
            result = self.runner.invoke(
                main,
                [
                    "swarm", "--redteam", "--raw", "--provider", "p",
                    "--report-path", "custom-report.md", "audit", "this",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(run.call_args.kwargs["report_path"], "custom-report.md")

    def test_scan_cwd_is_ignored_with_explicit_source(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "explicit.py"
            source.write_text("EXPLICIT = 1\n", encoding="utf-8")
            payload = _passing_redteam_payload()
            with (
                patch("omnicli.orchestrate.build_swarm", return_value=[("p", "openai/test")]),
                patch("omnicli.orchestrate.run_orchestration", return_value=payload) as run,
            ):
                result = self.runner.invoke(
                    main,
                    [
                        "swarm", "--redteam", "--scan-cwd", "--raw", "--provider", "p",
                        "--source-file", str(source), "audit", "this",
                    ],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--scan-cwd ignored", result.output)
        self.assertEqual(
            run.call_args.kwargs["source_files"],
            {str(source): "EXPLICIT = 1\n"},
        )

    def test_report_path_with_raw_warns_that_no_report_is_written(self):
        with (
            patch("omnicli.orchestrate.build_swarm", return_value=[("p", "openai/test")]),
            patch(
                "omnicli.orchestrate.run_orchestration",
                return_value=_passing_redteam_payload(),
            ) as run,
        ):
            result = self.runner.invoke(
                main,
                ["swarm", "--redteam", "--raw", "--report-path", "unused.md",
                 "--provider", "p", "audit", "this"],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("has no effect with --raw", result.output)
        self.assertEqual(run.call_args.kwargs["report_path"], "unused.md")

    def test_scan_cwd_is_ignored_when_an_explicit_but_empty_dir_is_given(self):
        with tempfile.TemporaryDirectory() as td:
            payload = _passing_redteam_payload()
            with (
                patch("omnicli.orchestrate.build_swarm", return_value=[("p", "openai/test")]),
                patch("omnicli.orchestrate.run_orchestration", return_value=payload) as run,
            ):
                result = self.runner.invoke(
                    main,
                    [
                        "swarm", "--redteam", "--scan-cwd", "--raw", "--provider", "p",
                        "--dir", td, "audit", "this",
                    ],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--scan-cwd ignored", result.output)
        self.assertEqual(run.call_args.kwargs["source_files"], {})

    def test_enrich_files_warns_and_is_ignored_in_redteam(self):
        with (
            patch("omnicli.orchestrate.build_swarm", return_value=[("p", "openai/test")]),
            patch(
                "omnicli.orchestrate.run_orchestration",
                return_value=_passing_redteam_payload(),
            ) as run,
        ):
            result = self.runner.invoke(
                main,
                ["swarm", "--redteam", "--enrich-files", "--raw", "--provider", "p",
                 "audit", "this"],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("ignored in red-team mode", result.output)
        self.assertTrue(run.call_args.kwargs["enrich_files"])


class TestNumericFlagRanges(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_temperature_is_bounded(self):
        for argv in (
            ["ask", "--temperature", "3", "hello"],
            ["ask", "--temperature", "-0.1", "hello"],
            ["chat", "--temperature", "2.1"],
        ):
            with self.subTest(argv=argv):
                result = self.runner.invoke(main, argv)
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertIn("Invalid value", result.output)

    def test_positive_integer_flags_reject_zero(self):
        cases = (
            ["ask", "--max-tokens", "0", "hello"],
            ["agent", "--max-turns", "0", "do", "work"],
            ["serve", "--port", "0"],
            ["scan", "--panel-size", "0"],
            ["scan", "--max-per-family", "0"],
        )
        for argv in cases:
            with self.subTest(argv=argv):
                result = self.runner.invoke(main, argv)
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertIn("Invalid value", result.output)


if __name__ == "__main__":
    unittest.main()
