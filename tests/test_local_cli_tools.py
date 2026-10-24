"""Local coding-CLI behavior: operate in the project, not just the CWD.

These tests pin the tool surface an operator gets from a repository
subdirectory: discover the project root, find/search/read local code,
and make targeted edits without needing shell access.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

SENTINEL = "LOCAL_CLI_SENTINEL_7"
SECRET = "LOCAL_CLI_SECRET_DO_NOT_RETURN"


class LocalCliWorkspaceCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.pkg = self.root / "pkg"
        self.sub = self.pkg / "sub"
        self.sub.mkdir(parents=True)
        (self.root / "README.md").write_text("# Demo\n", encoding="utf-8")
        (self.pkg / "auth_flow.py").write_text(
            f"# {SENTINEL}\n"
            "def check_session_cookie(req):\n"
            "    return req.cookies.get('sid') == 'ok'\n",
            encoding="utf-8",
        )
        (self.pkg / "billing.py").write_text(
            "def invoice_total(items):\n    return sum(items)\n",
            encoding="utf-8",
        )
        workflows = self.root / ".github" / "workflows"
        workflows.mkdir(parents=True)
        (workflows / "ci.yml").write_text(
            "name: CI_SENTINEL\non: [push]\n", encoding="utf-8"
        )
        (self.root / "Makefile").write_text(
            "test:\n\t@echo MAKE_SENTINEL\n", encoding="utf-8"
        )
        (self.root / "credentials.json").write_text(
            '{"token": "' + SECRET + '"}\n', encoding="utf-8"
        )
        (self.root / ".env").write_text(f"API_KEY={SECRET}\n", encoding="utf-8")
        (self.root / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
        (self.root / "ignored.py").write_text(
            "IGNORED_SENTINEL = True\n", encoding="utf-8"
        )
        # A real repo when git is available; a .git marker still proves
        # root discovery on systems where it is not.
        try:
            subprocess.run(
                ["git", "init", "-q"], cwd=self.root,
                check=True, capture_output=True, timeout=10,
            )
        except Exception:
            (self.root / ".git").mkdir(exist_ok=True)
        self._old_cwd = os.getcwd()
        os.chdir(self.sub)

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._tmp.cleanup()


class TestWorkspaceRoot(LocalCliWorkspaceCase):
    def test_root_is_discovered_from_repository_subdirectory(self):
        from omnicli.tools import check_workspace_jail, resolve_workspace_root

        self.assertEqual(resolve_workspace_root(), self.root)
        self.assertEqual(resolve_workspace_root(self.sub), self.root)
        self.assertEqual(
            check_workspace_jail("pkg/auth_flow.py"),
            self.pkg / "auth_flow.py",
        )
        with self.assertRaises(PermissionError):
            check_workspace_jail("../../outside.txt")

    def test_default_grounding_uses_project_root_not_launch_subdirectory(self):
        from omnicli.grounding import build_grounding

        grounding = build_grounding(
            "how does the auth flow check the session cookie?"
        )
        self.assertEqual(grounding["workspace"], str(self.root))
        self.assertIn(
            "pkg/auth_flow.py",
            [item["path"] for item in grounding["attached"]],
        )
        self.assertIn(SENTINEL, grounding["block"])
        self.assertNotIn(SECRET, grounding["block"])
        self.assertNotIn("ignored.py", grounding["workspace_map"])

    def test_grounding_sees_tracked_dot_github_and_extensionless_files(self):
        from omnicli.grounding import build_grounding

        ci = build_grounding("explain the CI workflow")
        self.assertIn(
            ".github/workflows/ci.yml",
            [item["path"] for item in ci["attached"]],
        )
        make = build_grounding("how do I run the make test target")
        self.assertIn("MAKE_SENTINEL", make["block"])


class TestLocalCodeTools(LocalCliWorkspaceCase):
    def test_find_files_returns_workspace_relative_paths(self):
        from omnicli.tools import tool_find_files

        out = tool_find_files("*.py")
        self.assertIn("pkg/auth_flow.py", out)
        self.assertIn("pkg/billing.py", out)
        self.assertNotIn("ignored.py", out)
        self.assertNotIn("credentials.json", out)

    def test_search_code_finds_symbol_from_subdirectory(self):
        from omnicli.tools import tool_search_code

        out = tool_search_code("check_session_cookie")
        self.assertIn("pkg/auth_flow.py:2:", out)
        self.assertIn("def check_session_cookie", out)
        self.assertNotIn(SECRET, out)
        self.assertNotIn("IGNORED_SENTINEL", out)

        regex_out = tool_search_code(
            r"check_.*cookie", regex=True, file_glob="*.py"
        )
        self.assertIn("pkg/auth_flow.py:2:", regex_out)

        secret_out = tool_search_code(SECRET)
        self.assertIn("No matches", secret_out)
        self.assertNotIn(SECRET, secret_out.split("No matches", 1)[-1])

    def test_read_file_supports_numbered_line_ranges_from_subdirectory(self):
        from omnicli.tools import tool_read_file

        out = tool_read_file(
            "pkg/auth_flow.py", start_line=2, end_line=3
        )
        self.assertIn("lines 2-3 of 3", out)
        self.assertIn("00002 | def check_session_cookie", out)
        self.assertIn("00003 |     return", out)
        self.assertNotIn(SENTINEL, out)

    def test_edit_file_requires_exact_match_unless_replace_all(self):
        from omnicli.tools import tool_edit_file

        out = tool_edit_file(
            "pkg/auth_flow.py",
            "return req.cookies.get('sid') == 'ok'",
            "return req.cookies.get('sid') == 'approved'",
        )
        self.assertIn("Successfully edited", out)
        content = (self.pkg / "auth_flow.py").read_text(encoding="utf-8")
        self.assertIn("'approved'", content)

        missing = tool_edit_file("pkg/auth_flow.py", "no such text", "x")
        self.assertIn("Error", missing)
        self.assertEqual(
            (self.pkg / "auth_flow.py").read_text(encoding="utf-8"), content
        )

        dup = self.pkg / "dup.py"
        dup.write_text("x = 1\nx = 1\n", encoding="utf-8")
        ambiguous = tool_edit_file("pkg/dup.py", "x = 1", "x = 2")
        self.assertIn("Error", ambiguous)
        self.assertIn("2", ambiguous)
        replaced = tool_edit_file(
            "pkg/dup.py", "x = 1", "x = 2", replace_all=True
        )
        self.assertIn("Successfully edited", replaced)
        self.assertEqual(dup.read_text(encoding="utf-8"), "x = 2\nx = 2\n")

    def test_execute_tool_call_dispatches_local_code_tools(self):
        from omnicli.tools import TOOL_DEFINITIONS, execute_tool_call

        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        self.assertTrue({"find_files", "search_code", "edit_file"} <= names)
        out = execute_tool_call(
            "search_code", {"query": "invoice_total"}
        )
        self.assertIn("pkg/billing.py:1:", out)


class TestReadOnlyToolset(LocalCliWorkspaceCase):
    def test_verification_toolset_can_search_but_cannot_edit(self):
        from omnicli.agent import verification_toolset

        names = {t["function"]["name"] for t in verification_toolset()}
        self.assertIn("find_files", names)
        self.assertIn("search_code", names)
        self.assertNotIn("edit_file", names)
        self.assertNotIn("write_file", names)


class TestAgentWorkspace(LocalCliWorkspaceCase):
    def test_agent_grounds_and_executes_tools_in_project_root(self):
        from omnicli.agent import AgentRunner

        runner = AgentRunner.__new__(AgentRunner)
        runner.cfg = MagicMock()
        runner.cfg.get_fallback_models.return_value = []
        runner.client = MagicMock()
        runner.provider = "p"
        runner.model = "openai/test"
        runner.max_turns = 1
        runner.console = MagicMock()
        runner.redteam = False
        runner.rotate_families = False
        runner.allow_shell = False
        runner.allow_local = False
        runner.allowed_hosts = []
        runner.allow_mutating_http = False
        runner.ground_local = True
        runner.workspace_root = self.root
        captured = {}

        def fake_call(provider, model, messages):
            captured["messages"] = messages
            return (
                {"choices": [{"message": {"content": "done", "tool_calls": []}}]},
                None,
            )

        runner._call_model = fake_call
        with patch("omnicli.memory.load_memory", return_value=""):
            answer = runner.run(
                "how does the auth flow check the session cookie?"
            )
        self.assertEqual(answer, "done")
        sent = "\n".join(
            m.get("content", "") for m in captured["messages"]
            if isinstance(m.get("content"), str)
        )
        self.assertIn(SENTINEL, sent)
        self.assertIn("pkg/auth_flow.py", sent)

    def test_agent_cli_forwards_workspace(self):
        from omnicli.cli import main

        with patch("omnicli.agent.AgentRunner") as runner_cls:
            result = CliRunner().invoke(
                main,
                ["agent", "--workspace", str(self.root),
                 "--max-turns", "2", "fix", "auth"],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            str(runner_cls.call_args.kwargs["workspace_root"]), str(self.root)
        )
        runner_cls.return_value.run.assert_called_once_with("fix auth")


if __name__ == "__main__":
    unittest.main()
