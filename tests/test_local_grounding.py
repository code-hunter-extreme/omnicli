"""Local-file grounding: orchestration must work on local files, not memory.

User-facing entry points (swarm/ask/agent/chat/workflow) ground prompts in
relevance-ranked files from the current workspace by default; the library
default stays bare so programmatic callers (e.g. Oracle's OmniCLIBackend,
which already embeds its own sources) never get cwd contents implicitly.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

SENTINEL = "LOCAL_SENTINEL_XYZPDQ_42"
SECRET_SENTINEL = "SHOULD_NEVER_LEAK_99"


def _make_workspace(root: Path) -> Path:
    (root / "auth_flow.py").write_text(
        f"# {SENTINEL}\n"
        "def check_session_cookie(req):\n"
        "    return req.cookies.get('sid') == 'ok'\n",
        encoding="utf-8",
    )
    (root / "billing.py").write_text(
        "def invoice_total(items):\n    return sum(items)\n", encoding="utf-8"
    )
    (root / ".env").write_text(f"API_KEY={SECRET_SENTINEL}\n", encoding="utf-8")
    (root / "README.md").write_text("# Demo project\n", encoding="utf-8")
    return root


class _Cfg:
    data = {"providers": {}}

    def get_api_key(self, provider):
        return "x"

    def get_default_provider(self):
        return "p"

    def get_default_model(self, provider):
        return "openai/test"

    def get_fallback_models(self, provider):
        return []


class _RecordingClient:
    def __init__(self, answer="generic answer"):
        self.answer = answer
        self.calls = []

    def chat_complete(self, provider_id=None, messages=None, model=None, **kwargs):
        self.calls.append(messages)
        return {"choices": [{"message": {"content": self.answer}}]}

    def all_text(self):
        return "\n".join(
            m.get("content", "")
            for call in self.calls
            for m in (call or [])
            if isinstance(m.get("content"), str)
        )


class _WorkspaceCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = _make_workspace(Path(self._tmp.name))
        self._old_cwd = os.getcwd()
        os.chdir(self.ws)

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._tmp.cleanup()


class TestBuildGrounding(_WorkspaceCase):
    def test_relevant_file_attached_with_line_numbers_and_rules(self):
        from omnicli.grounding import build_grounding

        gr = build_grounding(
            "how does the auth flow check the session cookie?", root=self.ws
        )
        attached_paths = [item["path"] for item in gr["attached"]]
        self.assertIn("auth_flow.py", attached_paths)
        self.assertIn(SENTINEL, gr["block"])
        self.assertIn("auth_flow.py", gr["block"])
        self.assertRegex(gr["block"], r"00001 \|")
        self.assertIn("cite", gr["block"].lower())
        self.assertIn("LOCAL WORKSPACE GROUNDING", gr["block"])

    def test_sensitive_files_are_never_attached_or_mapped(self):
        from omnicli.grounding import build_grounding

        gr = build_grounding("what is the api key in .env?", root=self.ws)
        self.assertNotIn(SECRET_SENTINEL, gr["block"])
        self.assertNotIn(".env", [item["path"] for item in gr["attached"]])

    def test_named_file_wins_over_unrelated_files(self):
        from omnicli.grounding import build_grounding

        gr = build_grounding("explain billing.py", root=self.ws)
        self.assertEqual(gr["attached"][0]["path"], "billing.py")

    def test_empty_workspace_produces_no_block(self):
        from omnicli.grounding import build_grounding

        with tempfile.TemporaryDirectory() as empty:
            gr = build_grounding("how does anything work?", root=Path(empty))
        self.assertEqual(gr["attached"], [])
        self.assertEqual(gr["block"], "")

    def test_irrelevant_prompt_attaches_no_source_bodies(self):
        from omnicli.grounding import build_grounding

        gr = build_grounding("zqxjkvb nonexistenttopic", root=self.ws)
        self.assertNotIn(SENTINEL, gr["block"])


class TestOrchestrationGrounding(_WorkspaceCase):
    def test_ground_local_reaches_collaborate_members(self):
        from omnicli.orchestrate import run_orchestration

        client = _RecordingClient()
        out = run_orchestration(
            "collaborate", _Cfg(), client,
            "how does the auth flow check the session cookie?",
            swarm=[("p", "openai/test")], synth=False, ground_local=True,
        )
        self.assertIn(SENTINEL, client.all_text())
        self.assertIn(
            "auth_flow.py",
            [item["path"] for item in out["grounding"]["attached"]],
        )

    def test_library_default_stays_bare_for_programmatic_callers(self):
        from omnicli.orchestrate import run_orchestration

        client = _RecordingClient()
        out = run_orchestration(
            "collaborate", _Cfg(), client,
            "how does the auth flow check the session cookie?",
            swarm=[("p", "openai/test")], synth=False,
        )
        self.assertNotIn(SENTINEL, client.all_text())
        self.assertIsNone(out.get("grounding"))

    def test_explicit_source_files_win_over_auto_grounding(self):
        from omnicli.orchestrate import run_orchestration

        client = _RecordingClient()
        run_orchestration(
            "collaborate", _Cfg(), client, "review the chosen file",
            swarm=[("p", "openai/test")], synth=False, ground_local=True,
            source_files={"chosen.py": "CHOSEN = 1\n"},
        )
        sent = client.all_text()
        self.assertIn("CHOSEN = 1", sent)
        self.assertNotIn(SENTINEL, sent)

    def test_redteam_never_auto_grounds_from_cwd(self):
        from omnicli.orchestrate import run_orchestration

        client = _RecordingClient()
        out = run_orchestration(
            "redteam", _Cfg(), client, "audit this",
            swarm=[("p", "openai/test")], ground_local=True,
        )
        self.assertEqual(out["members"], [])
        self.assertIn("EVIDENCE GATE: FAIL", out["final"])
        self.assertNotIn(SENTINEL, client.all_text())


class TestCliSwarmGrounding(_WorkspaceCase):
    def _invoke(self, argv):
        from omnicli.cli import main

        with (
            patch(
                "omnicli.orchestrate.build_swarm",
                return_value=[("p", "openai/test")],
            ),
            patch(
                "omnicli.orchestrate.run_orchestration",
                return_value={"members": [], "final": "ok", "metrics": {}},
            ) as run,
        ):
            result = CliRunner().invoke(main, argv)
        return result, run

    def test_swarm_collaborate_grounds_in_cwd_by_default(self):
        result, run = self._invoke(
            ["swarm", "--raw", "--provider", "p",
             "how does the auth flow check the session cookie"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        grounding = run.call_args.kwargs.get("grounding") or {}
        self.assertIn(
            "auth_flow.py", [item["path"] for item in grounding["attached"]]
        )
        self.assertIn("Local Grounding", result.output)

    def test_swarm_no_ground_sends_no_local_files(self):
        result, run = self._invoke(
            ["swarm", "--no-ground", "--raw", "--provider", "p",
             "how does the auth flow check the session cookie"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(run.call_args.kwargs.get("grounding"))

    def test_swarm_explicit_sources_skip_auto_grounding(self):
        result, run = self._invoke(
            ["swarm", "--raw", "--provider", "p",
             "--source-file", "billing.py", "review billing"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(run.call_args.kwargs.get("grounding"))
        self.assertEqual(
            list(run.call_args.kwargs["source_files"]), ["billing.py"]
        )


class TestCliAskGrounding(_WorkspaceCase):
    def _invoke(self, argv):
        from omnicli.cli import main

        client = _RecordingClient(answer="answer")
        with (
            patch("omnicli.cli.client", client),
            patch("omnicli.cli.cfg", _Cfg()),
        ):
            result = CliRunner().invoke(main, argv, input="")
        return result, client

    def test_ask_grounds_in_cwd_by_default(self):
        result, client = self._invoke(
            ["ask", "--no-stream", "--raw",
             "how does the auth flow check the session cookie"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(SENTINEL, client.all_text())

    def test_ask_unmatched_prompt_asks_before_general_fallback(self):
        # Owner law: local folder first. No local match -> never silently
        # answer from general knowledge; non-interactive runs must stop
        # and name the explicit opt-in.
        result, client = self._invoke(
            ["ask", "--no-stream", "--raw", "zqxjkvb nonexistenttopic quuxly"]
        )
        self.assertNotEqual(result.exit_code, 0, result.output)
        self.assertEqual(client.calls, [])
        self.assertIn("--allow-general", result.output)

    def test_ask_unmatched_with_allow_general_proceeds(self):
        result, client = self._invoke(
            ["ask", "--allow-general", "--no-stream", "--raw",
             "zqxjkvb nonexistenttopic quuxly"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(client.calls)

    def test_ask_no_ground_stays_bare(self):
        result, client = self._invoke(
            ["ask", "--no-ground", "--no-stream", "--raw",
             "how does the auth flow check the session cookie"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn(SENTINEL, client.all_text())


class TestAgentGrounding(_WorkspaceCase):
    def _run_agent(self, ground_local):
        from omnicli.agent import AgentRunner

        runner = AgentRunner.__new__(AgentRunner)
        runner.cfg = _Cfg()
        runner.client = MagicMock()
        runner.provider = "p"
        runner.model = "openai/test"
        runner.max_turns = 3
        runner.console = MagicMock()
        runner.redteam = False
        runner.rotate_families = False
        runner.allow_shell = False
        runner.allow_local = False
        runner.allowed_hosts = []
        runner.allow_mutating_http = False
        runner.ground_local = ground_local
        captured = {}

        def fake_call(provider, model, messages):
            captured["messages"] = messages
            return (
                {"choices": [{"message": {"content": "done", "tool_calls": []}}]},
                None,
            )

        runner._call_model = fake_call
        with patch("omnicli.memory.load_memory", return_value=""):
            answer = runner.run("how does the auth flow check the session cookie?")
        self.assertEqual(answer, "done")
        return "\n".join(
            m.get("content", "") for m in captured["messages"]
            if isinstance(m.get("content"), str)
        )

    def test_agent_pre_grounds_goal_in_local_files(self):
        self.assertIn(SENTINEL, self._run_agent(True))

    def test_agent_no_ground_stays_bare(self):
        self.assertNotIn(SENTINEL, self._run_agent(False))


class TestChatConsentGate(_WorkspaceCase):
    def _session(self, allow):
        from omnicli.chat import ChatSession

        session = ChatSession.__new__(ChatSession)
        session.console = MagicMock()
        session.cfg = _Cfg()
        session.client = MagicMock()
        session.provider = "p"
        session.model = "openai/test"
        session.mode = "collaborate"
        session.council_size = 1
        session.messages = [{"role": "system", "content": "sys"}]
        session.pending_files = {}
        session.last_report_path = None
        session.last_failed_prompt = None
        session.ground_local = True
        session.allow_general = allow
        return session

    def test_chat_unmatched_turn_aborts_without_consent(self):
        from omnicli.chat import ChatSession

        session = self._session(allow=False)
        with patch.object(ChatSession, "_apply_memory"), patch.object(
            ChatSession, "_confirm_general_fallback", return_value=False
        ), patch(
            "omnicli.orchestrate.run_orchestration",
            return_value={"members": [], "final": "ok"},
        ) as run:
            session.stream_turn("zqxjkvb nonexistenttopic quuxly")
        run.assert_not_called()

    def test_chat_unmatched_turn_proceeds_with_consent(self):
        from omnicli.chat import ChatSession

        session = self._session(allow=False)
        with patch.object(ChatSession, "_apply_memory"), patch.object(
            ChatSession, "_confirm_general_fallback", return_value=True
        ), patch(
            "omnicli.orchestrate.run_orchestration",
            return_value={"members": [], "final": "ok"},
        ) as run:
            session.stream_turn("zqxjkvb nonexistenttopic quuxly")
        run.assert_called_once()


class TestChatGrounding(_WorkspaceCase):
    def _session(self, gr_kwargs=None):
        from omnicli.chat import ChatSession

        session = ChatSession.__new__(ChatSession)
        session.console = MagicMock()
        session.cfg = _Cfg()
        session.client = MagicMock()
        session.provider = "p"
        session.model = "openai/test"
        session.mode = "collaborate"
        session.council_size = 1
        session.messages = [{"role": "system", "content": "sys"}]
        session.pending_files = {}
        session.last_report_path = None
        session.last_failed_prompt = None
        session.ground_local = True
        return session

    def test_chat_collaborate_turn_forwards_cwd_grounding(self):
        from omnicli.chat import ChatSession

        session = self._session()
        with patch.object(ChatSession, "_apply_memory"), patch(
            "omnicli.orchestrate.run_orchestration",
            return_value={"members": [], "final": "ok"},
        ) as run:
            session.stream_turn(
                "how does the auth flow check the session cookie?"
            )
        grounding = run.call_args.kwargs.get("grounding") or {}
        self.assertIn(
            "auth_flow.py", [item["path"] for item in grounding["attached"]]
        )


class TestWorkflowGrounding(_WorkspaceCase):
    def test_validate_accepts_ground_and_files(self):
        from omnicli.workflow import validate_workflow

        tasks = validate_workflow({"tasks": [
            {"id": "a", "prompt": "x", "ground": False, "files": ["auth_flow.py"]},
            {"id": "b", "prompt": "y"},
        ]})
        self.assertFalse(tasks[0]["ground"])
        self.assertEqual(tasks[0]["files"], ["auth_flow.py"])
        self.assertTrue(tasks[1]["ground"])

    def test_validate_rejects_bad_ground_and_files(self):
        from omnicli.workflow import validate_workflow

        for task in (
            {"id": "a", "prompt": "x", "ground": "yes"},
            {"id": "a", "prompt": "x", "files": "auth_flow.py"},
            {"id": "a", "prompt": "x", "files": ["a.py", "a.py"]},
        ):
            with self.subTest(task=task), self.assertRaises(ValueError):
                validate_workflow({"tasks": [task]})

    def _run_workflow_cli(self, tasks):
        from omnicli.cli import main

        workflow_file = self.ws / "workflow.json"
        workflow_file.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
        client = _RecordingClient(answer="done")
        with patch("omnicli.cli.client", client):
            result = CliRunner().invoke(
                main, ["workflow", "run", str(workflow_file)]
            )
        return result, client

    def test_workflow_task_grounds_by_default(self):
        result, client = self._run_workflow_cli([
            {"id": "s", "prompt": "how does the auth flow check the session cookie",
             "provider": "p", "model": "openai/test"},
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(SENTINEL, client.all_text())

    def test_workflow_task_ground_false_stays_bare(self):
        result, client = self._run_workflow_cli([
            {"id": "s", "prompt": "how does the auth flow check the session cookie",
             "provider": "p", "model": "openai/test", "ground": False},
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn(SENTINEL, client.all_text())

    def test_workflow_task_files_attach_named_file(self):
        result, client = self._run_workflow_cli([
            {"id": "s", "prompt": "summarize billing",
             "provider": "p", "model": "openai/test", "files": ["billing.py"]},
        ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("invoice_total", client.all_text())
        self.assertNotIn(SECRET_SENTINEL, client.all_text())


if __name__ == "__main__":
    unittest.main()


class TestGroundingRankingAndHomeGuard(unittest.TestCase):
    def test_definition_file_outranks_prose_mentions(self):
        # Owner report: asked about local code, got general-knowledge
        # answers. Cause: filename/prose density outranked the defining
        # code file. A class/def definition of the asked terms must win.
        from omnicli.grounding import build_grounding

        index = {
            "docs/SCOPE_NOTES.md": ("scope gate scope gate scope\n" * 6),
            "engine.py": (
                "class ScopeGate:\n"
                "    def is_in_scope(self, url):\n"
                "        return True\n"
            ),
        }
        gr = build_grounding(
            "how does the scope gate decide", root="/nonexistent-ws", index=index
        )
        self.assertTrue(gr["attached"])
        self.assertEqual(gr["attached"][0]["path"], "engine.py")

    def test_home_directory_is_not_a_workspace(self):
        # Launched from $HOME, grounding must say so fast and loud
        # instead of walking the whole home and grounding on noise.
        from omnicli.grounding import build_grounding

        gr = build_grounding("how does auth work", root=str(Path.home()))
        self.assertEqual(gr["files_indexed"], 0)
        self.assertEqual(gr["attached"], [])
        self.assertIn("not a project", gr["block"])
        self.assertIn("OMNICLI_WORKSPACE", gr["block"])

    def test_excerpt_budget_shared_across_files(self):
        # Big repos: first files must not eat the whole budget as 16k
        # excerpts while the defining file ranks just below them.
        from omnicli.grounding import build_grounding

        body = "class ScopeGate:\n    pass  # scope gate\n" + ("x = scope_gate_url  # scope gate url\n" * 900)
        index = {f"mod{i}.py": body for i in range(8)}
        gr = build_grounding(
            "how does the scope gate decide", root="/nonexistent-ws",
            index=index, max_chars=60_000,
        )
        self.assertGreaterEqual(len(gr["attached"]), 6)

    def test_oversized_defining_file_is_indexed(self):
        # The trio's largest product files (hunter ~400KB, scan4me
        # ~209KB) exceeded the old 200KB index cap and were invisible
        # to grounding — questions about their own code got doc echoes.
        import tempfile
        from pathlib import Path as _P

        from omnicli.grounding import build_grounding, index_workspace

        with tempfile.TemporaryDirectory() as td:
            big = _P(td) / "big_engine.py"
            big.write_text(
                "class ScopeGate:\n    pass  # scope gate\n"
                + ("# filler line to exceed the old cap\n" * 9000),
                encoding="utf-8",
            )
            self.assertGreater(big.stat().st_size, 200_000)
            idx = index_workspace(td)
            self.assertIn("big_engine.py", idx)
            gr = build_grounding("how does the scope gate decide", root=td)
            self.assertEqual(gr["attached"][0]["path"], "big_engine.py")

    def test_default_grounding_budget_fits_small_provider_limits(self):
        # Live failure 2026-10-26: default 60k grounding chars made the
        # default provider reject the ask outright (Groq 413 TPM 8000).
        # The shared default must stay small; swarm opts up explicitly.
        from omnicli.grounding import DEFAULT_MAX_CHARS, build_grounding

        self.assertLessEqual(DEFAULT_MAX_CHARS, 24_000)
        body = "class ScopeGate:\n    pass\n" + ("x = scope_gate_url\n" * 2000)
        index = {f"mod{i}.py": body for i in range(6)}
        gr = build_grounding(
            "how does the scope gate decide", root="/nonexistent-ws", index=index
        )
        attached_chars = sum(item["chars"] for item in gr["attached"])
        self.assertLessEqual(attached_chars, DEFAULT_MAX_CHARS)


class TestChatDeadFlagWarnings(_WorkspaceCase):
    def test_collaborate_warns_temperature_and_rotate_ignored(self):
        from unittest.mock import MagicMock as _MM

        from click.testing import CliRunner

        from omnicli.cli import main

        session = _MM()
        with patch("omnicli.cli.ChatSession", return_value=session):
            result = CliRunner().invoke(
                main,
                ["chat", "--collaborate", "--temperature", "1.5",
                 "--rotate-families"],
                input="",
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("ignored in collaborate mode", result.output)
