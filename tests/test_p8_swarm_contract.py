#!/usr/bin/env python3
"""
P8 OmniCLI swarm contract tests — error surfacing and round-robin fill.

All tests are offline / no live API calls:
  - _one_call: fake clients raise or return canned payloads
  - build_swarm: synthetic provider rosters via MagicMock config

    python3 -m pytest tests/test_p8_swarm_contract.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestOneCallSurfacesFailures(unittest.TestCase):
    """A failing provider must surface: verbatim message, never a falsy error."""

    class BoomClient:
        def __init__(self, exc):
            self.exc = exc

        def chat_complete(self, **kwargs):
            raise self.exc

    class OkClient:
        def chat_complete(self, **kwargs):
            return {"choices": [{"message": {"content": "hello"}}]}

    class EmptyClient:
        def chat_complete(self, **kwargs):
            return {"choices": [{"message": {"content": "   "}}]}

    def _call(self, client):
        from omnicli.orchestrate import _one_call

        return _one_call(
            client, "pa", "openai/gpt-4o", [{"role": "user", "content": "hi"}], 64
        )

    def test_messaged_failure_passes_through_verbatim(self):
        rec = self._call(self.BoomClient(RuntimeError("down")))
        self.assertEqual(rec["answer"], "")
        self.assertEqual(rec["provider"], "pa")
        self.assertEqual(rec["model"], "openai/gpt-4o")
        self.assertEqual(rec["error"], "down")

    def test_messageless_failure_falls_back_to_type_name(self):
        # str(exc) == "" used to collapse error to "" (falsy → read as success).
        rec = self._call(self.BoomClient(RuntimeError()))
        self.assertEqual(rec["error"], "RuntimeError")

    def test_empty_response_surfaces_provider_message(self):
        rec = self._call(self.EmptyClient())
        self.assertEqual(rec["answer"], "")
        self.assertEqual(rec["error"], "Provider returned an empty response")

    def test_success_unchanged(self):
        rec = self._call(self.OkClient())
        self.assertIsNone(rec["error"])
        self.assertEqual(rec["answer"], "hello")


class TestBuildSwarmRoundRobin(unittest.TestCase):
    """Pass-2 must interleave providers instead of draining one roster first."""

    def _cfg(self):
        cfg = MagicMock()
        cfg.data = {"providers": {"pa": {}, "pb": {}}}
        defaults = {"pa": "openai/gpt-4o", "pb": "meta/llama-3.3-70b"}
        fallbacks = {
            "pa": ["openai/gpt-4o-mini", "openai/o1-preview"],
            "pb": ["meta/llama-3.2-90b", "meta/codellama-70b"],
        }
        cfg.get_default_model.side_effect = lambda p: defaults.get(p, "")
        cfg.get_fallback_models.side_effect = lambda p: fallbacks.get(p, [])
        return cfg

    def test_pass2_round_robins_across_providers(self):
        from omnicli.orchestrate import build_swarm

        swarm = build_swarm(self._cfg(), size=5)
        self.assertEqual(
            swarm,
            [
                ("pa", "openai/gpt-4o"),
                ("pb", "meta/llama-3.3-70b"),
                ("pa", "openai/gpt-4o-mini"),
                ("pb", "meta/llama-3.2-90b"),
                ("pa", "openai/o1-preview"),
            ],
        )

    def test_oversize_returns_everything_without_dupes(self):
        from omnicli.orchestrate import build_swarm

        swarm = build_swarm(self._cfg(), size=99)
        self.assertEqual(len(swarm), 6)
        self.assertEqual(len(set(swarm)), 6)
        self.assertEqual(
            [provider for provider, _ in swarm],
            ["pa", "pb", "pa", "pb", "pa", "pb"],
        )

    def test_uneven_rosters_fill_without_hanging(self):
        from omnicli.orchestrate import build_swarm

        cfg = MagicMock()
        cfg.data = {"providers": {"pa": {}, "pb": {}}}
        cfg.get_default_model.side_effect = lambda p: {
            "pa": "openai/gpt-4o",
            "pb": "meta/llama-3.3-70b",
        }[p]
        cfg.get_fallback_models.side_effect = lambda p: {
            "pa": [],
            "pb": ["meta/llama-3.2-90b", "meta/codellama-70b"],
        }[p]
        swarm = build_swarm(cfg, size=9)
        self.assertEqual(len(swarm), 4)
        self.assertEqual(len(set(swarm)), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
