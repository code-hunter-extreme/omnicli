"""
Agentic execution loop with tool calling, multi-step reasoning, and cross-provider delegation.

Flags:
  redteam=True         — injects a red-team adversarial system prompt and wires the
                         deterministic verify tools (verify_http, verify_code,
                         verify_consensus) as the primary toolset the model is
                         instructed to use. Goal: find and prove issues, not describe them.

  rotate_families=True — after every completed turn the agent switches to the next
                         model family in the scanned roster. Consecutive turns use
                         different architectures. Good for: avoiding single-model blind
                         spots, cross-family corroboration without a full consensus call.

Auto-fallback: on API error / timeout the agent walks the roster regardless of flags.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from rich.console import Console
from rich.panel import Panel

from omnicli.client import OmniClient, ProviderError
from omnicli.config import ConfigManager
from omnicli.tools import TOOL_DEFINITIONS, execute_tool_call


# HTTP status codes that are permanent failures — no point retrying the same model.
_HARD_FAIL_CODES = {400, 401, 403, 404, 422}
# Status codes that justify trying the next model in the roster.
_RETRY_CODES = {429, 500, 502, 503, 504}


# Red-team system prompt — injected when --redteam is set.
# Instructs the model to act as an adversarial security analyst whose job is to
# find, prove, and document issues using the deterministic tools, not describe them.
_REDTEAM_SYSTEM = """\
You are an adversarial security analyst performing a red-team assessment.
Your job is to FIND and PROVE security issues — not describe them.

Rules:
1. Every claim MUST be verified with a tool before you state it.
   - Use verify_http to confirm wire behavior (status, body, headers).
   - Use verify_code to confirm code patterns or missing guards.
   - Use verify_consensus to cross-check a finding across model families before reporting.
2. Do not speculate. If you cannot prove it with a tool, say so and stop.
3. When you find something, produce:
   - CLAIM: one sentence stating the issue.
   - PROOF: the verify_http / verify_code / verify_consensus tool result.
   - IMPACT: what an attacker can do with this.
   - REPRO: the exact curl command to reproduce.
4. Use bash and read_file to gather evidence. Use consult_provider to get a second opinion.
5. Be thorough. Check auth, access control, information disclosure, and error handling.
"""

# Red-team tool definitions — the verify_* set only.
# When --redteam is active these replace the full TOOL_DEFINITIONS so the model
# is not distracted by bash/file_ops and stays focused on verification.
from omnicli.tools import TOOL_DEFINITIONS as _ALL_TOOLS

_REDTEAM_TOOLS = [t for t in _ALL_TOOLS if t["function"]["name"] in (
    "verify_http", "verify_code", "verify_consensus",
    "bash", "read_file", "consult_provider",
)]


class AgentRunner:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_turns: int = 10,
        console: Optional[Console] = None,
        redteam: bool = False,
        rotate_families: bool = False,
    ) -> None:
        self.cfg = ConfigManager()
        self.client = OmniClient(self.cfg)
        self.provider = (provider or self.cfg.get_default_provider()).lower()
        self.model = model or self.cfg.get_default_model(self.provider)
        self.max_turns = max_turns
        self.console = console or Console()
        self.redteam = redteam
        self.rotate_families = rotate_families

    def _model_roster(self) -> List[str]:
        """Return [primary] + fallback_models from config, deduped, primary first."""
        fallbacks = self.cfg.get_fallback_models(self.provider)
        seen = set()
        roster = []
        for m in [self.model] + fallbacks:
            if m and m not in seen:
                seen.add(m)
                roster.append(m)
        return roster

    @staticmethod
    def _next_family_index(current_idx: int, roster: List[str]) -> int:
        """Index of the next roster model whose family differs from the current one.

        Walks forward from current_idx + 1 with wraparound so families cycle
        indefinitely across turns. Returns current_idx unchanged when the roster
        holds only one family (or one model) — nothing to rotate to.
        """
        if len(roster) < 2:
            return current_idx
        current_family = _family_label(roster[current_idx])
        for step in range(1, len(roster)):
            cand = (current_idx + step) % len(roster)
            if _family_label(roster[cand]) != current_family:
                return cand
        return current_idx

    def _call_model(
        self,
        model: str,
        messages: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Single model call. Returns (data, None) on success or (None, error_msg).

        Caller decides whether error warrants a roster retry.
        """
        url = self.client.get_endpoint_url(self.provider, "chat/completions")
        headers = self.client.get_headers(self.provider)
        # In red-team mode expose only the verify_* set (+bash/read_file/consult)
        # so the model stays on proving findings, not writing files.
        tools = _REDTEAM_TOOLS if getattr(self, "redteam", False) else TOOL_DEFINITIONS
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        try:
            with httpx.Client(timeout=90.0) as http_client:
                res = http_client.post(url, headers=headers, json=payload)
                if res.status_code in _HARD_FAIL_CODES:
                    return None, f"hard_fail:{res.status_code}:{res.text[:120]}"
                if res.status_code in _RETRY_CODES:
                    return None, f"retry:{res.status_code}:{res.text[:120]}"
                if res.status_code != 200:
                    return None, f"error:{res.status_code}:{res.text[:120]}"
                return res.json(), None
        except httpx.TimeoutException:
            return None, "retry:timeout"
        except Exception as e:
            return None, f"retry:network:{e}"

    def run(self, goal: str, system_override: Optional[str] = None) -> str:
        """Run an autonomous agent loop with tools to achieve a user goal.

        On API error or stuck response, automatically tries the next model
        in the provider's fallback roster (built by `omni scan`).
        """
        if system_override:
            system_prompt = system_override
        elif getattr(self, "redteam", False):
            system_prompt = _REDTEAM_SYSTEM
        else:
            system_prompt = (
                "You are an expert autonomous AI agent equipped with local tools: "
                "'bash' (run commands), 'read_file', 'write_file', 'registry_search' "
                "(search ~6,400 local tools in Program Registry), and 'consult_provider' "
                "(ask NVIDIA NIM, XKiro, OpenRouter, Cloudflare, Bynara, or Apinex for "
                "specialized help). Always inspect and verify before answering. Use tools "
                "actively to gather truth."
            )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]

        roster = self._model_roster()
        active_model = roster[0]
        roster_idx = 0

        self.console.print(
            Panel(
                f"[bold cyan]OmniCLI Agent Goal:[/bold cyan] {goal}\n"
                f"[bold green]Supervisor:[/bold green] {self.provider} ({active_model})  |  "
                f"[dim]Roster: {len(roster)} model(s) available  |  "
                f"Tools: bash, file_ops, registry_search (6.4k tools), cross-provider[/dim]",
                title="Agent Active",
                border_style="cyan",
            )
        )

        turn = 0
        while turn < self.max_turns:
            turn += 1

            with self.console.status(
                f"[cyan]Turn {turn}: {self.provider}/{active_model}...[/cyan]",
                spinner="dots",
            ):
                data, err = self._call_model(active_model, messages)

            if err:
                is_retryable = err.startswith("retry:")
                self.console.print(
                    f"  [yellow]⚠ {self.provider}/{active_model}: {err}[/yellow]"
                )
                if is_retryable and roster_idx + 1 < len(roster):
                    roster_idx += 1
                    active_model = roster[roster_idx]
                    self.console.print(
                        f"  [cyan]↻ Falling back to {self.provider}/{active_model} "
                        f"(family: {_family_label(active_model)})[/cyan]"
                    )
                    turn -= 1  # don't burn a turn on a model swap
                    continue
                return f"Agent stopped: {err}"

            choices = data.get("choices", []) if data else []
            if not choices:
                return "No response choices returned by model."

            message = choices[0].get("message", {})
            messages.append(message)

            content = message.get("content")
            tool_calls = message.get("tool_calls", [])

            if content and not tool_calls:
                self.console.print("\n[bold green]Final Answer:[/bold green]")
                self.console.print(content)
                return content

            if tool_calls:
                for tc in tool_calls:
                    call_id = tc.get("id", f"call_{int(time.time()*1000)}")
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {}

                    arg_summary = ", ".join(f"{k}='{v}'" for k, v in list(args.items())[:3])
                    self.console.print(
                        f"  [bold yellow]⚡ Tool Call:[/bold yellow] "
                        f"[bold white]{name}[/bold white]({arg_summary})"
                    )

                    output = execute_tool_call(name, args, client_ref=self.client)

                    preview = output.strip().split("\n")[0][:120]
                    self.console.print(f"    [dim]↳ {preview}... ({len(output)} chars)[/dim]")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": output,
                    })

                # After a completed turn, optionally switch to the next model
                # family so consecutive turns use different architectures.
                if getattr(self, "rotate_families", False):
                    new_idx = self._next_family_index(roster_idx, roster)
                    if new_idx != roster_idx:
                        roster_idx = new_idx
                        active_model = roster[roster_idx]
                        self.console.print(
                            f"  [magenta]⟳ Rotating family → {self.provider}/{active_model} "
                            f"(family: {_family_label(active_model)})[/magenta]"
                        )
            else:
                if content:
                    self.console.print(content)
                    return content
                break

        return "Agent reached maximum turns limit."


def _family_label(model_id: str) -> str:
    """Quick family label for display — imports scanner lazily to avoid cycles."""
    try:
        from omnicli.scanner import classify_family
        return classify_family(model_id)
    except Exception:
        return "unknown"
