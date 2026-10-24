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

  ground_local=True    — (default) deterministically pre-grounds the goal in
                         relevance-ranked local workspace files before the
                         first model call (omnicli.grounding). CLI
                         --no-ground disables.

Auto-fallback: on API error / timeout the agent walks the roster regardless of flags.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from omnicli.client import OmniClient
from omnicli.config import ConfigManager
from omnicli.tools import TOOL_DEFINITIONS, execute_tool_call, resolve_workspace_root


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
   - Use analyze_python_security on Python files to check for direct input-to-SQL interpolation candidates; these are source-level leads, not proof of exploitability.
   - Use verify_code only for exact literal source-code patterns or supported built-in checks. Inspect the source first and use verbatim code fragments, not semantic labels, as patterns.
   - A REJECTED verify_code result means only that the submitted pattern/check was not satisfied. It does NOT prove the opposite security claim or prove that a vulnerability is absent.
   - A CONFIRMED pattern match proves only that the literal pattern exists (or the built-in check passed); explain any remaining assumptions before claiming exploitability.
   - Use verify_consensus to cross-check a finding across model families before reporting.
2. Do not speculate. If you cannot prove it with a tool, say so and stop.
3. When you find something, produce:
   - CLAIM: one sentence stating the issue.
   - PROOF: the verify_http / analyze_python_security / verify_code / verify_consensus tool result and how it supports this exact claim; analyzer candidates support only a source-level lead, not exploitability.
   - IMPACT: what an attacker can do with this.
   - REPRO: the exact curl command to reproduce.
4. Use find_files, search_code, list_directory, and read_file to explore the local project and gather evidence. Use bash only when enabled. Use consult_provider to get a second opinion.
5. Be thorough. Check auth, access control, information disclosure, and error handling.
6. Use verify_http only for operator-authorized hosts configured for this run. GET, HEAD, and OPTIONS are permitted by default; never use state-changing methods unless the operator explicitly enabled them.
"""

# Red-team tool definitions — bounded verification and read-only inspection tools.
# When --redteam is active these replace the full TOOL_DEFINITIONS so the model
# is not distracted by bash/file_ops and stays focused on verification.
from omnicli.tools import TOOL_DEFINITIONS as _ALL_TOOLS

_REDTEAM_TOOLS = [t for t in _ALL_TOOLS if t["function"]["name"] in (
    "verify_http", "verify_code", "analyze_python_security", "verify_consensus",
    "bash", "find_files", "search_code", "read_file", "consult_provider", "list_directory",
)]


def verification_toolset() -> List[Dict[str, Any]]:
    """The bounded verify toolset shared by the agent and tooled swarm members.

    Same list the red-team agent loop uses (verification, read-only inspection,
    and provider consultation — no write_file or registry_search).
    Returned as a fresh list so callers cannot mutate the shared set.
    """
    return list(_REDTEAM_TOOLS)


def call_model_with_tools(
    client: Any,
    provider: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    timeout: float = 90.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Single tool-calling model call. Returns (data, None) or (None, error).

    Shared transport for the sequential agent loop and for parallel tooled
    swarm members: identical payload shape, identical retry/hard-fail
    classification. Error prefixes are the contract — "retry:..." walks the
    roster, anything else stops.
    """
    url = client.get_endpoint_url(provider, "chat/completions")
    headers = client.get_headers(provider)
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    try:
        res = client.post_chat_completion(
            url=url,
            headers=headers,
            payload=payload,
            timeout=timeout,
        )
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


class AgentRunner:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_turns: int = 10,
        console: Optional[Console] = None,
        redteam: bool = False,
        rotate_families: bool = False,
        allow_shell: bool = False,
        allow_local: bool = False,
        allowed_hosts: Optional[List[str]] = None,
        allow_mutating_http: bool = False,
        ground_local: bool = True,
        workspace_root: Optional[Any] = None,
    ) -> None:
        self.cfg = ConfigManager()
        self.client = OmniClient(self.cfg)
        self.provider = (provider or self.cfg.get_default_provider()).lower()
        self.model = model or self.cfg.get_default_model(self.provider)
        self.max_turns = max_turns
        self.console = console or Console()
        self.redteam = redteam
        self.rotate_families = rotate_families
        self.allow_shell = allow_shell
        self.allow_local = allow_local
        self.allowed_hosts = list(allowed_hosts or [])
        self.allow_mutating_http = allow_mutating_http
        self.ground_local = ground_local
        self.grounding: Optional[Dict[str, Any]] = None
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve()
            if workspace_root else resolve_workspace_root()
        )

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

    def _build_roster(self) -> List[Tuple[str, str]]:
        """Roster of (provider, model) pairs the run walks for rotation/fallback.

        Default: the supervisor provider's own models. With rotate_families, a
        cross-provider, family-diverse roster spanning ALL configured providers,
        so rotation truly switches both family and API each turn.
        """
        if getattr(self, "rotate_families", False):
            try:
                from omnicli.orchestrate import build_swarm
                primary = (self.provider, self.model)
                pairs = [primary] + [p for p in build_swarm(self.cfg, size=12) if p != primary]
                seen: set = set()
                out: List[Tuple[str, str]] = []
                for pr in pairs:
                    if pr[1] and pr not in seen:
                        seen.add(pr)
                        out.append(pr)
                if out:
                    return out
            except Exception:
                pass
        return [(self.provider, m) for m in self._model_roster()]

    def _call_model(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Single model call on a specific provider. Returns (data, None) or (None, error).

        Caller decides whether error warrants a roster retry.
        """
        # In red-team mode expose only the verify_* set (+bash/read_file/consult)
        # so the model stays on proving findings, not writing files.
        tools = _REDTEAM_TOOLS if getattr(self, "redteam", False) else TOOL_DEFINITIONS
        return call_model_with_tools(
            self.client, provider, model, messages, tools
        )

    def run(self, goal: str, system_override: Optional[str] = None) -> str:
        """Run an autonomous agent loop with tools to achieve a user goal.

        On API error or stuck response, automatically tries the next model
        in the provider's fallback roster (built by `omni scan`).
        """
        configured_root = getattr(self, "workspace_root", None)
        self.workspace_root = (
            Path(configured_root).expanduser().resolve()
            if configured_root else resolve_workspace_root()
        )
        if system_override:
            system_prompt = system_override
        elif getattr(self, "redteam", False):
            system_prompt = _REDTEAM_SYSTEM
            allowed_hosts = getattr(self, "allowed_hosts", [])
            if allowed_hosts:
                system_prompt += "\n\nOperator-authorized HTTP hosts for this run: " + ", ".join(allowed_hosts)
            else:
                system_prompt += "\n\nNo HTTP hosts are authorized for this run; do not call verify_http."
            if getattr(self, "allow_mutating_http", False):
                system_prompt += "\nOperator explicitly enabled state-changing HTTP methods, limited to the authorized hosts."
            system_prompt += (
                f"\n\nLocal workspace root: {self.workspace_root}. "
                "Paths returned by local tools are workspace-relative."
            )
        else:
            system_prompt = (
                "You are an expert autonomous AI agent and local coding CLI "
                f"operating inside the local workspace {self.workspace_root}. "
                "Work on the actual local code, not from general knowledge. "
                "Local tools: 'find_files' and 'search_code' to locate code, "
                "'list_directory' and numbered 'read_file' ranges to inspect it, "
                "'edit_file' for targeted changes, 'write_file' for new files or "
                "complete rewrites, and 'bash' for commands/tests only when shell "
                "access is enabled. Paths returned by tools are workspace-relative. "
                "Inspect before editing, preserve unrelated changes, and verify with "
                "tool evidence. Never claim a test ran if bash is disabled or a tool "
                "result does not show it. Use 'registry_search' for the local Program "
                "Registry and 'consult_provider' for a second model opinion when useful. "
                "Cite local code as path:line and report files changed."
            )

        # Layer per-provider memory (context files) on top of the system prompt.
        try:
            from omnicli.memory import load_memory
            mem = load_memory(self.cfg, self.provider)
            if mem:
                system_prompt = f"{system_prompt}\n\n{mem}"
        except Exception:
            pass

        # Deterministic local pre-grounding: attach relevance-ranked
        # workspace files BEFORE the first model call, so the agent starts
        # from the operator's actual files (and cites path:line) instead
        # of answering from general knowledge without touching a tool.
        grounded_goal = goal
        self.grounding = None
        if getattr(self, "ground_local", True):
            try:
                from omnicli.grounding import build_grounding
                grounding = build_grounding(goal, root=self.workspace_root)
                if grounding.get("block"):
                    self.grounding = grounding
                    # Addendum rides in the user message: a caller's
                    # system_override stays byte-for-byte verbatim.
                    grounded_goal = (
                        f"{goal}\n\nLocal sources are attached below with "
                        "authoritative line numbers. Start from them, cite "
                        "path:line, and use search_code/find_files/read_file "
                        f"for anything beyond the attached lines.\n\n{grounding['block']}"
                    )
            except Exception:
                pass

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": grounded_goal},
        ]

        roster = self._build_roster()                  # list of (provider, model)
        models_only = [m for (_p, m) in roster]
        active_provider, active_model = roster[0]
        roster_idx = 0
        span = "cross-provider" if len({p for p, _ in roster}) > 1 else self.provider

        self.console.print(
            Panel(
                f"[bold cyan]OmniCLI Agent Goal:[/bold cyan] {goal}\n"
                f"[bold green]Supervisor:[/bold green] {active_provider} ({active_model})  |  "
                f"[bold green]Workspace:[/bold green] {escape(str(self.workspace_root))}\n"
                f"[dim]Roster: {len(roster)} model(s) across {span}  |  "
                f"Tools: find_files, search_code, read_file, edit_file, write_file, bash, registry_search, cross-provider  |  "
                f"Local grounding: {len(self.grounding['attached']) if self.grounding else 0} file(s) attached from workspace[/dim]",
                title="Agent Active",
                border_style="cyan",
            )
        )

        turn = 0
        while turn < self.max_turns:
            turn += 1

            with self.console.status(
                f"[cyan]Turn {turn}: {active_provider}/{active_model}...[/cyan]",
                spinner="dots",
            ):
                data, err = self._call_model(active_provider, active_model, messages)

            if err:
                is_retryable = err.startswith("retry:")
                self.console.print(
                    f"  [yellow]⚠ {active_provider}/{active_model}: {err}[/yellow]"
                )
                if is_retryable and roster_idx + 1 < len(roster):
                    roster_idx += 1
                    active_provider, active_model = roster[roster_idx]
                    self.console.print(
                        f"  [cyan]↻ Falling back to {active_provider}/{active_model} "
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
                self.console.print(escape(content))
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

                    arg_summary = escape(", ".join(f"{k}='{v}'" for k, v in list(args.items())[:3]))
                    self.console.print(
                        f"  [bold yellow]⚡ Tool Call:[/bold yellow] "
                        f"[bold white]{escape(name)}[/bold white]({arg_summary})"
                    )

                    output = execute_tool_call(
                        name, args, client_ref=self.client,
                        allow_shell=getattr(self, "allow_shell", False),
                        allow_local=getattr(self, "allow_local", False),
                        allowed_hosts=getattr(self, "allowed_hosts", None),
                        allow_mutating_http=getattr(self, "allow_mutating_http", False),
                        workspace_root=getattr(self, "workspace_root", None),
                    )

                    preview = escape(output.strip().split("\n")[0][:120])
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
                    new_idx = self._next_family_index(roster_idx, models_only)
                    if new_idx != roster_idx:
                        roster_idx = new_idx
                        active_provider, active_model = roster[roster_idx]
                        self.console.print(
                            f"  [magenta]⟳ Rotating family → {active_provider}/{active_model} "
                            f"(family: {_family_label(active_model)})[/magenta]"
                        )
            else:
                if content:
                    self.console.print(escape(content))
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
