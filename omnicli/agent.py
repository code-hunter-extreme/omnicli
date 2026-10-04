"""
Agentic execution loop with tool calling, multi-step reasoning, and cross-provider delegation.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from omnicli.client import OmniClient, ProviderError
from omnicli.config import ConfigManager
from omnicli.tools import TOOL_DEFINITIONS, execute_tool_call


class AgentRunner:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_turns: int = 10,
        console: Optional[Console] = None,
    ) -> None:
        self.cfg = ConfigManager()
        self.client = OmniClient(self.cfg)
        self.provider = (provider or self.cfg.get_default_provider()).lower()
        self.model = model or self.cfg.get_default_model(self.provider)
        self.max_turns = max_turns
        self.console = console or Console()

    def run(self, goal: str, system_override: Optional[str] = None) -> str:
        """Run an autonomous agent loop with tools to achieve a user goal."""
        system_prompt = system_override or (
            "You are an expert autonomous AI agent equipped with local tools: "
            "'bash' (run commands), 'read_file', 'write_file', 'registry_search' (search ~6,400 local tools in Program Registry), "
            "and 'consult_provider' (ask NVIDIA NIM, XKiro, OpenRouter, Cloudflare, Bynara, or Apinex for specialized help). "
            "Always inspect and verify before answering. Use tools actively to gather truth."
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]

        self.console.print(
            Panel(
                f"[bold cyan]OmniCLI Agent Goal:[/bold cyan] {goal}\n"
                f"[bold green]Supervisor:[/bold green] {self.provider} ({self.model})  |  "
                f"[dim]Tools: bash, file_ops, registry_search (6.4k tools), cross-provider consultation[/dim]",
                title="Agent Active",
                border_style="cyan",
            )
        )

        turn = 0
        while turn < self.max_turns:
            turn += 1
            url = self.client.get_endpoint_url(self.provider, "chat/completions")
            headers = self.client.get_headers(self.provider)

            payload = {
                "model": self.model,
                "messages": messages,
                "tools": TOOL_DEFINITIONS,
                "tool_choice": "auto",
                "temperature": 0.2,
                "max_tokens": 2048,
            }

            with self.console.status(f"[cyan]Turn {turn}: Thinking with {self.provider}/{self.model}...[/cyan]", spinner="dots"):
                try:
                    with httpx.Client(timeout=90.0) as http_client:
                        res = http_client.post(url, headers=headers, json=payload)
                        if res.status_code != 200:
                            err_msg = f"HTTP {res.status_code}: {res.text[:300]}"
                            self.console.print(f"[bold red]API Error from {self.provider}:[/bold red] {err_msg}")
                            return f"Agent stopped due to provider error: {err_msg}"
                        data = res.json()
                except Exception as e:
                    self.console.print(f"[bold red]Network/Client Error:[/bold red] {e}")
                    return f"Agent connection failed: {e}"

            choices = data.get("choices", [])
            if not choices:
                return "No response choices returned by model."

            message = choices[0].get("message", {})
            messages.append(message)

            content = message.get("content")
            tool_calls = message.get("tool_calls", [])

            if content and not tool_calls:
                # Agent completed work
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

                    # Pretty print the tool call
                    arg_summary = ", ".join(f"{k}='{v}'" for k, v in list(args.items())[:3])
                    self.console.print(f"  [bold yellow]⚡ Tool Call:[/bold yellow] [bold white]{name}[/bold white]({arg_summary})")

                    # Execute the tool
                    output = execute_tool_call(name, args, client_ref=self.client)

                    # Print short preview of output
                    preview = output.strip().split("\n")[0][:120]
                    self.console.print(f"    [dim]↳ Result: {preview}... ({len(output)} chars)[/dim]")

                    # Append tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": output,
                    })
            else:
                if content:
                    self.console.print(content)
                    return content
                break

        return "Agent reached maximum turns limit."
