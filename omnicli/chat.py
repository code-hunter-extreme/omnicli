"""
Interactive REPL Chat Session powered by Rich and Prompt Toolkit.
Provides real-time Markdown streaming, live slash commands, model auto-complete,
and multi-turn memory.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from omnicli.client import OmniClient, ProviderError
from omnicli.config import ConfigManager


class SlashCommandCompleter(Completer):
    """Auto-completion for slash commands, providers, and models."""

    def __init__(self, config_manager: ConfigManager, client: OmniClient):
        self.cfg = config_manager
        self.client = client
        self.commands = [
            ("/provider", "Switch active LLM provider"),
            ("/model", "Switch active model"),
            ("/models", "List or search models from active provider"),
            ("/providers", "List all configured providers"),
            ("/system", "View or set system prompt"),
            ("/temp", "Set generation temperature (0.0 - 2.0)"),
            ("/file", "Attach a file to chat context"),
            ("/save", "Save conversation to a Markdown file"),
            ("/clear", "Clear chat history"),
            ("/ping", "Test latency to all providers"),
            ("/help", "Show available commands"),
            ("/exit", "Exit the chat session"),
        ]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        words = text.split()

        if not text.startswith("/"):
            return

        if len(words) <= 1 and not text.endswith(" "):
            # Complete the command itself
            for cmd, desc in self.commands:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text), display_meta=desc)
            return

        cmd = words[0]
        arg = words[1] if len(words) > 1 else ""

        if cmd in ("/provider", "/p"):
            providers = list(self.cfg.data.get("providers", {}).keys())
            for p in providers:
                if p.startswith(arg):
                    yield Completion(p, start_position=-len(arg), display_meta=f"Switch to {p}")

        elif cmd in ("/model", "/m"):
            # Can suggest recent or default models
            provider = self.cfg.get_default_provider()
            default_m = self.cfg.get_default_model(provider)
            if default_m and default_m.startswith(arg):
                yield Completion(default_m, start_position=-len(arg), display_meta=f"Default {provider}")


class ChatSession:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> None:
        self.cfg = ConfigManager()
        self.client = OmniClient(self.cfg)
        self.console = Console()

        self.provider = (provider or self.cfg.get_default_provider()).lower()
        self.model = model or self.cfg.get_default_model(self.provider)
        self.system_prompt = system_prompt or self.cfg.data.get(
            "system_prompt", "You are a helpful, brilliant, concise AI assistant."
        )
        self.temperature = temperature
        self.messages: List[Dict[str, str]] = []

        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

        completer = SlashCommandCompleter(self.cfg, self.client)
        self.prompt_session = PromptSession(
            history=FileHistory(str(self.cfg.history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
        )

    def print_welcome(self) -> None:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_row("[bold cyan]OmniCLI — Local Multi-Provider AI Terminal[/bold cyan]")
        grid.add_row(f"[bold green]Provider:[/bold green] {self.provider}  |  [bold green]Model:[/bold green] {self.model}  |  [bold green]Temp:[/bold green] {self.temperature}")
        grid.add_row("[dim]Type /help for slash commands, /models to browse models, /clear to reset history.[/dim]\n")
        self.console.print(Panel(grid, border_style="cyan", padding=(0, 1)))

    def show_help(self) -> None:
        table = Table(title="OmniCLI Chat Commands", title_style="bold cyan", border_style="dim")
        table.add_column("Command", style="bold green", width=22)
        table.add_column("Description", style="white")

        table.add_row("/provider, /p <name>", "Switch provider (openrouter, nvidia, cloudflare, bynara, apinex, xkiro)")
        table.add_row("/model, /m <name>", "Switch model on current provider")
        table.add_row("/models [query]", "List / search models available on active provider")
        table.add_row("/providers", "List all configured providers and key statuses")
        table.add_row("/system, /s [prompt]", "Show or change the active system prompt")
        table.add_row("/temp <0.0-2.0>", "Adjust sampling temperature")
        table.add_row("/file <path>", "Attach file content to the next prompt")
        table.add_row("/clear", "Clear chat message history")
        table.add_row("/save [path]", "Save conversation transcript to a markdown file")
        table.add_row("/ping", "Test latency and health across all endpoints")
        table.add_row("/help", "Display this help menu")
        table.add_row("/exit, /quit", "Exit the chat session")

        self.console.print(table)

    def show_providers(self) -> None:
        table = Table(title="Configured Providers", title_style="bold cyan", border_style="dim")
        table.add_column("ID", style="bold green", width=12)
        table.add_column("Name", style="white", width=24)
        table.add_column("Base URL", style="cyan")
        table.add_column("Auth Key", style="yellow", width=14)
        table.add_column("Default Model", style="magenta")

        providers = self.cfg.data.get("providers", {})
        active = self.provider

        for pid, pdata in providers.items():
            key = self.cfg.get_api_key(pid)
            masked_key = f"{key[:4]}...{key[-3:]}" if key and len(key) > 7 else ("Set" if key else "[dim]Missing[/dim]")
            prefix = "[bold green]▶ [/bold green]" if pid == active else "  "
            base_url = self.cfg.get_base_url(pid)
            def_m = pdata.get("default_model", "none")
            table.add_row(f"{prefix}{pid}", pdata.get("name", pid), base_url, masked_key, def_m)

        self.console.print(table)

    def show_models(self, query: str = "") -> None:
        with self.console.status(f"[cyan]Fetching models from {self.provider}...[/cyan]", spinner="dots"):
            try:
                models = self.client.list_models(self.provider)
            except Exception as e:
                self.console.print(f"[bold red]Failed to fetch models:[/bold red] {e}")
                return

        if not models:
            self.console.print(f"[yellow]No models found or returned by {self.provider}.[/yellow]")
            return

        # Filter by search query if provided
        filtered = []
        q = query.lower().strip()
        for m in models:
            mid = m.get("id", "")
            if not q or q in mid.lower() or q in m.get("name", "").lower():
                filtered.append(m)

        table = Table(
            title=f"Models on {self.provider} ({len(filtered)}/{len(models)})",
            title_style="bold cyan",
            border_style="dim",
        )
        table.add_column("Model ID", style="bold green")
        table.add_column("Context", style="yellow", width=12)
        table.add_column("Pricing (in/out)", style="cyan", width=20)

        # Show first 40 if many
        limit = 40
        for m in filtered[:limit]:
            mid = m.get("id", "")
            ctx = str(m.get("context_length", m.get("context_window", "-")))
            pricing_data = m.get("pricing")
            price_str = "-"
            if isinstance(pricing_data, dict):
                inp = pricing_data.get("input", pricing_data.get("prompt", "-"))
                out = pricing_data.get("output", pricing_data.get("completion", "-"))
                price_str = f"${inp} / ${out}"
            elif "price" in m:
                price_str = str(m["price"])

            marker = " [bold cyan](current)[/bold cyan]" if mid == self.model else ""
            table.add_row(f"{mid}{marker}", ctx, price_str)

        self.console.print(table)
        if len(filtered) > limit:
            self.console.print(f"[dim]...showing first {limit} of {len(filtered)} models. Use '/models <filter>' to search.[/dim]")

    def handle_slash_command(self, text: str) -> bool:
        """Handle slash command. Returns True to continue prompt loop, False on exit."""
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit", "/q"):
            self.console.print("[dim]Goodbye![/dim]")
            return False

        elif cmd in ("/help", "/h", "/?"):
            self.show_help()

        elif cmd in ("/providers", "/p_list"):
            self.show_providers()

        elif cmd in ("/provider", "/p"):
            if not arg:
                self.console.print(f"Current provider: [bold green]{self.provider}[/bold green]")
                self.show_providers()
            else:
                prov = arg.lower()
                if prov not in self.cfg.data.get("providers", {}):
                    self.console.print(f"[bold red]Unknown provider:[/bold red] '{prov}'. Run /providers to view options.")
                else:
                    self.provider = prov
                    self.model = self.cfg.get_default_model(prov)
                    self.console.print(f"[bold green]Switched to provider:[/bold green] {self.provider} (model: {self.model})")

        elif cmd in ("/model", "/m"):
            if not arg:
                self.console.print(f"Current model: [bold green]{self.model}[/bold green] on [cyan]{self.provider}[/cyan]")
            else:
                self.model = arg
                self.console.print(f"[bold green]Model set to:[/bold green] {self.model}")

        elif cmd in ("/models", "/catalog"):
            self.show_models(arg)

        elif cmd in ("/system", "/s"):
            if not arg:
                self.console.print(Panel(self.system_prompt, title="System Prompt", border_style="cyan"))
            else:
                self.system_prompt = arg
                # Update first message if system role
                if self.messages and self.messages[0]["role"] == "system":
                    self.messages[0]["content"] = self.system_prompt
                else:
                    self.messages.insert(0, {"role": "system", "content": self.system_prompt})
                self.console.print("[bold green]System prompt updated.[/bold green]")

        elif cmd in ("/temp", "/temperature"):
            if not arg:
                self.console.print(f"Current temperature: [bold green]{self.temperature}[/bold green]")
            else:
                try:
                    val = float(arg)
                    if 0.0 <= val <= 2.0:
                        self.temperature = val
                        self.console.print(f"[bold green]Temperature set to {self.temperature}[/bold green]")
                    else:
                        self.console.print("[red]Temperature must be between 0.0 and 2.0[/red]")
                except ValueError:
                    self.console.print("[red]Invalid float number for temperature[/red]")

        elif cmd == "/clear":
            self.messages = []
            if self.system_prompt:
                self.messages.append({"role": "system", "content": self.system_prompt})
            self.console.print("[bold yellow]Chat history cleared.[/bold yellow]")

        elif cmd == "/ping":
            self.run_ping_all()

        elif cmd == "/save":
            path = arg or f"chat_transcript_{int(time.time())}.md"
            self.save_transcript(path)

        elif cmd == "/file":
            if not arg:
                self.console.print("[red]Usage: /file <path/to/file>[/red]")
            else:
                fpath = Path(arg).expanduser()
                if not fpath.exists() or not fpath.is_file():
                    self.console.print(f"[red]File not found: {fpath}[/red]")
                else:
                    try:
                        content = fpath.read_text(encoding="utf-8", errors="replace")
                        formatted = f"\n\n[File Content: {fpath.name}]\n```\n{content}\n```\n"
                        self.messages.append({"role": "user", "content": formatted})
                        self.console.print(f"[bold green]Loaded file into context:[/bold green] {fpath.name} ({len(content)} chars)")
                    except Exception as e:
                        self.console.print(f"[red]Error reading file: {e}[/red]")

        else:
            self.console.print(f"[yellow]Unknown command:[/yellow] {cmd}. Type /help for assistance.")

        return True

    def run_ping_all(self) -> None:
        table = Table(title="Provider Diagnostics & Health", title_style="bold cyan", border_style="dim")
        table.add_column("Provider", style="bold green", width=14)
        table.add_column("Status", width=16)
        table.add_column("Latency", justify="right", width=12)
        table.add_column("Models", justify="right", width=10)
        table.add_column("Details", style="dim")

        providers = list(self.cfg.data.get("providers", {}).keys())

        with self.console.status("[cyan]Pinging all providers...[/cyan]", spinner="dots"):
            for pid in providers:
                res = self.client.ping(pid)
                st = res["status"]
                if st == "healthy":
                    status_str = "[bold green]ONLINE (200)[/bold green]"
                elif st in ("unauthorized", "missing_key"):
                    status_str = "[yellow]AUTH NEEDED[/yellow]"
                else:
                    status_str = f"[red]{st.upper()}[/red]"

                latency_str = f"{res['latency_ms']} ms" if res["latency_ms"] else "-"
                model_str = str(res["model_count"]) if res["model_count"] else "-"
                details = res["error"] or "OK"
                table.add_row(pid, status_str, latency_str, model_str, details[:45])

        self.console.print(table)

    def save_transcript(self, filepath: str) -> None:
        p = Path(filepath).expanduser()
        lines = [f"# OmniCLI Chat Transcript\n- **Provider:** {self.provider}\n- **Model:** {self.model}\n\n---\n"]
        for msg in self.messages:
            role = msg["role"].capitalize()
            lines.append(f"### {role}\n\n{msg['content']}\n")

        p.write_text("\n".join(lines), encoding="utf-8")
        self.console.print(f"[bold green]Transcript saved to:[/bold green] {p.resolve()}")

    def stream_turn(self, user_prompt: str) -> None:
        self.messages.append({"role": "user", "content": user_prompt})

        full_content = ""
        t0 = time.perf_counter()
        first_token_time = None
        token_count = 0

        self.console.print()  # Empty line

        # Use Live markdown rendering
        with Live(
            Text("...", style="dim cyan"),
            console=self.console,
            refresh_per_second=15,
            auto_refresh=True,
        ) as live:
            for chunk in self.client.chat_stream(
                provider_id=self.provider,
                messages=self.messages,
                model=self.model,
                temperature=self.temperature,
            ):
                if chunk["type"] == "content":
                    delta = chunk["delta"]
                    if not first_token_time:
                        first_token_time = time.perf_counter()
                    full_content += delta
                    token_count += 1
                    # Render updated markdown
                    live.update(Markdown(full_content))
                elif chunk["type"] == "error":
                    live.update(Text(chunk["content"], style="bold red"))
                    self.messages.pop()  # discard failed user message
                    return

        total_time = time.perf_counter() - t0
        tps = round(token_count / total_time, 1) if total_time > 0 and token_count else 0
        ttft = round((first_token_time - t0) * 1000, 0) if first_token_time else 0

        # Stats bar
        stats = Text()
        stats.append(f"⚡ {self.provider}/{self.model}", style="bold cyan")
        stats.append(f"  •  {round(total_time, 2)}s", style="dim")
        if ttft:
            stats.append(f" (TTFT: {int(ttft)}ms)", style="dim")
        if tps:
            stats.append(f"  •  ~{tps} tok/s", style="dim green")
        self.console.print(stats)
        self.console.print()

        self.messages.append({"role": "assistant", "content": full_content})

    def run(self) -> None:
        self.print_welcome()
        while True:
            try:
                prompt_label = [
                    ("class:provider", f"{self.provider}"),
                    ("class:colon", ":"),
                    ("class:model", f"{self.model.split('/')[-1]}"),
                    ("class:arrow", " ❯ "),
                ]
                text = self.prompt_session.prompt(prompt_label)
                if not text or not text.strip():
                    continue

                clean = text.strip()
                if clean.startswith("/"):
                    cont = self.handle_slash_command(clean)
                    if not cont:
                        break
                    continue

                self.stream_turn(clean)

            except KeyboardInterrupt:
                self.console.print("\n[dim](Cancelled. Type /exit to quit)[/dim]")
                continue
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[bold red]Unexpected error:[/bold red] {e}")
