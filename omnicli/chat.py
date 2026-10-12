"""
Interactive REPL Chat Session powered by Rich and Prompt Toolkit.
Provides real-time Markdown streaming, live slash commands, model auto-complete,
and multi-turn memory.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from omnicli.client import OmniClient
from omnicli.config import ConfigManager

chat_style = Style.from_dict({
    "provider": "ansicyan bold",
    "colon": "ansigray",
    "model": "ansigreen",
    "mode": "ansimagenta bold",
    "arrow": "ansicyan bold",
})


# Adversarial persona for `omni chat --redteam`. Chat has no tool execution, so
# unlike the agent's red-team prompt this does NOT promise verify_* tools — it
# just makes the model a rigorous, proof-demanding security analyst.
_CHAT_REDTEAM_SYSTEM = (
    "You are an adversarial security analyst in an interactive session. "
    "Be rigorous and skeptical: challenge assumptions and probe for auth, "
    "access-control, injection, SSRF, and information-disclosure issues. For "
    "every claim, separate what is PROVEN from what is merely plausible and "
    "state exactly what evidence (a request, a code path, a response) would "
    "confirm or refute it. Never fabricate results."
)


class SlashCommandCompleter(Completer):
    """Auto-completion for slash commands, providers, and models."""

    def __init__(self, config_manager: ConfigManager, client: OmniClient):
        self.cfg = config_manager
        self.client = client
        self.commands = [
            ("/mode", "Switch mode: collaborate, redteam, single"),
            ("/collaborate", "Collaborative council mode (all models reason together)"),
            ("/redteam", "Adversarial red-team council mode"),
            ("/single", "Single-model direct chat mode"),
            ("/size", "Set council size for swarm orchestration"),
            ("/provider", "Switch active LLM provider"),
            ("/model", "Switch active model"),
            ("/models", "List or search models from active provider"),
            ("/providers", "List all configured providers"),
            ("/system", "View or set system prompt"),
            ("/temp", "Set generation temperature (0.0 - 2.0)"),
            ("/file", "Attach a file to chat context"),
            ("/save", "Save conversation to a Markdown file"),
            ("/clear", "Clear chat history"),
            ("/forget", "Delete persisted history for the active provider"),
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

        if cmd == "/mode":
            for m in ("collaborate", "redteam", "single"):
                if m.startswith(arg.lower()):
                    yield Completion(m, start_position=-len(arg), display_meta=f"Set mode: {m}")

        elif cmd in ("/provider", "/p"):
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
        redteam: bool = False,
        rotate_families: bool = False,
        mode: str = "collaborate",
        council_size: Optional[int] = None,
    ) -> None:
        self.cfg = ConfigManager()
        self.client = OmniClient(self.cfg)
        self.console = Console()

        self.provider = (provider or self.cfg.get_default_provider()).lower()
        self.model = model or self.cfg.get_default_model(self.provider)
        self.redteam = redteam
        self.rotate_families = rotate_families
        self.mode = "redteam" if redteam else mode.lower().strip()
        self.council_size = council_size
        if system_prompt:
            self.system_prompt = system_prompt
        elif self.mode == "redteam":
            self.system_prompt = _CHAT_REDTEAM_SYSTEM
        else:
            self.system_prompt = self.cfg.data.get(
                "system_prompt", "You are a helpful, brilliant, concise AI assistant."
            )
        self.temperature = temperature
        # Keep the operator's prompt separate so per-provider memory can be
        # re-layered on top whenever the active provider changes.
        self.base_system_prompt = self.system_prompt
        self.messages: List[Dict[str, str]] = []
        self.pending_files: Dict[str, str] = {}
        self.last_report_path: Optional[str] = None

        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        self._apply_memory()
        self._load_history_into_context()

        completer = SlashCommandCompleter(self.cfg, self.client)
        self.prompt_session = PromptSession(
            history=FileHistory(str(self.cfg.history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            style=chat_style,
        )

    def print_welcome(self) -> None:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_row("[bold cyan]OmniCLI — Local Multi-Provider AI Terminal & Council[/bold cyan]")
        grid.add_row(f"[bold green]Lead Provider:[/bold green] {self.provider}  |  [bold green]Model:[/bold green] {self.model}  |  [bold green]Temp:[/bold green] {self.temperature}")
        mode_style = {
            "collaborate": "[bold cyan]COLLABORATE (Multi-Model Council)[/bold cyan]",
            "redteam": "[bold red]RED-TEAM (Adversarial Audit)[/bold red]",
            "single": "[dim]SINGLE-MODEL (Direct Chat)[/dim]",
        }.get(self.mode, f"[bold cyan]{self.mode.upper()}[/bold cyan]")
        mode_line = f"[bold green]Active Mode:[/bold green] {mode_style}"
        if self.mode != "single":
            sz_str = str(self.council_size) if self.council_size else "all active providers"
            mode_line += f"  |  [bold green]Council Size:[/bold green] {sz_str}"
        if getattr(self, "rotate_families", False):
            mode_line += "  |  [bold magenta]ROTATE-FAMILIES[/bold magenta]"
        grid.add_row(mode_line)
        try:
            from omnicli.memory import memory_file_count
            mcount = memory_file_count(self.cfg, self.provider)
        except Exception:
            mcount = 0
        if mcount:
            grid.add_row(f"[bold green]Memory:[/bold green] {mcount} file(s) for {self.provider}")
        grid.add_row("[dim]Type /mode, /collaborate, /redteam, /single to switch modes. Type /help for all commands.[/dim]\n")
        self.console.print(Panel(grid, border_style="cyan", padding=(0, 1)))

    def show_help(self) -> None:
        table = Table(title="OmniCLI Chat Commands", title_style="bold cyan", border_style="dim")
        table.add_column("Command", style="bold green", width=24)
        table.add_column("Description", style="white")

        table.add_row("/mode [mode]", "Switch mode (collaborate, redteam, single)")
        table.add_row("/collaborate, /c", "Collaborative council (all models reason together)")
        table.add_row("/redteam, /rt", "Adversarial red-team council (vulnerability focus)")
        table.add_row("/single", "Direct 1-on-1 single-model chat")
        table.add_row("/size <1-10>", "Set council size for swarm orchestration")
        table.add_row("/accept", "Accept audit recommendations / remediations")
        table.add_row("/provider, /p <name>", "Switch lead provider (groq, openrouter, nvidia, etc.)")
        table.add_row("/model, /m <name>", "Switch model on current provider")
        table.add_row("/models [query]", "List / search models available on active provider")
        table.add_row("/providers", "List all configured providers and key statuses")
        table.add_row("/system, /s [prompt]", "Show or change the active system prompt")
        table.add_row("/temp <0.0-2.0>", "Adjust sampling temperature")
        table.add_row("/file <path>", "Attach file content to the next prompt")
        table.add_row("/clear", "Clear chat message history (in-session only)")
        table.add_row("/forget", "Delete persisted history file for the active provider")
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

        elif cmd in ("/mode",):
            if not arg:
                self.console.print(f"Current mode: [bold cyan]{self.mode}[/bold cyan] (options: collaborate, redteam, single)")
            else:
                m = arg.lower()
                if m in ("collaborate", "redteam", "single"):
                    self.mode = m
                    self.console.print(f"[bold green]Mode switched to:[/bold green] [bold cyan]{self.mode}[/bold cyan]")
                else:
                    self.console.print(f"[red]Unknown mode '{arg}'. Choose from: collaborate, redteam, single[/red]")

        elif cmd in ("/collaborate", "/collab", "/c"):
            self.mode = "collaborate"
            self.console.print("[bold green]Mode switched to:[/bold green] [bold cyan]collaborate[/bold cyan] (all models reason together)")

        elif cmd in ("/builder", "/build", "/b"):
            self.console.print("[red]Builder mode was removed: it never built or tested anything. Use collaborate, redteam, or single.[/red]")

        elif cmd in ("/redteam", "/rt"):
            self.mode = "redteam"
            self.console.print("[bold green]Mode switched to:[/bold green] [bold red]redteam[/bold red] (adversarial vulnerability review)")

        elif cmd in ("/single",):
            self.mode = "single"
            self.console.print(f"[bold green]Mode switched to:[/bold green] [dim]single[/dim] (direct 1-on-1 with {self.provider}/{self.model})")

        elif cmd in ("/size", "/council_size"):
            if not arg:
                self.console.print(f"Current council size: [bold cyan]{self.council_size}[/bold cyan]")
            else:
                try:
                    sz = int(arg)
                    if 1 <= sz <= 10:
                        self.council_size = sz
                        self.console.print(f"[bold green]Council size set to {self.council_size}[/bold green]")
                    else:
                        self.console.print("[red]Council size must be between 1 and 10[/red]")
                except ValueError:
                    self.console.print("[red]Invalid integer for council size[/red]")

        elif cmd in ("/accept", "/apply"):
            if getattr(self, "last_report_path", None):
                self.console.print(f"[bold green]✓ Remediations from {self.last_report_path} accepted.[/bold green]")
                self.console.print("[dim]You can now use 'omni agent' or local editing tools to apply the changes.[/dim]")
            else:
                self.console.print("[yellow]No recent audit report or remediation to accept.[/yellow]")

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
                    self._apply_memory()
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
                self.base_system_prompt = arg
                self._apply_memory()
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
            self.pending_files = {}
            if self.system_prompt:
                self.messages.append({"role": "system", "content": self.system_prompt})
            self.console.print("[bold yellow]Chat history cleared.[/bold yellow]")

        elif cmd == "/forget":
            from omnicli.memory import history_path
            p = history_path(self.cfg, self.provider)
            try:
                if p.is_file():
                    p.unlink()
                self.console.print(f"[bold yellow]Forgot persisted history for {self.provider}.[/bold yellow]")
            except Exception as e:
                self.console.print(f"[red]Could not clear history: {e}[/red]")

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
                        pending_files = getattr(self, "pending_files", {})
                        total_chars = sum(
                            len(existing)
                            for existing_path, existing in pending_files.items()
                            if existing_path != str(fpath)
                        ) + len(content)
                        if total_chars > 200_000:
                            self.console.print(
                                "[red]Attached source files exceed the 200,000-character limit.[/red]"
                            )
                        else:
                            pending_files[str(fpath)] = content
                            self.pending_files = pending_files
                            self.console.print(
                                f"[bold green]Loaded file for the next prompt:[/bold green] "
                                f"{fpath} ({len(content):,} chars)"
                            )
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

    def _load_history_into_context(self) -> None:
        """Resume persisted per-provider history (opt-in) after the system message."""
        try:
            from omnicli.memory import load_history
            hist = load_history(self.cfg, self.provider)
        except Exception:
            hist = []
        if hist:
            self.messages.extend(hist)
            self.console.print(
                f"[dim]Resumed {len(hist)} message(s) of history for {self.provider}.[/dim]"
            )

    def _apply_memory(self) -> None:
        """Set the system message to base prompt + current provider's memory."""
        try:
            from omnicli.memory import load_memory
            mem = load_memory(self.cfg, self.provider)
        except Exception:
            mem = ""
        content = self.base_system_prompt or ""
        if mem:
            content = f"{content}\n\n{mem}".strip()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = content
        elif content:
            self.messages.insert(0, {"role": "system", "content": content})

    def _next_family_model(self) -> Optional[str]:
        """Next model in the scanned roster whose family differs from the current.

        Returns None when the roster holds only one family (nothing to rotate to).
        """
        from omnicli.scanner import classify_family
        roster = [self.model] + list(self.cfg.get_fallback_models(self.provider))
        seen: set = set()
        deduped = []
        for m in roster:
            if m and m not in seen:
                seen.add(m)
                deduped.append(m)
        if len(deduped) < 2:
            return None
        current_family = classify_family(self.model)
        for m in deduped[1:]:
            if classify_family(m) != current_family:
                return m
        return None

    def stream_turn(self, user_prompt: str) -> None:
        self._apply_memory()  # refresh memory for the active provider
        pending_files = dict(getattr(self, "pending_files", {}))
        orchestration_prompt = user_prompt
        source_files = {}
        if self.mode == "redteam":
            from omnicli.evidence import extract_sources_from_prompt

            source_files = {
                **extract_sources_from_prompt(user_prompt),
                **pending_files,
            }
            if sum(len(content) for content in source_files.values()) > 200_000:
                self.console.print(
                    "[red]Red-team source files exceed the 200,000-character limit.[/red]"
                )
                return
        self.pending_files = {}
        if pending_files and self.mode != "redteam":
            orchestration_prompt += "\n\n" + "\n\n".join(
                f"### Source file: {path}\n```text\n{content}\n```"
                for path, content in pending_files.items()
            )
        self.messages.append({"role": "user", "content": orchestration_prompt})

        if self.mode in ("collaborate", "redteam"):
            from omnicli.orchestrate import run_orchestration
            from rich.markup import escape
            if self.mode == "redteam" and source_files:
                disclosed_files = "\n".join(
                    f"- {escape(path)} ({len(content):,} characters)"
                    for path, content in source_files.items()
                )
                self.console.print(Panel(
                    "These explicitly supplied files will be sent to the "
                    "configured model providers:\n" + disclosed_files,
                    title="Red-Team Source Disclosure",
                    border_style="yellow",
                ))
            t0 = time.perf_counter()
            with self.console.status(
                f"[bold cyan]Council ({self.mode.upper()}) reasoning across {self.council_size} models in parallel...[/bold cyan]",
                spinner="dots",
            ):
                out = run_orchestration(
                    mode=self.mode,
                    cfg=self.cfg,
                    client=self.client,
                    prompt=orchestration_prompt if self.mode != "redteam" else user_prompt,
                    size=self.council_size,
                    synth=True,
                    judge=(self.provider, self.model),
                    source_files=source_files,
                )
            total_time = time.perf_counter() - t0

            # Display member perspectives in compact panels
            for r in out.get("members", []):
                role_tag = f" [{r['role']}]" if "role" in r else ""
                head = escape(f"{r['family']}{role_tag} | {r['provider']}:{r['model']}")
                if r.get("error"):
                    self.console.print(Panel(f"[red]Error: {escape(r['error'])}[/red]", title=head, border_style="red"))
                elif r.get("answer"):
                    preview = escape(r["answer"][:300] + ("..." if len(r["answer"]) > 300 else ""))
                    self.console.print(Panel(preview, title=head, border_style="dim"))

            evidence_gate = out.get("evidence_gate")
            if self.mode == "redteam" and evidence_gate is None:
                evidence_gate = {
                    "status": "FAIL",
                    "security_conclusion": "NOT_ASSESSED",
                    "failed_checks": [
                        "Red-team run returned no deterministic evidence-gate result."
                    ],
                }

            # Display final merged answer
            final_ans = out.get("final") or "All swarm members failed to respond."
            from omnicli.orchestrate import swarm_final_title
            if self.mode == "redteam":
                title_text = (
                    "EVIDENCE-LED RED-TEAM TRIAGE (NOT A SECURITY VERDICT)"
                )
            else:
                title_text = swarm_final_title(
                    self.mode, out.get("synthesis")
                )
            title_style = "bold yellow" if self.mode == "redteam" else "bold green"
            title_mode = f"[{title_style}]{title_text}[/{title_style}]"
            final_border = (
                "red"
                if evidence_gate and evidence_gate.get("status") != "PASS_GROUNDING_ONLY"
                else "yellow"
                if self.mode == "redteam"
                else "green"
            )
            self.console.print(
                Panel(Markdown(final_ans), title=title_mode, border_style=final_border)
            )

            if evidence_gate:
                failures = evidence_gate.get("failed_checks", [])
                detail = "\n".join(f"- {item}" for item in failures)
                if not detail:
                    detail = evidence_gate.get(
                        "meaning", "Grounding only; security was not assessed."
                    )
                self.console.print(Panel(
                    f"EVIDENCE GATE: {evidence_gate.get('status', 'FAIL')}\n"
                    f"Security conclusion: "
                    f"{evidence_gate.get('security_conclusion', 'NOT_ASSESSED')}\n"
                    f"{detail}",
                    title="Deterministic Evidence Gate",
                    border_style=(
                        "yellow"
                        if evidence_gate.get("status") == "PASS_GROUNDING_ONLY"
                        else "red"
                    ),
                ))

            if self.mode == "redteam" and out.get("scenario_coverage"):
                rows = [
                    f"- {item['id']}: {item.get('completed_reviews', 0)}/"
                    f"{item.get('reviews', len(out.get('members', [])) if item.get('triggered') else 0)} "
                    "review responses complete"
                    for item in out["scenario_coverage"]
                ]
                self.console.print(Panel(
                    "\n".join(rows)
                    + "\n\nCompletion is not evidence of security or completeness.",
                    title="Source-Security Scenario Coverage",
                    border_style="cyan",
                ))

            if out.get("report_path"):
                self.last_report_path = out["report_path"]
                self.console.print(f"[bold green]📄 Red-Team Audit Report saved to:[/bold green] [cyan]{out['report_path']}[/cyan]")
                self.console.print("[dim]Type /accept to review and acknowledge remediations, or 'omni agent' to execute fixes.[/dim]\n")

            # Stats bar
            stats = Text()
            stats.append(f"⚡ Swarm [{self.mode.upper()}] • {len(out.get('members', []))} models", style="bold cyan")
            stats.append(f"  •  Synthesizer: {self.provider}/{self.model}", style="dim")
            stats.append(f"  •  {round(total_time, 2)}s", style="dim green")
            self.console.print(stats)
            self.console.print()

            self.messages.append({"role": "assistant", "content": final_ans})
            return

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

        # Persist the completed turn to per-provider history (opt-in).
        try:
            from omnicli.memory import append_history
            append_history(self.cfg, self.provider, "user", user_prompt)
            append_history(self.cfg, self.provider, "assistant", full_content)
        except Exception:
            pass

        # Optionally switch model family for the next turn.
        if getattr(self, "rotate_families", False):
            nxt = self._next_family_model()
            if nxt and nxt != self.model:
                from omnicli.scanner import classify_family
                self.model = nxt
                self.console.print(
                    f"[magenta]⟳ Rotating family → {self.provider}/{self.model} "
                    f"(family: {classify_family(self.model)})[/magenta]"
                )

    def run(self) -> None:
        self.print_welcome()
        while True:
            try:
                prompt_label = [
                    ("class:provider", f"{self.provider}"),
                    ("class:colon", ":"),
                    ("class:model", f"{self.model.split('/')[-1]}"),
                    ("class:mode", f" [{self.mode}]"),
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
