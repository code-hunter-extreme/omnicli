"""
Command-line interface entry point for OmniCLI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from omnicli.chat import ChatSession
from omnicli.client import OmniClient
from omnicli.config import ConfigManager
from omnicli.proxy import start_proxy

console = Console()
cfg = ConfigManager()
client = OmniClient(cfg)


class OmniGroup(click.Group):
    """Custom Click Group that seamlessly routes unknown args to 'ask' and empty args to 'chat'."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not args:
            return super().parse_args(ctx, ["chat"])
        # Check if first arg is an option or command
        first = args[0]
        if first in self.commands or first in ("--help", "-h"):
            return super().parse_args(ctx, args)
        # If first arg starts with '-' and might be an option for ask, or is a plain prompt string
        return super().parse_args(ctx, ["ask"] + args)


@click.group(cls=OmniGroup)
def main() -> None:
    """OmniCLI — Local Multi-Provider AI Terminal & Router Client.

    Supported Providers:
      • groq       (https://api.groq.com/openai/v1)
      • openrouter (OpenRouter API)
      • nvidia     (NVIDIA Build / NIM)
      • cloudflare (Cloudflare Workers AI)
      • bynara     (https://router.bynara.id/v1)
      • apinex     (https://api.apinex.bond/v1)
      • xkiro      (https://api.xkiro.com/v1)
    """
    pass


@main.command("ask")
@click.option("-p", "--provider", help="Provider ID (groq, openrouter, nvidia, cloudflare, bynara, apinex, xkiro)")
@click.option("-m", "--model", help="Model identifier")
@click.option("-s", "--system", help="System prompt")
@click.option("-t", "--temperature", type=float, default=0.7, help="Sampling temperature (0.0 - 2.0)")
@click.option("-f", "--file", "input_file", type=click.Path(exists=True), help="Attach file content")
@click.option("--max-tokens", type=int, default=2048, help="Max tokens to generate (default: 2048)")
@click.option("--no-stream", is_flag=True, help="Disable streaming output")
@click.option("--raw", is_flag=True, help="Output raw text without rich markdown formatting")
@click.argument("prompt_words", nargs=-1, required=False)
def cmd_ask(
    provider: Optional[str],
    model: Optional[str],
    system: Optional[str],
    temperature: float,
    input_file: Optional[str],
    max_tokens: int,
    no_stream: bool,
    raw: bool,
    prompt_words: tuple,
) -> None:
    """Execute a prompt or query models directly."""
    piped_input = ""
    if not sys.stdin.isatty():
        piped_input = sys.stdin.read().strip()

    prompt = " ".join(prompt_words).strip() if prompt_words else ""
    if piped_input:
        if prompt:
            prompt = f"{prompt}\n\nContext / Input:\n{piped_input}"
        else:
            prompt = piped_input

    if input_file:
        try:
            content = Path(input_file).read_text(encoding="utf-8", errors="replace")
            file_block = f"\n\n[Attached File: {Path(input_file).name}]\n```\n{content}\n```\n"
            prompt = f"{prompt}{file_block}" if prompt else file_block
        except Exception as e:
            console.print(f"[bold red]Failed to read attached file:[/bold red] {e}")
            sys.exit(1)

    if not prompt:
        console.print("[yellow]No prompt provided. Launching interactive chat...[/yellow]")
        session = ChatSession(
            provider=provider,
            model=model,
            system_prompt=system,
            temperature=temperature,
        )
        session.run()
        return

    run_one_shot(
        prompt=prompt,
        provider_id=provider,
        model_id=model,
        system_prompt=system,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=not no_stream,
        raw=raw,
    )


def run_one_shot(
    prompt: str,
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = True,
    raw: bool = False,
) -> None:
    prov = (provider_id or cfg.get_default_provider()).lower()
    mod = model_id or cfg.get_default_model(prov)
    sys_p = system_prompt or cfg.data.get("system_prompt", "You are a helpful, concise AI assistant.")

    messages = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": prompt},
    ]

    if not stream:
        with console.status(f"[cyan]Querying {prov}/{mod}...[/cyan]"):
            try:
                res = client.chat_complete(
                    provider_id=prov,
                    messages=messages,
                    model=mod,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                choices = res.get("choices", [])
                out = choices[0]["message"]["content"] if choices else ""
                if raw:
                    print(out)
                else:
                    console.print(Markdown(out))
            except Exception as e:
                console.print(f"[bold red]Error from {prov}:[/bold red] {e}")
                sys.exit(1)
        return

    # Streaming mode
    full_text = ""
    try:
        for chunk in client.chat_stream(
            provider_id=prov,
            messages=messages,
            model=mod,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if chunk["type"] == "content":
                delta = str(chunk.get("delta", ""))
                full_text += delta
                sys.stdout.write(delta)
                sys.stdout.flush()
            elif chunk["type"] == "error":
                sys.stderr.write(f"\nError: {chunk.get('content')}\n")
                sys.exit(1)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")


@main.command("chat")
@click.option("-p", "--provider", help="Provider ID")
@click.option("-m", "--model", help="Model name")
@click.option("-s", "--system", help="System prompt")
@click.option("-t", "--temperature", type=float, default=0.7, help="Temperature")
@click.option("--mode", type=click.Choice(["collaborate", "builder", "redteam", "single"]), default="collaborate",
              help="Council mode: collaborate (all models), builder (architects+redteam), redteam (adversarial), or single")
@click.option("--collaborate", "flag_collaborate", is_flag=True, default=False, help="Set mode to collaborate")
@click.option("--builder", "flag_builder", is_flag=True, default=False, help="Set mode to builder")
@click.option("--redteam", is_flag=True, default=False,
              help="Adversarial security-analyst persona (no tools; proof-demanding)")
@click.option("--single", "flag_single", is_flag=True, default=False, help="Set mode to single-model chat")
@click.option("--size", type=int, default=None, help="Council size for multi-model orchestration (default: all active providers)")
@click.option("--rotate-families", is_flag=True, default=False,
              help="Switch model family each turn from the scanned roster")
def cmd_chat(provider: Optional[str], model: Optional[str], system: Optional[str],
             temperature: float, mode: str, flag_collaborate: bool, flag_builder: bool,
             redteam: bool, flag_single: bool, size: Optional[int], rotate_families: bool) -> None:
    """Start an interactive chat session with multi-model orchestration."""
    active_mode = mode
    if flag_single:
        active_mode = "single"
    elif redteam:
        active_mode = "redteam"
    elif flag_builder:
        active_mode = "builder"
    elif flag_collaborate:
        active_mode = "collaborate"

    session = ChatSession(
        provider=provider,
        model=model,
        system_prompt=system,
        temperature=temperature,
        redteam=redteam,
        rotate_families=rotate_families,
        mode=active_mode,
        council_size=size,
    )
    session.run()


@main.command("ping")
def cmd_ping() -> None:
    """Test connection and measure latency to all configured endpoints."""
    table = Table(title="Provider Diagnostics & Health", title_style="bold cyan", border_style="dim")
    table.add_column("Provider", style="bold green", width=14)
    table.add_column("Endpoint", style="dim cyan", width=34)
    table.add_column("Status", width=16)
    table.add_column("Latency", justify="right", width=12)
    table.add_column("Models", justify="right", width=10)
    table.add_column("Details", style="dim")

    providers = list(cfg.data.get("providers", {}).keys())

    with console.status("[cyan]Testing connectivity across all endpoints...[/cyan]", spinner="dots"):
        for pid in providers:
            res = client.ping(pid)
            st = res["status"]
            if st == "healthy":
                status_str = "[bold green]ONLINE (200)[/bold green]"
            elif st in ("unauthorized", "missing_key"):
                status_str = "[yellow]AUTH NEEDED[/yellow]"
            else:
                status_str = f"[red]{st.upper()}[/red]"

            base_url = cfg.get_base_url(pid)
            latency_str = f"{res['latency_ms']} ms" if res["latency_ms"] else "-"
            model_str = str(res["model_count"]) if res["model_count"] else "-"
            details = res["error"] or "OK"
            table.add_row(pid, base_url[:33], status_str, latency_str, model_str, details[:35])

    console.print(table)


@main.command("heartbeat")
@click.option("--canary", is_flag=True, default=False,
              help="Also send a tiny canary prompt per healthy provider")
@click.option("--run-tasks", is_flag=True, default=False,
              help="Also run the user task file (heartbeat.tasks_file)")
@click.option("--no-write", is_flag=True, default=False,
              help="Do not write the status file")
def cmd_heartbeat(canary: bool, run_tasks: bool, no_write: bool) -> None:
    """Per-API heartbeat: health, latency, model count (built-in health checks)."""
    from omnicli import heartbeat as hb
    with console.status("[cyan]Heartbeat: checking all providers...[/cyan]", spinner="dots"):
        records = hb.run_heartbeat(cfg, client, canary=canary)

    table = Table(title="OmniCLI Heartbeat", title_style="bold cyan", border_style="dim")
    table.add_column("Provider", style="bold green", width=14)
    table.add_column("Status", width=16)
    table.add_column("Latency", justify="right", width=12)
    table.add_column("Models", justify="right", width=10)
    if canary:
        table.add_column("Canary", width=10)
    table.add_column("Details", style="dim")

    for rec in records:
        st = rec.get("status")
        if st == "healthy":
            status_str = "[bold green]ONLINE[/bold green]"
        elif st in ("unauthorized", "missing_key"):
            status_str = "[yellow]AUTH NEEDED[/yellow]"
        elif st == "config_needed":
            status_str = "[yellow]CONFIG[/yellow]"
        else:
            status_str = f"[red]{str(st).upper()}[/red]"
        latency_str = f"{rec['latency_ms']} ms" if rec.get("latency_ms") else "-"
        model_str = str(rec["model_count"]) if rec.get("model_count") else "-"
        row = [rec.get("provider", "?"), status_str, latency_str, model_str]
        if canary:
            can = rec.get("canary")
            if can is None:
                row.append("-")
            elif can.get("ok"):
                row.append("[green]OK[/green]")
            else:
                row.append("[red]FAIL[/red]")
        row.append((rec.get("error") or "OK")[:35])
        table.add_row(*row)

    console.print(table)

    task_results = None
    if run_tasks:
        tasks = hb.load_tasks(cfg)
        if not tasks:
            console.print("[dim]No heartbeat tasks configured (heartbeat.tasks_file).[/dim]")
            task_results = []
        else:
            with console.status(f"[cyan]Running {len(tasks)} heartbeat task(s)...[/cyan]", spinner="dots"):
                task_results = hb.run_tasks(cfg, client, tasks)
            ttable = Table(title="Heartbeat Tasks", title_style="bold cyan", border_style="dim")
            ttable.add_column("#", width=4)
            ttable.add_column("Type", style="bold green", width=10)
            ttable.add_column("OK", width=6)
            ttable.add_column("Detail", style="dim")
            for i, r in enumerate(task_results):
                ok_str = "[green]OK[/green]" if r.get("ok") else "[red]FAIL[/red]"
                detail = r.get("error") or r.get("reply") or r.get("answer") or \
                    (f"{r.get('models', r.get('providers', ''))}")
                ttable.add_row(str(i), str(r.get("type", "?")), ok_str, str(detail)[:50])
            console.print(ttable)

    if not no_write:
        path = hb.write_status(cfg, records, tasks=task_results)
        if path:
            console.print(f"[dim]Status written to {path}[/dim]")


@main.command("models")
@click.argument("provider", required=False)
@click.option("-s", "--search", help="Filter models by search substring")
def cmd_models(provider: Optional[str], search: Optional[str]) -> None:
    """List or search models from a provider."""
    target_prov = (provider or cfg.get_default_provider()).lower()
    with console.status(f"[cyan]Fetching models from {target_prov}...[/cyan]", spinner="dots"):
        try:
            models = client.list_models(target_prov)
        except Exception as e:
            console.print(f"[bold red]Failed to fetch models:[/bold red] {e}")
            return

    if not models:
        console.print(f"[yellow]No models found from {target_prov}.[/yellow]")
        return

    filtered = []
    q = (search or "").lower().strip()
    for m in models:
        mid = m.get("id", "")
        if not q or q in mid.lower() or q in m.get("name", "").lower():
            filtered.append(m)

    table = Table(
        title=f"Models on {target_prov} ({len(filtered)}/{len(models)})",
        title_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Model ID", style="bold green")
    table.add_column("Context", style="yellow", width=12)
    table.add_column("Pricing", style="cyan", width=22)

    limit = 50
    for m in filtered[:limit]:
        mid = m.get("id", "")
        ctx = str(m.get("context_length", m.get("context_window", "-")))
        pricing_data = m.get("pricing")
        price_str = "-"
        if isinstance(pricing_data, dict):
            inp = pricing_data.get("input", "-")
            out = pricing_data.get("output", "-")
            price_str = f"${inp} / ${out}"
        elif "price" in m:
            price_str = str(m["price"])

        table.add_row(mid, ctx, price_str)

    console.print(table)
    if len(filtered) > limit:
        console.print(f"[dim]Showing first {limit} models. Use -s/--search to narrow down results.[/dim]")


@main.command("serve")
@click.option("--host", default="127.0.0.1", help="Host interface to bind")
@click.option("--port", default=8000, type=int, help="Port to listen on")
def cmd_serve(host: str, port: int) -> None:
    """Run local OpenAI-compatible proxy server for external tools."""
    start_proxy(host=host, port=port)


@main.group("config")
def cmd_config() -> None:
    """View and manage OmniCLI configuration and API keys."""
    pass


@cmd_config.command("show")
def config_show() -> None:
    """Display current configuration."""
    table = Table(title="OmniCLI Configuration", title_style="bold cyan", border_style="dim")
    table.add_column("Setting / Provider", style="bold green", width=20)
    table.add_column("Value / Base URL", style="white")
    table.add_column("Key Status", style="yellow", width=15)

    table.add_row("Default Provider", cfg.get_default_provider(), "-")
    table.add_row("System Prompt", cfg.data.get("system_prompt", "")[:40] + "...", "-")
    table.add_row("Cloudflare Account ID", cfg.get_cloudflare_account_id() or "[dim]Not set[/dim]", "-")
    table.add_section()

    for pid, pdata in cfg.data.get("providers", {}).items():
        key = cfg.get_api_key(pid)
        key_status = f"{key[:4]}...{key[-3:]}" if key and len(key) > 7 else ("[green]Set[/green]" if key else "[dim]Missing[/dim]")
        url = cfg.get_base_url(pid)
        table.add_row(pid, f"{url} (model: {pdata.get('default_model')})", key_status)

    console.print(table)


@cmd_config.command("set-key")
@click.argument("provider")
@click.argument("key")
def config_set_key(provider: str, key: str) -> None:
    """Set API key for a provider."""
    prov = provider.lower()
    if prov not in cfg.data.get("providers", {}):
        console.print(f"[yellow]Warning: Provider '{prov}' is not yet in provider list. Adding it.[/yellow]")
    cfg.set_api_key(prov, key)
    console.print(f"[bold green]API key updated for {prov}.[/bold green]")


@cmd_config.command("set-account")
@click.argument("account_id")
def config_set_account(account_id: str) -> None:
    """Set Cloudflare Account ID."""
    cfg.set_cloudflare_account_id(account_id)
    console.print(f"[bold green]Cloudflare account ID set to {account_id}.[/bold green]")


@cmd_config.command("set-default")
@click.option("-p", "--provider", help="Set default provider")
@click.option("-m", "--model", help="Set default model for provider")
def config_set_default(provider: Optional[str], model: Optional[str]) -> None:
    """Set default provider or default model."""
    if provider:
        try:
            cfg.set_default_provider(provider)
            console.print(f"[bold green]Default provider set to:[/bold green] {provider}")
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            return
    if model:
        target_p = (provider or cfg.get_default_provider()).lower()
        cfg.set_default_model(target_p, model)
        console.print(f"[bold green]Default model for {target_p} set to:[/bold green] {model}")


@cmd_config.command("add-provider")
@click.argument("name")
@click.argument("base_url")
@click.option("--key", help="API key")
@click.option("--model", default="default", help="Default model name")
def config_add_provider(name: str, base_url: str, key: Optional[str], model: str) -> None:
    """Add a custom OpenAI-compatible provider."""
    cfg.add_custom_provider(provider_id=name, base_url=base_url, api_key=key, default_model=model)
    console.print(f"[bold green]Added custom provider '{name}' ({base_url}).[/bold green]")


@main.command("scan")
@click.option("-p", "--provider", help="Scan a single provider (default: all)")
@click.option("--panel-size", type=int, default=3, show_default=True,
              help="Max models in verification panel per provider")
@click.option("--max-per-family", type=int, default=2, show_default=True,
              help="Max fallback models per family per provider")
def cmd_scan(provider: Optional[str], panel_size: int, max_per_family: int) -> None:
    """Scan provider /models endpoints, classify by family, write fallback rosters to config.

    Run this once after setup (or whenever you want to refresh available models).
    Rosters built here are used by the agent for auto-fallback and by the
    consensus verifier for diverse-family red-team panels.
    """
    from omnicli.scanner import scan_all_providers, scan_provider as scan_one

    table = Table(title="Model Scan Results", title_style="bold cyan", border_style="dim")
    table.add_column("Provider",    style="bold green", width=14)
    table.add_column("Status",      width=10)
    table.add_column("Models",      justify="right", width=8)
    table.add_column("Families",    justify="right", width=10)
    table.add_column("Fallback roster (first 3)", style="dim")
    table.add_column("Verification panel",         style="cyan")

    with console.status("[cyan]Scanning providers for available models...[/cyan]", spinner="dots"):
        if provider:
            results = [scan_one(provider.lower(), panel_size=panel_size, max_per_family=max_per_family)]
        else:
            results = scan_all_providers(panel_size=panel_size, max_per_family=max_per_family)

    for r in results:
        ok = r["ok"]
        status_str = "[bold green]OK[/bold green]" if ok else "[bold red]FAIL[/bold red]"
        roster_preview = ", ".join(r["fallback_models"][:3]) if r["fallback_models"] else "[dim]run omni scan[/dim]"
        panel_preview  = ", ".join(
            f"{e['family']}:{e['model'].split('/')[-1][:18]}"
            for e in r["verification_panel"]
        ) if r["verification_panel"] else "[dim]none[/dim]"
        err = f"  [red]{r['error'][:50]}[/red]" if r.get("error") else ""
        table.add_row(
            r["provider"],
            status_str,
            str(r["model_count"]) if ok else "-",
            str(r["families_found"]) if ok else "-",
            roster_preview + err,
            panel_preview,
        )

    console.print(table)
    if not provider:
        total_ok = sum(1 for r in results if r["ok"])
        console.print(
            f"\n[bold green]✓[/bold green] Scanned {len(results)} providers, "
            f"{total_ok} OK — rosters written to config."
        )


@main.command("agent")
@click.option("-p", "--provider", help="Supervisor provider (default: openrouter or nvidia)")
@click.option("-m", "--model", help="Supervisor model")
@click.option("--max-turns", type=int, default=10, help="Max tool call turns (default 10)")
@click.option("--redteam", is_flag=True, default=False,
              help="Adversarial system prompt + verify_* tools as the primary toolset")
@click.option("--rotate-families", is_flag=True, default=False,
              help="Switch model family each turn from the scanned roster")
@click.option("--allow-shell", is_flag=True, default=False,
              help="Enable the bash tool (OFF by default; destructive commands still blocked)")
@click.option("--allow-local", is_flag=True, default=False,
              help="Permit verify_http against localhost/RFC1918 (metadata IPs stay blocked)")
@click.argument("goal", nargs=-1, required=True)
def cmd_agent(provider: Optional[str], model: Optional[str], max_turns: int,
              redteam: bool, rotate_families: bool, allow_shell: bool,
              allow_local: bool, goal: tuple) -> None:
    """Run an autonomous agent equipped with local tools and cross-provider delegation."""
    from omnicli.agent import AgentRunner
    task_goal = " ".join(goal).strip()
    runner = AgentRunner(
        provider=provider,
        model=model,
        max_turns=max_turns,
        console=console,
        redteam=redteam,
        rotate_families=rotate_families,
        allow_shell=allow_shell,
        allow_local=allow_local,
    )
    runner.run(task_goal)


@main.command("swarm")
@click.option("-n", "--size", type=int, default=None, help="Number of swarm members (default: all active providers)")
@click.option("--mode", type=click.Choice(["collaborate", "builder", "redteam"]), default="collaborate",
              help="Swarm mode: collaborate (consensus), builder (architects+redteam), or redteam (audit)")
@click.option("--collaborate", "flag_collaborate", is_flag=True, default=False, help="Run collaborative swarm")
@click.option("--builder", "flag_builder", is_flag=True, default=False, help="Run builder swarm (architects vs redteam)")
@click.option("--redteam", is_flag=True, default=False,
              help="Multi-model adversarial red-team review (different families together)")
@click.option("--raw", is_flag=True, default=False, help="Show each member's answer, skip synthesis")
@click.option("-p", "--provider", help="Restrict the swarm to one provider's families")
@click.argument("prompt", nargs=-1, required=True)
def cmd_swarm(size: Optional[int], mode: str, flag_collaborate: bool, flag_builder: bool,
              redteam: bool, raw: bool, provider: Optional[str], prompt: tuple) -> None:
    """Run MANY models (different families, across all providers) together on one task."""
    from rich.panel import Panel
    from rich.markup import escape
    from omnicli import orchestrate as orch
    task = " ".join(prompt).strip()
    swarm = orch.build_swarm(cfg, size=size, provider=provider)
    if not swarm:
        console.print("[red]No models available for a swarm. Run 'omni scan' first.[/red]")
        return

    active_mode = "redteam" if redteam else ("builder" if flag_builder else ("collaborate" if flag_collaborate else mode))
    roster = ", ".join(f"{p}:{m.split('/')[-1]} ({orch.classify_family(m)})" for p, m in swarm)
    console.print(Panel(
        f"[bold cyan]Swarm {active_mode.upper()}[/bold cyan] — {len(swarm)} members\n"
        f"[dim]{escape(roster)}[/dim]\n[bold green]Task:[/bold green] {escape(task)}",
        title="Swarm Active", border_style="cyan",
    ))
    with console.status(f"[cyan]Querying {len(swarm)} models in parallel ({active_mode})...[/cyan]", spinner="dots"):
        out = orch.run_orchestration(active_mode, cfg, client, task, size=size, synth=not raw)
    # Model/tool output is untrusted — escape it so it cannot inject Rich markup.
    for r in out["members"]:
        role_tag = f" [{r['role']}]" if "role" in r else ""
        head = escape(f"{r['family']}{role_tag} | {r['provider']}:{r['model']}")
        if r["error"]:
            console.print(Panel(f"[red]ERROR: {escape(r['error'])}[/red]", title=head, border_style="red"))
        else:
            body = escape(r["answer"]) if r["answer"] else "[dim](empty)[/dim]"
            console.print(Panel(body, title=head, border_style="dim"))
    if out.get("final"):
        title_mode = {
            "collaborate": "[bold green]SYNTHESIZED (swarm consensus)[/bold green]",
            "builder": "[bold green]HARDENED BUILD (synthesized against red-team)[/bold green]",
            "redteam": "[bold green]CONSOLIDATED (red-team findings)[/bold green]",
        }.get(active_mode, "[bold green]SYNTHESIZED[/bold green]")
        console.print(Panel(escape(out["final"]),
                            title=title_mode,
                            border_style="green"))
    if out.get("report_path"):
        console.print(f"\n[bold green]📄 Red-Team Audit Report saved to:[/bold green] [cyan]{out['report_path']}[/cyan]")
        try:
            accept = click.confirm("\nWould you like to review and accept the proposed remediations/changes?", default=True)
            if accept:
                console.print("[bold green]✓ Remediations accepted. You can use 'omni agent' or local editing tools to apply the changes.[/bold green]")
        except Exception:
            pass


@main.group("verify")
def cmd_verify() -> None:
    """Deterministic verification and multi-model consensus testing."""
    pass


@cmd_verify.command("http")
@click.argument("url")
@click.option("-m", "--method", default="GET", help="HTTP method (GET, POST, etc.)")
@click.option("-e", "--expected", required=True, help="Expected prediction string (e.g. 'status 200, body contains \"admin\"')")
@click.option("-b", "--body", help="Request body content")
def verify_http_cmd(url: str, method: str, expected: str, body: Optional[str]) -> None:
    """Verify HTTP wire behavior deterministically."""
    from rich.panel import Panel
    from omnicli.deterministic import HTTPVerifier
    with console.status(f"[cyan]Executing deterministic HTTP probe against {url}...[/cyan]"):
        res = HTTPVerifier.verify(url=url, method=method, expected_str=expected, body=body)
    verdict = res["verdict"]
    color = "green" if verdict == "CONFIRMED" else "red"
    console.print(Panel(
        f"[bold {color}]VERDICT: {verdict}[/bold {color}]\n"
        f"[bold white]Reason:[/bold white] {res.get('reason')}\n"
        f"[dim white]Observed Status:[/dim white] {res.get('observed', {}).get('status') if res.get('observed') else 'None'}",
        title="Deterministic HTTP Verification",
        border_style=color,
    ))


@cmd_verify.command("code")
@click.argument("path")
@click.option("--must-contain", multiple=True, help="Patterns that MUST exist in code")
@click.option("--must-not-contain", multiple=True, help="Patterns that MUST NOT exist")
@click.option("--function", help="Function name to scope check")
@click.option("--check-type", help="Built-in check (e.g. 'wp_auth_check')")
def verify_code_cmd(path: str, must_contain: tuple, must_not_contain: tuple, function: Optional[str], check_type: Optional[str]) -> None:
    """Verify local source code patterns and security guards deterministically."""
    from rich.panel import Panel
    from omnicli.deterministic import CodeVerifier
    res = CodeVerifier.verify_file(
        file_path=path,
        must_contain=list(must_contain) if must_contain else None,
        must_not_contain=list(must_not_contain) if must_not_contain else None,
        target_function=function,
        check_type=check_type,
    )
    verdict = res["verdict"]
    color = "green" if verdict == "CONFIRMED" else "red"
    console.print(Panel(
        f"[bold {color}]VERDICT: {verdict}[/bold {color}]\n"
        f"[bold white]Reason:[/bold white] {res.get('reason')}\n"
        f"[dim white]Target File:[/dim white] {res.get('file')}",
        title="Deterministic Code Pattern Verification",
        border_style=color,
    ))


@cmd_verify.command("consensus")
@click.argument("claim")
@click.option("-f", "--file", "filepath", help="File containing code/evidence to evaluate")
@click.option("-e", "--evidence", help="Raw inline evidence text")
@click.option("-m", "--model", "models", multiple=True, help="Model spec in provider:model format")
def verify_consensus_cmd(claim: str, filepath: Optional[str], evidence: Optional[str], models: tuple) -> None:
    """Run blind multi-model / multi-provider consensus verification."""
    from rich.panel import Panel
    from omnicli.deterministic import CrossModelConsensusVerifier
    ev_text = evidence or ""
    if filepath:
        p = Path(filepath).expanduser().resolve()
        if p.exists() and p.is_file():
            ev_text = p.read_text(encoding="utf-8", errors="replace")
        else:
            console.print(f"[bold red]Evidence file not found:[/bold red] {filepath}")
            return

    model_specs = []
    if models:
        for m in models:
            if ":" in m:
                p, mod = m.split(":", 1)
                model_specs.append((p.strip(), mod.strip()))
            else:
                model_specs.append((m.strip(), None))

    with console.status("[cyan]Querying model panel for blind verification consensus...[/cyan]"):
        verifier = CrossModelConsensusVerifier(client)
        res = verifier.verify_finding(claim, ev_text, models=model_specs or None)

    verdict = res["verdict"]
    color = "green" if verdict == "CONSENSUS_CONFIRMED" else ("red" if verdict == "CONSENSUS_REJECTED" else "yellow")

    table = Table(title="Cross-Model Verification Votes", border_style="dim")
    table.add_column("Model / Endpoint", style="bold cyan")
    table.add_column("Vote", style="bold")
    for mod_name, vote in res["votes"].items():
        v_col = "green" if vote == "TRUE_POSITIVE" else ("red" if vote == "FALSE_POSITIVE" else "yellow")
        table.add_row(mod_name, f"[{v_col}]{vote}[/{v_col}]")

    console.print(Panel(
        f"[bold {color}]CONSENSUS VERDICT: {verdict}[/bold {color}]\n"
        f"Corroboration: [green]{res['true_positives']} True Positives[/green] / [red]{res['false_positives']} False Positives[/red] "
        f"(TP {res['tp_ratio']} / FP {res['fp_ratio']} of {res['total_queried']})",
        title="Blind Multi-Model Verification Consensus",
        border_style=color,
    ))
    console.print(table)


@main.command("check")
@click.argument("target")
@click.option("-f", "--file", "filepath", help="Local code or evidence file to inspect")
@click.option("-e", "--expected", help="Expected deterministic condition (e.g. 'status 200' or pattern)")
@click.option("-m", "--model", "models", multiple=True, help="Specific provider:model panels to query")
def cmd_check(target: str, filepath: Optional[str], expected: Optional[str], models: tuple) -> None:
    """Unified check: tests target deterministically and across multiple API keys/models."""
    from rich.panel import Panel
    from omnicli.deterministic import HTTPVerifier, CodeVerifier, CrossModelConsensusVerifier

    target_str = target.strip()
    is_url = target_str.startswith("http://") or target_str.startswith("https://")
    is_local_file = False

    evidence_file = filepath
    if not is_url:
        p = Path(target_str).expanduser()
        if p.exists() and p.is_file():
            is_local_file = True
            evidence_file = str(p.resolve())

    console.print(Panel(
        f"[bold cyan]OmniCLI Check Target:[/bold cyan] {target_str}\n"
        f"[dim]Mode: {'HTTP Wire + Multi-API Verification' if is_url else ('Code AST + Multi-API Verification' if is_local_file else 'Adversarial Claim + Multi-API Consensus')}[/dim]",
        title="Omni System Check",
        border_style="cyan",
    ))

    # Phase 1: Deterministic Engine Execution
    deterministic_result = None
    if is_url:
        with console.status(f"[cyan]Phase 1: Running deterministic HTTP wire probe against {target_str}...[/cyan]"):
            exp = expected or "status 200"
            deterministic_result = HTTPVerifier.verify(url=target_str, expected_str=exp)
        d_verdict = deterministic_result["verdict"]
        d_col = "green" if d_verdict == "CONFIRMED" else "red"
        console.print(Panel(
            f"[bold {d_col}]GATE VERDICT: {d_verdict}[/bold {d_col}]\n"
            f"[bold white]Observation:[/bold white] {deterministic_result.get('reason')}\n"
            f"[dim white]Observed Status:[/dim white] {deterministic_result.get('observed', {}).get('status') if deterministic_result.get('observed') else 'None'}",
            title="Phase 1: Pure Deterministic Wire Verification (Zero LLM)",
            border_style=d_col,
        ))
    elif is_local_file:
        with console.status(f"[cyan]Phase 1: Running deterministic code AST checks on {target_str}...[/cyan]"):
            must_c = [expected] if expected else None
            deterministic_result = CodeVerifier.verify_file(
                file_path=evidence_file or target_str,
                must_contain=must_c,
                check_type="wp_auth_check" if target_str.endswith(".php") else None,
            )
        d_verdict = deterministic_result["verdict"]
        d_col = "green" if d_verdict == "CONFIRMED" else "yellow"
        console.print(Panel(
            f"[bold {d_col}]GATE VERDICT: {d_verdict}[/bold {d_col}]\n"
            f"[bold white]Observation:[/bold white] {deterministic_result.get('reason')}",
            title="Phase 1: Pure Deterministic AST Code Verification (Zero LLM)",
            border_style=d_col,
        ))

    # Phase 2: Multi-Model & Cross-Provider Consensus Verification
    code_context = ""
    if evidence_file:
        ep = Path(evidence_file).expanduser().resolve()
        if ep.exists() and ep.is_file():
            code_context = ep.read_text(encoding="utf-8", errors="replace")

    claim_text = (
        f"Verify security implications and vulnerabilities in endpoint: {target_str}"
        if is_url else
        (f"Analyze security posture and check for missing auth/sanitization in: {target_str}" if is_local_file else target_str)
    )

    model_specs = []
    if models:
        for m in models:
            if ":" in m:
                p, mod = m.split(":", 1)
                model_specs.append((p.strip(), mod.strip()))
            else:
                model_specs.append((m.strip(), None))

    with console.status("[cyan]Phase 2: Querying multiple API keys & model families for blind consensus...[/cyan]"):
        verifier = CrossModelConsensusVerifier(client)
        consensus_res = verifier.verify_finding(claim_text, code_context or target_str, models=model_specs or None)

    c_verdict = consensus_res["verdict"]
    c_col = "green" if c_verdict == "CONSENSUS_CONFIRMED" else ("red" if c_verdict == "CONSENSUS_REJECTED" else "yellow")

    table = Table(title="Multi-Provider / Multi-Family Vote Matrix", border_style="dim")
    table.add_column("Provider & Model", style="bold cyan")
    table.add_column("Independent Verdict", style="bold")
    for mod_name, vote in consensus_res["votes"].items():
        v_col = "green" if vote == "TRUE_POSITIVE" else ("red" if vote == "FALSE_POSITIVE" else "yellow")
        table.add_row(mod_name, f"[{v_col}]{vote}[/{v_col}]")

    console.print(Panel(
        f"[bold {c_col}]CROSS-MODEL CONSENSUS: {c_verdict}[/bold {c_col}]\n"
        f"Agreement: [green]{consensus_res['true_positives']} Confirming[/green] / [red]{consensus_res['false_positives']} Refuting[/red] "
        f"(TP {consensus_res['tp_ratio']} / FP {consensus_res['fp_ratio']} of {consensus_res['total_queried']})",
        title="Phase 2: Multi-Provider AI Consensus",
        border_style=c_col,
    ))
    console.print(table)


if __name__ == "__main__":
    main()


