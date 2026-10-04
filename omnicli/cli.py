"""
Command-line interface entry point for OmniCLI.
"""

from __future__ import annotations

import os
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
      • openrouter (OpenRouter API)
      • nvidia     (NVIDIA Build / NIM)
      • cloudflare (Cloudflare Workers AI)
      • bynara     (https://router.bynara.id/v1)
      • apinex     (https://api.apinex.bond/v1)
      • xkiro      (https://api.xkiro.com/v1)
    """
    pass


@main.command("ask")
@click.option("-p", "--provider", help="Provider ID (openrouter, nvidia, cloudflare, bynara, apinex, xkiro)")
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
                delta = chunk["delta"]
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
def cmd_chat(provider: Optional[str], model: Optional[str], system: Optional[str], temperature: float) -> None:
    """Start an interactive chat session."""
    session = ChatSession(
        provider=provider,
        model=model,
        system_prompt=system,
        temperature=temperature,
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


@main.command("agent")
@click.option("-p", "--provider", help="Supervisor provider (default: openrouter or nvidia)")
@click.option("-m", "--model", help="Supervisor model")
@click.option("--max-turns", type=int, default=10, help="Max tool call turns (default 10)")
@click.argument("goal", nargs=-1, required=True)
def cmd_agent(provider: Optional[str], model: Optional[str], max_turns: int, goal: tuple) -> None:
    """Run an autonomous agent equipped with local tools and cross-provider delegation."""
    from omnicli.agent import AgentRunner
    task_goal = " ".join(goal).strip()
    runner = AgentRunner(provider=provider, model=model, max_turns=max_turns, console=console)
    runner.run(task_goal)


if __name__ == "__main__":
    main()

