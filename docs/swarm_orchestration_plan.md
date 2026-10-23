# OmniCLI — Swarm Orchestration Plan

The chat REPL is one model at a time. Orchestration = **many models, different
families, across all providers, working together at once.** This plan builds that.

## Core engine: omnicli/orchestrate.py
A *swarm* = several `(provider, model)` pairs chosen for **family diversity across
ALL providers**, run **in parallel**, then brought together.

- `build_swarm(cfg, size, provider=None)` — pick N pairs, one per family,
  round-robin across every configured provider (so the swarm spans APIs, not one).
  `provider=X` restricts to that provider's families.
- `run_swarm(cfg, client, prompt, system, swarm, max_tokens)` — parallel fan-out
  (threads); one record per member `{provider, model, family, answer|error}`.
- `synthesize(cfg, client, prompt, results, judge)` — one member merges all
  answers into a single result (agreements kept, disagreements flagged, errors dropped).

## Modes (CLI: `omni swarm`, `omni chat`)
- **collaborate (default)** — fan out a task to the swarm, then merge claims
  deterministically. "Models working together." → item 2
- **redteam (`--redteam`)** — fan out the adversarial "find and PROVE" prompt to
  the swarm (different families attacking together), then run cross-examination,
  an evidence-fed typed judge, and a fail-closed evidence gate.
- **deliberate (`--mode deliberate`)** — two bounded rounds of independent
  proposals plus critiques, then a chair summary that preserves disagreement.
- **single (`--single`)** — direct 1-on-1 interaction with the lead provider/model
  (chat only; `omni swarm` is inherently multi-model).
- **builder** — removed. The old parity-assigned BUILDER/REDTEAM roles never
  built or tested anything. `--builder` is not a valid flag; `/builder` prints
  a removal notice.

`omni swarm "<task>" [-n SIZE] [--mode collaborate|redteam|deliberate] [--collaborate] [--redteam] [--raw] [-p PROVIDER]`
`omni chat [--mode collaborate|redteam|single] [--collaborate] [--redteam] [--single] [--size N]`

Conflicting mode flags are rejected instead of silently choosing one.

## In-Chat Slash Command Switching
- `/mode [collaborate|redteam|single]`
- `/collaborate`, `/c`
- `/redteam`, `/rt`
- `/single`
- `/size <N>`
- `/builder`, `/b` — removed; prints an explanation and leaves the mode unchanged

## Fix rotate-families (item 3) — cross-provider (item 4)
Today `_model_roster()` only lists the current provider's models, and
`_call_model` always calls `self.provider` — so rotation never leaves one API.
Fix:
- Roster becomes a list of `(provider, model)` pairs spanning **all providers**,
  family-diverse (reuse `build_swarm` ordering).
- `run()` tracks `(active_provider, active_model)`; `_call_model(provider, model, ...)`
  uses the pair's provider for URL/headers/keys.
- `--rotate-families` then truly switches family **and** API each turn; auto-fallback
  walks the same cross-provider roster.

## Tests (offline, fake multi-provider client)
- build_swarm: family-diverse, spans providers, respects size.
- run_swarm: one record per member, errors captured, parallel.
- synthesize: merges; single/none handled.
- redteam swarm: adversarial system prompt used; consolidated output.
- builder swarm: removed by design; requesting it raises in the library and
  prints a removal notice in chat.
- rotate-families: roster is cross-provider; _call_model uses the pair's provider.
- in-chat mode switching: slash commands toggle runtime modes cleanly.

## Status
- [x] 1 engine (build_swarm / run_swarm / synthesize)
- [x] 2 `omni swarm` collaborate + redteam
- [x] 3 rotate-families cross-provider fix (roster is (provider, model) pairs; _call_model uses the pair's provider)
- [x] 4 cross-provider throughout (build_swarm round-robins all providers; rotate roster spans APIs)
- [x] 5 builder mode cut; no fake build stage remains
- [x] 6 live in-chat multi-model orchestration with slash command switching
- [x] 7 Groq native provider integration and default starting model (`llama-3.3-70b-versatile`)
