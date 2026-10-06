# OmniCLI — Local Multi-Provider AI Terminal, Router & Red-Team Agent

A fast, local CLI for querying, comparing, routing, and **orchestrating** across
7 OpenAI-compatible AI providers — plus an autonomous agent with deterministic
verification tools, cross-model consensus, per-API memory, and a heartbeat.

Providers:

- **Groq** (`https://api.groq.com/openai/v1`) — ultra-fast LPU inference (default: `llama-3.3-70b-versatile`)
- **OpenRouter** (`https://openrouter.ai/api/v1`)
- **NVIDIA Build / NIM** (`https://integrate.api.nvidia.com/v1`)
- **Cloudflare Workers AI** (`https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1`)
- **Bynara** (`https://router.bynara.id/v1`)
- **Apinex** (`https://api.apinex.bond/v1`)
- **XKiro** (`https://api.xkiro.com/v1`)

Installed on `$PATH` as `omni` (aliases: `omnicli`, `nexus`).

---

## Command map

| Command | What it does | Single vs. multi-API |
|---|---|---|
| `omni` / `omni chat` | Interactive streaming REPL with **live multi-model council orchestration** (`collaborate`, `builder`, `redteam`) or `single` | **multi** or single |
| `omni ask "<prompt>"` | One-shot query (also `omni "<prompt>"`) | single |
| `omni agent "<goal>"` | Autonomous tool-using agent | **multi** (delegates, auto-fallback, cross-provider rotate) |
| `omni swarm "<task>"` | Many models, different families, **in parallel** → merged (`collaborate`, `builder`, `redteam`) | **multi** |
| `omni verify consensus "<claim>"` | Blind vote across models/providers | **multi** |
| `omni verify http\|code` | Deterministic HTTP / code verification | single |
| `omni scan` | Discover `/models` per provider, build rosters | **multi** |
| `omni heartbeat` | Health/latency/model-count per API (+ tasks) | **multi** |
| `omni ping` | Quick connectivity + latency | **multi** |
| `omni models [provider]` | List/search a provider's live catalog | single |
| `omni serve` | OpenAI-compatible proxy, route by prefix | **multi** |
| `omni config ...` | Keys, account id, settings | — |

---

## Quick start

```bash
omni config set-key openrouter sk-...        # set keys
omni config set-key nvidia nvapi-...
omni config set-account <CLOUDFLARE_ACCOUNT_ID>

omni scan            # discover models per provider, build fallback + consensus rosters
omni heartbeat       # health/latency/model count across every API
omni                 # interactive chat (add -p openrouter for a chat-tuned default)
```

---

## The agent (autonomous + red-team)

```bash
# Autonomous agent with local tools + cross-provider delegation
omni agent "summarize the risks in ./config.json"

# Red-team mode: adversarial "find and PROVE" prompt + deterministic verify tools
omni agent --redteam "audit ~/project for auth bypass; prove each finding"

# Authorize HTTP verification only for the target's in-scope hosts
omni agent --redteam --allowed-host api.example.com "audit https://api.example.com"

# Cross-architecture corroboration: switch model family each turn
omni agent --redteam --rotate-families "review these contracts for reentrancy"
```

Flags:

- `--redteam` — adversarial system prompt; toolset restricted to the verify set
  (`verify_http`, `verify_code`, `verify_consensus`, `bash`, `read_file`,
  `consult_provider`). The model must prove claims with tools, not assert them.
- `--rotate-families` — after each turn, switch to the next distinct model family
  from a **cross-provider** roster, so consecutive turns use a different family
  *and* a different API (e.g. openrouter → nvidia → cloudflare).
- `--allow-shell` — enable the `bash` tool. **Off by default**; even when on,
  destructive/download-to-shell commands are blocked.
- `--allow-local` — let `verify_http` reach localhost / RFC1918 targets.
  **Off by default**; cloud-metadata/link-local IPs stay blocked regardless.
- `--allowed-host HOST` — authorize a host for agent HTTP verification; repeat
  for every in-scope host. Agent verification is denied when no host is allowed.
- `--allow-mutating-http` — explicitly permit state-changing methods on those
  allowed hosts. GET, HEAD, and OPTIONS remain the only methods allowed by
  default.
- `--max-turns N` — tool-call budget (default 10).

Interactive red-team is also available in chat: `omni chat --redteam
[--rotate-families]` (adversarial persona; chat has no tool execution).

---

## Swarm — many models at once

The headline orchestration: run several models from **different families across
all providers in parallel**, then bring them together.

Modes:
- **`collaborate` (default)** — Models work together constructively. Parallel fanout + merged consensus.
- **`builder` (`--builder`)** — Council split: Builders/Architects propose the design/implementation while Red-Team Breakers stress-test and attack in parallel. The synthesizer produces a hardened build fortifying against the attacks.
- **`redteam` (`--redteam`)** — Adversarial audit council probing for vulnerabilities, consolidated by severity.

```bash
omni swarm "design a token-bucket rate limiter"          # collaborate: parallel + synthesized answer
omni swarm --builder "design an OAuth2 PKCE auth engine" # builder: architects propose, red-team attacks, synthesized into hardened build
omni swarm --redteam "audit ~/project for authz bugs"    # red-team: multi-model audit council, consolidated
omni swarm -n 5 "..."                                     # 5 members
omni swarm --raw "..."                                    # show each member, skip synthesis
omni swarm -p openrouter "..."                            # restrict to one provider's families
```

Each member is a distinct family (e.g. google + meta + openai), queried
concurrently; a synthesis pass merges agreements and flags disagreements.
`--builder` couples constructive architects with breaker red-teamers into a fortified solution.
`--redteam` fans an adversarial review across the families and consolidates the
candidate findings — a red-team *team*, not one model. Swarm members do not
execute deterministic checks: their results are hypotheses for human review,
and model agreement alone is never proof.

## Deterministic verification

Proof, not vibes. These power the agent's red-team tools and are usable directly:

```bash
omni verify http https://api.example.com/users -e 'status 200, body contains "id"'
omni verify code ./auth.py --function login --must-contain "verify_token"
omni verify consensus "endpoint leaks PII without auth" -f ./evidence.txt
```

- **HTTP** probes are SSRF-guarded: internal/metadata IPs are blocked and the
  connection is pinned to the validated address (DNS-rebind safe).
- **Code** checks fail **closed** — a scoped claim against a missing function, or
  a request with no constraints, is REJECTED, never vacuously confirmed.
- **Consensus** runs a blind vote across your scanned verification panel.

---

## Memory (per-API context + history)

Each provider can be given memory files that are prepended to the model's system
prompt. Configure in `~/.config/omnicli/config.json`:

```json
"memory": {
  "enabled": true,
  "dir": "~/.config/omnicli/memory",
  "global": ["house_rules.md"],
  "providers": { "nvidia": ["cuda_notes.md", "/abs/path/targets.md"] },
  "history": { "enabled": false, "max_turns": 40 }
}
```

- `global` is injected for every provider; `providers.<id>` layers on top.
- Entries are relative to `dir` or absolute → **multiple files / locations per API**.
- Missing files warn and are skipped (memory is additive, never fatal).
- `history.enabled` resumes the last `max_turns` per provider in chat; `/forget`
  deletes a provider's history. History files are written owner-only (`0600`).

---

## Heartbeat

```bash
omni heartbeat                # per-API status, latency, model count -> status file
omni heartbeat --canary       # also send a tiny prompt to confirm the model answers
omni heartbeat --run-tasks    # also run the user task file
```

Built-in health checks are the default. A separate **user task file**
(`heartbeat.tasks_file`, JSON list) defines tasks to run on a tick:

```json
[ { "type": "ask", "provider": "nvidia", "prompt": "say OK", "expect": "OK" },
  { "type": "rescan" } ]
```

Task types: `ask` (prompt a provider, optional `expect` substring), `rescan`
(refresh rosters), `agent` (run an autonomous goal). Unknown/failed tasks are
recorded, never fatal.

---

## Chat REPL commands

**Mode Switching (Live Orchestration):**
- `/mode [collaborate|builder|redteam|single]` — change orchestration council mode
- `/collaborate`, `/c` — all models reason together in parallel, synthesized consensus
- `/builder`, `/b` — builders construct implementations, red-team attacks in parallel, hardened build
- `/redteam`, `/rt` — adversarial audit council, ranked vulnerabilities
- `/single` — direct 1-on-1 single-model chat
- `/size <1-10>` — set council member count (e.g. `/size 4`)

**Session & Providers:**
`/provider` · `/model` · `/models [query]` · `/providers` · `/system` · `/temp` ·
`/file <path>` · `/save` · `/clear` (session) · `/forget` (persisted history) ·
`/ping` · `/help` · `/exit`

## Proxy mode

```bash
omni serve --port 8000
```

External tools connect to `http://localhost:8000/v1` and route by prefix, e.g.
`nvidia/meta/llama-3.3-70b-instruct`, `openrouter/cohere/command-r`,
`xkiro/minimax/minimax-m3`.

---

## Security posture

- Agent shell is **off by default** (`--allow-shell`); destructive commands are
  deny-listed even when enabled.
- Agent HTTP checks require an explicit `--allowed-host` scope, and automated
  state-changing methods require the separate `--allow-mutating-http` opt-in.
- `verify_http` blocks internal/metadata IPs, is DNS-rebind-safe, and rejects
  ambiguous URL authorities (parser-differential guard). `--allow-local` opts in
  to local targets.
- Verification fails **closed**: no proof means no pass.
- Memory/history files are owner-only.

See `docs/memory_heartbeat_plan.md` for the memory/heartbeat design.
