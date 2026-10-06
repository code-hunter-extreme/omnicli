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
- **`deliberate`** — Two bounded rounds: parallel independent proposals, then parallel critiques that receive the complete prior-round transcript, followed by a chair summary that preserves disagreement.

```bash
omni swarm "design a token-bucket rate limiter"          # collaborate: parallel + synthesized answer
omni swarm --builder "design an OAuth2 PKCE auth engine" # builder: architects propose, red-team attacks, synthesized into hardened build
omni swarm --redteam --source-file ./src/auth.py "audit authz bugs" # ground citations against source
omni swarm --mode deliberate "compare approaches for this design"
omni swarm -n 5 "..."                                     # 5 members
omni swarm --raw "..."                                    # show each member, skip synthesis
omni swarm -p openrouter "..."                            # restrict to one provider's families
```

Each member is a distinct family (e.g. google + meta + openai), queried
concurrently; a synthesis pass merges agreements and flags disagreements.
`--builder` couples constructive architects with breaker red-teamers into a fortified solution.
`--redteam` runs a staged review: each member performs separate passes over
authentication/authorization, tainted-input sinks, sensitive-data handling, and
state/concurrency/resource risks (even when only one member is configured).
The challenge stage cross-examines the combined candidate list, then OmniCLI
uses deterministic source/route facts to deduplicate and screen cited model
hypotheses. A generic missing-auth claim on a literal-only public health route
is not treated as actionable; a missing local guard on a mutating route remains
unverified until middleware, reachability, and impact are checked. Members
challenge the combined candidate list for unsupported paths, missing
preconditions, and false positives. When only one model is configured, this is
explicitly labeled self-critique, not independent corroboration. The final
red-team report itself is deterministic; it does not ask another model to vote
on or amplify the same claims. Every stage and provider failure is shown in the
result.
This is orchestration of model review, not target execution: the swarm does not
run tests or HTTP probes, and model agreement alone is never proof. Use
`omni agent --redteam` when tool-driven inspection and deterministic checks are
needed; HTTP checks still require an explicit host allowlist.
For source grounding, supply files explicitly with repeatable `--source-file`
options. Red-team answers are asked for structured `FINDING` / `FILE` / `LINE` /
`QUOTE` blocks; OmniCLI checks those lines and quotes against supplied files and
reports `quotes_grounded`, `unsupported_citations`, `unstructured_claims`,
`no_structured_findings`, or `no_source_provided`. Malformed citation syntax and
additional unstructured claims are not treated as an empty finding set.
Quote grounding confirms only that text exists at that location—it does not
prove reachability, exploitability, impact, or bounty eligibility. Unstructured
model claims remain ungrounded hypotheses.
Red-team runs also apply a fail-closed **evidence gate**: missing source,
provider-stage failures, malformed/unstructured candidate claims, or citations
that do not match the supplied file cause a prominent `EVIDENCE GATE: FAIL`
and CLI exit code `2`. `PASS_GROUNDING_ONLY` means only that the supplied
outputs passed these citation checks; it is not a security verdict. A red-team
result without a gate is itself treated as a failure.
Deliberate mode costs at least two member calls per selected model plus
synthesis when more than one member response is available, and currently does
not support checkpoint/resume. Member and chair outputs are checked for quoted
source claims when source files are supplied. It is an experiment, not a
quality guarantee; compare it against single-pass modes on the same reviewed
tasks before relying on its additional latency or cost.

The staged review and smoke-case design draw on the risk taxonomies and
evaluation framing in [OpenRedTeaming](https://github.com/Libr-AI/OpenRedTeaming),
the resource map in [Awesome-LLM-Red-Teaming](https://github.com/user1342/Awesome-LLM-Red-Teaming),
the explicit challenge objectives in Microsoft's
[AI Red Teaming Playground Labs](https://github.com/microsoft/AI-Red-Teaming-Playground-Labs),
and the category/severity metadata pattern in
[llm-red-teaming-dataset](https://github.com/navirocker/llm-red-teaming-dataset).
These are design references, not runtime dependencies. OmniCLI's regression
fixtures are purpose-built source-security cases; it does not import the
separate dataset's harmful-content prompt payloads or copy the playground app.
All four repositories currently expose MIT top-level licenses; check nested or
upstream third-party materials separately before reusing their code or assets.

### Dependency-aware workflows

Use `omni workflow run` for a small JSON DAG when work has dependencies as well
as independent parallel steps:

```json
{
  "tasks": [
    {"id": "inspect", "prompt": "Inspect the supplied project context.", "writes": ["review"]},
    {"id": "challenge", "prompt": "Find weaknesses in the review.", "depends_on": ["inspect"], "writes": ["critique"]},
    {"id": "summarize", "prompt": "Summarize the review and critique.", "depends_on": ["inspect", "challenge"]}
  ]
}
```

```bash
omni workflow run workflow.json --max-workers 4 --output workflow-result.json
```

Ready independent tasks run concurrently. A task receives the text outputs of
its direct dependencies; a failed task causes its descendants to be skipped,
while independent branches continue. Declare shared artifact names in `writes`:
unordered tasks that both write the same name are rejected before model calls.
Dependencies establish an order and allow an intentional downstream rewrite.
The JSON report includes each task's result/status and an ordered event log.
Workflow execution is advisory model orchestration—outputs still require review.

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
- Red-team agent mode exposes `analyze_python_security`, a bounded Python AST
  check for direct function-parameter/request-value interpolation into SQL
  execution sinks. Results are candidate source evidence, not confirmed
  vulnerabilities; indirect flows and framework-specific wrappers may be
  missed, and no finding is not proof of safety.
- Red-team swarm synthesis now receives the original enriched task/source
  evidence separately from the untrusted member hypotheses, rather than only a
  consolidation instruction. This improves grounding but remains model
  synthesis, not deterministic vulnerability proof.
- Run it directly with `omni verify python-sql path/to/module.py` (optionally
  scoped with `--function name`).
- Chat-completion requests retry transient rate-limit, server, and network
  failures up to three attempts, honoring `Retry-After` where available before
  using the configured model fallback roster. Permanent client errors are not
  retried, and model calls are not silently redirected to an unconfigured
  provider.
- Swarm runs can opt in to private local SQLite checkpoints and lifecycle events:

  ```bash
  omni swarm --mode redteam --checkpoint ./audit-run.sqlite3 "Review ./src"
  omni swarm --mode redteam --checkpoint ./audit-run.sqlite3 --resume "Review ./src"
  ```

  Checkpoints store the prompt (which can include source code) and model
  responses on disk, unencrypted. Protect the file accordingly. Resume requires
  the same mode, prompt, model roster, and run options; it reuses successful
  member results and retries failed members. No checkpoint is written unless
  `--checkpoint` is provided.
- Verification fails **closed**: no proof means no pass.
- Memory/history files are owner-only.

See `docs/memory_heartbeat_plan.md` for the memory/heartbeat design.

### Native orchestration patterns

OmniCLI adapts selected patterns from the reviewed projects without importing
their frameworks as runtime dependencies:

| Reference | OmniCLI application | State |
|---|---|---|
| Agent Squad, TrustGraph | Keep original source/evidence available to the synthesis step and clearly isolate untrusted model hypotheses. | Implemented; deterministic citation validation remains future work. |
| Microsoft Agent Framework, LangGraph | Opt-in local run checkpoints, resumable successful members, retry of failed members, and ordered lifecycle events. | Implemented for swarm runs; workflow DAG checkpoints remain future work. |
| Dynamiq | Dependency-aware task execution, parallel independent work, failure propagation, and conflicting-write detection. | Basic declarative workflow DAG implemented. |
| Solace Agent Mesh | Event-driven visibility into run/member/synthesis lifecycle without requiring its deprecated Python runtime. | Basic local event log implemented; external event transport is not. |
| LiteLLM | Bounded shared provider retries and transparent configured-roster failover. | Shared retry transport implemented; broader adapter unification remains planned. |
| RouteLLM | Compare models using measured security-audit quality and optional per-model price data, not intuition or general-chat scores. | Offline-labeled evaluation and recommendation command implemented; automatic routing remains disabled. |

Checkpoint files contain prompts (which may include source code) and model
responses in plaintext. They are opt-in and created with owner-only file
permissions; do not store sensitive prompts in a location accessible to others.

### Labeled security evaluation

Compare candidates on an operator-reviewed JSONL dataset:

```bash
omni evaluate-security \
  --dataset security-cases.jsonl \
  --model openrouter:provider/model-a \
  --model nvidia:provider/model-b \
  --static-baseline \
  --staged-redteam \
  --minimum-cases 20 \
  --repetitions 2 \
  --output evaluation-results.json
```

Each non-empty line is one case. The dataset must include both positive and clean
cases. Optional `split` metadata must be `development` or `held_out`; cases using
it also require a category, positive severity and preconditions, or a clean-case
rationale. Expected evidence must match the labeled source line; a true positive
requires an exact category, file, line, and quoted evidence match:

```json
{"id":"sqli-01","file":"app/query.py","prompt":"Review this function for SQL injection.","source":"def find(value):\n    cursor.execute(f\"SELECT {value}\")\n","expected_findings":[{"category":"sql_injection","file":"app/query.py","line":2,"evidence":"cursor.execute(f\"SELECT {value}\")"}]}
{"id":"clean-01","file":"app/safe.py","prompt":"Review this function for SQL injection.","source":"def find(value):\n    cursor.execute(\"SELECT * FROM t WHERE id = %s\", (value,))\n","expected_findings":[]}
```

The command reports precision, recall, F1, clean-case specificity, exact
source-line quote accuracy, per-category scores, 95% Wilson intervals, request
errors, p50 latency, and cost only when an optional `--pricing` JSON file
supplies per-model `input_per_million` / `output_per_million` rates and provider
responses include token usage. The JSON report preserves each case's
normalized predicted findings and citation quotes for adjudication.
`--staged-redteam` adds the actual four-focus attack plus challenge pipeline for
every case and model, reported beside the single-pass and optional static
results. This uses at least five model calls per case per repetition; request
it deliberately and include its latency/cost in comparisons. Stage or
evidence-gate failures count as failed trials and expected findings are scored
as missed. Staged results are not eligible for the single-pass model
recommendation.
`--static-baseline` adds a local deterministic pattern baseline using the same
cases; its candidates are heuristics, not security verdicts or model
recommendations. `--repetitions` repeats every model/case pair (default 1);
errors remain explicit and block recommendations. Recommendations are
advisory, require the configured minimum number of error-free cases and at
least one exact match, and do not change normal model routing. The evaluator
can compare staged versus single-pass behavior but has not yet established
whether orchestration improves quality on representative or held-out cases. A
benchmark is only as reliable as its independently reviewed labels and
representative cases. The self-authored development fixtures at
`tests/fixtures/redteam/cases.dev.jsonl` are for tuning, not a held-out quality
claim. See `docs/REDTEAM_QUALITY_PLAN.md` for
the bounded transition gate and current limitations.
The candidate 60-case held-out set in
`tests/fixtures/redteam/cases.held_out.provisional.jsonl` is provisional, not
approved for scoring. A second reviewer must independently verify every gold
label, line, rationale, and precondition before updating its adjudication
metadata; the gate refuses to call providers on provisional labels.

To run the transition check after assembling a held-out JSONL set:

```bash
omni evaluate-security \
  --dataset security-held-out.jsonl \
  --split held_out \
  --model openrouter:provider/model-a \
  --model nvidia:provider/model-b \
  --model openrouter:provider/model-c \
  --staged-redteam \
  --quality-gate \
  --repetitions 2 \
  --output quality-gate-results.json
```

The gate requires at least 30 positive and 30 benign held-out cases, paired
single-pass/staged results from at least three healthy model families, and two
or more repetitions. Every held-out case must also record
`"adjudication_status":"independently_reviewed"` and `"reviewer_count":2`;
provisional labels are rejected before provider calls. It exits with status 2
on failure. A pass is bounded to those labels, models, and settings; it does
not establish universal audit quality.
