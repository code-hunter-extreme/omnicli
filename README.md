# OmniCLI — Local Multi-Provider AI Terminal & Unified Router

A fast, lightweight, and extensible local CLI tool for seamlessly querying, comparing, and routing between 6 OpenAI-compatible AI providers:

- **OpenRouter** (`https://openrouter.ai/api/v1`)
- **NVIDIA Build / NIM** (`https://integrate.api.nvidia.com/v1`)
- **Cloudflare Workers AI** (`https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1`)
- **Bynara** (`https://router.bynara.id/v1`)
- **Apinex** (`https://api.apinex.bond/v1`)
- **XKiro** (`https://api.xkiro.com/v1`)

---

## Quick Start

The command is installed in your `$PATH` as `omni` (also aliased as `omnicli` and `nexus`):

```bash
# 1. Interactive terminal chat (with live markdown streaming)
omni

# 2. Check health and latency across all 6 providers
omni ping

# 3. View configuration & masked keys
omni config show

# 4. Set API keys
omni config set-key nvidia nvapi-...
omni config set-key xkiro sk-...
omni config set-key bynara sk-...
omni config set-key apinex sk-...
omni config set-key cloudflare <API_TOKEN>
omni config set-account <CLOUDFLARE_ACCOUNT_ID>

# 5. One-shot command line query
omni -p nvidia -m meta/llama-3.3-70b-instruct "Write a quicksort in C"

# 6. Pipe code/logs into any provider
cat error.log | omni -p xkiro "What caused this error?"

# 7. Start local OpenAI proxy server for Aider, Cursor, Open-WebUI, etc.
omni serve --port 8000
```

---

## Using The APIs Together

### 1. In Interactive Chat (`omni`)
Within the interactive REPL, switch providers and models on the fly while retaining conversation history:
- `/provider nvidia` (or `/p openrouter`, `/p xkiro`, etc.)
- `/model meta/llama-3.3-70b-instruct`
- `/models llama` (searches live model catalog on the current provider)
- `/system <prompt>` (updates system instructions)
- `/file <path>` (attaches file contents into context)
- `/ping` (tests latency to all providers)

### 2. In Local Proxy Mode (`omni serve`)
External tools can connect to `http://localhost:8000/v1` and route by prefix:
- `nvidia/meta/llama-3.3-70b-instruct`
- `openrouter/cohere/north-mini-code:free`
- `xkiro/minimax/minimax-m3`
- `bynara/gpt-4o`
- `apinex/gpt-4o`
