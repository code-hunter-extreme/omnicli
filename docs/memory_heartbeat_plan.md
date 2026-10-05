# OmniCLI — Memory + Heartbeat Plan

One plan, four increments. Build order: **A → C → B → D**. Each increment is
independently shippable, TDD, committed on its own. Tests live in
`oracle/tests/test_omnicli_engine.py` (offline).

## Config additions (config.json, top-level)

```json
"memory": {
  "enabled": true,
  "dir": "~/.config/omnicli/memory",
  "global": [],
  "providers": {
    "nvidia": ["cuda.md", "/abs/path/targets.md"]
  },
  "history": {
    "enabled": false,
    "max_turns": 40
  }
},
"heartbeat": {
  "canary": false,
  "status_file": "~/.config/omnicli/heartbeat.json",
  "tasks_file": "~/.config/omnicli/heartbeat_tasks.json"
}
```

Both blocks are optional; accessors return safe defaults when absent so existing
configs keep working. Memory file entries resolve relative to `memory.dir`, or
may be absolute → "multiple files / multiple locations per API".

Decision — a configured-but-missing memory file **warns loudly and is skipped**,
it does not crash the session (memory is additive).

---

## Increment A — Memory: injected context per API  [BUILD FIRST]

**Module `omnicli/memory.py`**
- `load_memory(cfg, provider) -> str`: if `memory.enabled`, gather `global` +
  `providers[provider]` files, read each, concatenate with a `# Memory: <name>`
  header. Missing file → warn + skip. Disabled/none → `""`.
- `memory_file_count(cfg, provider) -> int`: for UI display.

**config.py**: `get_memory_config()`, `get_memory_files(provider)` (global +
provider, resolved paths), default empty `memory` block in `_load`.

**Wiring**: `ChatSession` and `AgentRunner` append `load_memory(...)` to the base
system prompt (after redteam/custom prompt). Banner/panel shows `Memory: N files`.

**Tests**: global+provider merge; missing-file warn; disabled→empty; chat/agent
system prompt contains the memory text.

---

## Increment C — Heartbeat: system status  [BUILD SECOND]

**Module `omnicli/heartbeat.py`**
- `run_heartbeat(cfg, client, canary=False) -> list[dict]`: per provider, reuse
  `client.ping()` (status/latency/model_count); if `canary`, send a tiny prompt
  and record ok/err. Returns one record per provider.
- `write_status(cfg, records)`: dump to `heartbeat.status_file` with a timestamp.

**cli.py**: `omni heartbeat [--canary] [--no-write]` → run, print a rich table,
write the status file. Health checks are the DEFAULT (built in here), separate
from the task file (D).

**Tests**: record shape from a fake client; status-file written; canary path
toggles an extra field. (Offline via fake client.)

---

## Increment B — Memory: persistent history per API  [BUILD THIRD]

**memory.py additions**
- `history_path(cfg, provider) -> Path`: `memory.dir/history/<provider>.jsonl`.
- `append_history(cfg, provider, role, content)`: append one line; opt-in via
  `memory.history.enabled`.
- `load_history(cfg, provider, max_turns) -> list[dict]`: last N turns.

**Wiring**: `ChatSession` appends each user/assistant turn when history enabled;
on start, optionally resume the last `max_turns`. Size-capped by `max_turns`.
`/clear` does not wipe the file; a `/forget` command truncates it.

**Tests**: append then load round-trip; max_turns truncation; disabled→no file.

---

## Increment D — Heartbeat: task-runner  [BUILD LAST]

Health checks are default (C). The **task file is user-populated**; its contents
are whatever the user sets up. v1 is a minimal, extensible runner.

**heartbeat.py additions**
- `load_tasks(cfg) -> list[dict]`: read `heartbeat.tasks_file` (JSON list). Each
  task: `{ "type": "ask|agent|rescan", ...params }`.
- `run_tasks(cfg, client, tasks) -> list[dict]`: dispatch by type, collect
  results, append to the status file. Unknown type → recorded error, not a crash.

**cli.py**: `omni heartbeat --run-tasks` runs default status + the task file.

**Tests**: task dispatch by type with a fake client; unknown type handled;
missing tasks file → empty list.

NOTE: D's task semantics are intentionally minimal/extensible pending a follow-up
spec — confirm the task shapes before relying on them.

---

## Status checklist
- [x] A — memory inject
- [x] C — heartbeat status
- [x] B — memory history
- [x] D — heartbeat tasks  (v1: ask/rescan/agent; task shapes may evolve)
