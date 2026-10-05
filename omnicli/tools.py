"""
Agentic Tool execution engine for OmniCLI.
Provides standard OpenAI function schemas and local execution for:
- bash command execution
- file reading and writing
- local Program Registry search (6,400+ tools)
- provider-to-provider LLM delegation (cross-API synergy)
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


from omnicli.deterministic import HTTPVerifier, CodeVerifier, CrossModelConsensusVerifier, ScopeGate


REGISTRY_DB_PATH = Path("/home/phantomcore/program_registry/core/program_registry.db")
if not REGISTRY_DB_PATH.exists():
    REGISTRY_DB_PATH = Path("/home/phantomcore/program_registry.db")


def tool_verify_http(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    expected: str = "",
    allow_local: bool = False,
) -> str:
    """Execute deterministic HTTP verification and return verdict.

    A ScopeGate is always applied so a model-chosen URL cannot reach internal
    or cloud-metadata addresses (SSRF). allow_local (--allow-local) opts into
    loopback + RFC1918 targets for deliberate local testing; link-local
    (metadata) stays blocked regardless.
    """
    gate = ScopeGate(allow_localhost=allow_local, allow_private=allow_local)
    res = HTTPVerifier.verify(
        url=url,
        method=method,
        headers=headers,
        body=body,
        expected_str=expected,
        scope_gate=gate,
    )
    return json.dumps(res, indent=2)


def _coerce_patterns(val: Any) -> Optional[List[str]]:
    """Normalise a must_(not_)contain argument into a list of strings.

    Models frequently pass a JSON-encoded list string (e.g. '["a","b"]') or a
    single bare pattern instead of an array; both are coerced to a real list so
    the verifier does not iterate a string character-by-character.
    """
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except json.JSONDecodeError:
                pass
        return [val]
    return [str(val)]


def tool_verify_code(
    path: str,
    must_contain: Optional[List[str]] = None,
    must_not_contain: Optional[List[str]] = None,
    target_function: Optional[str] = None,
    check_type: Optional[str] = None,
) -> str:
    """Execute deterministic code pattern verification and return verdict."""
    res = CodeVerifier.verify_file(
        file_path=path,
        must_contain=_coerce_patterns(must_contain),
        must_not_contain=_coerce_patterns(must_not_contain),
        target_function=target_function,
        check_type=check_type,
    )
    return json.dumps(res, indent=2)


def tool_verify_consensus(
    claim: str,
    evidence: str,
    models: Optional[List[str]] = None,
    client_ref: Any = None,
) -> str:
    """Run cross-model / cross-provider verification consensus."""
    if client_ref is None:
        return "Error: Client reference required for consensus verification."
    model_specs = []
    if models:
        for m in models:
            if ":" in m:
                p, mod = m.split(":", 1)
                model_specs.append((p.strip(), mod.strip()))
            else:
                model_specs.append((m.strip(), None))
    verifier = CrossModelConsensusVerifier(client_ref)
    res = verifier.verify_finding(claim, evidence, models=model_specs or None)
    return json.dumps(res, indent=2)


# Destructive / exfil-to-shell command patterns refused even when the shell is
# enabled. This is a backstop against a prompt-injected model, not a full sandbox.
_BASH_DENY_PATTERNS = [
    r"\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+(/|~|\$HOME|\*)",  # rm -rf / ~ * $HOME
    r"\brm\s+-[a-z]*f[a-z]*r[a-z]*\s+(/|~|\$HOME|\*)",  # rm -fr ...
    r"\bmkfs\b",                                        # format a filesystem
    r"\bdd\b[^\n]*\bof=/dev/",                          # overwrite a block device
    r">\s*/dev/(sd|nvme|hd|vd)",                        # clobber a disk
    r":\s*\(\s*\)\s*\{.*\}\s*;\s*:",                    # fork bomb :(){ :|:& };:
    r"\b(shutdown|reboot|halt|poweroff|init\s+0)\b",   # host power control
    r"\bchmod\s+-R\s+0*777\s+/",                        # chmod -R 777 /
    r"\bchown\s+-R\b[^\n]*\s+/\s*$",                    # chown -R ... /
    r"(curl|wget)\b[^|;&\n]*\|\s*(sudo\s+)?(ba)?sh",   # pipe a download into a shell
    r">\s*/etc/",                                       # overwrite system config
]


def _bash_denied(command: str) -> Optional[str]:
    """Return the matched deny-pattern if command is destructive, else None."""
    for pat in _BASH_DENY_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return pat
    return None


def tool_bash(command: str, timeout: int = 30, allow_shell: bool = False) -> str:
    """Execute a bash command in the local environment.

    Shell execution is OFF by default: the agent must be started with
    --allow-shell (allow_shell=True) to enable it. Even when enabled, commands
    matching _BASH_DENY_PATTERNS are refused as a backstop against a
    prompt-injected model steering the agent into a destructive command.
    """
    if not allow_shell:
        return (
            "Error: shell execution is disabled. Re-run the agent with "
            "--allow-shell to enable the bash tool."
        )
    denied = _bash_denied(command)
    if denied:
        return (
            f"Error: command refused by safety policy (matched deny pattern: {denied}). "
            "Destructive or download-to-shell commands are blocked."
        )
    try:
        try:
            timeout = int(timeout) if timeout not in (None, "None", "") else 30
        except (ValueError, TypeError):
            timeout = 30
        res = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path.home()),
        )
        out = res.stdout
        err = res.stderr
        code = res.returncode
        result = []
        if out:
            result.append(f"STDOUT:\n{out}")
        if err:
            result.append(f"STDERR:\n{err}")
        result.append(f"Exit Code: {code}")
        return "\n".join(result)
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing bash: {e}"


def check_workspace_jail(path: str | Path) -> Path:
    """Validate path and prevent access to sensitive host directories."""
    p = Path(path).expanduser().resolve()
    blocked_prefixes = [
        Path("/etc"), Path("/proc"), Path("/sys"), Path("/dev"),
        Path.home() / ".ssh", Path.home() / ".gnupg", Path.home() / ".aws"
    ]
    for bp in blocked_prefixes:
        try:
            if p == bp or p.is_relative_to(bp):
                raise PermissionError(f"Security Policy: path '{path}' resolves inside protected directory '{bp}'")
        except AttributeError:
            if str(p).startswith(str(bp)):
                raise PermissionError(f"Security Policy: path '{path}' resolves inside protected directory '{bp}'")
    return p


def format_tool_error(code: str, message: str, remedy: str = "") -> str:
    """Standardized tool error envelope for immediate agent comprehension."""
    err = {
        "status": "error",
        "code": code,
        "message": message,
    }
    if remedy:
        err["remedy"] = remedy
    return f"Error: {message}\n" + json.dumps(err, indent=2)


def tool_read_file(path: str, max_lines: int = 300) -> str:
    """Read the contents of a local file with path jail verification."""
    try:
        p = check_workspace_jail(path)
    except PermissionError as pe:
        return format_tool_error("E_PERMISSION", str(pe), "Choose a path inside the project workspace.")

    if not p.exists():
        return format_tool_error("E_NOT_FOUND", f"File '{path}' does not exist.", "Check file path spelling and location.")
    if not p.is_file():
        return format_tool_error("E_NOT_FILE", f"'{path}' is not a regular file.", "Specify a file path, not a directory.")
    try:
        try:
            max_lines = int(max_lines) if max_lines not in (None, "None", "") else 300
        except (ValueError, TypeError):
            max_lines = 300
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = [f.readline() for _ in range(max_lines)]
        total_chars = sum(len(l) for l in lines)
        header = f"--- {p} (showing first {len(lines)} lines, {total_chars} chars) ---\n"
        return header + "".join(lines)
    except Exception as e:
        return format_tool_error("E_IO", f"Error reading file: {e}")


def net_open_brackets(text: str) -> int:
    """Net count of unclosed (, [, and {."""
    open_count = 0
    for ch in text:
        if ch in "([{":
            open_count += 1
        elif ch in ")]}":
            open_count -= 1
    return open_count


def detect_truncated_rewrite(old_content: str, new_content: str) -> Optional[Dict[str, Any]]:
    """Truncated write guard: refuses replacing a file with a much shorter copy
    that stops mid-expression with unclosed syntax."""
    old_lines = len(old_content.splitlines())
    if old_lines < 30:
        return None
    if len(new_content) >= len(old_content) * 0.6:
        return None
    if net_open_brackets(old_content) != 0:
        return None
    open_brackets = net_open_brackets(new_content)
    if open_brackets <= 0:
        return None
    trimmed = new_content.rstrip()
    last_line = trimmed.splitlines()[-1].strip() if trimmed.splitlines() else ""
    return {
        "old_lines": old_lines,
        "new_lines": len(new_content.splitlines()),
        "open_brackets": open_brackets,
        "last_line": last_line[-100:] if len(last_line) > 100 else last_line,
    }


def tool_write_file(path: str, content: str) -> str:
    """Write or overwrite a local file with truncation and jail protection."""
    try:
        p = check_workspace_jail(path)
    except PermissionError as pe:
        return format_tool_error("E_PERMISSION", str(pe), "Choose a path inside the project workspace.")

    if p.exists() and p.is_file():
        try:
            old_c = p.read_text(encoding="utf-8", errors="replace")
            trunc = detect_truncated_rewrite(old_c, content)
            if trunc:
                return (
                    f"Error: Refused to write '{p}': The new content ({trunc['new_lines']} lines) "
                    f"is far shorter than current file ({trunc['old_lines']} lines) and ends with "
                    f"{trunc['open_brackets']} unclosed bracket(s) (last line: '{trunc['last_line']}'). "
                    f"Truncated-write-guard prevented file corruption. Re-issue full content."
                )
        except Exception:
            pass
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {p}"
    except Exception as e:
        return format_tool_error("E_IO", f"Error writing file: {e}")


def tool_registry_search(query: str, limit: int = 10) -> str:
    """Search the Program Registry database (~6,400 tools and programs)."""
    if not REGISTRY_DB_PATH.exists():
        return f"Program registry database not found at {REGISTRY_DB_PATH}"

    try:
        try:
            limit = int(limit) if limit not in (None, "None", "") else 10
        except (ValueError, TypeError):
            limit = 10
        conn = sqlite3.connect(str(REGISTRY_DB_PATH))
        c = conn.cursor()
        # Query programs table
        c.execute(
            """
            SELECT id, name, category, description, path, is_critical
            FROM programs
            WHERE name LIKE ? OR description LIKE ? OR category LIKE ?
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", f"%{query}%", limit),
        )
        rows = c.fetchall()
        conn.close()

        if not rows:
            return f"No programs or tools found matching '{query}' in registry."

        results = []
        for r in rows:
            results.append(
                f"• ID: {r[0]} | Name: {r[1]} | Category: {r[2]} | Path: {r[4] or 'N/A'}\n  Description: {r[3] or 'No description'}"
            )
        return f"Found {len(rows)} tools in Program Registry matching '{query}':\n" + "\n".join(results)
    except Exception as e:
        return f"Database query error: {e}"


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in the local environment and return stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command to run",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a local file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"},
                    "max_lines": {"type": "integer", "description": "Max lines to return (default 300)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Full file content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "registry_search",
            "description": "Search the local Program Registry database containing over 6,400 system tools, binaries, and dependencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Tool name, keyword, or category to search for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consult_provider",
            "description": "Delegate a sub-task or question to another AI provider (e.g. ask NVIDIA NIM for CUDA/high-end reasoning, XKiro for Minimax/Grok, or OpenRouter for diverse models).",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Target provider: 'groq', 'openrouter', 'nvidia', 'cloudflare', 'bynara', 'apinex', or 'xkiro'",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model ID to use on the target provider (optional, uses default if omitted)",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The specific query, task, or code snippet to ask the other provider",
                    },
                },
                "required": ["provider", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_http",
            "description": "Deterministically verify an HTTP endpoint behavior against an Expected prediction string (status code, body predicates, header requirements). Returns strict binary CONFIRMED or REJECTED verdict.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Target HTTP/HTTPS URL"},
                    "method": {"type": "string", "description": "HTTP method (GET, POST, etc.)", "default": "GET"},
                    "expected": {
                        "type": "string",
                        "description": "Prediction string, e.g.: 'status 200, body contains \"user_id\", header x-frame-options absent'",
                    },
                    "body": {"type": "string", "description": "Optional request body"},
                },
                "required": ["url", "expected"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_code",
            "description": "Deterministically verify source code patterns or security guards in local files. Inspects AST/code for required authorization checks or forbidden sinks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to local code file to inspect"},
                    "must_contain": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of string patterns that MUST be present in the target code",
                    },
                    "must_not_contain": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of string patterns that MUST NOT be present",
                    },
                    "target_function": {"type": "string", "description": "Optional function name to scope inspection"},
                    "check_type": {
                        "type": "string",
                        "description": "Optional built-in check type (e.g. 'wp_auth_check')",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_consensus",
            "description": "Run blind verification consensus across multiple AI models / model families (same API key or cross-provider) to corroboratively validate or reject a security hypothesis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "The specific vulnerability or behavior claim to evaluate"},
                    "evidence": {"type": "string", "description": "Code snippet, HTTP trace, or execution evidence"},
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of provider:model specs (e.g. ['cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast', 'cloudflare:@cf/qwen/qwen2.5-coder-32b-instruct', 'nvidia:meta/llama-3.2-11b-vision-instruct'])",
                    },
                },
                "required": ["claim", "evidence"],
            },
        },
    },
]


def execute_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    client_ref: Any = None,
    allow_shell: bool = False,
    allow_local: bool = False,
) -> str:
    """Execute a function call by name and arguments."""
    if tool_name == "bash":
        return tool_bash(command=arguments.get("command", ""), allow_shell=allow_shell)
    elif tool_name == "read_file":
        return tool_read_file(path=arguments.get("path", ""), max_lines=arguments.get("max_lines", 300))
    elif tool_name == "write_file":
        return tool_write_file(path=arguments.get("path", ""), content=arguments.get("content", ""))
    elif tool_name == "registry_search":
        return tool_registry_search(query=arguments.get("query", ""), limit=arguments.get("limit", 10))
    elif tool_name == "verify_http":
        return tool_verify_http(
            url=arguments.get("url", ""),
            method=arguments.get("method", "GET"),
            headers=arguments.get("headers"),
            body=arguments.get("body"),
            expected=arguments.get("expected", ""),
            allow_local=allow_local,
        )
    elif tool_name == "verify_code":
        return tool_verify_code(
            path=arguments.get("path", ""),
            must_contain=arguments.get("must_contain"),
            must_not_contain=arguments.get("must_not_contain"),
            target_function=arguments.get("target_function"),
            check_type=arguments.get("check_type"),
        )
    elif tool_name == "verify_consensus":
        return tool_verify_consensus(
            claim=arguments.get("claim", ""),
            evidence=arguments.get("evidence", ""),
            models=arguments.get("models"),
            client_ref=client_ref,
        )
    elif tool_name == "consult_provider":
        if client_ref is None:
            return "Error: Client reference not provided to execute cross-provider consultation."
        provider = arguments.get("provider", "openrouter").lower()
        model = arguments.get("model")
        prompt = arguments.get("prompt", "")
        try:
            res = client_ref.chat_complete(
                provider_id=provider,
                messages=[
                    {"role": "system", "content": "You are a specialized expert model providing a direct, precise answer."},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                max_tokens=2048,
            )
            choices = res.get("choices", [])
            if choices:
                return f"[{provider.upper()} RESPONSE]:\n" + choices[0]["message"]["content"]
            return f"[{provider.upper()} RESPONSE]: No content returned."
        except Exception as e:
            return f"Error consulting {provider}: {e}"
    else:
        return f"Unknown tool: '{tool_name}'"

