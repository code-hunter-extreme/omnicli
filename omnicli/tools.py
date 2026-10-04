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
import os
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


REGISTRY_DB_PATH = Path("/home/phantomcore/program_registry/core/program_registry.db")
if not REGISTRY_DB_PATH.exists():
    REGISTRY_DB_PATH = Path("/home/phantomcore/program_registry.db")


def tool_bash(command: str, timeout: int = 30) -> str:
    """Execute a bash command in the local environment."""
    try:
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


def tool_read_file(path: str, max_lines: int = 300) -> str:
    """Read the contents of a local file."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"Error: File '{path}' does not exist."
    if not p.is_file():
        return f"Error: '{path}' is not a file."
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = [f.readline() for _ in range(max_lines)]
        total_chars = sum(len(l) for l in lines)
        header = f"--- {p} (showing first {len(lines)} lines, {total_chars} chars) ---\n"
        return header + "".join(lines)
    except Exception as e:
        return f"Error reading file: {e}"


def tool_write_file(path: str, content: str) -> str:
    """Write or overwrite a local file."""
    p = Path(path).expanduser().resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {p}"
    except Exception as e:
        return f"Error writing file: {e}"


def tool_registry_search(query: str, limit: int = 10) -> str:
    """Search the Program Registry database (~6,400 tools and programs)."""
    if not REGISTRY_DB_PATH.exists():
        return f"Program registry database not found at {REGISTRY_DB_PATH}"

    try:
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
                        "description": "Target provider: 'openrouter', 'nvidia', 'cloudflare', 'bynara', 'apinex', or 'xkiro'",
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
]


def execute_tool_call(tool_name: str, arguments: Dict[str, Any], client_ref: Any = None) -> str:
    """Execute a function call by name and arguments."""
    if tool_name == "bash":
        return tool_bash(command=arguments.get("command", ""))
    elif tool_name == "read_file":
        return tool_read_file(path=arguments.get("path", ""), max_lines=arguments.get("max_lines", 300))
    elif tool_name == "write_file":
        return tool_write_file(path=arguments.get("path", ""), content=arguments.get("content", ""))
    elif tool_name == "registry_search":
        return tool_registry_search(query=arguments.get("query", ""), limit=arguments.get("limit", 10))
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
