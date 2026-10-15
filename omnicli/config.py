"""
Configuration manager for OmniCLI.
Handles provider definitions, API keys, endpoints, and persistent settings.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_DIR = Path.home() / ".config" / "omnicli"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history"
TRIO_KEYS_FILE = Path.home() / ".config" / "trio" / "keys.env"

# Model family keywords used by `omni scan` to classify discovered models.
# Order matters: first match wins. Extend here to recognise new families.
MODEL_FAMILY_KEYWORDS: list = [
    ("anthropic",  ["claude"]),
    ("openai",     ["gpt-", "o1-", "o3-", "o4-"]),
    ("meta",       ["llama"]),
    ("google",     ["gemini", "gemma"]),
    ("mistral",    ["mistral", "mixtral"]),
    ("qwen",       ["qwen"]),
    ("deepseek",   ["deepseek"]),
    ("xai",        ["grok"]),
    ("minimax",    ["minimax"]),
    ("nvidia",     ["nemotron", "nv-"]),
    ("cohere",     ["command"]),
    ("databricks", ["dbrx"]),
    ("microsoft",  ["phi-", "phi3", "wizardlm"]),
    ("glm",        ["glm"]),
    ("01ai",       ["yi-"]),
]

DEFAULT_PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "google/gemini-2.5-flash",
        "description": "Multi-model router with 300+ models",
        "requires_auth_for_models": False,
        # Populated by `omni scan` — not hardcoded.
        "fallback_models": [],
        "verification_panel": [],
    },
    "nvidia": {
        "name": "NVIDIA Build / NIM",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_API_KEY",
        "alt_env_keys": ["NVAPI_KEY"],
        "default_model": "meta/llama-3.2-11b-vision-instruct",
        "description": "NVIDIA NIM high-throughput inference endpoints",
        "requires_auth_for_models": False,
        "fallback_models": [],
        "verification_panel": [],
    },
    "cloudflare": {
        "name": "Cloudflare Workers AI",
        "base_url_template": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "env_key": "CLOUDFLARE_API_TOKEN",
        "alt_env_keys": ["CLOUDFLARE_API_KEY"],
        "account_id_env": "CLOUDFLARE_ACCOUNT_ID",
        "default_model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "description": "Cloudflare Workers AI serverless inference",
        "requires_auth_for_models": True,
        "fallback_models": [],
        "verification_panel": [],
    },
    "bynara": {
        "name": "Bynara Router",
        "base_url": "https://router.bynara.id/v1",
        "env_key": "BYNARA_API_KEY",
        "default_model": "agnes-2.5-flash",
        "description": "Bynara AI Gateway & Router (7M tokens/day)",
        "requires_auth_for_models": True,
        "fallback_models": [
            "agnes-3-flash",
            "deepseek-v4-flash-alibaba",
        ],
        "verification_panel": [
            {"model": "agnes-2.5-flash", "family": "google"},
        ],
    },
    "apinex": {
        "name": "Apinex Router",
        "base_url": "https://api.apinex.bond/v1",
        "env_key": "APINEX_API_KEY",
        "default_model": "deepseek-v4-flash",
        "description": "Apinex AI Model Proxy (10M tokens/24h)",
        "requires_auth_for_models": True,
        "fallback_models": [],
        "verification_panel": [],
    },
    "xkiro": {
        "name": "XKiro Router",
        "base_url": "https://api.xkiro.com/v1",
        "env_key": "XKIRO_API_KEY",
        "default_model": "qwen/qwen3.6-27b:free",
        "description": "XKiro Model Router & Gateway (1M/day, 30M pool)",
        "requires_auth_for_models": False,
        "fallback_models": [
            "qwen/qwen3.7-flash:free",
            "inclusionai/ling-3.0-flash-sante:free",
        ],
        "verification_panel": [
            {"model": "qwen/qwen3.6-27b:free", "family": "qwen"},
        ],
    },
    "dahl": {
        "name": "Dahl Inference",
        "base_url": "https://inference.dahl.global/v1",
        "env_key": "DAHL_API_KEY",
        "default_model": "zai-org/GLM-5.3-Flash",
        "description": "Dahl Global high-capacity inference gateway (100M tokens)",
        "requires_auth_for_models": True,
        "fallback_models": [
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            "MiniMaxAI/MiniMax-M2.7",
        ],
        "verification_panel": [
            {"model": "zai-org/GLM-5.3-Flash", "family": "glm"},
            {"model": "deepseek-ai/DeepSeek-V4-Flash-0731", "family": "deepseek"},
        ],
    },
    "llm7": {
        "name": "LLM7 Gateway",
        "base_url": "https://api.llm7.io/v1",
        "env_key": "LLM7_API_KEY",
        "default_model": "default",
        "description": "LLM7 multi-provider gateway (free tier selectors)",
        "requires_auth_for_models": False,
        "fallback_models": [
            "fast",
        ],
        "verification_panel": [
            {"model": "default", "family": "other"},
        ],
    },
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
        "description": "Groq ultra-fast LPU inference endpoints",
        "requires_auth_for_models": True,
        "fallback_models": [
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ],
        "verification_panel": [
            {"model": "openai/gpt-oss-120b", "family": "openai"},
            {"model": "qwen/qwen3.8-27b", "family": "qwen"},
        ],
    },
    "monkeycode": {
        "name": "MonkeyCode AI",
        "base_url": "https://api.monkeycode-ai.net/v1",
        "env_key": "MONKEYCODE_API_KEY",
        "default_model": "monkeycode-basic/qwen3.8-flash",
        "description": "Chaitin MonkeyCode AI Platform (10M daily tokens, Google OAuth alexander.sorrell.it)",
        "requires_auth_for_models": True,
        "fallback_models": [
            "monkeycode-basic/qwen3.8-flash",
            "monkeycode-basic/deepseek-flash",
            "monkeycode-pro/glm-5.2",
        ],
        "verification_panel": [
            {"model": "monkeycode-basic/qwen3.8-flash", "family": "qwen"},
            {"model": "monkeycode-basic/deepseek-flash", "family": "deepseek"},
        ],
    },
}

# Per-provider memory (context injection + optional history). File entries are
# relative to "dir" or absolute, so each API can layer several memory sources.
DEFAULT_MEMORY = {
    "enabled": True,
    "dir": str(CONFIG_DIR / "memory"),
    "global": [],
    "providers": {},
    "history": {"enabled": False, "max_turns": 40},
}

# Heartbeat: built-in health checks (status) + an optional user task file.
DEFAULT_HEARTBEAT = {
    "canary": False,
    "status_file": str(CONFIG_DIR / "heartbeat.json"),
    "tasks_file": str(CONFIG_DIR / "heartbeat_tasks.json"),
}


class ConfigManager:
    """Provider/key settings rooted at a caller-chosen directory.

    config_dir defaults to ~/.config/omnicli (standalone CLI use). Library
    consumers pass their own root (e.g. a tmp dir or the host app's config
    dir) so importing and driving OmniCLI never touches the user's home.
    """

    def __init__(self, config_dir: Optional[Any] = None) -> None:
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
        self.config_file = self.config_dir / "config.json"
        self.history_file = self.config_dir / "history"
        self._ensure_dir()
        self.data: Dict[str, Any] = self._load()

    def _ensure_dir(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            self.history_file.touch(mode=0o600, exist_ok=True)

    def _load(self) -> Dict[str, Any]:
        if not self.config_file.exists():
            initial_data = {
                "default_provider": "groq",
                "providers": copy.deepcopy(DEFAULT_PROVIDERS),
                "keys": {},
                "cloudflare_account_id": "",
                "system_prompt": "You are a helpful, brilliant, concise AI assistant.",
                "temperature": 0.7,
                "timeout": 60,
                "memory": copy.deepcopy(DEFAULT_MEMORY),
                "heartbeat": copy.deepcopy(DEFAULT_HEARTBEAT),
            }
            # Auto-check aider oauth-keys if available
            aider_env = Path.home() / ".aider" / "oauth-keys.env"
            if aider_env.exists():
                try:
                    with open(aider_env, "r", encoding="utf-8") as f:
                        for line in f:
                            if "OPENROUTER_API_KEY=" in line:
                                k = line.split("=", 1)[1].strip().strip("\"'")
                                if k:
                                    initial_data["keys"]["openrouter"] = k
                except Exception:
                    pass

            self._save(initial_data)
            return initial_data

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Ensure all default providers exist
            if "providers" not in data:
                data["providers"] = {}
            for k, v in DEFAULT_PROVIDERS.items():
                if k not in data["providers"]:
                    data["providers"][k] = copy.deepcopy(v)
            if "keys" not in data:
                data["keys"] = {}
            # Migrate older configs: add memory/heartbeat blocks if absent.
            changed = False
            if "memory" not in data:
                data["memory"] = copy.deepcopy(DEFAULT_MEMORY)
                changed = True
            if "heartbeat" not in data:
                data["heartbeat"] = copy.deepcopy(DEFAULT_HEARTBEAT)
                changed = True
            if changed:
                self._save(data)
            return data
        except Exception:
            return {
                "default_provider": "openrouter",
                "providers": copy.deepcopy(DEFAULT_PROVIDERS),
                "keys": {},
                "cloudflare_account_id": "",
                "system_prompt": "You are a helpful, brilliant, concise AI assistant.",
                "temperature": 0.7,
                "timeout": 60,
            }

    def _save(self, data: Optional[Dict[str, Any]] = None) -> None:
        if data is None:
            data = self.data
        # config.json holds provider API keys — it must be owner-only, never
        # world-readable (the history file is already 0600). O_CREAT applies the
        # mode on first write; chmod also tightens a file created before this fix.
        fd = os.open(
            self.config_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        try:
            os.chmod(self.config_file, 0o600)
        except OSError:
            pass

    def save(self) -> None:
        self._save(self.data)

    def get_default_provider(self) -> str:
        return self.data.get("default_provider", "openrouter")

    def set_default_provider(self, provider_id: str) -> None:
        provider_id = provider_id.lower()
        if provider_id not in self.data["providers"]:
            raise ValueError(f"Provider '{provider_id}' is not configured.")
        self.data["default_provider"] = provider_id
        self.save()

    def get_provider_info(self, provider_id: str) -> Dict[str, Any]:
        provider_id = provider_id.lower()
        if provider_id not in self.data["providers"]:
            raise KeyError(f"Unknown provider '{provider_id}'. Available: {list(self.data['providers'].keys())}")
        return self.data["providers"][provider_id]

    def _read_env_files(self) -> Dict[str, str]:
        env_dict: Dict[str, str] = {}
        home = Path.home()
        candidate_paths = [
            TRIO_KEYS_FILE,
            home / "oracle" / ".env",
            home / "oracle-fresh" / ".env",
            home / ".aider" / "oauth-keys.env",
            home / ".config" / "smallcode" / ".env",
        ]
        for path in candidate_paths:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                k = k.strip()
                                v = v.strip().strip("\"'")
                                if k and v and k not in env_dict:
                                    env_dict[k] = v
                except Exception:
                    pass
        return env_dict

    def get_api_key(self, provider_id: str) -> str:
        provider_id = provider_id.lower()
        # 1. Check direct config keys
        key = self.data.get("keys", {}).get(provider_id, "")
        if key:
            return key

        # 2. Check environment variables
        prov = self.data["providers"].get(provider_id, {})
        env_key = prov.get("env_key")
        if env_key and os.getenv(env_key):
            return os.getenv(env_key, "").strip()

        # 3. Check alt env keys
        for alt in prov.get("alt_env_keys", []):
            if os.getenv(alt):
                return os.getenv(alt, "").strip()

        # 4. Check known .env files (e.g., ~/oracle/.env)
        env_dict = self._read_env_files()
        if env_key and env_key in env_dict:
            return env_dict[env_key]

        for alt in prov.get("alt_env_keys", []):
            if alt in env_dict:
                return env_dict[alt]

        return ""

    def set_api_key(self, provider_id: str, key: str) -> None:
        provider_id = provider_id.lower()
        if "keys" not in self.data:
            self.data["keys"] = {}
        self.data["keys"][provider_id] = key.strip()
        self.save()

    def get_cloudflare_account_id(self) -> str:
        acc = self.data.get("cloudflare_account_id", "")
        if acc:
            return acc.strip()
        if os.getenv("CLOUDFLARE_ACCOUNT_ID"):
            return os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()

        env_dict = self._read_env_files()
        if "CLOUDFLARE_ACCOUNT_ID" in env_dict:
            return env_dict["CLOUDFLARE_ACCOUNT_ID"]

        return ""

    def set_cloudflare_account_id(self, account_id: str) -> None:
        self.data["cloudflare_account_id"] = account_id.strip()
        self.save()

    def get_monkeycode_signing_secret(self) -> str:
        if os.getenv("MONKEYCODE_SIGNING_SECRET"):
            return os.getenv("MONKEYCODE_SIGNING_SECRET", "").strip()
        env_dict = self._read_env_files()
        return env_dict.get("MONKEYCODE_SIGNING_SECRET", "")

    def get_monkeycode_session_cookie(self) -> str:
        if os.getenv("MONKEYCODE_SESSION_COOKIE"):
            return os.getenv("MONKEYCODE_SESSION_COOKIE", "").strip()
        env_dict = self._read_env_files()
        return env_dict.get("MONKEYCODE_SESSION_COOKIE", "")

    def get_base_url(self, provider_id: str) -> str:
        provider_id = provider_id.lower()
        prov = self.get_provider_info(provider_id)
        if provider_id == "cloudflare":
            account_id = self.get_cloudflare_account_id()
            if not account_id:
                # default placeholder or instructions
                return "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
            return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        return prov.get("base_url", "")

    def get_default_model(self, provider_id: str) -> str:
        prov = self.get_provider_info(provider_id)
        return prov.get("default_model", "")

    def get_fallback_models(self, provider_id: str) -> list:
        """Ordered fallback model roster for a provider (populated by omni scan)."""
        prov = self.get_provider_info(provider_id)
        return list(prov.get("fallback_models") or [])

    def get_memory_config(self) -> dict:
        """Memory settings block (enabled, dir, global, providers, history)."""
        return dict(self.data.get("memory") or {})

    def get_memory_files(self, provider_id: str) -> list:
        """Configured memory file names for a provider: global first, then its own."""
        mem = self.data.get("memory") or {}
        files = list(mem.get("global") or [])
        files += list((mem.get("providers") or {}).get(provider_id.lower()) or [])
        return files

    def get_heartbeat_config(self) -> dict:
        """Heartbeat settings block (canary, status_file, tasks_file)."""
        return dict(self.data.get("heartbeat") or {})

    def get_verification_panel(self, provider_id: str) -> list:
        """Diverse-family verification panel (populated by omni scan).

        Each entry is {"model": str, "family": str}.
        """
        prov = self.get_provider_info(provider_id)
        return list(prov.get("verification_panel") or [])

    def save_scan_results(self, provider_id: str,
                          fallback_models: list,
                          verification_panel: list) -> None:
        """Persist live scan results (roster + panel) for a provider."""
        provider_id = provider_id.lower()
        prov = self.data["providers"].get(provider_id)
        if prov is None:
            return
        prov["fallback_models"] = fallback_models
        prov["verification_panel"] = verification_panel
        self.save()

    def set_default_model(self, provider_id: str, model_id: str) -> None:
        provider_id = provider_id.lower()
        prov = self.get_provider_info(provider_id)
        prov["default_model"] = model_id.strip()
        self.save()

    def add_custom_provider(
        self,
        provider_id: str,
        base_url: str,
        api_key: Optional[str] = None,
        default_model: str = "default",
        name: Optional[str] = None,
    ) -> None:
        provider_id = provider_id.lower().strip()
        self.data["providers"][provider_id] = {
            "name": name or provider_id.capitalize(),
            "base_url": base_url.rstrip("/"),
            "env_key": f"{provider_id.upper()}_API_KEY",
            "default_model": default_model,
            "description": f"Custom OpenAI-compatible provider: {base_url}",
            "requires_auth_for_models": True,
        }
        if api_key:
            self.set_api_key(provider_id, api_key)
        else:
            self.save()

    def remove_provider(self, provider_id: str) -> bool:
        provider_id = provider_id.lower().strip()
        if provider_id in self.data["providers"]:
            del self.data["providers"][provider_id]
            if provider_id in self.data.get("keys", {}):
                del self.data["keys"][provider_id]
            self.save()
            return True
        return False
