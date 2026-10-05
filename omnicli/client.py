"""
Unified HTTP / SSE client for OpenAI-compatible LLM endpoints.
Handles streaming, headers, model discovery, authentication, and error reporting.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple

import httpx

from omnicli.config import ConfigManager


class ProviderError(Exception):
    def __init__(self, provider: str, status_code: Optional[int], message: str) -> None:
        self.provider = provider
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{provider.upper()}] Error {status_code or ''}: {message}")


class OmniClient:
    def __init__(self, config_manager: Optional[ConfigManager] = None) -> None:
        self.cfg = config_manager or ConfigManager()

    def get_headers(self, provider_id: str) -> Dict[str, str]:
        provider_id = provider_id.lower()
        key = self.cfg.get_api_key(provider_id)

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "OmniCLI/1.0",
        }

        if key:
            headers["Authorization"] = f"Bearer {key}"

        if provider_id == "monkeycode" and key:
            headers["Cookie"] = f"monkeycode_ai_session={key}"

        if provider_id == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/phantomcore/omnicli"
            headers["X-Title"] = "OmniCLI Local Assistant"

        return headers

    def get_endpoint_url(self, provider_id: str, path: str) -> str:
        base = self.cfg.get_base_url(provider_id).rstrip("/")
        path = path.lstrip("/")
        return f"{base}/{path}"

    def list_models(self, provider_id: str, timeout: float = 10.0) -> List[Dict[str, Any]]:
        """Fetch available models from the provider.

        Cloudflare uses a different endpoint than the OpenAI-compatible /models:
          GET /accounts/{id}/ai/models/search?task=Text+Generation&per_page=100
        All other providers use the standard /v1/models path.
        """
        provider_id = provider_id.lower()

        if provider_id == "cloudflare":
            account_id = self.cfg.get_cloudflare_account_id()
            if not account_id:
                raise ProviderError(
                    provider_id, None,
                    "Cloudflare Account ID not set. Run: omni config set-account <ACCOUNT_ID>",
                )
            return self._list_cloudflare_models(account_id, timeout=timeout)

        url = self.get_endpoint_url(provider_id, "models")
        headers = self.get_headers(provider_id)

        try:
            with httpx.Client(timeout=timeout) as client:
                res = client.get(url, headers=headers)
                if res.status_code in (401, 403):
                    raise ProviderError(
                        provider_id, res.status_code,
                        f"Authentication required for {provider_id}. "
                        f"Set key via: omni config set-key {provider_id} <KEY>",
                    )
                if res.status_code != 200:
                    raise ProviderError(provider_id, res.status_code, res.text[:300])

                data = res.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], list):
                        return data["data"]
                    if "result" in data and isinstance(data["result"], list):
                        return data["result"]
                    return []
                return []
        except httpx.RequestError as e:
            raise ProviderError(provider_id, None, f"Network connection failed: {e}")

    def _list_cloudflare_models(self, account_id: str, timeout: float = 10.0) -> List[Dict[str, Any]]:
        """Cloudflare model listing via /ai/models/search — returns text-generation models only.

        The standard /ai/v1/models returns 405. Cloudflare's actual catalog endpoint is:
          GET /accounts/{id}/ai/models/search?task=Text+Generation&per_page=100
        Returns OpenAI-compatible dicts: {"id": model_name, ...}
        """
        base = f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        url = f"{base}/ai/models/search"
        headers = self.get_headers("cloudflare")
        params = {"task": "Text Generation", "per_page": "100"}

        try:
            with httpx.Client(timeout=timeout) as client:
                res = client.get(url, headers=headers, params=params)
                if res.status_code in (401, 403):
                    raise ProviderError("cloudflare", res.status_code, "Authentication failed")
                if res.status_code != 200:
                    raise ProviderError("cloudflare", res.status_code, res.text[:300])
                data = res.json()
                raw = data.get("result", []) if isinstance(data, dict) else []
                # Normalise to OpenAI-compatible shape: {"id": name, "name": name, ...}
                return [
                    {
                        "id":   m.get("name", ""),
                        "name": m.get("name", ""),
                        "description": m.get("description", ""),
                    }
                    for m in raw
                    if m.get("name")
                ]
        except httpx.RequestError as e:
            raise ProviderError("cloudflare", None, f"Network connection failed: {e}")

    def ping(self, provider_id: str) -> Dict[str, Any]:
        """Test provider connectivity, measure latency, and count models."""
        provider_id = provider_id.lower()
        has_key = bool(self.cfg.get_api_key(provider_id))

        if provider_id == "cloudflare" and not self.cfg.get_cloudflare_account_id():
            return {
                "status": "config_needed",
                "code": None,
                "latency_ms": None,
                "model_count": 0,
                "has_key": has_key,
                "error": "Set account ID: omni config set-account <ID>",
            }

        url = self.get_endpoint_url(provider_id, "models")
        headers = self.get_headers(provider_id)

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(url, headers=headers)
                latency = round((time.perf_counter() - t0) * 1000, 1)

                if res.status_code == 200:
                    try:
                        data = res.json()
                        count = len(data.get("data", data.get("result", [])))
                    except Exception:
                        count = 0
                    return {
                        "status": "healthy",
                        "code": 200,
                        "latency_ms": latency,
                        "model_count": count,
                        "has_key": has_key,
                        "error": None,
                    }
                elif res.status_code in (401, 403):
                    return {
                        "status": "unauthorized" if has_key else "missing_key",
                        "code": res.status_code,
                        "latency_ms": latency,
                        "model_count": 0,
                        "has_key": has_key,
                        "error": "Authentication failed or key missing",
                    }
                else:
                    return {
                        "status": f"http_{res.status_code}",
                        "code": res.status_code,
                        "latency_ms": latency,
                        "model_count": 0,
                        "has_key": has_key,
                        "error": res.text[:120],
                    }
        except httpx.RequestError as e:
            latency = round((time.perf_counter() - t0) * 1000, 1)
            return {
                "status": "unreachable",
                "code": None,
                "latency_ms": latency,
                "model_count": 0,
                "has_key": has_key,
                "error": str(e),
            }

    def chat_complete(
        self,
        provider_id: str,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Perform a non-streaming chat completion."""
        provider_id = provider_id.lower()
        model_name = model or self.cfg.get_default_model(provider_id)
        url = self.get_endpoint_url(provider_id, "chat/completions")
        headers = self.get_headers(provider_id)

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            with httpx.Client(timeout=timeout) as client:
                res = client.post(url, headers=headers, json=payload)
                if res.status_code != 200:
                    if res.status_code in (400, 404) and any(
                        x in res.text.lower() for x in ("model_not_found", "does not exist", "not have access", "invalid_model")
                    ):
                        try:
                            from omnicli.scanner import scan_provider
                            scan_provider(provider_id, cfg=self.cfg, client=self)
                            new_model = self.cfg.get_default_model(provider_id)
                            if new_model and new_model != payload["model"]:
                                payload["model"] = new_model
                                retry_res = client.post(url, headers=headers, json=payload)
                                if retry_res.status_code == 200:
                                    return retry_res.json()
                        except Exception:
                            pass
                    raise ProviderError(provider_id, res.status_code, res.text)
                return res.json()
        except httpx.RequestError as e:
            raise ProviderError(provider_id, None, f"Request failed: {e}")

    def chat_stream(
        self,
        provider_id: str,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 90.0,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream chat completions token by token.
        Yields dict with {"type": "content"|"usage"|"error", "delta": "...", "raw": ...}
        """
        provider_id = provider_id.lower()
        model_name = model or self.cfg.get_default_model(provider_id)
        url = self.get_endpoint_url(provider_id, "chat/completions")
        headers = self.get_headers(provider_id)

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        err_body = response.read().decode("utf-8", errors="replace")
                        if response.status_code in (400, 404) and any(
                            x in err_body.lower() for x in ("model_not_found", "does not exist", "not have access", "invalid_model")
                        ):
                            try:
                                from omnicli.scanner import scan_provider
                                scan_provider(provider_id, cfg=self.cfg, client=self)
                                new_model = self.cfg.get_default_model(provider_id)
                                if new_model and new_model != payload["model"]:
                                    payload["model"] = new_model
                                    with client.stream("POST", url, headers=headers, json=payload) as retry_resp:
                                        if retry_resp.status_code == 200:
                                            for line in retry_resp.iter_lines():
                                                if not line:
                                                    continue
                                                line_str = line.strip()
                                                if line_str.startswith("data: "):
                                                    data_chunk = line_str[6:].strip()
                                                    if data_chunk == "[DONE]":
                                                        break
                                                    try:
                                                        parsed = json.loads(data_chunk)
                                                        choices = parsed.get("choices", [])
                                                        if choices:
                                                            delta = choices[0].get("delta", {})
                                                            content = delta.get("content", "")
                                                            if content is not None and content != "":
                                                                yield {"type": "content", "delta": str(content), "raw": parsed}
                                                        if "usage" in parsed and parsed["usage"]:
                                                            yield {"type": "usage", "usage": parsed["usage"]}
                                                    except json.JSONDecodeError:
                                                        continue
                                            return
                            except Exception:
                                pass
                        yield {
                            "type": "error",
                            "status_code": response.status_code,
                            "content": f"Provider error ({response.status_code}): {err_body}",
                        }
                        return

                    for line in response.iter_lines():
                        if not line:
                            continue
                        line_str = line.strip()
                        if line_str.startswith("data: "):
                            data_chunk = line_str[6:].strip()
                            if data_chunk == "[DONE]":
                                break
                            try:
                                parsed = json.loads(data_chunk)
                                choices = parsed.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content is not None and content != "":
                                        yield {"type": "content", "delta": str(content), "raw": parsed}
                                if "usage" in parsed and parsed["usage"]:
                                    yield {"type": "usage", "usage": parsed["usage"]}
                            except json.JSONDecodeError:
                                continue
        except httpx.RequestError as e:
            yield {"type": "error", "content": f"Connection error: {e}"}
