"""
Local OpenAI-compatible proxy server for OmniCLI.
Exposes standard /v1/chat/completions and /v1/models endpoints locally so any
client (Aider, Cursor, Continue, Open-WebUI, curl, etc.) can query all providers.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from omnicli.config import ConfigManager

app = FastAPI(title="OmniCLI Local Proxy", version="1.0.0")

# Restrict browser callers to localhost; a wildcard origin with credentials is
# both invalid per the CORS spec and lets any site drive the operator's keys.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = ConfigManager()


def _require_proxy_auth(request: Request) -> None:
    """Gate the key-forwarding endpoints behind an operator token.

    The proxy injects the operator's stored provider keys, so an unauthenticated
    caller who can reach the port would spend the operator's quota. When
    OMNICLI_PROXY_TOKEN is set, every forwarding request must present it as a
    bearer token. The token is required whenever the server is reachable off
    loopback (see start_proxy), so exposure without a token fails closed.
    """
    expected = os.environ.get("OMNICLI_PROXY_TOKEN")
    if not expected:
        return
    sent = request.headers.get("authorization", "")
    if not sent.startswith("Bearer ") or sent[7:].strip() != expected:
        raise HTTPException(status_code=401, detail="invalid or missing proxy token")


def resolve_provider_and_model(model_id: str) -> tuple[str, str]:
    """
    Parse provider and model from strings like 'nvidia/meta/llama-3.3'
    or 'openrouter/google/gemini-2.5-flash'.
    If no prefix matches a known provider, use the configured default provider.
    """
    providers = cfg.data.get("providers", {})
    parts = model_id.split("/", 1)
    if len(parts) == 2 and parts[0].lower() in providers:
        return parts[0].lower(), parts[1]

    # Check if exact provider match
    if model_id.lower() in providers:
        p = model_id.lower()
        return p, cfg.get_default_model(p)

    # Fallback to default provider
    default_p = cfg.get_default_provider()
    return default_p, model_id


@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "service": "OmniCLI Local OpenAI Gateway",
        "default_provider": cfg.get_default_provider(),
        "providers": list(cfg.data.get("providers", {}).keys()),
    }


@app.get("/v1/models")
async def list_models(request: Request):
    """Aggregate or list models from all providers or default provider."""
    # Check if user specified a provider in query param, e.g. /v1/models?provider=nvidia
    target_prov = request.query_params.get("provider")
    results = []

    providers_to_check = [target_prov] if target_prov and target_prov in cfg.data.get("providers", {}) else list(cfg.data.get("providers", {}).keys())

    async with httpx.AsyncClient(timeout=8.0) as client:
        for pid in providers_to_check:
            base_url = cfg.get_base_url(pid).rstrip("/")
            key = cfg.get_api_key(pid)
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            try:
                r = await client.get(f"{base_url}/models", headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    models = data.get("data", data.get("result", []))
                    if isinstance(models, list):
                        for m in models:
                            mid = m.get("id") if isinstance(m, dict) else str(m)
                            if mid:
                                # Prefix with provider for distinct routing
                                results.append({
                                    "id": f"{pid}/{mid}",
                                    "object": "model",
                                    "owned_by": pid,
                                    "root": mid,
                                })
            except Exception:
                continue

    # Fallback dummy models if none returned
    if not results:
        for pid in cfg.data.get("providers", {}):
            def_m = cfg.get_default_model(pid)
            results.append({
                "id": f"{pid}/{def_m}",
                "object": "model",
                "owned_by": pid,
            })

    return {"object": "list", "data": results}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _require_proxy_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    raw_model = body.get("model", "")
    provider_id, target_model = resolve_provider_and_model(raw_model)

    is_stream = body.get("stream", False)
    base_url = cfg.get_base_url(provider_id).rstrip("/")
    url = f"{base_url}/chat/completions"

    key = cfg.get_api_key(provider_id)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "OmniCLI-Proxy/1.0",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if provider_id == "openrouter":
        headers["HTTP-Referer"] = "http://localhost:8000"
        headers["X-Title"] = "OmniCLI-Proxy"

    forward_body = dict(body)
    forward_body["model"] = target_model

    client = httpx.AsyncClient(timeout=120.0)

    if not is_stream:
        try:
            resp = await client.post(url, headers=headers, json=forward_body)
            await client.aclose()
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception as e:
            await client.aclose()
            raise HTTPException(status_code=502, detail=f"Proxy error forwarding to {provider_id}: {e}")

    # Streaming mode. Open the upstream response BEFORE returning the stream so
    # an upstream failure (401/429/5xx) is forwarded with its real status code
    # instead of being masked as a local 200 stream of error text.
    req = client.build_request("POST", url, headers=headers, json=forward_body)
    try:
        resp = await client.send(req, stream=True)
    except Exception as e:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Proxy error forwarding to {provider_id}: {e}")
    if resp.status_code >= 400:
        detail = (await resp.aread()).decode("utf-8", "replace")
        await resp.aclose()
        await client.aclose()
        return JSONResponse(
            status_code=resp.status_code,
            content={"error": {"message": detail or f"upstream {resp.status_code}",
                               "type": "upstream_error", "provider": provider_id}},
        )

    async def sse_generator() -> AsyncGenerator[bytes, None]:
        try:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                yield (line + "\n\n").encode("utf-8")
        except Exception as e:
            err_msg = json.dumps({"error": {"message": str(e), "type": "proxy_error"}})
            yield f"data: {err_msg}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


def start_proxy(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    # Fail closed: binding off loopback exposes the operator's keys to the
    # network, so refuse unless an access token is configured to gate callers.
    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("OMNICLI_PROXY_TOKEN"):
        raise RuntimeError(
            f"refusing to bind proxy to {host!r} without OMNICLI_PROXY_TOKEN set — "
            "an unauthenticated listener would let any caller spend your provider keys"
        )
    print(f"Starting OmniCLI local proxy on http://{host}:{port}/v1 ...")
    uvicorn.run(app, host=host, port=port, log_level="warning")
