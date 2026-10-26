"""Unit tests for omnicli.quota (the ``omni credits`` quota layer).

Parsers are tested against the exact payload shapes observed live from each
provider, so a provider changing its schema fails here loudly.
"""

import pytest

from omnicli import quota


# --- account parsers (real observed payloads) -------------------------------


def test_parse_openrouter_usd_used():
    info = quota.parse_openrouter(
        {"data": {"total_credits": 0, "total_usage": 0.19726279}}
    )
    assert info.source == "account" and info.unit == "USD"
    assert info.used == pytest.approx(0.19726279)
    assert info.total is None  # 0 credits -> unbounded, not a real cap


def test_parse_bynara_token_quota():
    info = quota.parse_bynara(
        {
            "quota": {"remaining": 99868037, "limit": 100000000},
            "usage": {"requests_today": 5, "requests_month": 159},
        }
    )
    assert info.unit == "tokens" and info.total == 100000000
    assert info.remaining == 99868037
    assert info.used == 100000000 - 99868037
    assert "req/mo" in info.detail


def test_parse_apinex_subscription():
    info = quota.parse_apinex(
        {"token_limit": 3000000, "tokens_used": 0, "tokens_remaining": 3000000}
    )
    assert info.total == 3000000 and info.used == 0 and info.remaining == 3000000


def test_parse_xkiro_daily():
    info = quota.parse_xkiro(
        {
            "free_tokens": {
                "used_today": 264656,
                "limit_per_day": 1000000,
                "remaining": 735344,
            }
        }
    )
    assert info.unit == "tokens/day" and info.total == 1000000
    assert info.used == 264656 and info.remaining == 735344


def test_parse_llm7_balance_string_coerced():
    info = quota.parse_llm7({"email": "x@y.z", "balance_usd": "4.20"})
    assert info.unit == "USD" and info.remaining == pytest.approx(4.20)
    assert info.used is None and info.total is None


# --- rate-limit headers (groq) ----------------------------------------------


def test_parse_rate_headers_tokens():
    info = quota.parse_rate_headers(
        {
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-tokens": "7927",
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "999",
        }
    )
    assert info.source == "rate" and info.unit == "tokens/min"
    assert info.total == 8000 and info.remaining == 7927 and info.used == 73
    assert "req/day" in info.detail


def test_parse_bynara_null_requests_month_not_zero():
    info = quota.parse_bynara(
        {
            "quota": {"remaining": 100, "limit": 200},
            "usage": {"requests_month": None},
        }
    )
    assert info.detail == ""


def test_parse_rate_headers_missing_raises():
    with pytest.raises(quota.QuotaError):
        quota.parse_rate_headers({"content-type": "application/json"})


# --- rendering (honest units + n/a) -----------------------------------------


def test_render_used_total_tokens():
    info = quota.QuotaInfo(
        unit="tokens", source="account", used=131963, total=100000000
    )
    assert quota.render_quota(info) == "131,963 / 100,000,000 tokens"


def test_render_zero_total_preserved():
    info = quota.QuotaInfo(unit="tokens", source="account", used=0, total=0)
    assert quota.render_quota(info) == "0 / 0 tokens"


def test_render_remaining_only_usd():
    info = quota.QuotaInfo(unit="USD", source="account", remaining=4.2)
    assert quota.render_quota(info) == "$4.2 USD left"


def test_render_doc_cap_no_usage():
    info = quota.QuotaInfo(unit="rpm", source="doc", total=40)
    assert quota.render_quota(info) == "— / 40 rpm"


def test_render_none_is_na():
    assert quota.render_quota(None) == "n/a"


# --- fetch_quota orchestration (no real network) ----------------------------


class _FakeCfg:
    def __init__(self, keys, providers):
        self._keys = keys
        self._providers = providers

    def get_api_key(self, pid):
        return self._keys.get(pid, "")

    def get_base_url(self, pid):
        return self._providers.get(pid, {}).get("base_url", "")

    def get_default_model(self, pid):
        return self._providers.get(pid, {}).get("default_model", "")

    def get_provider_info(self, pid):
        return self._providers.get(pid, {})


def test_fetch_quota_missing_key_raises():
    cfg = _FakeCfg({}, {"openrouter": {"base_url": "https://x/v1"}})
    with pytest.raises(quota.QuotaError):
        quota.fetch_quota("openrouter", cfg)


def test_fetch_quota_unknown_provider_returns_none():
    cfg = _FakeCfg({"dahl": "k"}, {"dahl": {"base_url": "https://x/v1"}})
    assert quota.fetch_quota("dahl", cfg) is None


def test_fetch_quota_doc_cap():
    cfg = _FakeCfg(
        {"nvidia": "k"},
        {
            "nvidia": {
                "base_url": "https://x/v1",
                "quota_doc": {"value": 40, "unit": "rpm"},
            }
        },
    )
    info = quota.fetch_quota("nvidia", cfg)
    assert (
        info is not None
        and info.source == "doc"
        and info.total == 40
        and info.unit == "rpm"
    )


def test_fetch_quota_account_uses_parser(monkeypatch):
    cfg = _FakeCfg(
        {"apinex": "k"}, {"apinex": {"base_url": "https://api.apinex.bond/v1"}}
    )
    captured = {}

    def fake_get(url, key):
        captured["url"] = url
        return {"token_limit": 3000000, "tokens_used": 10, "tokens_remaining": 2999990}

    monkeypatch.setattr(quota, "_http_get_json", fake_get)
    info = quota.fetch_quota("apinex", cfg)
    assert captured["url"] == "https://api.apinex.bond/v1/subscription"
    assert info.total == 3000000 and info.used == 10


def test_fetch_quota_malformed_schema_raises_quota_error(monkeypatch):
    cfg = _FakeCfg(
        {"apinex": "k"}, {"apinex": {"base_url": "https://api.apinex.bond/v1"}}
    )
    monkeypatch.setattr(quota, "_http_get_json", lambda u, k: {"wrong": 1})
    with pytest.raises(quota.QuotaError, match="unexpected quota schema"):
        quota.fetch_quota("apinex", cfg)


def test_http_get_json_never_forwards_bearer_on_redirect():
    # S5: quota lookups carry the provider Bearer key; a 3xx must become
    # QuotaError, never a forwarded key at the redirect target.
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received = []

    class Sink(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    sink = HTTPServer(("127.0.0.1", 0), Sink)
    threading.Thread(target=sink.serve_forever, daemon=True).start()

    class Redirector(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header(
                "Location", f"http://127.0.0.1:{sink.server_port}/sink"
            )
            self.end_headers()

        def log_message(self, *args):
            pass

    src = HTTPServer(("127.0.0.1", 0), Redirector)
    threading.Thread(target=src.serve_forever, daemon=True).start()
    try:
        with pytest.raises(quota.QuotaError):
            quota._http_get_json(
                f"http://127.0.0.1:{src.server_port}/credits", "dummy-key"
            )
        assert received == [], "Bearer key reached the redirect target"
    finally:
        src.shutdown()
        sink.shutdown()


def test_require_http_url_blocks_remote_http_allows_loopback():
    quota._require_http_url("https://openrouter.ai/api/v1/credits")
    quota._require_http_url("http://127.0.0.1:3271/v1")
    quota._require_http_url("http://localhost:8000/v1")
    with pytest.raises(quota.QuotaError):
        quota._require_http_url("http://api.example.com/v1/credits")

