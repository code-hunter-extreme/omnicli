"""Provider quota/usage fetchers for the ``omni credits`` command.

Each provider that exposes a usable quota signal has an entry here. The signal
comes in three honest flavours, and every row carries its own native unit and a
``source`` marker so nothing is misread:

* ``account`` -- a real balance/usage endpoint (openrouter, bynara, apinex,
  xkiro, llm7). Numbers are live.
* ``rate`` -- no account endpoint, but the provider returns ``x-ratelimit-*``
  response headers (groq). Numbers describe the current rate window.
* ``doc`` -- no live signal at all, but the provider documents a fixed cap the
  operator recorded in config under ``quota_doc`` (e.g. nvidia 40 rpm). The
  cap is shown with a ``(doc)`` marker and ``used`` stays unknown.

Providers with none of the above have no fetcher and the command shows an
honest ``n/a`` -- never a fabricated zero. Adding a provider later is one entry
in :data:`ACCOUNT_SPECS` / :data:`RATE_HEADER_PROVIDERS` or a ``quota_doc`` in
its config block.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_TIMEOUT = 12.0


class QuotaError(RuntimeError):
    """A quota lookup was attempted for a provider but failed loudly.

    Raised instead of returning a misleading zero, so the caller can show the
    provider's own error in its row without aborting the whole table.
    """


@dataclass(frozen=True)
class QuotaInfo:
    """Normalised quota reading for one provider.

    ``used``/``total``/``remaining`` are provider-native amounts in ``unit``;
    any of them may be ``None`` when the provider does not report that facet
    (e.g. a pure remaining-balance, or a documented cap with no usage meter).
    """

    unit: str
    source: str
    used: Optional[float] = None
    total: Optional[float] = None
    remaining: Optional[float] = None
    detail: str = ""


# --- pure parsers (one per account provider; unit-tested without network) ---


def _num(value: Any) -> Optional[float]:
    """Coerce an API field to float, or ``None`` when it is not a number."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().lstrip("$"))
        except ValueError:
            return None
    return None


def parse_openrouter(payload: Dict[str, Any]) -> QuotaInfo:
    """Parse OpenRouter ``GET /credits`` -> USD used against purchased credits."""
    data = payload["data"]
    used = _num(data["total_usage"])
    total = _num(data["total_credits"])
    remaining = None
    if used is not None and total:
        remaining = total - used
    return QuotaInfo(
        unit="USD",
        source="account",
        used=used,
        total=total or None,
        remaining=remaining,
    )


def parse_bynara(payload: Dict[str, Any]) -> QuotaInfo:
    """Parse Bynara ``GET /me`` -> token quota (limit/remaining)."""
    quota = payload["quota"]
    total = _num(quota["limit"])
    remaining = _num(quota["remaining"])
    used = (
        (total - remaining) if (total is not None and remaining is not None) else None
    )
    usage = payload.get("usage") or {}
    detail = ""
    req_mo = _num(usage.get("requests_month"))
    if req_mo is not None:
        detail = f"{int(req_mo):,} req/mo"
    return QuotaInfo(
        unit="tokens",
        source="account",
        used=used,
        total=total,
        remaining=remaining,
        detail=detail,
    )


def parse_apinex(payload: Dict[str, Any]) -> QuotaInfo:
    """Parse Apinex ``GET /subscription`` -> token allowance (used/limit)."""
    total = _num(payload["token_limit"])
    used = _num(payload["tokens_used"])
    remaining = _num(payload.get("tokens_remaining"))
    return QuotaInfo(
        unit="tokens",
        source="account",
        used=used,
        total=total,
        remaining=remaining,
    )


def parse_xkiro(payload: Dict[str, Any]) -> QuotaInfo:
    """Parse xKiro ``GET /usage`` -> daily free-token budget."""
    ft = payload["free_tokens"]
    total = _num(ft["limit_per_day"])
    used = _num(ft["used_today"])
    remaining = _num(ft.get("remaining"))
    return QuotaInfo(
        unit="tokens/day",
        source="account",
        used=used,
        total=total,
        remaining=remaining,
    )


def parse_llm7(payload: Dict[str, Any]) -> QuotaInfo:
    """Parse LLM7 ``GET /balance`` -> remaining USD balance."""
    remaining = _num(payload["balance_usd"])
    return QuotaInfo(unit="USD", source="account", remaining=remaining)


def parse_rate_headers(headers: Dict[str, str]) -> QuotaInfo:
    """Parse ``x-ratelimit-*`` token headers (e.g. groq) -> per-window usage."""
    low = {k.lower(): v for k, v in headers.items()}
    total = _num(low.get("x-ratelimit-limit-tokens"))
    remaining = _num(low.get("x-ratelimit-remaining-tokens"))
    if total is None and remaining is None:
        raise QuotaError("no x-ratelimit token headers returned")
    used = (
        (total - remaining) if (total is not None and remaining is not None) else None
    )
    detail = ""
    rr = _num(low.get("x-ratelimit-remaining-requests"))
    lr = _num(low.get("x-ratelimit-limit-requests"))
    if rr is not None and lr is not None:
        detail = f"{int(rr):,}/{int(lr):,} req/day"
    return QuotaInfo(
        unit="tokens/min",
        source="rate",
        used=used,
        total=total,
        remaining=remaining,
        detail=detail,
    )


# provider_id -> (endpoint path appended to base_url, parser)
ACCOUNT_SPECS: Dict[str, tuple[str, Callable[[Dict[str, Any]], QuotaInfo]]] = {
    "openrouter": ("/credits", parse_openrouter),
    "bynara": ("/me", parse_bynara),
    "apinex": ("/subscription", parse_apinex),
    "xkiro": ("/usage", parse_xkiro),
    "llm7": ("/balance", parse_llm7),
}

# providers with no account endpoint but usable x-ratelimit-* response headers
RATE_HEADER_PROVIDERS = {"groq"}


# --- network glue (thin; parsers above hold all the logic) ------------------


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: urllib would otherwise re-send the provider
    Bearer key to the redirect target, any origin or scheme (S5)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _no_redirect_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirectHandler)


def _require_http_url(url: str) -> None:
    """Require https:// for remote quota URLs; allow http:// only on loopback."""
    parsed = urllib.parse.urlsplit(url or "")
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if scheme == "https" and host:
        return
    if scheme == "http" and host in ("127.0.0.1", "localhost", "::1"):
        return
    raise QuotaError(f"refusing non-https/non-loopback quota URL: {url[:40]}")


def _http_get_json(url: str, api_key: str) -> Dict[str, Any]:
    """GET a provider account endpoint and return its parsed JSON body."""
    _require_http_url(url)
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": _UA,
            "Accept": "application/json",
        },
    )
    try:
        with _no_redirect_opener().open(
            req, timeout=_TIMEOUT
        ) as resp:  # nosec B310 - scheme checked above
            return json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        raise QuotaError(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, ValueError, OSError) as exc:
        raise QuotaError(str(exc)[:80]) from exc


def _fetch_rate_headers(base_url: str, api_key: str, model: str) -> QuotaInfo:
    """Make a 1-token call and read ``x-ratelimit-*`` headers into a QuotaInfo."""
    url = base_url.rstrip("/") + "/chat/completions"
    _require_http_url(url)
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": _UA,
            "Content-Type": "application/json",
        },
    )
    try:
        with _no_redirect_opener().open(
            req, timeout=_TIMEOUT
        ) as resp:  # nosec B310 - scheme checked above
            return parse_rate_headers(dict(resp.headers.items()))
    except urllib.error.HTTPError as exc:
        return parse_rate_headers(dict(exc.headers.items()))
    except (urllib.error.URLError, OSError) as exc:
        raise QuotaError(str(exc)[:80]) from exc


def fetch_quota(provider_id: str, cfg: Any) -> Optional[QuotaInfo]:
    """Resolve one provider's quota, or ``None`` when it exposes no signal.

    Raises :class:`QuotaError` when a signal is expected but the lookup fails,
    so the caller renders the error rather than a fabricated number.
    """
    provider_id = provider_id.lower()
    api_key = cfg.get_api_key(provider_id)
    if not api_key:
        raise QuotaError("no API key configured")

    if provider_id in ACCOUNT_SPECS:
        path, parser = ACCOUNT_SPECS[provider_id]
        base = cfg.get_base_url(provider_id).rstrip("/")
        payload = _http_get_json(base + path, api_key)
        try:
            return parser(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise QuotaError(f"unexpected quota schema: {exc}") from exc

    if provider_id in RATE_HEADER_PROVIDERS:
        base = cfg.get_base_url(provider_id)
        return _fetch_rate_headers(base, api_key, cfg.get_default_model(provider_id))

    quota_doc = cfg.get_provider_info(provider_id).get("quota_doc")
    if isinstance(quota_doc, dict) and "value" in quota_doc and "unit" in quota_doc:
        return QuotaInfo(
            unit=str(quota_doc["unit"]),
            source="doc",
            total=_num(quota_doc["value"]),
        )

    return None


# --- rendering (shared by the CLI; kept here so it is unit-testable) --------


def _fmt_amount(value: float, unit: str) -> str:
    """Format an amount for display: ``$`` money for USD, grouped ints else."""
    if unit == "USD":
        return f"${value:,.4f}".rstrip("0").rstrip(".") if value else "$0"
    return f"{int(value):,}" if float(value).is_integer() else f"{value:,.2f}"


def render_quota(info: Optional[QuotaInfo]) -> str:
    """Build the human cell for a quota reading (honest ``n/a`` when None)."""
    if info is None:
        return "n/a"
    u = info.unit
    if info.used is not None and info.total is not None:
        return f"{_fmt_amount(info.used, u)} / {_fmt_amount(info.total, u)} {u}"
    if info.used is not None:
        return f"{_fmt_amount(info.used, u)} {u} used"
    if info.remaining is not None and info.total is not None:
        used = info.total - info.remaining
        return f"{_fmt_amount(used, u)} / {_fmt_amount(info.total, u)} {u}"
    if info.remaining is not None:
        return f"{_fmt_amount(info.remaining, u)} {u} left"
    if info.total is not None:
        return f"— / {_fmt_amount(info.total, u)} {u}"
    return "n/a"
