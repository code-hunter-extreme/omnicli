"""
OmniCLI Native Deterministic Verification & Cross-Model Consensus Engine.

Provides:
  1. ScopeGate: Fail-closed boundary check for hosts, IP ranges, and excluded paths.
  2. HTTPVerifier: Pure Python HTTP execution and strict predicate verification.
  3. CodeVerifier: Deterministic static AST and pattern seam verification.
  4. AssertionCardVerifier: 5-Gate test runner for structured binary assertions.
  5. ConsensusVerifier: Blind multi-model cross-checking (same provider / cross-provider).
"""

from __future__ import annotations

import contextlib
import ipaddress
import re
import socket
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx


def _is_under(path: Path, root: Path) -> bool:
    """True when path is root or a descendant of root."""
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


GLOBAL_BLOCKED_HOSTS = {
    "169.254.169.254",          # AWS / Azure / GCP metadata IP
    "metadata.google.internal",
    "metadata.azure.com",
    "google.com", "googleapis.com", "gstatic.com", "google-analytics.com",
    "googletagmanager.com", "doubleclick.net", "facebook.com",
    "twitter.com", "x.com", "github.com", "cloudflare.com"
}


class ScopeGate:
    """Deterministic boundary gate for URL and code targets."""

    def __init__(
        self,
        allowed_hosts: Optional[List[str]] = None,
        blocked_hosts: Optional[List[str]] = None,
        allow_localhost: bool = False,
        allow_private: bool = False,
    ) -> None:
        self.allowed_hosts = [h.lower().strip() for h in (allowed_hosts or []) if h]
        self.blocked_hosts = set(GLOBAL_BLOCKED_HOSTS)
        if blocked_hosts:
            self.blocked_hosts.update(h.lower().strip() for h in blocked_hosts if h)
        self.allow_localhost = allow_localhost
        # Permit RFC1918 / unique-local targets (e.g. testing a LAN instance).
        # Link-local (169.254/16, fe80::/10 — cloud metadata) stays blocked
        # regardless, as do reserved/multicast/unspecified ranges.
        self.allow_private = allow_private

    @staticmethod
    def extract_host(target: str) -> str:
        """Extract clean hostname from URL or target string, stripping userinfo and ports."""
        target = target.strip()
        if "://" in target:
            try:
                parsed = urllib.parse.urlparse(target)
                host = parsed.netloc or parsed.path
            except ValueError:
                host = target.split("://", 1)[1].split("/")[0]
        else:
            host = target.split("/")[0]

        if "@" in host:
            host = host.split("@")[-1]

        if ":" in host and not host.startswith("["):
            host = host.split(":")[0]

        return host.strip().lower()

    def is_in_scope(self, target: str) -> Tuple[bool, str]:
        """Check if target host is allowed and not blocked (verdict + reason)."""
        ok, reason, _ = self.screen(target)
        return ok, reason

    def screen(self, target: str) -> Tuple[bool, str, Optional[str]]:
        """Full scope decision plus the validated IP to pin the connection to.

        Returns (ok, reason, pinned_ip). pinned_ip is the resolved, scope-checked
        address the caller should connect to directly, so a second DNS lookup at
        connect time cannot rebind to an internal host.
        """
        host = self.extract_host(target)
        if not host:
            return False, "Empty or invalid target host", None

        # Parser-differential guard: an authority with backslash, whitespace,
        # control chars, or multiple '@' can be parsed one way by extract_host and
        # another by the HTTP client, letting the connection reach a different host
        # than we validate. Reject it outright (fail closed).
        authority = target.split("://", 1)[-1].split("/", 1)[0] if "://" in target else target.split("/", 1)[0]
        if "\\" in authority or authority.count("@") > 1 or any(
            c.isspace() or ord(c) < 0x20 for c in authority
        ):
            return False, "out_of_scope: ambiguous/unsafe URL authority", None

        if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
            if self.allow_localhost:
                pin = "::1" if host == "::1" else "127.0.0.1"
                return True, "in_scope: localhost permitted", pin
            return False, "out_of_scope: localhost access blocked", None

        if host in self.blocked_hosts or any(host.endswith("." + b) for b in self.blocked_hosts):
            return False, f"EXPLICIT_BLOCKED: host '{host}' is in blacklisted/metadata domains", None

        # Resolve the host and reject if ANY address is non-public. This catches
        # IP literals in odd encodings (decimal/hex), 127.0.0.0/8 beyond
        # 127.0.0.1, RFC1918, IPv4-mapped IPv6, and hostnames that resolve to
        # internal addresses. Runs before the whitelist so an explicit allow
        # still can't reach an internal IP without allow_private/allow_localhost.
        ok_ip, ip_reason, pin_ip = self._check_resolved_ips(host)
        if not ok_ip:
            return False, ip_reason, None

        if not self.allowed_hosts:
            # If no explicit whitelist is defined, allow any valid public host not in blacklist
            return True, "in_scope: general public target", pin_ip

        for pattern in self.allowed_hosts:
            if pattern.startswith("*."):
                # Anchor both ends: *.victim.com must match api.victim.com
                # but NOT api.victim.com.evil.net (suffix match is not enough).
                suffix = re.escape(pattern[2:])
                anchored = re.compile(rf"^(?:[a-zA-Z0-9_\-]+\.)*{suffix}$", re.IGNORECASE)
                if anchored.match(host):
                    return True, f"in_scope: matches wildcard rule '{pattern}'", pin_ip
            elif host == pattern:
                return True, f"in_scope: matches explicit rule '{pattern}'", pin_ip

        return False, f"out_of_scope: host '{host}' does not match permitted scopes", None

    @staticmethod
    def _resolve_host(host: str) -> List[Any]:
        """Resolve a host to a deduped list of ipaddress objects.

        An IP literal is parsed directly; a hostname is resolved via
        getaddrinfo (A + AAAA). Returns [] when resolution fails — the caller
        treats that as out-of-scope (fail closed).
        """
        h = host.strip().strip("[]")
        try:
            return [ipaddress.ip_address(h)]
        except ValueError:
            pass
        try:
            infos = socket.getaddrinfo(h, None, proto=socket.IPPROTO_TCP)
        except (socket.gaierror, UnicodeError, OSError):
            return []
        seen: set = set()
        out: List[Any] = []
        for info in infos:
            addr = info[4][0]
            try:
                ip = ipaddress.ip_address(addr)
            except ValueError:
                continue
            if ip not in seen:
                seen.add(ip)
                out.append(ip)
        return out

    @staticmethod
    def _ip_category(ip: Any) -> str:
        """Classify an ip into loopback/linklocal/private/.../global.

        Order matters: loopback/link-local/multicast/reserved are checked before
        the broad is_private so 'private' means the RFC1918 / unique-local
        remainder. IPv4-mapped IPv6 is unwrapped first.
        """
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if ip.is_unspecified:
            return "unspecified"
        if ip.is_loopback:
            return "loopback"
        if ip.is_link_local:
            return "linklocal"
        if ip.is_multicast:
            return "multicast"
        if ip.is_reserved:
            return "reserved"
        if ip.is_private:
            return "private"
        return "global"

    def _check_resolved_ips(self, host: str) -> Tuple[bool, str, Optional[str]]:
        """Fail-closed SSRF check: every resolved IP must be public (or opted in).

        loopback requires allow_localhost; RFC1918/unique-local requires
        allow_private; link-local (incl. cloud metadata), reserved, multicast
        and unspecified are always rejected. Returns the first validated address
        as the third element so the caller can pin the connection to it.
        """
        ips = self._resolve_host(host)
        if not ips:
            return False, f"out_of_scope: host '{host}' could not be resolved", None
        for ip in ips:
            cat = self._ip_category(ip)
            if cat == "global":
                continue
            if cat == "loopback" and self.allow_localhost:
                continue
            if cat == "private" and self.allow_private:
                continue
            return False, (
                f"SSRF_BLOCKED: host '{host}' resolves to {ip} ({cat}); "
                "non-public address not permitted"
            ), None
        return True, f"in_scope: '{host}' resolves to public address(es)", str(ips[0])


def parse_expected_predicates(expected_str: str) -> Dict[str, Any]:
    """Parse an Expected line into structured machine predicates.
    
    Example: 'status 200, body contains "records", header x-frame-options absent'
    """
    out: Dict[str, Any] = {
        "status": None,
        "body_contains": [],
        "body_not_contains": [],
        "body_contains_any": [],
        "headers_absent": [],
        "headers_contains": {},
    }
    if not expected_str:
        return out

    s = expected_str.strip()

    m_status = re.search(r"status\s+(\d{3})", s, re.IGNORECASE)
    if m_status:
        out["status"] = int(m_status.group(1))

    m_any = re.search(r"body\s+contains\s+any\s+of\s+(.+?)(?:;|$)", s, re.IGNORECASE)
    if m_any:
        out["body_contains_any"] = [x.strip("\"'") for x in re.findall(r'["\']([^"\']+)["\']', m_any.group(1))]
        s = (s[:m_any.start()] + s[m_any.end():]).strip()

    for m in re.finditer(r'body\s+contains\s+["\']([^"\']+)["\']', s, re.IGNORECASE):
        out["body_contains"].append(m.group(1))

    for m in re.finditer(r'body\s+not\s+contains\s+["\']([^"\']+)["\']', s, re.IGNORECASE):
        out["body_not_contains"].append(m.group(1))

    # Fallback for unquoted single token: 'body contains admin'
    if not out["body_contains"]:
        m_unq = re.search(r'body\s+contains\s+([^\s,;]+)', s, re.IGNORECASE)
        if m_unq and m_unq.group(1).lower() not in ('any', 'not'):
            out["body_contains"].append(m_unq.group(1).strip("\"'"))

    for m in re.finditer(r'header\s+([a-z0-9-]+)\s+absent', s, re.IGNORECASE):
        out["headers_absent"].append(m.group(1).lower())

    for m in re.finditer(r'header\s+([a-z0-9-]+)\s*(?:contains|:)\s*["\']?([^,"\']+)', s, re.IGNORECASE):
        name, value = m.group(1).lower(), m.group(2).strip()
        if value and value.lower() != "absent":
            out["headers_contains"][name] = value

    return out


@contextlib.contextmanager
def _pin_getaddrinfo(hostname: str, ip: str):
    """Force `hostname` to resolve to `ip` for the duration of the block.

    The ScopeGate already resolved and validated the host; pinning the connection
    to that exact IP means httpx cannot trigger a second DNS lookup that rebinds
    to an internal address. Single-threaded use only (patches the global
    resolver), which matches the agent's sequential request loop.
    """
    real_getaddrinfo = socket.getaddrinfo

    def patched(host, port, family=0, socktype=0, proto=0, flags=0):
        if host == hostname:
            if ":" in ip:
                return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, port or 0, 0, 0))]
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]
        return real_getaddrinfo(host, port, family, socktype, proto, flags)

    socket.getaddrinfo = patched
    try:
        yield
    finally:
        socket.getaddrinfo = real_getaddrinfo


class HTTPVerifier:
    """Deterministic HTTP wire probe engine."""

    @staticmethod
    def verify(
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        expected_str: str = "",
        scope_gate: Optional[ScopeGate] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Perform request and verify response against expected predicates."""
        pin_ip: Optional[str] = None
        if scope_gate:
            ok, reason, pin_ip = scope_gate.screen(url)
            if not ok:
                return {
                    "verdict": "REJECTED",
                    "reason": f"Scope Gate blocked request: {reason}",
                    "url": url,
                    "observed": None,
                }

        predicates = parse_expected_predicates(expected_str)
        if not (
            predicates["status"] is not None
            or predicates["body_contains"]
            or predicates["body_not_contains"]
            or predicates["body_contains_any"]
            or predicates["headers_absent"]
            or predicates["headers_contains"]
        ):
            return {
                "verdict": "REJECTED",
                "reason": "No valid expected predicates were provided.",
                "url": url,
                "observed": None,
                "predicates": predicates,
            }

        req_headers = headers or {}

        pin_host = ScopeGate.extract_host(url) if pin_ip else None
        pin_ctx = _pin_getaddrinfo(pin_host, pin_ip) if (pin_host and pin_ip) else contextlib.nullcontext()

        try:
            with pin_ctx:
                with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                    res = client.request(method=method.upper(), url=url, headers=req_headers, content=body)
                    obs_status = res.status_code
                    obs_body = res.text
                    obs_headers = {k.lower(): v for k, v in res.headers.items()}
        except Exception as e:
            return {
                "verdict": "REJECTED",
                "reason": f"HTTP request failed: {e}",
                "url": url,
                "observed": None,
            }

        observed = {"status": obs_status, "headers": obs_headers, "body_preview": obs_body[:300]}
        failures = []

        if predicates["status"] is not None and obs_status != predicates["status"]:
            failures.append(f"expected status {predicates['status']}, got {obs_status}")

        for pat in predicates["body_contains"]:
            if pat not in obs_body:
                failures.append(f'body missing "{pat}"')

        if predicates["body_contains_any"]:
            if not any(pat in obs_body for pat in predicates["body_contains_any"]):
                failures.append(f"body contains none of {predicates['body_contains_any']}")

        for pat in predicates["body_not_contains"]:
            if pat in obs_body:
                failures.append(f'body unexpectedly contains "{pat}"')

        for h in predicates["headers_absent"]:
            if h in obs_headers:
                failures.append(f'header "{h}" is present')

        for h, val in predicates["headers_contains"].items():
            obs_val = obs_headers.get(h, "")
            if val.lower() not in obs_val.lower():
                failures.append(f'header "{h}" does not contain "{val}" (got "{obs_val}")')

        if failures:
            return {
                "verdict": "REJECTED",
                "reason": f"Wire behavior did not match prediction: {'; '.join(failures)}",
                "url": url,
                "observed": observed,
                "predicates": predicates,
            }

        return {
            "verdict": "CONFIRMED",
            "reason": f"Wire matches prediction (status {obs_status}) — binary conflict proven",
            "url": url,
            "observed": observed,
            "predicates": predicates,
        }


# Directories CodeVerifier is allowed to read.
# Extend by passing allowed_dirs to verify_file, or update this list.
_DEFAULT_ALLOWED_DIRS: List[Path] = [
    Path.home(),          # anything under ~
    Path("/tmp"),
    Path("/var/tmp"),
]


class CodeVerifier:
    """Deterministic source code and AST seam verifier."""

    @staticmethod
    def _extract_python_block(content: str, name: str, kind: str = "def") -> Optional[str]:
        """Return the source of a Python `kind` (def/class) named `name`, or None."""
        header = re.compile(r"^(\s*)" + re.escape(kind) + r"\s+" + re.escape(name) + r"\s*[\(:]")
        lines = content.splitlines()
        for i, line in enumerate(lines):
            m = header.match(line)
            if not m:
                continue
            indent = len(m.group(1))
            block = [line]
            for nxt in lines[i + 1:]:
                if nxt.strip() == "":
                    block.append(nxt)
                    continue
                if (len(nxt) - len(nxt.lstrip())) <= indent:
                    break
                block.append(nxt)
            return "\n".join(block)
        return None

    @staticmethod
    def verify_file(
        file_path: str,
        must_contain: Optional[List[str]] = None,
        must_not_contain: Optional[List[str]] = None,
        target_function: Optional[str] = None,
        check_type: Optional[str] = None,
        allowed_dirs: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        p = Path(file_path).expanduser().resolve()

        # Path guard: reject anything outside allowed directories.
        # This prevents a hallucinating agent from reading /etc/passwd or ~/.ssh/id_rsa.
        roots = allowed_dirs if allowed_dirs is not None else _DEFAULT_ALLOWED_DIRS
        if not any(_is_under(p, root) for root in roots):
            return {
                "verdict": "REJECTED",
                "reason": (
                    f"Path '{p}' is outside the allowed workspace directories. "
                    f"Allowed roots: {[str(r) for r in roots]}"
                ),
                "file": str(p),
            }

        if not p.exists():
            return {"verdict": "REJECTED", "reason": f"File does not exist: {file_path}", "file": str(p)}
        if not p.is_file():
            return {"verdict": "REJECTED", "reason": f"Target is not a file: {file_path}", "file": str(p)}

        supported_check_types = {"wp_auth_check"}
        if check_type and check_type not in supported_check_types:
            return {
                "verdict": "REJECTED",
                "reason": f"Unsupported verification check type: {check_type}",
                "file": str(p),
            }

        # A function name scopes inspection; it does not assert a security property.
        if not (must_contain or must_not_contain or check_type):
            return {
                "verdict": "REJECTED",
                "reason": (
                    "No verification constraints provided "
                    "(must_contain / must_not_contain / supported check_type); "
                    "target_function only scopes inspection."
                ),
                "file": str(p),
            }

        patterns = (must_contain or []) + (must_not_contain or [])
        if any(not isinstance(pattern, str) or not pattern.strip() for pattern in patterns):
            return {
                "verdict": "REJECTED",
                "reason": "Verification patterns must be non-empty strings.",
                "file": str(p),
            }

        content = p.read_text(encoding="utf-8", errors="replace")
        failures = []

        # If scoped to a function, extract function block
        inspect_text = content
        if target_function:
            # Accept "Class.method" by scoping on the final name segment.
            name = target_function.split(".")[-1]
            fn_regex = rf"(function\s+{re.escape(name)}\s*\([^)]*\)\s*\{{(?:[^{{}}]*|\{{(?:[^{{}}]*|\{{[^{{}}]*\}})*\}})*\}})"
            m = re.search(fn_regex, content, re.DOTALL | re.IGNORECASE)
            if m:
                inspect_text = m.group(1)          # JavaScript/TS function
            else:
                block = (CodeVerifier._extract_python_block(content, name, "def")
                         or CodeVerifier._extract_python_block(content, name, "class"))
                if block is not None:
                    inspect_text = block            # Python def/class
                else:
                    # Scope requested but not found: FAIL CLOSED. Verifying a
                    # scoped claim against the whole file would be fail-open —
                    # the pattern could match unrelated code elsewhere.
                    failures.append(
                        f"Scope '{target_function}' not found; cannot verify a scoped claim"
                    )
                    inspect_text = ""

        # Specific framework checks
        if check_type == "wp_auth_check":
            # Must contain capability or nonce check
            wp_guards = ["current_user_can", "wp_verify_nonce", "check_ajax_referer", "check_admin_referer"]
            if not any(g in inspect_text for g in wp_guards):
                failures.append("No WordPress authorization guard (current_user_can, nonce) detected in target scope")

        if must_contain:
            for item in must_contain:
                if item not in inspect_text:
                    failures.append(f"Required pattern '{item}' not found in code")

        if must_not_contain:
            for item in must_not_contain:
                if item in inspect_text:
                    failures.append(f"Forbidden pattern '{item}' unexpectedly present in code")

        if failures:
            return {
                "verdict": "REJECTED",
                "reason": f"Code verification failed: {'; '.join(failures)}",
                "file": str(p),
            }

        return {
            "verdict": "CONFIRMED",
            "reason": "Code pattern constraints satisfied deterministically",
            "file": str(p),
        }


class CrossModelConsensusVerifier:
    """Blind multi-model, multi-family consensus verifier.

    Supports:
      - Panel built by `omni scan` (from config) — one model per family per provider
      - Explicit panel passed at call time (provider, model) tuples
      - Cross-provider: mix NVIDIA + Cloudflare + OpenRouter in one panel

    Vote math:
      - Denominator = total_queried (not just models that answered — errors count)
      - CONSENSUS_CONFIRMED: majority TRUE_POSITIVE AND tp > fp
      - CONSENSUS_REJECTED:  majority FALSE_POSITIVE AND fp > tp
      - SPLIT_DECISION:      neither majority
      - INCONCLUSIVE:        no valid votes at all (all errors / all INCONCLUSIVE)
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    @staticmethod
    def _default_panel(cfg: Any = None) -> List[Tuple[str, Optional[str]]]:
        """Build a cross-provider panel from config scan results.

        Picks the first entry from each provider's verification_panel (already
        family-diverse per provider). If config has no scan results yet, falls
        back to a minimal 2-provider hardcoded panel so verify_consensus works
        before the first `omni scan` run.
        """
        try:
            from omnicli.config import ConfigManager
            c = cfg or ConfigManager()
            panel: List[Tuple[str, Optional[str]]] = []
            seen_families: set = set()
            for provider_id in c.data.get("providers", {}):
                vp = c.get_verification_panel(provider_id)
                for entry in vp:
                    family = entry.get("family", "")
                    model = entry.get("model", "")
                    if family and model and family not in seen_families:
                        panel.append((provider_id, model))
                        seen_families.add(family)
                    if len(panel) >= 4:
                        break
                if len(panel) >= 4:
                    break
            if panel:
                return panel
        except Exception:
            pass
        # Fallback: minimal 2-provider panel (works before first scan)
        return [
            ("nvidia",     "meta/llama-3.2-11b-vision-instruct"),
            ("cloudflare", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
        ]

    def verify_finding(
        self,
        claim_description: str,
        code_or_evidence: str,
        models: Optional[List[Tuple[str, Optional[str]]]] = None,
    ) -> Dict[str, Any]:
        """Query models independently with a blind verification rubric.

        models: explicit [(provider, model_id), ...] list, or None to use
                the panel from config (built by `omni scan`).

        Returns consensus vote, per-model critiques, and honest ratios.
        """
        model_panel = models or self._default_panel()
        total_queried = len(model_panel)

        system_prompt = (
            "You are a rigorous, skeptical application security verifier. "
            "You will be given a security claim and code/evidence. "
            "Evaluate whether the claim is a TRUE POSITIVE vulnerability or "
            "a FALSE POSITIVE / NON-ISSUE. "
            "Be strict: reject speculative or non-state-mutating issues. "
            "Your output MUST end with a final line: "
            "'VERDICT: TRUE_POSITIVE' or 'VERDICT: FALSE_POSITIVE'."
        )
        user_prompt = (
            f"### CLAIM:\n{claim_description}\n\n"
            f"### CODE / EVIDENCE:\n{code_or_evidence}\n\n"
            "Analyze the vulnerability mechanics and provide your final verdict."
        )

        votes: Dict[str, str] = {}
        critiques: Dict[str, str] = {}

        for provider, model_id in model_panel:
            label = f"{provider}:{model_id or 'default'}"
            try:
                res = self.client.chat_complete(
                    provider_id=provider,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    model=model_id,
                    max_tokens=600,
                )
                choices = res.get("choices", [])
                content = choices[0]["message"]["content"] if choices else ""
                critiques[label] = content
                upper = content.upper()
                if "VERDICT: TRUE_POSITIVE" in upper:
                    votes[label] = "TRUE_POSITIVE"
                elif "VERDICT: FALSE_POSITIVE" in upper:
                    votes[label] = "FALSE_POSITIVE"
                else:
                    votes[label] = "INCONCLUSIVE"
            except Exception as e:
                critiques[label] = f"Error: {e}"
                votes[label] = "ERROR"

        tp_count  = sum(1 for v in votes.values() if v == "TRUE_POSITIVE")
        fp_count  = sum(1 for v in votes.values() if v == "FALSE_POSITIVE")
        inc_count = sum(1 for v in votes.values() if v in ("INCONCLUSIVE", "ERROR"))

        # Ratio over total_queried — errors don't silently inflate the ratio.
        tp_ratio = round(tp_count / total_queried, 2)
        fp_ratio = round(fp_count / total_queried, 2)

        majority = (total_queried // 2) + 1
        if tp_count == 0 and fp_count == 0:
            consensus_verdict = "INCONCLUSIVE"
        elif tp_count >= majority and tp_count > fp_count:
            consensus_verdict = "CONSENSUS_CONFIRMED"
        elif fp_count >= majority and fp_count > tp_count:
            consensus_verdict = "CONSENSUS_REJECTED"
        else:
            consensus_verdict = "SPLIT_DECISION"

        return {
            "verdict":          consensus_verdict,
            "true_positives":   tp_count,
            "false_positives":  fp_count,
            "inconclusive":     inc_count,
            "total_queried":    total_queried,
            "tp_ratio":         tp_ratio,
            "fp_ratio":         fp_ratio,
            "votes":            votes,
            "critiques":        critiques,
        }
