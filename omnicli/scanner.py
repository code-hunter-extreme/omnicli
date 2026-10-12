"""
OmniCLI Model Scanner — live discovery and roster builder.

Hits every provider's /models endpoint, classifies each model into a family
using MODEL_FAMILY_KEYWORDS, then writes two things to config per provider:

  fallback_models   — ordered list of model IDs, one per family, for auto-retry
  verification_panel — [{model, family}, ...], one per family, for consensus checks

Usage:
    from omnicli.scanner import scan_all_providers, scan_provider
    results = scan_all_providers()          # all providers
    result  = scan_provider("openrouter")  # single provider
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from omnicli.config import ConfigManager, MODEL_FAMILY_KEYWORDS
from omnicli.client import OmniClient


def classify_family(model_id: str) -> str:
    """Return the family name for a model ID using MODEL_FAMILY_KEYWORDS.

    Matches on lowercase model_id substrings in keyword order — first match wins.
    Returns "other" when no keyword matches.
    """
    mid = model_id.lower()
    for family, keywords in MODEL_FAMILY_KEYWORDS:
        if any(kw in mid for kw in keywords):
            return family
    return "other"


def is_text_or_chat_model(mid: str) -> bool:
    """Filter out non-chat / special-purpose models like whisper, embeddings, guards."""
    low = mid.lower()
    return not any(
        x in low for x in ("whisper", "guard", "safeguard", "embed", "tts", "moderation", "audio", "orpheus", "vision-guard")
    )


def build_roster_and_panel(
    models: List[Dict[str, Any]],
    max_per_family: int = 2,
    panel_size: int = 3,
) -> Tuple[List[str], List[Dict[str, str]]]:
    """From a raw /models list build fallback_models and verification_panel.

    fallback_models: up to max_per_family models per known family, ordered by
    family diversity so the first retries are genuinely different from each
    other, followed by every unclassified ("other") text model. Unknown-family
    models are never dropped: no family signal exists to rank them, and
    dropping them hollows out router-native catalogs (e.g. bynara agnes-*).

    verification_panel: one model per known family first, then unclassified
    models fill remaining slots, capped at panel_size entries — used by
    CrossModelConsensusVerifier for blind voting. Non-empty whenever any
    text/chat model was discovered.
    """
    seen_families: Dict[str, List[str]] = {}
    for m in models:
        mid = m.get("id", "")
        if not mid or not is_text_or_chat_model(mid):
            continue
        family = classify_family(mid)
        seen_families.setdefault(family, []).append(mid)

    # Build fallback roster: round-robin across known families so family
    # diversity is maximised across the first N retries, not bunched by
    # family. Unclassified models append after — never dropped.
    family_iters = {
        f: iter(ids) for f, ids in seen_families.items() if f != "other"
    }
    roster: List[str] = []
    active = list(family_iters.keys())
    counts: Dict[str, int] = {f: 0 for f in active}
    while active:
        next_active = []
        for family in active:
            try:
                mid = next(family_iters[family])
                roster.append(mid)
                counts[family] += 1
                if counts[family] < max_per_family:
                    next_active.append(family)
            except StopIteration:
                pass
        active = next_active
    roster.extend(
        mid for mid in seen_families.get("other", []) if mid not in roster
    )

    # Build verification panel: one per known family, then unclassified
    # models fill remaining slots so router-native catalogs still get a panel.
    panel: List[Dict[str, str]] = []
    used_families: set = set()
    for mid in roster:
        if len(panel) >= panel_size:
            break
        family = classify_family(mid)
        if family == "other":
            if mid not in {entry["model"] for entry in panel}:
                panel.append({"model": mid, "family": family})
        elif family not in used_families:
            panel.append({"model": mid, "family": family})
            used_families.add(family)

    return roster, panel


def scan_provider(
    provider_id: str,
    cfg: Optional[ConfigManager] = None,
    client: Optional[OmniClient] = None,
    save: bool = True,
    panel_size: int = 3,
    max_per_family: int = 2,
) -> Dict[str, Any]:
    """Scan a single provider: fetch /models, classify, build roster + panel.

    Additive-only: scanned models merge INTO the stored roster/panel — a scan
    never removes existing entries and never rewrites default_model. Router
    catalogs rename models freely; heuristic rewrites destroyed working
    hand-written fallbacks, so the operator's config always wins.

    Returns:
        {
          "provider": str,
          "ok": bool,
          "model_count": int,
          "families_found": int,
          "fallback_models": [...],      # merged (existing first, then new)
          "verification_panel": [...],   # merged by model id
          "error": str | None,
          "latency_ms": float,
        }
    """
    cfg = cfg or ConfigManager()
    client = client or OmniClient(cfg)

    t0 = time.perf_counter()
    try:
        models = client.list_models(provider_id, timeout=10.0)
    except Exception as e:
        return {
            "provider": provider_id,
            "ok": False,
            "model_count": 0,
            "families_found": 0,
            "fallback_models": [],
            "verification_panel": [],
            "error": str(e),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    latency = round((time.perf_counter() - t0) * 1000, 1)
    roster, panel = build_roster_and_panel(
        models, max_per_family=max_per_family, panel_size=panel_size
    )
    families = {classify_family(m.get("id", "")) for m in models if m.get("id")}
    families.discard("other")

    # Additive merge: existing entries survive every scan; only genuinely
    # new models append. default_model is never touched here.
    existing_roster = list(cfg.get_fallback_models(provider_id) or [])
    merged_roster = existing_roster + [
        mid for mid in roster if mid not in existing_roster
    ]
    existing_panel = list(cfg.get_verification_panel(provider_id) or [])
    known_panel_models = {
        entry.get("model") for entry in existing_panel
        if isinstance(entry, dict)
    }
    merged_panel = existing_panel + [
        entry for entry in panel
        if entry.get("model") not in known_panel_models
    ]

    if save:
        cfg.save_scan_results(provider_id, merged_roster, merged_panel)

    return {
        "provider": provider_id,
        "ok": True,
        "model_count": len(models),
        "families_found": len(families),
        "fallback_models": merged_roster,
        "verification_panel": merged_panel,
        "error": None,
        "latency_ms": latency,
    }


def scan_all_providers(
    cfg: Optional[ConfigManager] = None,
    panel_size: int = 3,
    max_per_family: int = 2,
) -> List[Dict[str, Any]]:
    """Scan every configured provider and persist results.

    Providers that error (no key, unreachable) are reported but don't abort
    the scan — you get results for every provider that answered.
    """
    cfg = cfg or ConfigManager()
    client = OmniClient(cfg)
    results = []
    for provider_id in cfg.data.get("providers", {}):
        result = scan_provider(
            provider_id,
            cfg=cfg,
            client=client,
            save=True,
            panel_size=panel_size,
            max_per_family=max_per_family,
        )
        results.append(result)
    return results
