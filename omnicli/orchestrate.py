"""
OmniCLI swarm orchestration.

A *swarm* runs several models from DIFFERENT families, spanning ALL configured
providers, on the same task AT ONCE (in parallel) and brings their outputs
together — the opposite of the one-model-at-a-time chat.

Modes:
  - collaborate: fan out a task, then synthesize one merged answer.
  - redteam:     fan out an adversarial review across families, then consolidate
                 the findings (multi-model red-team, not one model).
"""

from __future__ import annotations

import concurrent.futures as cf
from typing import Any, Dict, List, Optional, Tuple

from omnicli.scanner import classify_family

# Adversarial persona for a swarm member. Members have no tools, so (unlike the
# agent's tool-driven red-team prompt) this demands reasoned, proof-oriented
# review: separate proven from plausible, say what would confirm/refute, no fabrication.
_SWARM_REDTEAM_SYSTEM = (
    "You are one member of an adversarial red-team swarm. Rigorously probe the "
    "target for security issues (auth, access control, injection, SSRF, info "
    "disclosure). For every issue: state it as a one-line CLAIM, separate what is "
    "PROVEN from what is merely PLAUSIBLE, and say exactly what evidence would "
    "confirm or refute it. Never fabricate results. Be concise and specific."
)


def build_swarm(
    cfg: Any,
    size: int = 3,
    provider: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Pick up to `size` (provider, model) pairs, family-diverse across providers.

    Round-robins across every configured provider (or just `provider` if given),
    choosing one model per not-yet-seen family first so the swarm spans both APIs
    and architectures. Falls back to filling remaining slots if there aren't
    enough distinct families.
    """
    providers = [provider.lower()] if provider else list(cfg.data.get("providers", {}).keys())

    per_provider: Dict[str, List[str]] = {}
    for pid in providers:
        models: List[str] = []
        dm = cfg.get_default_model(pid)
        if dm:
            models.append(dm)
        models += list(cfg.get_fallback_models(pid))
        seen: set = set()
        ordered: List[str] = []
        for m in models:
            if m and m not in seen:
                seen.add(m)
                ordered.append(m)
        per_provider[pid] = ordered

    swarm: List[Tuple[str, str]] = []
    chosen_families: set = set()
    idx: Dict[str, int] = {pid: 0 for pid in providers}
    active = [pid for pid in providers if per_provider[pid]]

    # Pass 1: round-robin, one new family per provider per lap.
    while active and len(swarm) < size:
        for pid in list(active):
            ms = per_provider[pid]
            picked: Optional[Tuple[str, str]] = None
            while idx[pid] < len(ms):
                m = ms[idx[pid]]
                idx[pid] += 1
                fam = classify_family(m)
                if fam not in chosen_families:
                    picked = (pid, m)
                    chosen_families.add(fam)
                    break
            if picked:
                swarm.append(picked)
                if len(swarm) >= size:
                    break
            if idx[pid] >= len(ms):
                active.remove(pid)

    # Pass 2: not enough distinct families — fill remaining slots with any models.
    if len(swarm) < size:
        for pid in providers:
            for m in per_provider[pid]:
                if (pid, m) not in swarm:
                    swarm.append((pid, m))
                    if len(swarm) >= size:
                        break
            if len(swarm) >= size:
                break

    return swarm[:size]


def _one_call(client: Any, provider: str, model: str,
              messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
    """Single swarm-member call. Errors are captured, never raised."""
    try:
        res = client.chat_complete(
            provider_id=provider, messages=messages, model=model, max_tokens=max_tokens
        )
        content = ((res.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "") or ""
        return {"provider": provider, "model": model, "family": classify_family(model),
                "answer": content.strip(), "error": None}
    except Exception as e:  # noqa: BLE001
        return {"provider": provider, "model": model, "family": classify_family(model),
                "answer": "", "error": str(e)[:160]}


def run_swarm(
    cfg: Any,
    client: Any,
    prompt: str,
    system: Optional[str] = None,
    swarm: Optional[List[Tuple[str, str]]] = None,
    size: int = 3,
    max_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """Fan a prompt out to the swarm IN PARALLEL; one record per member, in swarm order."""
    swarm = swarm if swarm is not None else build_swarm(cfg, size=size)
    if not swarm:
        return []
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    results: List[Dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, len(swarm))) as ex:
        futs = [ex.submit(_one_call, client, p, m, messages, max_tokens) for (p, m) in swarm]
        for fut in cf.as_completed(futs):
            results.append(fut.result())
    order = {(p, m): i for i, (p, m) in enumerate(swarm)}
    results.sort(key=lambda r: order.get((r["provider"], r["model"]), 1_000_000))
    return results


def synthesize(
    cfg: Any,
    client: Any,
    prompt: str,
    results: List[Dict[str, Any]],
    judge: Optional[Tuple[str, str]] = None,
) -> str:
    """Merge swarm answers into one via a single member; keep agreements, flag disagreements."""
    good = [r for r in results if not r["error"] and r["answer"]]
    if not good:
        return "All swarm members failed to respond."
    if len(good) == 1:
        return good[0]["answer"]
    panel_text = "\n\n".join(
        f"[{r['family']} | {r['provider']}:{r['model']}]\n{r['answer']}" for r in good
    )
    jp, jm = judge if judge else (good[0]["provider"], good[0]["model"])
    sys = (
        "You are the synthesizer for a multi-model swarm. Below are independent "
        "answers from different model families to the same task. Produce ONE best "
        "answer: keep what they agree on, explicitly flag genuine disagreements, "
        "and drop anything that looks like an error. Be concise."
    )
    user = f"### TASK:\n{prompt}\n\n### SWARM ANSWERS:\n{panel_text}\n\n### MERGED ANSWER:"
    try:
        res = client.chat_complete(
            provider_id=jp,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            model=jm,
            max_tokens=1024,
        )
        out = ((res.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "") or ""
        return out.strip() or panel_text
    except Exception as e:  # noqa: BLE001
        return f"[synthesis failed: {e}]\n\n{panel_text}"


_SWARM_BUILDER_SYSTEM = (
    "You are the BUILDER / ARCHITECT in a multi-model engineering council. "
    "Your objective is to produce the cleanest, most robust, production-grade "
    "design, code, and implementation for the task. Focus on modularity, correct "
    "architecture, and concrete executable steps."
)

_SWARM_BREAKER_SYSTEM = (
    "You are the RED-TEAM / BREAKER in a multi-model engineering council. "
    "Your objective is to stress-test and attack the proposed solutions for this task. "
    "Identify edge cases, security vulnerabilities, race conditions, scaling bottlenecks, "
    "and invalid assumptions. Be specific and rigorous."
)


def swarm_collaborate(cfg: Any, client: Any, prompt: str, size: int = 3,
                      max_tokens: int = 512, synth: bool = True,
                      judge: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
    """Models work together on one task: parallel fan-out + synthesis."""
    members = run_swarm(cfg, client, prompt, size=size, max_tokens=max_tokens)
    final = synthesize(cfg, client, prompt, members, judge=judge) if synth else None
    return {"members": members, "final": final}


def swarm_redteam(cfg: Any, client: Any, goal: str, size: int = 3,
                  max_tokens: int = 512, synth: bool = True,
                  judge: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
    """Multi-model red-team: different families review the target, then consolidate."""
    members = run_swarm(cfg, client, goal, system=_SWARM_REDTEAM_SYSTEM,
                        size=size, max_tokens=max_tokens)
    final = None
    if synth:
        final = synthesize(
            cfg, client,
            f"Consolidate these independent red-team reviews of: {goal}. "
            "Merge overlapping findings, rank by severity, keep only proof-backed or "
            "clearly-plausible issues.",
            members,
            judge=judge,
        )
    return {"members": members, "final": final}


def swarm_builder(
    cfg: Any,
    client: Any,
    prompt: str,
    size: int = 3,
    max_tokens: int = 512,
    synth: bool = True,
    judge: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Builder mode: builders propose solutions while red-teamers stress-test in parallel,
    synthesized into a hardened, battle-tested solution."""
    swarm = build_swarm(cfg, size=max(2, size))
    if not swarm:
        return {"members": [], "final": None}

    results: List[Dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, len(swarm))) as ex:
        futs = []
        for i, (p, m) in enumerate(swarm):
            role = "BUILDER" if (i % 2 == 0) else "REDTEAM"
            sys_prompt = _SWARM_BUILDER_SYSTEM if role == "BUILDER" else _SWARM_BREAKER_SYSTEM
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
            futs.append((role, ex.submit(_one_call, client, p, m, messages, max_tokens)))
        for role, fut in futs:
            res = fut.result()
            res["role"] = role
            results.append(res)

    order = {(p, m): i for i, (p, m) in enumerate(swarm)}
    results.sort(key=lambda r: order.get((r["provider"], r["model"]), 1_000_000))

    final = None
    if synth:
        synth_prompt = (
            f"You are the lead architect synthesizing a battle-hardened solution for:\n{prompt}\n\n"
            "Below are independent proposals from BUILDER models and adversarial stress-tests from REDTEAM models.\n"
            "Produce the final, hardened build: take the best architectural solutions from the Builders, "
            "and directly fortify them against the vulnerabilities and failure modes identified by the Red Team. "
            "Deliver the concrete, robust solution."
        )
        final = synthesize(cfg, client, synth_prompt, results, judge=judge)
    return {"members": results, "final": final}


def run_orchestration(
    mode: str,
    cfg: Any,
    client: Any,
    prompt: str,
    size: int = 3,
    max_tokens: int = 512,
    synth: bool = True,
    judge: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Unified dispatcher for swarm modes: 'collaborate', 'builder', 'redteam'."""
    normalized = (mode or "collaborate").lower().strip()
    if normalized in ("builder", "build", "b"):
        return swarm_builder(cfg, client, prompt, size=size, max_tokens=max_tokens, synth=synth, judge=judge)
    elif normalized in ("redteam", "red", "rt"):
        return swarm_redteam(cfg, client, prompt, size=size, max_tokens=max_tokens, synth=synth, judge=judge)
    else:
        return swarm_collaborate(cfg, client, prompt, size=size, max_tokens=max_tokens, synth=synth, judge=judge)
