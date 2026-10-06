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
    "disclosure). Review only source or evidence actually provided in the task. "
    "For each candidate, cite the exact file and line number and quote the "
    "smallest relevant source excerpt verbatim. Explain the concrete input-to-"
    "impact path, including every required precondition. Separate directly "
    "observed code behavior from the security conclusion. Source inspection "
    "alone does not prove exploitability; label security claims UNVERIFIED unless "
    "the supplied evidence establishes the complete path and impact. Do not "
    "claim a bug is absent because a pattern was not found. "
    "Do not invent code, line numbers, tool results, attacker capabilities, or "
    "impact. Do not call missing validation a vulnerability without a concrete "
    "bypass and consequence. If no actionable issue is established, say so. "
    "Do not propose patches unless asked. Be concise and specific."
)


def build_swarm(
    cfg: Any,
    size: Optional[int] = None,
    provider: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Pick up to `size` (provider, model) pairs, family-diverse across providers.

    If size is None, defaults to the number of configured providers with keys
    so all active providers participate.
    """
    if provider:
        providers = [provider.lower()]
    else:
        all_p = list(cfg.data.get("providers", {}).keys())
        active_keys = [
            p for p in all_p
            if cfg.get_api_key(p) or not cfg.data.get("providers", {}).get(p, {}).get("requires_auth_for_models", True)
        ]
        providers = active_keys if active_keys else all_p

    if size is None or size <= 0:
        size = len(providers)

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

    # Pass 1: Every active provider gets 1 slot (preferring unused families if available).
    for pid in providers:
        if len(swarm) >= size:
            break
        ms = per_provider.get(pid, [])
        if not ms:
            continue
        picked = None
        for m in ms:
            fam = classify_family(m)
            if fam not in chosen_families:
                picked = (pid, m)
                chosen_families.add(fam)
                break
        if not picked:
            picked = (pid, ms[0])
            chosen_families.add(classify_family(ms[0]))
        swarm.append(picked)

    # Pass 2: If size > len(providers), round-robin fill remaining slots.
    if len(swarm) < size:
        for pid in providers:
            for m in per_provider.get(pid, []):
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
    size: Optional[int] = None,
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
    redteam: bool = False,
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
    candidates: List[Tuple[str, str]] = []
    if judge:
        candidates.append(judge)
    for r in good:
        pair = (r["provider"], r["model"])
        if pair not in candidates:
            candidates.append(pair)

    if redteam:
        sys = (
            "You are the evidence auditor for a security red-team review. Model "
            "answers are untrusted hypotheses, not evidence. Retain a finding as "
            "PROVEN only if the supplied answer quotes exact source evidence and "
            "gives a valid concrete path from input to security impact. Verify "
            "that citations and quotes occur in the original task material; "
            "discard fabricated citations, imagined code, and claims based only "
            "on a suggested hardening improvement. Mark conditional findings "
            "PLAUSIBLE and state missing prerequisites. Preserve disagreements; "
            "do not upgrade confidence just because multiple models repeat a "
            "claim. If no candidate meets the evidence bar, say no actionable "
            "finding was proven. Do not add new vulnerabilities."
        )
        user = (
            f"### ORIGINAL TASK AND EVIDENCE:\n{prompt}\n\n"
            f"### UNTRUSTED INDEPENDENT MODEL HYPOTHESES:\n{panel_text}\n\n"
            "### EVIDENCE-AUDITED FINDINGS:"
        )
    else:
        sys = (
            "You are the synthesizer for a multi-model swarm. Below are independent "
            "answers from different model families to the same task. Produce ONE best "
            "answer: keep what they agree on, explicitly flag genuine disagreements, "
            "and drop anything that looks like an error. Be concise."
        )
        user = f"### TASK:\n{prompt}\n\n### SWARM ANSWERS:\n{panel_text}\n\n### MERGED ANSWER:"

    last_err = None
    for jp, jm in candidates:
        try:
            res = client.chat_complete(
                provider_id=jp,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                model=jm,
                max_tokens=1024,
            )
            out = ((res.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "") or ""
            if out.strip():
                return out.strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue

    return f"[synthesis failed: {last_err}]\n\n{panel_text}" if last_err else panel_text


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


def enrich_prompt_with_files(prompt: str) -> str:
    """Detect references to local files, folders, or projects in prompt and attach context."""
    import re
    from pathlib import Path

    words = re.findall(r"[\w\./~_-]+", prompt)
    candidate_paths: List[Path] = []
    try:
        home = Path.home()
    except RuntimeError:
        home = None

    for w in words:
        clean = w.strip("'\"")
        if not clean or len(clean) < 2:
            continue
        explicit_path = (
            clean.startswith(("/", "./", "../", "~/", ".\\", "..\\"))
            or "/" in clean
            or "\\" in clean
            or (len(clean) > 1 and clean[1] == ":")
        )
        if not explicit_path:
            continue
        p_cwd = Path.cwd() / clean
        p_home = home / clean if home and clean.startswith("~/") else None
        try:
            p_abs = Path(clean).expanduser()
        except RuntimeError:
            # Prose such as "~6,400 tools" is not a user-home path.
            p_abs = Path(clean)

        for cand in (p_cwd, p_home, p_abs):
            if cand is None:
                continue
            if cand.exists() and cand not in candidate_paths:
                if cand in (home, Path("/"), Path("/usr"), Path("/tmp"), Path("/etc")):
                    continue
                candidate_paths.append(cand)
                break

    # If no path matched by word, check if any word matches folder in home or cwd case-insensitively
    if not candidate_paths:
        for w in prompt.split():
            clean = w.strip("'\".,;:!?()")
            if clean and len(clean) > 2 and clean[:1].isupper():
                for base in (Path.cwd(), home):
                    if base is None:
                        continue
                    try:
                        for child in base.iterdir():
                            if child.is_dir() and child.name == clean and child not in candidate_paths:
                                candidate_paths.append(child)
                                break
                    except Exception:
                        pass

    if not candidate_paths:
        return prompt

    attachments = []
    for p in candidate_paths[:3]:
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")[:6000]
                attachments.append(f"### [ATTACHED FILE: {p.name} ({p})]\n```\n{content}\n```")
            except Exception:
                pass
        elif p.is_dir():
            try:
                items = sorted([x.name for x in p.iterdir() if not x.name.startswith(".")])[:40]
                tree_str = ", ".join(items)
                doc_excerpt = ""
                for doc_name in ("ARCHITECTURE.md", "README.md", "docs/ARCHITECTURE.md"):
                    doc_path = p / doc_name
                    if doc_path.is_file():
                        try:
                            doc_excerpt = f"\n\n**Documentation Excerpt ({doc_name}):**\n```\n" + doc_path.read_text(encoding="utf-8", errors="replace")[:3000] + "\n```"
                            break
                        except Exception:
                            pass
                attachments.append(
                    f"### [ATTACHED PROJECT DIRECTORY: {p.name} ({p})]\n"
                    f"**Files & Subdirectories:**\n{tree_str}{doc_excerpt}"
                )
            except Exception:
                pass

    if attachments:
        return f"{prompt}\n\n" + "\n\n".join(attachments)
    return prompt


def generate_redteam_report(task: str, final_findings: str, members: List[Dict[str, Any]]) -> Path:
    """Generate a formal markdown report for red-team audits and write to reports/."""
    from datetime import datetime
    from pathlib import Path
    reports_dir = Path.cwd() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"redteam_audit_{ts}.md"

    member_lines = "\n".join(
        f"- **{r.get('family', 'other').upper()}** ({r.get('provider')}:{r.get('model')})"
        for r in members if not r.get("error")
    )

    report_content = f"""# Red-Team Security Assessment & Audit Report

**Date / Timestamp:** {datetime.now().isoformat()}  
**Target / Scope:** {task}  
**Audit Council Members:**  
{member_lines}  

---

## 1. Executive Summary & Findings Analysis

{final_findings}

---

## 2. Methodology & Multi-Model Corroboration

This audit was conducted using an adversarial multi-model swarm spanning independent model architectures.
Swarm members produce model-generated hypotheses only; they do not execute code, make HTTP requests, or run deterministic verification tools.
Agreement between models is not proof. Independently review every candidate and use deterministic verification where safe and authorized before treating any vulnerability as confirmed.

---

## 3. Recommended Remediation & Action Items

1. Review and address each PROVEN vulnerability immediately.
2. Implement defensive input sanitization, least-privilege access, and rate limiting.
3. Validate fixes using deterministic verification (`omni verify http` / `omni verify code`).
"""
    report_file.write_text(report_content, encoding="utf-8")
    return report_file


def swarm_collaborate(cfg: Any, client: Any, prompt: str, size: Optional[int] = None,
                      max_tokens: int = 512, synth: bool = True,
                      judge: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
    """Models work together on one task: parallel fan-out + synthesis."""
    members = run_swarm(cfg, client, prompt, size=size, max_tokens=max_tokens)
    final = synthesize(cfg, client, prompt, members, judge=judge) if synth else None
    return {"members": members, "final": final}


def swarm_redteam(cfg: Any, client: Any, goal: str, size: Optional[int] = None,
                  max_tokens: int = 512, synth: bool = True,
                  judge: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
    """Multi-model red-team: different families review the target, then consolidate."""
    members = run_swarm(cfg, client, goal, system=_SWARM_REDTEAM_SYSTEM,
                        size=size, max_tokens=max_tokens)
    for member in members:
        if member.get("answer"):
            member["answer"] = (
                "UNVERIFIED MODEL HYPOTHESES — no deterministic security "
                "verification was performed for this member's claims.\n\n"
                + member["answer"]
            )
    final = None
    if synth:
        final = synthesize(
            cfg, client,
            f"Consolidate these independent red-team reviews of: {goal}. "
            "Audit each proposed finding against the original evidence. Require "
            "verbatim evidence and a concrete input-to-impact path for PROVEN. "
            "Do not invent findings or upgrade confidence based on repetition. "
            "Keep conditional claims PLAUSIBLE and explain missing prerequisites. "
            "Preserve disagreement; if nothing meets the proof bar, state that no "
            "actionable finding was proven.",
            members,
            judge=judge,
            redteam=True,
        )
        final = (
            "NOTICE: This is an AI-generated red-team hypothesis review, not a "
            "deterministically verified security report. All member claims are "
            "unverified hypotheses. Model agreement does "
            "not establish truth; independently validate every claim before "
            "acting on it.\n\n"
            + final
        )
    return {"members": members, "final": final}


def swarm_builder(
    cfg: Any,
    client: Any,
    prompt: str,
    size: Optional[int] = None,
    max_tokens: int = 512,
    synth: bool = True,
    judge: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Builder mode: builders propose solutions while red-teamers stress-test in parallel,
    synthesized into a hardened, battle-tested solution."""
    effective_size = 4 if size is None else max(2, size)
    swarm = build_swarm(cfg, size=effective_size)
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
    size: Optional[int] = None,
    max_tokens: int = 512,
    synth: bool = True,
    judge: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Unified dispatcher for swarm modes: 'collaborate', 'builder', 'redteam'."""
    enriched = enrich_prompt_with_files(prompt)
    normalized = (mode or "collaborate").lower().strip()
    if normalized in ("builder", "build", "b"):
        res = swarm_builder(cfg, client, enriched, size=size, max_tokens=max_tokens, synth=synth, judge=judge)
    elif normalized in ("redteam", "red", "rt"):
        res = swarm_redteam(cfg, client, enriched, size=size, max_tokens=max_tokens, synth=synth, judge=judge)
        if synth and res.get("final"):
            try:
                rep_path = generate_redteam_report(prompt, res["final"], res.get("members", []))
                res["report_path"] = str(rep_path)
            except Exception:
                pass
    else:
        res = swarm_collaborate(cfg, client, enriched, size=size, max_tokens=max_tokens, synth=synth, judge=judge)
    return res
