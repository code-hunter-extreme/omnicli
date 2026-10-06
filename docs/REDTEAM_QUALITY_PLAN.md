# OmniCLI Red-Team Quality Plan

## Goal

Improve OmniCLI's source-code red-team usefulness by reducing unsupported
findings and missed, evidence-backed findings. Do not use model consensus,
citation presence, or a clean report as proof of security. This is a bounded,
measurable gate on reviewed cases, not a claim of universal coverage.

## How the four downloaded references are used

The repositories are design and evaluation references, not runtime
dependencies:

- **OpenRedTeaming** informs threat-surface coverage and the separation of
  attack hypotheses, defenses/challenges, and evaluation metrics.
- **Awesome-LLM-Red-Teaming** is a map of methods and research resources; use it
  to identify applicable techniques, not as an implementation to import.
- **Microsoft AI Red-Teaming Playground Labs** informs explicit challenge
  objectives, scoped scenarios, difficulty, and repeatable scoring.
- **llm-red-teaming-dataset** informs category, subcategory, and severity
  metadata. Its harmful prompt payloads are outside OmniCLI's source-code
  security scope and are not imported.

All four repositories have top-level MIT license files. This plan does not
copy their source, assets, notebooks, or prompt payloads; inspect nested and
upstream notices separately before considering reuse.

## Implementation phases

1. **Curate a development benchmark.** Add labeled, self-contained Python
   source cases covering authorization, tainted input/sinks, data exposure,
   and safe near-misses. Every positive has exact evidence, expected category,
   impact/preconditions, and a reviewed rationale. Every negative explains
   why the behavior is intentionally safe. Keep tuning/development and held-out
   cases separate.
2. **Improve deterministic context.** Extend route facts with bounded
   middleware/decorator, helper/sink, and call-site context. Reuse existing
   deterministic analyzers where their supported claim matches the code.
   Static absence is never proof of safety; distinguish observations,
   unverified candidates, unsupported/refuted claims, and runtime-confirmed
   behavior.
3. **Build repeatable mode comparisons.** Evaluate deterministic checks,
   single-pass, and staged red-team on identical cases, model rosters, and
   repeated runs. Record per-category recall, false positives/specificity,
   exact citation accuracy, unsupported outputs, evidence-gate failures,
   provider failures, latency, tokens, and provider-reported/priced cost when
   available. Include confidence intervals and retain raw outputs for review.
4. **Tune only on development cases.** Change prompts/triage from observed
   failure classes; re-run all development cases. Do not use held-out results
   to tune.
5. **Apply the transition gate.** On at least 30 held-out vulnerable and 30
   held-out benign cases, across at least three healthy model families and two
   repetitions, require pooled recall >=85%, specificity >=90%, and exact
   citation accuracy >=95%. Publish per-category results and confidence
   intervals. Staged mode must not regress recall versus the single-pass
   baseline without an explicit, documented trade-off. A provider error is a
   failed trial, not a skipped case.
6. **Then move focus to Oracle.** Once the bounded gate passes, begin Oracle's
   authorized target-to-reproducible-finding workflow. OmniCLI may be used
   internally by Oracle; OmniCLI must not depend on or call Oracle.

## Current status

- Implemented: source citation grounding, fail-closed evidence gate, staged
  attack/challenge orchestration, conservative Python route facts.
- Current development benchmark: 12 self-authored, labeled cases (6 positive,
  6 benign) with exact source-line evidence, split/category/severity metadata,
  positive preconditions, and clean-case rationales. It is a tuning fixture,
  not a held-out benchmark.
- Implemented evaluation increment: repeated direct and staged trials; exact
  category/file/line/evidence scoring; per-category counts, clean-case
  specificity, separately measured source-line quote accuracy, 95% Wilson
  intervals, latency, token totals, optional cost, and explicit provider/format
  errors. Staged mode runs the actual four-focus attack/challenge pipeline on
  the same source cases; stage failures fail the trial. A local deterministic
  pattern-based baseline also runs on the same cases, remains a heuristic, and
  is excluded from model recommendations. `--quality-gate --split held_out`
  checks the minimum data/model/repetition requirements, pooled thresholds,
  mode recall non-regression, and emits per-category scores and intervals;
  failure exits with code 2.
- Implemented context increment: source facts now expose function-local
  request-derived variables, selected security sinks/call sites, and
  same-module request hooks. Hooks are surfaced as context, not treated as
  proof that a route is guarded; a health-handler authorization hypothesis is
  no longer screened as non-actionable when a same-module hook exists.
- Not yet established: representative recall, specificity, citation accuracy,
  or performance across model families. The prior live pilot produced false
  claims; do not describe OmniCLI as a dependable vulnerability auditor.
- The evaluator now retains raw per-case model outputs and computes
  `citation_accuracy` separately from exact gold-finding matches. Exact scoring
  requires category, file, line, and evidence; citation accuracy only asks
  whether each quoted excerpt matches its cited source line. Neither metric
  proves exploitability.
- Remaining before gate: more independently reviewed development cases,
  independently authored/adjudicated held-out benchmark, and healthy live
  runs from three model families over at least two repetitions. The current
  12-case fixture is deliberately only a development smoke set. No cross-family
  live trials have been run; the quality gate is not met. Do not promote
  development cases into held-out merely to satisfy its numeric threshold.

## Next execution batch

1. Broaden the development set around control-flow/data-flow near misses and
   framework-level authorization context; independently review expected
   evidence and preconditions.
2. Prepare a separate held-out set with no development overlap, independently
   adjudicated by a second reviewer. If a second reviewer is unavailable,
   label it provisional and do not use it to claim the gate passed.
3. Check configured model-family availability without exposing credentials;
   coordinate any substantial live evaluation cost before running the
   three-family, two-repetition matrix.
4. Run the same held-out cases through single-pass and staged modes, publish
   the complete machine-readable report and concise adjudication, then apply
   the bounded gate. Oracle stays blocked until that evidence passes review.
