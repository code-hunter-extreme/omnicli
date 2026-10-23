# OmniCLI Red-Team Quality Plan

## Goal

Improve OmniCLI's source-code red-team usefulness by reducing unsupported
findings and missed, evidence-backed findings. Do not use model consensus,
citation presence, or a clean report as proof of security. This is a bounded,
measurable gate on reviewed cases, not a claim of universal coverage.

## How downloaded references are used

The repositories are design and evaluation references, not runtime
dependencies:

- **OpenRedTeaming** informs threat-surface coverage and the separation of
  attack hypotheses, defenses/challenges, and evaluation metrics. OmniCLI now
  executes objective-labeled scenarios and reports completion/cue coverage;
  unrelated jailbreak payloads are not copied into source review.
- **Awesome-LLM-Red-Teaming** is a map of methods and research resources; use it
  to identify applicable techniques, not as an implementation to import. The
  category structure informs coverage tracking, not source-security verdicts.
- **Microsoft AI Red-Teaming Playground Labs** informs explicit challenge
  objectives, scoped scenarios, difficulty, and repeatable scoring. OmniCLI
  applies this structure to source-security questions; the playground app and
  out-of-scope harmful-content challenges are not imported.
- **llm-red-teaming-dataset** informs category, subcategory, and severity
  metadata only. Its harmful-content prompts test a different problem and are
  not imported.
- **PrimeVul** informs paired vulnerable/fixed examples, duplicate control, and
  leakage-aware splits. Its separately hosted C/C++ dataset is not imported.
- **OWASP BenchmarkJava** informs explicit positive/negative expected-result
  labels and scorecards. Its Java app is not run or imported.

The four AI-safety references have top-level MIT license files; OWASP
BenchmarkJava is GPL-2.0. This plan does not copy source, assets, notebooks,
or prompt payloads; inspect nested and upstream notices separately before
considering reuse.

Additional agent red-team and orchestration references are useful for
harness design, but do not provide source-code vulnerability labels:

- **Acme Refund Agent** and **Doomcaller** separate adversarial interaction,
  the system under test, and deterministic grading of observed actions or
  transcripts. Their fail-closed treatment of non-answers motivates keeping
  incomplete/provider-failed trials distinct from clean reviews. OmniCLI's
  source stages do not execute target code, so citations and AST cues remain
  leads rather than runtime proof.
- **Hexbreaker** demonstrates generated cases, a separate answer key,
  reproducible artifacts, and explicit reporting of leakage and scope limits.
  Keep the OmniCLI holdout blinded and independently reviewed; neither its
  forensic cases nor its reported scores transfer to code-review accuracy.
- **CLI-Fleet/fleetcode** and **Agent Colosseum** suggest non-overlapping
  parallel roles, model diversity, and injected provider failures for
  orchestration tests. Use these as test ideas for OmniCLI's existing
  dispatcher, not as another runtime dependency. Their file-mailbox design's
  documented polling and deduplication limitations are not adopted.
- **Invisible Seam** reinforces running deterministic checks against claims
  where feasible. OmniCLI's quote validation and labeled benchmark scoring
  are useful controls, but a source quote alone does not prove a security
  behavior.

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

### Red-team trust and completeness safeguards

- Red-team mode reads only source supplied through `--source-file` or an
  explicitly labeled source block; it does not infer local files from prompt
  paths or project names. The CLI lists the files and character counts before
  sending them to configured model providers.
- A citation must match the complete cited range, start on its cited first
  line, and span no more than eight lines. Broader surrounding context may
  still be useful to a model, but cannot be used to attribute a quote to a
  different line.
- Empty provider replies are stage failures. Every attack focus must return a
  structured candidate or explicitly say `NO ACTIONABLE CANDIDATE`; every
  cross-examination must disposition each indexed candidate, or say
  `NO CANDIDATES TO CHALLENGE` when the index is empty.
- Staged and direct offline evaluation both run at temperature 0. Interactive
  runs continue to use provider defaults; record model/provider settings when
  comparing those results.
- Optional `pair_id`, `pair_role`, and `pair_category` metadata connects a
  vulnerable source case to its fixed counterpart. Paired accuracy is reported
  separately: a pair counts as correct only when the vulnerable case is
  exactly detected and the fixed case has no false positive. This supplements
  per-case metrics; it is not evidence of broad performance.
- Four objective-labeled source-security scenarios execute as pipeline stages.
  After first-pass calls, deterministic Python AST observations identify
  security-relevant request-to-sink and unguarded-mutation locations not
  covered by grounded citations. If gaps exist, one bounded adaptive follow-up
  per model examines those exact cues. The report exposes scenario completion
  and per-cue citation telemetry; neither is a security verdict.
- Reports retain the matched quote, category, model-described attack path and
  preconditions, challenge opinions, and suggested safe tests, clearly marked
  as unverified and not executed.

### Held-out adjudication procedure

The second reviewer should work from a copy with `expected_findings`,
`benign_rationale`, and severity concealed. Independently record finding
category, exact file/line/evidence, impact and required attacker preconditions,
or explain why the case is benign. Compare both label sets only after the
independent pass. Resolve disagreements in writing; remove ambiguous cases
from the held-out set rather than tuning the implementation to them. Mark cases
`independently_reviewed` with `reviewer_count: 2` only after the comparison and
resolution are complete. Do not edit held-out labels after evaluating model
outputs; disputed cases return to development and require a new held-out case.

## Current status

- Implemented: source citation grounding, fail-closed evidence gate, staged
  objective-labeled attack/challenge scenarios, conditional adaptive
  evidence-gap follow-up, and conservative Python route facts.
- Current development benchmark: 24 self-authored, labeled cases (12 positive,
  12 benign) with exact source-line evidence, split/category/severity metadata,
  positive preconditions, and clean-case rationales. It includes sink-control
  near misses and same-module auth-hook context cases. It is a tuning fixture,
  not a held-out benchmark.
- One development pair contrasts SQL interpolation with parameter binding.
  The deterministic baseline detects the vulnerable variant and leaves the
  fixed variant unflagged in this one-pair smoke check; the sample size is not
  meaningful evidence of general accuracy.
- Development-only tuning observation: the local static baseline now detects
  12/12 labeled positives with 0/12 benign-case false positives on this 24-case
  fixture. These cases were used to develop the analyzer; this score is
  expected to be optimistic and is not a quality-gate result.
- A separate 60-case held-out candidate set now exists at
  `tests/fixtures/redteam/cases.held_out.provisional.jsonl` (30 positive, 30
  benign across six source-security categories). It is explicitly marked
  `provisional` with one reviewer; the CLI and gate refuse to make provider
  calls or pass the gate until a second reviewer has independently checked
  every label and the records are updated to two-person reviewed status.
- Implemented evaluation increment: repeated direct and staged trials; exact
  category/file/line/evidence scoring; per-category counts, clean-case
  specificity, separately measured source-line quote accuracy, 95% Wilson
  intervals, latency, token totals, optional cost, and explicit provider/format
  errors. Staged mode runs four objective-labeled attack scenarios, followed
  by a challenge and a conditional evidence-gap follow-up, on the same source
  cases; stage failures fail the trial. A local deterministic
  pattern-based baseline also runs on the same cases, remains a heuristic, and
  is excluded from model recommendations. `--quality-gate --split held_out`
  first checks dataset size, recognized model-family count, and repetitions
  and two-person adjudication metadata before any provider calls, then checks
  paired trials, pooled and per-family thresholds, provider/evidence failures,
  and mode recall non-regression. Missing model pairs and failed calls cannot
  be hidden by excluding them from the scored set. Provisional labels cannot
  pass. It emits per-category and per-family scores and intervals; failure
  exits with code 2.
- Evaluator diagnostics retain raw direct-model responses, distinguish failed
  clean-case trials from false positives, and separately expose exact-evidence
  category mismatches. Failed staged trials retain candidate observations for
  review but do not count those observations as valid false positives or
  grounded citations. This improves failure diagnosis; it does not improve
  model accuracy or replace an actual multi-provider council evaluation.
- `evaluate-security --staged-redteam` runs the selected roster together
  through the actual cross-model attack/challenge pipeline; absent explicit
  `--model` options, it uses OmniCLI's existing active default swarm.
  Held-out `--quality-gate` additionally runs per-model staged comparisons for
  per-family metrics; one-member challenges are self-critique, not independent
  corroboration. Challenge assignments now carry a per-candidate `self_review`
  flag, and the run metrics distinguish independent, mixed, and self-only
  review. A four-case pilot did not evaluate the full roster and cannot
  establish orchestration quality.
  No additional council or roster-selection flags were added.
- The staged prompt now explicitly tells models to omit the five-digit line
  display prefix from source quotes. The grounding validator may strip that
  prefix only when its number exactly matches the cited line; quotes are then
  validated against original source code and an incorrect line number still
  fails.
- Implemented triage hardening: a same-module request hook no longer suppresses
  deterministic authorization candidates; hook presence stays contextual and
  unverified. Challenge outputs now refer to stable candidate IDs, and their
  dispositions are preserved as model challenge opinions without overriding
  the underlying unverified-candidate status.
- The live development council exposed oversized cross-examination context and
  provider-specific triage formats. Cross-examination now distributes concise,
  evidence-carrying candidate cards across reviewers and checks that each
  assigned card received a parsed disposition; it no longer sends every
  challenger the entire attack transcript. Parsing accepts the structured
  output variations observed in the live provider responses. This improves
  orchestration reliability, not measured detection quality.
- Staged evaluation reports now retain stage calls, attack/challenge
  validations, candidate triage, scenario coverage, and adaptive-cue outcomes
  so invalid live trials can be diagnosed from artifacts rather than aggregate
  gate counts alone.
- Implemented a single conditional format-only correction for attack responses
  that contain grounded findings plus uncited prose. A correction replaces the
  original response only when it validates, preserves the exact set of
  grounded file/range/category/quote citations, and retains all required
  structured finding fields including `CATEGORY`; rejected corrections remain
  visible as stage errors and do not relax the evidence gate. Unit tests cover
  accepted corrections, citation-changing corrections, and omitted categories.
- Bounded live development check after this change:
  `reports/redteam_live_format_correction_20261006.json` records a two-case,
  two-model run with 23 staged member calls. The benign case passed
  `PASS_GROUNDING_ONLY`; the vulnerable case remained invalid because all three
  Cloudflare format corrections failed exact-citation preservation and the
  council also returned unsupported/unstructured attack output. The run had
  one evidence-gate failure, no exact true positives, and 25,182 input / 3,622
  output tokens; provider cost was not reported. This confirms fail-closed
  behavior, not improved vulnerability detection or general council quality.
- A subsequent same-case experiment tightened scenario-scope wording and the
  correction instructions. It still failed both trials (one incomplete attack
  review and one unstructured benign response); the scope wording was reverted
  rather than claiming a gain from this noisy two-case run. The report is
  `reports/redteam_live_scope_correction_20261006.json`. The format-correction
  instruction now explicitly requires preserving `CATEGORY`, addressing the
  concrete omission seen in prior correction responses; this prompt adjustment
  has regression coverage but has not yet been revalidated in a live run.
- Implemented context increment: source facts now expose function-local
  request-derived variables, selected security sinks/call sites, and
  same-module request hooks. Hooks are surfaced as context, not treated as
  proof that a route is guarded; a health-handler authorization hypothesis is
  no longer screened as non-actionable when a same-module hook exists.
- Red-team output now includes scenario-level review completion and records
  which uncovered AST cues receive grounded citations after the adaptive
  follow-up. These are process-coverage signals, not accuracy claims.
- Staged-review candidate identity and deduplication use normalized cited
  evidence (path, line range, and quote), not model-written titles or category
  names. When reviewers label the same exact evidence with different
  categories, keep the category variants on one evidence candidate and
  challenge it once; do not erase the disagreement or call a category exact.
  Candidate checkpoint version 5 forces old, category-specific IDs to be
  regenerated rather than silently reusing them.
- Each evidence candidate is assigned to as many as two non-reporting council
  members for cross-examination; reviewer selection balances assignments and
  prefers distinct classified model families. This reuses the already
  scheduled per-member challenge calls, adding no provider calls, while giving
  each candidate more than one independent critique when the roster permits.
  A small roster may provide only one non-reporting challenger; if every member
  reported the evidence, the pipeline falls back to self-critique and must not
  present that as independent review. Challenge dispositions remain opinions.
- Live council report
  `reports/redteam_live_three_provider_challenge_20261006.json` exposed a
  challenge-parser mismatch: XKiro returned `TRIAGE: UNVERIFIED`, which had
  not counted as a disposition, despite including both assigned candidate IDs.
  The parser now normalizes `UNVERIFIED` to `UNRESOLVED` (an opinion, not a
  verdict). Replaying the captured authorization-case outputs with this parser
  changed its evidence gate from `FAIL` (one incomplete assignment) to
  `PASS_GROUNDING_ONLY`; all three challenge assignments parsed complete.
  This is a demonstrated orchestration-completeness fix on actual provider
  output, not a new model call or a measured increase in vulnerability
  accuracy. The same live run's benign case remained failed because Bynara
  returned a DNS/network error; cost and complete token usage were unavailable.
- The bounded Python source index now links a route's call sites to matching
  helper definitions and exposes request hooks from other explicitly supplied
  files. It does not discover files, infer runtime wiring, or treat hooks as
  proof of protection. These compact links are included in attack,
  adaptive-gap, and challenge context. Regression tests cover a cross-file
  mutating helper and request hook; a model-quality gain has not been
  established.
- An incomplete challenge assignment now gets one recovery call, limited to
  that reviewer's missing candidate IDs. Parsed first-round dispositions are
  retained and only valid missing-ID recovery dispositions are merged. A
  failed or still-incomplete retry remains an evidence-gate failure, while
  retry errors are attached to challenge diagnostics. Regression tests cover
  successful completion, still-incomplete output, provider error, and
  preservation of a valid first-round opinion. This improves bounded review
  completion; it does not establish improved detection accuracy.
- Live development smoke evaluation
  `reports/redteam_live_context_recovery_dev_case.json` ran the existing
  authorization-positive and clean-health pair through NVIDIA and XKiro: 20
  council calls, zero provider errors, zero evidence-gate failures, and
  `PASS_GROUNDING_ONLY` on both cases. No challenge recovery was triggered
  because all assigned challenge reviews completed normally; the cases were
  single-file and do not validate the cross-file context path live. Council
  results had 0 exact true positives, 3 false positives, and 1 false negative
  on the vulnerable case; the clean case had no false positives. Usage was
  23,364 input and 2,460 output tokens, cost unavailable. The small smoke test
  checks end-to-end operation, not a quality gain.
- The prior captured authorization review replayed through this evidence
  identity: eight model finding mentions collapsed into two distinct
  file/range/quote candidates, retaining both category labels on the duplicated
  evidence candidate. A two-case live regression run after this change used
  the existing Cloudflare/XKiro roster and 20 council calls, but both case
  trials failed because Cloudflare returned HTTP 429 after exhausting its daily
  free allocation. The run therefore does not measure whether the candidate
  deduplication improved model accuracy; full usage and cost were unavailable.
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
  24-case fixture is deliberately only a development smoke set. No qualifying
  cross-family held-out trials have been run; the quality gate is not met. Do
  not promote development cases into held-out merely to satisfy its numeric
  threshold.
- The 60-case held-out candidate set remains provisional pending independent
  second review. A blinded reviewer packet is available under
  `reports/redteam_review_packet_20261006.jsonl`; keep the private answer key
  separate and reconcile disagreements only after the reviewer returns labels.
- Rowan's reference-based judge and typed stage-graph rebuild is being tested
   separately at `/home/phantomcore/tools/omnicli-redteam-hardening` so the
   existing changes in this primary checkout are not overwritten. Its direct
   design references are garak, DeepTeam, PyRIT, Microsoft Agent Framework,
   LangGraph, CrewAI, and AutoGen; the rebuild documents which patterns each
   informs in `docs/REPOSITORIES_SURVEYED.md`. The earlier source-code and
   adversarial-evaluation references above remain complementary; no additional
   repository is currently needed to address the observed judge failures.
- The rebuild's evaluator now scores final judge `UPHOLD` verdicts instead of
   raw probe mentions. Probe citation failures remain visible telemetry and do
   not erase a post-judge score; provider errors still fail the trial. After
   diagnosis, the rebuild also added request-derived-argument evidence to
   authorization route facts, constrained KILL-to-UPHOLD contradiction flips
   to grounded candidates, and re-anchors an authorization candidate to a
   unique mutating call only when a route has no detected local guard or
   auth-enforcing hook. Original model citations are retained for audit, and
   normalized duplicates are judged once.
- Rebuild validation: the isolated worktree passes 334 tests and 51 subtests,
   plus compilation and whitespace checks. A two-case, one-provider live
   diagnostic changed from 0 TP / 3 FP / 1 FN in the staged run to 1 TP / 0 FP
   / 0 FN; the paired single-pass result was 0 TP / 1 FP / 1 FN. The staged run
   still had 54.55% raw probe citation accuracy and two evidence-gate failures.
   This tiny run is only a debugging signal, not a quality claim. The primary
   checkout has not yet received the rebuild; integration must preserve its
   existing dirty changes and be followed by command-path and provider-free
   council tests.
- The rebuild's development-only fixture is 26 cases (13 positive, 13
   benign), separate from the primary checkout's 24-case development fixture.
   Its static baseline scores 13/13 positives and 0/13 benign false positives
   on those self-authored tuning cases; this optimistic result is not held-out
   evidence. Do not silently mix the fixtures or promote development cases into
   the blinded held-out set.

## Next execution batch

1. Review and integrate the validated Rowan rebuild into the primary checkout
    without replacing or discarding its existing edits; run focused command-path
    and provider-free council tests in the integrated checkout.
2. Continue expanding and reviewing development-only cases around
    control-flow/data-flow near misses and framework-level authorization
    context. Diagnose any live failure before tuning, and compare changes on
    identical paired cases.
3. Complete independent adjudication of the blinded held-out packet; resolve
   disagreements before changing the gold labels.
4. Check configured model-family availability without exposing credentials;
   coordinate any substantial live evaluation cost before running the
   three-family, two-repetition matrix.
5. Run the same held-out cases through single-pass and staged modes, publish
   the complete machine-readable report and concise adjudication, then apply
   the bounded gate. Oracle stays blocked until that evidence passes review.
