# Transition-Gate Status (P4)

The Transition Gate is defined in
[REDTEAM_QUALITY_PLAN.md](REDTEAM_QUALITY_PLAN.md) (phase 5) and enforced by
`assess_redteam_quality_gate` in
[omnicli/evaluation.py](../omnicli/evaluation.py). The roadmap's narrative is
in [OMNICLI_PROGRESS_AND_ROADMAP_20261007.md](OMNICLI_PROGRESS_AND_ROADMAP_20261007.md)
(section 4). This file records what exists, what the P4 anatomy check covers,
and what remains for a human reviewer. No qualifying run has passed yet.

## What exists (verified 2026-10-12)

- Held-out candidate set:
  `tests/fixtures/redteam/cases.held_out.provisional.jsonl` — 60 cases
  (30 positive, 30 benign), `adjudication_status: provisional`,
  `reviewer_count: 1`. Case counts already meet the 30/30 minimum; only the
  second review is missing.
- Blinded review packet for the 2nd reviewer:
  `reports/redteam_blinded_review_packet_20261007.json` (60 neutralized cases
  with SHA-256 integrity hash) plus the review guide
  `reports/redteam_review_packet_guide_20261006.md`.
- Gate enforcement: the CLI refuses provider calls until the dataset is 30/30
  held-out, two-person adjudicated, across 3 model families x 2 repetitions
  (`omnicli/cli.py`); the gate then requires paired single-pass/staged trials,
  pooled and per-family bars, zero excluded failures, and no staged-recall
  regression (`omnicli/evaluation.py`).
- Qualifying-run anatomy validator: `validate_qualifying_run` in
  `omnicli/evaluation.py` re-derives families from the paired rows and
  re-checks every bar, so a hand-edited artifact cannot pass.
- Gate test: `tests/test_p4_council_rulings.py::test_qualifying_run_artifact_exists_and_validates`
  FAILS until `reports/redteam_qualifying_run.json` exists and validates.

## Binding bars (note the discrepancy)

The roadmap's Task-3 checklist lists looser numbers (F1 >= 0.70,
specificity >= 0.85, citation >= 0.90). The quality plan (phase 5) and the
implemented gate enforce stricter bars, and those are authoritative:

- pooled recall >= 0.85 (single-pass and staged, overall and per family)
- clean-case specificity >= 0.90 (single-pass and staged, overall and per family)
- exact citation accuracy >= 0.95 (single-pass and staged, overall and per family)
- staged recall must not regress against the single-pass baseline
- zero provider errors, zero evidence-gate failures, zero failed trials

The validator pins these bars; if the gate ever changes them, the validator
must change in the same edit.

## What remains for a human reviewer

1. **Second-reviewer adjudication (human, required).** An independent reviewer
   completes the blinded packet per the review guide, without consulting the
   gold labels. Compare both label sets only after the independent pass.
2. **Reconcile in writing (human, required).** Resolve disagreements in
   writing; remove ambiguous cases rather than tuning to them. Then record
   `adjudication_status: independently_reviewed` with `reviewer_count: 2` on
   every held-out case.
3. **Full qualifying run (human operator, live providers, required).**
   Three model families x two repetitions on the held-out split:
   `omni evaluate-security --dataset tests/fixtures/redteam/cases.held_out.provisional.jsonl --split held_out --model cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast --model xkiro:google/gemini-2.5-flash:free --model apinex:free/deepseek-v4.1-flash --repetitions 2 --staged-redteam --quality-gate --output reports/redteam_qualifying_run.json`
4. **Re-run the gate test.** It passes only on a genuine PASS_BOUNDED_GATE
   artifact. No human may hand-write or hand-edit the artifact.
5. **Oracle hand-off.** Only after the gate passes: verify OmniCLI exposes its
   CLI/SDK cleanly without referencing Oracle, and mark the transition
   complete. Oracle stays blocked until then.
