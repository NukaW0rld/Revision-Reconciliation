# Delta Preservation

## What This Is

Delta Preservation is a brownfield aerospace drawing-reconciliation system with two validated surfaces: a standalone pipeline that compares Rev A and Rev B drawings against AS9102 Form 3 data, and a web workflow for review, sign-off, and debug inspection. `v1.0` shipped a deterministic maintainer-facing debug workflow: completed runs are evaluated against stable part-level ground truth, conforming rows auto-pass, exception rows drive the manual queue, and accepted alternates can be reused later without mutating canonical truth.

## Core Value

Improve reconciliation accuracy across many parts through a consistent, evidence-rich debug workflow that strengthens the algorithm without teaching it part-specific hacks.

## Current State

- `v1.0` Ground Truth Debug Workflow shipped on 2026-04-12.
- Completed runs now load immutable `ground_truth.json`, apply canonical requirement and snippet evaluation, narrow manual review to exceptions, and emit `debug_report.json` with canonical, unresolved, and history-backed states.
- Approved acceptable alternates are persisted separately from ground truth and can auto-conform later same-part reruns when the reviewed fingerprint matches exactly.
- The milestone audit was accepted as tech debt because Phases 1 and 2 are missing `VERIFICATION.md` artifacts even though the current test suite is green.

## Requirements

### Validated

- ✓ Automated reconciliation pipeline compares Rev A and Rev B drawings against FAIR/Form 3 characteristics and emits classifications with evidence artifacts — existing
- ✓ Web application supports run submission, background processing, review queues, and sign-off workflow for completed runs — existing
- ✓ Admin debug mode captures per-item verdicts and exports a debug report once a run has been fully reviewed — existing
- ✓ Each part run can automatically compare pipeline output against that part's `ground_truth.json` with deterministic requirement and snippet checks — v1.0
- ✓ Conforming characteristics auto-pass and nonconforming rows flow into an exception-only review queue plus `debug_report.json` — v1.0
- ✓ Accepted alternates persist in a separate history layer and can auto-conform later same-part reruns without changing canonical truth — v1.0

### Active

- [ ] Detect contradiction risk across accepted outcomes before a reviewer approves a new alternate
- [ ] Surface benchmark-wide trend summaries across parts and runs so algorithm changes can be judged against stable history
- [ ] Flag potential overfitting patterns when a locally accepted fix does not generalize across parts

### Out of Scope

- Automatic updates to `ground_truth.json` — the canonical ground truth must remain stable and manually curated
- Automatic algorithm self-tuning from review history — this milestone stops at better debugging infrastructure and evidence, not a learning loop
- End-user-facing debug tooling — this workflow remains for the maintainer, not production users
- Multi-user debug collaboration features — the current debug loop is still optimized for one maintainer

## Context

Shipped `v1.0` across 3 phases, 8 plans, and 17 tasks between 2026-04-10 and 2026-04-11. The repository now contains roughly 25k tracked lines of Python/TypeScript/Swift, with the delivered milestone concentrated in pipeline evaluation, review routing, admin templates, and accepted-alternate persistence. The priority now shifts from per-run throughput to cross-part consistency: the next milestone should detect contradiction risk, summarize benchmark trends, and help the maintainer judge whether accepted alternates improve general behavior rather than encode part-specific hacks.

## Constraints

- **Generalization**: Debug decisions must improve behavior across many parts — the system cannot be shaped around one drawing pair at the expense of others
- **Ground truth stability**: `ground_truth.json` files are canonical references and must not be auto-edited by the workflow
- **Human review boundary**: Nonconforming results still require manual inspection because more than one outcome may be acceptable in some cases
- **Developer-only scope**: This milestone is optimized for one maintainer's iterative run -> review -> rerun loop, not a multi-user debug product
- **Snippet tolerance**: Snippet evaluation should favor useful visible context over exact center-coordinate matching

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as a brownfield debugging-infrastructure milestone instead of a new product feature | The current need is faster and more consistent algorithm refinement on real drawing pairs | Shipped in v1.0 |
| Keep `ground_truth.json` immutable during runs | Stable ground truth prevents accidental drift and preserves a trusted baseline across iterations | Enforced in v1.0 |
| Use packet order plus `ReviewItem.id` as the debug row identity contract | Duplicate and null `char_no` rows made `char_no` sorting unsafe for queue/export alignment | Shipped in v1.0 |
| Auto-pass conforming characteristics and focus human review on nonconforming items | Manual review should be reserved for disagreement, ambiguity, and contradiction detection | Shipped in v1.0 |
| Store alternate acceptable outcomes in a separate exceptions/history layer | Some mismatches may still be acceptable, and that nuance should not overwrite the canonical truth | Shipped in v1.0 |
| Reuse history only for exact same-part reviewed fingerprints | Broader reuse would risk overfitting and false auto-conformance across parts | Shipped in v1.0 |
| Stop short of automatic learning/self-correction | The immediate goal remains better debugging evidence and consistency, not training the algorithm from review history | Confirmed for v1.0 |

## Next Milestone Goals

- Detect contradictions between accepted outcomes across parts and runs.
- Warn on potential overfitting before a reviewer accepts a new alternate outcome.
- Summarize benchmark performance trends across many runs so algorithm changes can be judged against stable ground truth.

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone**:
1. Full review of all sections
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-12 after v1.0 milestone*
