# Delta Preservation

## What This Is

Delta Preservation is a brownfield aerospace drawing-reconciliation system with two validated surfaces: a standalone pipeline that compares Rev A and Rev B drawings against AS9102 Form 3 data, and a web workflow for review, sign-off, and debug inspection. `v1.0` shipped a deterministic maintainer-facing debug workflow with ground-truth evaluation and accepted-alternate history. `v1.1` refined the core algorithm across 9 debug-corpus parts — GD&T parsing, classification logic, added-characteristic detection, and title-block exclusion — with regression tests locking the baseline at 500 tests green, and sign-off gating that blocks export on unresolved debug exceptions.

## Core Value

Improve reconciliation accuracy across many parts through a consistent, evidence-rich debug workflow that strengthens the algorithm without teaching it part-specific hacks.

## Current State

- `v1.1` Cross-part Characteristic Matching Refinement shipped on 2026-04-19.
- GD&T parser handles compact tokens, word-form names, and composite multi-compartment FCFs.
- Classification logic suppresses adjacency bleed false positives, reconciles removed+added pairs as changed, and detects asymmetric tolerance changes — all surfaced via `confidence_flags`.
- Added-characteristic detection claims 33/35 corpus-wide truth rows (up from 7/35 pre-v1.1); 2 documented architectural deferrals remain (Part 5 indexes 16+17).
- Title-block exclusion contract (`span_is_excluded_for_annotation_search`) is shared across anchors, matching, rescue scans, and added detection.
- Sign-off is gated on unresolved debug exceptions; exported artifacts preserve the same advisory state the maintainer saw during review.
- Full `/runs/new` → packet → review → debug → sign-off → export workflow proven by automated E2E tests using real corpus assets.
- Test suite: 500 passed, 2 xfailed. Python LOC: ~34k.

## Requirements

### Validated

- ✓ Automated reconciliation pipeline compares Rev A and Rev B drawings against FAIR/Form 3 characteristics and emits classifications with evidence artifacts — existing
- ✓ Web application supports run submission, background processing, review queues, and sign-off workflow for completed runs — existing
- ✓ Admin debug mode captures per-item verdicts and exports a debug report once a run has been fully reviewed — existing
- ✓ Each part run can automatically compare pipeline output against that part's `ground_truth.json` with deterministic requirement and snippet checks — v1.0
- ✓ Conforming characteristics auto-pass and nonconforming rows flow into an exception-only review queue plus `debug_report.json` — v1.0
- ✓ Accepted alternates persist in a separate history layer and can auto-conform later same-part reruns without changing canonical truth — v1.0
- ✓ GD&T compact token parser correctly handles concatenated frames, word-form names, and composite FCFs — v1.1
- ✓ Adjacency bleed detection suppresses false count_added classification signal and surfaces a confidence flag — v1.1
- ✓ Removed+unmatched-added pairs at close spatial proximity are resolved as a single changed characteristic — v1.1
- ✓ Asymmetric tolerance change (±T → +a/−b) is correctly detected as changed — v1.1
- ✓ Added characteristics are generated for ground-truth-added rows across the corpus (33/35 claimed, 2 architectural deferrals documented) — v1.1
- ✓ Title block region is reliably excluded from characteristic search windows — v1.1
- ✓ Per-cluster regression tests and cross-part benchmark guard aggregate accuracy — v1.1
- ✓ Sign-off gating on unresolved debug exceptions with advisory flag surfacing — v1.1
- ✓ Live web workflow proven end-to-end with automated integration tests — v1.1

### Active

- [ ] Cross-part contradiction detection between accepted alternates
- [ ] Benchmark trend summaries across runs
- [ ] Overfitting warning before reviewer accepts a new alternate
- [ ] Part 5 thread/countersink matching-layer support (indexes 16+17)
- [ ] Part 9 explained-by-match suppressor false-absorption fix (truth_index 42)

### Out of Scope

- Automatic updates to `ground_truth.json` — the canonical ground truth must remain stable and manually curated
- Automatic algorithm self-tuning from review history — the system provides debugging evidence, not a learning loop
- End-user-facing debug tooling — this workflow remains for the maintainer, not production users
- Multi-user debug collaboration features — the current debug loop is optimized for one maintainer
- Composite GD&T with positional-zone modifiers — deferred to future milestone

## Context

Shipped `v1.1` across 9 phases, 32 plans, and 155 commits between 2026-04-12 and 2026-04-19 (7 days). The repository now contains ~34k tracked lines of Python with 500 passing tests. The v1.1 milestone concentrated on algorithm accuracy (GD&T parsing, classification logic, added-characteristic detection) and process maturity (regression benchmarks, sign-off gating, E2E web-workflow automation). The priority now shifts to cross-part consistency tooling: contradiction detection, benchmark trend analysis, and overfitting guards.

## Constraints

- **Generalization**: Debug decisions must improve behavior across many parts — the system cannot be shaped around one drawing pair at the expense of others
- **Ground truth stability**: `ground_truth.json` files are canonical references and must not be auto-edited by the workflow
- **Human review boundary**: Nonconforming results still require manual inspection because more than one outcome may be acceptable in some cases
- **Developer-only scope**: This milestone is optimized for one maintainer's iterative run → review → rerun loop, not a multi-user debug product
- **Snippet tolerance**: Snippet evaluation should favor useful visible context over exact center-coordinate matching

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as a brownfield debugging-infrastructure milestone instead of a new product feature | The current need is faster and more consistent algorithm refinement on real drawing pairs | ✓ Good — v1.0 |
| Keep `ground_truth.json` immutable during runs | Stable ground truth prevents accidental drift and preserves a trusted baseline across iterations | ✓ Good — v1.0 |
| Use packet order plus `ReviewItem.id` as the debug row identity contract | Duplicate and null `char_no` rows made `char_no` sorting unsafe for queue/export alignment | ✓ Good — v1.0 |
| Auto-pass conforming characteristics and focus human review on nonconforming items | Manual review should be reserved for disagreement, ambiguity, and contradiction detection | ✓ Good — v1.0 |
| Store alternate acceptable outcomes in a separate exceptions/history layer | Some mismatches may still be acceptable, and that nuance should not overwrite the canonical truth | ✓ Good — v1.0 |
| Reuse history only for exact same-part reviewed fingerprints | Broader reuse would risk overfitting and false auto-conformance across parts | ✓ Good — v1.0 |
| Use `confidence_flags` as packet-native advisory data | Flags travel with the packet and are rendered directly, not re-derived from reasons text | ✓ Good — v1.1 |
| Replace ±200 pt same-row sweep with fixed-point companion walk | Broad sweep caused false absorption of independent annotations | ✓ Good — v1.1 |
| Key `advisory_flags_by_item_id` by `ReviewItem.id` not `char_no` | Tolerates duplicate and None characteristic numbers in the review queue | ✓ Good — v1.1 |
| Gate sign-off on strict `review_needed` exception contract | Avoids false positives from items without evaluation | ✓ Good — v1.1 |
| Signed debug snapshot at sign-off time becomes single source of truth for export | Ensures exported artifacts match what the maintainer reviewed | ✓ Good — v1.1 |
| Defer Part 5 thread/countersink matching to future milestone | Requires matching-layer architectural changes beyond v1.1 scope | ⚠️ Revisit — v1.1 |
| Accept 2/35 corpus-wide added-truth deferrals as known limitation | Part 5 indexes 16+17 need architectural work; documented, not hacked | ✓ Good — v1.1 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone**:
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-19 after v1.1 milestone*
