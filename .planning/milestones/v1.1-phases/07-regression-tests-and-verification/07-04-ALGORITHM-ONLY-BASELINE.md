# 07-04 Algorithm-Only Baseline

**Provenance:** Baseline commit: db265b5 (post-Phase-9 full-corpus closure)
**Capture method:** `uv run python run.py partN` from repo root (HEAD = db265b5, after Phase 9 Plans 01-05)
**Captured:** 2026-04-18 (Phase 9 Plan 06 refresh)
**Prior baseline commit:** 8a611c9 (Phase 9 Plans 01-03), c5c19f0 (Phase 7 original capture)

## Why this baseline exists

The earlier verification report (07-03) anchored comparisons against
`assets/debug_report_part{1..9}.json`.  Those files are web-export snapshots
that mix review-queue metadata with algorithm output; they were exported from
database runs before Phase 6's added-characteristic detection improvements
were applied.  Using them as a regression baseline produced three false
positives (parity artifacts): Parts 1, 8, and 9 appeared to regress when
the comparison was actually against a stale, pre-Phase-6 snapshot.

This document records the algorithm-only baseline derived from running the
standalone pipeline.  Phase 7 (c5c19f0 → c3980fe) added only planning
documents and test scaffolding — no algorithm source files changed.  Phase 9
(Plans 01-05) closed all remaining missing-added misses: canonical token
contract, leading-zero normalization, detector-side owner signatures, GD&T
anchor symbols, surface-finish grouping, material-modifier spacing
normalization, zone-aware boilerplate filter, and dimensional-incompatibility
guard in the matching layer.

## Baseline counts (Phase 9 full-corpus closure — HEAD db265b5)

| Part | Conforming | Review-Needed | Missing-Added (indexes) |
|------|-----------|---------------|------------------------|
| part1 | 22 | 17 | 0 |
| part2 | 18 | 6  | 0 |
| part3 | 10 | 11 | 0 |
| part4 | 9  | 10 | 0 |
| part5 | 14 | 6  | 0 |
| part6 | 20 | 1  | 0 |
| part7 | 11 | 7  | 0 |
| part8 | 8  | 6  | 0 |
| part9 | 23 | 16 | 0 |

## Improvements over the Phase 7 (c5c19f0) baseline

| Part | Old conforming | New conforming | Old missing-added | New missing-added |
|------|---------------|----------------|-------------------|-------------------|
| part1 | 22 | 22 | 1 ([38]) | 0 |
| part2 | 18 | 19 | 1 ([22]) | 0 |
| part3 | 10 | 10 | 2 ([19, 20]) | 0 |
| part4 | 9  | 9  | 3 ([11, 14, 15]) | 0 |
| part5 | 12 | 14 | 3 ([16, 17, 18]) | 0 |
| part6 | 17 | 20 | 0 | 0 |
| part7 | 11 | 11 | 0 | 0 |
| part8 | 7  | 8  | 0 | 0 |
| part9 | 23 | 23 | 1 ([42]) | 0 |

## Phase 9 full-corpus closure

Part 5 indexes 16 (`800`) and 17 (`3X Ø18 ↧30`), previously deferred as
architectural gaps, are now closed:

- **Index 16 (`800`)**: Closed by Plan 04 — zone-aware boilerplate filter
  replaced the overly broad 120 pt proximity sweep with
  `span_is_excluded_for_annotation_search` plus a tight same-row companion
  check.  The drawing-body integer is no longer suppressed.
- **Index 17 (`3X Ø18 ↧30`)**: Closed by Plan 05 —
  `_candidate_is_dimensionally_compatible` guard in `assign_matches` prevents
  grouped candidates from claiming dimensionally unrelated anchors, and
  source-span pruning frees foreign sub-spans for added detection.  CLS-02
  primary-value threshold tightened from 15% to 10%.

`max_missing_added=0` now holds for every part.

## Historical Phase 6 assets

`assets/debug_report_part1.json` through `assets/debug_report_part9.json`
remain historical Phase 6 corpus evidence and are intentionally left
untouched.  They serve as the reference corpus for
`tests/test_phase6_asset_regression.py` and must not be modified by any
later phase.

## Fixture location

Algorithm-only fixtures are committed at:
`tests/fixtures/phase7_algorithm_only/part{1..9}-debug-report.json`

Each fixture contains: `baseline_commit`, `packet_run_id`, `part_name`,
`items`, `missing_added_truth_indexes`, `conforming_count`, `review_needed_count`.
