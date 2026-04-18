# 07-04 Algorithm-Only Baseline

**Provenance:** Baseline commit: 8a611c9 (post-Phase-9 closure)
**Capture method:** `uv run python run.py partN` from repo root (HEAD = 8a611c9, after Phase 9 Plans 01-03)
**Captured:** 2026-04-18 (Phase 9 Plan 03 refresh)
**Prior baseline commit:** c5c19f0 (Phase 7 original capture)

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
(Plans 01-03) then closed the remaining missing-added misses by fixing the
canonical token contract, leading-zero normalization, detector-side owner
signatures, GD&T anchor symbols, surface-finish grouping, and
material-modifier spacing normalization.

## Baseline counts (Phase 9 post-closure — HEAD 8a611c9)

| Part | Conforming | Review-Needed | Missing-Added (indexes) |
|------|-----------|---------------|------------------------|
| part1 | 22 | 17 | 0 |
| part2 | 19 | 4  | 0 |
| part3 | 10 | 11 | 0 |
| part4 | 9  | 10 | 0 |
| part5 | 13 | 5  | 2 ([16, 17]) |
| part6 | 20 | 1  | 0 |
| part7 | 11 | 7  | 0 |
| part8 | 7  | 7  | 0 |
| part9 | 23 | 16 | 0 |

### Part 5 deferred items

Part 5 indexes 16 (`800`) and 17 (`3X Ø18 ↧30`) remain unclaimed:

- **Index 16 (`800`)**: Filtered by the near-boilerplate proximity check —
  `UNLESS OTHERWISE SPECIFIED` text is ~54 pt away, within the 120 pt
  threshold.  Reducing the threshold risks false positives; deferred.
- **Index 17 (`3X Ø18 ↧30`)**: All three source spans are matched to char 1
  (`Ø35.2 / Ø34.8`) by the matching layer before Pass 0 can detect them.
  Requires a matching-layer architectural fix outside Phase 9 scope.

Both were explicitly deferred in Phase 9 Plan 02 (see 09-02-SUMMARY.md).

## Improvements over the Phase 7 (c5c19f0) baseline

| Part | Old conforming | New conforming | Old missing-added | New missing-added |
|------|---------------|----------------|-------------------|-------------------|
| part1 | 22 | 22 | 1 ([38]) | 0 |
| part2 | 18 | 19 | 1 ([22]) | 0 |
| part3 | 10 | 10 | 2 ([19, 20]) | 0 |
| part4 | 9  | 9  | 3 ([11, 14, 15]) | 0 |
| part5 | 12 | 13 | 3 ([16, 17, 18]) | 2 ([16, 17]) |
| part6 | 17 | 20 | 0 | 0 |
| part7 | 11 | 11 | 0 | 0 |
| part8 | 7  | 7  | 0 | 0 |
| part9 | 23 | 23 | 1 ([42]) | 0 |

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
