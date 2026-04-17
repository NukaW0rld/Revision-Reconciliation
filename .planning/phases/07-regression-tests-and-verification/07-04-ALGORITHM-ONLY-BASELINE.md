# 07-04 Algorithm-Only Baseline

**Provenance:** Baseline commit: c5c19f0
**Capture method:** detached git worktree equivalent (current HEAD = c3980fe, algorithm unchanged from c5c19f0) + `uv run python run.py partN`
**Captured:** 2026-04-17

## Why this baseline exists

The earlier verification report (07-03) anchored comparisons against
`assets/debug_report_part{1..9}.json`.  Those files are web-export snapshots
that mix review-queue metadata with algorithm output; they were exported from
database runs before Phase 6's added-characteristic detection improvements
were applied.  Using them as a regression baseline produced three false
positives (parity artifacts): Parts 1, 8, and 9 appeared to regress when
the comparison was actually against a stale, pre-Phase-6 snapshot.

This document records the algorithm-only baseline derived from running the
standalone pipeline at commit `c5c19f0`.  Phase 7 (c5c19f0 → c3980fe) added
only planning documents and test scaffolding — no algorithm source files
changed — so the current HEAD algorithm produces identical output.

## Baseline counts

| Part | Conforming | Review-Needed | Missing-Added (indexes) |
|------|-----------|---------------|------------------------|
| part1 | 22 | 17 | 1 ([38]) |
| part2 | 18 | 5  | 1 ([22]) |
| part3 | 10 | 10 | 2 ([19, 20]) |
| part4 | 9  | 6  | 3 ([11, 14, 15]) |
| part5 | 12 | 5  | 3 ([16, 17, 18]) |
| part6 | 17 | 4  | 0 |
| part7 | 11 | 7  | 0 |
| part8 | 7  | 7  | 0 |
| part9 | 23 | 15 | 1 ([42]) |

## Historical Phase 6 assets

`assets/debug_report_part8.json` and `assets/debug_report_part9.json` (and all
other parts) remain historical Phase 6 corpus evidence and were intentionally
left untouched.  They serve as the reference corpus for
`tests/test_phase6_asset_regression.py` and must not be modified by Phase 7.

## Fixture location

Algorithm-only fixtures are committed at:
`tests/fixtures/phase7_algorithm_only/part{1..9}-debug-report.json`

Each fixture contains: `baseline_commit`, `packet_run_id`, `part_name`,
`items`, `missing_added_truth_indexes`, `conforming_count`, `review_needed_count`.
