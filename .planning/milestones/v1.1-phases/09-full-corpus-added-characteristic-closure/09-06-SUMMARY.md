# 09-06 Summary: Final Evidence Refresh

**Status:** COMPLETE
**Commit:** 9ae06c7 (fixtures + baseline), 9e4f723 (docs + plan-02 fix)

## What Changed

### Task 1: Regenerate 9-part fixtures and pin BASELINE_COUNTS at zero

Ran `uv run python run.py partN` for all 9 parts at HEAD db265b5 (Plans 01-05
applied). Regenerated all `tests/fixtures/phase7_algorithm_only/partN-debug-report.json`
fixtures. All 9 parts now have `missing_added_truth_indexes: []`.

Updated `BASELINE_COUNTS` in `tests/test_phase7_benchmark.py`:
- All parts: `max_missing_added=0`
- Part 2: conforming 19→18 (no added-truth regression)
- Part 5: conforming 13→14
- Part 8: conforming 7→8

### Task 2: Fix plan-02 contains mismatch

Updated `09-02-PLAN.md` `contains` field from the incorrect
`test_live_part8_packet_preserves_separate_runout_and_diameter_rows` to the
actual function name `test_live_part8_pipeline_keeps_added_rows_distinct`.

### Task 3: Refresh 06/07/09 verification docs

- `07-04-ALGORITHM-ONLY-BASELINE.md`: Updated Part 5 row to `[]`, added
  "Phase 9 full-corpus closure" section documenting how indexes 16+17 were
  closed by Plans 04 and 05.
- `07-VERIFICATION.md`: Extended Phase 9 Closure Note with Plans 04-05
  follow-up paragraph.
- `06-VERIFICATION.md`: Updated ADD-01 row to
  `VERIFIED (Phase 9 full-corpus closure, parts 1–9 at zero missing-added)`.
- `09-VERIFICATION.md`: Changed status `gaps_found`→`verified`, score→`3/3`,
  added `## Phase 9 Re-verification` section with Part 5 gaps closed table.

## Verification

- All 9 fixtures have `missing_added_truth_indexes == []`
- `BASELINE_COUNTS` covers parts 1–9 with `max_missing_added=0` for all
- `git diff --name-only -- assets/debug_report_part*.json` returns no output
- Full test suite: 487 passed, 2 xfailed, 0 failures
