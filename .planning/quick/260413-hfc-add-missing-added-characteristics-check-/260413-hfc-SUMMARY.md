---
phase: 260413-hfc
plan: 01
subsystem: shop/services/review
tags: [debug-queue, ground-truth, added-characteristics, tdd]
dependency_graph:
  requires: [delta_preservation.evaluation.loader, delta_preservation.evaluation.conformance, delta_preservation.evaluation.contracts]
  provides: [missing_added_truth_indexes in build_debug_queue_state, synthetic exception rows in build_run_debug_summary]
  affects: [debug queue sign-off gate, assemble_debug_report_payload payload]
tech_stack:
  added: []
  patterns: [load_ground_truth_packet mock in tests, GroundTruthContractError broad catch for silent fallback]
key_files:
  created: []
  modified:
    - shop/services/review.py
    - tests/test_debug_row_identity.py
decisions:
  - Broad Exception catch (not just GroundTruthContractError) in _load_missing_added_truth_indexes so any fixture-load failure is silent
  - missing_added_truth rows have no saved_verdict path so they permanently block debug_report_ready until resolved externally
  - assemble_debug_report_payload exposes missing_added_truth_indexes in top-level payload for downstream consumers
metrics:
  duration: ~8 min
  completed: 2026-04-13T17:38:42Z
  tasks_completed: 1
  files_modified: 2
---

# Phase 260413-hfc Plan 01: Add Missing Added Characteristics Check Summary

**One-liner:** Added `_load_missing_added_truth_indexes` helper and wired unclaimed truth "added" row detection into `build_debug_queue_state`, `build_run_debug_summary`, and `assemble_debug_report_payload` so debug sign-off is blocked when the algorithm misses added characteristics.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Failing tests for missing added truth detection | ddb3287 | tests/test_debug_row_identity.py |
| 1 (GREEN) | Implement missing added truth index detection | 33f27f4 | shop/services/review.py |

## What Was Built

### `_load_missing_added_truth_indexes(run, packet_data, packet_rows) -> list[int]`
New helper function in `shop/services/review.py` that:
1. Resolves the truth fixture key from `packet_data["inputs"]["truth_fixture_key"]` or `run.part_number`
2. Loads the ground truth packet via `load_ground_truth_packet`
3. Collects all "added" characteristic indexes from the truth packet
4. Scans packet row evaluations for `"added:N"` tokens in `matched_truth_char_no`
5. Returns sorted list of unclaimed added truth indexes
6. Returns `[]` silently on any fixture load error

### `build_debug_queue_state` updates
- Calls `_load_missing_added_truth_indexes` after building `packet_rows`
- Adds `missing_added_truth_indexes` key to return dict
- `debug_total` now = `len(exception_items) + len(missing_added_truth_indexes)`

### `build_run_debug_summary` updates
- After processing packet rows, appends one synthetic exception row per missing added truth index
- Each synthetic row: `row_state="missing_added_truth"`, `char_no=None`, mismatch code `"missing_added_characteristic"`
- These rows have no `saved_verdict` path, so they permanently count toward `unresolved_exception_count`, blocking `debug_report_ready`

### `assemble_debug_report_payload` updates
- Exposes `missing_added_truth_indexes` list in the top-level payload

## Tests Added

Three new tests in `tests/test_debug_row_identity.py`:
- `test_missing_added_truth_indexes_detected_when_packet_misses_a_truth_added_row`: 2 truth added rows, packet claims index 0 only → `missing_added_truth_indexes == [1]`, `debug_total == exception_count + 1`
- `test_no_missing_added_when_all_truth_added_rows_are_claimed`: 1 truth added row, packet claims index 0 → `missing_added_truth_indexes == []`
- `test_missing_added_silently_skipped_when_no_fixture`: fixture load raises exception → `missing_added_truth_indexes == []`, no exception propagated

All 38 tests in the verification suite pass.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed GroundTruthCharacteristic validation in test helper**
- **Found during:** RED phase test run
- **Issue:** `_make_truth_packet` helper passed `requirement_revB=None` and `snippet_center_revB=None` for "added" rows, but the model validator requires both for non-"removed" classifications
- **Fix:** Updated helper to supply per-classification minimum required fields (`requirement_revB` and `snippet_center_revB` for "added", `snippet_center_revA` for non-added)
- **Files modified:** tests/test_debug_row_identity.py
- **Commit:** ddb3287 (part of RED commit, fixed before GREEN)

None - implementation executed exactly as specified in the plan.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. The `_load_missing_added_truth_indexes` helper reads from an existing read-only assets directory path already covered by T-hfc-01 in the plan's threat model.

## Self-Check: PASSED

- `shop/services/review.py` exists and contains `missing_added_truth_indexes`
- `tests/test_debug_row_identity.py` exists and contains new tests
- Commits ddb3287 and 33f27f4 exist in git log
- 38/38 tests pass
