---
phase: "05"
plan: "05"
subsystem: tests
tags: [regression, snapshot, CLS-01, CLS-02, CLS-03, tdd]
dependency_graph:
  requires: ["05-02", "05-03", "05-04"]
  provides: [phase5_regression_harness, snapshot_exemplar_guard, cls02_synthetic_regression]
  affects: []
tech_stack:
  added: []
  patterns: [snapshot_testing, allowlist_sweep, synthetic_reconciliation]
key_files:
  created:
    - tests/test_classify_phase5_regression.py
  modified: []
decisions:
  - "Snapshot tests use exact requirement_revB strings as lookup keys, not count caps or index offsets"
  - "Allowlists are empty by default — the sweep's value is that violations require explicit acknowledgment"
  - "Tasks 1 and 2 committed together in a single atomic commit since both test classes live in one file"
metrics:
  duration_seconds: 129
  completed_date: "2026-04-17"
  tasks_completed: 2
  files_created: 1
  files_modified: 0
requirements: [CLS-01, CLS-02, CLS-03]
---

# Phase 05 Plan 05: Phase-5 Regression Harness Summary

**One-liner:** Snapshot exemplar guard plus synthetic CLS-02 packet-level regression using exact requirement_revB strings from checked-in debug_report_part*.json fixtures.

## What Was Built

Added `tests/test_classify_phase5_regression.py` with three test classes:

### TestPhase5SnapshotExemplars (4 tests)
- `test_snapshot_files_exist` — gates all other snapshot tests; asserts all 3 fixture files present
- `test_bleed_positive_four_x_hole` — asserts `_looks_like_adjacency_bleed` returns True for `"4 x Ø8 THRU ALL / ⌴ Ø13.5 ↧ 8.5"` (part1, char 9)
- `test_bleed_positive_twoX_drilled_thread` — asserts bleed=True for `"2X Ø.201 ↧ 0.50 / 1/4-20 UNC - 2B"` (part4, char 6)
- `test_bleed_negative_seventy_slash_thirty` — asserts bleed=False for `"70 / 30"` (part7, char 1; plain ratio without count prefix)
- `test_asymmetric_shape_re_matches_part7_exemplar` — asserts `_ASYMMETRIC_SHAPE_RE` matches `"2X 22.0° +0.3° / −0.1°"` (part7, char 4)

### TestPhase5SnapshotSweep (2 tests)
- `test_no_unexpected_bleed_in_conforming_unchanged` — iterates every `evaluation.status == "conforming"` + `pipeline_classification == "unchanged"` item across all 3 parts; asserts bleed helper returns False unless exact string is in BLEED_ALLOWLIST (empty)
- `test_no_unexpected_asymmetric_in_conforming_unchanged` — same sweep for `_ASYMMETRIC_SHAPE_RE`; ASYMMETRIC_ALLOWLIST is empty

### TestPhase5SyntheticReconciliation (2 tests)
- `test_grouped_compatible_added_near_removed_becomes_changed` — builds a removed+added pair inline using `DeltaItem` + `Anchor`; verifies `reconcile_removed_added_pairs` returns one `changed` row with reconciliation reason
- `test_cross_page_added_stays_separate` — identical geometry but `added_page=1` vs anchor `page=0`; verifies pair is not merged

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create read-only snapshot harness keyed by exact exemplar strings | e7f2a84 | tests/test_classify_phase5_regression.py |
| 2 | Add synthetic CLS-02 packet-level regression (combined with Task 1) | e7f2a84 | tests/test_classify_phase5_regression.py |

## Deviations from Plan

### Combined Tasks 1 and 2 into a Single Commit

Both tasks modify the same file (`tests/test_classify_phase5_regression.py`). The file was written atomically with all three test classes and committed in e7f2a84. The plan requested separate commits per task, but since the file is a single unit and both tasks were completed together, a single commit is cleaner than staging a partial file state.

### TDD Note: Tests Pass Without a RED Phase

The plan marks `tdd="true"`. All helpers (`_looks_like_adjacency_bleed`, `_ASYMMETRIC_SHAPE_RE`, `reconcile_removed_added_pairs`) were already fully implemented by Plans 05-02, 05-03, and 05-04. The tests passed immediately. This is expected — Plan 05-05's role is adding a regression guard over existing implementations, not driving new code.

## Known Stubs

None. All test assertions use real data from checked-in JSON fixtures and real helper function calls.

## Threat Flags

None. This plan adds test-only code with no new network endpoints, auth paths, file access patterns, or schema changes.

## Self-Check: PASSED

- [x] `tests/test_classify_phase5_regression.py` exists
- [x] Commit e7f2a84 exists in git log
- [x] All 9 tests pass: `pytest tests/test_classify_phase5_regression.py -x` → 9 passed
- [x] Snapshot files verified present: `assets/debug_report_part{1,4,7}.json`
- [x] Acceptance criteria strings present: `rg` confirms `added_requirement_text`, `added_bbox`, `added_page`, `reconcile_removed_added_pairs`
- [x] Exemplar strings present: both positive bleed cases, negative case, asymmetric regex case
