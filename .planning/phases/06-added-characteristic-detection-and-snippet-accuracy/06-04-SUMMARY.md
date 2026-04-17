---
phase: 06-added-characteristic-detection-and-snippet-accuracy
plan: "04"
subsystem: tests
tags: [regression, testing, assets, part8, part9]
dependency_graph:
  requires: [06-01, 06-02, 06-03]
  provides: [asset-backed-regression-harness]
  affects: []
tech_stack:
  added: []
  patterns: [asset-backed regression harness, read-only corpus fixtures, TDD regression pinning]
key_files:
  created:
    - tests/test_phase6_asset_regression.py
  modified: []
decisions:
  - "Tests use exact exemplar strings and canonical centers from checked-in assets, not vague count assertions (T-06-09)"
  - "Harness stays read-only with respect to assets/ and ground_truth.json — never regenerates or rewrites them (T-06-10)"
  - "TestPhase6AssetInvariants pins snapshot file existence and exact added-row text/count assertions"
  - "TestPhase6Part8Exemplars pins .045 A / ⚪ .005 / ⌰ .002 A as the three Part 8 corpus exemplars"
  - "TestPhase6Part9DuplicateAddedRows loads live GroundTruthCharacteristic rows from the JSON fixture to avoid test drift"
metrics:
  duration: "3 min"
  completed: "2026-04-17"
  tasks_completed: 1
  files_modified: 1
---

# Phase 06 Plan 04: Asset-Backed Phase 6 Regression Harness Summary

Read-only asset regression harness pinned to exact Part 8 / Part 9 exemplar strings and canonical centers from the checked-in debug corpus.

## What Was Built

Created `tests/test_phase6_asset_regression.py` with three test classes and 28 test methods:

### TestPhase6AssetInvariants (12 tests)
- Asserts all four snapshot files exist (`debug_report_part8.json`, `debug_report_part9.json`, `assets/part8/ground_truth.json`, `assets/part9/ground_truth.json`)
- Pins Part 8 canonical added rows by exact text: `Ø10.000±.001`, `⌰ .015 B`, `⌰ .002 A`
- Pins Part 9 duplicate groups by exact multiplicity: `Ø.250 ±.008` (×2), `⌖ ∅.015 D H` (×2), `↧.50 ±.05` (×2)
- Pins `missing_added_truth_indexes=[10]` for Part 8 and `[35..42]` for Part 9

### TestPhase6Part8Exemplars (6 tests)
- Pins `.045 A` as a false-positive added fragment present in the Part 8 debug report
- Pins `⚪ .005` as a false-positive circularity fragment in the Part 8 debug report
- Pins `⌰ .002 A` as the canonical missing added row (algorithm_error, row_state=algorithm_error)
- Asserts that `detect_added_characteristics` can produce a row containing `⌰` and `.002` (full callout for `⌰ .002 A`)
- Asserts fragment `.045 A` does not survive when `◎ ∅.045 A` already owns the annotation

### TestPhase6Part9DuplicateAddedRows (10 tests)
- Loads exact duplicate truth centers from `assets/part9/ground_truth.json` at runtime
- Asserts group-A and group-B centers match the research report values
- Uses `select_truth_row_for_item` with tight bboxes around each canonical center to confirm distinct-token claiming
- Pins the truncated `⌖∅` exemplar from the Part 9 debug report as distinct from `⌖ ∅.015 D H`
- Asserts `evaluate_packet_against_truth` assigns distinct added-pool tokens to both `⌖ ∅.015 D H` items

## Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create asset regression harness | 03f7be6 | tests/test_phase6_asset_regression.py |

## Verification

All acceptance criteria met:
- `tests/test_phase6_asset_regression.py` exists and contains all three required test classes
- `rg -n "\.045 A|⚪ \.005|⌰ \.002 A|⌖∅|⌖ ∅\.015 D H|Ø\.250 ±\.008|↧\.50 ±\.05" tests/test_phase6_asset_regression.py` returns all pinned exemplar strings
- `uv run pytest tests/test_phase6_asset_regression.py -x` exits 0 (28 passed)
- Full suite: 415 passed, 2 xfailed

## Deviations from Plan

None — plan executed exactly as written.

The TDD RED phase: one test initially failed due to `GroundTruthPacket` requiring `part_name` and `general_notes` fields (Pydantic validation). Fixed inline during RED by loading these from the fixture JSON rather than constructing an anonymous packet. This is a correctness adjustment (Rule 1), not a plan deviation.

## Known Stubs

None — all assertions are keyed to exact corpus strings and live fixture data.

## Threat Flags

None — the harness is read-only and introduces no new network endpoints, auth paths, or file write operations.

## Self-Check: PASSED
