---
phase: 06-added-characteristic-detection-and-snippet-accuracy
plan: "03"
subsystem: evaluation/conformance
tags: [tdd, added-truth-selection, tie-break, bbox-evidence, duplicate-disambiguation]
dependency_graph:
  requires: ["06-02"]
  provides: ["deterministic-duplicate-added-truth-claiming"]
  affects: ["delta_preservation/evaluation/conformance.py"]
tech_stack:
  added: ["ADDED_TRUTH_TIEBREAK_MAX_DISTANCE_PT constant", "geometry helpers: _coerce_packet_bbox, _bbox_center, _point_inside_bbox, _distance"]
  patterns: ["TDD RED/GREEN", "two-stage tie-break: bbox containment then nearest-center", "conservative ambiguity fallback"]
key_files:
  created:
    - tests/test_added_truth_selection.py
  modified:
    - delta_preservation/evaluation/conformance.py
    - tests/test_debug_row_identity.py
decisions:
  - "Use 100.0 pt as ADDED_TRUTH_TIEBREAK_MAX_DISTANCE_PT — 1 inch is a comfortable radius that prevents cross-cluster confusion on typical aerospace drawings"
  - "Stage 1 (bbox containment) takes priority over Stage 2 (nearest-center) to minimize risk of distance-based misassignment"
  - "Float equality comparison for min_dist used in nearest-center — two centers at identical distances preserve ambiguity"
  - "All geometry helpers are self-contained in conformance.py; no private imports from snippet_rules.py"
metrics:
  duration_minutes: 4
  completed_date: "2026-04-17"
  tasks_completed: 2
  files_changed: 3
---

# Phase 06 Plan 03: Duplicate Added-Truth Tie-Break Summary

Deterministic duplicate added-truth claiming using packet Rev B bbox evidence in `select_truth_row_for_item`.

## One-Liner

Two-stage geometry tie-break (bbox containment → nearest-center within 100 pt) lets duplicate Part 9 added rows claim distinct canonical truth indexes instead of collapsing into `truth_ambiguity`.

## What Was Built

### Task 1 — Failing tests (RED)

- Created `tests/test_added_truth_selection.py` with `class TestDuplicateAddedTruthSelection` (17 test methods) covering:
  1. Unique exact-text fast path (unchanged behavior)
  2. Bbox containment disambiguation for all three Part 9 duplicate pairs (`Ø.250 ±.008`, `⌖ ∅.015 D H`, `↧.50 ±.05`) — group A and group B variants
  3. Nearest-center fallback when no center falls inside bbox
  4. Ambiguity preservation when bbox covers both centers or centers are equidistant
  5. Invalid/missing revB evidence stays ambiguous (None, empty list, 3-element list)
  6. Reserved index exclusion

- Extended `tests/test_debug_row_identity.py` with `test_duplicate_added_truth_rows_claim_distinct_indexes_from_revb_evidence` — queue-facing regression that processes all six Part 9 duplicate added packet rows sequentially, verifying each claims a distinct truth index using checked-in ground-truth centers as coordinate source.

### Task 2 — Implementation (GREEN)

Extended `delta_preservation/evaluation/conformance.py`:

- `ADDED_TRUTH_TIEBREAK_MAX_DISTANCE_PT = 100.0` — explicit distance threshold constant
- `_coerce_packet_bbox()` — validates and coerces raw Evidence bbox to a 4-tuple or None
- `_bbox_center()` — computes center of a validated bbox
- `_point_inside_bbox()` — inclusive containment check
- `_distance()` — Euclidean distance between two points
- `select_truth_row_for_item()` extended with two-stage tie-break after duplicate exact-text detection:
  - Stage 1: if exactly one truth `snippet_center_revB` lies inside the packet bbox, select it
  - Stage 2: if exactly one truth center is the unique nearest within `ADDED_TRUTH_TIEBREAK_MAX_DISTANCE_PT`, select it
  - Conservative fallback: preserve the original ambiguity message when evidence is missing or non-unique

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 (RED) | fdc55bd | test(06-03): add failing tests for duplicate added-truth tie-break |
| 2 (GREEN) | 26f88a7 | feat(06-03): add deterministic Rev B evidence tie-break for duplicate added-truth selection |

## Verification

```
uv run pytest tests/test_added_truth_selection.py tests/test_debug_row_identity.py -x
# 20 passed

uv run pytest -x
# 387 passed, 2 xfailed (up from 351 in Phase 05)
```

Pattern checks:
- `ADDED_TRUTH_TIEBREAK_MAX_DISTANCE_PT` present in conformance.py ✓
- `snippet_center_revB` referenced in tie-break logic ✓
- `inside the packet bbox` comment present ✓
- Conservative ambiguity path `multiple canonical added truth rows share the same normalized requirement text` retained ✓

## Deviations from Plan

None — plan executed exactly as written.

## Threat Model Compliance

| Threat ID | Disposition | Implemented |
|-----------|-------------|-------------|
| T-06-07 (I) | mitigate | Only selects when exactly one candidate qualifies via bbox containment or nearest-center uniqueness |
| T-06-08 (T) | mitigate | Conservative ambiguity fallback preserved whenever evidence is missing, invalid, or non-unique |

## Known Stubs

None. All selection logic is fully wired; no placeholder values or TODO paths.

## Self-Check: PASSED

- `delta_preservation/evaluation/conformance.py` — modified ✓
- `tests/test_added_truth_selection.py` — created ✓
- `tests/test_debug_row_identity.py` — modified ✓
- Commits fdc55bd and 26f88a7 exist in git log ✓
- 387 tests pass, 0 failures ✓
