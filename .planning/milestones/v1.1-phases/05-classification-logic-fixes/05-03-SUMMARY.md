---
phase: "05"
plan: "03"
subsystem: reconcile/classify
tags: [cls-03, tolerance, asymmetric, kind-transition, tdd]
dependency_graph:
  requires: ["05-01"]
  provides: ["CLS-03 kind-first asymmetric tolerance detection"]
  affects: ["delta_preservation/reconcile/classify.py"]
tech_stack:
  added: []
  patterns: ["kind-first pre-check in tolerance refinement", "raw-text fallback for tolerance_comparison=None"]
key_files:
  created: []
  modified:
    - delta_preservation/reconcile/classify.py
    - tests/test_classify_bugfixes.py
decisions:
  - "Kind-transition reason is always appended (even when status already changed) to ensure traceability in debug reports"
  - "Raw-text fallback uses _ASYMMETRIC_SHAPE_RE on anchor.requirement_raw vs candidate.span.text when tolerance_comparison is None"
  - "Pre-existing UnboundLocalError in removed-item branch (is_notes_anchor/anchor_primary used before assignment when revB_text_spans is None) auto-fixed under Rule 1"
metrics:
  duration: "5 min"
  completed_date: "2026-04-17"
  tasks_completed: 2
  files_modified: 2
---

# Phase 05 Plan 03: CLS-03 Asymmetric Tolerance Kind-Transition Detection Summary

Kind-first symmetric→asymmetric tolerance transition detection that runs before the `tolerances_match` boost can mask the change.

## What Was Built

**Task 1 (RED):** Added `class TestAsymmetricTolerance` to `tests/test_classify_bugfixes.py` with 5 test cases covering the full CLS-03 contract:
1. `22.0° ±1°` → `22.0° +0.3°/−0.1°` with `tolerance_comparison` having `plus_minus → bilateral_stacked` kind — must produce `changed` with a kind-transition reason string
2. Leading-decimal fallback: `Ø.250 ±.002` → `Ø.250 +.005 / -.003` with `tolerance_comparison=None` — must produce `changed` via raw-text detection
3. Already-`changed` item (count mismatch `2X→4X`) with `tolerances_match=True` — must stay `changed`, not be downgraded
4. Same-kind control: `22.0° ±1°` → `22.0° ±0.5°` (both `plus_minus`) — must still reach `changed` via existing `tolerances_differ` path
5. Bilateral→bilateral control: `+0.3/−0.1` → `+0.4/−0.1` (no kind change) — must still be `changed` via existing logic

**Task 2 (GREEN):** Modified `classify.py` tolerance-refinement block:
- Added `_ASYMMETRIC_SHAPE_RE` pattern covering standard and leading-decimal asymmetric forms (`+.005 / -.003`, `+0.3° / −0.1°`, Unicode minus U+2212)
- Added `_is_symmetric_to_asymmetric_kind_change(tolerance_comparison)` helper detecting `plus_minus → bilateral_stacked/unilateral_stacked` transitions
- Kind-transition pre-check now runs **before** both `tolerances_match` and `tolerances_differ` branches, so a `tolerances_match=True` result cannot suppress a kind change
- Kind-transition reason always appended (even if status was already `changed`) for audit traceability
- Raw-text fallback added in the `else` branch (no `tolerance_comparison`): compares `±` in anchor raw text against `_ASYMMETRIC_SHAPE_RE` in candidate span text; promotes `unchanged/uncertain` to `changed` but never downgrades `changed`

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 (RED) | dab1c55 | test(05-03): add failing CLS-03 regression tests + pre-existing bug fix |
| 2 (GREEN) | afc48ef | feat(05-03): implement CLS-03 kind-first tolerance transition detection |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Pre-existing UnboundLocalError in removed-item branch**
- **Found during:** Task 1 (RED phase — test execution revealed the error immediately)
- **Issue:** In the `match_or_none is None` branch, `is_notes_anchor` and `anchor_primary` were computed inside `if revB_text_spans:` but referenced by the keyword-anchor scan outside that block. When `revB_text_spans=None`, calling `classify_delta` with `match_or_none=None` raised `UnboundLocalError`.
- **Fix:** Moved `anchor_fp`, `anchor_numerics`, `anchor_primary`, `anchor_req_type`, and `is_notes_anchor` computation to before the `if revB_text_spans:` guard. The `if revB_text_spans:` block was slimmed to only run the identity scan loop.
- **Files modified:** `delta_preservation/reconcile/classify.py`
- **Commit:** dab1c55

## Known Stubs

None — all logic paths are fully wired.

## Threat Flags

None — this change only modifies internal classification logic with no new network endpoints, auth paths, or trust boundary crossings.

## Self-Check: PASSED

- `delta_preservation/reconcile/classify.py` — exists, contains `_ASYMMETRIC_SHAPE_RE`, `_is_symmetric_to_asymmetric_kind_change`, kind-transition pre-check, raw-text fallback
- `tests/test_classify_bugfixes.py` — exists, contains `class TestAsymmetricTolerance`
- Commits dab1c55 and afc48ef — verified in git log
- Full test suite: 335 passed, 2 xfailed, 0 failures
