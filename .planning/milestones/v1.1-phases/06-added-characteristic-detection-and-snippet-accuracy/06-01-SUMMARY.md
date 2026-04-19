---
phase: 06-added-characteristic-detection-and-snippet-accuracy
plan: "01"
subsystem: reconcile
tags: [exclusion, refactor, shared-contract, tdd]
dependency_graph:
  requires: []
  provides: [shared-exclusion-contract]
  affects: [anchors, match, classify]
tech_stack:
  added: [delta_preservation/reconcile/exclusion.py]
  patterns: [shared-helper module, thin-wrapper delegation]
key_files:
  created:
    - delta_preservation/reconcile/exclusion.py
    - tests/test_phase6_exclusion.py
  modified:
    - delta_preservation/reconcile/anchors.py
    - delta_preservation/reconcile/match.py
    - delta_preservation/reconcile/classify.py
decisions:
  - "Thin wrappers kept in match.py (_is_boilerplate_candidate_text, _span_is_excluded_for_matching) so existing internal call sites work unchanged"
  - "estimate_page_dimensions uses min_width=612 / min_height=792 floors matching pre-existing behavior in match.py"
  - "Tests use portrait page dimensions (612x792) to be consistent with estimate_page_dimensions min_height floor"
  - "detect_added_characteristics retains local is_in_exclusion_zone() as a thin wrapper delegating to span_is_excluded_for_annotation_search"
metrics:
  duration: "6 min"
  completed: "2026-04-17"
  tasks: 2
  files: 5
---

# Phase 06 Plan 01: Shared Exclusion Contract — Summary

Centralized title-block/boilerplate exclusion into a single `exclusion.py` module and migrated all four annotation-search callers (anchor lookup, candidate generation, rescue scans, added detection) to the shared contract instead of three drifting local heuristics.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add exclusion-focused unit tests (TDD RED) | f9a5f45 | tests/test_phase6_exclusion.py |
| 2 | Extract shared exclusion module and migrate all callers (GREEN) | 443a29d | exclusion.py, anchors.py, match.py, classify.py, test_phase6_exclusion.py |

## What Was Built

**`delta_preservation/reconcile/exclusion.py`** — new shared utility exposing:
- `estimate_page_dimensions(spans, *, min_width=612.0, min_height=792.0) -> (float, float)`
- `is_boilerplate_candidate_text(text: str) -> bool`
- `span_is_excluded_for_annotation_search(span, *, page_width, page_height) -> bool`

The module preserves all existing normalized-text checks from `match.py`, the minimum page-size floors, and the span-centre-based geometric zone tests (not raw `x0`/`y0` corners).

**Caller migrations:**

- `anchors.py` — replaced raw `y0 > 85% / x0 > 80%` heuristic with `span_is_excluded_for_annotation_search()`; uses `estimate_page_dimensions()` for page-size estimation.
- `match.py` — local `_is_boilerplate_candidate_text` and `_span_is_excluded_for_matching` reduced to thin wrappers delegating to the shared module; `generate_candidates()` uses `estimate_page_dimensions()`.
- `classify.py` — keyword rescue scan now calls `estimate_page_dimensions()` + `span_is_excluded_for_annotation_search()` instead of inline coordinate checks + manual keyword list; `detect_added_characteristics()` `is_in_exclusion_zone()` reduced to a thin wrapper; unmatched-span pre-filter uses shared helper directly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed NameError in classify.py keyword rescue scan**
- **Found during:** Task 2 GREEN phase
- **Issue:** When removing the explicit `_st_upper = span.text.strip().upper()` assignment (part of the old boilerplate keyword check block), the subsequent `_st_norm = _st_upper.replace(...)` line was left referencing an undefined name.
- **Fix:** Re-added `_st_upper = span.text.strip().upper()` immediately before `_st_norm` to restore the variable that the substring check needs.
- **Files modified:** `delta_preservation/reconcile/classify.py`
- **Commit:** 443a29d

**2. [Rule 1 - Bug] Test fixture used landscape page proportions inconsistent with height floor**
- **Found during:** Task 2 GREEN verification
- **Issue:** Tests initially used `pw=792, ph=612` (landscape). `estimate_page_dimensions` applies a `min_height=792` floor, so a span at `cy = 612 * 0.92 = 563` falls below `0.85 * 792 = 673`, evading the title-block zone check.
- **Fix:** Updated all rescue-scan and added-detection test scenarios to use portrait `pw=612, ph=792` so the 792-pt min_height floor matches the actual page height and exclusion-zone proportions are correct.
- **Files modified:** `tests/test_phase6_exclusion.py`
- **Commit:** 443a29d

## Verification

```
uv run pytest tests/test_phase6_exclusion.py tests/test_alignment_multishift.py::test_generate_candidates_excludes_tolerance_block_boilerplate -x
# 11 passed

uv run pytest -x
# 361 passed, 2 xfailed
```

## Known Stubs

None — all wiring is live and exercised by the test suite.

## Threat Flags

None — no new network endpoints, auth paths, or trust-boundary surfaces introduced. This plan is a pure internal refactor of reconcile helpers.

## Self-Check: PASSED

- `delta_preservation/reconcile/exclusion.py` — exists ✓
- `tests/test_phase6_exclusion.py` — exists ✓
- Commit f9a5f45 (TDD RED) — exists ✓
- Commit 443a29d (feat GREEN) — exists ✓
- 361 tests pass, 0 regressions ✓
