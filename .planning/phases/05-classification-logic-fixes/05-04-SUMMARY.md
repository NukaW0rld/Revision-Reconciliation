---
phase: 05-classification-logic-fixes
plan: "04"
subsystem: reconciliation
tags: [cls-02, removed-added, reconciliation, post-pass, geometry, tdd]
dependency_graph:
  requires: [05-01]
  provides: [reconcile_removed_added_pairs, CLS02_MAX_DISTANCE_PT, added_requirement_text, added_bbox, added_page]
  affects: [delta_preservation/reconcile/classify.py, delta_preservation/cli.py, tests/test_classify_bugfixes.py]
tech_stack:
  added: []
  patterns: [post-pass reconciliation, getattr legacy-object tolerance, closest-wins one-to-one matching, TDD RED-GREEN]
key_files:
  created: []
  modified:
    - delta_preservation/reconcile/classify.py
    - delta_preservation/cli.py
    - tests/test_classify_bugfixes.py
decisions:
  - Use getattr fallback for added_bbox/added_page/added_requirement_text to tolerate legacy _FakeInternalDeltaItem test doubles
  - Page equality gate (anchor.page == added_page) prevents cross-page false merges
  - req_bbox centroid with balloon_bbox fallback on removed side; added_bbox centroid on added side
  - Closest-wins one-to-one pairing: each removed item claims the nearest compatible added item
metrics:
  duration: 4 min
  completed_date: "2026-04-17"
  tasks_completed: 3
  files_modified: 3
---

# Phase 05 Plan 04: CLS-02 Removed+Added Reconciliation Post-Pass Summary

**One-liner:** Spatial post-pass merges close-proximity removed+added pairs into single changed rows using page-gated, distance-bounded, type-compatible one-to-one pairing with req_bbox/balloon_bbox centroid geometry.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Add mechanism-level tests for close-pair reconciliation | a4b05e3 | tests/test_classify_bugfixes.py |
| 2 (GREEN) | Extend internal added-item metadata and implement reconcile_removed_added_pairs | e4e9b5e | delta_preservation/reconcile/classify.py |
| 3 | Wire the reconciliation post-pass into cli.py | 90414ec | delta_preservation/cli.py |

## Decisions Made

- **getattr for legacy tolerance:** `reconcile_removed_added_pairs` uses `getattr(it, "added_bbox", None)` instead of direct attribute access so that pre-CLS-02 fake test doubles (`_FakeInternalDeltaItem`) do not raise `AttributeError`. This mirrors the existing `getattr(delta_internal, 'confidence_flags', [])` pattern established in Plan 05-01.
- **Removed-side geometry:** centroid of `anchor.req_bbox` if present, otherwise centroid of `anchor.balloon_bbox`. Explicit contract prevents pairing inconsistency for anchors whose text annotation was not found.
- **Added-side geometry:** centroid of `added_bbox` (union bbox for grouped GD&T items, union of pair spans for stacked limits), not just `added_span.bbox_pdf`.
- **One-to-one closest-wins:** each removed item claims the single nearest compatible added item; consumed items are excluded from subsequent pairings.
- **Type gate:** `are_requirement_types_incompatible` reuses the existing normalize infrastructure rather than introducing a new type comparison path.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] _FakeInternalDeltaItem AttributeError in reconcile post-pass**
- **Found during:** Task 3 full test suite run (post-wiring)
- **Issue:** `test_run_pipeline_omits_revA_evidence_for_added_characteristics` injects a `_FakeInternalDeltaItem` that pre-dates the new `added_bbox`/`added_page` fields. The filter list comprehension raised `AttributeError: '_FakeInternalDeltaItem' object has no attribute 'added_bbox'`
- **Fix:** Changed both the added_candidates filter and inner loop references to use `getattr(it, "added_bbox", None)` / `getattr(added, "added_page", None)` / `getattr(added, "added_requirement_text", None)`
- **Files modified:** `delta_preservation/reconcile/classify.py`
- **Commit:** 9b8a089

## TDD Gate Compliance

| Gate | Commit | Status |
|------|--------|--------|
| RED — `test(05-04)` commit | a4b05e3 | PASS — 1 test failed (ImportError) as expected |
| GREEN — `feat(05-04)` commit | e4e9b5e | PASS — 7 tests pass |

## Verification Results

```
pytest tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation -x
7 passed in 0.01s

python3 -c "from delta_preservation.reconcile.classify import CLS02_MAX_DISTANCE_PT; assert CLS02_MAX_DISTANCE_PT == 150.0; print('ok')"
ok

pytest tests/ (full suite)
342 passed, 2 xfailed, 2 warnings
```

## Known Stubs

None — all added metadata fields are populated in the three detection passes and consumed by the reconciliation function.

## Threat Flags

None — no new network endpoints, auth paths, or file access patterns introduced. Internal pipeline transformation only.

## Self-Check: PASSED
- `delta_preservation/reconcile/classify.py` modified: FOUND
- `delta_preservation/cli.py` modified: FOUND
- `tests/test_classify_bugfixes.py` modified: FOUND
- Commit a4b05e3 (RED): FOUND
- Commit e4e9b5e (GREEN): FOUND
- Commit 90414ec (wiring): FOUND
- Commit 9b8a089 (bug fix): FOUND
