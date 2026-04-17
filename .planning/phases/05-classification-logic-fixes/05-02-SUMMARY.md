---
phase: "05"
plan: "02"
subsystem: classification
tags: [CLS-01, bleed-detection, count-added, confidence-flags]
dependency_graph:
  requires: ["05-01"]
  provides: ["anchor-aware-bleed-suppressor", "TestAdjacencyBleed"]
  affects: ["delta_preservation/reconcile/classify.py", "tests/test_classify_bugfixes.py"]
tech_stack:
  added: []
  patterns: ["module-level regex constants", "pure-predicate helper", "result_flags accumulator"]
key_files:
  created: []
  modified:
    - delta_preservation/reconcile/classify.py
    - tests/test_classify_bugfixes.py
decisions:
  - "Use _BLEED_SPLIT_RE with whitespace-bounded slash to exclude embedded fractions (1/4-20, H7/p6)"
  - "Anchor-bearing detection uses parse_requirement numeric tokens, not string contains, for robust float matching"
  - "result_flags accumulator initialized before decision tree so all branches have the variable in scope"
  - "Bleed detection runs only on count_added path (not count_changed) per plan spec"
metrics:
  duration: "3 min"
  completed: "2026-04-17"
  tasks_completed: 2
  files_modified: 2
---

# Phase 05 Plan 02: Anchor-Aware Adjacency Bleed Suppression (CLS-01) Summary

**One-liner:** Anchor-aware bleed suppressor using `_BLEED_SPLIT_RE` + `parse_requirement` numeric matching prevents slash-merged multi-balloon spans from triggering false `count_added` signals while preserving real count changes.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Add failing TestAdjacencyBleed tests | d019b34 | tests/test_classify_bugfixes.py |
| 2 (GREEN) | Implement anchor-aware bleed suppression | 6b718cb | delta_preservation/reconcile/classify.py |

## What Was Built

### `_BLEED_FLAG` and `_BLEED_SPLIT_RE` (classify.py lines 15–22)

Module-level constants:
- `_BLEED_FLAG = "Rev B text may contain adjacent balloon content"` — the exact advisory phrase stored in `DeltaItem.confidence_flags`.
- `_BLEED_SPLIT_RE = re.compile(r"\s+/\s+")` — whitespace-bounded slash splitter that excludes embedded fractions (`1/4-20 UNC`, `H7/p6`).

### `_looks_like_adjacency_bleed(span_text, anchor_text) -> bool`

Pure predicate that:
1. Splits `span_text` on `_BLEED_SPLIT_RE`; returns `False` if fewer than 2 chunks.
2. Parses `anchor_text` with `parse_requirement` to extract primary numeric and non-stopword keywords.
3. Marks a chunk "anchor-bearing" when it contains the anchor's primary float value (via `parse_requirement` numeric tokens) or ≥2 anchor keywords.
4. Marks a different chunk "foreign-count-bearing" when it has a count prefix (`2X`, `4 x`) and lacks the anchor's primary numeric.
5. Returns `True` only when both signals appear in distinct chunks.

### Integration in `classify_delta` count_added branch

- `result_flags: List[str] = []` accumulator initialized before the decision tree.
- Inside the `count_added` sub-branch: if `_looks_like_adjacency_bleed` returns `True`, status is demoted to `"unchanged"`, confidence floor is `max(0.55, ...)`, a human-readable suppression reason is appended, and `result_flags = [_BLEED_FLAG]`.
- Final `DeltaItem` return wired with `confidence_flags=result_flags`.

### `TestAdjacencyBleed` (test_classify_bugfixes.py)

Five test cases:
1. `test_part1_style_bleed_not_changed` — counterbore + slash-merged span, expects status ≠ `"changed"` and bleed flag present.
2. `test_part4_style_bleed_no_count_added` — thread callout + slash-merged span, expects bleed flag.
3. `test_benign_slash_ratio_no_bleed` — `"70 / 30"` for anchor `"30 ±0.5 mm"`, no bleed flag.
4. `test_legitimate_asymmetric_tolerance_no_bleed_stays_changed` — asymmetric tolerance `/` in value, no bleed flag, status remains `"changed"`.
5. `test_no_slash_regression_guard_stays_changed` — no slash, original count_added fires, status `"changed"`, no bleed flag.

## Verification Results

```
pytest tests/test_classify_bugfixes.py::TestAdjacencyBleed tests/test_classify_bugfixes.py::TestCountAdded -x
7 passed in 0.02s

pytest (full suite)
330 passed, 2 xfailed, 2 warnings in 32.57s
```

Plan inline verification:
```
_looks_like_adjacency_bleed('4 x Ø8 THRU ALL / ⌴ Ø13.5 ↧ 8.5', 'Counterbore Diameter (13.5 +/- 0.2 mm)') → True  ✓
_looks_like_adjacency_bleed('70 / 30', '30 ±0.5 mm') → False  ✓
```

## Deviations from Plan

None — plan executed exactly as written.

The plan specified `Optional[int]` usage in `_looks_like_adjacency_bleed` for index tracking; the implementation uses `Optional[int]` consistently for `anchor_bearing_idx` and `foreign_count_idx`.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. The helper is a pure in-memory predicate with no I/O.

## Self-Check: PASSED

- `delta_preservation/reconcile/classify.py` — FOUND (modified with bleed suppressor)
- `tests/test_classify_bugfixes.py` — FOUND (TestAdjacencyBleed added)
- Commit d019b34 — FOUND (test RED phase)
- Commit 6b718cb — FOUND (implementation GREEN phase)
- `_BLEED_FLAG` at line 15, `_looks_like_adjacency_bleed` at line 59 — FOUND
- Exactly one `count_changed or count_added` match — CONFIRMED
