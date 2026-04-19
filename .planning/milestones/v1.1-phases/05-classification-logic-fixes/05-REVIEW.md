---
phase: 05-classification-logic-fixes
reviewed: 2026-04-16T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - delta_preservation/cli.py
  - delta_preservation/reconcile/classify.py
  - delta_preservation/types.py
  - tests/test_classify_bugfixes.py
  - tests/test_classify_phase5_regression.py
  - tests/test_output_formatting.py
  - tests/test_pipeline_semantic_packet.py
findings:
  critical: 0
  warning: 4
  info: 5
  total: 9
status: issues_found
---

# Phase 05: Code Review Report

**Reviewed:** 2026-04-16
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

The phase-5 classification-logic fixes are well-structured and the overall design is sound. The core algorithms (`classify_delta`, `detect_added_characteristics`, `reconcile_removed_added_pairs`) correctly address the documented bug clusters (CLS-01 adjacency bleed, CLS-02 removed/added reconciliation, CLS-03 asymmetric tolerance kind transition). The test suite is thorough and the Pydantic/dataclass boundary is correctly maintained.

Four warnings were found, all in `classify.py` and `cli.py`. None are security issues. Three are correctness risks that could produce wrong classification results in specific edge cases; one is a missing error-handling gap. Five informational items covering code quality are also noted.

## Warnings

### WR-01: `is_in_revA` tolerance grid scan misses positions beyond ±15 pts by 5-pt stride

**File:** `delta_preservation/reconcile/classify.py:1206-1208`
**Issue:** The nested `range(-tol, tol + 1, 5)` loop iterates at a stride of 5 PDF points. With the default tolerance of 15 pts, checked offsets are {-15, -10, -5, 0, 5, 10, 15}. A position whose rounded centre falls 7 pts from the anchor (e.g., cx = round(42.5) = 42, signature cx = 49) is never tested — the iterator jumps from 5 to 10 and skips 7. In practice, spans that moved by a sub-stride amount between revisions could be incorrectly classified as "not in Rev A" and emitted as false added items. The intent is a continuous ±15 pt window, but the stride means only ~50% of offsets within that window are covered.

**Fix:** Use stride 1 for correctness, or switch to a set-based lookup that stores all rounded centre positions and use a tight comparison directly:
```python
def is_in_revA(span: TextSpan, position_tolerance: float = 15.0) -> bool:
    if not revA_span_signatures:
        return False
    x0, y0, x1, y1 = span.bbox_pdf
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    text = span.text.strip()
    tol = int(position_tolerance)
    # stride-1 so no offset within the window is skipped
    for dx in range(-tol, tol + 1):
        for dy in range(-tol, tol + 1):
            if (text, round(cx) + dx, round(cy) + dy) in revA_span_signatures:
                return True
    return False
```
Alternatively, store `revA_span_signatures` as a dict keyed by (text, cx, cy) and do a single distance check per entry, which is O(n) but avoids the grid scan entirely.

---

### WR-02: `reconcile_removed_added_pairs` mutates items in-place while iterating the same list

**File:** `delta_preservation/reconcile/classify.py:1799-1807`
**Issue:** `removed_items` is a filtered view constructed from the same `items` list passed in. Inside the loop, `removed.status = "changed"` mutates the object that still lives in the original `items` list (and inside `removed_items`). Although the mutation does not affect the iteration order of `removed_items` (it was pre-built), it does mean the function has an implicit side-effect on the caller's list that is not documented. The docstring says "Rewrite the removed item in place" but callers may not expect the original list to be modified when the function also returns a filtered copy. In `cli.py` line 511 the result is reassigned, but if a future caller passes the list and ignores the return value, the side-effect would produce a corrupted list with items that are silently "changed" without being filtered.

**Fix:** Either document the side-effect explicitly at the function signature level, or avoid mutating the original object by constructing a new `DeltaItem` with the updated fields and substituting it in the output list:
```python
# Instead of mutating:
# removed.status = "changed"
# ...
# Return a rebuilt list that includes a replacement object, not the mutated original
new_item = dataclasses.replace(
    removed,
    status="changed",
    confidence=max(removed.confidence, 0.70),
    added_span=best_added.added_span,
    reasons=removed.reasons + [reconcile_reason],
)
```

---

### WR-03: `classify_delta` notes-block path falls through silently when `revB_text_spans` is None

**File:** `delta_preservation/reconcile/classify.py:637-756`
**Issue:** The notes-block branch at line 637 checks `if "NOTES" in matched_fp.norm_text or "NOTE" in matched_fp.norm_text`. When `True`, it enters a sub-branch that checks `if revB_text_spans is not None` (line 641). When `revB_text_spans` is None, the entire content-comparison block is skipped and execution falls through to the unconditional `return DeltaItem(status="unchanged", confidence=0.85, ...)` at line 745. This means a notes anchor matched against a span containing "NOTE" will always be classified as "unchanged" with high confidence when called without `revB_text_spans`, even if the notes content changed significantly. This is a silent correctness issue: callers are not warned and get a confidently-wrong "unchanged" result.

**Fix:** Either require `revB_text_spans` to be provided for notes anchors (raise or return "uncertain"), or add an explicit reason explaining the degraded path:
```python
if revB_text_spans is None:
    # Cannot compare notes content without Rev B spans — degrade confidence
    return DeltaItem(
        char_no=anchor.char_no,
        status="uncertain",
        confidence=0.50,
        reasons=["Notes block matched but content comparison skipped: revB_text_spans not provided"],
        component_scores={...},
        match=match_or_none,
    )
```

---

### WR-04: `_FakeInternalDeltaItem` in tests lacks CLS-02 fields — `getattr` fallback silently masks attribute errors

**File:** `tests/test_output_formatting.py:57-66`  
**File:** `tests/test_pipeline_semantic_packet.py:47-55`
**Issue:** `_FakeInternalDeltaItem` does not define `added_requirement_text`, `added_bbox`, or `added_page`. The production `reconcile_removed_added_pairs` code accesses these via `getattr(it, "added_bbox", None)` (classify.py:1747). This means if a test accidentally passes a `_FakeInternalDeltaItem` with `status="added"` into `reconcile_removed_added_pairs`, the item will be silently ignored (not reconciled and not raised as an error) rather than causing a visible test failure. A production `DeltaItem` dataclass always has these fields, so the gap is only in the fake, but it makes the fakes misleading for CLS-02 test coverage purposes.

**Fix:** Add the three CLS-02 fields to both `_FakeInternalDeltaItem` definitions with `None` defaults so that they accurately reflect the production contract:
```python
class _FakeInternalDeltaItem:
    def __init__(self, *, char_no, status, confidence, reasons, component_scores,
                 match=None, added_span=None, confidence_flags=None,
                 added_requirement_text=None, added_bbox=None, added_page=None):
        ...
        self.added_requirement_text = added_requirement_text
        self.added_bbox = added_bbox
        self.added_page = added_page
```

---

## Info

### IN-01: Repeated `import re as _re` statements inside function body

**File:** `delta_preservation/reconcile/classify.py:398, 651, 680, 695, 849`
**Issue:** `re` is already imported at module level (line 4). Five additional `import re as _re*` statements appear inside the function body at various branch points, each under a different alias (`_re`, `_re_notes_filt`, `_re_ent`, `_re_kw`). This is redundant — Python caches imports, so there is no runtime cost, but it clutters the code and signals that these branches were added incrementally without considering the module-level import. It also causes linters to emit "redefined-outer-name" warnings.

**Fix:** Remove all inner `import re as ...` lines and use the module-level `re` directly, or create a single alias at module scope if needed.

---

### IN-02: `ExpandedWrapper` class defined inside a tight per-item loop

**File:** `delta_preservation/cli.py:580-583`
**Issue:** A one-off `ExpandedWrapper` class is defined inside a `for delta_internal in delta_items_internal` loop body (inside the `if is_notes_type:` branch). Python recreates the class object on every iteration that reaches this branch. The class is simple and the cost is minor, but a named `dataclass` or `SimpleNamespace` at module scope is cleaner and avoids class creation overhead.

**Fix:**
```python
# At module scope:
from types import SimpleNamespace

# In the loop:
expanded = SimpleNamespace(bbox=expanded_bbox)
```

---

### IN-03: `test_classify_phase5_regression.py` snapshot sweep relies on `pipeline_classification` field not present in the production `DeltaItem` schema

**File:** `tests/test_classify_phase5_regression.py:224`
**Issue:** `TestPhase5SnapshotSweep._conforming_unchanged_items()` filters on `item.get("pipeline_classification") == "unchanged"`. This field exists in the debug-report snapshot JSON (which comes from the review UI layer, not the pipeline packet), but it is not part of the `DeltaItem` Pydantic model in `types.py`. If the sweep were ever pointed at a raw `delta_packet.json` instead of the debug report, all items would silently fail the filter and the test would pass vacuously. The test is correct for the current fixture format but this implicit coupling is fragile.

**Fix:** Add a comment documenting that `SNAPSHOT_FILES` must point to debug-report JSON (not raw delta packet JSON) and that `pipeline_classification` is a debug-report-specific field. Alternatively, also assert `len(list(_conforming_unchanged_items())) > 0` to guard against vacuous pass.

---

### IN-04: Magic constant `CLS02_MAX_DISTANCE_PT = 150.0` is not tested at the boundary

**File:** `delta_preservation/reconcile/classify.py:1714`
**Issue:** `TestRemovedAddedReconciliation.test_far_apart_pair_stays_separate` uses a distance of 200 pts (clearly over the 150-pt limit). There is no test at exactly 150 pts (boundary should merge) or 151 pts (boundary should not merge). Off-by-one at the boundary (`> CLS02_MAX_DISTANCE_PT` vs `>= CLS02_MAX_DISTANCE_PT`) is not currently exercised.

**Fix:** Add a boundary test:
```python
def test_boundary_distance_exactly_at_threshold_merges(self):
    # Place added centroid exactly 150 pt from removed centroid
    # e.g., removed centroid at (60, 54), added centroid at (210, 54)
    ...
```

---

### IN-05: `classify_delta` notes-block keyword guard uses `import re as _re_notes_filt` but the alias is never used

**File:** `delta_preservation/reconcile/classify.py:651`
**Issue:** Line 651 imports `re` as `_re_notes_filt` inside the notes-block content-comparison branch. The alias `_re_notes_filt` is never referenced in the subsequent code — a different alias `_re_ent` is imported at line 680 and `_re` at line 695. The `_re_notes_filt` import is a dead import that was likely left over from a refactor.

**Fix:** Remove line 651 (`import re as _re_notes_filt`) entirely.

---

_Reviewed: 2026-04-16_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
