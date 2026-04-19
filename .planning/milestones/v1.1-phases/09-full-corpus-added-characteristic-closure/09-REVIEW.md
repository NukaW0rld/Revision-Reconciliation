---
phase: 09-full-corpus-added-characteristic-closure
reviewed: 2026-04-18T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - delta_preservation/evaluation/conformance.py
  - delta_preservation/reconcile/classify.py
  - shop/services/review.py
  - tests/test_added_detection_phase6.py
  - tests/test_added_truth_selection.py
  - tests/test_debug_row_identity.py
  - tests/test_phase7_benchmark.py
findings:
  critical: 0
  warning: 4
  info: 5
  total: 9
status: issues_found
---

# Phase 09: Code Review Report

**Reviewed:** 2026-04-18T00:00:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Seven files were reviewed: the conformance evaluator, the main classification engine, the shop review service, and four test modules. The code is well-structured overall, with clear domain intent. No security vulnerabilities or data-loss bugs were found. The warnings are correctness edge cases — a potential `None`-dereference inside the suppressor, a suppressor bypass when `cand_bbox` is `None`, and two logic inconsistencies in `classify.py`. The info items are dead code, repeated in-function imports, and minor naming/code-quality concerns.

---

## Warnings

### WR-01: `cand_bbox` is `None` — suppressor skips the bbox gate entirely, allowing over-suppression

**File:** `delta_preservation/reconcile/classify.py:2215-2226`

**Issue:** In the explained-by-match suppressor loop, when `cand_bbox` is `None` (i.e., an added candidate has no `added_bbox` and no `added_span`), the `if cand_bbox is not None:` block is skipped and the overlap guard is never evaluated. The suppressor then fires solely on the content-subset check (`_is_content_subset`), which can match on common token substrings. A candidate with no geometry evidence could be suppressed by an entirely unrelated matched annotation if they share one or two tokens (e.g., a bare `"A"` datum token or a tolerance value like `".015"`). This violates the stated invariant that "proximity alone is NOT sufficient."

**Fix:**
```python
# Before line 2219 (currently `suppressed = True`):
if cand_bbox is None:
    # No geometry evidence — skip this candidate to avoid false suppression.
    continue
```
Alternatively, require `cand_bbox is not None` as a precondition before entering the suppressor loop for a given candidate.

---

### WR-02: `_bbox_overlap_ratio` called twice on line 2224 with a potentially `None` `cand_bbox`

**File:** `delta_preservation/reconcile/classify.py:2224`

**Issue:** Inside the suppressor, the `suppression_reason` f-string calls `_bbox_overlap_ratio(cand_bbox, match_bbox)` unconditionally, but `cand_bbox` might be `None` at that point (the outer guard only wraps lines 2215-2218). If `cand_bbox` is `None` and the content-subset check passes on line 2212, `suppressed` is set to `True` and execution reaches line 2223, where `_bbox_overlap_ratio(None, match_bbox)` will unpack `None` and raise `TypeError`.

In practice this only occurs if the `if cand_bbox is not None` block is NOT reached (see WR-01), but the second call is still unguarded. Even after fixing WR-01, the f-string on line 2224 re-calls the function redundantly instead of reusing the already-computed `overlap`.

**Fix:**
```python
# Reuse the computed `overlap` value instead of recomputing:
suppression_reason = (
    f"explained by an existing matched characteristic: "
    f"candidate grouped text '{cand_text}' is a content subset of "
    f"matched annotation '{match_text[:80]}' "
    f"(bbox containment={overlap:.2f})"
)
```

---

### WR-03: `semantic_contract_for_item` always queries the packet twice — and returns the wrong row when multiple items share `char_no=None`

**File:** `shop/services/review.py:488-491`

**Issue:** `semantic_contract_for_item` delegates to `semantic_contracts_by_char(run)`, which builds a dict keyed by `char_no`. When `item.char_no is None` (all added items), it calls `.get(None)`. But `semantic_contracts_by_char` builds the dict by iterating the packet and overwriting earlier entries under the same key — so if multiple added items (all with `char_no=None`) exist, only the last one is stored under the `None` key. Any call to `semantic_contract_for_item` for any but the last added item will silently return the wrong semantic contract.

Furthermore, `semantic_contracts_by_char` re-reads and re-parses the full packet from disk on every call; `semantic_contract_for_item` is called once per queue item during rendering, making this O(N) reads for N items.

**Fix:** This function should not be exposed as a per-item API if `char_no=None` is ambiguous. Either add an item-id–keyed variant (which already exists as `semantic_contracts_by_item_id`), or document the limitation explicitly and stop calling this function for added items:
```python
def semantic_contract_for_item(run: Run, item: ReviewItem) -> dict | None:
    # NOTE: ambiguous for added items (char_no=None); prefer semantic_contracts_by_item_id
    if item.char_no is None:
        return None  # Cannot resolve without item_id
    return semantic_contracts_by_char(run).get(item.char_no)
```

---

### WR-04: `is_in_revA` position-tolerance loop uses integer steps of 5, silently misses positions between integer-5 grid points

**File:** `delta_preservation/reconcile/classify.py:1324-1327`

**Issue:** The `is_in_revA` helper iterates `range(-tol, tol + 1, 5)` over `dx` and `dy` to find a matching Rev A span within `position_tolerance=15.0` points. With step 5 the grid is: -15, -10, -5, 0, 5, 10, 15. A span whose PDF-rounded center is at offset (+3, +3) from the Rev B center (well inside the 15 pt radius) would be missed because none of the 5-pt grid points hit it. The method returns `False` for legitimate Rev A spans and allows them to be detected as added characteristics. The step size of 5 creates gaps for typical sub-pixel offsets from PDF rounding.

**Fix:** Use a continuous distance check rather than a discrete grid scan:
```python
def is_in_revA(span: TextSpan, position_tolerance: float = 15.0) -> bool:
    if not revA_span_signatures:
        return False
    x0, y0, x1, y1 = span.bbox_pdf
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    text = span.text.strip()
    for (sig_text, sig_cx, sig_cy) in revA_span_signatures:
        if sig_text == text and math.hypot(cx - sig_cx, cy - sig_cy) <= position_tolerance:
            return True
    return False
```
Note: `revA_span_signatures` currently stores `(text, int_cx, int_cy)` tuples; switching to float storage enables the continuous check.

---

## Info

### IN-01: `import unicodedata` inside `_normalize_for_suppression` is a dead import — `unicodedata` is never used

**File:** `delta_preservation/reconcile/classify.py:2089`

**Issue:** `_normalize_for_suppression` imports `unicodedata` but never calls any function from it. The docstring mentions keeping "GD&T symbols" but the implementation uses only regex whitespace collapse and `.upper()`. This is a dead import that was likely left over from an earlier implementation draft.

**Fix:** Remove line 2089 (`import unicodedata`).

---

### IN-02: Repeated in-function `import re as _re*` aliases — `re` is already imported at module scope

**File:** `delta_preservation/reconcile/classify.py:516, 769, 798, 813, 967`

**Issue:** `re` is imported at module top level (line 4). Five locations inside `classify_delta` and its nested blocks re-import it under local aliases (`_re`, `_re_notes_filt`, `_re_ent`, `_re_kw`). While this causes no runtime error, it obscures the code and makes the import graph harder to audit. The aliases are unnecessary since `re` is already in scope.

**Fix:** Remove the in-function `import re as ...` statements and use the module-level `re` directly.

---

### IN-03: `suppression_reason` is assigned but silently discarded — dead variable

**File:** `delta_preservation/reconcile/classify.py:2209-2231`

**Issue:** `suppression_reason` is populated with a detailed diagnostic string whenever an added candidate is suppressed, but then immediately discarded via `_ = suppression_reason`. The comment says "available for future debug export" but there is no mechanism to surface it — it is never logged, stored, or returned. This is dead code.

**Fix:** Either remove `suppression_reason` entirely and the `_ = suppression_reason` no-op, or integrate it into a structured debug channel (e.g., attach it to the candidate's `reasons` list before skipping, or store it in a debug side-channel). If debug export is genuinely planned, add a `TODO` with a ticket reference rather than a no-op assignment.

---

### IN-04: `_normalize_for_suppression` is imported but `unicodedata` not used — function comment does not match implementation

**File:** `delta_preservation/reconcile/classify.py:2081-2093`

**Issue:** The docstring for `_normalize_for_suppression` states "Strips GD&T punctuation, whitespace, and case-irrelevant separators so that '∅.045 A' and '◎ ∅.045 A' can be compared as subsets." The implementation, however, only collapses whitespace and uppercases — it does NOT strip GD&T punctuation (∅, ◎, ⌖, etc.) from either string. In practice, `"∅.045 A"` normalized is still `"∅.045 A"` — the subset check in `_is_content_subset` does find it inside `"◎ ∅.045 A"` as a direct substring, so the function works for that case. But the docstring implies broader normalization that isn't happening.

**Fix:** Either update the docstring to accurately describe the implementation, or implement the described stripping:
```python
def _normalize_for_suppression(text: str) -> str:
    """Normalize annotation text: collapse whitespace, uppercase."""
    return re.sub(r'\s+', ' ', text.strip()).upper()
```

---

### IN-05: `test_phase7_benchmark.py` — fixture-missing assertion message reveals filesystem paths that may vary across environments

**File:** `tests/test_phase7_benchmark.py:58`

**Issue:** The assertion `assert path.exists(), f"Missing required fixture: {path}"` emits the full absolute path in the failure message. On CI this path includes the workspace root (e.g., `/home/runner/work/...`). This is minor but can make CI failure messages environment-specific. Not a correctness issue.

**Fix:** Use a relative path in the message:
```python
assert path.exists(), f"Missing required fixture: {path.relative_to(Path(__file__).parent.parent)}"
```

---

_Reviewed: 2026-04-18T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
