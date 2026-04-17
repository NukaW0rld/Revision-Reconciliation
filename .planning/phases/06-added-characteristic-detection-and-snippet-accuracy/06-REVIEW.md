---
phase: 06-added-characteristic-detection-and-snippet-accuracy
reviewed: 2026-04-17T00:00:00Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - delta_preservation/cli.py
  - delta_preservation/evaluation/conformance.py
  - delta_preservation/reconcile/anchors.py
  - delta_preservation/reconcile/classify.py
  - delta_preservation/reconcile/exclusion.py
  - delta_preservation/reconcile/match.py
  - tests/test_added_detection_phase6.py
  - tests/test_added_truth_selection.py
  - tests/test_debug_row_identity.py
  - tests/test_phase6_asset_regression.py
  - tests/test_phase6_exclusion.py
findings:
  critical: 0
  warning: 6
  info: 5
  total: 11
status: issues_found
---

# Phase 06: Code Review Report

**Reviewed:** 2026-04-17
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

This phase introduces added characteristic detection (grouped callout evidence, GD&T FCF grouping, explained-by-match suppression) and duplicate added-truth disambiguation using packet Rev B bbox evidence. The implementation is architecturally sound and the shared exclusion contract (`exclusion.py`) is cleanly centralized. No critical security issues or data-loss bugs were found.

Six warnings were identified. The most significant are: a silent `except Exception: pass` in the identity-check scan that can mask parser errors; mutation of a `DeltaItem.status` field in place inside `reconcile_removed_added_pairs` (creates hidden aliasing risk); a bare `except Exception` block in the bleed-detection helper; floating-point equality comparison that is fragile across platforms; a `cand_bbox is None` suppression bypass that silently skips the geometric gate; and an O(n²) companion-span scan loop in Pass 0 that could become a correctness concern if the revB span count is large (flagged as a quality issue, not a performance issue).

Five info items cover: a runtime import inside a hot loop, a redundant inner `import re as _re_*` pattern, dead exception variable (`ValueError as e`), magic numbers without named constants, and a minor test coupling concern.

---

## Warnings

### WR-01: Silent `except Exception: pass` masks parser errors in identity-check scan

**File:** `delta_preservation/reconcile/classify.py:155-160`

**Issue:** In `_looks_like_adjacency_bleed`, the inner `try/except Exception: pass` silently swallows any exception from `parse_requirement(chunk)`. If `parse_requirement` raises due to unexpected input, the chunk is silently classified as non-anchor-bearing, potentially producing a wrong `True` return (false bleed detection). The same pattern appears at lines 173-177 in the foreign-count check.

**Fix:**
```python
# Replace bare except with explicit ValueError/AttributeError
try:
    chunk_fp = parse_requirement(chunk)
    chunk_nums = {v for v, _ in chunk_fp.numeric_tokens}
    if anchor_primary in chunk_nums:
        is_anchor_bearing = True
except (ValueError, AttributeError):
    pass  # parse_requirement contract guarantees it won't raise; this is defensive only
```

---

### WR-02: In-place mutation of `DeltaItem` in `reconcile_removed_added_pairs` creates aliasing risk

**File:** `delta_preservation/reconcile/classify.py:2078-2087`

**Issue:** `reconcile_removed_added_pairs` mutates `removed.status`, `removed.confidence`, `removed.added_span`, and `removed.reasons` directly on the objects already in the shared `items` list. Because `DeltaItem` is a plain `@dataclass` (not frozen), callers that hold references to these objects — including the loop in `cli.py` that iterates `delta_items_internal` — will see the mutation silently. This works correctly today only because the caller processes the returned list rather than the original objects, but it is a hidden contract violation that could produce double-mutation if the function is ever called twice on the same list, or if debug logging captures item state before the call.

**Fix:** Replace in-place mutation with a copy:
```python
# Instead of mutating removed.status etc. in place:
new_removed = DeltaItem(
    char_no=removed.char_no,
    status="changed",
    confidence=max(removed.confidence, 0.70),
    reasons=removed.reasons + [reconciliation_reason],
    component_scores=removed.component_scores,
    match=removed.match,
    added_span=best_added.added_span,
    added_requirement_text=removed.added_requirement_text,
    added_bbox=removed.added_bbox,
    added_page=removed.added_page,
    confidence_flags=removed.confidence_flags,
)
# Replace in final list rather than mutating
```

---

### WR-03: Suppression silently skips geometric gate when `cand_bbox` is None

**File:** `delta_preservation/reconcile/classify.py:1963-1966`

**Issue:** In the explained-by-match suppressor, when `cand_bbox is None`, the `if cand_bbox is not None: ... if overlap < 0.3: continue` block is skipped entirely, meaning suppression can fire on content-only ownership without any geometric check. A candidate with `added_bbox=None` (a legacy single-span item with no grouped bbox) will be suppressed by content match alone regardless of how far away the matched annotation is. This is the opposite of the stated contract ("Proximity alone is NOT sufficient").

**Fix:**
```python
# Gate 2: bbox ownership — require geometric confirmation
if cand_bbox is None:
    # No bbox evidence: cannot confirm ownership geometrically → do not suppress
    continue
overlap = _bbox_overlap_ratio(cand_bbox, match_bbox)
if overlap < 0.3:
    continue
```

---

### WR-04: Floating-point equality comparison in nearest-center tie-break

**File:** `delta_preservation/evaluation/conformance.py:239-251`

**Issue:** The nearest-center tie-break at line 244 selects candidates where `d == min_dist`. Floating-point `==` on computed distances (using `math.hypot`) is unreliable: two distances computed from different arithmetic paths may differ in the last ULP even when conceptually equal, causing the uniqueness check to fail or produce spurious false uniqueness. This can cause correct tie-break results to be silently discarded as ambiguous.

**Fix:**
```python
# Use a small epsilon for "same distance" comparison
_DIST_EPSILON = 1e-6
nearest = [
    (ti, tr)
    for ti, tr, d in candidates_with_distance
    if abs(d - min_dist) <= _DIST_EPSILON
]
```

---

### WR-05: `is_in_revA` brute-force grid search has quadratic complexity in position tolerance

**File:** `delta_preservation/reconcile/classify.py:1205-1208`

**Issue:** `is_in_revA` iterates `range(-tol, tol + 1, 5)` × `range(-tol, tol + 1, 5)` for every unmatched span, performing up to `(2 * tol / 5 + 1)² ≈ 49` set-membership probes per span. With `tol=15`, that is 49 probes × N unmatched spans × `revA_span_signatures` lookup. While set lookups are O(1), the double loop itself is a code-quality issue because the intent (find any signature within a radius) can be expressed more clearly and is error-prone if `tol` increases. More critically, using integer step-5 increments means positions at offsets 1–4, 6–9, etc. are never checked — the radius is not uniformly covered.

**Fix:**
```python
def is_in_revA(span: TextSpan, position_tolerance: float = 15.0) -> bool:
    """Return True if this span exists in Rev A at approximately the same position."""
    if not revA_span_signatures:
        return False
    x0, y0, x1, y1 = span.bbox_pdf
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    text = span.text.strip()
    # Check a grid of integer offsets within the tolerance radius
    tol = int(position_tolerance)
    for dx in range(-tol, tol + 1):
        for dy in range(-tol, tol + 1):
            if dx*dx + dy*dy > tol*tol:
                continue  # outside circular radius
            if (text, round(cx) + dx, round(cy) + dy) in revA_span_signatures:
                return True
    return False
```
Alternatively, store signatures in a spatial index (dict keyed by text → list of positions) and use a single linear scan.

---

### WR-06: `_expand_standard_added_span` excludes matched-span companions, causing incomplete grouping for cross-ownership cases

**File:** `delta_preservation/reconcile/classify.py:1590-1608`

**Issue:** The `_expand_standard_added_span` helper explicitly skips any span whose key is in `matched_span_keys` (line 1592-1593). This is correct to prevent consuming already-matched spans as new added items, but it means that if a legitimate companion of a new annotation happens to share its bbox_pdf with a matched span (due to a PDF extraction glitch that generates duplicate spans at the same coordinates), the companion will be silently dropped from the grouped text. The result is a truncated `added_requirement_text` that misses one companion. This is the same class of bug that Phase 6 is trying to fix for GD&T frames (Pass 0 includes matched companions for text but not consumption). Pass 2 should do the same.

**Fix:** Mirror the Pass 0 pattern: include matched spans in group text but do not add them to `companion_keys` (so they are not marked as newly consumed):
```python
for other in revB_spans:
    other_key = (other.block_id, other.line_id, other.span_id, other.bbox_pdf)
    if other_key == seed_key or other_key in consumed_keys:
        continue
    # ... proximity checks ...
    group_spans.append(other)
    # Only mark unmatched spans consumed — matched spans included for text only
    if other_key not in matched_span_keys:
        companion_keys.add(other_key)
```

---

## Info

### IN-01: Runtime import inside hot inner loop

**File:** `delta_preservation/reconcile/classify.py:397`

**Issue:** `import re as _re` appears inside the `classify_delta` function body (executed on every call), alongside several other inner `import re as _re_*` aliases at lines 650, 679, 694, 848. Python caches module imports after the first resolution, so this is not a correctness issue, but it is a code-quality smell that creates confusion about which `re` alias is in scope at any point.

**Fix:** Move all `re` imports to the module-level `import re` already present at line 4. Remove all `import re as _re` inside function bodies.

---

### IN-02: Unused exception variable `e` in `cli.py`

**File:** `delta_preservation/cli.py:761, 782`

**Issue:** Both `except ValueError as e:` blocks in the snippet-generation section capture the exception but never reference `e` (the body is just a comment + assignment). Python will hold the exception in scope unnecessarily.

**Fix:**
```python
except ValueError:
    # Bbox invalid, record evidence without image
    revA_evidence = Evidence(...)
```

---

### IN-03: Magic number `0.12` duplicated between `generate_candidates` and `assign_matches`

**File:** `delta_preservation/reconcile/match.py:448, 865`

**Issue:** The viability threshold `0.12` is declared as `MIN_MATCH_SCORE = 0.12` inside `assign_matches` (line 865) but also hard-coded in `generate_candidates` at line 448 (`not any(c.total_score >= 0.12 for c in candidates)`). If the threshold changes in `assign_matches`, `generate_candidates` will silently use a stale value and the global fallback trigger will be out of sync.

**Fix:** Extract to a module-level constant:
```python
# At module level
_MIN_VIABLE_CANDIDATE_SCORE: float = 0.12

# In generate_candidates:
no_viable_candidate = not any(c.total_score >= _MIN_VIABLE_CANDIDATE_SCORE for c in candidates)

# In assign_matches:
MIN_MATCH_SCORE = _MIN_VIABLE_CANDIDATE_SCORE
```

---

### IN-04: `test_debug_row_identity.py` imports `shop.*` which is an undiscovered dependency

**File:** `tests/test_debug_row_identity.py:8-9`

**Issue:** The test file imports `from sqlalchemy.orm import sessionmaker`, `from shop.models import Run`, and `from shop.services.review import build_debug_queue_state`. None of these modules are part of the `delta_preservation` package reviewed in this phase. If the `shop` package is not installed in the test environment, the entire module will fail to import and all four tests will silently be skipped or produce a collection error rather than a meaningful failure. The other test files in this phase do not have this dependency.

**Fix:** Confirm `shop` is installed in CI. If the `shop` package is optional or environment-specific, add a `pytest.importorskip("shop")` guard at the top of the file:
```python
shop = pytest.importorskip("shop", reason="shop package not installed")
```

---

### IN-05: `_normalize_for_suppression` imports `unicodedata` but does not use it

**File:** `delta_preservation/reconcile/classify.py:1847`

**Issue:** The function `_normalize_for_suppression` has `import unicodedata` inside its body but the function body never calls any `unicodedata` function. The import is dead code left from a prior implementation.

**Fix:** Remove the `import unicodedata` line inside `_normalize_for_suppression`.

---

_Reviewed: 2026-04-17_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
