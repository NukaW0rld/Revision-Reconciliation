# Phase 5: Classification Logic Fixes — Pattern Map

**Mapped:** 2026-04-16
**Files analyzed:** 6 (2 to modify, 1 to extend, 1 new directory, 2 read-only analog sources)
**Analogs found:** 6 / 6 — every touchpoint has an in-repo analog. No "no analog" entries.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `delta_preservation/reconcile/classify.py` (MODIFY — dataclass + helpers + two decision branches + new post-pass function) | classifier / decision-tree + post-pass reconciler | transform (in-memory, synchronous, per-anchor + cross-item) | itself — pre-existing `DeltaItem` dataclass (line 23), `count_added` branch (line 656/733), `tolerance_comparison` branch (line 904), `is_near_matched_span` proximity helper (line 1097), `detect_added_characteristics` post-pass (line 941) | exact — same file, same role |
| `delta_preservation/types.py` (MODIFY — additive `confidence_flags: List[str]` field on Pydantic `DeltaItem`) | schema (Pydantic) | persisted (JSON blob inside `DeltaPacket`) | `types.DeltaItem.reasons: List[str]` (line 277), `GdtSemanticPayload.datum_refs` / `.modifiers` / `.compartments` (lines 85-99), `AcceptedAlternate.mismatch_codes` (line 197), `PacketEvaluation.mismatches` (line 230) | exact — same file, same role, same container type |
| `delta_preservation/cli.py` (MODIFY — one-line call at ~508 + kwarg at ~842 on Pydantic constructor) | pipeline wiring | orchestration (reads anchors + items, emits packet) | itself — pre-existing `delta_items_internal.extend(added_items)` (line 507) and `DeltaItem(…)` constructor call (line 842) | exact — same file |
| `tests/test_classify_bugfixes.py` (EXTEND — three new test classes) | unit test | request-response (synchronous, pure-function) | existing `TestCountAdded` (line 104), `TestSpuriousMatchGuard` (line 126), `TestGdtAddedDetection` (line 163), `TestPlainDecimalDetection` (line 202), `TestToleranceOverlapThreshold` (line 238) | exact — same file, same helpers |
| `tests/fixtures/classify/` (NEW directory) | test fixture data | file I/O (read-only loaded by tests) | **no analog in repo** — Phase 5 research references "JSON fixtures extracted from `assets/debug_report_part*.json`" but no existing `tests/fixtures/` directory exists today. Closest analog: `assets/debug_report_part*.json` themselves (read-only inputs for tests) | partial — new directory, but file-shape convention can mirror `assets/debug_report_part*.json` |
| `delta_preservation/reconcile/classify.py` — mutable dataclass `DeltaItem` field add (line 24) | data-class schema | in-memory transport | itself — `reasons: List[str]` on the same dataclass (line 29) | exact |

## Pattern Assignments

### `delta_preservation/reconcile/classify.py` — DeltaItem dataclass field addition (CLS-01 plumbing)

**Analog (same file, self-analog):** `delta_preservation/reconcile/classify.py` lines 23-32

**Current code pattern (to copy from — the `reasons` field is the precise template):**
```python
# classify.py:23-32
@dataclass
class DeltaItem:
    """Classification result for a single Rev A characteristic."""
    char_no: int
    status: str  # "unchanged", "changed", "removed", "uncertain", "added"
    confidence: float
    reasons: List[str]
    component_scores: Dict[str, float]
    match: Optional[Match] = None
    added_span: Optional[TextSpan] = None  # For added characteristics
```

**What to copy:** Add `confidence_flags: List[str] = field(default_factory=list)` immediately after `added_span`. Because the existing dataclass has non-default fields before defaulted fields, the new field must come at the END (after `added_span`) OR after `reasons` but before any default-valued field — Python dataclass rule. Simplest placement: after `added_span: Optional[TextSpan] = None`.

**Required import addition at top of file** (classify.py:6 currently reads `from dataclasses import dataclass`):
```python
from dataclasses import dataclass, field
```

---

### `delta_preservation/types.py` — Pydantic DeltaItem field addition (CLS-01 persistence)

**Analog (same file, self-analog):** `delta_preservation/types.py`

**Imports pattern (already in file):**
```python
# types.py top (already present)
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
```

**Core pattern — `reasons` field (line 277) is the shape-match, but it uses a required `Field(...)` with no default. For backward-compat deserialization of legacy packets, prefer the `default_factory=list` pattern used by `GdtSemanticPayload.datum_refs` (line 85):**
```python
# types.py:85-99 — EXACT template for confidence_flags
    datum_refs: List[str] = Field(default_factory=list, description="Referenced datums for this compartment")
    modifiers: List[str] = Field(default_factory=list, description="Applied modifiers for this compartment")
```
```python
# types.py:197-200 — another default_factory=list analog on a persisted model
    mismatch_codes: List[str] = Field(
        default_factory=list,
        description="Stable mismatch-code fingerprint captured when the alternate was approved",
    )
```

**What to copy:** Add after line 277 (after `reasons:` and before `scores:`):
```python
    confidence_flags: List[str] = Field(
        default_factory=list,
        description="Advisory flags annotating classifier reasoning (e.g., bleed warnings). Not authoritative — see `status` for verdict.",
    )
```

**Why `default_factory=list`, not `Field(...)`:** The `reasons` field uses `Field(...)` (required). That would break JSON deserialization of legacy SQLite-stored packets that don't have `confidence_flags`. `default_factory=list` matches the pattern at lines 85, 86, 95, 96, 99, 198, 231 — all list-typed optional/new fields on persisted models use `default_factory=list`.

**Pitfall (from RESEARCH Pitfall 7):** Do NOT use a bare `= []` default — Pydantic v2 may not always deep-copy it, and even if it does, the idiom across this file is `default_factory=list`.

---

### `delta_preservation/reconcile/classify.py` — `_looks_like_adjacency_bleed` helper (CLS-01)

**Analog (same file, self-analog):** The existing regex module-level pattern near line 468 (notes-item regex) — same file uses `re.compile` module-level constants.

**Imports pattern (already in file at classify.py:1-5):**
```python
from __future__ import annotations
import math
import re
from typing import Optional, Dict, List, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass
```

**Module-level regex analog pattern (classify.py already uses this style — see existing `re.compile` usages at classify.py:253, 468, 1225):**
- Place the two `re.compile` constants (`_BLEED_SLASH_RE`, `_COUNT_PREFIX_RE`) at module-level near the top, grouped with other classifier module constants.
- Name with leading underscore (private module helper) — matches the house style of private helpers throughout `classify.py`.

**What to copy:** See RESEARCH.md Pattern 1 for the full helper body. The helper signature is:
```python
def _looks_like_adjacency_bleed(span_text: str) -> bool:
```
Return type `bool`, single string input, no mutation — pure predicate. This shape matches the existing `is_near_matched_span` helper at classify.py:1097 (pure predicate returning bool).

---

### `delta_preservation/reconcile/classify.py` — CLS-01 integration into `count_added` branch

**Analog (same file, self-analog):** `delta_preservation/reconcile/classify.py` lines 732-740 (exact target branch)

**Current code (lines 732-740) — this is the exact block to modify:**
```python
    # Priority 1: Primary dimension value match is the strongest indicator
    if count_changed or count_added:
        # Count explicitly changed or added (e.g., none → "2X", or "2X" → "4X") → changed
        status = "changed"
        confidence = 0.5 * location_score + 0.3 * numeric_overlap + 0.2
        if count_changed:
            reasons.append(f"Count changed: {anchor_count} → {matched_count}")
        else:
            reasons.append(f"Count added in Rev B: {matched_count} (was absent in Rev A)")
```

**Pre-existing sibling suppressor (lines 700-730) — DO NOT DUPLICATE.** This guard fires earlier, for pure-text anchors with no numerics. The CLS-01 bleed check is orthogonal: it fires when the anchor DOES have numerics but the Rev B span text is `/`-merged. See RESEARCH Pitfall 2.

**What to copy:** Replace the `if count_changed or count_added:` body with a branched form that runs `_looks_like_adjacency_bleed(candidate.span.text)` ONLY when `count_added` is the trigger (NOT when `count_changed`), and when bleed is detected, set status back to `unchanged` and append to `confidence_flags`. See RESEARCH Pattern 1 "Integration point" block for the precise code.

**Verification target (existing passing test, must continue to pass):** `tests/test_classify_bugfixes.py::TestCountAdded::test_count_added_marks_changed` (line 107) — anchor `"Ø 8"` + candidate `"2X Ø 8"` has no `/` → bleed check returns False → `status = "changed"` as before.

---

### `delta_preservation/reconcile/classify.py` — CLS-03 kind-based asymmetry branch

**Analog (same file, self-analog):** `delta_preservation/reconcile/classify.py` lines 903-922 (exact target branch, the existing tolerance_comparison block)

**Current code (lines 903-922):**
```python
    # --- Tolerance refinement ---
    if tolerance_comparison is not None:
        if tolerance_comparison.tolerances_match:
            if status == "unchanged":
                confidence += 0.05
                reasons.extend(tolerance_comparison.reasons)
            elif status == "uncertain":
                status = "unchanged"
                confidence += 0.1
                reasons.append("Tolerance agreement resolves uncertainty")
                reasons.extend(tolerance_comparison.reasons)
        elif tolerance_comparison.tolerances_differ:
            if status == "unchanged":
                status = "changed"
                reasons.append("Tolerance changed despite same dimension")
                reasons.extend(tolerance_comparison.reasons)
            elif status == "uncertain":
                status = "changed"
                reasons.append("Tolerance difference resolves uncertainty")
                reasons.extend(tolerance_comparison.reasons)
```

**Related read-only consumption (tolerance_pdf.py:41-47, 96-106):**
```python
# tolerance_pdf.py:41-47
PdfToleranceKind = Literal[
    "none",
    "plus_minus",
    "bilateral_stacked",
    "unilateral_stacked",
    "limits_stacked",
]
```
```python
# tolerance_pdf.py:96-106
@dataclass(frozen=True)
class ToleranceComparison:
    char_no: int
    revA_tolerance: PdfTolerance
    revB_tolerance: PdfTolerance
    tolerances_match: bool
    tolerances_differ: bool
    has_tolerance: bool
    reasons: List[str]
```
Where `PdfTolerance` (line 54) has a `.kind: PdfToleranceKind` attribute (line 68).

**What to copy:** Append the kind-based check inside the `if tolerance_comparison is not None:` block, AFTER the existing match/differ branches. The check must only upgrade `status == "unchanged"` → `"changed"` (never downgrade). See RESEARCH Pattern 3 "Algorithm sketch" for the exact code. Plus a fallback path outside the `tolerance_comparison is not None` block for the case where unit tests pass `tolerance_comparison=None`.

**Style note:** Use `reasons.append(...)` (single string) — consistent with the existing branch style at line 917. Use `max(confidence, 0.65)` not `confidence = 0.65` — consistent with the `max(0.65, …)` pattern at line 773.

---

### `delta_preservation/reconcile/classify.py` — CLS-02 `reconcile_removed_added_pairs` function

**Analog (same file, self-analog):** `delta_preservation/reconcile/classify.py`

**Placement guidance:** Place the new function AFTER `detect_added_characteristics` (starts at line 941). This groups all "list-of-items cross-scan" functions together.

**Proximity-math analog (lines 1097-1103) — the exact `math.sqrt` style to copy:**
```python
# classify.py:1097-1103
def is_near_matched_span(bbox: tuple, threshold: float = 25.0) -> bool:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for mx, my in matched_centers:
        if math.sqrt((cx - mx) ** 2 + (cy - my) ** 2) <= threshold:
            return True
    return False
```
Copy this centroid-distance-comparison style exactly. Use `math.sqrt`, not a helper; the call count is ~O(N²) where N≤20, so trivially fast.

**Type-gate analog — normalize helpers already imported at classify.py:10-14:**
```python
# classify.py:10-14 (imports already present — reuse, don't re-import)
from delta_preservation.reconcile.normalize import (
    are_requirement_types_incompatible,
    classify_requirement_type,
    parse_requirement,
)
```

**What to copy:** Full `reconcile_removed_added_pairs(items, anchors)` function per RESEARCH Pattern 2 "Algorithm sketch". Key conventions to follow:
- Named module-level constants `CLS02_MAX_DISTANCE_PT = 150.0` and `CLS02_STRICT_DISTANCE_PT = 75.0` at top of file (near regex module constants) — matches the existing style of named thresholds in the same file.
- Mutate `removed` items in place (DeltaItem is a mutable dataclass) — do NOT construct new items. This preserves list order, per RESEARCH Pitfall 6.
- Filter paired added items out of the return list; do NOT delete from the input list.
- Guard `added_span is not None` in the list comprehension (RESEARCH Pitfall 3).
- Guard `anchor_by_char.get(r.char_no)` (returns None if missing) — matches the defensive style at classify.py:532 in cli.py.

---

### `delta_preservation/cli.py` — CLS-02 post-pass wiring (line ~508)

**Analog (same file, self-analog):** `delta_preservation/cli.py` lines 500-508 (exact injection site)

**Current code (lines 500-508):**
```python
    # Detect added characteristics (new in Rev B, not in Rev A)
    max_char_no = max(a.char_no for a in anchors) if anchors else 0
    added_items = detect_added_characteristics(
        revB_text_spans, matches, next_char_no=max_char_no + 1,
        page_width=page_width_b, page_height=page_height_b,
        revA_spans=revA_text_spans,
    )
    delta_items_internal.extend(added_items)
    print(f"  Found {len(added_items)} added characteristics in Rev B")
```

**What to copy:** Insert the reconciliation call between line 507 (`extend`) and line 508 (`print`) — or, safer, between the `print` and the `export_run_tolerance_debug` call at line 510. New line:
```python
    delta_items_internal = reconcile_removed_added_pairs(delta_items_internal, anchors)
```
Import addition at top of cli.py (add to existing `from delta_preservation.reconcile.classify import …` block):
```python
    reconcile_removed_added_pairs,
```

**Locate by symbol, not line number (RESEARCH Pitfall 1):** The executor MUST find `delta_items_internal.extend(added_items)` by text search, then insert the new call immediately below it — NOT by hard-coded line 508.

---

### `delta_preservation/cli.py` — Pydantic constructor kwarg (line ~842)

**Analog (same file, self-analog):** `delta_preservation/cli.py` lines 841-853 (exact target)

**Current code (lines 841-853):**
```python
        # Create Pydantic DeltaItem
        delta_pydantic = DeltaItem(
            char_no=delta_internal.char_no,
            status=delta_internal.status,
            confidence=delta_internal.confidence,
            reasons=delta_internal.reasons,
            scores=delta_internal.component_scores,
            revA=revA_evidence,
            revB=revB_evidence,
            requirement_revB=requirement_revB,
            semantic_callout=semantic_callout,
            snippet_rule_family=snippet_rule_family,
        )
```

**What to copy:** Add one kwarg line. Place it adjacent to `reasons=delta_internal.reasons,` so the `reasons` / `confidence_flags` pair stays visually grouped (mirroring how they're grouped on the dataclass):
```python
            confidence_flags=delta_internal.confidence_flags,
```

---

### `tests/test_classify_bugfixes.py` — three new test classes (CLS-01, CLS-02, CLS-03)

**Analog (same file, self-analog):** `tests/test_classify_bugfixes.py`

**Existing helper pattern — use unchanged (lines 35-97):**
```python
# test_classify_bugfixes.py:35-66 — _span + _anchor helpers
def _span(
    text: str,
    *,
    block_id: int = 0,
    line_id: int = 0,
    span_id: int = 0,
    x0: float = 10.0,
    y0: float = 10.0,
    width: float = 20.0,
    height: float = 8.0,
) -> TextSpan:
    return TextSpan(
        text=text,
        bbox_pdf=(x0, y0, x0 + width, y0 + height),
        font_size=10.0,
        block_id=block_id,
        line_id=line_id,
        span_id=span_id,
    )


def _anchor(requirement_raw: str, char_no: int = 1) -> Anchor:
    anchor_span = _span(requirement_raw)
    return Anchor(
        char_no=char_no,
        page=0,
        balloon_bbox=(0.0, 0.0, 5.0, 5.0),
        req_bbox=anchor_span.bbox_pdf,
        requirement_raw=requirement_raw,
        requirement_norm=requirement_raw,
        local_context=[anchor_span],
    )
```

```python
# test_classify_bugfixes.py:73-97 — _classify helper
def _classify(anchor: Anchor, candidate_span: TextSpan):
    """Run classify_delta through the full candidate/match pipeline."""
    anchor_semantic = extract_semantic_callout(
        pdf_spans=[_span(anchor.requirement_raw)],
        form3_requirement=anchor.requirement_raw,
    )
    matched_semantic = extract_semantic_callout(pdf_spans=[candidate_span])
    revb_semantic_by_span = {_span_key(candidate_span): matched_semantic}

    candidates = generate_candidates(
        anchor,
        [candidate_span],
        _Transform(),
        anchor_semantic_callout=anchor_semantic,
        revB_semantic_callouts_by_span_key=revb_semantic_by_span,
    )
    assert candidates, "Expected at least one candidate"
    match = Match(char_no=anchor.char_no, candidate=candidates[0])
    delta = classify_delta(
        anchor,
        match,
        anchor_semantic_callout=anchor_semantic,
        matched_semantic_callout=matched_semantic,
    )
    return delta
```

**Existing test class shape (copy this structure verbatim — one Test class per bug cluster):**
```python
# test_classify_bugfixes.py:104-119 — TestCountAdded (the pattern to follow)
class TestCountAdded:
    """When Rev B adds a count prefix (e.g., none → '2X Ø8'), status must be 'changed'."""

    def test_count_added_marks_changed(self):
        anchor = _anchor("Ø 8")
        candidate = _span("2X Ø 8", block_id=1, line_id=0, span_id=0, x0=12.0, y0=11.0)
        delta = _classify(anchor, candidate)
        assert delta.status == "changed", f"Expected 'changed', got '{delta.status}'"
        assert any("Count added" in r or "count" in r.lower() for r in delta.reasons)

    def test_count_same_stays_unchanged(self):
        """Sanity: when both sides have the same count the status stays unchanged."""
        anchor = _anchor("2X Ø 8")
        candidate = _span("2X Ø 8", block_id=1, line_id=0, span_id=0, x0=12.0, y0=11.0)
        delta = _classify(anchor, candidate)
        assert delta.status == "unchanged"
```

**What to copy for the three new classes:**
- Preserve the one-line class-level docstring (states which requirement ID the class covers: CLS-01 / CLS-02 / CLS-03).
- Method-level: no docstring if the name is self-explanatory, otherwise a single-line docstring — matches the existing mix (some methods have docstrings, some don't).
- Use `_anchor(...)` + `_span(...)` + `_classify(...)` exclusively for CLS-01 and CLS-03 tests (request-response pattern).
- For CLS-02, bypass `_classify` — construct `DeltaItem` instances directly (same way `TestGdtAddedDetection` bypasses by calling `detect_added_characteristics` directly at line 173). See RESEARCH Example 2 for the full body.
- Use `assert delta.status == "…", f"Expected …, got '{delta.status}'"` failure message style — consistent with line 111.
- Use `assert any(pattern in r for r in delta.reasons)` for reason/flag checks — consistent with line 112.

**Direct-import pattern for CLS-02 (follows the top-of-file import at line 11-15):**
```python
from delta_preservation.reconcile.classify import (
    DeltaItem,
    classify_delta,
    detect_added_characteristics,
    reconcile_removed_added_pairs,  # NEW for CLS-02
)
```

---

### `tests/fixtures/classify/` — new fixture directory

**Analog:** `assets/debug_report_part*.json` — the current read-only JSON reference corpus.

**What to copy (convention):**
- JSON files are the project's standard static-fixture format (see `assets/part*/ground_truth.json` and `assets/debug_report_part*.json`).
- File naming convention to propose (there is no existing `tests/fixtures/` to inherit from): `bleed_part1_char11.json`, `remadd_pair_part2_char1.json`, `asym_tolerance_part7_char4.json` — one JSON per exemplar.
- Load in tests via `pathlib.Path(__file__).parent / "fixtures" / "classify" / "..."` + `json.loads()` — the standard stdlib approach used elsewhere.

**Alternative (recommended by pattern parsimony):** Skip the fixtures directory for Phase 5. All three test classes can be fully exercised with inline `_span(...)` + `_anchor(...)` constructions. The `tests/fixtures/classify/` directory is only worth creating if multiple test modules need to share fixture data. RESEARCH says "JSON fixtures extracted from `assets/debug_report_part*.json`" but RESEARCH's own test examples (Examples 1-3) all use inline literals — no fixture file is actually required. Let the planner decide whether to scope this in or out.

---

## Shared Patterns

### Pattern A: Module-level private helpers named with `_leading_underscore`
**Source:** `delta_preservation/reconcile/classify.py` — `_kw_tokens` (line 702), `_BLEED_SLASH_RE` / `_COUNT_PREFIX_RE` convention, `_span_key` in the test file (line 69).
**Apply to:** `_looks_like_adjacency_bleed`, `_BLEED_SLASH_RE`, `_COUNT_PREFIX_RE`, any private helper in `classify.py`.

```python
# Inspired by the pattern at classify.py:701-705 (inline private def)
# For module-level helpers, use the same underscore-prefix convention.
_BLEED_SLASH_RE = re.compile(r"\s*/\s*")
_COUNT_PREFIX_RE = re.compile(r"\b\d+\s*[Xx](?!\s*\d)")


def _looks_like_adjacency_bleed(span_text: str) -> bool:
    ...
```

### Pattern B: `reasons.append(...)` rather than `reasons += [...]` or `reasons.extend([...])` for single strings
**Source:** `delta_preservation/reconcile/classify.py` — lines 738, 740, 774, 779, 781, 816, 822, 886, 892, 897, 901, 917, 921 (every branch uses `.append` for single messages and `.extend(tolerance_comparison.reasons)` only when folding in a pre-built list).
**Apply to:** All new code in `classify_delta` branches touched by Phase 5.

### Pattern C: `confidence = max(prev, new)` to avoid clobbering a higher pre-existing score
**Source:** `delta_preservation/reconcile/classify.py` line 773 (`confidence = max(0.65, 0.4 * location_score + 0.3 * numeric_overlap + 0.2)`).
**Apply to:** CLS-03 kind-based asymmetry branch, CLS-01 bleed-demotion branch, CLS-02 reconciliation.

### Pattern D: Mutable dataclass in-place mutation (DO NOT reconstruct)
**Source:** `delta_preservation/reconcile/classify.py` — `DeltaItem` is `@dataclass` (line 23), not `@dataclass(frozen=True)`. The CLI already relies on this (cli.py:491, 507) via `append` / `extend`.
**Apply to:** CLS-02's `reconcile_removed_added_pairs` — mutate `r.status = "changed"` and `r.reasons = [...]` in place; don't construct a new `DeltaItem`.

### Pattern E: Default-factory for Pydantic list fields on persisted models
**Source:** `delta_preservation/types.py` lines 55-57, 85, 86, 95-99, 197-200, 230-233.
**Apply to:** `DeltaItem.confidence_flags` — `List[str] = Field(default_factory=list, description="...")`.

### Pattern F: Type-import guard with `TYPE_CHECKING`
**Source:** `delta_preservation/reconcile/classify.py` lines 19-20.
```python
if TYPE_CHECKING:
    from delta_preservation.reconcile.tolerance_pdf import ToleranceComparison
```
**Apply to:** Any new typing-only imports in classify.py (unlikely for Phase 5, but noted for consistency if a new forward-reference is needed).

### Pattern G: Existing `reasons.extend(tolerance_comparison.reasons)` style for folding pre-built lists
**Source:** `delta_preservation/reconcile/classify.py` lines 908, 913, 918, 922.
**Apply to:** CLS-02 reconciliation — when merging paired added item's reasons into the rewritten removed item, use `extend` plus a header line (per RESEARCH Pattern 2 code sketch).

### Pattern H: Test failure messages use f-string with expected+actual
**Source:** `tests/test_classify_bugfixes.py` line 111 (`f"Expected 'changed', got '{delta.status}'"`) and line 156.
**Apply to:** All new test assertions.

### Pattern I: Import-direct test imports (no `conftest.py` fixture indirection)
**Source:** `tests/test_classify_bugfixes.py` lines 9-17 — direct imports from `delta_preservation.io.pdf`, `.reconcile.anchors`, `.reconcile.classify`, `.reconcile.match`, `.reconcile.normalize`. No `pytest.fixture` decorators, no `conftest.py` dependency.
**Apply to:** All three new test classes — use direct imports and the existing `_span` / `_anchor` / `_classify` helpers.

---

## No Analog Found

**None.** Every touchpoint in this phase has a close in-repo analog:
- Dataclass field addition → `reasons: List[str]` on the same dataclass
- Pydantic field addition → six existing `default_factory=list` examples in the same file
- Module-level regex helper → existing module-level regex patterns in `classify.py`
- Post-pass reconciler → `detect_added_characteristics` (same file, same "cross-item list transformer" shape)
- Proximity math → `is_near_matched_span` (same file, same Euclidean-centroid pattern)
- Test class → four existing test classes in the same test file with the same helper harness
- CLI wiring → the exact two injection sites already live in the file, surrounded by sibling `extend` / kwarg patterns

The lone "partial" entry is `tests/fixtures/classify/` (no `tests/fixtures/` directory currently exists). Recommendation: defer or skip — RESEARCH's test examples (Examples 1-3) use inline literals, so fixtures aren't structurally required for Phase 5.

---

## Metadata

**Analog search scope:**
- `delta_preservation/reconcile/classify.py` (full file, ~1478 lines)
- `delta_preservation/reconcile/tolerance_pdf.py` (lines 30-150 — `PdfToleranceKind`, `PdfTolerance`, `ToleranceComparison`, `compare_tolerances`)
- `delta_preservation/reconcile/anchors.py` (lines 1-60 — `Anchor` dataclass)
- `delta_preservation/types.py` (lines 50-295 — every `Field(default_factory=list)` instance)
- `delta_preservation/cli.py` (lines 485-860 — per-anchor classify loop, post-pass injection site, Pydantic constructor)
- `tests/test_classify_bugfixes.py` (full file — all four existing test classes + helpers)

**Files scanned:** 6 source files + 1 test file + 1 roadmap + 1 research doc = 9.

**Pattern extraction date:** 2026-04-16

**Confidence:** HIGH — every analog is in-file or in a tightly-coupled sibling module. No extrapolation from external projects. RESEARCH.md pre-verified all cited line numbers this same session; this PATTERNS.md re-read each cited region to confirm.
