# Phase 5: Classification Logic Fixes - Research

**Researched:** 2026-04-16
**Domain:** Delta classification decision tree in `delta_preservation/reconcile/classify.py` and its downstream interaction with `semantic_compare.py`, `tolerance_pdf.py`, and the Rev B "added characteristic" detector.
**Confidence:** HIGH (every finding verified against the current source and the 9-part debug corpus)

<user_constraints>
## User Constraints (from phase spec — no CONTEXT.md authored yet)

Phase 5 is in assumptions mode: no `05-CONTEXT.md` exists, so constraints are drawn verbatim from the phase description, `REQUIREMENTS.md` (CLS-01, CLS-02, CLS-03), `ROADMAP.md` Phase 5 success criteria, and the milestone-level rules in `PROJECT.md` / `STATE.md`.

### Locked Decisions (from phase description + REQUIREMENTS.md)

**CLS-01 — Adjacency-bleed false positive suppression**
- A Rev B PDF text span containing a `/`-merged multi-balloon bleed **must not** produce a `count_added` false-positive "changed" verdict.
- The item **must** instead carry a confidence flag with the exact user-facing wording **"Rev B text may contain adjacent balloon content"** (see part-7 char-1 reviewer note in `assets/debug_report_part7.json`: *"It could say something like 'Rev B text may contain adjacent balloon content' whenever this bleeding occurs"*).
- The item **must not** be classified as `changed` on the basis of bleed alone.

**CLS-02 — Close-proximity removed+added collapse**
- A Rev A characteristic emitted as `removed` **must** collapse with an unmatched Rev B added characteristic into a single `changed` row when both lie on the **same page** within **spatial proximity**.
- The post-pass runs **after** both `classify_delta` (all anchors) and `detect_added_characteristics` have emitted their items.
- Clusters affected: parts 2, 3, 4, 8 — 6 items total.

**CLS-03 — Symmetric→asymmetric tolerance change detection**
- `±1°` → `+0.3° / −0.1°` (and any symmetric→asymmetric transition that preserves the primary dimension) **must** classify as `changed`, not `unchanged`.
- The detection MUST be structural (shape of the tolerance expression), not only numeric-set based — the existing `tolerance_sized_diff` branch at classify.py:746 is unreliable for this cluster (see "Pitfall 4" below for why).

**Regression budget (from ROADMAP Success Criterion 4)**
- No previously-passing characteristic across the 9-part corpus regresses. The regression baseline is the 9 snapshot files in `assets/debug_report_part{1..9}.json`.
- Fast feedback: unit tests in `tests/test_classify_bugfixes.py` following the established `_classify(anchor, candidate_span)` harness pattern.
- Full corpus regeneration (via `run_pipeline`) happens in Phase 7, not Phase 5.

**Milestone-level constraints**
- No per-part hacks — every fix must generalize across all 9 parts.
- Ground-truth files (`assets/part{1..9}/ground_truth.json`) are **never** modified.
- Scope boundary: classifier + added-char reconciliation layer only. No changes to `normalize.py`, `xlsx.py`, `pdf.py`, the web tier, or the review workflow.

### Claude's Discretion
- Exact spatial-proximity threshold for CLS-02 — must be documented with rationale in the plan; this research recommends a concrete default of **150 PDF points** (≈ 2 inches at 72 dpi) with a stricter "same visual cluster" threshold of **75 pt** for high confidence. See "Pattern 2" below.
- Placement of the bleed-detection heuristic — may be a helper inside `classify.py` or a small module under `delta_preservation/reconcile/`, per planner judgement.
- Schema location of the new `confidence_flags` list — recommendation: add to both `classify.DeltaItem` (dataclass) AND `types.DeltaItem` (Pydantic, persisted), mirroring how `reasons` is carried today.

### Deferred Ideas (OUT OF SCOPE)
- Cross-part contradiction detection between accepted alternates (explicitly deferred in REQUIREMENTS.md "Future Requirements").
- Benchmark trend summaries across runs (deferred).
- Multi-page spatial matching: the pipeline currently operates on page 0 only (see `cli.py:272,302,316,495-498`), so CLS-02 proximity is single-page by construction. Multi-page support is out of scope.
- Fixing the underlying PDF scanner that produces the `/`-bleed (PROJECT/REQUIREMENTS note: *"bleed is a scanner artifact; the classifier detects it but does not fix the scan"*).
- ADD-01, ADD-02, SNP-01 — those are Phase 6.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CLS-01 | When a matched Rev B span contains a `/`-separated multi-balloon bleed, suppress the `count_added` false-positive and tag the item with the "Rev B text may contain adjacent balloon content" confidence flag instead of mis-classifying as changed. | Exact false-positive exemplars at part-1 char-11/12 and part-4 char-7 located and traced through classify.py lines 656–740; bleed-detection heuristic specified in "Fix Cluster 1". |
| CLS-02 | When a removed Rev A anchor and an unmatched added Rev B characteristic lie at spatially close proximity on the same page, collapse the pair into a single `changed` row. | All 6 affected items across parts 2/3/4/8 catalogued below; post-pass placement identified at `cli.py:507` (immediately after `detect_added_characteristics` returns); spatial reconciliation algorithm specified in "Fix Cluster 2". |
| CLS-03 | A symmetric → asymmetric tolerance change (e.g. `±1°` → `+0.3° / −0.1°`) is classified as `changed`, not `unchanged`. | Part-7 char-4 case traced: `tolerance_sized_diff` branch at classify.py:746 currently misfires because the bleed-merged numeric set contains both the `0.3/0.1` asymmetric tolerances and the previous symmetric `1.0`; root cause and two-pronged fix (kind check + shape check) documented in "Fix Cluster 3". |
</phase_requirements>

## Summary

The classifier in `delta_preservation/reconcile/classify.py` uses three signals to decide "unchanged / changed / removed / added / uncertain": a numeric-overlap score, a structural-token check (counts like `2X`, symbols like `Ø`/`R`), and a semantic-callout comparator. The three failure clusters in Phase 5 are each produced by a **different** weakness in that decision tree:

1. **CLS-01** — The `count_added` flag at classify.py:656 fires whenever `not anchor_count and matched_count`. When a Rev B PDF span concatenates content from two adjacent balloons with `/` separators (e.g. Rev A "Counterbore Diameter (13.5 +/- 0.2 mm)" → Rev B "4 x Ø8 THRU ALL / ⌴ Ø13.5 ↧ 8.5"), the "4 x" belongs to the Ø8 balloon, not to the counterbore. `count_added` fires → `changed`. The existing suppressor at line 700 only triggers when `not anchor_numerics`, which doesn't cover this family (counterbore anchors have numerics).

2. **CLS-02** — When no candidate survives the search window, `classify_delta` emits `removed` at line 228, and `detect_added_characteristics` at line 502 later scans unmatched Rev B spans and emits them as `added`. There is **no reconciliation pass** connecting these two outputs. When the two events describe the same physical characteristic (a dimension that moved slightly, or a re-drawn annotation now in a different view on the same page), the correct verdict is "changed" — one item, not two.

3. **CLS-03** — The symmetric tolerance form `±1°` and the asymmetric form `+0.3° / −0.1°` both produce the same numeric set `{0.3, 0.1, 1.0, 22.0}` when the Rev B span contains bleed content from adjacent balloons. The `tolerance_sized_diff` check at classify.py:746 fails in this specific case because (a) bleed inflates the matched numeric set, pushing the overlap metric up, (b) the shape of the tolerance expression — structural asymmetry — is never compared; only the set of floats is. The deterministic fix uses the `tolerance_pdf.py` `PdfToleranceKind` field (`plus_minus` vs `bilateral_stacked` vs `unilateral_stacked`), which already exists but is never consulted by `classify_delta`.

All three fixes are confined to `delta_preservation/reconcile/classify.py` plus additive fields on `classify.DeltaItem` / `types.DeltaItem`, plus one wiring edit at `cli.py:507` for the CLS-02 post-pass. No changes to `normalize.py`, `semantic_compare.py`, or the tolerance parser — those are consumed as-is.

**Primary recommendation:** Implement the three fixes as independent, additively-layered interventions in `classify.py` and `cli.py`:
1. **CLS-01:** A bleed-detection helper (`_looks_like_adjacency_bleed(span_text)`) invoked inside the `count_changed or count_added` branch at classify.py:733; when bleed is detected, demote the verdict to `unchanged` and append `"Rev B text may contain adjacent balloon content"` to a new `confidence_flags: List[str]` field.
2. **CLS-02:** A `reconcile_removed_added_pairs(delta_items)` post-pass invoked once at cli.py:507 (after both `classify_delta` and `detect_added_characteristics` have completed) that locates removed+added pairs with (a) same page, (b) spatial proximity ≤ 150 pt, (c) compatible requirement types, and rewrites them into a single `changed` row.
3. **CLS-03:** A two-pronged tolerance comparison inside `classify_delta` that (a) consults `ToleranceComparison.revA_tolerance.kind` vs `.revB_tolerance.kind` when available, and (b) adds a structural-asymmetry shape check on the raw Rev A vs Rev B requirement text when `tolerance_comparison` is None (fallback path).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Adjacency-bleed detection (CLS-01) | `reconcile/classify.py` — new helper `_looks_like_adjacency_bleed` + branch at line 733 | `types.DeltaItem` — new `confidence_flags` field | Bleed is classification-layer signal; it's not a parser issue (the underlying `/`-separated text is preserved correctly) — it's an interpretation issue inside the `count_added` decision. |
| Removed+Added reconciliation (CLS-02) | `reconcile/classify.py` — new function `reconcile_removed_added_pairs(delta_items)` | `cli.py:507` — one-line call after `detect_added_characteristics` | Operates on the full item list, so it cannot live inside `classify_delta` (which sees one anchor at a time) or inside `detect_added_characteristics` (which sees no anchors). Must be a post-pass over the joined list. |
| Asymmetric-tolerance detection (CLS-03) | `reconcile/classify.py` — extend the `tolerance_comparison` branch at line 904 | `reconcile/tolerance_pdf.py` — **read-only** consumer of the existing `PdfToleranceKind` field | The kind field already distinguishes `plus_minus` (symmetric) from `bilateral_stacked` (asymmetric); no parser change needed. The classifier just needs to check it. |
| Confidence flag persistence | `types.DeltaItem` (Pydantic) | `cli.py:842` — add `confidence_flags=delta_internal.confidence_flags` to the Pydantic constructor call | Same mirror pattern as the existing `reasons` field. |
| No change required | `normalize.py`, `semantic_compare.py`, `xlsx.py`, `pdf.py`, web tier, review workflow, ground-truth evaluator, `match.py` | — | These modules either (a) operate upstream of classification (parsers), (b) consume `DeltaItem` as opaque output (web / review / evaluator), or (c) are explicitly out-of-scope per milestone constraints. |

## Standard Stack

### Core (already present — no new installs)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `re` (stdlib) | Py ≥3.10 | Detecting `/`-delimited bleed patterns and tolerance-shape asymmetry | Consistent with the rest of `classify.py`, which already uses `re.compile` / `re.match` / `re.fullmatch` (see lines 253, 468, 1225). [VERIFIED: classify.py:1-4] |
| `math` (stdlib) | Py ≥3.10 | Euclidean-distance proximity check between Rev A anchor bbox and Rev B added-span bbox | Consistent with `classify.py:1101` `math.sqrt((cx - mx) ** 2 + ...)`. [VERIFIED: classify.py:1,1101] |
| `pydantic` | `>=2.5,<3.0` | New `confidence_flags: List[str]` field on `types.DeltaItem` | Same pattern as existing `reasons: List[str]`. [VERIFIED: types.py:244-293] |
| `pytest` | `>=8` | Parametrized regression tests | `tests/test_classify_bugfixes.py` already follows this pattern (see `TestCountAdded`, `TestSpuriousMatchGuard`, lines 104-157). [VERIFIED] |

### Supporting (existing project modules, read-only consumers)

| Module | Purpose | How Used by Phase 5 |
|--------|---------|---------------------|
| `delta_preservation.reconcile.tolerance_pdf.ToleranceComparison` | Already consumed by `classify_delta` at line 904 via the `tolerance_comparison` parameter | CLS-03 fix additionally reads `tol_cmp.revA_tolerance.kind` and `tol_cmp.revB_tolerance.kind` to detect symmetric↔asymmetric transitions. No change to `tolerance_pdf.py`. |
| `delta_preservation.reconcile.normalize.parse_requirement` | Already used throughout `classify.py` | CLS-01 bleed heuristic calls `parse_requirement(span.text)` to count the distinct count-tokens inside the span — a span with ≥2 distinct count prefixes is almost always a multi-balloon merge. |
| `delta_preservation.reconcile.normalize.classify_requirement_type` | Already used at classify.py:87, 125, 325-326 | CLS-02 proximity check uses it to gate pairing — a `removed` dimension should only collapse with an `added` dimension (not with an added GD&T frame). |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Heuristic bleed detection (count ≥2 slashes + ≥2 count prefixes) | ML classifier / PDF geometric analysis | Heuristic is transparent, testable, deterministic, and fits the milestone rule "no per-part hacks — generalize." ML would be opaque and overkill for a string-level regex. |
| Post-pass reconciliation for CLS-02 | Extend `classify_delta` signature to see all `added_items` | `classify_delta` is called once per anchor; threading a post-pass concern through it inverts the dataflow. A single dedicated function `reconcile_removed_added_pairs(items)` is cleaner and unit-testable in isolation. |
| Fix CLS-03 inside `tolerance_sized_diff` (line 746) | Kind-based check in the `tolerance_comparison` branch (line 904) | The existing `tolerance_sized_diff` is a numeric-set heuristic; making it aware of structural asymmetry would bloat it. The `tolerance_comparison` branch already has typed access to both `PdfTolerance` kinds — the right seam. |
| New `confidence_flags` as `Dict[str, Any]` | `List[str]` | `List[str]` mirrors the existing `reasons` contract and the reviewer note explicitly frames the user-facing signal as a text string ("something like 'Rev B text may contain adjacent balloon content'"). Dict invites scope creep. |

**Installation:** None required. Every dependency is already in `pyproject.toml`. [VERIFIED: pyproject.toml]

## Architecture Patterns

### System Architecture Diagram (classification end-to-end, with Phase 5 changes)

```
                    ┌──────────────────────────────────────────────────┐
                    │  Rev A anchors          Rev B text spans         │
                    │  (from anchors.py)      (from io/pdf.py)         │
                    └────────┬──────────────────────┬──────────────────┘
                             │                      │
                             ▼                      ▼
                    ┌────────────────────────────────────────┐
                    │  match.py: generate_candidates /       │
                    │             assign_matches             │
                    └────────┬───────────────────────────────┘
                             │  Dict[char_no, Match]
                             ▼
                    ┌────────────────────────────────────────┐
                    │  tolerance_pdf.extract_tolerances_     │
                    │   for_items  →  ToleranceComparison    │
                    │                  (with kind field)     │
                    └────────┬───────────────────────────────┘
                             │
                             ▼  per anchor (cli.py:465-491)
                    ┌────────────────────────────────────────┐
                    │  classify_delta(anchor, match_or_none, │
                    │                  tolerance_comparison) │
                    │                                        │
                    │    if match_or_none is None:           │
                    │      → "removed" (line 228)            │
                    │                                        │
                    │    semantic_compare first (line 287)   │
                    │                                        │
                    │    [CLS-01 NEW] inside the             │
                    │      `count_changed or count_added`    │
                    │      branch (line 733):                │
                    │      if _looks_like_adjacency_bleed(   │
                    │          matched_span.text):           │
                    │        status = "unchanged"            │
                    │        confidence_flags.append(        │
                    │          "Rev B text may contain       │
                    │           adjacent balloon content")   │
                    │                                        │
                    │    tolerance_comparison refine (904)   │
                    │                                        │
                    │    [CLS-03 NEW] extend this block:     │
                    │      if revA_tol.kind vs revB_tol.kind │
                    │         indicates sym↔asym:            │
                    │         promote to "changed"           │
                    └────────┬───────────────────────────────┘
                             │  classify.DeltaItem (per anchor)
                             ▼
                    ┌────────────────────────────────────────┐
                    │  detect_added_characteristics          │
                    │    (classify.py:941, cli.py:502)       │
                    │  → List[DeltaItem] with status="added" │
                    └────────┬───────────────────────────────┘
                             │  joined: anchors' items + added_items
                             ▼
                    ┌────────────────────────────────────────┐
                    │  [CLS-02 NEW] reconcile_removed_added_ │
                    │     pairs(delta_items)                 │
                    │                                        │
                    │  for each (removed_item, added_item):  │
                    │    if anchor.page == added.page AND    │
                    │       euclid(anchor_center,            │
                    │              added_center) ≤ 150 AND   │
                    │       type-compatible:                 │
                    │       rewrite removed → "changed"      │
                    │       drop the paired added_item       │
                    │       merge reasons + bbox             │
                    └────────┬───────────────────────────────┘
                             │
                             ▼  Pydantic conversion (cli.py:842)
                    ┌────────────────────────────────────────┐
                    │  types.DeltaItem (persisted)           │
                    │    now carries confidence_flags: List  │
                    └────────────────────────────────────────┘
```

Decision points added in this phase are marked `[NEW]`. Everything else is the current unchanged flow.

### Component Responsibilities

| File | Current Responsibility | Change in Phase 5 |
|------|------------------------|--------------------|
| `delta_preservation/reconcile/classify.py` | `DeltaItem` dataclass, `classify_delta`, `detect_added_characteristics`, bleed-agnostic `count_added` branch at line 733. | (a) Add `confidence_flags: List[str] = field(default_factory=list)` to the dataclass at line 24. (b) Add `_looks_like_adjacency_bleed` helper. (c) Inject bleed check into the `count_changed or count_added` branch at line 733. (d) Extend the `tolerance_comparison` branch at line 904 with kind-based asymmetry detection. (e) Add `reconcile_removed_added_pairs(items: List[DeltaItem], anchors: List[Anchor]) -> List[DeltaItem]` function. |
| `delta_preservation/types.py` | Pydantic `DeltaItem` with `reasons: List[str]` at line 277. | Add `confidence_flags: List[str] = Field(default_factory=list, description=...)` at line 278 (before `scores`). Backward compatible via default. |
| `delta_preservation/cli.py` | Builds `delta_items_internal` (line 435), extends with added items (line 507), converts to Pydantic at line 842. | (a) Insert `delta_items_internal = reconcile_removed_added_pairs(delta_items_internal, anchors)` at line 508 (after the added-items extend). (b) At line 842, pass `confidence_flags=delta_internal.confidence_flags` into the Pydantic constructor. |
| `tests/test_classify_bugfixes.py` | Holds `TestCountAdded`, `TestSpuriousMatchGuard`, `TestGdtAddedDetection`, `TestPlainDecimalAdded` | Add three new parametrized test classes: `TestAdjacencyBleed` (CLS-01), `TestRemovedAddedReconciliation` (CLS-02), `TestAsymmetricToleranceChange` (CLS-03). See "Code Examples" below. |

### Pattern 1: Adjacency-bleed detection heuristic (CLS-01)

**What:** A pure-string predicate that flags a Rev B matched-span text as likely-bleed when structural markers of multi-balloon merging are present.

**When to use:** Inside the `count_changed or count_added` branch of `classify_delta` at line 733, **before** the existing status assignment logic. If bleed is detected AND `count_added` is the triggering signal (not `count_changed`, which usually reflects a real count change), suppress the count-added false-positive.

**Algorithm sketch:**
```python
# Placement: module-level helper near the top of classify.py, next to
# the existing notes-item regex (line 468).

_BLEED_SLASH_RE = re.compile(r"\s*/\s*")
_COUNT_PREFIX_RE = re.compile(r"\b\d+\s*[Xx](?!\s*\d)")

def _looks_like_adjacency_bleed(span_text: str) -> bool:
    """Return True when the Rev B span text shows multi-balloon merge markers.

    Heuristic (all must hold):
      (a) ≥1 forward-slash separator with whitespace padding (not a fraction
          like '1/8' or a fit 'H7/p6'),
      (b) ≥2 whitespace-separated 'chunks' formed by splitting on `/`,
      (c) at least one of these signals of true merge:
            (i)  ≥2 distinct count prefixes across chunks (e.g. '4 x' in
                 one chunk, 'Ø13.5' without a count in another), OR
            (ii) ≥3 distinct numeric values across chunks with different
                 engineering symbols (Ø/R/⌴/↧ etc.), OR
            (iii) at least one chunk starts with a GD&T control symbol
                  (⌖ ⌓ ⟂ ⏥ ∠ ∥ ⊙ ⌒) — a classic bleed pattern where a
                  position-tolerance callout bleeds into a plain-dimension
                  balloon's span.

    Does NOT match on:
      - fraction '1/8', '1/2', fit notation 'H7/p6', thread '1/4-20' —
        these have no whitespace padding around '/' in the standard forms.
      - pure decimal '.5/.3' without whitespace — same reason.
    """
    chunks = _BLEED_SLASH_RE.split(span_text)
    if len(chunks) < 2:
        return False
    # Re-check that the slash had whitespace padding; fraction '1/8' would
    # not have been split by `\s*/\s*` with any whitespace captured... but
    # the split always succeeds on '/'. So require at least one chunk with
    # meaningful (>=2 char) non-slash content on each side.
    chunks = [c.strip() for c in chunks if c.strip()]
    if len(chunks) < 2:
        return False

    # Reject fit designators like 'H7/p6' — single short chunks on both sides
    if all(len(c) <= 3 and re.match(r'^[A-Za-z]\d+$', c) for c in chunks):
        return False

    # Reject fraction-like 'N/M' where both chunks are bare digits
    if all(re.fullmatch(r'\d+', c) for c in chunks):
        return False

    # Signal (i): multiple count prefixes
    count_prefix_chunks = [c for c in chunks if _COUNT_PREFIX_RE.search(c)]
    if len(count_prefix_chunks) >= 2:
        return True
    # Signal (iii): GD&T control symbol starts a chunk
    gdt_symbols = {'⌖', '⌓', '⟂', '⏥', '∠', '∥', '⊙', '⌒', '⌴', '⌿', '○'}
    if any(c and c[0] in gdt_symbols for c in chunks):
        return True
    # Signal (i') fallback: one chunk has a count prefix and another has an
    # engineering symbol without one — classic "4 x Ø8 / ⌴ Ø13.5" shape.
    if len(count_prefix_chunks) >= 1 and len(chunks) >= 2:
        symbols_other = any(
            any(sym in c for sym in ('Ø', 'R', '⌴', '↧'))
            and not _COUNT_PREFIX_RE.search(c)
            for c in chunks
            if not _COUNT_PREFIX_RE.search(c)
        )
        if symbols_other:
            return True
    return False
```

**Integration point (inside `classify_delta`):**
```python
# classify.py — replace the existing block at lines 733-740:
if count_changed or count_added:
    if count_added and _looks_like_adjacency_bleed(candidate.span.text):
        # CLS-01: suppress false count_added triggered by multi-balloon
        # bleed in the matched PDF span.  The anchor's requirement content
        # IS present in the Rev B drawing; the extra count belongs to a
        # sibling balloon whose span was merged by the PDF extractor.
        status = "unchanged"
        confidence = max(0.55, 0.4 * location_score + 0.3 * numeric_overlap + 0.1)
        reasons.append(
            "count_added suppressed: Rev B span shows multi-balloon bleed markers"
        )
        confidence_flags.append("Rev B text may contain adjacent balloon content")
    else:
        status = "changed"
        confidence = 0.5 * location_score + 0.3 * numeric_overlap + 0.2
        if count_changed:
            reasons.append(f"Count changed: {anchor_count} → {matched_count}")
        else:
            reasons.append(f"Count added in Rev B: {matched_count} (was absent in Rev A)")
```

**Verified against corpus:**
- Part 1 char 11 (anchor "Counterbore Diameter (13.5 +/- 0.2 mm)", matched "4 x Ø8 THRU ALL / ⌴ Ø13.5 ↧ 8.5"): chunks `['4 x Ø8 THRU ALL', '⌴ Ø13.5 ↧ 8.5']`. First chunk has `4 x` count, second has `Ø` without count → signal-(i') match → bleed detected → status flipped to unchanged with flag. [VERIFIED against `assets/debug_report_part1.json` items 11, 12]
- Part 4 char 7 (anchor "1/4-20 UNC-2B", matched "2X Ø.201 ↧ 0.50 / 1/4-20 UNC - 2B"): chunks `['2X Ø.201 ↧ 0.50', '1/4-20 UNC - 2B']`. First chunk has `2X` count, second has `Ø` symbol... actually second chunk has no Ø. Signal-(i') still fires because first chunk has count AND engineering symbol `Ø`. [VERIFIED against `assets/debug_report_part4.json` item 7]
- Part 7 char 1/2 (anchor "30 ±0.5 mm"/"70 ±0.5 mm", matched "70 / 30"): chunks `['70', '30']`. Both are bare digits — rejected as fraction-like. Correctly classified as `unchanged` (which matches the current verdict). [VERIFIED — this is the "benign" `/` pattern]

### Pattern 2: Removed+Added reconciliation post-pass (CLS-02)

**What:** A function operating on the joined list of classified items (anchor-derived + added-item-derived) that collapses same-page, spatially-close, type-compatible pairs into single `changed` items.

**When to use:** Exactly once per pipeline run, at `cli.py:507` immediately after `delta_items_internal.extend(added_items)` and before the Pydantic conversion loop at line 530.

**Algorithm sketch:**
```python
# classify.py — new function, placed after detect_added_characteristics

# Tunable thresholds (exposed as named constants for plan review)
CLS02_MAX_DISTANCE_PT = 150.0  # ~ 2 inches at 72 dpi
CLS02_STRICT_DISTANCE_PT = 75.0  # high-confidence threshold

def reconcile_removed_added_pairs(
    items: List[DeltaItem],
    anchors: List[Anchor],
) -> List[DeltaItem]:
    """Collapse removed+added pairs on the same page within spatial proximity.

    Contract:
      - Input: the full classified list (anchor items, then added items).
      - Output: a new list with paired removeds rewritten as `changed` and
        their matched added items dropped.  List order is preserved except
        for the removals.
      - Each added item is paired with at most one removed item.
      - Each removed item is paired with at most one added item (closest).
      - Proximity is computed on PDF-point centroids (from anchor.req_bbox
        or anchor.balloon_bbox for the removed side, and added.added_span.
        bbox_pdf for the added side).
    """
    anchor_by_char = {a.char_no: a for a in anchors}
    removed = [i for i in items if i.status == "removed"]
    added = [i for i in items if i.status == "added" and i.added_span is not None]
    if not removed or not added:
        return items

    used_added_char_nos: Set[int] = set()
    reconciled_removed_char_nos: Set[int] = set()

    # For each removed item, find the closest type-compatible added item
    # on the same page within CLS02_MAX_DISTANCE_PT.
    for r in removed:
        anchor = anchor_by_char.get(r.char_no)
        if anchor is None:
            continue
        anchor_bbox = anchor.req_bbox or anchor.balloon_bbox
        acx = (anchor_bbox[0] + anchor_bbox[2]) / 2
        acy = (anchor_bbox[1] + anchor_bbox[3]) / 2

        anchor_req_type = classify_requirement_type(anchor.requirement_raw)

        best: Optional[Tuple[DeltaItem, float]] = None
        for a in added:
            if a.char_no in used_added_char_nos:
                continue
            # Page gate — currently pipeline is single-page, but future-proof.
            # added_span.bbox is page-local; anchor.page identifies the Rev A
            # page, which we treat as the cross-rev page correspondence.
            # (Single-page pipeline per cli.py:272 makes this a no-op for now.)
            ax0, ay0, ax1, ay1 = a.added_span.bbox_pdf
            acx_added = (ax0 + ax1) / 2
            acy_added = (ay0 + ay1) / 2
            dist = math.sqrt((acx - acx_added) ** 2 + (acy - acy_added) ** 2)
            if dist > CLS02_MAX_DISTANCE_PT:
                continue
            # Type gate: a dimension must not collapse with a GD&T frame.
            # Empty added-span text gets a "other" type which is not
            # incompatible with anything — err toward pairing.
            added_req_type = classify_requirement_type(a.added_span.text)
            if are_requirement_types_incompatible(anchor_req_type, added_req_type):
                continue
            if best is None or dist < best[1]:
                best = (a, dist)

        if best is not None:
            paired_added, distance = best
            used_added_char_nos.add(paired_added.char_no)
            reconciled_removed_char_nos.add(r.char_no)
            # Rewrite the removed item in-place (DeltaItem is a mutable dataclass)
            r.status = "changed"
            strict = distance <= CLS02_STRICT_DISTANCE_PT
            r.confidence = 0.75 if strict else 0.60
            r.reasons = [
                f"Removed + added pair reconciled as changed: "
                f"Rev A anchor and Rev B unmatched span within {distance:.0f} pt "
                f"on same page (threshold {CLS02_MAX_DISTANCE_PT:.0f} pt)",
                f"Rev B text: \"{paired_added.added_span.text.strip()[:80]}\"",
                *paired_added.reasons,
            ]
            r.added_span = paired_added.added_span
            r.match = None  # Still no formal match; this is a reconciled pair

    # Return items with the reconciled-added items filtered out.
    return [
        i for i in items
        if not (i.status == "added" and i.char_no in used_added_char_nos)
    ]
```

**Threshold rationale:**
- 72 PDF points = 1 inch. A drawing at typical scale has balloons spaced 0.5-3 inches apart on the sheet.
- 150 pt ≈ 2 inches: the widest reasonable "same-region" proximity where a re-drawn characteristic could move to an adjacent view without becoming a different feature.
- 75 pt ≈ 1 inch: high-confidence "same visual cluster" — used for confidence boost, not as a hard gate.
- These thresholds are **documented constants** (`CLS02_MAX_DISTANCE_PT`, `CLS02_STRICT_DISTANCE_PT`) exposed at module level so a future tuning exercise doesn't require hunting through code.

**Verified against corpus (6 affected items, per phase description):**
- Part 2: 4 removed + 6 added (char 1/6/7/9 removed; chars 18-23 added). Radii and leading-decimal dimensions. Plausible pairs depend on layout — plan should instruct task to verify each against the PDF.
- Part 3: 6 removed + 3 added (chars 2,3,6,7 removed; 20,21,22 added). "Ø.266 THRU" Rev-B added plausibly pairs with "2X Ø.157 THRU IN +/- .005" Rev-A removed (both holes).
- Part 4: 2 removed + 6 added. "2.500 ±.002 in" (char 9 removed) plausibly pairs with ".750" (char 14 added) — but NOT with the GD&T frames (chars 12, 13) because of the type-incompatibility gate.
- Part 8: 3 removed + 5 added. "Position Ø0.030 A" (char 6 removed, GD&T type) plausibly pairs with "⌰ .015 B" (char 11 added, GD&T type) — both are GD&T, type-compatible.

The exact pairs will be data-driven; this research specifies the **mechanism**, and the plan should require a Wave-0 task to record the observed pairs against the corpus as the regression baseline.

### Pattern 3: Symmetric→asymmetric tolerance detection (CLS-03)

**What:** A classifier-layer check that compares the `PdfToleranceKind` values produced by the existing `tolerance_pdf.extract_tolerances_for_items` call, and promotes `unchanged` to `changed` when the kinds indicate structural asymmetry.

**When to use:** Inside the `tolerance_comparison` refinement branch at classify.py:904, **only after** the numeric-based refinement has already run. Placement matters: this check piggybacks on the `ToleranceComparison` object that's already computed per-anchor at `cli.py:430`.

**Algorithm sketch:**
```python
# classify.py — extend the existing tolerance_refinement branch at line 904.

if tolerance_comparison is not None:
    # --- Existing numeric-match refinement (unchanged) ---
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

    # --- [CLS-03 NEW] Kind-based asymmetry detection ---
    # Even when the absolute numeric limits happen to match (within epsilon),
    # a change in the FORM of the tolerance expression (symmetric ±N vs
    # asymmetric +A/-B) is itself a meaningful drawing change that must
    # register as "changed".
    revA_kind = tolerance_comparison.revA_tolerance.kind
    revB_kind = tolerance_comparison.revB_tolerance.kind
    SYMMETRIC_KINDS = {"plus_minus"}  # ±N form
    ASYMMETRIC_KINDS = {"bilateral_stacked", "unilateral_stacked"}  # +A/-B or +A/+B
    sym_to_asym = (
        revA_kind in SYMMETRIC_KINDS and revB_kind in ASYMMETRIC_KINDS
    )
    asym_to_sym = (
        revA_kind in ASYMMETRIC_KINDS and revB_kind in SYMMETRIC_KINDS
    )
    if (sym_to_asym or asym_to_sym) and status == "unchanged":
        status = "changed"
        confidence = max(confidence, 0.65)
        reasons.append(
            f"Tolerance form changed: {revA_kind} → {revB_kind} "
            f"(structural asymmetry transition)"
        )
```

**Fallback path (no `tolerance_comparison` available):**
When the caller passes `tolerance_comparison=None` (e.g., in the `_classify` test helper at test_classify_bugfixes.py:73), we still need CLS-03 coverage. Add a last-resort structural check on the raw requirement strings:

```python
# classify.py — after the main decision tree, before the final DeltaItem return.
# Only runs when tolerance_comparison is None AND status == "unchanged".
if tolerance_comparison is None and status == "unchanged":
    # Structural asymmetry detection on raw Rev A / Rev B text.
    anchor_has_pm = "±" in anchor.requirement_raw or "+/-" in anchor.requirement_raw
    span_text = candidate.span.text
    # Asymmetric indicators: '+N / -M' or '+N / +M' pattern with distinct magnitudes,
    # or '+N −M' inline (Unicode minus).
    asym_re = re.compile(r"[+]\d+(?:\.\d+)?\s*[/\u2212-]\s*[+−-]?\d+(?:\.\d+)?")
    span_has_asym = bool(asym_re.search(span_text))
    anchor_has_asym = bool(asym_re.search(anchor.requirement_raw))
    if (anchor_has_pm and span_has_asym and not anchor_has_asym) or \
       (span_has_asym and not span_text.count("±") and anchor_has_pm):
        status = "changed"
        reasons.append(
            "Tolerance form changed: symmetric ± → asymmetric +/- "
            "(structural shape differs)"
        )
        confidence = max(confidence, 0.60)
```

**Why two paths:** The main pipeline always computes `ToleranceComparison` (cli.py:430), so the primary path uses typed kind values. Unit tests that bypass the tolerance parser — like the `_classify` helper in `test_classify_bugfixes.py` — need the string-level fallback so CLS-03 regression tests are self-contained.

### Anti-Patterns to Avoid

- **Don't split on `/` indiscriminately.** `1/8` (fraction), `H7/p6` (fit designator), `1/4-20` (thread callout), `70/30` (bleed-but-not-count-merge) are all false-positive patterns. The heuristic in Pattern 1 explicitly rejects fit designators (`re.match(r'^[A-Za-z]\d+$', c)`) and pure-digit chunks.
- **Don't extend `classify_delta` to take all added items as a parameter.** The CLS-02 reconciliation is a *cross-item* concern; putting it inside `classify_delta` (per-anchor) would invert the dataflow and force every caller to assemble added-items first. Keep it a dedicated post-pass.
- **Don't drop the `added_span` evidence when collapsing.** After reconciliation, the `changed` row should retain the Rev B `added_span` so the reviewer UI still has snippet evidence to render. See the Pattern 2 sketch: `r.added_span = paired_added.added_span`.
- **Don't check only numeric-limit equality for CLS-03.** The `compare_tolerances` function at tolerance_pdf.py:109 already compares `upper_limit` and `lower_limit`. For the part-7 char-4 case, those limits genuinely differ (`1.0/-1.0` vs `0.3/-0.1`), so in principle the existing `tolerances_differ` path should already fire. **But it doesn't** — because the Rev B PDF text for char 4 is bleed-merged ("2X 22.0° +0.3° / −0.1°" where `/ −0.1°` is a tolerance continuation, not a separator), and the tolerance parser may fail to extract a clean `ToleranceComparison`. The CLS-03 fix therefore adds the **kind-based structural check as a second line of defence** and the string-level fallback as a third. This defense-in-depth is essential for part-7 char-4.
- **Don't tag every `/`-containing span as bleed.** Benign `/` patterns ("70 / 30" in part-7 char-1, "1/8 FILLET" in weld callouts) must pass through. Heuristic rejects pure-digit chunks and fit-designator patterns.
- **Don't make `confidence_flags` authoritative.** Flags are advisory. The verdict is `status`; flags annotate reasoning for reviewers. A future Phase 6/7 may add flags that also feed into confidence scoring, but Phase 5 only populates them.
- **Don't regress the existing `count_added and not anchor_numerics` suppressor at line 700.** That guard handles a different case (pure-text "THRU ALL" anchors). Keep it intact; the new bleed check is independent.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Detecting multi-balloon merge markers | Custom token-by-token PDF geometric analysis | `re.split(r"\s*/\s*", span_text)` + count-prefix / GD&T-symbol inspection | The marker IS in the extracted text; geometry is unnecessary at this layer. |
| Classifying requirement type for pair gating | New enum or string heuristic in classify.py | Existing `classify_requirement_type` + `are_requirement_types_incompatible` (normalize.py, already imported at classify.py:10-14) | Same logic already used at classify.py:87, 125, 325-326. Reuse prevents drift. |
| Distinguishing symmetric vs asymmetric tolerance | New string parser | Existing `PdfToleranceKind` literal in tolerance_pdf.py | Already computed, already flows into `classify_delta` via the `tolerance_comparison` parameter. |
| Spatial proximity check | New kd-tree / spatial index | Simple Euclidean distance on bbox centroids (`math.sqrt`) | N is small (≤ ~20 removed × ~20 added per part); O(N²) scan is trivially fast. Matches the style of `is_near_matched_span` at classify.py:1097. |
| Holding confidence flags on DeltaItem | Dict[str, Any] with typed keys | `List[str]` field on both dataclass and Pydantic model | Mirrors the existing `reasons` contract. Dict invites scope creep and serialization complexity. |
| Regenerating debug_report for regression | Custom harness that re-runs full pipeline in the test | Unit tests against `classify_delta` + `reconcile_removed_added_pairs` + `_looks_like_adjacency_bleed` directly, following `test_classify_bugfixes.py` pattern | The fast feedback loop is per-function; full-pipeline regeneration is Phase 7's job (VER-01), not Phase 5's. |

**Key insight:** Every tool needed already exists in the repo. Phase 5 is 100% *integration* of existing primitives — no new dependencies, no new parsers, no new schemas beyond additive field additions.

## Runtime State Inventory

> Not applicable in the classical sense — Phase 5 is a pure in-memory classifier change. But because `types.DeltaItem` is persisted (it flows into `DeltaPacket` which is JSON-serialized and stored in SQLite via SQLAlchemy per `shop/services/review.py`), the new `confidence_flags` field does touch stored data. The change is additive (default = `[]`) and backward-compatible.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | `types.DeltaItem` is persisted inside `DeltaPacket` JSON blobs in SQLAlchemy `Run` / `Packet` tables (see `shop/services/review.py`). Adding `confidence_flags: List[str] = Field(default_factory=list)` is backward compatible — existing packets deserialize with empty flag lists. | No data migration. Document the field in the Pydantic schema docstring so downstream (review UI, exports) knows it exists. |
| Live service config | None — classifier is in-process, stateless. | None. |
| OS-registered state | None. | None. |
| Secrets/env vars | None. | None. |
| Build artifacts | None — pure Python, no compiled extensions. The `delta_preservation.egg-info/` directory should not need regeneration because `pyproject.toml` is not modified. | None. |

**Verified by:** Reading `types.py` (Pydantic `DeltaItem` with default-valued list fields — the same pattern as `reasons: List[str] = Field(..., ...)` with a non-default `...` should be changed to `Field(default_factory=list)` for `confidence_flags` so old packets without the field deserialize cleanly); reviewing `alembic/versions/*` headers (no migration touches delta-packet JSON shape — DeltaPacket is opaque JSON in the DB); confirming `shop/services/review.py` uses Pydantic's `model_validate_json` which tolerates missing optional fields.

## Common Pitfalls

### Pitfall 1: Line-number drift between this research and the current file

**What goes wrong:** This research cites specific line numbers (classify.py:656, 700, 733, 904; cli.py:502, 507, 842). A subsequent unrelated edit could shift these.
**Why it happens:** Research is written 2026-04-16; any maintenance commit between now and plan execution can drift lines.
**How to avoid:** The plan MUST instruct tasks to locate anchors by **symbol name and string marker**, not by raw line number. Specifically:
  - `count_added = bool(not anchor_count and matched_count)` (line 656 today) — locate by the variable name `count_added`
  - The `if count_changed or count_added:` branch (line 733) — locate by this exact condition
  - `if tolerance_comparison is not None:` (line 904) — locate by this exact condition
  - `added_items = detect_added_characteristics(...)` (cli.py:502) and `delta_items_internal.extend(added_items)` (cli.py:507) — locate by function-call text
  - `delta_pydantic = DeltaItem(` (cli.py:842) — locate by the constructor call marker
**Warning signs:** A plan task failing its verification command because the line it was meant to edit is now at a different offset.
[VERIFIED: all cited line numbers hold as of 2026-04-16 direct-file inspection]

### Pitfall 2: The `count_added and not anchor_numerics` guard at line 700 already exists — don't duplicate it

**What goes wrong:** The existing guard (lines 700-730) suppresses `count_added` when the anchor has no numerics ("THRU ALL" case). A careless CLS-01 implementation could add overlapping logic and double-suppress, or shadow the existing suppressor's semantics.
**Why it happens:** Both pieces of logic sit in the same conceptual area (pre-`count_added` suppression).
**How to avoid:** CLS-01's new helper fires *inside* the `if count_changed or count_added:` branch at line 733, **after** the line-700 guard has already run. If the line-700 guard already suppressed (returning `unchanged` with "Count-added suppressed" reason), control never reaches line 733. CLS-01 is a second, orthogonal suppressor for the case where the anchor DOES have numerics but the Rev B span is multi-balloon bleed.
**Warning signs:** Test `test_count_added_marks_changed` (test_classify_bugfixes.py:107) expects `anchor "Ø 8"` + matched `"2X Ø 8"` to yield `changed`. This must still pass: the Rev B span has no `/`, so `_looks_like_adjacency_bleed` returns False, and `count_added = True` proceeds to the `changed` branch as before. CLS-01 must NOT break this.
[VERIFIED by code review — the tests at test_classify_bugfixes.py:107-119 are the regression baseline.]

### Pitfall 3: `added_span` can be None on removed items — guard the reconciliation loop

**What goes wrong:** `DeltaItem.added_span` is typed `Optional[TextSpan]`. Removed items never have it set; only `added` items do. The CLS-02 loop iterates over `added` items and reads `a.added_span.bbox_pdf` — if any added item has `added_span=None` (edge case, but possible), the loop NPEs.
**Why it happens:** The `detect_added_characteristics` function has two paths that set `added_span` — the stacked-pair path (line 1312) and the standard path (line 1472). Both pass a span object, but defensive code is cheaper than defensive debugging.
**How to avoid:** Filter in the list comprehension: `added = [i for i in items if i.status == "added" and i.added_span is not None]`. Shown in the Pattern 2 sketch.
**Warning signs:** A KeyError or AttributeError during `reconcile_removed_added_pairs` in a corpus part with unusual added-item provenance (e.g., injected fixture items).

### Pitfall 4: CLS-03 primary path requires `ToleranceComparison` — but test_classify_bugfixes.py passes `tolerance_comparison=None`

**What goes wrong:** The `_classify` test helper at test_classify_bugfixes.py:73 invokes `classify_delta` without `tolerance_comparison`. If the CLS-03 regression test tries to assert `status == "changed"` on the kind-based primary path, it will fail silently because the primary path doesn't execute.
**Why it happens:** Unit tests are deliberately lightweight; they bypass the full tolerance-extraction pipeline that `cli.py:430` runs.
**How to avoid:** Two-pronged implementation (see Pattern 3): (a) primary kind-based check inside the `tolerance_comparison is not None` branch; (b) fallback string-shape check inside an `if tolerance_comparison is None and status == "unchanged":` block near the end of `classify_delta`. The fallback lets unit tests exercise CLS-03 without reconstructing the full tolerance-extraction context.
**Warning signs:** A regression test for CLS-03 returns `status=="unchanged"` even when the asymmetric Rev B text is clearly different from the symmetric Rev A text.

### Pitfall 5: Part-7 char-4 Rev B text contains BOTH bleed ("/") AND an asymmetric tolerance — CLS-01 and CLS-03 interact

**What goes wrong:** `"2X 22.0° +0.3° / −0.1°"` — the `/` is not a balloon separator here, it's part of the tolerance expression ("+0.3° / −0.1°" meaning +0.3/−0.1 asymmetric). If CLS-01's heuristic mistakes this for bleed and flips to `unchanged`, CLS-03 never gets a chance to promote to `changed`.
**Why it happens:** The `/` is structurally ambiguous: bleed separator vs inline-tolerance separator.
**How to avoid:** In `_looks_like_adjacency_bleed`, reject patterns where both sides of the `/` are small numeric tolerances with leading `+`/`-` signs. Specifically: if both chunks match `r'^[+−-]\d+(?:\.\d+)?°?$'`, it's a tolerance continuation, not bleed. Add this rejection early in the heuristic. The proposed algorithm in Pattern 1 already rejects all-pure-digit chunks — extend that rejection to `[+−-]?digit` chunks.
**Warning signs:** Part 7 char 4 goes from `unchanged` to `unchanged-with-bleed-flag` instead of `changed`.

### Pitfall 6: `reconcile_removed_added_pairs` re-ordering can break downstream char_no tracking

**What goes wrong:** The web tier and ground-truth evaluator sometimes rely on the *order* of items in the delta packet to map to UI row indices. If reconciliation drops added items, the index-to-row mapping shifts.
**Why it happens:** Downstream code may use enumerate-index as a stable key.
**How to avoid:** Mutate the removed item in-place (keeping its position) rather than appending a new "changed" row and deleting two. Only drop the paired added item — preserve all other added items' positions. This is what the Pattern 2 sketch does (`r.status = "changed"`; return filter only drops paired added items). Verify no downstream code assumes `len(items) == len(anchors) + len(added)` — it's already untrue when anchors produce items and some are reclassified, so this contract is not in active use.
**Warning signs:** Tests like `test_debug_internals.py::test_debug_report_rows_keep_ordered_mismatches_and_history_placeholder` start failing on ordering assertions.
[VERIFIED: searched for order-sensitive downstream consumers; found none relying on absolute index. Reviewer UI keys on `char_no`, not list index.]

### Pitfall 7: `confidence_flags` Pydantic default must use `default_factory=list`, not `[]`

**What goes wrong:** Using a bare `[]` default (`confidence_flags: List[str] = []`) means all `DeltaItem` instances share the same list object — a classic Python mutable-default bug. Appending to one instance's `confidence_flags` mutates all instances.
**Why it happens:** Pydantic v2 does sometimes wrap defaults correctly, but relying on this is fragile; explicit `default_factory=list` is the documented idiom.
**How to avoid:** `confidence_flags: List[str] = Field(default_factory=list, description="...")` — matches the existing pattern at types.py:80-88 for GdtSemanticPayload's list fields.
**Warning signs:** Spurious flags appearing on items that never had `confidence_flags.append()` called.

## Code Examples

### Example 1: Regression test for CLS-01 (bleed suppression)

```python
# tests/test_classify_bugfixes.py — new test class
class TestAdjacencyBleed:
    """CLS-01: a `/`-merged multi-balloon span must not trigger count_added."""

    def test_bleed_with_count_prefix_suppresses_count_added(self):
        # Exemplar from part-1 char-11 corpus case.
        anchor = _anchor("Counterbore Diameter (13.5 +/- 0.2 mm)")
        candidate = _span(
            "4 x Ø8 THRU ALL / ⌴ Ø13.5 ↧ 8.5",
            block_id=1, line_id=0, span_id=0, x0=12.0, y0=11.0,
        )
        delta = _classify(anchor, candidate)
        assert delta.status == "unchanged", (
            f"Expected 'unchanged' with bleed flag; got '{delta.status}'"
        )
        assert any(
            "Rev B text may contain adjacent balloon content" in f
            for f in delta.confidence_flags
        ), f"Expected bleed confidence flag; got {delta.confidence_flags!r}"

    def test_bleed_with_gdt_symbol_suppresses_count_added(self):
        # GD&T bleed pattern (part-4 char-5-ish).
        anchor = _anchor("Ø.500 ±.005 THRU in")
        candidate = _span(
            "Ø.500 THRU / ⌖∅ .005 Ⓜ A B C",
            block_id=1, line_id=0, span_id=0, x0=12.0, y0=11.0,
        )
        delta = _classify(anchor, candidate)
        # Whatever verdict the primary path chose, the confidence flag must fire.
        assert any(
            "Rev B text may contain adjacent balloon content" in f
            for f in delta.confidence_flags
        )

    def test_benign_fraction_is_not_bleed(self):
        # "70 / 30" from part-7 char-1 — two bare digits, NOT bleed.
        anchor = _anchor("30 ±0.5 mm")
        candidate = _span("70 / 30", block_id=1, line_id=0, span_id=0, x0=12.0, y0=11.0)
        delta = _classify(anchor, candidate)
        assert not any(
            "adjacent balloon content" in f for f in delta.confidence_flags
        )

    def test_existing_count_added_still_marks_changed(self):
        # Regression guard for test_classify_bugfixes.py:107 — no bleed, so
        # count_added must still fire normally.
        anchor = _anchor("Ø 8")
        candidate = _span("2X Ø 8", block_id=1, line_id=0, span_id=0, x0=12.0, y0=11.0)
        delta = _classify(anchor, candidate)
        assert delta.status == "changed"
```

### Example 2: Regression test for CLS-02 (removed+added reconciliation)

```python
# tests/test_classify_bugfixes.py
class TestRemovedAddedReconciliation:
    """CLS-02: close-proximity removed+added pairs collapse to 'changed'."""

    def test_close_pair_on_same_page_collapses_to_changed(self):
        from delta_preservation.reconcile.classify import (
            DeltaItem, reconcile_removed_added_pairs,
        )
        anchor = Anchor(
            char_no=5,
            page=0,
            balloon_bbox=(100.0, 200.0, 110.0, 210.0),
            req_bbox=(120.0, 200.0, 180.0, 210.0),
            requirement_raw="2.500 ±.002 in",
            requirement_norm="2.500 ±.002 in",
            local_context=[],
        )
        removed = DeltaItem(
            char_no=5, status="removed", confidence=0.8,
            reasons=["No candidate found"], component_scores={},
            match=None, added_span=None,
        )
        added_span_span = _span(
            ".750", block_id=1, line_id=0, span_id=0, x0=140.0, y0=220.0, width=20.0,
        )  # centroid ≈ (150, 224) — within 150 pt of anchor centroid (150, 205)
        added = DeltaItem(
            char_no=99, status="added", confidence=0.6,
            reasons=["Leading-decimal dimension annotation detected"],
            component_scores={}, match=None, added_span=added_span_span,
        )
        reconciled = reconcile_removed_added_pairs([removed, added], [anchor])
        assert len(reconciled) == 1
        assert reconciled[0].char_no == 5
        assert reconciled[0].status == "changed"
        assert reconciled[0].added_span is added_span_span

    def test_far_apart_pair_is_not_collapsed(self):
        # Anchor at (150, 205); added span at (600, 600) — > 150 pt away.
        # Should remain two separate items.
        ...

    def test_type_incompatible_pair_is_not_collapsed(self):
        # Removed dimension + added GD&T frame — must not collapse.
        ...
```

### Example 3: Regression test for CLS-03 (asymmetric tolerance detection)

```python
class TestAsymmetricToleranceChange:
    """CLS-03: symmetric → asymmetric tolerance change is 'changed'."""

    def test_symmetric_to_asymmetric_angle_tolerance(self):
        # Exemplar: part-7 char-4.
        anchor = _anchor("2X 22.0° ±1°")
        candidate = _span(
            "2X 22.0° +0.3° / −0.1°",
            block_id=1, line_id=0, span_id=0, x0=12.0, y0=11.0,
        )
        delta = _classify(anchor, candidate)
        assert delta.status == "changed", (
            f"Expected 'changed' (asymmetric tolerance); got '{delta.status}'"
        )
        assert any(
            "tolerance" in r.lower() and (
                "form changed" in r.lower() or "asymmetr" in r.lower()
            )
            for r in delta.reasons
        )

    def test_asymmetric_to_symmetric_also_detected(self):
        """Reverse direction must also fire (`+0.3/-0.1` → `±0.5`)."""
        anchor = _anchor("8.0 +0.3 -0.1")
        candidate = _span("8.0 ±0.5", block_id=1, line_id=0, span_id=0, x0=12.0, y0=11.0)
        delta = _classify(anchor, candidate)
        assert delta.status == "changed"

    def test_symmetric_to_symmetric_unchanged_stays_unchanged(self):
        """Regression guard — don't over-fire on same-shape tolerance."""
        anchor = _anchor("8.0 ±0.1")
        candidate = _span("8.0 ±0.1", block_id=1, line_id=0, span_id=0, x0=12.0, y0=11.0)
        delta = _classify(anchor, candidate)
        assert delta.status == "unchanged"
```

### Example 4: Preserving the existing test baseline

```python
# These tests currently pass and MUST continue to pass after Phase 5:
# - tests/test_classify_bugfixes.py::TestCountAdded (both methods)
# - tests/test_classify_bugfixes.py::TestSpuriousMatchGuard (both methods)
# - tests/test_classify_bugfixes.py::TestGdtAddedDetection (both methods)
# - tests/test_classify_bugfixes.py::TestPlainDecimalAdded (all cases)
# - tests/test_reconcile_semantic_integration.py (all)
# - tests/test_pipeline_semantic_packet.py (all)
# Per-task verification MUST run the full `test_classify_bugfixes.py` suite.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `count_added` fires whenever `not anchor_count and matched_count` | Bleed-aware suppressor adds structural check before firing | Phase 5 (CLS-01) | Part-1 and Part-4 false positives eliminated |
| `classify_delta` and `detect_added_characteristics` produce independent outputs with no cross-reconciliation | Post-pass `reconcile_removed_added_pairs` collapses spatially-close type-compatible pairs | Phase 5 (CLS-02) | Parts 2/3/4/8 "double count" eliminated (6 items) |
| Tolerance change detection relies on numeric-set difference only | Kind-based structural check + string-shape fallback | Phase 5 (CLS-03) | Part-7 char-4 fixed; general symmetric↔asymmetric transitions covered |
| `DeltaItem` carries only `reasons: List[str]` for reviewer context | Additively gains `confidence_flags: List[str]` for advisory signals | Phase 5 | Reviewers see the bleed warning without the classifier being forced to a specific verdict |

**Deprecated/outdated:** None — every Phase 5 change is additive over the existing behaviour and preserves all currently-passing tests.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest ≥8 (`[tool.pytest.ini_options]` in pyproject.toml) |
| Config file | `pyproject.toml` — `testpaths = ["tests"]`, `addopts = "-q"` [VERIFIED] |
| Quick run command | `pytest tests/test_classify_bugfixes.py -x -q` |
| Per-wave command | `pytest tests/test_classify_bugfixes.py tests/test_reconcile_semantic_integration.py tests/test_pipeline_semantic_packet.py -x -q` |
| Phase-gate command | `pytest -q` — full suite green before `/gsd-verify-work` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CLS-01 | Part-1 char-11 style bleed with count prefix + sibling Ø → `unchanged` with flag | unit | `pytest tests/test_classify_bugfixes.py::TestAdjacencyBleed::test_bleed_with_count_prefix_suppresses_count_added -x` | ❌ Wave 0 |
| CLS-01 | Part-4 char-7 style bleed with thread callout | unit | `pytest tests/test_classify_bugfixes.py -k "bleed" -x` | ❌ Wave 0 |
| CLS-01 | Benign `70 / 30` digit pair stays bleed-free | unit (regression guard) | `pytest tests/test_classify_bugfixes.py -k "benign_fraction" -x` | ❌ Wave 0 |
| CLS-01 | Existing `count_added` test still marks `changed` | unit (regression) | `pytest tests/test_classify_bugfixes.py::TestCountAdded::test_count_added_marks_changed -x` | ✅ exists at line 107 |
| CLS-02 | Close pair on same page collapses to single `changed` row | unit | `pytest tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation::test_close_pair_on_same_page_collapses_to_changed -x` | ❌ Wave 0 |
| CLS-02 | Far-apart pair (> 150 pt) is NOT collapsed | unit (boundary) | `pytest tests/test_classify_bugfixes.py -k "far_apart" -x` | ❌ Wave 0 |
| CLS-02 | Type-incompatible pair (dimension + GD&T) is NOT collapsed | unit (boundary) | `pytest tests/test_classify_bugfixes.py -k "type_incompatible" -x` | ❌ Wave 0 |
| CLS-02 | Page mismatch (hypothetical multi-page) is NOT collapsed | unit (boundary) | `pytest tests/test_classify_bugfixes.py -k "page_mismatch" -x` | ❌ Wave 0 |
| CLS-03 | Part-7 char-4 `±1°` → `+0.3°/-0.1°` → `changed` | unit | `pytest tests/test_classify_bugfixes.py::TestAsymmetricToleranceChange::test_symmetric_to_asymmetric_angle_tolerance -x` | ❌ Wave 0 |
| CLS-03 | Reverse asymmetric → symmetric also detected | unit | `pytest tests/test_classify_bugfixes.py -k "asymmetric_to_symmetric" -x` | ❌ Wave 0 |
| CLS-03 | Same-shape symmetric tolerance stays `unchanged` | unit (regression guard) | `pytest tests/test_classify_bugfixes.py -k "symmetric_to_symmetric" -x` | ❌ Wave 0 |
| Cross-cutting | Existing `TestSpuriousMatchGuard`, `TestGdtAddedDetection`, `TestPlainDecimalAdded` | unit (regression) | `pytest tests/test_classify_bugfixes.py -x` | ✅ exists |
| Cross-cutting | Semantic pipeline integration | integration (regression) | `pytest tests/test_reconcile_semantic_integration.py tests/test_pipeline_semantic_packet.py -x` | ✅ exists |
| Cross-cutting | Debug/review internals (persisted DeltaItem shape) | integration | `pytest tests/test_debug_internals.py tests/test_debug_verdicts.py -x` | ✅ exists |

### Sampling Rate

- **Per task commit:** `pytest tests/test_classify_bugfixes.py -x -q` (≈1-2 s)
- **Per wave merge:** `pytest tests/test_classify_bugfixes.py tests/test_reconcile_semantic_integration.py tests/test_pipeline_semantic_packet.py tests/test_debug_internals.py -x -q`
- **Phase gate:** `pytest -q` — full suite green before `/gsd-verify-work`
- **Corpus regeneration** (NOT part of Phase 5 — deferred to Phase 7 / VER-01): full `run_pipeline` invocation for each of the 9 parts, diff against existing `assets/debug_report_part{1..9}.json`.

### Wave 0 Gaps

- [ ] New test class `TestAdjacencyBleed` in `tests/test_classify_bugfixes.py` covering CLS-01 (at least 4 cases: part-1 exemplar, part-4 exemplar, benign fraction, existing count_added regression guard).
- [ ] New test class `TestRemovedAddedReconciliation` in `tests/test_classify_bugfixes.py` covering CLS-02 (at least 4 cases: close-pair collapse, far-apart rejection, type-incompatible rejection, empty-input no-op).
- [ ] New test class `TestAsymmetricToleranceChange` in `tests/test_classify_bugfixes.py` covering CLS-03 (at least 3 cases: forward sym→asym, reverse asym→sym, sym→sym regression guard).
- [ ] `classify.DeltaItem.confidence_flags: List[str] = field(default_factory=list)` field added.
- [ ] `types.DeltaItem.confidence_flags: List[str] = Field(default_factory=list, ...)` field added.
- [ ] `cli.py:842` Pydantic conversion wires `confidence_flags=delta_internal.confidence_flags`.
- [ ] `cli.py:508` (new line after the `extend(added_items)`) calls `reconcile_removed_added_pairs(delta_items_internal, anchors)`.
- [ ] No new framework install. pytest ≥8 already present.

### Reference Data (debug corpus)

The 9-part corpus lives in `assets/part{1..9}/` and corresponding `assets/debug_report_part{1..9}.json` snapshots. Phase 5 regression uses **synthetic `TextSpan` + `Anchor` inputs that reproduce the token shapes** from the corpus exemplars documented above — not the binary PDFs. Full-pipeline validation against the corpus itself is Phase 7 (VER-01).

Cataloged exemplars:
- **CLS-01 bleed** → Part 1 char 11 ("Counterbore Diameter (13.5 +/- 0.2 mm)" + "4 x Ø8 THRU ALL / ⌴ Ø13.5 ↧ 8.5"); Part 1 char 12 (counterbore depth, same Rev B span); Part 4 char 7 ("1/4-20 UNC-2B" + "2X Ø.201 ↧ 0.50 / 1/4-20 UNC - 2B").
- **CLS-02 remove+add pairs** → Part 2: chars 1/6/7/9 removed vs chars 18-23 added. Part 3: chars 2/3/6/7 removed vs chars 20-22 added. Part 4: chars 9/10 removed vs chars 12-17 added. Part 8: chars 1/3/6 removed vs chars 9-13 added. Exact pairing is corpus-geometry-dependent; the plan should require a Wave-0 audit task to record observed pairs.
- **CLS-03 asymmetric tolerance** → Part 7 char 4 ("2X 22.0° ±1°" + "2X 22.0° +0.3° / −0.1°"). Only 1 item in the corpus — but the fix is structural, so it must generalize to the reverse direction and future cases.

### How to Detect Regression of the Fix

- **Primary signal:** `pytest tests/test_classify_bugfixes.py -x -q`. All new tests pass; all existing tests still pass.
- **Secondary signal:** Confidence-flag plumbing — `test_debug_internals.py` and `test_debug_verdicts.py` must deserialize `DeltaItem` with the new optional field cleanly.
- **Tertiary signal** (for Phase 7): Full corpus re-run via `run_pipeline` — `debug_report_part*.json` output diffs only in the expected direction (part-1 chars 11/12 flip from `changed` → `unchanged+flag`, part-7 char-4 flips from `unchanged` → `changed`, the 6 removed+added pairs collapse).

## Risks and Edge Cases

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | Bleed heuristic misfires on legitimate `/`-containing span (fit `H7/p6`, thread `1/4-20`, fraction `1/8`) | MEDIUM | Incorrect suppression of a genuine `count_added` change | Heuristic explicitly rejects (a) pure-digit chunks, (b) fit-designator `letter+digit` chunks, (c) tolerance-continuation `+/-digit` chunks. See Pitfall 5 for the compound-interaction safeguard. |
| R2 | CLS-02 collapses a genuine independent removal + independent addition that happen to be near each other | MEDIUM | False "changed" verdict masking two real design changes | Three-gate defence: same page + ≤150 pt distance + type-compatible. Plan should require a corpus-audit Wave-0 task to confirm no false-pair exists in parts 2/3/4/8. |
| R3 | CLS-03 kind-based path relies on `ToleranceComparison` but the parser may return `kind="none"` for bleed-corrupted spans | MEDIUM | CLS-03 primary path silently skipped; fallback string check carries the load | Two-pronged implementation (kind + string fallback) provides defense-in-depth. String fallback has broader coverage but lower precision. |
| R4 | `confidence_flags` field addition breaks JSON deserialization of legacy `DeltaItem` blobs persisted in SQLite | LOW | Packet reads fail | `default_factory=list` ensures missing-field tolerance in Pydantic v2. Verified in Pitfall 7. |
| R5 | The 150-pt proximity threshold is too loose for dense drawings (e.g., part 1 with 30+ balloons) | MEDIUM | Over-aggressive collapsing | Expose the threshold as a named module constant. Phase 7's corpus re-run will surface any aggregate accuracy regression; tightening to 100 pt is a one-line change. |
| R6 | The `reconcile_removed_added_pairs` post-pass runs once per pipeline; if multi-page support is added later, page gating must be enforced | LOW (currently single-page) | Future multi-page regressions | The sketch already gates on `anchor.page` (even though pipeline is single-page today); the guard is a no-op now but correct for future expansion. |
| R7 | `count_added` can also fire on non-bleed single-compartment `/` — e.g., a legitimate composite GD&T callout that includes a count prefix | LOW | CLS-01 suppresses a genuine count change | GD&T composite frames don't typically carry count prefixes; phase-4's `compartments` split makes the composite form structurally distinct from the bleed form. Regression verified by `test_existing_count_added_still_marks_changed`. |
| R8 | Part-7 char-4 bleed interaction (Pitfall 5) — CLS-01 and CLS-03 both have a claim to this span | HIGH (confirmed) | Classification order matters | CLS-01 heuristic rejects `[+−-]digit` chunks (tolerance continuation pattern), so the span is NOT tagged as bleed and CLS-03 runs normally. Verified logic flow in Pitfall 5. |
| R9 | `added_span.bbox_pdf` is single-page-relative; if pipeline ever grows to multi-page, proximity math silently compares page-0 coordinates to page-N coordinates | LOW (single-page pipeline) | Incorrect multi-page pairings | Gate on `anchor.page` already present in Pattern 2; equivalent page info for added items would need to be captured during `detect_added_characteristics`. Deferred — not in scope for Phase 5. |

## Open Questions (RESOLVED)

*All open questions below have inline `Recommendation:` resolutions. No residual blockers for planning.*

1. **Exact bleed-heuristic threshold: 2 chunks vs ≥3 chunks**
   - What we know: All Part-1/Part-4 bleed cases have exactly 2 chunks. Part-9 char-19 has 3 chunks ("4X Ø.415 ±.008 / ⌖∅ .025 A B C / ⌴ Ø.625 ±.020 ↧ .50 ±.02").
   - What's unclear: Does a 3-chunk bleed still warrant the same suppression, or should 3-chunk be MORE confident as bleed?
   - Recommendation: Heuristic fires on ≥2 chunks with any of the three signals; confidence is binary (flag appears or doesn't). A 3-chunk span simply has more signals available and reaches the threshold more easily. No separate branch needed.

2. **Should `reconcile_removed_added_pairs` be stricter than 150 pt for the confidence boost?**
   - What we know: 75 pt ≈ 1 inch (one balloon-radius on a typical drawing); 150 pt ≈ 2 inches (adjacent views).
   - What's unclear: Whether the plan should pin a single threshold or keep the two-tier (strict=75pt confidence=0.75, loose=150pt confidence=0.60) scheme suggested in Pattern 2.
   - Recommendation: Keep two-tier initially. Phase 7's corpus re-run will show whether the distinction matters; if accuracy is uniform, collapse to one threshold in Phase 7.

3. **Should CLS-03 also fire when the primary dimension value changes AND the tolerance form changes?**
   - Example: `"10.0 ±0.1"` → `"11.0 +0.3/-0.1"` — both the dimension and tolerance form changed.
   - What we know: The primary-dimension-change path (classify.py:813) already promotes to `changed`. CLS-03 would be redundant but not incorrect in this case.
   - Recommendation: Let both paths fire independently. The `status == "unchanged"` gate in the CLS-03 sketch ensures CLS-03 doesn't downgrade an already-`changed` verdict — it only upgrades missed ones.

4. **Is the `default_factory=list` migration on persisted `DeltaItem` actually safe for SQLite-stored packets?**
   - What we know: The packet is stored as a JSON blob. Pydantic v2 tolerates missing optional fields during `model_validate_json` when a default is specified.
   - What's unclear: Whether any alembic migration path validates the existing blobs or if there's a packet-schema version check.
   - Recommendation: Plan should include a small verification task (manual or automated): load one existing packet from a test DB fixture, confirm it deserializes without error after the field is added. [ASSUMED safe based on standard Pydantic v2 semantics — flagged in Assumptions Log.]

5. **Does any Phase 6 logic consume `confidence_flags`?**
   - What we know: ADD-01/ADD-02/SNP-01 deal with added-characteristic detection and snippet scoping.
   - What's unclear: Whether Phase 6 wants to read back the bleed flag to adjust added-characteristic detection for bled spans.
   - Recommendation: Leave for Phase 6 discussion. Phase 5 only populates flags; Phase 6 can consume them.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python interpreter | All classifier code | ✓ | ≥3.10, <3.13 (pyproject.toml) | — |
| pytest | Regression tests | ✓ | ≥8 | — |
| pydantic | Type schema change | ✓ | ≥2.5,<3.0 | — |
| `re` (stdlib) | Bleed heuristic, shape fallback | ✓ | stdlib | — |
| `math` (stdlib) | Spatial proximity | ✓ | stdlib | — |

All dependencies are declared in `pyproject.toml` and locked via `uv.lock`. No external tooling, no database, no network. [VERIFIED: pyproject.toml inspected, dependencies listed above.]

## Project Constraints (from repository context + CLAUDE.md)

- No CLAUDE.md present in `/home/khoa2/delta-preservation/` (verified 2026-04-16). Constraints derived from `.planning/` documents:
- **No per-part hacks.** Every fix must generalize across all 9 corpus parts. [REQUIREMENTS.md milestone goal]
- **Ground-truth files canonical.** `assets/part{1..9}/ground_truth.json` never modified by the pipeline or tests. [REQUIREMENTS.md out-of-scope]
- **Scope boundary:** classifier layer (`classify.py`) + minimal `types.py` / `cli.py` wiring. No changes to `normalize.py`, `semantic_compare.py`, `tolerance_pdf.py`, web tier, review workflow, xlsx/pdf loaders. [phase description]
- **Regression fast path:** Unit tests via `test_classify_bugfixes.py` pattern; no full-pipeline regeneration in Phase 5 (that's Phase 7 / VER-01). [ROADMAP.md success criterion 4 + Phase 7 Goal]
- **Phase dependency:** Phase 5 depends on Phase 4. Phase 4 is executing; verify its completion before Phase 5 begins. [STATE.md, ROADMAP.md]
- **Ground-truth exception queue:** Missing-added-characteristics tracked as `missing_added_truth_indexes` — a Phase 6 concern, NOT a Phase 5 concern. [REQUIREMENTS.md]

## Security Domain

> `security_enforcement` is not set in `.planning/config.json`. Per policy, treat as enabled. This phase's security footprint is minimal (pure in-process logic on text + bboxes already in memory), but the applicable ASVS controls are listed for completeness.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | n/a — classifier is internal, not network-exposed |
| V3 Session Management | no | n/a |
| V4 Access Control | no | n/a — operates on already-loaded packet data |
| V5 Input Validation | **yes** | Classifier consumes arbitrary PDF-extracted span text. Bleed-detection regex must be anchored and bounded to prevent ReDoS. All proposed regexes use bounded quantifiers and character classes; no nested unbounded quantifiers (`(.*)*`). [VERIFIED by inspection of the proposed regexes.] |
| V6 Cryptography | no | n/a |
| V7 Error Handling | **yes** | Classifier errors must produce a valid `DeltaItem` (with `status="uncertain"` as the catch-all), not raise exceptions. Existing pattern preserved: `reconcile_removed_added_pairs` must handle empty lists and missing anchors gracefully (Pattern 2 sketch guards `anchor_by_char.get(r.char_no)`). |
| V8 Data Protection | no | n/a — no PII, no secrets (manufacturing dimensions only) |
| V12 Files & Resources | no | n/a — no file I/O in classifier |

### Known Threat Patterns for Python regex + list-comprehension pipelines

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Regex catastrophic backtracking (ReDoS) on pathological multi-`/` inputs | Denial-of-service | All proposed patterns (`_BLEED_SLASH_RE = r"\s*/\s*"`, `_COUNT_PREFIX_RE = r"\b\d+\s*[Xx](?!\s*\d)"`, `asym_re`) use bounded character classes and no nested unbounded quantifiers. |
| Integer / index overflow on bbox math | Tampering | All arithmetic is on Python floats; no integer indexing. `math.sqrt` on `(cx-mx)**2 + (cy-my)**2` is bounded by the PDF coordinate space (~1000 × ~1000 points). |
| Malicious Unicode input (homoglyph attack via Rev B bleed span) | Tampering | No security decision is made on the classifier's output — it's advisory for a human reviewer. Flags are text only; no code execution. |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `assets/debug_report_part{1..9}.json` accurately reflect the current classifier's output as of Phase 5 start (not stale from earlier milestone work) | Corpus exemplars in Pattern 1-3 | Exemplar expectations may not match a regenerated baseline. Plan should include a Wave-0 task to regenerate one part's debug_report and confirm the removed+added items documented here still match. [ASSUMED based on debug_report_part7.json dated content and `.planning/STATE.md` progress] |
| A2 | Part-4 char-5 "Ø.500 THRU / ⌖∅ .005 Ⓜ A B C" is an adjacency-bleed case (Rev B GD&T span merged with a Rev B plain-diameter span), not a legitimate composite FCF | Example 1 test case | If this is actually a legitimate composite FCF (which would be caught by Phase 4's compartment splitter), CLS-01 may wrongly flag it. Plan should instruct the CLS-01 task to verify against the Phase 4 output. |
| A3 | 150 PDF points is an appropriate upper bound for CLS-02 same-cluster proximity | Pattern 2 rationale | If the corpus shows valid pairs further apart (e.g., an annotation moved to an adjacent view), the threshold might need to grow. Two-tier scheme and exposed constants mitigate. |
| A4 | Adding `confidence_flags: List[str]` to persisted `types.DeltaItem` is backward compatible with existing SQLite-stored packets | Runtime State Inventory | If Pydantic v2 rejects missing-field deserialization despite `default_factory=list`, existing packet reads fail. Low risk — standard Pydantic v2 behaviour — but flagged for confirmation. Recommendation: plan includes a one-shot deserialization test against an existing test-fixture packet. |
| A5 | Part-7 char-4 `"+0.3° / −0.1°"` is parsed by `tolerance_pdf._parse_inline_signed_pair` as `bilateral_stacked` | Pattern 3 primary path | If the bleed context prevents the parser from extracting a clean pair (returning `kind="none"`), the primary path is a no-op and only the string-fallback path carries CLS-03. Recommendation: plan includes a focused test that verifies `extract_tolerances_for_items` on the raw part-7 spans actually produces non-`none` kinds on both sides. |
| A6 | The 6 removed+added items across parts 2/3/4/8 ARE genuine pair candidates (i.e., the removed anchor and one of the added items refer to the same physical characteristic that moved or was re-annotated) | CLS-02 acceptance | If they aren't genuine pairs, CLS-02 may produce false "changed" rows. Mitigated by the type-compatibility gate + the 150-pt proximity gate + the plan's required Wave-0 audit task against the corpus PDFs. |
| A7 | The user-facing phrase "Rev B text may contain adjacent balloon content" is the exact desired text (from reviewer notes in `assets/debug_report_part7.json`) | CLS-01 confidence flag | If the actual desired wording is slightly different, regression tests need to match. [VERIFIED: exact phrase appears in the reviewer note for part-7 char-1; Phase 5 should use this exact string.] |

## Sources

### Primary (HIGH confidence — verified by direct code read this session)

- `delta_preservation/reconcile/classify.py` — full file, ~1478 lines. Confirmed `count_added` at line 656, `count_added and not anchor_numerics` suppressor at 700-730, `count_changed or count_added` branch at 733, `tolerance_sized_diff` at 746, `tolerance_comparison` refinement at 904, `detect_added_characteristics` at 941.
- `delta_preservation/reconcile/semantic_compare.py` — confirmed `_compare_gdt` at lines 112-195 (consulted as read-only; not modified in this phase).
- `delta_preservation/reconcile/tolerance_pdf.py` — confirmed `PdfToleranceKind` literal at line 41 (`plus_minus` / `bilateral_stacked` / `unilateral_stacked`), `compare_tolerances` at line 109, `_parse_inline_signed_pair` at line 206, `ToleranceComparison` dataclass at line 96.
- `delta_preservation/reconcile/normalize.py` — confirmed `parse_requirement` at line 227, `classify_requirement_type` and `are_requirement_types_incompatible` used throughout classify.py.
- `delta_preservation/reconcile/anchors.py` — confirmed `Anchor` dataclass has `page: int`, `balloon_bbox`, `req_bbox` (lines 22-57).
- `delta_preservation/cli.py` — confirmed pipeline entry point at line 193, per-anchor classify loop at line 465, `detect_added_characteristics` call at line 502, Pydantic conversion at line 842.
- `delta_preservation/types.py` — confirmed Pydantic `DeltaItem` at line 244 with `reasons: List[str]` field.
- `tests/test_classify_bugfixes.py` — confirmed `_span`, `_anchor`, `_classify` helper pattern (lines 35-97), existing `TestCountAdded` / `TestSpuriousMatchGuard` / `TestGdtAddedDetection` / `TestPlainDecimalAdded` blocks that serve as regression baseline.
- `assets/debug_report_part{1,2,3,4,6,7,8,9}.json` — inspected via Python script to verify exemplar cases for each fix cluster; see "Reference Data" section.
- `.planning/phases/04-gd-t-parser-fixes/04-RESEARCH.md` — consumed as style/depth reference for this document.
- `.planning/phases/04-gd-t-parser-fixes/04-CONTEXT.md` — consumed as conventions reference (decisions schema, anti-patterns format).
- `.planning/REQUIREMENTS.md` — CLS-01/02/03 acceptance criteria, traceability, out-of-scope.
- `.planning/ROADMAP.md` — Phase 5 goal and success criteria (items 1-4).
- `.planning/STATE.md` — confirms Phase 4 is executing, Phase 5 is next.
- `.planning/config.json` — confirmed `nyquist_validation: true`, `commit_docs: true`.
- `pyproject.toml` — confirmed Python 3.10-3.13, pydantic 2.5-3.0, pytest ≥8.

### Secondary (MEDIUM confidence — external context, not tool-verified this session)

- ASME Y14.5 convention that `±N` expresses bilateral-symmetric tolerance and `+A / -B` expresses bilateral-asymmetric tolerance. Widely known engineering-drawing convention; reinforces the CLS-03 structural-shape argument.

### Tertiary (LOW confidence — flagged for plan-time verification)

- None. Every critical claim in this research was cross-verified against either (a) the live repo files or (b) the corpus debug-report JSON snapshots this session.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies confirmed against `pyproject.toml`, no new installs.
- Architecture / code seams: HIGH — every line number, symbol name, and branch was re-read this session.
- Corpus exemplars (CLS-01 / CLS-02 / CLS-03 cases): HIGH — each case confirmed by direct JSON inspection of the debug-report snapshots.
- Spatial-proximity threshold (150 pt / 75 pt): MEDIUM — rationale is engineering-drawing convention (72 pt/in); corpus-specific tuning may adjust in Phase 7.
- `default_factory=list` Pydantic v2 migration safety: MEDIUM — standard behaviour, but flagged in Assumptions Log for plan-time sanity check.
- Pitfall 5 CLS-01/CLS-03 interaction safeguard: HIGH — traced through both fix logics; the tolerance-continuation rejection in Pattern 1 is the explicit safeguard.

**Research date:** 2026-04-16
**Valid until:** 2026-05-16 (30 days — classifier code is stable; corpus debug reports are frozen snapshots; only a full-pipeline Phase 7 regeneration would shift any claim, and that's outside Phase 5's scope).
