---
phase: 09-full-corpus-added-characteristic-closure
plan: "02"
subsystem: reconcile/classify — added detection and explained-by-match suppressor
tags: [added-detection, gdt-anchors, surface-finish-grouping, owner-signatures, regression-tests]
dependency_graph:
  requires:
    - 09-01 (canonical added-row token contract and leading-zero normalization)
  provides:
    - Narrowed matched-owner signature construction (companion-only, not ±200 pt sweep)
    - ⌵ countersink symbol recognized as GD&T anchor in Pass 0
    - Surface-finish same-block expansion in _expand_standard_added_span
    - Regression tests for Part 9 flatness survival and Parts 3-5 exemplar families
  affects:
    - delta_preservation/reconcile/classify.py
    - tests/test_added_detection_phase6.py
tech_stack:
  added: []
  patterns:
    - Fixed-point companion walk for owner signature construction (replaces ±200 pt sweep)
    - Surface-finish same-block expansion for multi-line annotation grouping
    - GD&T anchor symbol extension for countersink callouts
key_files:
  created: []
  modified:
    - delta_preservation/reconcile/classify.py (matched-owner signatures + ⌵ anchor + surface-finish expansion)
    - tests/test_added_detection_phase6.py (7 new tests across 2 new test functions + 1 new test class)
decisions:
  - "Replace ±200 pt same-row sweep in matched-owner signature building with fixed-point _spans_are_annotation_companions walk"
  - "Add ⌵ (U+2375 countersink) to GDT_ANCHOR_SYMBOLS so Pass 0 seeds from it rather than falling through to the 25 pt proximity-filtered Pass 2"
  - "Surface-finish seeds use same-block vertical sweep (≤25 pt gap) to capture non-numeric prefix rows (e.g. FINISH TURN) that share a PDF annotation block"
metrics:
  duration: "28 min"
  completed_date: "2026-04-18"
  tasks_completed: 2
  files_modified: 2
requirements: [ADD-01, ADD-02, SNP-01]
---

# Phase 9 Plan 02: Detector-Side Miss Closure — Owner Signatures and Parts 3-5 Families

One-liner: Narrowed matched-owner signatures to companion-only walks plus ⌵ anchor and surface-finish block expansion close the Part 9 flatness false-suppression and Part 3 detection misses, with full exemplar regression coverage for Parts 3-5.

## What Was Built

### Task 1 — Narrow matched-owner signature construction

**Problem confirmed by Phase 9 debug:** `detect_added_characteristics()` built matched-owner annotation signatures by sweeping all same-row unmatched spans within a ±200 pt horizontal window into the owner's synthetic text/bbox. This caused the independent Part 9 block-46 flatness frame (`⏥ .01`) to be absorbed into matched char 8's owner signature (`4X INDIVIDUALLY 2X CR.50±.02 ⏥ .01`). The subsequent content-subset + bbox-overlap suppressor check then falsely removed the real added flatness row.

**Fix:** Replaced the `±200 pt` window condition with a fixed-point companion walk using `_spans_are_annotation_companions()`. Owner signatures now expand only to spans that are genuine annotation companions (same-row and horizontally adjacent within GD&T limits, or stacked numeric fragments). Unrelated same-row callouts hundreds of points away are no longer absorbed.

The companion walk is also iterative (fixed-point) so transitively adjacent companions remain correctly included.

**Tests added:**
- `test_part9_flatness_survives_when_unrelated_same_row_match_exists` — reproduces the Part 9 geometry (match at x≈335, added flatness at x≈712, ~376 pt apart on same row); proves the added flatness item now survives.
- `test_part8_fragment_still_suppressed_after_owner_signature_narrowing` — proves that true stacked companion fragments (`.045 A` below matched `◎ ∅.045 A`) are still absorbed into the owner signature and correctly suppressed.

### Task 2 — Close Parts 3-5 miss families through shared detector/grouping changes

**Root causes identified per exemplar family:**

| Exemplar | Root cause | Fix |
|----------|-----------|-----|
| `⌵ Ø.531 X 82.0°` (Part 3 truth_index 19) | `⌵` not in `GDT_ANCHOR_SYMBOLS` → fell to Pass 2 → `Ø.531 X 82.0°` companion filtered by 25 pt near-matched-span check (17.7 pt from matched `Ø.266 THRU`) | Added `⌵` to `GDT_ANCHOR_SYMBOLS`; Pass 0 now seeds from `⌵` (41.6 pt from match — passes 12 pt guard) |
| `FINISH TURN 1000 Ra` (Part 3 truth_index 20) | `FINISH TURN` prefix has no numeric payload and no GDT chars → failed `_spans_are_annotation_companions` in `_expand_standard_added_span` (stacked dx=14.7 > 14.0 non-GDT limit) | Added `is_surface_finish_seed` parameter; when seed has `Ra` indicator, same-block spans within 25 pt vertical gap are swept in regardless of companion geometry |
| `⌖ ∅.005Ⓜ A B C` (Part 4 truth_index 11) | Falsely suppressed by wide ±200 pt owner-signature sweep (Task 1 bug) | Fixed by Task 1 owner-signature narrowing |
| `1.250` rows (Part 4 truth_indexes 14, 15) | Same ±200 pt suppressor over-sweep as above | Fixed by Task 1 owner-signature narrowing |
| `3X Ø18 ↧30` (Part 5 truth_index 17) | `3X Ø18 ↧30` source spans are all matched to char 1 (matching mis-assignment) — fully consumed before Pass 0 even starts | Unit-level detector test uses unmatched spans to prove the shared detection path works; live-pipeline fix requires matching-layer corrections outside Plan 02 scope |
| `M20x2.5 − 6H ↧6` (Part 5 truth_index 18) | Already detected by live pipeline as `M20x2.5 − 6H ↧ 6` (minor spacing diff, evaluator normalization handles it) | Regression test added to guard the detection path |

**Tests added (new class `TestPhase9ExemplarFamilies`):**
- `test_surface_finish_multiline_grouping_produces_finish_turn_1000_ra`
- `test_countersink_symbol_detected_as_gdt_anchor_for_csink_callout`
- `test_position_callout_with_modifier_and_datums_detected` (`⌖ ∅.005Ⓜ A B C`)
- `test_3x_hole_with_depth_callout_detected` (`3X Ø18 ↧30`)
- `test_threaded_hole_with_depth_m20_detected` (`M20x2.5 − 6H ↧6`)

## Verification

- `uv run pytest -q tests/test_added_detection_phase6.py -k "part9_flatness_survives_when_unrelated_same_row_match_exists or part8_fragment_still_suppressed_after_owner_signature_narrowing or suppression" -x` → 5 passed
- `uv run pytest -q tests/test_added_detection_phase6.py tests/test_phase6_asset_regression.py -x` → 51 passed
- `git diff --name-only -- assets/debug_report_part*.json` → no output (frozen assets unchanged)

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1    | 4ae7b75 | feat(09-02): narrow matched-owner signatures to companion spans only |
| 2    | 59cf80e | feat(09-02): close Parts 3-5 miss families via shared detector/grouping fixes |

## Deviations from Plan

### Out-of-scope discovery: Part 5 `3X Ø18 ↧30` is a matching mis-assignment

**Found during:** Task 2 investigation
**Issue:** The `3X Ø18 ↧30` callout is not a detector miss in isolation — all three of its source spans (`3X Ø18`, `↧`, `30`) are grouped and matched to char 1 (`Ø35.2 / Ø34.8`) by the matching algorithm. This consumes the spans before Pass 0 can detect them. The root cause is at the matching layer, not the detector/grouping layer.
**Action taken:** Added unit-level regression test using unmatched spans to prove the shared detection path correctly handles the callout. Documented as a deferred matching fix for Plan 03 or subsequent work.
**Rule:** Rule 4 boundary (would require matching-layer architectural change) — documented without fixing in this plan.

### Part 5 `800` not closed

**Found during:** Task 2 investigation
**Issue:** `800` (plain integer ≥ 100 at center (615, 581)) is filtered by the near-boilerplate check — `UNLESS OTHERWISE SPECIFIED` is ~54 pt away, within the 120 pt threshold. The threshold was intentionally set to be conservative, and reducing it risks false positives. `800` was not in the required acceptance criteria literal strings, so this is deferred.
**Action:** Deferred. Not a regression — `800` was already missing in the Phase 7 algorithm-only baseline.

## Known Stubs

None.

## Threat Flags

None — changes are confined to internal detection and grouping logic within `classify.py`. No new network endpoints, auth paths, file access patterns, or schema changes introduced. Historical Phase 6 asset snapshots (`assets/debug_report_part*.json`) are untouched.

## Self-Check: PASSED

- `delta_preservation/reconcile/classify.py` does NOT contain `ox0 > mx1 + 200.0 or ox1 < mx0 - 200.0` ✓
- `delta_preservation/reconcile/classify.py` contains `matched_annotation_signatures` ✓
- `delta_preservation/reconcile/classify.py` contains `⌵` in GDT_ANCHOR_SYMBOLS block ✓
- `delta_preservation/reconcile/classify.py` contains `is_surface_finish_seed` ✓
- `delta_preservation/reconcile/classify.py` contains `span_is_excluded_for_annotation_search(` ✓
- `tests/test_added_detection_phase6.py` contains `test_part9_flatness_survives_when_unrelated_same_row_match_exists` ✓
- `tests/test_added_detection_phase6.py` contains `test_part8_fragment_still_suppressed_after_owner_signature_narrowing` ✓
- `tests/test_added_detection_phase6.py` contains `FINISH TURN 1000 Ra` ✓
- `tests/test_added_detection_phase6.py` contains `⌵ Ø.531 X 82.0°` ✓
- `tests/test_added_detection_phase6.py` contains `⌖ ∅.005Ⓜ A B C` ✓
- `tests/test_added_detection_phase6.py` contains `3X Ø18 ↧30` ✓
- `tests/test_added_detection_phase6.py` contains `M20x2.5 − 6H ↧6` ✓
- Commits 4ae7b75 and 59cf80e exist in git log ✓
- 51 tests pass in `tests/test_added_detection_phase6.py tests/test_phase6_asset_regression.py` ✓
- No `assets/debug_report_part*.json` files modified ✓
