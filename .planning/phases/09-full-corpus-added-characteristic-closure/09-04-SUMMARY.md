# 09-04 Summary: Zone-Aware Plain-Integer Boilerplate Guard

**Status:** COMPLETE
**Commit:** 4451f41

## What Changed

### Task 1: Replace 120 pt proximity loop with zone-aware filter in `classify.py`

Removed the blunt 120 pt omni-directional proximity sweep from the plain-integer
near-boilerplate guard inside `detect_added_characteristics()`. Replaced it with
a two-branch zone-aware check:

1. **Zone branch** — delegates to `span_is_excluded_for_annotation_search()` from
   `delta_preservation.reconcile.exclusion`. If the candidate span itself sits in
   the shared boilerplate zone (title block, revision table, bottom-centre tolerance
   zone, or its own text is recognised as boilerplate), it is suppressed.
2. **Same-row companion branch** — scans `revB_spans` for a boilerplate phrase
   (`is_boilerplate_candidate_text`) whose centre is vertically co-linear
   (≤ 10 pt in y) AND within 40 pt centre-to-centre of the candidate. Only a
   direct same-row companion suppresses the span.

The `_boilerplate_kws` local tuple and the `120.0` literal were removed entirely.
Page dimensions are estimated once per `detect_added_characteristics()` call and
reused; no per-candidate overhead was added.

**Root cause closed:** Part 5 truth_index 16 (`800`) sat ~54 pt from
`UNLESS OTHERWISE SPECIFIED` but was outside the tolerance zone. The old 120 pt
sweep suppressed it; the new zone-aware filter lets it through.

**Imports updated** — `is_boilerplate_candidate_text` and `estimate_page_dimensions`
added alongside the existing `span_is_excluded_for_annotation_search` import from
`delta_preservation.reconcile.exclusion` (lines 65–68 of `classify.py`).

### Task 2: Positive and negative regression tests

Added two tests to `TestPhase9ExemplarFamilies` in
`tests/test_added_detection_phase6.py`:

- **`test_plain_integer_800_detected_when_tolerance_block_is_far_away`** — Verifies
  that `800` at drawing-body centre `(615, 581)` survives when
  `UNLESS OTHERWISE SPECIFIED` sits in the bottom-centre tolerance zone at
  `(500, 650)` (>40 pt away, not same-row).
- **`test_plain_integer_in_general_tolerance_block_still_suppressed`** — Verifies
  that `150` whose centre is inside the tolerance zone at `(420, 650)` is still
  suppressed when a same-row boilerplate companion at `(395, 650)` is within
  10 pt vertical / 25 pt centre-to-centre distance.

## Verification

- `rg "_boilerplate_kws\s*=" classify.py` → no matches ✅
- `rg "<= 120\.0" classify.py` → no matches ✅
- `span_is_excluded_for_annotation_search`, `is_boilerplate_candidate_text`,
  `estimate_page_dimensions` all imported from `exclusion` ✅
- `uv run pytest -q tests/test_added_detection_phase6.py tests/test_phase6_asset_regression.py -x` → 53 passed ✅
- `git diff --name-only -- assets/debug_report_part*.json` → no output (historical assets untouched) ✅
