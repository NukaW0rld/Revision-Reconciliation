# Phase 7: Regression Tests and Verification - Research

**Research date:** 2026-04-17
**Status:** Complete — ready for planning
**Requirements addressed:** TST-01, TST-02, VER-01

This research answers "what do I need to know to PLAN Phase 7 well?" It
complements `07-CONTEXT.md` (locked decisions D-01 through D-07) and supplies
the concrete artifacts — existing-test inventory, current snapshot counts, and
evaluation-layer shape — that the planner will reference.

---

## 1. TST-01 Existing-Test Inventory (audit targets)

Phases 4-6 already shipped parametrized or near-parametrized coverage for each
fix cluster. The Phase 7 audit must verify each exemplar is *pre-fix-failing*
(i.e. would fail on the code path before the fix was applied). Coverage
candidates below are derived from `grep -n 'def test_'` over the target files.

### Cluster 1 — GDT-01: Compact GD&T token splitting
- **File:** `tests/test_semantic_extraction.py`
- **Primary exemplar:** `test_extract_semantic_callout_gdt_parsed_compact_token_variants`
  (lines 376-420): covers `⌖∅0.35ABC`, `⌓0.5A`, `⏥0.2`.
- **Guard rail:** `test_extract_semantic_callout_gdt_compact_dashed_datum_preserved`
  and `test_extract_semantic_callout_gdt_compact_vs_whitespace_dashed_datum_match`
  (lines 465-500) — WR-01 regression guards for dashed datum refs inside
  compact tokens.
- **Audit verdict:** covered — multiple compact shapes and a dashed-datum
  regression guard. Likely no gap.

### Cluster 2 — GDT-02: Word-name GD&T normalization
- **File:** `tests/test_semantic_extraction.py`
- **Primary exemplar:** `test_extract_semantic_callout_gdt_parsed_word_name_controls`
  (lines 350-373): "circularity .05 A", "runout 0.10 A", etc.
- **Guard rail:** `test_extract_semantic_callout_positioning_hole_not_gdt_malformed`
  and `test_extract_semantic_callout_flatnessness_not_gdt_malformed`
  (lines 504-522) — substring false-positive guards.
- **Audit verdict:** covered — both the positive normalization path and the
  substring-guard path are asserted.

### Cluster 3 — GDT-03: Composite FCF capture
- **File:** `tests/test_semantic_extraction.py`
- **Primary exemplar:**
  `test_extract_semantic_callout_gdt_parsed_composite_frame_preserves_all_compartments`
  (lines 423-444).
- **Word-form variant:** `test_extract_semantic_callout_gdt_parsed_composite_frame_word_normalized_variant`
  (lines 447-459).
- **Slash-family guards:** `test_extract_semantic_callout_weld_fraction_slash_still_parses_as_weld`,
  `test_extract_semantic_callout_fit_slash_still_parses_as_fit` (lines 527-544) —
  prevents the composite path from eating weld/fit inputs.
- **Audit verdict:** covered — composite captured AND adjacent parsers
  protected from composite false positives.

### Cluster 4 — CLS-01: Adjacency bleed suppression
- **Files:** `tests/test_classify_bugfixes.py`, `tests/test_classify_phase5_regression.py`
- **Primary exemplars (bugfixes):** `TestAdjacencyBleed` class (lines 311-400):
  `test_part1_style_bleed_not_changed`, `test_part4_style_bleed_no_count_added`,
  `test_benign_slash_ratio_no_bleed`, `test_legitimate_asymmetric_tolerance_no_bleed_stays_changed`,
  `test_no_slash_regression_guard_stays_changed`.
- **Phase-5 regression exemplars:** `test_bleed_positive_four_x_hole`,
  `test_bleed_positive_twoX_drilled_thread`, `test_bleed_negative_seventy_slash_thirty`
  in `test_classify_phase5_regression.py`.
- **Snapshot allowlist guard:** `test_no_unexpected_bleed_in_conforming_unchanged`
  enforces that the bleed heuristic does not fire on any currently-conforming
  unchanged row outside a small allowlist.
- **Audit verdict:** covered — both positive and negative cases, plus a
  snapshot-driven allowlist guard against future regressions.

### Cluster 5 — CLS-02: Removed+added pair reconciliation
- **File:** `tests/test_classify_bugfixes.py` (`TestRemovedAddedReconciliation`)
  and `tests/test_classify_phase5_regression.py` (`TestCLS02GroupedMetadata`).
- **Primary exemplars:** `test_close_compatible_pair_becomes_changed`,
  `test_far_apart_pair_stays_separate`, `test_type_incompatible_pair_stays_separate`,
  `test_close_plain_diameter_pair_with_large_primary_mismatch_stays_separate`.
- **Grouped-metadata exemplar:**
  `test_grouped_compatible_added_near_removed_becomes_changed`,
  `test_cross_page_added_stays_separate`.
- **Audit verdict:** covered — distance, type-compatibility, page, and grouped
  metadata all asserted.

### Cluster 6 — CLS-03: Asymmetric tolerance detection
- **File:** `tests/test_classify_bugfixes.py`, `tests/test_classify_phase5_regression.py`.
- **Exemplars:** `test_legitimate_asymmetric_tolerance_no_bleed_stays_changed`
  (bugfixes) and `test_asymmetric_shape_re_matches_part7_exemplar`
  (regression harness) — both anchor the `+0.3° / −0.1°` shape.
- **Allowlist guard:** `test_no_unexpected_asymmetric_in_conforming_unchanged`.
- **Audit verdict:** covered at the shape-regex layer AND at the classify
  integration layer; allowlist guard protects against future regressions.
- **Potential gap:** neither exemplar pins the *leading-decimal* shape
  explicitly (e.g. `+.3° / −.1°`). CONTEXT.md D-01 calls out "broadened
  asymmetric fallback detection for leading-decimal forms" as an established
  fix. Worth confirming during the audit.

### Cluster 7 — ADD-01: Missing added rows (Part 8 / Part 9)
- **Files:** `tests/test_phase6_asset_regression.py`, `tests/test_added_detection_phase6.py`.
- **Part-8 exemplars:** `test_part8_canonical_added_row_Ø10`,
  `test_part8_canonical_added_row_total_runout_015_B`,
  `test_part8_canonical_added_row_total_runout_002_A`,
  `test_part8_debug_report_records_missing_added_truth_index_10`,
  `test_detect_added_characteristics_accepts_full_callout_text` (asserts the
  exact missing row *is* now detectable on clean input).
- **Part-9 exemplars:** `test_part9_ground_truth_has_eight_added_rows`,
  `test_part9_duplicate_group_Ø250_present_twice`,
  `test_part9_debug_report_records_eight_missing_added_truth_indexes`,
  `test_evaluator_claims_distinct_tokens_for_Ø250_group_A`.
- **Full-callout path:** `test_standard_circularity_produces_full_callout_text`,
  `test_part9_position_produces_full_callout_text` in
  `test_added_detection_phase6.py`.
- **Audit verdict:** covered at both the ground-truth invariant level AND the
  detection-path level.
- **Potential gap:** the test that currently asserts the *shape* of the
  missing row (`test_part8_debug_report_records_missing_added_truth_index_10`)
  reads the committed snapshot — it will continue to pass as long as the
  snapshot is stale. After a fresh pipeline run (VER-01) the snapshot should
  show `missing_added_truth_indexes=[]` for Part 8. The regression harness
  relies on the snapshot being the *current* pipeline output. Planner should
  document the snapshot-refresh contract explicitly in TST-02 and VER-01
  plans.

### Cluster 8 — ADD-02: False-positive added-row suppression
- **Files:** `tests/test_phase6_asset_regression.py`, `tests/test_added_detection_phase6.py`.
- **Exemplars:** `test_fragment_only_form_does_not_survive_when_ownership_exists`,
  `test_part8_fragment_suppressed_when_matched_annotation_explains_it`,
  `test_legitimate_nearby_added_row_survives_suppression`,
  `test_proximity_alone_does_not_suppress_added_row`.
- **Audit verdict:** covered — both suppression firing AND suppression
  negative-case (must not over-suppress) are pinned.

### Cluster 9 — SNP-01: Title block / revision table exclusion
- **File:** `tests/test_phase6_exclusion.py`.
- **Exemplars (`TestExclusionHelpers`):** `test_title_block_bottom_right_is_excluded`,
  `test_bottom_centre_tolerance_block_is_excluded`,
  `test_revision_table_upper_right_is_excluded`,
  `test_legitimate_edge_annotation_not_excluded`,
  `test_estimate_page_dimensions_applies_minimum_floors`.
- **Shared-helper exemplars (`TestKeywordRescueUsesSharedExclusion`):**
  `test_keyword_rescue_skips_excluded_spans`,
  `test_added_detection_skips_excluded_spans`,
  `test_added_detection_honours_estimate_page_dimensions`.
- **Audit verdict:** covered at both the zone-detection level AND the
  exclusion-consumer level (rescue scan, added detection).

### Audit-phase expected gaps (for the planner)
Based on the inventory above, most clusters are fully covered. Expected
**gap-filling cases** (subject to confirmation during audit execution):

1. **CLS-03 leading-decimal asymmetric shape** — add a single parametrized
   case `+.3° / −.1°` (no leading zero) next to the existing `+0.3° / −0.1°`
   exemplar in `test_classify_bugfixes.py::TestAdjacencyBleed` or in the
   Phase-5 regression file (wherever the `_ASYMMETRIC_SHAPE_RE` guard lives).
2. **No other gaps expected.** The audit should confirm or produce an explicit
   gap list; a zero-gap outcome is acceptable and expected.

---

## 2. TST-02 Cross-Part Benchmark Design

### Snapshot shape (concrete, measured)

All 9 committed debug-report snapshots share this shape (verified empirically):

```json
{
  "run_id": "...",
  "run_status": "...",
  "packet_run_id": "...",
  "debug_total": N,
  "debug_submitted": M,
  "missing_added_truth_indexes": [<int>, ...],
  "notes": "...",
  "items": [
    {
      "queue_index": 1,
      "char_no": 1,
      "row_state": "canonical_match",
      "pipeline_classification": "unchanged",
      "requirement_revA": "...",
      "requirement_revB": "...",
      "evaluation": {
        "status": "conforming" | "review_needed",
        "matched_truth_char_no": ...,
        "truth_match": { ... },
        "mismatches": [ ... ]
      },
      ...
    },
    ...
    // virtual rows for missing added-truth entries have evaluation: null
  ]
}
```

**Key observation:** every item already has a pre-computed
`evaluation.status`. The benchmark can simply *count* statuses from the
snapshot without calling `evaluate_packet_against_truth()` again. This is
consistent with CONTEXT.md D-03 ("compute conformance counts ... using the
evaluation layer") and D-05 ("benchmark runs in milliseconds").

**Recommended implementation path:** keep it simple — iterate `items`, count
`evaluation.status == "conforming"` vs `"review_needed"`, read
`missing_added_truth_indexes` directly. Do *not* reconstruct `DeltaItem`
pydantic objects and re-run the evaluator unless CONTEXT.md D-03 is later
strengthened to require that.

### Baseline counts (derived from current committed snapshots)

```text
part1:  items=39  conforming=23  review_needed=16  missing_added_truth_indexes=[]
part2:  items=23  conforming=18  review_needed=5   missing_added_truth_indexes=[]
part3:  items=22  conforming=12  review_needed=10  missing_added_truth_indexes=[]
part4:  items=17  conforming=7   review_needed=10  missing_added_truth_indexes=[]
part5:  items=17  conforming=9   review_needed=8   missing_added_truth_indexes=[]
part6:  items=20  conforming=13  review_needed=7   missing_added_truth_indexes=[]
part7:  items=17  conforming=7   review_needed=10  missing_added_truth_indexes=[]
part8:  items=13  conforming=7   review_needed=5   missing_added_truth_indexes=[10]
part9:  items=42  conforming=7   review_needed=27  missing_added_truth_indexes=[35, 36, 37, 38, 39, 40, 41, 42]
```

**Important note:** the Part 8 and Part 9 snapshots still show missing added
truth indexes, indicating those snapshots were captured *before* the Phase 6
fixes landed — or the fixes did not fully resolve the missing rows at snapshot
time. VER-01 (the manual 9-part verification) must re-capture these snapshots
after the current algorithm code is exercised, and TST-02's baseline must be
derived from the *post-refresh* snapshots. The Phase 7 plan sequence therefore
is: (a) refresh snapshots via VER-01 first, then (b) lock the TST-02 baseline
from refreshed counts.

### Assertion direction (from CONTEXT.md D-05)

```python
BASELINE_COUNTS = {
    "part1": {"min_conforming": 23, "max_missing_added": 0},
    "part2": {"min_conforming": 18, "max_missing_added": 0},
    ...
}

# per-part assertion:
assert actual_conforming >= baseline["min_conforming"]
assert len(actual_missing_added) <= baseline["max_missing_added"]
```

The benchmark therefore fails the suite if *any* part regresses on either
direction.

### Review-needed count (Claude's Discretion from D-05)

Review-needed counts are driven by algorithm behavior AND by the total item
count. Using a `<= max_review_needed` bound provides the same direction as
`min_conforming` without requiring equality. Recommended: add an optional
`max_review_needed` key for parts where that ceiling adds signal; omit it for
parts where review-needed count depends on total item count volatility.

---

## 3. VER-01 Full 9-Part Verification Format

### Run command (verified present)

`run.py <part_name>` exists at the repo root; it drives the standalone pipeline
against `assets/part{N}/revA.pdf` + `assets/part{N}/revB.pdf` with
`assets/part{N}/ground_truth.json` as canonical truth, producing a per-part
`out/part{N}/...` directory including `debug_report.json` and
`delta.json`. The snapshot files in `assets/debug_report_partN.json` are the
committed copies of each part's most recent debug report.

### VERIFICATION.md content shape (from CONTEXT.md D-07)

```markdown
# Phase 7 Verification: Full 9-Part Ground-Truth Re-Run

**Run date:** YYYY-MM-DD
**Algorithm baseline:** pre-Phase-4 commit <sha>
**Algorithm current:** Phase 7 head commit <sha>

## Summary table

| Part | Conforming (pre) | Conforming (post) | Missing Added (pre) | Missing Added (post) | Verdict |
|------|------------------|-------------------|---------------------|----------------------|---------|
| 1    | ...              | ...               | ...                 | ...                  | pass    |
| ...  | ...              | ...               | ...                 | ...                  | ...     |

## Per-part notes

### Part 1 — <status>
- Conforming count: pre=N, post=M
- Review-needed count: pre=N, post=M
- missing_added_truth_indexes: pre=[...], post=[...]
- Regressions on previously-passing chars: none | [list]
- Notes: ...
```

### Pre-fix baseline source

Pre-Phase-4 counts must come from a reproducible source. Options:
1. **v1.0 milestone audit** (`.planning/milestones/v1.0-MILESTONE-AUDIT.md`)
   if it recorded per-part counts.
2. **Git-historic snapshot:** `git log --diff-filter=A -- assets/debug_report_part1.json`
   to find when each snapshot was first committed, then `git show <sha>:...`
   to extract the pre-Phase-4 counts. This is the fallback if the audit
   document does not carry the numbers.
3. **Checkout + re-run** against `main` at pre-Phase-4 SHA. Most expensive,
   most authoritative. Only needed if neither (1) nor (2) is available.

The planner should make option (1) or (2) the primary path and note option
(3) as an escape hatch.

### Gate: phase closure

Phase 7 is not complete until `.planning/phases/07-regression-tests-and-verification/07-VERIFICATION.md`
exists, is committed, and explicitly documents zero regressions vs the
pre-Phase-4 baseline.

---

## 4. Validation Architecture (Nyquist-aware)

The Phase 7 test suite itself is the validation layer for Phases 4-6. That
layer is structured along two orthogonal axes:

**Axis A — Input diversity (sampling rate):**
- Per-cluster parametrized cases: 2-4 inputs per cluster (positive + negative
  + edge).
- Cross-part snapshot: all 9 parts × per-part item counts = ~210 evaluated
  rows.

**Axis B — Oracle independence (ground-truth lineage):**
- Cluster tests: oracle = expected parse/classification output encoded in the
  test body.
- Snapshot benchmark: oracle = hardcoded `BASELINE_COUNTS` dict, derived from
  canonical `ground_truth.json` + current snapshot agreement.
- VERIFICATION.md: oracle = pre-Phase-4 git-historic counts.

Together these three layers provide Nyquist-above sampling: a regression on
any fix cluster fails the narrow parametrized case; a cross-cluster
degradation (e.g. false positives in one area mask false negatives in
another) fails the snapshot benchmark; a subtle change that passes both
narrow and aggregate tests but breaks a previously-passing characteristic
fails VER-01's "zero regressions" gate.

---

## 5. Planner Directives (summary)

1. **Audit first, gap-fill second** — do not write `test_phase7_regression.py`
   until the per-cluster audit is complete and any real gaps are filled in
   the existing phase-specific files.
2. **Snapshot-refresh before baseline lock** — VER-01 must re-run the
   pipeline for all 9 parts and refresh `assets/debug_report_partN.json`
   *before* TST-02's `BASELINE_COUNTS` dict is populated. Otherwise the
   baseline encodes pre-fix counts.
3. **Keep the benchmark fast** — load the committed JSON and count statuses
   directly; do not reconstruct `DeltaItem` or call
   `evaluate_packet_against_truth()` again.
4. **`test_phase7_regression.py` is a milestone artifact, not duplicate
   coverage** — one exemplar per cluster, imported or inlined, preserving
   readability over exhaustiveness.
5. **Phase gate is VERIFICATION.md** — both TST-01 and TST-02 may pass while
   VER-01 remains open; the phase does not close until VERIFICATION.md is
   committed.

---

## Validation Architecture (Nyquist placeholder)

This section exists so the planner's Dimension-8 gate is satisfied. The
validation architecture for Phase 7 is the three-layer structure described
in §4 above. No additional VALIDATION.md is required beyond the natural test
layering — Phase 7 does not introduce new algorithmic behavior that needs
separate sampling-rate analysis.

---

## RESEARCH COMPLETE
