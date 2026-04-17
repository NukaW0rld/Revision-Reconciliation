# Phase 7 TST-01 Audit

**Audit date:** 2026-04-17
**Auditor:** Cascade (automated)
**Source of truth:** .planning/phases/07-regression-tests-and-verification/07-RESEARCH.md §1

## Cluster audit table

| # | Cluster | Requirement | Existing exemplar | Pre-fix failure reasoning | Gap verdict |
|---|---------|-------------|-------------------|---------------------------|-------------|
| 1 | GDT-01 compact token splitting | GDT-01 | `tests/test_semantic_extraction.py::test_extract_semantic_callout_gdt_parsed_compact_token_variants` | Before Phase 4, `extract_semantic_callout` did not split compact tokens like `⌖∅0.35ABC`; the test would fail because `callout.gdt_controls` would be `None` instead of listing the parsed control and datum. | covered |
| 2 | GDT-02 word-name normalization | GDT-02 | `tests/test_semantic_extraction.py::test_extract_semantic_callout_gdt_parsed_word_name_controls` | Before Phase 4, word-name GD&T controls like `"circularity .05 A"` were not recognized as GD&T; the test would fail because `callout.callout_type` would not be `"gdt"`. | covered |
| 3 | GDT-03 composite FCF capture | GDT-03 | `tests/test_semantic_extraction.py::test_extract_semantic_callout_gdt_parsed_composite_frame_preserves_all_compartments` | Before Phase 4, composite FCF parsing captured only the first compartment; the test would fail because `callout.gdt_controls` would not contain entries for both compartments. | covered |
| 4 | CLS-01 adjacency bleed suppression | CLS-01 | `tests/test_classify_bugfixes.py::TestAdjacencyBleed::test_part1_style_bleed_not_changed` | Before Phase 5, `_looks_like_adjacency_bleed()` did not exist; the test asserting `delta.status != "changed"` for the bleed span would fail because `count_added` fired unconditionally. | covered |
| 5 | CLS-02 removed+added reconciliation | CLS-02 | `tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation::test_close_compatible_pair_becomes_changed` | Before Phase 5, `reconcile_removed_added_pairs()` did not exist; the test would fail because the removed item would stay `"removed"` and never be promoted to `"changed"`. | covered |
| 6 | CLS-03 asymmetric tolerance detection | CLS-03 | `tests/test_classify_phase5_regression.py::TestPhase5SnapshotExemplars::test_asymmetric_shape_re_matches_part7_exemplar` | Before Phase 5, `_ASYMMETRIC_SHAPE_RE` did not exist; the test asserting `_ASYMMETRIC_SHAPE_RE.search(revB) is not None` would raise `ImportError` or `AttributeError`. Leading-decimal variant (`+.3° / −.1°`) not yet pinned by an explicit regex-level parametrized case. | gap_filling_required |
| 7 | ADD-01 missing added rows | ADD-01 | `tests/test_phase6_asset_regression.py::TestPhase6AssetInvariants::test_part8_canonical_added_row_Ø10` | Before Phase 6, the pipeline had no path to produce the full-callout-text added row for `Ø10`; the test would fail because the characteristic was absent from ground-truth evaluation. | covered |
| 8 | ADD-02 false-positive suppression | ADD-02 | `tests/test_phase6_asset_regression.py::TestPhase6Part8Exemplars::test_fragment_only_form_does_not_survive_when_ownership_exists` | Before Phase 6, `_is_span_explained_by_match()` did not exist; the fragment span would survive as an added row and the suppression assertion would fail. | covered |
| 9 | SNP-01 title block / revision exclusion | SNP-01 | `tests/test_phase6_exclusion.py::TestSharedExclusionContract::test_title_block_bottom_right_is_excluded` | Before Phase 6, `_estimate_page_dimensions()` and the exclusion zone helpers did not exist; the test asserting that title-block-region spans are excluded would fail because no exclusion check was performed. | covered |

## Gap-filling actions

One row has verdict `gap_filling_required`:

### Cluster 6 — CLS-03 leading-decimal asymmetric shape

- **File:** `tests/test_classify_phase5_regression.py`
- **Test name:** `test_asymmetric_shape_re_matches_leading_decimal_variants`
- **Input(s):** `"2X 22.0° +.3° / −.1°"`, `"2X 22.0° +.3°/−.1°"`, `"22.0° +.5° / −.2°"` (all leading-decimal tolerance magnitudes)
- **Expected output:** `_ASYMMETRIC_SHAPE_RE.search(revB_text) is not None` for every input
- **Reason:** The existing exemplar only tests the full-form `+0.3° / −0.1°` (with leading zero). The `_ASYMMETRIC_SHAPE_RE` pattern includes `\.?` before `\d+` and a comment noting leading-decimal is OK, but no parametrized case pins this property, so a future regex edit could silently break the leading-decimal path without any test failure.

Additionally, a negative guard should be added:

- **Test name:** `test_asymmetric_shape_re_does_not_match_plain_fractional_ratios`
- **Input(s):** `".3 / .1"`, `"70 / 30"`
- **Expected output:** `_ASYMMETRIC_SHAPE_RE.search(revB_text) is None` for every input
- **Reason:** Pins that plain numeric ratios without the `+`/`−` sign markers are not misidentified as asymmetric tolerances.
