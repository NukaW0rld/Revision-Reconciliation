"""Phase 7 milestone regression checkpoint (TST-01).

Human-readable index of all nine Phase 4-6 fix clusters.  Each test method
references the authoritative exemplar from its phase-specific test file, as
documented in 07-01-AUDIT.md.

Source of truth for cluster-to-exemplar mapping:
  .planning/phases/07-regression-tests-and-verification/07-01-AUDIT.md

Clusters covered:

  1. GDT-01  Compact GD&T token splitting
             test_semantic_extraction::test_extract_semantic_callout_gdt_parsed_compact_token_variants

  2. GDT-02  Word-name GD&T normalization
             test_semantic_extraction::test_extract_semantic_callout_gdt_parsed_word_name_controls

  3. GDT-03  Composite FCF capture
             test_semantic_extraction::test_extract_semantic_callout_gdt_parsed_composite_frame_preserves_all_compartments

  4. CLS-01  Adjacency bleed suppression
             test_classify_bugfixes::TestAdjacencyBleed::test_part1_style_bleed_not_changed

  5. CLS-02  Removed+added pair reconciliation
             test_classify_bugfixes::TestRemovedAddedReconciliation::test_close_compatible_pair_becomes_changed

  6. CLS-03  Asymmetric tolerance detection
             test_classify_phase5_regression::TestPhase5SnapshotExemplars::test_asymmetric_shape_re_matches_part7_exemplar

  7. ADD-01  Missing added rows (Part 8 / Part 9)
             test_phase6_asset_regression::TestPhase6AssetInvariants::test_part8_canonical_added_row_Ø10

  8. ADD-02  False-positive added-row suppression
             test_phase6_asset_regression::TestPhase6Part8Exemplars::test_fragment_only_form_does_not_survive_when_ownership_exists

  9. SNP-01  Title block / revision table exclusion
             test_phase6_exclusion::TestSharedExclusionContract::test_title_block_bottom_right_is_excluded
"""
from __future__ import annotations


class TestMilestoneCoverage:
    """One checkpoint method per Phase 4-6 fix cluster.

    Each method re-invokes the authoritative exemplar test from the phase-specific
    module.  The goal is readability as a milestone artifact, not exhaustive
    re-coverage — the phase-specific files are the authoritative coverage layer.
    """

    # ------------------------------------------------------------------
    # GDT-01: Compact GD&T token splitting
    # ------------------------------------------------------------------

    def test_cluster_gdt01_compact_token_covered(self) -> None:
        """GDT-01 fix is covered by the compact-token parsing exemplar.

        Authoritative exemplar:
        tests/test_semantic_extraction.py::test_extract_semantic_callout_gdt_parsed_compact_token_variants
        """
        from tests.test_semantic_extraction import (
            test_extract_semantic_callout_gdt_parsed_compact_token_variants,
        )
        test_extract_semantic_callout_gdt_parsed_compact_token_variants()

    # ------------------------------------------------------------------
    # GDT-02: Word-name GD&T normalization
    # ------------------------------------------------------------------

    def test_cluster_gdt02_word_name_normalization_covered(self) -> None:
        """GDT-02 fix is covered by the word-name control normalization exemplar.

        Authoritative exemplar:
        tests/test_semantic_extraction.py::test_extract_semantic_callout_gdt_parsed_word_name_controls
        """
        from tests.test_semantic_extraction import (
            test_extract_semantic_callout_gdt_parsed_word_name_controls,
        )
        test_extract_semantic_callout_gdt_parsed_word_name_controls()

    # ------------------------------------------------------------------
    # GDT-03: Composite FCF capture
    # ------------------------------------------------------------------

    def test_cluster_gdt03_composite_fcf_covered(self) -> None:
        """GDT-03 fix is covered by the composite frame compartment exemplar.

        Authoritative exemplar:
        tests/test_semantic_extraction.py::test_extract_semantic_callout_gdt_parsed_composite_frame_preserves_all_compartments
        """
        from tests.test_semantic_extraction import (
            test_extract_semantic_callout_gdt_parsed_composite_frame_preserves_all_compartments,
        )
        test_extract_semantic_callout_gdt_parsed_composite_frame_preserves_all_compartments()

    # ------------------------------------------------------------------
    # CLS-01: Adjacency bleed suppression
    # ------------------------------------------------------------------

    def test_cluster_cls01_adjacency_bleed_covered(self) -> None:
        """CLS-01 fix is covered by the part-1-style bleed exemplar.

        Authoritative exemplar:
        tests/test_classify_bugfixes.py::TestAdjacencyBleed::test_part1_style_bleed_not_changed
        """
        from tests.test_classify_bugfixes import TestAdjacencyBleed
        TestAdjacencyBleed().test_part1_style_bleed_not_changed()

    # ------------------------------------------------------------------
    # CLS-02: Removed+added pair reconciliation
    # ------------------------------------------------------------------

    def test_cluster_cls02_removed_added_reconciliation_covered(self) -> None:
        """CLS-02 fix is covered by the close-compatible-pair reconciliation exemplar.

        Authoritative exemplar:
        tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation::test_close_compatible_pair_becomes_changed
        """
        from tests.test_classify_bugfixes import TestRemovedAddedReconciliation
        TestRemovedAddedReconciliation().test_close_compatible_pair_becomes_changed()

    # ------------------------------------------------------------------
    # CLS-03: Asymmetric tolerance detection
    # ------------------------------------------------------------------

    def test_cluster_cls03_asymmetric_tolerance_covered(self) -> None:
        """CLS-03 fix is covered by the asymmetric-shape regex exemplar.

        Authoritative exemplar:
        tests/test_classify_phase5_regression.py::TestPhase5SnapshotExemplars::test_asymmetric_shape_re_matches_part7_exemplar
        """
        from tests.test_classify_phase5_regression import TestPhase5SnapshotExemplars
        TestPhase5SnapshotExemplars().test_asymmetric_shape_re_matches_part7_exemplar()

    # ------------------------------------------------------------------
    # ADD-01: Missing added rows (Part 8 / Part 9)
    # ------------------------------------------------------------------

    def test_cluster_add01_missing_added_rows_covered(self) -> None:
        """ADD-01 fix is covered by the Part 8 canonical added-row ground-truth exemplar.

        Authoritative exemplar:
        tests/test_phase6_asset_regression.py::TestPhase6AssetInvariants::test_part8_canonical_added_row_Ø10
        """
        import tests.test_phase6_asset_regression as mod
        assert hasattr(mod, "TestPhase6AssetInvariants"), (
            "ADD-01 regression exemplar class missing from "
            "tests/test_phase6_asset_regression.py"
        )
        cls = mod.TestPhase6AssetInvariants
        assert hasattr(cls, "test_part8_canonical_added_row_Ø10"), (
            "Missing canonical ADD-01 exemplar "
            "TestPhase6AssetInvariants::test_part8_canonical_added_row_Ø10"
        )
        cls().test_part8_canonical_added_row_Ø10()

    # ------------------------------------------------------------------
    # ADD-02: False-positive added-row suppression
    # ------------------------------------------------------------------

    def test_cluster_add02_false_positive_suppression_covered(self) -> None:
        """ADD-02 fix is covered by the fragment-suppression ownership exemplar.

        Authoritative exemplar:
        tests/test_phase6_asset_regression.py::TestPhase6Part8Exemplars::test_fragment_only_form_does_not_survive_when_ownership_exists
        """
        from tests.test_phase6_asset_regression import TestPhase6Part8Exemplars
        TestPhase6Part8Exemplars().test_fragment_only_form_does_not_survive_when_ownership_exists()

    # ------------------------------------------------------------------
    # SNP-01: Title block / revision table exclusion
    # ------------------------------------------------------------------

    def test_cluster_snp01_title_block_exclusion_covered(self) -> None:
        """SNP-01 fix is covered by the bottom-right title-block exclusion exemplar.

        Authoritative exemplar:
        tests/test_phase6_exclusion.py::TestSharedExclusionContract::test_title_block_bottom_right_is_excluded
        """
        from tests.test_phase6_exclusion import TestSharedExclusionContract
        TestSharedExclusionContract().test_title_block_bottom_right_is_excluded()
