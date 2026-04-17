"""Focused evaluator coverage for unique, duplicate, and ambiguous added-row selection.

This module tests ``select_truth_row_for_item`` in ``delta_preservation.evaluation.conformance``
for correctness under five conditions:

1. Unique exact-text: single matching added truth row → accepted immediately
2. Duplicate with bbox containment: packet revB.bbox contains exactly one truth center → that row wins
3. Duplicate with nearest-center fallback: no center inside bbox but exactly one uniquely nearest
   within ``ADDED_TRUTH_TIEBREAK_MAX_DISTANCE_PT`` → that row wins
4. Duplicate still ambiguous: two or more rows still qualify → returns truth_ambiguity
5. Missing or invalid revB evidence: duplicate matches stay ambiguous
"""
from __future__ import annotations

import pytest

from delta_preservation.evaluation.conformance import (
    ADDED_POOL_TOKEN_PREFIX,
    TRUTH_AMBIGUITY_CODE,
    select_truth_row_for_item,
)
from delta_preservation.evaluation.contracts import GroundTruthCharacteristic
from delta_preservation.types import DeltaItem, Evidence

# ---------------------------------------------------------------------------
# Part 9 ground-truth constants (from assets/part9/ground_truth.json)
# These are the six duplicate added rows that come in pairs of 3.
# ---------------------------------------------------------------------------

# Group A: around (162, 328), (178, 348), (156, 364)
_P9_A0_REQ = "Ø.250 ±.008"
_P9_A0_CENTER = (162.0, 328.0)
_P9_A1_REQ = "⌖ ∅.015 D H"
_P9_A1_CENTER = (178.0, 348.0)
_P9_A2_REQ = "↧.50 ±.05"
_P9_A2_CENTER = (156.0, 364.0)

# Group B: around (602, 288), (618, 306), (594, 322)
_P9_B0_REQ = "Ø.250 ±.008"
_P9_B0_CENTER = (602.0, 288.0)
_P9_B1_REQ = "⌖ ∅.015 D H"
_P9_B1_CENTER = (618.0, 306.0)
_P9_B2_REQ = "↧.50 ±.05"
_P9_B2_CENTER = (594.0, 322.0)


def _make_added_truth(requirement: str, center_revB: tuple[float, float]) -> GroundTruthCharacteristic:
    """Return a minimal added GroundTruthCharacteristic."""
    return GroundTruthCharacteristic(
        char_no=None,
        classification="added",
        requirement_revB=requirement,
        snippet_center_revA=None,
        snippet_center_revB=center_revB,
    )


def _make_added_item(
    requirement: str,
    bbox: list[float] | None = None,
    page: int = 0,
) -> DeltaItem:
    """Return a minimal added DeltaItem, optionally with revB Evidence."""
    revB = Evidence(page=page, bbox=bbox, image_path=None) if bbox is not None else None
    return DeltaItem(
        char_no=None,
        status="added",
        confidence=0.9,
        reasons=["test"],
        scores={},
        revA=None,
        revB=revB,
        requirement_revB=requirement,
    )


def _make_added_item_with_bad_revB(requirement: str, bad_bbox: list[float]) -> DeltaItem:
    """Return an added DeltaItem with a malformed revB bbox."""
    bad_evidence = Evidence(page=0, bbox=bad_bbox, image_path=None)
    return DeltaItem(
        char_no=None,
        status="added",
        confidence=0.9,
        reasons=["test"],
        scores={},
        revA=None,
        revB=bad_evidence,
        requirement_revB=requirement,
    )


def _sel(item: DeltaItem, truth_rows: list[GroundTruthCharacteristic], reserved: set[int] | None = None):
    """Convenience wrapper around select_truth_row_for_item."""
    return select_truth_row_for_item(item, truth_rows, reserved_added_truth_indexes=reserved or set())


class TestDuplicateAddedTruthSelection:
    """Tests for the duplicate added-truth tie-break in select_truth_row_for_item."""

    def _pool_Ø250(self) -> list[GroundTruthCharacteristic]:
        """Two added truth rows sharing requirement text 'Ø.250 ±.008'."""
        return [
            _make_added_truth(_P9_A0_REQ, _P9_A0_CENTER),  # pool index 0
            _make_added_truth(_P9_B0_REQ, _P9_B0_CENTER),  # pool index 1
        ]

    def _pool_position_mark(self) -> list[GroundTruthCharacteristic]:
        """Two added truth rows sharing requirement text '⌖ ∅.015 D H'."""
        return [
            _make_added_truth(_P9_A1_REQ, _P9_A1_CENTER),  # pool index 0
            _make_added_truth(_P9_B1_REQ, _P9_B1_CENTER),  # pool index 1
        ]

    def _pool_depth(self) -> list[GroundTruthCharacteristic]:
        """Two added truth rows sharing requirement text '↧.50 ±.05'."""
        return [
            _make_added_truth(_P9_A2_REQ, _P9_A2_CENTER),  # pool index 0
            _make_added_truth(_P9_B2_REQ, _P9_B2_CENTER),  # pool index 1
        ]

    # ------------------------------------------------------------------
    # 1. Unique exact-text: fast path unchanged
    # ------------------------------------------------------------------

    def test_unique_exact_text_selects_single_matching_row(self):
        """A unique exact-text match selects without consulting bbox evidence."""
        truth_rows = [
            _make_added_truth("⌓ .02 A B C", (750.0, 430.0)),  # index 0 — unique
            _make_added_truth("⏥ .01", (726.0, 446.0)),         # index 1 — unique
        ]
        item = _make_added_item("⌓ .02 A B C")  # no bbox needed
        selection = _sel(item, truth_rows)
        assert selection.truth_row is not None, "Expected the unique truth row to be selected"
        assert selection.truth_index == 0
        assert selection.truth_row.snippet_center_revB == (750.0, 430.0)
        assert selection.ambiguity_message is None

    def test_unique_exact_text_selects_second_row_when_first_reserved(self):
        """Reserved index 0 → only index 1 qualifies; unique → selected."""
        truth_rows = [
            _make_added_truth("⌓ .02 A B C", (750.0, 430.0)),  # index 0 — reserved
            _make_added_truth("⌓ .02 A B C", (200.0, 200.0)),  # index 1 — available
        ]
        item = _make_added_item("⌓ .02 A B C")
        selection = _sel(item, truth_rows, reserved={0})
        assert selection.truth_row is not None
        assert selection.truth_index == 1

    # ------------------------------------------------------------------
    # 2. Bbox containment disambiguates duplicates
    # ------------------------------------------------------------------

    def test_bbox_containment_selects_group_a_Ø250(self):
        """Packet bbox contains group-A center → selects index 0 (Ø.250 ±.008 group A)."""
        truth_rows = self._pool_Ø250()
        bbox = [130.0, 300.0, 200.0, 360.0]  # contains (162, 328); not (602, 288)
        item = _make_added_item(_P9_A0_REQ, bbox=bbox)
        selection = _sel(item, truth_rows)
        assert selection.truth_row is not None, "Expected tie-break to select group-A row"
        assert selection.truth_index == 0
        assert selection.truth_row.snippet_center_revB == _P9_A0_CENTER
        assert selection.ambiguity_message is None

    def test_bbox_containment_selects_group_b_Ø250(self):
        """Packet bbox contains group-B center → selects index 1 (Ø.250 ±.008 group B)."""
        truth_rows = self._pool_Ø250()
        bbox = [570.0, 260.0, 640.0, 320.0]  # contains (602, 288); not (162, 328)
        item = _make_added_item(_P9_B0_REQ, bbox=bbox)
        selection = _sel(item, truth_rows)
        assert selection.truth_row is not None, "Expected tie-break to select group-B row"
        assert selection.truth_index == 1
        assert selection.truth_row.snippet_center_revB == _P9_B0_CENTER
        assert selection.ambiguity_message is None

    def test_bbox_containment_selects_group_a_position_mark(self):
        """Packet bbox around group-A center disambiguates ⌖ ∅.015 D H → index 0."""
        truth_rows = self._pool_position_mark()
        bbox = [150.0, 320.0, 210.0, 375.0]  # contains (178, 348)
        item = _make_added_item(_P9_A1_REQ, bbox=bbox)
        selection = _sel(item, truth_rows)
        assert selection.truth_row is not None
        assert selection.truth_index == 0
        assert selection.truth_row.snippet_center_revB == _P9_A1_CENTER

    def test_bbox_containment_selects_group_b_position_mark(self):
        """Packet bbox around group-B center disambiguates ⌖ ∅.015 D H → index 1."""
        truth_rows = self._pool_position_mark()
        bbox = [590.0, 280.0, 650.0, 330.0]  # contains (618, 306)
        item = _make_added_item(_P9_B1_REQ, bbox=bbox)
        selection = _sel(item, truth_rows)
        assert selection.truth_row is not None
        assert selection.truth_index == 1
        assert selection.truth_row.snippet_center_revB == _P9_B1_CENTER

    def test_bbox_containment_accepts_spacing_variant_for_position_mark(self):
        """Spacing-only variants must still reach the duplicate-row bbox tie-break."""
        truth_rows = self._pool_position_mark()
        bbox = [590.0, 280.0, 650.0, 330.0]  # contains (618, 306)
        item = _make_added_item("⌖∅ .015 D H", bbox=bbox)
        selection = _sel(item, truth_rows)
        assert selection.truth_row is not None
        assert selection.truth_index == 1
        assert selection.truth_row.snippet_center_revB == _P9_B1_CENTER

    def test_bbox_containment_selects_group_a_depth(self):
        """Packet bbox around group-A center disambiguates ↧.50 ±.05 → index 0."""
        truth_rows = self._pool_depth()
        bbox = [125.0, 335.0, 190.0, 395.0]  # contains (156, 364)
        item = _make_added_item(_P9_A2_REQ, bbox=bbox)
        selection = _sel(item, truth_rows)
        assert selection.truth_row is not None
        assert selection.truth_index == 0
        assert selection.truth_row.snippet_center_revB == _P9_A2_CENTER

    def test_bbox_containment_selects_group_b_depth(self):
        """Packet bbox around group-B center disambiguates ↧.50 ±.05 → index 1."""
        truth_rows = self._pool_depth()
        bbox = [565.0, 295.0, 625.0, 350.0]  # contains (594, 322)
        item = _make_added_item(_P9_B2_REQ, bbox=bbox)
        selection = _sel(item, truth_rows)
        assert selection.truth_row is not None
        assert selection.truth_index == 1
        assert selection.truth_row.snippet_center_revB == _P9_B2_CENTER

    def test_bbox_containment_accepts_spacing_variant_for_depth(self):
        """Spacing-only variants must still resolve duplicate depth rows."""
        truth_rows = self._pool_depth()
        bbox = [565.0, 295.0, 625.0, 350.0]  # contains (594, 322)
        item = _make_added_item("↧ .50 ±.05", bbox=bbox)
        selection = _sel(item, truth_rows)
        assert selection.truth_row is not None
        assert selection.truth_index == 1
        assert selection.truth_row.snippet_center_revB == _P9_B2_CENTER

    # ------------------------------------------------------------------
    # 3. Nearest-center fallback within tolerance
    # ------------------------------------------------------------------

    def test_nearest_center_fallback_selects_closest_within_tolerance(self):
        """No truth center inside bbox → unique nearest center within tolerance wins.

        Group A center: (162, 328), Group B center: (602, 288).
        Bbox centered at (200, 360) → distance to A ≈ 47 pt, distance to B ≈ 410 pt.
        Group A wins if the threshold allows ~47 pt.
        """
        truth_rows = self._pool_Ø250()
        # bbox just outside group-A center but far from group B
        bbox = [185.0, 345.0, 215.0, 375.0]  # center (200, 360); closest to A
        item = _make_added_item(_P9_A0_REQ, bbox=bbox)
        selection = _sel(item, truth_rows)
        assert selection.truth_row is not None, (
            "Expected nearest-center fallback to select group-A when group-A is uniquely closest"
        )
        assert selection.truth_index == 0

    # ------------------------------------------------------------------
    # 4. Still ambiguous when two rows still qualify
    # ------------------------------------------------------------------

    def test_bbox_contains_both_centers_stays_ambiguous(self):
        """bbox that encloses both truth centers → ambiguity preserved."""
        truth_rows = self._pool_Ø250()
        bbox = [100.0, 250.0, 700.0, 400.0]  # encloses both (162, 328) and (602, 288)
        item = _make_added_item(_P9_A0_REQ, bbox=bbox)
        selection = _sel(item, truth_rows)
        assert selection.truth_row is None
        assert selection.ambiguity_message is not None
        assert "multiple canonical added truth rows share the same normalized requirement text" in (
            selection.ambiguity_message
        )

    def test_equidistant_centers_stays_ambiguous(self):
        """Two truth rows equidistant from packet bbox center → preserves ambiguity."""
        left_center = (100.0, 300.0)
        right_center = (300.0, 300.0)
        truth_rows = [
            _make_added_truth("Ø.500 ±.010", left_center),
            _make_added_truth("Ø.500 ±.010", right_center),
        ]
        # bbox center at (200, 300) — equidistant from both
        bbox = [190.0, 290.0, 210.0, 310.0]
        item = _make_added_item("Ø.500 ±.010", bbox=bbox)
        selection = _sel(item, truth_rows)
        assert selection.truth_row is None
        assert selection.ambiguity_message is not None

    # ------------------------------------------------------------------
    # 5. Missing or invalid revB evidence stays ambiguous
    # ------------------------------------------------------------------

    def test_no_revB_evidence_stays_ambiguous(self):
        """Duplicate matches without any revB bbox remain truth_ambiguity."""
        truth_rows = self._pool_Ø250()
        item = _make_added_item(_P9_A0_REQ, bbox=None)
        selection = _sel(item, truth_rows)
        assert selection.truth_row is None
        assert selection.ambiguity_message is not None
        assert "multiple canonical added truth rows share the same normalized requirement text" in (
            selection.ambiguity_message
        )

    def test_empty_bbox_stays_ambiguous(self):
        """An empty bbox list is treated as missing evidence → ambiguity preserved."""
        truth_rows = self._pool_Ø250()
        item = _make_added_item_with_bad_revB(_P9_A0_REQ, bad_bbox=[])
        selection = _sel(item, truth_rows)
        assert selection.truth_row is None
        assert selection.ambiguity_message is not None

    def test_three_element_bbox_stays_ambiguous(self):
        """bbox with only 3 elements is invalid → ambiguity preserved."""
        truth_rows = self._pool_Ø250()
        item = _make_added_item_with_bad_revB(_P9_A0_REQ, bad_bbox=[100.0, 100.0, 200.0])
        selection = _sel(item, truth_rows)
        assert selection.truth_row is None
        assert selection.ambiguity_message is not None

    # ------------------------------------------------------------------
    # 6. Reserved index excluded from consideration
    # ------------------------------------------------------------------

    def test_already_reserved_truth_index_is_excluded(self):
        """Reserved index 0 is skipped; only index 1 remains → unique exact-text match."""
        truth_rows = self._pool_Ø250()
        # Tight bbox around group-A, but index 0 is already reserved.
        # Index 1 (group B) is the only candidate with matching text → unique match.
        bbox = [130.0, 300.0, 200.0, 360.0]
        item = _make_added_item(_P9_A0_REQ, bbox=bbox)
        selection = _sel(item, truth_rows, reserved={0})
        assert selection.truth_row is not None
        assert selection.truth_index == 1
