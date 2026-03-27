"""
Tests for drawing grid inference.

Success criteria (from feature spec):
- part1/revA and part1/revB: columns 8→1 left-to-right, rows H→A top-to-bottom
- part2/revA and part2/revB: columns 4→1 left-to-right, rows D→A top-to-bottom
- part3/revA and part3/revB: columns 2→1 left-to-right, rows B→A top-to-bottom
- part4/revA:                 columns 4→1 left-to-right, rows D→A top-to-bottom
"""

from pathlib import Path

import pytest

from delta_preservation.vision.grid import (
    DrawingGrid,
    _parse_zone_parts,
    infer_grid,
    parse_zone_from_reference_location,
)

ASSETS = Path(__file__).parent.parent / "assets"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def col_order(grid: DrawingGrid) -> list[str]:
    """Column labels sorted left → right (as-is from DrawingGrid.col_labels)."""
    return [lbl for lbl, _ in grid.col_labels]


def row_order(grid: DrawingGrid) -> list[str]:
    """Row labels sorted top → bottom (as-is from DrawingGrid.row_labels)."""
    return [lbl for lbl, _ in grid.row_labels]


# ---------------------------------------------------------------------------
# Grid inference — success criteria
# ---------------------------------------------------------------------------


class TestInferGrid:
    def test_part1_revA_columns(self):
        grid = infer_grid(ASSETS / "part1" / "revA.pdf")
        assert grid is not None
        assert col_order(grid) == ["8", "7", "6", "5", "4", "3", "2", "1"]

    def test_part1_revA_rows(self):
        grid = infer_grid(ASSETS / "part1" / "revA.pdf")
        assert grid is not None
        assert row_order(grid) == ["H", "G", "F", "E", "D", "C", "B", "A"]

    def test_part1_revB_columns(self):
        grid = infer_grid(ASSETS / "part1" / "revB.pdf")
        assert grid is not None
        assert col_order(grid) == ["8", "7", "6", "5", "4", "3", "2", "1"]

    def test_part1_revB_rows(self):
        grid = infer_grid(ASSETS / "part1" / "revB.pdf")
        assert grid is not None
        assert row_order(grid) == ["H", "G", "F", "E", "D", "C", "B", "A"]

    def test_part2_revA_columns(self):
        grid = infer_grid(ASSETS / "part2" / "revA.pdf")
        assert grid is not None
        assert col_order(grid) == ["4", "3", "2", "1"]

    def test_part2_revA_rows(self):
        grid = infer_grid(ASSETS / "part2" / "revA.pdf")
        assert grid is not None
        assert row_order(grid) == ["D", "C", "B", "A"]

    def test_part2_revB_columns(self):
        grid = infer_grid(ASSETS / "part2" / "revB.pdf")
        assert grid is not None
        assert col_order(grid) == ["4", "3", "2", "1"]

    def test_part2_revB_rows(self):
        grid = infer_grid(ASSETS / "part2" / "revB.pdf")
        assert grid is not None
        assert row_order(grid) == ["D", "C", "B", "A"]

    def test_part3_revA_columns(self):
        grid = infer_grid(ASSETS / "part3" / "revA.pdf")
        assert grid is not None
        assert col_order(grid) == ["2", "1"]

    def test_part3_revA_rows(self):
        grid = infer_grid(ASSETS / "part3" / "revA.pdf")
        assert grid is not None
        assert row_order(grid) == ["B", "A"]

    def test_part3_revB_columns(self):
        grid = infer_grid(ASSETS / "part3" / "revB.pdf")
        assert grid is not None
        assert col_order(grid) == ["2", "1"]

    def test_part3_revB_rows(self):
        grid = infer_grid(ASSETS / "part3" / "revB.pdf")
        assert grid is not None
        assert row_order(grid) == ["B", "A"]

    def test_part4_revA_columns(self):
        grid = infer_grid(ASSETS / "part4" / "revA.pdf")
        assert grid is not None
        assert col_order(grid) == ["4", "3", "2", "1"]

    def test_part4_revA_rows(self):
        grid = infer_grid(ASSETS / "part4" / "revA.pdf")
        assert grid is not None
        assert row_order(grid) == ["D", "C", "B", "A"]


# ---------------------------------------------------------------------------
# zone_to_bbox
# ---------------------------------------------------------------------------


class TestZoneToBbox:
    def test_bbox_is_tuple_of_four_floats(self):
        grid = infer_grid(ASSETS / "part4" / "revA.pdf")
        assert grid is not None
        bbox = grid.zone_to_bbox("B.2")
        assert bbox is not None
        assert len(bbox) == 4
        x0, y0, x1, y1 = bbox
        assert x0 < x1
        assert y0 < y1

    def test_bbox_within_page_bounds(self):
        grid = infer_grid(ASSETS / "part4" / "revA.pdf")
        assert grid is not None
        bbox = grid.zone_to_bbox("B.2")
        assert bbox is not None
        x0, y0, x1, y1 = bbox
        assert x0 >= 0
        assert y0 >= 0
        assert x1 <= grid.page_width
        assert y1 <= grid.page_height

    def test_unknown_zone_returns_none(self):
        grid = infer_grid(ASSETS / "part4" / "revA.pdf")
        assert grid is not None
        assert grid.zone_to_bbox("Z.9") is None

    def test_concatenated_format(self):
        grid = infer_grid(ASSETS / "part4" / "revA.pdf")
        assert grid is not None
        # "B2" and "B.2" should resolve to the same bbox
        assert grid.zone_to_bbox("B2") == grid.zone_to_bbox("B.2")

    def test_corner_zone_starts_at_page_edge(self):
        grid = infer_grid(ASSETS / "part4" / "revA.pdf")
        assert grid is not None
        # First column, first row zone should start at (0, 0)
        first_col = grid.col_labels[0][0]
        first_row = grid.row_labels[0][0]
        bbox = grid.zone_to_bbox(f"{first_row}.{first_col}")
        assert bbox is not None
        x0, y0, x1, y1 = bbox
        assert x0 == pytest.approx(0.0)
        assert y0 == pytest.approx(0.0)

    def test_last_zone_ends_at_page_edge(self):
        grid = infer_grid(ASSETS / "part4" / "revA.pdf")
        assert grid is not None
        last_col = grid.col_labels[-1][0]
        last_row = grid.row_labels[-1][0]
        bbox = grid.zone_to_bbox(f"{last_row}.{last_col}")
        assert bbox is not None
        x0, y0, x1, y1 = bbox
        assert x1 == pytest.approx(grid.page_width)
        assert y1 == pytest.approx(grid.page_height)


# ---------------------------------------------------------------------------
# parse_zone_from_reference_location
# ---------------------------------------------------------------------------


class TestParseZoneFromReferenceLocation:
    def test_full_format(self):
        assert parse_zone_from_reference_location("revA pg.1, Zone B.2") == "B.2"

    def test_zone_prefix_only(self):
        assert parse_zone_from_reference_location("Zone B2") == "B2"

    def test_bare_zone_dot(self):
        assert parse_zone_from_reference_location("B.2") == "B.2"

    def test_bare_zone_concat(self):
        assert parse_zone_from_reference_location("B2") == "B2"

    def test_no_zone_returns_none(self):
        assert parse_zone_from_reference_location("revA pg.1") is None

    def test_empty_returns_none(self):
        assert parse_zone_from_reference_location("") is None

    def test_case_insensitive_zone_keyword(self):
        assert parse_zone_from_reference_location("zone b.2") == "b.2"


# ---------------------------------------------------------------------------
# _parse_zone_parts (internal)
# ---------------------------------------------------------------------------


class TestParseZoneParts:
    def test_letter_dot_digit(self):
        assert _parse_zone_parts("B.2") == ("B", "2")

    def test_digit_dot_letter(self):
        assert _parse_zone_parts("2.B") == ("2", "B")

    def test_letter_concat_digit(self):
        assert _parse_zone_parts("B2") == ("B", "2")

    def test_digit_concat_letter(self):
        assert _parse_zone_parts("2B") == ("2", "B")

    def test_uppercase_normalisation(self):
        assert _parse_zone_parts("b.2") == ("B", "2")

    def test_invalid_returns_none(self):
        assert _parse_zone_parts("XY") is None
        assert _parse_zone_parts("12") is None
        assert _parse_zone_parts("") is None
