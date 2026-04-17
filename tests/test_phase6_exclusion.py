"""
Phase 6: Unit tests for the shared exclusion contract.

Covers:
- TestSharedExclusionContract: geometric and boilerplate exclusion rules
- TestRescueAndAddedDetectionExclusion: rescue scan and added detection respect the
  same exclusion rules instead of carrying per-module heuristics

These tests are expected to FAIL before exclusion.py is created and callers are
migrated.  Once exclusion.py is in place and anchors.py, match.py, classify.py
import it, all tests must pass.
"""

import pytest

from delta_preservation.io.pdf import TextSpan


# ---------------------------------------------------------------------------
# Lightweight fixture helpers
# ---------------------------------------------------------------------------

def _span(
    text: str,
    *,
    cx: float,
    cy: float,
    width: float = 30.0,
    height: float = 10.0,
    block_id: int = 0,
    line_id: int = 0,
    span_id: int = 0,
) -> TextSpan:
    """Build a fake TextSpan centred at (cx, cy)."""
    x0 = cx - width / 2
    y0 = cy - height / 2
    return TextSpan(
        text=text,
        bbox_pdf=(x0, y0, x0 + width, y0 + height),
        font_size=10.0,
        block_id=block_id,
        line_id=line_id,
        span_id=span_id,
    )


# ---------------------------------------------------------------------------
# TestSharedExclusionContract
# ---------------------------------------------------------------------------

class TestSharedExclusionContract:
    """
    Tests that the shared exclusion module (exclusion.py) correctly identifies
    boilerplate and title-block regions.
    """

    def _excluded(self, span: TextSpan, *, page_width: float, page_height: float) -> bool:
        from delta_preservation.reconcile.exclusion import span_is_excluded_for_annotation_search
        return span_is_excluded_for_annotation_search(
            span, page_width=page_width, page_height=page_height
        )

    # 1. Bottom-right title block text is excluded
    def test_title_block_bottom_right_is_excluded(self):
        pw, ph = 792.0, 612.0
        # Centre in the bottom-right corner: cx > 70% width AND cy > 85% height
        span = _span(
            "UNLESS OTHERWISE SPECIFIED",
            cx=pw * 0.80,
            cy=ph * 0.92,
        )
        assert self._excluded(span, page_width=pw, page_height=ph)

    # 2. Bottom-centre tolerance-block boilerplate is excluded
    def test_bottom_centre_tolerance_block_is_excluded(self):
        pw, ph = 792.0, 612.0
        # Centre in bottom-centre band: 25%–72% width, 78%–92% height
        span = _span(
            "UNLESS OTHERWISE SPECIFIED",
            cx=pw * 0.50,
            cy=ph * 0.85,
        )
        assert self._excluded(span, page_width=pw, page_height=ph)

    # 3. Revision table (upper-right) is excluded
    def test_revision_table_upper_right_is_excluded(self):
        pw, ph = 792.0, 612.0
        # Centre in upper-right: cx > 75% width AND cy < 15% height
        span = _span(
            "REV.",
            cx=pw * 0.85,
            cy=ph * 0.08,
        )
        assert self._excluded(span, page_width=pw, page_height=ph)

    # 4. A legitimate drawing annotation near the edge is NOT excluded
    def test_legitimate_edge_annotation_not_excluded(self):
        pw, ph = 792.0, 612.0
        # Centre near left edge but in drawing area, with engineering content
        span = _span(
            "Ø6.0 ±0.1",
            cx=pw * 0.05,
            cy=ph * 0.50,
        )
        assert not self._excluded(span, page_width=pw, page_height=ph)

    # 5. Page-dimension estimation uses the same floors as match.py
    def test_estimate_page_dimensions_applies_minimum_floors(self):
        from delta_preservation.reconcile.exclusion import estimate_page_dimensions

        # Spans that are very small — floors should kick in
        tiny_spans = [
            _span("X", cx=100.0, cy=100.0, width=10.0, height=10.0),
        ]
        w, h = estimate_page_dimensions(tiny_spans)
        assert w >= 612.0, f"Expected width floor 612.0, got {w}"
        assert h >= 792.0, f"Expected height floor 792.0, got {h}"

    # 5b. When spans are larger than the floor, the actual extents are used
    def test_estimate_page_dimensions_uses_actual_when_larger(self):
        from delta_preservation.reconcile.exclusion import estimate_page_dimensions

        large_spans = [
            _span("X", cx=400.0, cy=500.0, width=800.0, height=20.0),
        ]
        w, h = estimate_page_dimensions(large_spans)
        assert w >= 800.0, f"Expected width >= 800, got {w}"

    # 6. Known boilerplate text is detected by is_boilerplate_candidate_text
    def test_is_boilerplate_candidate_text_detects_tolerance_block(self):
        from delta_preservation.reconcile.exclusion import is_boilerplate_candidate_text

        assert is_boilerplate_candidate_text("UNLESS OTHERWISE SPECIFIED")
        assert is_boilerplate_candidate_text(".XX = ±.020")
        assert is_boilerplate_candidate_text("ANGULAR = ±0.3°")
        assert not is_boilerplate_candidate_text("Ø6.0 ±0.1")
        assert not is_boilerplate_candidate_text("R3.5")


# ---------------------------------------------------------------------------
# TestRescueAndAddedDetectionExclusion
# ---------------------------------------------------------------------------

class TestRescueAndAddedDetectionExclusion:
    """
    Tests that the keyword rescue path in classify_delta() and
    detect_added_characteristics() both honour the shared exclusion contract
    instead of implementing their own corner-only heuristics.
    """

    def _make_anchor(
        self,
        requirement_raw: str,
        *,
        char_no: int = 1,
        balloon_cx: float = 200.0,
        balloon_cy: float = 300.0,
    ):
        from delta_preservation.reconcile.anchors import Anchor
        return Anchor(
            char_no=char_no,
            page=0,
            balloon_bbox=(
                balloon_cx - 5.0,
                balloon_cy - 5.0,
                balloon_cx + 5.0,
                balloon_cy + 5.0,
            ),
            req_bbox=None,
            requirement_raw=requirement_raw,
            requirement_norm=requirement_raw,
            local_context=[],
        )

    # 1. Keyword rescue skips spans in excluded regions
    def test_keyword_rescue_skips_excluded_spans(self):
        """
        The keyword rescue scan inside classify_delta() must use the shared
        exclusion helper.  A keyword anchor whose matching span is in the
        title-block region must not be rescued as 'unchanged'.
        """
        from delta_preservation.reconcile.classify import classify_delta
        from delta_preservation.reconcile.exclusion import estimate_page_dimensions

        pw, ph = 792.0, 612.0
        # Span text matches anchor keyword but sits in the bottom-right title block
        excluded_span = _span(
            "THRU",
            cx=pw * 0.80,
            cy=ph * 0.92,
            block_id=0,
            line_id=0,
            span_id=0,
        )

        anchor = self._make_anchor("THRU", char_no=1)
        # No match — forcing classify_delta into the rescue path
        result = classify_delta(
            anchor,
            match_or_none=None,
            revB_text_spans=[excluded_span],
        )
        # The rescue scan should reject the title-block span, so the result
        # must NOT be rescued as 'unchanged' — it should be 'removed' or 'uncertain'
        # but definitely NOT 'unchanged'.
        assert result.status != "unchanged", (
            f"Rescue scan should have rejected the title-block span; "
            f"got status={result.status!r}, reasons={result.reasons}"
        )

    # 2. detect_added_characteristics skips excluded spans
    def test_added_detection_skips_excluded_spans(self):
        """
        detect_added_characteristics() must exclude spans in the title-block
        or revision-table regions.  The old local is_in_exclusion_zone() used
        raw x0/y0 coordinates; the shared helper uses span centres.
        This test passes an excluded span whose centre is in the title block
        and verifies no added row is generated.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        pw, ph = 792.0, 612.0
        # Span in title block region (bottom-right): centre at (0.80 * pw, 0.92 * ph)
        title_block_span = _span(
            "Ø6.0",
            cx=pw * 0.80,
            cy=ph * 0.92,
            block_id=5,
            line_id=0,
            span_id=0,
        )

        added = detect_added_characteristics(
            revB_spans=[title_block_span],
            matches={},
            next_char_no=100,
            page_width=pw,
            page_height=ph,
        )
        assert added == [], (
            f"Title-block span should be excluded from added detection; "
            f"got {len(added)} added items"
        )

    # 3. detect_added_characteristics uses shared estimate_page_dimensions internally
    def test_added_detection_honours_estimate_page_dimensions(self):
        """
        When page_width / page_height are supplied as defaults (612 / 792)
        the exclusion zones should still reject revision-table spans.
        This test verifies the contract is applied even with default page sizes.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        pw, ph = 612.0, 792.0
        # Span in revision table: cx > 75% pw, cy < 15% ph
        revision_span = _span(
            "Ø5.0",
            cx=pw * 0.85,
            cy=ph * 0.08,
            block_id=6,
            line_id=0,
            span_id=0,
        )

        added = detect_added_characteristics(
            revB_spans=[revision_span],
            matches={},
            next_char_no=100,
            page_width=pw,
            page_height=ph,
        )
        assert added == [], (
            f"Revision-table span should be excluded from added detection; "
            f"got {len(added)} added items"
        )
