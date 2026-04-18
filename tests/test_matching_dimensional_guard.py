"""Phase 9 Plan 05: Focused unit coverage for the dimensional-incompatibility guard.

Tests exercise ``assign_matches`` in isolation with hand-built Anchor/Candidate
inputs to prove that the ``_candidate_is_dimensionally_compatible`` guard:

  * Rejects candidates whose numeric payload has zero overlap with the anchor's
    primary numeric values.
  * Accepts candidates that carry the anchor's primary value in their own text
    or in their grouped source_spans.
  * Leaves non-numeric anchors (notes, drawing-notes) completely unaffected.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from delta_preservation.io.pdf import TextSpan
from delta_preservation.reconcile.anchors import Anchor
from delta_preservation.reconcile.match import (
    Candidate,
    GroupedSpan,
    assign_matches,
)


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def _make_span(
    text: str,
    *,
    block_id: int = 0,
    line_id: int = 0,
    span_id: int = 0,
    x0: float = 100.0,
    y0: float = 100.0,
    width: float = 60.0,
    height: float = 12.0,
) -> TextSpan:
    """Build a minimal TextSpan for test purposes."""
    return TextSpan(
        text=text,
        bbox_pdf=(x0, y0, x0 + width, y0 + height),
        font_size=10.0,
        block_id=block_id,
        line_id=line_id,
        span_id=span_id,
    )


def _make_anchor(
    char_no: int,
    requirement_raw: str,
) -> Anchor:
    """Build a minimal Anchor for test purposes."""
    return Anchor(
        char_no=char_no,
        page=0,
        balloon_bbox=(50.0, 50.0, 70.0, 70.0),
        req_bbox=(100.0, 100.0, 200.0, 112.0),
        requirement_raw=requirement_raw,
        requirement_norm=requirement_raw,
        local_context=[],
    )


def _make_candidate(
    span: TextSpan,
    *,
    total_score: float = 0.5,
    reasons: Optional[List[str]] = None,
    from_global_fallback: bool = False,
    source_span_keys: Optional[List[tuple]] = None,
) -> Candidate:
    """Build a Candidate for test purposes."""
    return Candidate(
        span=span,
        total_score=total_score,
        location_score=0.3,
        text_score=0.2,
        context_score=0.0,
        reasons=reasons or ["test candidate"],
        from_global_fallback=from_global_fallback,
        source_span_keys=source_span_keys,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDimensionalIncompatibilityGuard:
    """Verify the dimensional-incompatibility guard in assign_matches."""

    def test_anchor_with_primary_value_rejects_candidate_with_no_numeric_overlap(self):
        """A Ø35.2 / Ø34.8 anchor must reject a 3X Ø18 ↧30 candidate.

        The candidate carries numerics {3, 18, 30} — none of which are within
        tolerance of the anchor's {35.2, 34.8}.  The guard should block the edge
        and the returned matches dict should be empty.
        """
        anchor = _make_anchor(char_no=1, requirement_raw="Ø35.2 / Ø34.8")

        # Build a grouped span with three source sub-spans
        src1 = _make_span("3X Ø18", block_id=10, line_id=0, span_id=0, x0=200.0)
        src2 = _make_span("↧", block_id=10, line_id=0, span_id=1, x0=260.0, width=12.0)
        src3 = _make_span("30", block_id=10, line_id=0, span_id=2, x0=272.0, width=14.0)

        grouped = GroupedSpan(
            text="3X Ø18 ↧ 30",
            bbox_pdf=(200.0, 100.0, 286.0, 112.0),
            font_size=10.0,
            block_id=10,
            line_id=0,
            span_id=0,
            source_spans=[src1, src2, src3],
        )

        candidate = _make_candidate(
            span=grouped,
            total_score=0.5,
            from_global_fallback=False,
            source_span_keys=[
                (src1.block_id, src1.line_id, src1.span_id, src1.bbox_pdf),
                (src2.block_id, src2.line_id, src2.span_id, src2.bbox_pdf),
                (src3.block_id, src3.line_id, src3.span_id, src3.bbox_pdf),
            ],
        )

        matches = assign_matches(
            anchors=[anchor],
            candidates_by_anchor={1: [candidate]},
        )

        assert 1 not in matches, (
            f"Expected the guard to block the edge but char_no=1 was assigned: {matches}"
        )

    def test_anchor_accepts_candidate_whose_source_spans_carry_primary_value(self):
        """A Ø35.2 / Ø34.8 anchor must accept a candidate that includes 35.2.

        The candidate's grouped text includes both the anchor primary value (35.2)
        and a foreign value (18).  The guard must let it through because at least
        one anchor numeric overlaps with the candidate's numeric payload.
        """
        anchor = _make_anchor(char_no=1, requirement_raw="Ø35.2 / Ø34.8")

        # Build a grouped span that includes the anchor's value
        src1 = _make_span("Ø35.2", block_id=20, line_id=0, span_id=0, x0=200.0)
        src2 = _make_span("Ø34.8", block_id=20, line_id=0, span_id=1, x0=260.0)
        src3 = _make_span("3X Ø18", block_id=20, line_id=0, span_id=2, x0=320.0)

        grouped = GroupedSpan(
            text="Ø35.2 Ø34.8 3X Ø18",
            bbox_pdf=(200.0, 100.0, 380.0, 112.0),
            font_size=10.0,
            block_id=20,
            line_id=0,
            span_id=0,
            source_spans=[src1, src2, src3],
        )

        candidate = _make_candidate(
            span=grouped,
            total_score=0.5,
            reasons=["primary=35.2 exact match"],
            from_global_fallback=False,
            source_span_keys=[
                (src1.block_id, src1.line_id, src1.span_id, src1.bbox_pdf),
                (src2.block_id, src2.line_id, src2.span_id, src2.bbox_pdf),
                (src3.block_id, src3.line_id, src3.span_id, src3.bbox_pdf),
            ],
        )

        matches = assign_matches(
            anchors=[anchor],
            candidates_by_anchor={1: [candidate]},
        )

        assert 1 in matches, (
            f"Expected the guard to accept the candidate but char_no=1 was not assigned: {matches}"
        )

    def test_notes_anchor_without_numeric_tokens_is_unaffected_by_the_guard(self):
        """A notes anchor with no numeric tokens must accept any candidate.

        The guard must be permissive for non-numeric anchors so that drawing-notes
        and textual requirements are not accidentally blocked.
        """
        anchor = _make_anchor(char_no=10, requirement_raw="DRAWING NOTES: SEE SHEET 2")

        span = _make_span(
            "DRAWING NOTES",
            block_id=30, line_id=0, span_id=0,
            x0=50.0, y0=500.0, width=100.0,
        )

        candidate = _make_candidate(
            span=span,
            total_score=0.1,
            reasons=["notes text match"],
            from_global_fallback=False,
        )

        matches = assign_matches(
            anchors=[anchor],
            candidates_by_anchor={10: [candidate]},
        )

        assert 10 in matches, (
            f"Expected the notes anchor to accept the candidate but char_no=10 was not assigned: {matches}"
        )
