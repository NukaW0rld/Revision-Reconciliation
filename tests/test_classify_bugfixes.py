"""Tests for four classify.py bugfixes (2026-03-30).

Bug 1: count_added — Rev B adds a multiplicity prefix (e.g., none → "2X Ø8")
Bug 2: spurious match guard — grid labels with zero numeric overlap → "removed"
Bug 3: GD&T feature control frames detected as added characteristics
Bug 4: Plain decimal dimensions (e.g., "1.250") detected as added characteristics
"""

from delta_preservation.io.pdf import TextSpan
from delta_preservation.reconcile.anchors import Anchor
from delta_preservation.reconcile.classify import (
    DeltaItem,
    classify_delta,
    detect_added_characteristics,
)
from delta_preservation.reconcile.match import Match, Candidate, generate_candidates
from delta_preservation.reconcile.normalize import extract_semantic_callout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Transform:
    """Minimal stand-in for the homography transform required by generate_candidates."""

    def __init__(self):
        import numpy as np

        self.H = np.eye(3, dtype=float)
        self.inliers = 8
        self.inlier_ratio = 1.0


def _span(
    text: str,
    *,
    block_id: int = 0,
    line_id: int = 0,
    span_id: int = 0,
    x0: float = 10.0,
    y0: float = 10.0,
    width: float = 20.0,
    height: float = 8.0,
) -> TextSpan:
    return TextSpan(
        text=text,
        bbox_pdf=(x0, y0, x0 + width, y0 + height),
        font_size=10.0,
        block_id=block_id,
        line_id=line_id,
        span_id=span_id,
    )


def _anchor(requirement_raw: str, char_no: int = 1) -> Anchor:
    anchor_span = _span(requirement_raw)
    return Anchor(
        char_no=char_no,
        page=0,
        balloon_bbox=(0.0, 0.0, 5.0, 5.0),
        req_bbox=anchor_span.bbox_pdf,
        requirement_raw=requirement_raw,
        requirement_norm=requirement_raw,
        local_context=[anchor_span],
    )


def _span_key(span: TextSpan):
    return (span.block_id, span.line_id, span.span_id, span.bbox_pdf)


def _classify(anchor: Anchor, candidate_span: TextSpan):
    """Run classify_delta through the full candidate/match pipeline."""
    anchor_semantic = extract_semantic_callout(
        pdf_spans=[_span(anchor.requirement_raw)],
        form3_requirement=anchor.requirement_raw,
    )
    matched_semantic = extract_semantic_callout(pdf_spans=[candidate_span])
    revb_semantic_by_span = {_span_key(candidate_span): matched_semantic}

    candidates = generate_candidates(
        anchor,
        [candidate_span],
        _Transform(),
        anchor_semantic_callout=anchor_semantic,
        revB_semantic_callouts_by_span_key=revb_semantic_by_span,
    )
    assert candidates, "Expected at least one candidate"
    match = Match(char_no=anchor.char_no, candidate=candidates[0])
    delta = classify_delta(
        anchor,
        match,
        anchor_semantic_callout=anchor_semantic,
        matched_semantic_callout=matched_semantic,
    )
    return delta


# ---------------------------------------------------------------------------
# Bug 1: count_added — Rev B gains a multiplicity prefix
# ---------------------------------------------------------------------------

class TestCountAdded:
    """When Rev B adds a count prefix (e.g., none → '2X Ø8'), status must be 'changed'."""

    def test_count_added_marks_changed(self):
        anchor = _anchor("Ø 8")
        candidate = _span("2X Ø 8", block_id=1, line_id=0, span_id=0, x0=12.0, y0=11.0)
        delta = _classify(anchor, candidate)
        assert delta.status == "changed", f"Expected 'changed', got '{delta.status}'"
        assert any("Count added" in r or "count" in r.lower() for r in delta.reasons)

    def test_count_same_stays_unchanged(self):
        """Sanity: when both sides have the same count the status stays unchanged."""
        anchor = _anchor("2X Ø 8")
        candidate = _span("2X Ø 8", block_id=1, line_id=0, span_id=0, x0=12.0, y0=11.0)
        delta = _classify(anchor, candidate)
        assert delta.status == "unchanged"


# ---------------------------------------------------------------------------
# Bug 2: spurious match guard — zero overlap + divergent primary → "removed"
# ---------------------------------------------------------------------------

class TestSpuriousMatchGuard:
    """When a match candidate has zero numeric overlap and a significantly
    different primary value, the characteristic should be classified as 'removed'."""

    def test_spurious_grid_label_match_returns_removed(self):
        anchor = _anchor("Ø 12.5")
        # Candidate text is a completely unrelated grid label value
        candidate = _span("A 45.0", block_id=2, line_id=3, span_id=0, x0=100.0, y0=200.0)
        delta = _classify(anchor, candidate)
        assert delta.status == "removed", f"Expected 'removed', got '{delta.status}'"
        assert any("grid label" in r.lower() or "spurious" in r.lower() or "unrelated" in r.lower() for r in delta.reasons)

    def test_genuine_match_not_affected(self):
        """When primary values match, the guard should NOT fire."""
        anchor = _anchor("Ø 12.5")
        candidate = _span("Ø 12.5", block_id=2, line_id=0, span_id=0, x0=12.0, y0=11.0)
        delta = _classify(anchor, candidate)
        assert delta.status in ("unchanged", "changed"), f"Unexpected '{delta.status}'"


# ---------------------------------------------------------------------------
# Bug 3: GD&T feature control frames detected as added
# ---------------------------------------------------------------------------

class TestGdtAddedDetection:
    """Pass 0 in detect_added_characteristics should group GD&T symbol spans
    with their companion tolerance/datum spans and emit a single added item."""

    def test_gdt_fcf_detected_as_added(self):
        # GD&T symbol span (no numerics) plus a companion tolerance span (numerics, no symbols)
        sym_span = _span("⟂", block_id=5, line_id=0, span_id=0, x0=100.0, y0=200.0, width=10.0)
        tol_span = _span("Ø 0.01 A B", block_id=5, line_id=0, span_id=1, x0=112.0, y0=200.0, width=40.0)

        # No existing matches — everything is unmatched
        results = detect_added_characteristics(
            revB_spans=[sym_span, tol_span],
            matches={},
            next_char_no=100,
        )
        assert len(results) >= 1, "Expected at least one added GD&T item"
        gdt_items = [r for r in results if "GD&T" in " ".join(r.reasons)]
        assert gdt_items, "Expected a GD&T-specific added item"
        assert gdt_items[0].status == "added"

    def test_gdt_companion_spans_not_double_counted(self):
        """Companion spans consumed by Pass 0 should not produce separate items in Pass 2."""
        sym_span = _span("⌖", block_id=5, line_id=0, span_id=0, x0=100.0, y0=200.0, width=10.0)
        tol_span = _span("Ø 0.05 A", block_id=5, line_id=0, span_id=1, x0=112.0, y0=200.0, width=35.0)

        results = detect_added_characteristics(
            revB_spans=[sym_span, tol_span],
            matches={},
            next_char_no=100,
        )
        # Should have exactly one item for the group, not two
        gdt_items = [r for r in results if "GD&T" in " ".join(r.reasons)]
        assert len(gdt_items) == 1, f"Expected 1 GD&T group, got {len(gdt_items)}"


# ---------------------------------------------------------------------------
# Bug 4: plain decimal dimensions detected as added
# ---------------------------------------------------------------------------

class TestPlainDecimalDetection:
    """Plain decimal values like '1.250' should be detected as added characteristics."""

    def test_plain_decimal_detected_as_added(self):
        span = _span("1.250", block_id=3, line_id=0, span_id=0, x0=100.0, y0=200.0)
        results = detect_added_characteristics(
            revB_spans=[span],
            matches={},
            next_char_no=100,
        )
        assert len(results) >= 1, "Expected '1.250' to be detected as added"
        assert results[0].status == "added"
        assert any("decimal" in r.lower() for r in results[0].reasons)

    def test_leading_decimal_still_works(self):
        """Existing leading-decimal detection (.750) must continue to work."""
        span = _span(".750", block_id=3, line_id=0, span_id=0, x0=100.0, y0=200.0)
        results = detect_added_characteristics(
            revB_spans=[span],
            matches={},
            next_char_no=100,
        )
        assert len(results) >= 1, "Expected '.750' to be detected as added"
        assert results[0].status == "added"

    def test_plain_decimal_3_750_detected(self):
        span = _span("3.750", block_id=3, line_id=0, span_id=0, x0=100.0, y0=200.0)
        results = detect_added_characteristics(
            revB_spans=[span],
            matches={},
            next_char_no=100,
        )
        assert len(results) >= 1, "Expected '3.750' to be detected as added"
        assert results[0].status == "added"
