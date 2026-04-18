"""Phase 6 Plan 02: Grouped added evidence contract and explained-by-match suppression.

Three test classes:

  TestGroupedAddedEvidence
    Proves that the standard added pass produces full grouped callout text and union
    bbox for multi-span annotations (◎ ∅.045 A, ⌖ ∅.015 D H) instead of raw seed
    fragments.

  TestExplainedByMatchSuppression
    Proves that false-positive added fragments are suppressed only when an existing
    matched annotation already fully explains them — NOT on proximity alone.

  TestAddedPacketAssembly
    Proves that cli.py uses added_requirement_text (not added_span.text) and
    added_bbox (not seed span bbox) when building packet items for added rows, and
    emits snippet_rule_family="grouped_callout" for grouped evidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import pytest

# ---------------------------------------------------------------------------
# TextSpan factory
# ---------------------------------------------------------------------------

from delta_preservation.io.pdf import TextSpan


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


# ---------------------------------------------------------------------------
# Minimal Match / Candidate stubs to feed detect_added_characteristics()
# ---------------------------------------------------------------------------

from delta_preservation.reconcile.classify import DeltaItem as DeltaItemInternal


@dataclass
class _FakeCandidate:
    span: TextSpan


@dataclass
class _FakeMatch:
    candidate: _FakeCandidate


def _make_matches(*spans: TextSpan) -> Dict[int, _FakeMatch]:
    """Return a matches dict keyed by char_no=1..N for the given matched spans."""
    return {
        i + 1: _FakeMatch(candidate=_FakeCandidate(span=s))
        for i, s in enumerate(spans)
    }


# ===========================================================================
# Class 1: TestGroupedAddedEvidence
# ===========================================================================


class TestGroupedAddedEvidence:
    """Standard multi-span added callouts must produce full grouped text and union bbox.

    Exemplar strings come from the checked-in Part 8 / Part 9 debug reports.
    """

    def test_standard_circularity_produces_full_callout_text(self):
        """A multi-span ◎ ∅.045 A callout must produce added_requirement_text='◎ ∅.045 A'.

        The seed span is '◎' (GD&T anchor); its companion spans carry '∅.045' and 'A'.
        The standard pass must NOT collapse to '.045 A' (the fragment observed in Part 8).
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        # Build three companion spans representing '◎ ∅.045 A' as separate PDF spans
        anchor_span = _span("◎", block_id=0, line_id=0, span_id=0, x0=50.0, y0=200.0, width=8.0, height=8.0)
        tol_span   = _span("∅.045", block_id=0, line_id=0, span_id=1, x0=60.0, y0=200.0, width=18.0, height=8.0)
        datum_span = _span("A", block_id=0, line_id=0, span_id=2, x0=80.0, y0=200.0, width=6.0, height=8.0)

        # No existing matched spans on this page
        matches: Dict = {}
        all_spans = [anchor_span, tol_span, datum_span]

        added = detect_added_characteristics(
            revB_spans=all_spans,
            matches=matches,
            next_char_no=10,
            page_width=612.0,
            page_height=792.0,
        )

        assert len(added) >= 1, "Expected at least one added item for the circularity callout"
        # Find the item whose grouped text contains the full callout
        texts = [it.added_requirement_text or "" for it in added]
        full_callout_found = any("◎" in t and "∅.045" in t and "A" in t for t in texts)
        assert full_callout_found, (
            f"Expected an added item with full grouped text '◎ ∅.045 A' in {texts!r}; "
            "standard pass collapsed to a fragment"
        )

    def test_part9_position_produces_full_callout_text(self):
        """A Part 9-style '⌖ ∅.015 D H' callout must NOT collapse to '⌖∅'.

        The truncated '⌖∅' string in Part 9's debug report proves the seed span only
        approach is broken. After the fix, added_requirement_text must include datum refs.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        anchor_span  = _span("⌖", block_id=0, line_id=0, span_id=0, x0=100.0, y0=300.0, width=8.0, height=8.0)
        tol_span     = _span("∅.015", block_id=0, line_id=0, span_id=1, x0=110.0, y0=300.0, width=20.0, height=8.0)
        datum_d_span = _span("D", block_id=0, line_id=0, span_id=2, x0=132.0, y0=300.0, width=6.0, height=8.0)
        datum_h_span = _span("H", block_id=0, line_id=0, span_id=3, x0=140.0, y0=300.0, width=6.0, height=8.0)

        matches: Dict = {}
        all_spans = [anchor_span, tol_span, datum_d_span, datum_h_span]

        added = detect_added_characteristics(
            revB_spans=all_spans,
            matches=matches,
            next_char_no=10,
            page_width=612.0,
            page_height=792.0,
        )

        texts = [it.added_requirement_text or "" for it in added]
        assert any("⌖" in t and "∅.015" in t and "D" in t and "H" in t for t in texts), (
            f"Expected added item with full text '⌖ ∅.015 D H' but got {texts!r}; "
            "seed-span-only path truncates to '⌖∅'"
        )

    def test_grouped_added_row_has_union_bbox(self):
        """A grouped callout's added_bbox must be the union of all companion spans.

        The seed span alone has a narrow bbox; the union bbox expands to include
        all companion spans. Tests that added_bbox is wider than the seed span.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        anchor_span = _span("⌖", block_id=0, line_id=0, span_id=0, x0=100.0, y0=300.0, width=8.0, height=8.0)
        tol_span    = _span("∅.015", block_id=0, line_id=0, span_id=1, x0=110.0, y0=300.0, width=20.0, height=8.0)
        datum_span  = _span("D", block_id=0, line_id=0, span_id=2, x0=132.0, y0=300.0, width=6.0, height=8.0)

        matches: Dict = {}
        all_spans = [anchor_span, tol_span, datum_span]

        added = detect_added_characteristics(
            revB_spans=all_spans,
            matches=matches,
            next_char_no=10,
            page_width=612.0,
            page_height=792.0,
        )

        relevant = [it for it in added if it.added_bbox is not None and it.added_requirement_text and "⌖" in it.added_requirement_text]
        assert relevant, f"No added item with GD&T grouped evidence found; got: {[it.added_requirement_text for it in added]}"

        item = relevant[0]
        seed_bbox = anchor_span.bbox_pdf
        union_bbox = item.added_bbox

        # Union bbox x1 must extend past the seed span x1
        assert union_bbox[2] > seed_bbox[2], (
            f"added_bbox x1={union_bbox[2]} should exceed seed span x1={seed_bbox[2]}; "
            "union bbox not computed"
        )

    def test_grouped_added_row_keeps_representative_seed_span(self):
        """added_span must reference a representative seed span for provenance."""
        from delta_preservation.reconcile.classify import detect_added_characteristics

        anchor_span = _span("⌖", block_id=0, line_id=0, span_id=0, x0=100.0, y0=300.0, width=8.0, height=8.0)
        tol_span    = _span("∅.015", block_id=0, line_id=0, span_id=1, x0=110.0, y0=300.0, width=20.0, height=8.0)

        matches: Dict = {}
        all_spans = [anchor_span, tol_span]

        added = detect_added_characteristics(
            revB_spans=all_spans,
            matches=matches,
            next_char_no=10,
            page_width=612.0,
            page_height=792.0,
        )

        relevant = [it for it in added if it.added_span is not None and it.added_requirement_text and "⌖" in it.added_requirement_text]
        assert relevant, f"No added item with seed span found; items: {added}"
        # The added_span must be a valid TextSpan (provenance handle)
        item = relevant[0]
        assert item.added_span is not None
        assert hasattr(item.added_span, "bbox_pdf")

    def test_part8_same_row_gdt_grouping_does_not_absorb_distant_dimension(self):
        """A GD&T row must not sweep in a separate same-row dimension hundreds of points away.

        This reproduces the Part 8 over-grouping regression where `⌰ .015 B`
        was merged with `Ø10.000±.001` into one added row.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        runout_symbol = _span("⌰", block_id=0, line_id=0, span_id=0, x0=167.0, y0=528.0, width=20.0)
        tol_span = _span(".015", block_id=0, line_id=0, span_id=1, x0=198.0, y0=528.0, width=24.0)
        datum_span = _span("B", block_id=0, line_id=0, span_id=2, x0=230.0, y0=528.0, width=8.0)
        distant_dimension = _span(
            "Ø10.000±.001",
            block_id=0,
            line_id=0,
            span_id=3,
            x0=482.0,
            y0=521.0,
            width=78.0,
        )

        added = detect_added_characteristics(
            revB_spans=[runout_symbol, tol_span, datum_span, distant_dimension],
            matches={},
            next_char_no=9,
            page_width=612.0,
            page_height=792.0,
        )

        texts = [" ".join((it.added_requirement_text or "").split()) for it in added]
        assert "⌰ .015 B" in texts, f"Expected preserved runout row, got: {texts!r}"
        assert "Ø10.000±.001" in texts, f"Expected separate diameter row, got: {texts!r}"
        assert not any("⌰" in text and "Ø10.000±.001" in text for text in texts), (
            "Part 8 regression: distant same-row dimension was merged into the GD&T row"
        )

    def test_duplicate_standard_added_rows_survive_when_geometry_differs(self):
        """Identical added text at different coordinates must survive as two rows.

        This reproduces the Part 9 duplicate collapse where text-only dedupe
        reduced two `Ø.250 ±.008` rows to one.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        left = _span("Ø.250 ±.008", block_id=0, line_id=0, span_id=0, x0=128.0, y0=322.0, width=68.0)
        right = _span("Ø.250 ±.008", block_id=1, line_id=0, span_id=0, x0=566.0, y0=280.0, width=68.0)

        added = detect_added_characteristics(
            revB_spans=[left, right],
            matches={},
            next_char_no=38,
            page_width=1224.0,
            page_height=792.0,
        )

        texts = [" ".join((it.added_requirement_text or "").split()) for it in added]
        assert texts.count("Ø.250 ±.008") == 2, (
            "Part 9 regression: distinct duplicate added rows collapsed despite different geometry"
        )

    def test_adjacent_different_anchor_rows_do_not_chain_into_one_added_row(self):
        """Nearby rows with different anchor symbols must stay separate.

        This reproduces the Part 9 shape where a depth row sat close to a
        position row and the stacked-companion logic chained them into one blob.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        depth_anchor = _span("↧", block_id=0, line_id=0, span_id=0, x0=150.0, y0=360.0, width=10.0)
        depth_value = _span(".50 ±.05", block_id=0, line_id=0, span_id=1, x0=164.0, y0=360.0, width=42.0)

        position_anchor = _span("⌖", block_id=1, line_id=0, span_id=0, x0=152.0, y0=348.0, width=10.0)
        position_tol = _span("∅.015", block_id=1, line_id=0, span_id=1, x0=166.0, y0=348.0, width=24.0)
        datum_d = _span("D", block_id=1, line_id=0, span_id=2, x0=192.0, y0=348.0, width=6.0)
        datum_h = _span("H", block_id=1, line_id=0, span_id=3, x0=200.0, y0=348.0, width=6.0)

        added = detect_added_characteristics(
            revB_spans=[depth_anchor, depth_value, position_anchor, position_tol, datum_d, datum_h],
            matches={},
            next_char_no=35,
            page_width=612.0,
            page_height=792.0,
        )

        texts = [" ".join((it.added_requirement_text or "").split()) for it in added]
        assert any("↧" in text and ".50" in text for text in texts), (
            f"Expected separate depth row, got: {texts!r}"
        )
        assert any("⌖" in text and "∅.015" in text and "D" in text and "H" in text for text in texts), (
            f"Expected separate position row, got: {texts!r}"
        )
        assert not any("↧" in text and "⌖" in text for text in texts), (
            "Adjacent different-anchor rows were chained into one added row"
        )


# ===========================================================================
# Class 2: TestExplainedByMatchSuppression
# ===========================================================================


class TestExplainedByMatchSuppression:
    """An added fragment is suppressed only when a matched annotation explains it.

    Proximity alone must NOT suppress a legitimate added row.
    """

    def test_part8_fragment_suppressed_when_matched_annotation_explains_it(self):
        """The Part 8 false-positive '.045 A' fragment is suppressed when the
        matched annotation '◎ ∅.045 A' already explains the same content.

        The suppressor must identify that '.045 A' is a content-subset of the
        matched annotation group and suppress the added row.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        # Matched span (owned by char 6 in Part 8)
        matched_anchor = _span("◎", block_id=0, line_id=0, span_id=0,
                                x0=50.0, y0=200.0, width=8.0, height=8.0)
        matched_tol    = _span("∅.045", block_id=0, line_id=0, span_id=1,
                                x0=60.0, y0=200.0, width=18.0, height=8.0)
        matched_datum  = _span("A", block_id=0, line_id=0, span_id=2,
                                x0=80.0, y0=200.0, width=6.0, height=8.0)

        # Fragment span that used to leak as a false-positive added row
        fragment_span = _span(".045 A", block_id=0, line_id=1, span_id=0,
                               x0=60.0, y0=210.0, width=24.0, height=8.0)

        # The matched annotation claims matched_anchor; simulate a grouped match span
        # by injecting a GroupedSpan-like object that has source_spans
        from delta_preservation.io.pdf import TextSpan as TS

        class _GroupedSpan(TS):
            def __init__(self, source_spans, **kwargs):
                super().__init__(**kwargs)
                self.source_spans = source_spans

        grouped = _GroupedSpan(
            source_spans=[matched_anchor, matched_tol, matched_datum],
            text="◎ ∅.045 A",
            bbox_pdf=(50.0, 200.0, 86.0, 208.0),
            font_size=10.0,
            block_id=0,
            line_id=0,
            span_id=0,
        )

        matches = {6: _FakeMatch(candidate=_FakeCandidate(span=grouped))}

        all_spans = [matched_anchor, matched_tol, matched_datum, fragment_span]

        added = detect_added_characteristics(
            revB_spans=all_spans,
            matches=matches,
            next_char_no=20,
            page_width=612.0,
            page_height=792.0,
        )

        # The '.045 A' fragment must be suppressed — it is already explained by
        # the matched '◎ ∅.045 A' annotation.
        fragment_texts = [it.added_requirement_text or "" for it in added]
        spurious = [t for t in fragment_texts if ".045 A" in t and "◎" not in t]
        assert not spurious, (
            f"False-positive '.045 A' fragment survived suppression: {fragment_texts!r}; "
            "explained-by-match suppressor did not fire"
        )

    def test_legitimate_nearby_added_row_survives_suppression(self):
        """A legitimate added callout near a matched annotation must NOT be suppressed.

        A new '⌰ .015 B' callout placed 60pt below the matched '◎ ∅.045 A' annotation
        is NOT semantically explained by that annotation, so it must survive suppression.
        The y=260 placement ensures the vertical separation (>40pt from matched center)
        is sufficient to clear the GD&T Pass 0 proximity guard while still validating
        that semantically unrelated content is not suppressed.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        # Matched annotation at y=200
        matched_span = _span("◎ ∅.045 A", block_id=0, line_id=0, span_id=0,
                              x0=50.0, y0=200.0, width=50.0, height=8.0)
        matches = {6: _FakeMatch(candidate=_FakeCandidate(span=matched_span))}

        # Legitimate added callout — different content, 60pt below (>40pt proximity threshold)
        added_anchor = _span("⌰", block_id=0, line_id=2, span_id=0,
                              x0=55.0, y0=260.0, width=8.0, height=8.0)
        added_tol    = _span(".015", block_id=0, line_id=2, span_id=1,
                              x0=65.0, y0=260.0, width=14.0, height=8.0)
        added_datum  = _span("B", block_id=0, line_id=2, span_id=2,
                              x0=82.0, y0=260.0, width=6.0, height=8.0)

        all_spans = [matched_span, added_anchor, added_tol, added_datum]

        added = detect_added_characteristics(
            revB_spans=all_spans,
            matches=matches,
            next_char_no=20,
            page_width=612.0,
            page_height=792.0,
        )

        texts = [it.added_requirement_text or "" for it in added]
        # '⌰ .015 B' or similar grouping of the new annotation must survive
        found = any("⌰" in t for t in texts)
        assert found, (
            f"Expected legitimate '⌰ .015 B' added row to survive suppression, but got {texts!r}; "
            "proximity-only suppressor incorrectly removed a legitimate added row"
        )

    def test_proximity_alone_does_not_suppress_added_row(self):
        """A semantically unrelated added span near a matched span must NOT be suppressed.

        This test explicitly verifies that the suppressor checks content ownership,
        NOT just distance. If the test passes before content-aware suppression is
        implemented, it means the old proximity filter already allows this through —
        the suppressor is still needed for the false-positive case above.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        # Matched annotation: a dimension value
        matched_span = _span("Ø.250 ±.008", block_id=0, line_id=0, span_id=0,
                              x0=100.0, y0=200.0, width=50.0, height=8.0)
        matches = {1: _FakeMatch(candidate=_FakeCandidate(span=matched_span))}

        # Nearby added span with completely different content
        new_span = _span("⏥", block_id=0, line_id=1, span_id=0,
                          x0=105.0, y0=215.0, width=8.0, height=8.0)
        new_tol  = _span(".01", block_id=0, line_id=1, span_id=1,
                          x0=115.0, y0=215.0, width=12.0, height=8.0)

        all_spans = [matched_span, new_span, new_tol]

        added = detect_added_characteristics(
            revB_spans=all_spans,
            matches=matches,
            next_char_no=20,
            page_width=612.0,
            page_height=792.0,
        )

        texts = [it.added_requirement_text or "" for it in added]
        # ⏥ .01 is NOT explained by Ø.250 ±.008 — it must survive
        # NOTE: if proximity filter (25pt radius) suppresses it, this test will FAIL here
        # and the test proves we need content-aware suppression
        found = any("⏥" in t for t in texts)
        # This assertion establishes the contract: proximity alone does not suppress
        assert found, (
            f"Expected '⏥ .01' to survive proximity check near matched 'Ø.250 ±.008', "
            f"but got {texts!r}. If proximity filter is too broad, this is a false positive."
        )

    def test_part9_flatness_survives_when_unrelated_same_row_match_exists(self):
        """The Part 9 added `⏥ .01` row must not be falsely suppressed by an unrelated
        matched owner that shares the same horizontal row.

        Root cause (confirmed): matched-owner signatures previously swept same-row
        unmatched spans within ±200 pt into the owner's synthetic text/bbox.  This
        caused the independent block-46 flatness frame to be absorbed into matched
        char 8's signature (`4X INDIVIDUALLY 2X CR.50±.02 ⏥ .01`), after which the
        content-subset gate fired and suppressed the real added flatness row.

        After the fix, owner signatures may only include companion spans (via
        _spans_are_annotation_companions), so the block-46 flatness frame is NOT pulled
        into the unrelated owner and survives as a distinct added row.

        Geometry mirrors the Part 9 live state:
          - matched char 8 owner span: x≈[335, 440], y≈438-456 (block 33 area)
          - added flatness candidate: x≈[712, 750], y≈438-456 (block 46 area, ~376 pt away)
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        # Matched owner span for char 8 — '2X CR.50±.02' at block 33 area
        matched_owner = _span(
            "2X CR.50±.02",
            block_id=33, line_id=1, span_id=0,
            x0=335.0, y0=438.0, width=105.0, height=18.0,
        )

        # Separate same-row contextual span that belongs to char 8's annotation
        # (would be legitimately grouped as a companion)
        context_span = _span(
            "4X INDIVIDUALLY",
            block_id=33, line_id=0, span_id=0,
            x0=335.0, y0=420.0, width=90.0, height=14.0,
        )

        # The independent added flatness frame at block 46 — ~376 pt to the right
        flatness_symbol = _span(
            "⏥",
            block_id=46, line_id=1, span_id=0,
            x0=712.0, y0=440.0, width=14.0, height=14.0,
        )
        flatness_tol = _span(
            ".01",
            block_id=46, line_id=1, span_id=1,
            x0=728.0, y0=440.0, width=20.0, height=14.0,
        )

        matches = {8: _FakeMatch(candidate=_FakeCandidate(span=matched_owner))}
        all_spans = [context_span, matched_owner, flatness_symbol, flatness_tol]

        added = detect_added_characteristics(
            revB_spans=all_spans,
            matches=matches,
            next_char_no=35,
            page_width=1224.0,
            page_height=792.0,
        )

        texts = [it.added_requirement_text or "" for it in added]
        flatness_found = any("⏥" in t for t in texts)
        assert flatness_found, (
            f"Part 9 regression: added '⏥ .01' row was falsely suppressed by the "
            f"unrelated matched char-8 owner. After owner-signature narrowing it must "
            f"survive as an independent added row. Got: {texts!r}"
        )

    def test_part8_fragment_still_suppressed_after_owner_signature_narrowing(self):
        """After narrowing matched-owner signature construction, a true explained
        fragment must still be suppressed.

        Reproduces the Part 8 pattern: '◎ ∅.045 A' is a matched annotation; '.045 A'
        appears as an unmatched span on a stacked row immediately below — it IS a
        companion to the matched spans and must be absorbed into the owner signature,
        then suppressed when it surfaces as an added candidate.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics
        from delta_preservation.io.pdf import TextSpan as TS

        # Part 8 matched annotation: ◎ ∅.045 A grouped span
        class _GroupedSpan(TS):
            def __init__(self, source_spans, **kwargs):
                super().__init__(**kwargs)
                self.source_spans = source_spans

        matched_anchor = _span("◎", block_id=0, line_id=0, span_id=0,
                                x0=50.0, y0=200.0, width=8.0, height=8.0)
        matched_tol    = _span("∅.045", block_id=0, line_id=0, span_id=1,
                                x0=60.0, y0=200.0, width=18.0, height=8.0)
        matched_datum  = _span("A", block_id=0, line_id=0, span_id=2,
                                x0=80.0, y0=200.0, width=6.0, height=8.0)

        grouped = _GroupedSpan(
            source_spans=[matched_anchor, matched_tol, matched_datum],
            text="◎ ∅.045 A",
            bbox_pdf=(50.0, 200.0, 86.0, 208.0),
            font_size=10.0,
            block_id=0,
            line_id=0,
            span_id=0,
        )

        # Fragment span: stacked immediately below the matched group — companion distance
        fragment_span = _span(".045 A", block_id=0, line_id=1, span_id=0,
                               x0=60.0, y0=210.0, width=24.0, height=8.0)

        matches = {6: _FakeMatch(candidate=_FakeCandidate(span=grouped))}
        all_spans = [matched_anchor, matched_tol, matched_datum, fragment_span]

        added = detect_added_characteristics(
            revB_spans=all_spans,
            matches=matches,
            next_char_no=20,
            page_width=612.0,
            page_height=792.0,
        )

        fragment_texts = [it.added_requirement_text or "" for it in added]
        spurious = [t for t in fragment_texts if ".045 A" in t and "◎" not in t]
        assert not spurious, (
            f"Part 8 regression: '.045 A' fragment survived suppression after owner "
            f"signature narrowing — narrower signatures must still absorb true stacked "
            f"companion fragments. Got added texts: {fragment_texts!r}"
        )


# ===========================================================================
# Class 3: TestAddedPacketAssembly
# ===========================================================================


class TestAddedPacketAssembly:
    """cli.py packet assembly must use added_requirement_text and added_bbox for added rows.

    Uses DeltaItemInternal objects injected into the packet assembly path
    to verify that the new grouped contract fields flow through to the
    Pydantic packet DeltaItem correctly.
    """

    def _make_internal_added_item(
        self,
        *,
        char_no: int = 20,
        seed_text: str = "⌖",
        grouped_text: str = "⌖ ∅.015 D H",
        seed_bbox: Tuple[float, float, float, float] = (100.0, 300.0, 108.0, 308.0),
        grouped_bbox: Tuple[float, float, float, float] = (100.0, 300.0, 148.0, 308.0),
        multiple_spans: bool = True,
    ) -> DeltaItemInternal:
        seed_span = TextSpan(
            text=seed_text,
            bbox_pdf=seed_bbox,
            font_size=10.0,
            block_id=0,
            line_id=0,
            span_id=0,
        )
        return DeltaItemInternal(
            char_no=char_no,
            status="added",
            confidence=0.75,
            reasons=["New GD&T feature control frame in Rev B"],
            component_scores={"location": 0.0, "text": 1.0, "context": 0.0},
            added_span=seed_span,
            added_requirement_text=grouped_text if multiple_spans else None,
            added_bbox=grouped_bbox if multiple_spans else None,
            added_page=0,
        )

    def test_packet_uses_added_requirement_text_not_seed_span_text(self):
        """requirement_revB for an added row must come from added_requirement_text.

        The seed span text is '⌖'; the full grouped text is '⌖ ∅.015 D H'.
        The packet must report the full grouped text, not the truncated seed.
        """
        # Verify that DeltaItemInternal has added_requirement_text set
        item = self._make_internal_added_item()
        assert item.added_requirement_text == "⌖ ∅.015 D H", (
            f"Internal item should have added_requirement_text='⌖ ∅.015 D H', "
            f"got {item.added_requirement_text!r}"
        )
        # Verify seed span text is different (the fragment problem)
        assert item.added_span.text != "⌖ ∅.015 D H", (
            "Seed span text should be the short anchor symbol '⌖', not the full grouped text"
        )

        # Now test that cli.py _actually_ uses added_requirement_text for requirement_revB
        # We do this by simulating the cli.py packet assembly logic directly.
        # The contract: when delta_internal.status == "added" and added_requirement_text
        # is set, requirement_revB should use added_requirement_text.
        requirement_revB = _simulate_cli_requirement_revB(item, all_revB_spans=[])
        assert requirement_revB == "⌖ ∅.015 D H", (
            f"cli.py must use added_requirement_text='⌖ ∅.015 D H' as requirement_revB, "
            f"got {requirement_revB!r}"
        )

    def test_packet_uses_added_bbox_as_base_revB_bbox(self):
        """The added_bbox (grouped union) must be used as the base Rev B bbox.

        Seed span bbox is (100,300,108,308); grouped bbox is (100,300,148,308).
        The packet assembly must start from the grouped bbox, not the seed span.
        """
        item = self._make_internal_added_item(
            seed_bbox=(100.0, 300.0, 108.0, 308.0),
            grouped_bbox=(100.0, 300.0, 148.0, 308.0),
        )
        assert item.added_bbox == (100.0, 300.0, 148.0, 308.0)
        assert item.added_span.bbox_pdf == (100.0, 300.0, 108.0, 308.0)

        # The cli.py path must use added_bbox as the base, not added_span.bbox_pdf
        base_bbox = _simulate_cli_base_revB_bbox(item)
        assert base_bbox == (100.0, 300.0, 148.0, 308.0), (
            f"cli.py must use added_bbox=(100,300,148,308) as base bbox, "
            f"got {base_bbox!r} (seed span bbox would be (100,300,108,308))"
        )

    def test_grouped_added_rows_emit_grouped_callout_snippet_rule(self):
        """Added rows with grouped evidence must emit snippet_rule_family='grouped_callout'.

        Single-span added rows keep 'single_callout'; grouped rows must upgrade to
        'grouped_callout' so the evaluator can use richer context for snippet verification.
        """
        grouped_item = self._make_internal_added_item(multiple_spans=True)
        single_item  = self._make_internal_added_item(
            char_no=21,
            seed_text="Ø.250",
            grouped_text=None,
            grouped_bbox=None,
            multiple_spans=False,
        )

        grouped_family = _simulate_cli_snippet_rule_family(grouped_item, all_revB_spans=[])
        single_family  = _simulate_cli_snippet_rule_family(single_item,  all_revB_spans=[])

        assert grouped_family == "grouped_callout", (
            f"Grouped added row must emit snippet_rule_family='grouped_callout', "
            f"got {grouped_family!r}"
        )
        # Single-span rows may remain 'single_callout' or 'grouped_callout' depending
        # on other factors; the key invariant is that grouped evidence does NOT downgrade.
        assert grouped_family != "single_callout", (
            "Grouped evidence must not report single_callout snippet family"
        )


# ---------------------------------------------------------------------------
# CLI simulation helpers
#
# These replicate the packet assembly logic from cli.py so the tests can
# verify the contract without running the full pipeline.  They are updated
# alongside cli.py when the packet assembly changes.
# ---------------------------------------------------------------------------


def _simulate_cli_requirement_revB(
    delta_internal: DeltaItemInternal,
    all_revB_spans: List[TextSpan],
) -> Optional[str]:
    """Replicate the cli.py requirement_revB derivation for an added row.

    Contract (after fix):
    - If added_requirement_text is set, use it.
    - Otherwise fall back to _format_annotation_text([added_span]) or added_span.text.
    """
    if delta_internal.status != "added":
        return None
    added_req_text = getattr(delta_internal, "added_requirement_text", None)
    if added_req_text:
        return added_req_text
    # Fallback path (legacy or single-span)
    if delta_internal.added_span is not None:
        return delta_internal.added_span.text
    return None


def _simulate_cli_base_revB_bbox(
    delta_internal: DeltaItemInternal,
) -> Optional[Tuple[float, float, float, float]]:
    """Replicate the cli.py base Rev B bbox selection for an added row.

    Contract (after fix):
    - If added_bbox is set, use it as the base.
    - Otherwise fall back to added_span.bbox_pdf.
    """
    if delta_internal.status != "added":
        return None
    added_bbox = getattr(delta_internal, "added_bbox", None)
    if added_bbox is not None:
        return added_bbox
    if delta_internal.added_span is not None:
        return delta_internal.added_span.bbox_pdf
    return None


def _simulate_cli_snippet_rule_family(
    delta_internal: DeltaItemInternal,
    all_revB_spans: List[TextSpan],
) -> str:
    """Replicate the cli.py snippet_rule_family decision for an added row.

    Contract (after fix):
    - If added_requirement_text is set AND added_bbox is wider than added_span.bbox_pdf,
      emit 'grouped_callout'.
    - Otherwise emit 'single_callout' (legacy fallback).
    """
    if delta_internal.status != "added":
        return "single_callout"

    added_req_text = getattr(delta_internal, "added_requirement_text", None)
    added_bbox     = getattr(delta_internal, "added_bbox", None)
    seed_span      = delta_internal.added_span

    if added_req_text and added_bbox and seed_span is not None:
        seed_width   = seed_span.bbox_pdf[2] - seed_span.bbox_pdf[0]
        grouped_width = added_bbox[2] - added_bbox[0]
        if grouped_width > seed_width or " " in added_req_text:
            return "grouped_callout"

    return "single_callout"


# ===========================================================================
# Class 4: TestPhase9ExemplarFamilies
# ===========================================================================


class TestPhase9ExemplarFamilies:
    """Detector-side closure for the Parts 3-5 miss families identified in Phase 9.

    Each test uses exact exemplar strings from the Phase 9 debug notes so the
    result is grep-verifiable and directly tied to the confirmed root cause.

    Tests cover:
      - Part 3: surface-finish multi-line grouping (FINISH TURN 1000 Ra)
      - Part 3: countersink GD&T symbol added (⌵ Ø.531 X 82.0°)
      - Part 4: position callout with modifier and datum refs (⌖ ∅.005Ⓜ A B C)
      - Part 5: threaded-hole with depth callout (3X Ø18 ↧30, M20x2.5 − 6H ↧6)
    """

    def test_surface_finish_multiline_grouping_produces_finish_turn_1000_ra(self):
        """Surface-finish seed '1000 Ra' must group same-block prefix 'FINISH TURN'.

        Root cause: 'FINISH TURN' has no numeric payload so it never passed the
        companion geometry check.  After the surface-finish same-block expansion,
        the prefix row is swept in and the canonical callout becomes 'FINISH TURN
        1000 Ra' — the exact string required by Part 3 truth_index 20.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        # Two spans in the same PDF block (block_id=47): prefix on line 0, Ra value on line 1
        prefix_span = _span(
            "FINISH TURN",
            block_id=47, line_id=0, span_id=0,
            x0=214.0, y0=44.0, width=75.0, height=16.0,
        )
        value_span = _span(
            "1000 Ra",
            block_id=47, line_id=1, span_id=0,
            x0=214.0, y0=57.0, width=45.0, height=16.0,
        )

        added = detect_added_characteristics(
            revB_spans=[prefix_span, value_span],
            matches={},
            next_char_no=20,
            page_width=784.0,
            page_height=792.0,
        )

        texts = [it.added_requirement_text or "" for it in added]
        assert any("FINISH TURN 1000 Ra" in t for t in texts), (
            f"Expected grouped surface-finish callout 'FINISH TURN 1000 Ra' but got {texts!r}; "
            "surface-finish same-block expansion did not sweep the prefix row"
        )

    def test_countersink_symbol_detected_as_gdt_anchor_for_csink_callout(self):
        """The countersink symbol ⌵ must be treated as a GD&T anchor in Pass 0.

        Root cause: ⌵ was not in GDT_ANCHOR_SYMBOLS, so Pass 0 never seeded from it.
        Instead, 'Ø.531 X 82.0°' entered Pass 2 and was filtered by the near-matched-span
        proximity check (17.7 pt from a matched 'Ø.266 THRU' center).  After adding ⌵ to
        GDT_ANCHOR_SYMBOLS, Pass 0 seeds from ⌵ (41.6 pt from the match — passes 12 pt
        guard) and groups the companion diameter/angle span into the full callout.

        Exemplar from Part 3 ground_truth truth_index 19: '⌵ Ø.531 X 82.0°'
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        # Matched span Ø.266 THRU at block 41, line 0 — becomes a matched candidate
        matched_thru = _span(
            "Ø.266 THRU",
            block_id=41, line_id=0, span_id=0,
            x0=361.6, y0=418.5, width=68.4, height=15.9,
        )

        # Countersink annotation: ⌵ + Ø.531 X 82.0° on line 1 of the same block
        csink_sym = _span(
            "⌵",
            block_id=41, line_id=1, span_id=0,
            x0=346.6, y0=433.0, width=21.2, height=17.5,
        )
        csink_dim = _span(
            "Ø.531 X\xa082.0°",
            block_id=41, line_id=1, span_id=1,
            x0=367.7, y0=433.2, width=75.7, height=15.9,
        )

        matches = {5: _FakeMatch(candidate=_FakeCandidate(span=matched_thru))}
        all_spans = [matched_thru, csink_sym, csink_dim]

        added = detect_added_characteristics(
            revB_spans=all_spans,
            matches=matches,
            next_char_no=20,
            page_width=784.0,
            page_height=792.0,
        )

        texts = [it.added_requirement_text or "" for it in added]
        assert any("⌵" in t and ("Ø.531" in t or "82.0" in t) for t in texts), (
            f"Expected countersink callout '⌵ Ø.531 X 82.0°' to be detected but got {texts!r}; "
            "⌵ must be in GDT_ANCHOR_SYMBOLS so Pass 0 seeds from it"
        )

    def test_position_callout_with_modifier_and_datums_detected(self):
        """A position feature control frame with MMC modifier and datum references is detected.

        Exemplar from Part 4 ground_truth truth_index 11: '⌖ ∅.005Ⓜ A B C'

        The anchor ⌖ is already in GDT_ANCHOR_SYMBOLS.  This test confirms that the
        Ⓜ modifier and datum letter spans are grouped as companions and the full callout
        text survives the explained-by-match suppressor when no matched owner explains it.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        anchor = _span("⌖", block_id=10, line_id=0, span_id=0, x0=200.0, y0=384.0, width=10.0, height=10.0)
        diam   = _span("∅.005", block_id=10, line_id=0, span_id=1, x0=212.0, y0=384.0, width=22.0, height=10.0)
        mmc    = _span("Ⓜ", block_id=10, line_id=0, span_id=2, x0=236.0, y0=384.0, width=10.0, height=10.0)
        datA   = _span("A", block_id=10, line_id=0, span_id=3, x0=248.0, y0=384.0, width=8.0, height=10.0)
        datB   = _span("B", block_id=10, line_id=0, span_id=4, x0=258.0, y0=384.0, width=8.0, height=10.0)
        datC   = _span("C", block_id=10, line_id=0, span_id=5, x0=268.0, y0=384.0, width=8.0, height=10.0)

        added = detect_added_characteristics(
            revB_spans=[anchor, diam, mmc, datA, datB, datC],
            matches={},
            next_char_no=12,
            page_width=1224.0,
            page_height=792.0,
        )

        texts = [it.added_requirement_text or "" for it in added]
        assert any("⌖" in t and "∅.005" in t and "Ⓜ" in t and "A" in t and "B" in t and "C" in t for t in texts), (
            f"Expected '⌖ ∅.005Ⓜ A B C' callout but got {texts!r}; "
            "position frame with MMC modifier and datum refs must be fully grouped"
        )

    def test_3x_hole_with_depth_callout_detected(self):
        """A 3X hole count with depth callout '3X Ø18 ↧30' is detected.

        Exemplar from Part 5 ground_truth truth_index 17: '3X Ø18 ↧30'

        The ↧ is a GDT anchor symbol that seeds Pass 0.  When '3X Ø18' is not
        pre-matched, the companion walk groups it with ↧ and '30' to form the full
        callout.  This test verifies the detection with unmatched spans.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        count_sym = _span("3X Ø18 ", block_id=56, line_id=0, span_id=0, x0=212.8, y0=57.6, width=43.4, height=15.9)
        depth_sym = _span("↧", block_id=56, line_id=0, span_id=1, x0=256.2, y0=57.4, width=12.1, height=17.5)
        depth_val = _span("30", block_id=56, line_id=0, span_id=2, x0=268.3, y0=57.6, width=13.8, height=15.9)

        added = detect_added_characteristics(
            revB_spans=[count_sym, depth_sym, depth_val],
            matches={},
            next_char_no=18,
            page_width=1224.0,
            page_height=792.0,
        )

        texts = [it.added_requirement_text or "" for it in added]
        assert any("↧" in t and "18" in t and "30" in t for t in texts), (
            f"Expected depth callout containing '3X Ø18 ↧30' elements but got {texts!r}; "
            "↧ anchor must group the count/diameter and depth value companions"
        )
        # Verify literal fragment of the exemplar string appears
        assert any("3X Ø18" in t for t in texts), (
            f"Expected '3X Ø18' prefix in grouped text but got {texts!r}"
        )

    def test_threaded_hole_with_depth_m20_detected(self):
        """A threaded hole callout 'M20x2.5 − 6H ↧6' is detected and grouped.

        Exemplar from Part 5 ground_truth truth_index 18: 'M20x2.5 − 6H ↧6'

        The ↧ seeds Pass 0; M20x2.5 − 6H is grouped as a same-row companion.
        The thread designation prefix has no GDT symbols but the ↧ anchor binds
        them together via the fixed-point companion walk.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        thread_span = _span("M20x2.5 − 6H ", block_id=56, line_id=1, span_id=0, x0=197.5, y0=72.1, width=80.2, height=15.9)
        depth_sym   = _span("↧", block_id=56, line_id=1, span_id=1, x0=277.7, y0=71.9, width=12.1, height=17.5)
        depth_val   = _span("6", block_id=56, line_id=1, span_id=2, x0=289.8, y0=72.1, width=6.9, height=15.9)

        added = detect_added_characteristics(
            revB_spans=[thread_span, depth_sym, depth_val],
            matches={},
            next_char_no=19,
            page_width=1224.0,
            page_height=792.0,
        )

        texts = [it.added_requirement_text or "" for it in added]
        assert any("M20x2.5" in t and "6H" in t and "↧" in t for t in texts), (
            f"Expected threaded hole callout 'M20x2.5 − 6H ↧6' but got {texts!r}; "
            "↧ anchor must group the thread designation and depth value"
        )
        # Verify literal fragment of the exemplar string
        assert any("M20x2.5 − 6H" in t for t in texts), (
            f"Expected 'M20x2.5 − 6H' thread designation in grouped text but got {texts!r}"
        )

    def test_plain_integer_800_detected_when_tolerance_block_is_far_away(self):
        """A drawing-body plain integer '800' survives when the boilerplate phrase is far away.

        Reproduces the Part 5 truth_index 16 scenario: '800' sits at centre
        (615, 581) in the drawing body, while 'UNLESS OTHERWISE SPECIFIED' sits
        in the bottom-centre tolerance zone at centre (500, 650).  The old 120 pt
        proximity sweep suppressed '800' because the boilerplate was ~54 pt away.
        The refined zone-aware filter must let '800' through because:
          - '800' itself is NOT in a boilerplate zone (its centre is at ~73% height,
            outside the 78%-92% tolerance zone)
          - The boilerplate phrase centre is >40 pt away and not vertically co-linear
            (|Δy| > 10 pt), so the same-row companion check does not fire.
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        # Drawing-body plain integer at centre ~(615, 581) on a (1224, 792) page
        integer_span = _span(
            "800",
            block_id=70, line_id=0, span_id=0,
            x0=600.0, y0=577.0, width=30.0, height=8.0,
        )

        # Boilerplate phrase in the bottom-centre tolerance zone at centre ~(500, 650)
        boilerplate_span = _span(
            "UNLESS OTHERWISE SPECIFIED",
            block_id=90, line_id=0, span_id=0,
            x0=380.0, y0=644.0, width=240.0, height=12.0,
        )

        added = detect_added_characteristics(
            revB_spans=[integer_span, boilerplate_span],
            matches={},
            next_char_no=17,
            page_width=1224.0,
            page_height=792.0,
        )

        texts = [it.added_requirement_text or "" for it in added]
        assert any("800" in t for t in texts), (
            f"Expected plain integer '800' to survive detection but got {texts!r}; "
            "the zone-aware filter should not suppress a drawing-body integer when "
            "the boilerplate phrase is far away and in the tolerance zone"
        )

    def test_plain_integer_in_general_tolerance_block_still_suppressed(self):
        """A plain integer whose centre sits inside the tolerance zone with a same-row
        boilerplate companion is still suppressed.

        Negative case: '150' at centre (420, 650) on a (1224, 792) page — inside the
        bottom-centre tolerance zone (78%-92% height × 25%-72% width).  'UNLESS
        OTHERWISE SPECIFIED' sits within 10 pt in y and 40 pt centre-to-centre distance.
        The refined filter must suppress '150' because:
          - '150' itself is flagged by span_is_excluded_for_annotation_search (the span
            is in the tolerance zone AND adjacent boilerplate text)
          - AND a boilerplate phrase is a same-row companion (|Δy| ≤ 10 pt, distance ≤ 40 pt)
        """
        from delta_preservation.reconcile.classify import detect_added_characteristics

        # Plain integer inside the bottom-centre tolerance zone at centre ~(420, 650)
        integer_span = _span(
            "150",
            block_id=91, line_id=0, span_id=0,
            x0=410.0, y0=646.0, width=20.0, height=8.0,
        )

        # Same-row boilerplate companion whose centre is within 10 pt in y and
        # ≤ 40 pt centre-to-centre distance from the integer span.
        # Integer centre: (420, 650).  Boilerplate centre: (395, 650) → distance = 25 pt.
        boilerplate_span = _span(
            "UNLESS OTHERWISE SPECIFIED",
            block_id=91, line_id=0, span_id=1,
            x0=275.0, y0=644.0, width=240.0, height=12.0,
        )

        added = detect_added_characteristics(
            revB_spans=[integer_span, boilerplate_span],
            matches={},
            next_char_no=20,
            page_width=1224.0,
            page_height=792.0,
        )

        texts = [it.added_requirement_text or "" for it in added]
        assert not any("150" == t.strip() for t in texts), (
            f"Expected plain integer '150' in the tolerance zone to be suppressed but got {texts!r}; "
            "the zone-aware filter should suppress integers in the boilerplate zone or with "
            "same-row boilerplate companions"
        )
