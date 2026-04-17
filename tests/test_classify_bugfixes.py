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

        anchor_semantic = extract_semantic_callout(
            pdf_spans=[_span(anchor.requirement_raw)],
            form3_requirement=anchor.requirement_raw,
        )
        matched_semantic = extract_semantic_callout(pdf_spans=[candidate])
        revb_semantic_by_span = {_span_key(candidate): matched_semantic}
        candidates = generate_candidates(
            anchor,
            [candidate],
            _Transform(),
            anchor_semantic_callout=anchor_semantic,
            revB_semantic_callouts_by_span_key=revb_semantic_by_span,
        )

        assert candidates == []

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


class TestConfidenceFlagsCompatibility:
    """Scaffold compatibility tests for the additive confidence_flags field.

    These tests assert concrete backward-compat properties on both the internal
    (dataclass) and persisted (Pydantic) DeltaItem models without relying on any
    fixture files.
    """

    def test_internal_deltaitem_defaults_to_independent_lists(self):
        """Each internal DeltaItem must default confidence_flags to an independent list.

        Two separate instances must not share the same default list object — a
        common mistake when a mutable default is used directly.
        """
        from delta_preservation.reconcile.classify import DeltaItem as InternalDeltaItem

        item_a = InternalDeltaItem(
            char_no=1, status="unchanged", confidence=0.9, reasons=[], component_scores={}
        )
        item_b = InternalDeltaItem(
            char_no=2, status="changed", confidence=0.7, reasons=[], component_scores={}
        )

        assert item_a.confidence_flags == [], f"Expected [], got {item_a.confidence_flags!r}"
        assert item_b.confidence_flags == [], f"Expected [], got {item_b.confidence_flags!r}"
        # Independence check — mutating one must not affect the other
        item_a.confidence_flags.append("test_flag")
        assert item_b.confidence_flags == [], "confidence_flags default lists are not independent"

    def test_packet_deltaitem_validates_legacy_payload_without_confidence_flags(self):
        """A legacy JSON payload that omits confidence_flags must still parse successfully.

        This verifies backward-compat: clients serializing DeltaItem before the
        confidence_flags field was added must not break on deserialization.
        """
        from delta_preservation.types import DeltaItem as PacketDeltaItem

        legacy_payload = {
            "char_no": 5,
            "status": "unchanged",
            "confidence": 0.85,
            "reasons": ["Primary dimension matches: 12.5"],
            "scores": {"location": 0.9, "text": 0.8, "context": 0.7},
            # confidence_flags intentionally absent — simulates legacy JSON
        }

        item = PacketDeltaItem.model_validate(legacy_payload)
        assert item.confidence_flags == [], (
            f"Legacy payload without confidence_flags should default to [], got {item.confidence_flags!r}"
        )

    def test_packet_deltaitem_model_dump_includes_confidence_flags(self):
        """model_dump() on a PacketDeltaItem must include the confidence_flags key."""
        from delta_preservation.types import DeltaItem as PacketDeltaItem

        item = PacketDeltaItem(
            char_no=3,
            status="removed",
            confidence=0.6,
            reasons=["No candidate found"],
            scores={"location": 0.0, "text": 0.0, "context": 0.0},
            confidence_flags=["adjacency_bleed_risk"],
        )

        dumped = item.model_dump()
        assert "confidence_flags" in dumped, "model_dump() must include confidence_flags"
        assert dumped["confidence_flags"] == ["adjacency_bleed_risk"], (
            f"Expected ['adjacency_bleed_risk'], got {dumped['confidence_flags']!r}"
        )


class TestAdjacencyBleed:
    """CLS-01: slash-merged bleed spans must not cause false count_added positive."""

    def test_part1_style_bleed_not_changed(self):
        """part-1-style positive exemplar: counterbore span with slash-merged sibling.

        Anchor: Counterbore Diameter (13.5 +/- 0.2 mm)
        Span:   4 x Ø8 THRU ALL / ⌴ Ø13.5 ↧ 8.5

        The span contains the anchor's numeric (13.5) in the second chunk, and the
        first chunk contains a foreign count prefix (4 x).  This is adjacency bleed.
        Status must NOT be 'changed' solely because of the bleed prefix, and the
        bleed flag must be present in confidence_flags.
        """
        anchor = _anchor("Counterbore Diameter (13.5 +/- 0.2 mm)")
        candidate = _span(
            "4 x Ø8 THRU ALL / ⌴ Ø13.5 ↧ 8.5",
            block_id=1, line_id=0, span_id=0, x0=12.0, y0=11.0, width=80.0
        )
        delta = _classify(anchor, candidate)
        assert delta.status != "changed", (
            f"Bleed should suppress count_added 'changed'; got '{delta.status}'"
        )
        assert "Rev B text may contain adjacent balloon content" in delta.confidence_flags, (
            f"Bleed flag missing; confidence_flags={delta.confidence_flags!r}"
        )

    def test_part4_style_bleed_no_count_added(self):
        """part-4-style positive exemplar: thread callout with slash-merged sibling.

        Anchor: 1/4-20 UNC-2B
        Span:   2X Ø.201 ↧ 0.50 / 1/4-20 UNC - 2B

        The span contains the anchor keyword in the second chunk, and the first chunk
        carries a foreign count prefix (2X).  Bleed must be detected and the bleed
        flag must appear in confidence_flags.
        """
        anchor = _anchor("1/4-20 UNC-2B")
        candidate = _span(
            "2X Ø.201 ↧ 0.50 / 1/4-20 UNC - 2B",
            block_id=2, line_id=0, span_id=0, x0=12.0, y0=11.0, width=80.0
        )
        delta = _classify(anchor, candidate)
        assert "Rev B text may contain adjacent balloon content" in delta.confidence_flags, (
            f"Bleed flag missing for part-4-style case; confidence_flags={delta.confidence_flags!r}"
        )

    def test_benign_slash_ratio_no_bleed(self):
        """Benign slash ratio negative case: '70 / 30' for anchor '30 ±0.5 mm'.

        The span is just a fractional ratio — both chunks share numeric context,
        neither has a foreign count prefix.  No bleed flag expected.
        """
        anchor = _anchor("30 ±0.5 mm")
        candidate = _span(
            "70 / 30",
            block_id=3, line_id=0, span_id=0, x0=12.0, y0=11.0, width=30.0
        )
        delta = _classify(anchor, candidate)
        assert "Rev B text may contain adjacent balloon content" not in delta.confidence_flags, (
            f"Bleed flag should not fire for benign ratio; confidence_flags={delta.confidence_flags!r}"
        )

    def test_legitimate_asymmetric_tolerance_no_bleed_stays_changed(self):
        """Legitimate asymmetric tolerance negative case: '2X 22.0° +0.3° / −0.1°'.

        The slash separates the plus/minus parts of an asymmetric tolerance, NOT
        an adjacent balloon.  No bleed flag; status must remain 'changed'.
        """
        anchor = _anchor("2X 22.0° ±1°")
        candidate = _span(
            "2X 22.0° +0.3° / −0.1°",
            block_id=4, line_id=0, span_id=0, x0=12.0, y0=11.0, width=60.0
        )
        delta = _classify(anchor, candidate)
        assert "Rev B text may contain adjacent balloon content" not in delta.confidence_flags, (
            f"Bleed flag should not fire for asymmetric tolerance; confidence_flags={delta.confidence_flags!r}"
        )
        assert delta.status == "changed", (
            f"Asymmetric tolerance case must remain 'changed'; got '{delta.status}'"
        )

    def test_no_slash_regression_guard_stays_changed(self):
        """No-slash regression guard: anchor 'Ø 8' + span '2X Ø 8'.

        Original count_added detection must still fire when there is no slash.
        Status must be 'changed' and no bleed flag.
        """
        anchor = _anchor("Ø 8")
        candidate = _span(
            "2X Ø 8",
            block_id=5, line_id=0, span_id=0, x0=12.0, y0=11.0, width=30.0
        )
        delta = _classify(anchor, candidate)
        assert delta.status == "changed", (
            f"No-slash count_added must still be 'changed'; got '{delta.status}'"
        )
        assert "Rev B text may contain adjacent balloon content" not in delta.confidence_flags, (
            f"Bleed flag must not fire for no-slash case; confidence_flags={delta.confidence_flags!r}"
        )


class TestToleranceOverlapThreshold:
    """Tolerance-only numeric changes in the 65-70% overlap band should be marked changed."""

    def test_angle_tolerance_change_above_old_threshold_marks_changed(self):
        anchor = _anchor("2X 22.0° ±1°")
        candidate = _span("2X 22.0° +0.3° -0.1°", block_id=4, line_id=0, span_id=0, x0=12.0, y0=11.0, width=48.0)

        delta = _classify(anchor, candidate)

        assert delta.status == "changed", f"Expected 'changed', got '{delta.status}' with reasons={delta.reasons}"
        assert any("Tolerance value changed" in reason for reason in delta.reasons)


# ---------------------------------------------------------------------------
# CLS-03: Asymmetric tolerance kind transition detection
# ---------------------------------------------------------------------------

class TestAsymmetricTolerance:
    """CLS-03: Symmetric-to-asymmetric tolerance kind transitions must be detected
    even when absolute limits happen to match (tolerances_match=True) and even
    when tolerance_comparison is None (raw-text fallback path)."""

    def _make_tolerance_comparison(
        self,
        revA_kind: str,
        revB_kind: str,
        nominal: float = 22.0,
        revA_tol: float = 1.0,
        revB_plus: float = 0.3,
        revB_minus: float = 0.1,
        limits_match: bool = False,
        char_no: int = 1,
    ):
        """Build a fabricated ToleranceComparison for unit-testing the kind-transition branch."""
        from delta_preservation.reconcile.tolerance_pdf import (
            PdfTolerance,
            ToleranceComparison,
        )

        revA_tolerance = PdfTolerance(
            kind=revA_kind,
            nominal_value=nominal,
            upper_limit=nominal + revA_tol,
            lower_limit=nominal - revA_tol,
            confidence=0.95,
            source_spans=[],
            reasons=["fabricated for test"],
        )

        if limits_match:
            # Rev B limits equal Rev A limits even though kind differs
            revB_upper = nominal + revA_tol
            revB_lower = nominal - revA_tol
        else:
            revB_upper = nominal + revB_plus
            revB_lower = nominal - revB_minus

        revB_tolerance = PdfTolerance(
            kind=revB_kind,
            nominal_value=nominal,
            upper_limit=revB_upper,
            lower_limit=revB_lower,
            confidence=0.90,
            source_spans=[],
            reasons=["fabricated for test"],
        )

        # Determine match/differ from actual limit values
        import math
        _eps = 1e-6
        upper_ok = abs(revA_tolerance.upper_limit - revB_tolerance.upper_limit) < _eps
        lower_ok = abs(revA_tolerance.lower_limit - revB_tolerance.lower_limit) < _eps
        tol_match = upper_ok and lower_ok
        tol_differ = not tol_match

        return ToleranceComparison(
            char_no=char_no,
            revA_tolerance=revA_tolerance,
            revB_tolerance=revB_tolerance,
            tolerances_match=tol_match,
            tolerances_differ=tol_differ,
            has_tolerance=True,
            reasons=["fabricated comparison for CLS-03 test"],
        )

    def test_symmetric_to_asymmetric_kind_change_detected_before_tolerances_match(self):
        """CLS-03 positive case 1: 22.0° ±1° → 22.0° +0.3°/−0.1° with high numeric overlap.

        The anchor and candidate share primary value 22.0 so the main classification
        would normally emit 'unchanged' (primary_matches=True, numeric_overlap≥50%).
        With a ToleranceComparison that has tolerances_differ=True AND a kind transition
        from plus_minus → bilateral_stacked, the kind-transition branch must fire and
        produce 'changed' with a reason naming the kind transition.
        """
        # Anchor: "22.0° ±1°" — numeric tokens {22.0, 1.0}
        anchor = _anchor("22.0° ±1°")
        # Candidate: "22.0° +0.3° / −0.1°" — shares primary 22.0
        candidate = _span("22.0° +0.3° / −0.1°", block_id=10, line_id=0, span_id=0, x0=12.0, y0=11.0, width=60.0)

        tc = self._make_tolerance_comparison(
            revA_kind="plus_minus",
            revB_kind="bilateral_stacked",
            nominal=22.0,
            revA_tol=1.0,
            revB_plus=0.3,
            revB_minus=0.1,
        )
        # tc.tolerances_match is False here (limits differ); tc.tolerances_differ is True.
        # The kind-first check must fire and name the kind transition in reasons.
        from delta_preservation.reconcile.match import Match, Candidate
        cand = Candidate(
            span=candidate,
            total_score=0.85,
            location_score=0.85,
            text_score=0.9,
            context_score=0.7,
            reasons=["fabricated for test"],
        )
        match = Match(char_no=anchor.char_no, candidate=cand)

        delta = classify_delta(anchor, match, tolerance_comparison=tc)
        assert delta.status == "changed", (
            f"Kind transition (plus_minus → bilateral_stacked) must produce 'changed'; "
            f"got '{delta.status}' with reasons={delta.reasons}"
        )
        # The reason must mention kind transition, not just generic tolerance change
        assert any(
            "kind" in r.lower() or "asymmetric" in r.lower() or "transition" in r.lower()
            for r in delta.reasons
        ), f"Expected kind-transition reason; got reasons={delta.reasons}"

    def test_leading_decimal_fallback_detected_without_tolerance_comparison(self):
        """CLS-03 positive case 2: leading-decimal asymmetric form with no tolerance_comparison.

        anchor='Ø.250 ±.002'  →  candidate='Ø.250 +.005 / -.003'
        tolerance_comparison=None (fallback path using raw text).
        Status must be 'changed'.
        """
        anchor = _anchor("Ø.250 ±.002")
        candidate = _span("Ø.250 +.005 / -.003", block_id=11, line_id=0, span_id=0, x0=12.0, y0=11.0, width=50.0)

        delta = _classify(anchor, candidate)
        assert delta.status == "changed", (
            f"Leading-decimal asymmetric form must produce 'changed'; "
            f"got '{delta.status}' with reasons={delta.reasons}"
        )

    def test_already_changed_not_downgraded(self):
        """CLS-03 negative case 3: item already 'changed' before tolerance refinement.

        When count/numeric mismatch already set status='changed', the new kind-transition
        logic must not downgrade it back to 'unchanged'.
        """
        from delta_preservation.reconcile.match import Match, Candidate

        # Anchor has count "2X", candidate drops the count → count_changed fires → changed
        anchor = _anchor("2X Ø 8")
        candidate = _span("4X Ø 8", block_id=12, line_id=0, span_id=0, x0=12.0, y0=11.0, width=30.0)

        # Build a tolerance_comparison that says tolerances_match=True (same kind, same limits)
        tc = self._make_tolerance_comparison(
            revA_kind="plus_minus",
            revB_kind="plus_minus",
            limits_match=True,
        )

        cand = Candidate(
            span=candidate,
            total_score=0.85,
            location_score=0.85,
            text_score=0.9,
            context_score=0.7,
            reasons=["fabricated for test"],
        )
        match = Match(char_no=anchor.char_no, candidate=cand)

        delta = classify_delta(anchor, match, tolerance_comparison=tc)
        # Count changed from 2X → 4X; tolerance_comparison says match.
        # The result must still be 'changed' — tolerances_match must not override count change.
        assert delta.status == "changed", (
            f"Count-changed item must stay 'changed' even if tolerances_match=True; "
            f"got '{delta.status}' with reasons={delta.reasons}"
        )

    def test_same_kind_symmetric_routes_through_existing_logic(self):
        """CLS-03 negative case 4 (control): 22.0° ±1° → 22.0° ±0.5°.

        No kind transition (both plus_minus); this is a same-kind change and must
        still be detected as 'changed' via the existing tolerances_differ path.
        The new kind-transition branch must NOT be entered when kinds are equal.
        """
        from delta_preservation.reconcile.match import Match, Candidate

        anchor = _anchor("22.0° ±1°")
        candidate = _span("22.0° ±0.5°", block_id=13, line_id=0, span_id=0, x0=12.0, y0=11.0, width=40.0)

        # Both plus_minus; limits differ (±1 vs ±0.5)
        tc = self._make_tolerance_comparison(
            revA_kind="plus_minus",
            revB_kind="plus_minus",
            nominal=22.0,
            revA_tol=1.0,
            revB_plus=0.5,
            revB_minus=0.5,
        )
        cand = Candidate(
            span=candidate,
            total_score=0.8,
            location_score=0.8,
            text_score=0.5,
            context_score=0.5,
            reasons=["fabricated for test"],
        )
        match = Match(char_no=anchor.char_no, candidate=cand)

        delta = classify_delta(anchor, match, tolerance_comparison=tc)
        assert delta.status == "changed", (
            f"Same-kind tolerance change must still be 'changed'; "
            f"got '{delta.status}' with reasons={delta.reasons}"
        )

    def test_benign_asymmetric_no_transition_still_changes(self):
        """CLS-03 negative case 5 (control): +0.3/-0.1 → +0.4/-0.1.

        Both sides are bilateral_stacked (asymmetric); no kind transition.
        Limit change must still produce 'changed' via the existing tolerances_differ path.
        """
        from delta_preservation.reconcile.match import Match, Candidate

        anchor = _anchor("22.0 +0.3 / -0.1")
        candidate = _span("22.0 +0.4 / -0.1", block_id=14, line_id=0, span_id=0, x0=12.0, y0=11.0, width=45.0)

        tc = self._make_tolerance_comparison(
            revA_kind="bilateral_stacked",
            revB_kind="bilateral_stacked",
            revA_tol=0.3,
            revB_plus=0.4,
            revB_minus=0.1,
        )
        cand = Candidate(
            span=candidate,
            total_score=0.85,
            location_score=0.85,
            text_score=0.8,
            context_score=0.7,
            reasons=["fabricated for test"],
        )
        match = Match(char_no=anchor.char_no, candidate=cand)

        delta = classify_delta(anchor, match, tolerance_comparison=tc)
        assert delta.status == "changed", (
            f"Bilateral→bilateral tolerance change must be 'changed'; "
            f"got '{delta.status}' with reasons={delta.reasons}"
        )
