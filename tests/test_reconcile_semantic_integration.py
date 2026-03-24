from delta_preservation.io.pdf import TextSpan
from delta_preservation.reconcile.anchors import Anchor
from delta_preservation.reconcile.classify import classify_delta
from delta_preservation.reconcile.match import Match, generate_candidates
from delta_preservation.reconcile.normalize import extract_semantic_callout


class _Transform:
    def __init__(self):
        import numpy as np

        self.H = np.eye(3, dtype=float)
        self.inliers = 8
        self.inlier_ratio = 1.0


def _span(text: str, *, block_id: int = 0, line_id: int = 0, span_id: int = 0, x0: float = 10.0, y0: float = 10.0) -> TextSpan:
    return TextSpan(
        text=text,
        bbox_pdf=(x0, y0, x0 + 20.0, y0 + 8.0),
        font_size=10.0,
        block_id=block_id,
        line_id=line_id,
        span_id=span_id,
    )


def _anchor(requirement_raw: str) -> Anchor:
    anchor_span = _span(requirement_raw, x0=10.0, y0=10.0)
    return Anchor(
        char_no=1,
        page=0,
        balloon_bbox=(0.0, 0.0, 5.0, 5.0),
        req_bbox=anchor_span.bbox_pdf,
        requirement_raw=requirement_raw,
        requirement_norm=requirement_raw,
        local_context=[anchor_span],
    )


def _span_key(span: TextSpan):
    return (span.block_id, span.line_id, span.span_id, span.bbox_pdf)


def test_reconcile_semantic_integration_keeps_equivalent_gdt_callout_unchanged():
    anchor = _anchor("⌖ ⌀0.10 M A B C")
    candidate_span = _span("⌖    ⌀0.10   M   A  B C", block_id=4, line_id=2, span_id=1, x0=12.0, y0=11.0)

    anchor_semantic = extract_semantic_callout(pdf_spans=[_span(anchor.requirement_raw)], form3_requirement=anchor.requirement_raw)
    revb_semantic_by_span = {
        _span_key(candidate_span): extract_semantic_callout(pdf_spans=[candidate_span])
    }

    candidates = generate_candidates(
        anchor,
        [candidate_span],
        _Transform(),
        anchor_semantic_callout=anchor_semantic,
        revB_semantic_callouts_by_span_key=revb_semantic_by_span,
    )

    assert candidates
    assert any("semantic GD&T match" in reason for reason in candidates[0].reasons)

    match = Match(char_no=1, candidate=candidates[0])
    delta = classify_delta(
        anchor,
        match,
        anchor_semantic_callout=anchor_semantic,
        matched_semantic_callout=revb_semantic_by_span[_span_key(candidate_span)],
    )

    assert delta.status == "unchanged"
    assert any("semantic GD&T match" in reason for reason in delta.reasons)
    assert any("Semantic family agreement" in reason for reason in delta.reasons)


def test_reconcile_semantic_integration_marks_changed_weld_semantic_delta():
    anchor = _anchor("1/8 FILLET BOTH SIDES ALL AROUND 1.50-3.00 FLUSH TAIL: FIELD")
    candidate_span = _span("3/16 FILLET BOTH SIDES ALL AROUND 1.50-3.00 FLUSH TAIL: FIELD", block_id=5, line_id=1, span_id=0, x0=11.0, y0=10.5)

    anchor_semantic = extract_semantic_callout(pdf_spans=[_span(anchor.requirement_raw)], form3_requirement=anchor.requirement_raw)
    matched_semantic = extract_semantic_callout(pdf_spans=[candidate_span])
    revb_semantic_by_span = {_span_key(candidate_span): matched_semantic}

    candidates = generate_candidates(
        anchor,
        [candidate_span],
        _Transform(),
        anchor_semantic_callout=anchor_semantic,
        revB_semantic_callouts_by_span_key=revb_semantic_by_span,
    )

    assert candidates
    assert any("semantic weld changed: size 1/8 → 3/16" in reason for reason in candidates[0].reasons)

    match = Match(char_no=1, candidate=candidates[0])
    delta = classify_delta(
        anchor,
        match,
        anchor_semantic_callout=anchor_semantic,
        matched_semantic_callout=matched_semantic,
    )

    assert delta.status == "changed"
    assert any("semantic weld changed: size 1/8 → 3/16" in reason for reason in delta.reasons)
    assert any("meaningful drawing requirement change" in reason for reason in delta.reasons)


def test_reconcile_semantic_integration_fallback_to_numeric_reasoning_when_semantics_unparsed():
    anchor = _anchor("12.0 ± 0.1")
    candidate_span = _span("12.0", block_id=2, line_id=0, span_id=0, x0=11.0, y0=10.5)

    anchor_semantic = extract_semantic_callout(pdf_spans=[_span(anchor.requirement_raw)], form3_requirement=anchor.requirement_raw)
    matched_semantic = extract_semantic_callout(pdf_spans=[candidate_span])
    revb_semantic_by_span = {_span_key(candidate_span): matched_semantic}

    candidates = generate_candidates(
        anchor,
        [candidate_span],
        _Transform(),
        anchor_semantic_callout=anchor_semantic,
        revB_semantic_callouts_by_span_key=revb_semantic_by_span,
    )

    assert candidates
    assert any("semantic comparison fallback" in reason for reason in candidates[0].reasons)

    match = Match(char_no=1, candidate=candidates[0])
    delta = classify_delta(
        anchor,
        match,
        anchor_semantic_callout=anchor_semantic,
        matched_semantic_callout=matched_semantic,
    )

    assert delta.status == "unchanged"
    assert any("semantic comparison fallback" in reason for reason in delta.reasons)
    assert any("Primary dimension matches" in reason for reason in delta.reasons)
