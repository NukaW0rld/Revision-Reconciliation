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


def _span(text: str, *, block_id: int = 0, line_id: int = 0, span_id: int = 0, x0: float = 10.0, y0: float = 10.0, width: float = 20.0, height: float = 8.0) -> TextSpan:
    return TextSpan(
        text=text,
        bbox_pdf=(x0, y0, x0 + width, y0 + height),
        font_size=10.0,
        block_id=block_id,
        line_id=line_id,
        span_id=span_id,
    )


def _anchor(requirement_raw: str) -> Anchor:
    anchor_span = _span(requirement_raw, x0=10.0, y0=10.0, width=120.0)
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


def _classify(anchor: Anchor, candidate_span: TextSpan):
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
    match = Match(char_no=anchor.char_no, candidate=candidates[0])
    delta = classify_delta(
        anchor,
        match,
        anchor_semantic_callout=anchor_semantic,
        matched_semantic_callout=matched_semantic,
    )
    return candidates[0], delta, anchor_semantic, matched_semantic


def test_reconcile_semantic_integration_keeps_equivalent_gdt_callout_unchanged():
    anchor = _anchor("⌖ ⌀0.10 M A B C")
    candidate_span = _span("⌖    ⌀0.10   M   A  B C", block_id=4, line_id=2, span_id=1, x0=12.0, y0=11.0)

    candidate, delta, anchor_semantic, matched_semantic = _classify(anchor, candidate_span)

    assert anchor_semantic.status.parser_family == "gdt"
    assert matched_semantic.status.parser_family == "gdt"
    assert any("semantic GD&T match" in reason for reason in candidate.reasons)
    assert delta.status == "unchanged"
    assert any("semantic GD&T match" in reason for reason in delta.reasons)
    assert any("Semantic family agreement" in reason for reason in delta.reasons)


def test_reconcile_semantic_integration_marks_changed_weld_semantic_delta():
    anchor = _anchor("1/8 FILLET BOTH SIDES ALL AROUND 1.50-3.00 FLUSH TAIL: FIELD")
    candidate_span = _span("3/16 FILLET BOTH SIDES ALL AROUND 1.50-3.00 FLUSH TAIL: FIELD", block_id=5, line_id=1, span_id=0, x0=11.0, y0=10.5, width=120.0)

    anchor_semantic = extract_semantic_callout(pdf_spans=[_span(anchor.requirement_raw)], form3_requirement=anchor.requirement_raw)
    matched_semantic = extract_semantic_callout(pdf_spans=[candidate_span])
    synthetic_candidate = type(
        "SyntheticCandidate",
        (),
        {
            "span": candidate_span,
            "total_score": 0.8,
            "location_score": 0.9,
            "text_score": 0.9,
            "context_score": 0.0,
            "reasons": ["semantic weld changed: size 1/8 → 3/16"],
            "from_global_fallback": False,
        },
    )()
    delta = classify_delta(
        anchor,
        Match(char_no=anchor.char_no, candidate=synthetic_candidate),
        anchor_semantic_callout=anchor_semantic,
        matched_semantic_callout=matched_semantic,
    )

    assert anchor_semantic.status.parser_family == "weld"
    assert matched_semantic.status.parser_family == "weld"
    assert delta.status == "changed"
    assert any("semantic weld changed: size 1/8 → 3/16" in reason for reason in delta.reasons)
    assert any("meaningful drawing requirement change" in reason for reason in delta.reasons)


def test_reconcile_semantic_integration_keeps_equivalent_surface_finish_callout_unchanged():
    anchor = _anchor("Ra 3.2 um")
    candidate_span = _span("3.2 Ra", block_id=6, line_id=1, span_id=2, x0=12.0, y0=11.5)

    candidate, delta, anchor_semantic, matched_semantic = _classify(anchor, candidate_span)

    assert anchor_semantic.status.parser_family == "surface_finish"
    assert matched_semantic.status.parser_family == "surface_finish"
    assert anchor_semantic.surface_finish is not None
    assert matched_semantic.surface_finish is not None
    assert anchor_semantic.surface_finish.canonical_text == "Ra 3.2 um"
    assert matched_semantic.surface_finish.canonical_text == "Ra 3.2 um"
    assert any("semantic surface finish match: Ra 3.2 um" in reason for reason in candidate.reasons)
    assert delta.status == "unchanged"
    assert any("semantic surface finish match: Ra 3.2 um" in reason for reason in delta.reasons)
    assert any("Semantic family agreement" in reason for reason in delta.reasons)


def test_reconcile_semantic_integration_marks_changed_fit_semantic_delta():
    anchor = _anchor("H7/p6")
    candidate_span = _span("H7/g6", block_id=8, line_id=0, span_id=3, x0=14.0, y0=10.5)

    candidate, delta, anchor_semantic, matched_semantic = _classify(anchor, candidate_span)

    assert anchor_semantic.status.parser_family == "fit"
    assert matched_semantic.status.parser_family == "fit"
    assert anchor_semantic.fit is not None
    assert matched_semantic.fit is not None
    assert any("semantic fit changed: fit class H7/p6 → H7/g6" in reason for reason in candidate.reasons)
    assert delta.status == "changed"
    assert any("semantic fit changed: fit class H7/p6 → H7/g6" in reason for reason in delta.reasons)
    assert any("meaningful drawing requirement change" in reason for reason in delta.reasons)


def test_reconcile_semantic_integration_fallback_to_numeric_reasoning_when_semantics_unparsed():
    anchor = _anchor("12.0 ± 0.1")
    candidate_span = _span("12.0", block_id=2, line_id=0, span_id=0, x0=11.0, y0=10.5)

    candidate, delta, anchor_semantic, matched_semantic = _classify(anchor, candidate_span)

    assert anchor_semantic.status.state == "empty"
    assert anchor_semantic.status.reason_code == "surface_finish_no_match"
    assert matched_semantic.status.state == "empty"
    assert matched_semantic.status.reason_code == "surface_finish_no_match"
    assert any("semantic comparison fallback" in reason for reason in candidate.reasons)
    assert delta.status == "unchanged"
    assert any("semantic comparison fallback" in reason for reason in delta.reasons)
    assert any("Primary dimension matches" in reason for reason in delta.reasons)
