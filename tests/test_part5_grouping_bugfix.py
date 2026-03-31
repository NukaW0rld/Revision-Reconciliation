from delta_preservation.io.pdf import TextSpan
from delta_preservation.reconcile.anchors import Anchor
from delta_preservation.reconcile.classify import classify_delta, detect_added_characteristics
from delta_preservation.reconcile.match import Match, generate_candidates
from delta_preservation.reconcile.normalize import extract_semantic_callout


class _Transform:
    def __init__(self):
        import numpy as np
        self.H = np.eye(3, dtype=float)
        self.inliers = 8
        self.inlier_ratio = 1.0


def _span(text: str, *, block_id: int, line_id: int, span_id: int, x0: float, y0: float, width: float = 20.0, height: float = 8.0) -> TextSpan:
    return TextSpan(
        text=text,
        bbox_pdf=(x0, y0, x0 + width, y0 + height),
        font_size=10.0,
        block_id=block_id,
        line_id=line_id,
        span_id=span_id,
    )


def _anchor(requirement_raw: str, char_no: int = 1, x0: float = 10.0, y0: float = 10.0) -> Anchor:
    return Anchor(
        char_no=char_no,
        page=0,
        balloon_bbox=(0.0, 0.0, 5.0, 5.0),
        req_bbox=(x0, y0, x0 + 20.0, y0 + 8.0),
        requirement_raw=requirement_raw,
        requirement_norm=requirement_raw,
        local_context=[],
    )


def _span_key(span: TextSpan):
    return (span.block_id, span.line_id, span.span_id, span.bbox_pdf)


def test_generate_candidates_groups_split_limit_callout_before_matching():
    anchor = _anchor("Ø35 +0.2/-0.2", char_no=2)
    spans = [
        _span("Ø", block_id=10, line_id=0, span_id=0, x0=10, y0=10, width=6),
        _span("35.2", block_id=10, line_id=0, span_id=1, x0=17, y0=8, width=20),
        _span("34.8", block_id=11, line_id=0, span_id=0, x0=17, y0=18, width=20),
    ]

    semantic_by_key = {_span_key(s): extract_semantic_callout(pdf_spans=[s]) for s in spans}
    anchor_semantic = extract_semantic_callout(pdf_spans=[], form3_requirement=anchor.requirement_raw)
    candidates = generate_candidates(
        anchor,
        spans,
        _Transform(),
        anchor_semantic_callout=anchor_semantic,
        revB_semantic_callouts_by_span_key=semantic_by_key,
    )

    assert candidates, "expected grouped candidate"
    best = candidates[0]
    assert "35.2" in best.span.text
    assert "34.8" in best.span.text
    assert "Ø" in best.span.text

    delta = classify_delta(anchor, Match(char_no=2, candidate=best))
    assert delta.status == "unchanged"
    assert delta.confidence > 0.0


def test_generate_candidates_groups_gdt_symbol_with_numeric_value():
    anchor = _anchor("⟂1.5A", char_no=12)
    spans = [
        _span("⟂", block_id=40, line_id=0, span_id=0, x0=10, y0=10, width=8),
        _span("1.5", block_id=40, line_id=0, span_id=1, x0=20, y0=10, width=16),
        _span("A", block_id=40, line_id=0, span_id=2, x0=38, y0=10, width=8),
    ]
    semantic_by_key = {_span_key(s): extract_semantic_callout(pdf_spans=[s]) for s in spans}
    anchor_semantic = extract_semantic_callout(pdf_spans=[], form3_requirement=anchor.requirement_raw)
    candidates = generate_candidates(anchor, spans, _Transform(), anchor_semantic_callout=anchor_semantic, revB_semantic_callouts_by_span_key=semantic_by_key)
    assert candidates
    assert "⟂" in candidates[0].span.text
    assert "1.5" in candidates[0].span.text
    assert "A" in candidates[0].span.text


def test_assign_matches_does_not_reuse_same_grouped_source_spans_for_second_anchor():
    anchor1 = _anchor("Ø35 +0.2/-0.2", char_no=1)
    anchor2 = _anchor("Ø35 +0/-0.2", char_no=4)
    spans = [
        _span("Ø", block_id=10, line_id=0, span_id=0, x0=10, y0=10, width=6),
        _span("35.2", block_id=10, line_id=0, span_id=1, x0=17, y0=8, width=20),
        _span("34.8", block_id=11, line_id=0, span_id=0, x0=17, y0=18, width=20),
    ]
    semantic_by_key = {_span_key(s): extract_semantic_callout(pdf_spans=[s]) for s in spans}
    candidates1 = generate_candidates(anchor1, spans, _Transform(), anchor_semantic_callout=extract_semantic_callout(pdf_spans=[], form3_requirement=anchor1.requirement_raw), revB_semantic_callouts_by_span_key=semantic_by_key)
    candidates2 = generate_candidates(anchor2, spans, _Transform(), anchor_semantic_callout=extract_semantic_callout(pdf_spans=[], form3_requirement=anchor2.requirement_raw), revB_semantic_callouts_by_span_key=semantic_by_key)

    from delta_preservation.reconcile.match import assign_matches
    matches = assign_matches([anchor1, anchor2], {1: candidates1, 4: candidates2})
    assert 1 in matches
    assert 4 not in matches


def test_detect_added_characteristics_does_not_readd_grouped_match_companions():
    matched_group = _span("Ø 35.2 34.8", block_id=20, line_id=0, span_id=0, x0=100, y0=100, width=45)
    companion_spans = [
        _span("Ø", block_id=21, line_id=0, span_id=0, x0=100, y0=100, width=6),
        _span("35.2", block_id=21, line_id=0, span_id=1, x0=107, y0=98, width=20),
        _span("34.8", block_id=22, line_id=0, span_id=0, x0=107, y0=108, width=20),
    ]
    revb_spans = [matched_group, *companion_spans]
    matches = {1: Match(char_no=1, candidate=type('C', (), {'span': matched_group})())}

    added = detect_added_characteristics(revb_spans, matches, next_char_no=2)
    assert added == []


def test_generate_candidates_groups_diameter_with_split_tolerances_for_moved_annotation():
    anchor = _anchor("Ø20 +0.05/-0.1", char_no=7)
    spans = [
        _span("Ø20", block_id=30, line_id=0, span_id=0, x0=12, y0=10, width=18),
        _span("+0.05", block_id=30, line_id=0, span_id=1, x0=31, y0=8, width=24),
        _span("-0.1", block_id=31, line_id=0, span_id=0, x0=31, y0=18, width=22),
    ]
    semantic_by_key = {_span_key(s): extract_semantic_callout(pdf_spans=[s]) for s in spans}
    anchor_semantic = extract_semantic_callout(pdf_spans=[], form3_requirement=anchor.requirement_raw)
    candidates = generate_candidates(
        anchor,
        spans,
        _Transform(),
        anchor_semantic_callout=anchor_semantic,
        revB_semantic_callouts_by_span_key=semantic_by_key,
    )

    assert candidates
    best = candidates[0]
    assert "Ø20" in best.span.text
    assert "+0.05" in best.span.text
    assert "-0.1" in best.span.text
