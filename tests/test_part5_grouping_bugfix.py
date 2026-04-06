from delta_preservation.io.pdf import TextSpan
from delta_preservation.reconcile.anchors import Anchor
from delta_preservation.reconcile.classify import classify_delta, detect_added_characteristics
from delta_preservation.reconcile.match import Candidate, Match, assign_matches, generate_candidates
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


def _candidate(span: TextSpan, *, total_score: float, location_score: float, source_span_keys: list[tuple] | None = None,
               from_global_fallback: bool = False) -> Candidate:
    return Candidate(
        span=span,
        total_score=total_score,
        location_score=location_score,
        text_score=0.5,
        context_score=0.0,
        reasons=[
            f"location_score={location_score:.3f}",
            f"from_global_fallback={from_global_fallback}",
        ],
        from_global_fallback=from_global_fallback,
        source_span_keys=source_span_keys or [_span_key(span)],
    )


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


def test_assign_matches_prefers_stronger_local_owner_for_same_grouped_source_span_set():
    owner_anchor = _anchor("Ø35 +0.2/-0.2", char_no=1, x0=480.0, y0=72.0)
    removed_anchor = _anchor("Ø35 +0/-0.2", char_no=4, x0=350.0, y0=320.0)
    grouped_span = _span("35.2 Ø 34.8", block_id=45, line_id=0, span_id=0, x0=490.0, y0=74.0, width=45)
    grouped_source_span_keys = [
        (45, 0, 0, (490.0, 74.0, 535.0, 82.0)),
        (45, 1, 0, (498.0, 64.0, 520.0, 72.0)),
        (46, 0, 0, (498.0, 84.0, 520.0, 92.0)),
    ]

    stronger_local_owner = _candidate(
        grouped_span,
        total_score=0.58,
        location_score=0.94,
        source_span_keys=grouped_source_span_keys,
    )
    weaker_but_higher_total = _candidate(
        grouped_span,
        total_score=0.64,
        location_score=0.41,
        source_span_keys=grouped_source_span_keys,
    )

    matches = assign_matches(
        [owner_anchor, removed_anchor],
        {
            1: [stronger_local_owner],
            4: [weaker_but_higher_total],
        },
    )

    assert 1 in matches, (
        "Expected owning char_no=1 to keep grouped span via stronger local ownership; "
        f"got match keys={set(matches)}"
    )
    assert matches[1].candidate is stronger_local_owner, (
        "Expected grouped span owner char_no=1 to win same-source-span collision; "
        f"got candidate reasons={matches[1].candidate.reasons}"
    )
    assert 4 not in matches, (
        "Expected removed competitor char_no=4 to remain unmatched on grouped source-span collision; "
        f"got match keys={set(matches)}, owner_reasons={stronger_local_owner.reasons}, "
        f"competitor_reasons={weaker_but_higher_total.reasons}"
    )



def test_same_source_span_tiebreak_does_not_break_true_combined_annotation_sharing():
    depth_anchor = _anchor("10", char_no=21, x0=100.0, y0=100.0)
    angle_anchor = _anchor("90°", char_no=22, x0=180.0, y0=180.0)
    combined_span = _span("10 x 90°", block_id=60, line_id=0, span_id=0, x0=102.0, y0=102.0, width=38)
    combined_source_span_keys = [
        (60, 0, 0, (102.0, 102.0, 140.0, 110.0)),
        (60, 0, 1, (142.0, 102.0, 154.0, 110.0)),
    ]

    better_local_owner = _candidate(
        combined_span,
        total_score=0.57,
        location_score=0.92,
        source_span_keys=combined_source_span_keys,
    )
    second_characteristic = _candidate(
        combined_span,
        total_score=0.61,
        location_score=0.35,
        source_span_keys=combined_source_span_keys,
    )

    matches = assign_matches(
        [depth_anchor, angle_anchor],
        {
            21: [better_local_owner],
            22: [second_characteristic],
        },
    )

    assert 21 in matches, "Expected first characteristic to claim combined annotation in greedy pass"
    assert 22 in matches, (
        "Expected shared-span fallback to keep true combined annotation reusable after same-source-span tie-break; "
        f"got match keys={set(matches)}"
    )
    assert matches[22].candidate is second_characteristic
