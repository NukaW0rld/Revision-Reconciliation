from delta_preservation.io.pdf import TextSpan
from delta_preservation.reconcile.anchors import Anchor
from delta_preservation.reconcile.match import generate_candidates
from delta_preservation.vision.alignment import (
    Transform,
    estimate_transform_candidates_from_text_spans,
)


def _span(
    text: str,
    *,
    x0: float,
    y0: float,
    width: float = 24.0,
    height: float = 8.0,
    block_id: int = 0,
    line_id: int = 0,
    span_id: int = 0,
) -> TextSpan:
    return TextSpan(
        text=text,
        bbox_pdf=(x0, y0, x0 + width, y0 + height),
        font_size=10.0,
        block_id=block_id,
        line_id=line_id,
        span_id=span_id,
    )


def _anchor(requirement_raw: str, *, char_no: int = 1, x0: float = 10.0, y0: float = 10.0) -> Anchor:
    return Anchor(
        char_no=char_no,
        page=0,
        balloon_bbox=(0.0, 0.0, 5.0, 5.0),
        req_bbox=(x0, y0, x0 + 24.0, y0 + 8.0),
        requirement_raw=requirement_raw,
        requirement_norm=requirement_raw,
        local_context=[],
    )


def _transform(tx: float, ty: float = 0.0) -> Transform:
    import numpy as np

    return Transform(
        H=np.array([
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ], dtype=float),
        inliers=3,
        inlier_ratio=1.0,
        quality_ok=True,
    )


def test_estimate_transform_candidates_from_text_spans_returns_secondary_shift_group():
    revA_spans = [
        _span("Ø10", x0=10, y0=10, block_id=1, span_id=0),
        _span("Ø12", x0=30, y0=20, block_id=2, span_id=0),
        _span("Ø14", x0=50, y0=30, block_id=3, span_id=0),
        _span("Ø16", x0=70, y0=40, block_id=4, span_id=0),
        _span("Ø18", x0=90, y0=50, block_id=5, span_id=0),
    ]
    revB_spans = [
        _span("Ø10", x0=110, y0=10, block_id=11, span_id=0),
        _span("Ø12", x0=130, y0=20, block_id=12, span_id=0),
        _span("Ø14", x0=150, y0=30, block_id=13, span_id=0),
        _span("Ø16", x0=290, y0=40, block_id=14, span_id=0),
        _span("Ø18", x0=310, y0=50, block_id=15, span_id=0),
    ]

    transforms = estimate_transform_candidates_from_text_spans(revA_spans, revB_spans)

    assert len(transforms) >= 2
    txs = [transform.H[0, 2] for transform in transforms]
    assert any(abs(tx - 100.0) <= 1.0 for tx in txs)
    assert any(abs(tx - 220.0) <= 1.0 for tx in txs)


def test_generate_candidates_uses_alternate_transform_prediction():
    anchor = _anchor("Ø10", x0=10.0, y0=10.0)
    distant_match = _span("Ø10", x0=370.0, y0=10.0, block_id=20, span_id=0)

    candidates = generate_candidates(
        anchor,
        [distant_match],
        [_transform(0.0), _transform(360.0)],
    )

    assert candidates, "expected alternate transform to pull the candidate into range"
    assert candidates[0].span.text == "Ø10"


def test_generate_candidates_excludes_tolerance_block_boilerplate():
    anchor = _anchor("R.110 +/- 0.3 in", x0=10.0, y0=10.0)
    tolerance_block_span = _span(
        "ANGULAR = ±0.3 °",
        x0=10.0,
        y0=10.0,
        width=80.0,
        block_id=30,
        span_id=0,
    )

    candidates = generate_candidates(anchor, [tolerance_block_span], _transform(0.0))

    assert candidates == []
