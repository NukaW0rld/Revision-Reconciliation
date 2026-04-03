import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from delta_preservation.cli import _format_annotation_text, run_pipeline
from delta_preservation.io.pdf import TextSpan


class _FakeTransform:
    def __init__(self):
        import numpy as np
        self.inliers = 8
        self.inlier_ratio = 1.0
        self.H = np.eye(3, dtype=float)


class _FakeDoc:
    def __init__(self):
        self._page = SimpleNamespace(rect=SimpleNamespace(width=100.0, height=100.0))

    def load_page(self, index):
        return self._page

    def close(self):
        pass


class _FakeAnchor:
    def __init__(self, char_no: int, requirement_raw: str):
        self.char_no = char_no
        self.page = 0
        self.balloon_bbox = (0.0, 0.0, 5.0, 5.0)
        self.req_bbox = (10.0, 10.0, 30.0, 18.0)
        self.requirement_raw = requirement_raw
        self.requirement_norm = requirement_raw
        self.local_context = []
        self.zone_bbox = None


class _FakeCandidate:
    def __init__(self, span):
        self.span = span


class _FakeMatch:
    def __init__(self, span):
        self.char_no = 1
        self.candidate = _FakeCandidate(span)


class _FakeInternalDeltaItem:
    def __init__(self, *, char_no, status, confidence, reasons, component_scores, match=None, added_span=None):
        self.char_no = char_no
        self.status = status
        self.confidence = confidence
        self.reasons = reasons
        self.component_scores = component_scores
        self.match = match
        self.added_span = added_span


def _span(text: str, *, x0: float, y0: float, width: float = 20.0, height: float = 8.0, block_id: int = 0, line_id: int = 0, span_id: int = 0):
    return TextSpan(
        text=text,
        bbox_pdf=(x0, y0, x0 + width, y0 + height),
        font_size=10.0,
        block_id=block_id,
        line_id=line_id,
        span_id=span_id,
    )


def test_format_annotation_text_moves_depth_symbol_before_value():
    spans = [
        _span("0.50", x0=20.0, y0=10.0, block_id=1, span_id=0),
        _span("↧", x0=10.0, y0=10.0, width=6.0, block_id=1, span_id=1),
    ]

    assert _format_annotation_text(spans) == "↧ 0.50"


def test_format_annotation_text_renders_vertical_diameter_limits_compactly():
    spans = [
        _span("Ø", x0=10.0, y0=10.0, width=6.0, block_id=1, span_id=0),
        _span("35.2", x0=18.0, y0=8.0, block_id=2, span_id=0),
        _span("34.8", x0=18.0, y0=18.0, block_id=3, span_id=0),
    ]

    assert _format_annotation_text(spans) == "Ø35.2 / Ø34.8"


def test_run_pipeline_omits_revA_evidence_for_added_characteristics(tmp_path):
    revA = tmp_path / "revA.pdf"
    revB = tmp_path / "revB.pdf"
    form3 = tmp_path / "form3.xlsx"
    revA.write_bytes(b"%PDF-1.4")
    revB.write_bytes(b"%PDF-1.4")
    form3.write_bytes(b"PK")

    anchor = _FakeAnchor(char_no=3, requirement_raw="LENGTH 12.0")
    matched_span = _span("12.0", x0=20.0, y0=14.0, block_id=1, span_id=0)
    added_span = _span("NEW DIM", x0=60.0, y0=48.0, block_id=9, line_id=4, span_id=2)

    classified_item = _FakeInternalDeltaItem(
        char_no=3,
        status="unchanged",
        confidence=0.93,
        reasons=["Stable requirement"],
        component_scores={"location": 0.95, "text": 0.9, "context": 0.89},
        match=_FakeMatch(matched_span),
    )
    added_item = _FakeInternalDeltaItem(
        char_no=4,
        status="added",
        confidence=0.61,
        reasons=["New requirement detected in Rev B"],
        component_scores={"location": 0.0, "text": 1.0, "context": 0.0},
        added_span=added_span,
    )

    render_stub = np.zeros((40, 40, 3), dtype=np.uint8)

    def fake_extract_text_spans(pdf_path, page_index=0):
        name = Path(pdf_path).name
        if name == "revA.pdf":
            return [_span("LENGTH 12.0", x0=10.0, y0=10.0, block_id=0, span_id=0)]
        return [matched_span, added_span]

    with patch("delta_preservation.cli.load_form3", return_value=[SimpleNamespace(char_no=3, requirement="LENGTH 12.0")]), \
         patch("delta_preservation.cli.detect_balloons", return_value=[{"char_no": 3}]), \
         patch("delta_preservation.cli.extract_text_spans", side_effect=fake_extract_text_spans), \
         patch("delta_preservation.cli.build_revA_anchors", return_value=[anchor]), \
         patch("delta_preservation.cli.render_page", return_value=render_stub), \
         patch("delta_preservation.cli.estimate_transform", return_value=_FakeTransform()), \
         patch("delta_preservation.cli.estimate_transform_from_text_spans", return_value=None), \
         patch("delta_preservation.cli.estimate_transform_candidates_from_text_spans", return_value=[]), \
         patch("delta_preservation.cli.generate_candidates", return_value=[_FakeCandidate(matched_span)]), \
         patch("delta_preservation.cli.assign_matches", return_value={3: _FakeMatch(matched_span)}), \
         patch("delta_preservation.cli.extract_tolerances_for_items", return_value={}), \
         patch("delta_preservation.cli.classify_delta", return_value=classified_item), \
         patch("delta_preservation.cli.detect_added_characteristics", return_value=[added_item]), \
         patch("delta_preservation.cli.export_run_tolerance_debug"), \
         patch("delta_preservation.cli.pdf_to_img_coords", return_value=(0, 0, 10, 10)), \
         patch("delta_preservation.cli.crop_with_padding", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
         patch("delta_preservation.cli.save_snippet", side_effect=["a.png", "b.png", "c.png"]), \
         patch("delta_preservation.cli.fitz.open", side_effect=[_FakeDoc(), _FakeDoc()]):
        run_dir = run_pipeline(
            revA_pdf=str(revA),
            revB_pdf=str(revB),
            form3_xlsx=str(form3),
            out_dir=str(tmp_path / "out"),
            part_name="output-format-added",
        )

    packet = json.loads((run_dir / "delta_packet.json").read_text())
    added_row = next(item for item in packet["items"] if item["char_no"] == 4)
    assert added_row["status"] == "added"
    assert added_row.get("revA") is None
    assert added_row["revB"]["image_path"] == "snippets/c.png"


def test_run_pipeline_aggregates_full_notes_text_for_requirement_revb(tmp_path):
    revA = tmp_path / "revA.pdf"
    revB = tmp_path / "revB.pdf"
    form3 = tmp_path / "form3.xlsx"
    revA.write_bytes(b"%PDF-1.4")
    revB.write_bytes(b"%PDF-1.4")
    form3.write_bytes(b"PK")

    anchor = _FakeAnchor(char_no=11, requirement_raw="NOTES: SEE DRAWING")
    notes_header = _span("NOTES:", x0=10.0, y0=10.0, width=28.0, block_id=5, span_id=0)
    notes_line_1 = _span("1. FIRST NOTE", x0=12.0, y0=22.0, width=80.0, block_id=6, span_id=0)
    notes_line_2 = _span("2. SECOND NOTE", x0=12.0, y0=34.0, width=90.0, block_id=7, span_id=0)

    classified_item = _FakeInternalDeltaItem(
        char_no=11,
        status="changed",
        confidence=0.8,
        reasons=["Notes changed"],
        component_scores={"location": 0.8, "text": 0.7, "context": 0.6},
        match=_FakeMatch(notes_header),
    )

    render_stub = np.zeros((40, 40, 3), dtype=np.uint8)

    def fake_extract_text_spans(pdf_path, page_index=0):
        name = Path(pdf_path).name
        if name == "revA.pdf":
            return [_span("NOTES:", x0=10.0, y0=10.0, width=28.0, block_id=0, span_id=0)]
        return [notes_header, notes_line_1, notes_line_2]

    with patch("delta_preservation.cli.load_form3", return_value=[SimpleNamespace(char_no=11, requirement="NOTES: SEE DRAWING")]), \
         patch("delta_preservation.cli.detect_balloons", return_value=[{"char_no": 11}]), \
         patch("delta_preservation.cli.extract_text_spans", side_effect=fake_extract_text_spans), \
         patch("delta_preservation.cli.build_revA_anchors", return_value=[anchor]), \
         patch("delta_preservation.cli.render_page", return_value=render_stub), \
         patch("delta_preservation.cli.estimate_transform", return_value=_FakeTransform()), \
         patch("delta_preservation.cli.estimate_transform_from_text_spans", return_value=None), \
         patch("delta_preservation.cli.estimate_transform_candidates_from_text_spans", return_value=[]), \
         patch("delta_preservation.cli.generate_candidates", return_value=[_FakeCandidate(notes_header)]), \
         patch("delta_preservation.cli.assign_matches", return_value={11: _FakeMatch(notes_header)}), \
         patch("delta_preservation.cli.extract_tolerances_for_items", return_value={}), \
         patch("delta_preservation.cli.classify_delta", return_value=classified_item), \
         patch("delta_preservation.cli.detect_added_characteristics", return_value=[]), \
         patch("delta_preservation.cli.export_run_tolerance_debug"), \
         patch("delta_preservation.cli.expand_notes_block", return_value=(10.0, 10.0, 120.0, 46.0)), \
         patch("delta_preservation.cli.pdf_to_img_coords", return_value=(0, 0, 10, 10)), \
         patch("delta_preservation.cli.crop_with_padding", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
         patch("delta_preservation.cli.save_snippet", side_effect=["a.png", "b.png"]), \
         patch("delta_preservation.cli.fitz.open", side_effect=[_FakeDoc(), _FakeDoc()]):
        run_dir = run_pipeline(
            revA_pdf=str(revA),
            revB_pdf=str(revB),
            form3_xlsx=str(form3),
            out_dir=str(tmp_path / "out"),
            part_name="output-format-notes",
        )

    packet = json.loads((run_dir / "delta_packet.json").read_text())
    row = packet["items"][0]
    assert row["requirement_revB"] == "NOTES: 1. FIRST NOTE 2. SECOND NOTE"
