import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from delta_preservation.io.pdf import TextSpan


class _FakeDoc:
    def __init__(self, width: float = 612.0, height: float = 792.0):
        self._page = SimpleNamespace(rect=SimpleNamespace(width=width, height=height))

    def load_page(self, index: int):
        return self._page

    def close(self):
        return None


class _FakeAnchor:
    def __init__(self, char_no: int, requirement_raw: str, req_bbox=(10.0, 10.0, 30.0, 20.0)):
        self.char_no = char_no
        self.requirement_raw = requirement_raw
        self.req_bbox = req_bbox
        self.balloon_bbox = (5.0, 5.0, 15.0, 15.0)
        self.page = 0


class _FakeCandidate:
    def __init__(self, span: TextSpan):
        self.span = span
        self.location_score = 0.91
        self.text_score = 0.87
        self.context_score = 0.83


class _FakeMatch:
    def __init__(self, span: TextSpan):
        self.candidate = _FakeCandidate(span)


class _FakeInternalDeltaItem:
    def __init__(self, *, char_no: int, status: str, confidence: float, reasons, component_scores, match=None, added_span=None):
        self.char_no = char_no
        self.status = status
        self.confidence = confidence
        self.reasons = reasons
        self.component_scores = component_scores
        self.match = match
        self.added_span = added_span


class _FakeTransform:
    def __init__(self):
        self.inliers = 12
        self.inlier_ratio = 0.9
        self.H = np.eye(3)


def _span(text: str, *, block_id: int, line_id: int, span_id: int, x0: float, y0: float) -> TextSpan:
    return TextSpan(
        text=text,
        bbox_pdf=(x0, y0, x0 + 12.0, y0 + 6.0),
        font_size=10.0,
        block_id=block_id,
        line_id=line_id,
        span_id=span_id,
    )


def test_run_pipeline_persists_semantic_callouts_in_delta_packet(tmp_path):
    from delta_preservation.cli import run_pipeline

    revA = tmp_path / "revA.pdf"
    revB = tmp_path / "revB.pdf"
    form3 = tmp_path / "form3.xlsx"
    revA.write_bytes(b"%PDF-1.4")
    revB.write_bytes(b"%PDF-1.4")
    form3.write_bytes(b"PK")

    anchor = _FakeAnchor(char_no=7, requirement_raw="SURFACE FINISH 63 MICROINCH")
    revb_semantic_span = _span("⟂ 0.05 A B", block_id=4, line_id=2, span_id=1, x0=40.0, y0=22.0)

    classified_item = _FakeInternalDeltaItem(
        char_no=7,
        status="changed",
        confidence=0.88,
        reasons=["Matched span text changed"],
        component_scores={"location": 0.91, "text": 0.87, "context": 0.83},
        match=_FakeMatch(revb_semantic_span),
    )

    render_stub = np.zeros((40, 40, 3), dtype=np.uint8)

    def fake_extract_text_spans(pdf_path, page_index=0):
        name = Path(pdf_path).name
        if name == "revA.pdf":
            return [_span("FORM3 ANCHOR", block_id=0, line_id=0, span_id=0, x0=12.0, y0=12.0)]
        return [revb_semantic_span]

    with patch("delta_preservation.cli.load_form3", return_value=[SimpleNamespace(char_no=7, requirement="SURFACE FINISH 63 MICROINCH")]), \
         patch("delta_preservation.cli.detect_balloons", return_value=[{"char_no": 7}]), \
         patch("delta_preservation.cli.extract_text_spans", side_effect=fake_extract_text_spans), \
         patch("delta_preservation.cli.build_revA_anchors", return_value=[anchor]), \
         patch("delta_preservation.cli.render_page", return_value=render_stub), \
         patch("delta_preservation.cli.estimate_transform", return_value=_FakeTransform()), \
         patch("delta_preservation.cli.estimate_transform_from_text_spans", return_value=None), \
         patch("delta_preservation.cli.generate_candidates", return_value=[_FakeCandidate(revb_semantic_span)]), \
         patch("delta_preservation.cli.assign_matches", return_value={7: _FakeMatch(revb_semantic_span)}), \
         patch("delta_preservation.cli.extract_tolerances_for_items", return_value={}), \
         patch("delta_preservation.cli.classify_delta", return_value=classified_item), \
         patch("delta_preservation.cli.detect_added_characteristics", return_value=[]), \
         patch("delta_preservation.cli.export_run_tolerance_debug"), \
         patch("delta_preservation.cli.pdf_to_img_coords", return_value=(0, 0, 10, 10)), \
         patch("delta_preservation.cli.crop_with_padding", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
         patch("delta_preservation.cli.save_snippet", side_effect=["a.png", "b.png"]), \
         patch("delta_preservation.cli.fitz.open", side_effect=[_FakeDoc(), _FakeDoc()]):
        run_dir = run_pipeline(
            revA_pdf=str(revA),
            revB_pdf=str(revB),
            form3_xlsx=str(form3),
            out_dir=str(tmp_path / "out"),
            part_name="semantic-packet",
        )

    packet = json.loads((run_dir / "delta_packet.json").read_text())
    item = packet["items"][0]
    semantic = item["semantic_callout"]

    assert item["char_no"] == 7
    assert item["status"] == "changed"
    assert semantic["provenance"]["authority"] == "pdf"
    assert semantic["provenance"]["source_type"] == "drawing_pdf"
    assert semantic["provenance"]["source_ref"] == "pdf:block:4/line:2/span:1"
    assert semantic["status"]["state"] == "not_implemented"
    assert semantic["status"]["reason_code"] == "not_implemented_in_slice"
    assert semantic["metadata"]["dispatcher"] == "semantic_dispatch"
    assert semantic["metadata"]["authority_source"] == "pdf"
    assert semantic["metadata"]["planned_families"] == "gdt,weld,surface_finish,fit"
    assert semantic["raw_text"] == "⟂ 0.05 A B"
    assert "gdt" in semantic
    assert "weld" in semantic
    assert "surface_finish" in semantic
    assert "fit" in semantic


def test_run_pipeline_uses_pdf_span_inputs_for_matched_and_added_semantics(tmp_path):
    from delta_preservation.cli import run_pipeline
    from delta_preservation.types import SemanticCallout, SemanticParserStatus, SemanticProvenance

    revA = tmp_path / "revA.pdf"
    revB = tmp_path / "revB.pdf"
    form3 = tmp_path / "form3.xlsx"
    revA.write_bytes(b"%PDF-1.4")
    revB.write_bytes(b"%PDF-1.4")
    form3.write_bytes(b"PK")

    anchor = _FakeAnchor(char_no=3, requirement_raw="LENGTH 12.0")
    matched_span = _span("12.0", block_id=1, line_id=0, span_id=0, x0=20.0, y0=14.0)
    added_span = _span("NEW DIM", block_id=9, line_id=4, span_id=2, x0=60.0, y0=48.0)

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
            return [_span("LENGTH 12.0", block_id=0, line_id=0, span_id=0, x0=10.0, y0=10.0)]
        return [matched_span, added_span]

    semantic_inputs = []

    def fake_extract_semantic_callout(pdf_spans, form3_requirement=None):
        semantic_inputs.append(
            {
                "pdf_texts": [span.text for span in pdf_spans],
                "form3_requirement": form3_requirement,
            }
        )
        authority = "pdf" if pdf_spans else "form3"
        source_type = "drawing_pdf" if pdf_spans else "form3_requirement"
        source_ref = (
            f"pdf:block:{pdf_spans[0].block_id}/line:{pdf_spans[0].line_id}/span:{pdf_spans[0].span_id}"
            if pdf_spans
            else "form3:requirement"
        )
        raw_text = pdf_spans[0].text if pdf_spans else form3_requirement
        return SemanticCallout(
            provenance=SemanticProvenance(
                authority=authority,
                source_type=source_type,
                source_ref=source_ref,
                notes=["stub semantic extraction for packet propagation test"],
            ),
            status=SemanticParserStatus(
                state="not_implemented",
                parser_family="semantic_dispatch",
                reason_code="not_implemented_in_slice",
                detail="stub",
            ),
            raw_text=raw_text,
            normalized_text=raw_text,
            metadata={"authority_source": authority},
        )

    with patch("delta_preservation.cli.load_form3", return_value=[SimpleNamespace(char_no=3, requirement="LENGTH 12.0")]), \
         patch("delta_preservation.cli.detect_balloons", return_value=[{"char_no": 3}]), \
         patch("delta_preservation.cli.extract_text_spans", side_effect=fake_extract_text_spans), \
         patch("delta_preservation.cli.build_revA_anchors", return_value=[anchor]), \
         patch("delta_preservation.cli.render_page", return_value=render_stub), \
         patch("delta_preservation.cli.estimate_transform", return_value=_FakeTransform()), \
         patch("delta_preservation.cli.estimate_transform_from_text_spans", return_value=None), \
         patch("delta_preservation.cli.generate_candidates", return_value=[_FakeCandidate(matched_span)]), \
         patch("delta_preservation.cli.assign_matches", return_value={3: _FakeMatch(matched_span)}), \
         patch("delta_preservation.cli.extract_tolerances_for_items", return_value={}), \
         patch("delta_preservation.cli.classify_delta", return_value=classified_item), \
         patch("delta_preservation.cli.detect_added_characteristics", return_value=[added_item]), \
         patch("delta_preservation.cli.export_run_tolerance_debug"), \
         patch("delta_preservation.cli.pdf_to_img_coords", return_value=(0, 0, 10, 10)), \
         patch("delta_preservation.cli.crop_with_padding", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
         patch("delta_preservation.cli.save_snippet", side_effect=["a.png", "b.png", "c.png", "d.png"]), \
         patch("delta_preservation.cli.extract_semantic_callout", side_effect=fake_extract_semantic_callout), \
         patch("delta_preservation.cli.fitz.open", side_effect=[_FakeDoc(), _FakeDoc()]):
        run_dir = run_pipeline(
            revA_pdf=str(revA),
            revB_pdf=str(revB),
            form3_xlsx=str(form3),
            out_dir=str(tmp_path / "out"),
            part_name="semantic-packet-added",
        )

    packet = json.loads((run_dir / "delta_packet.json").read_text())
    assert [item["char_no"] for item in packet["items"]] == [3, 4]
    assert semantic_inputs == [
        {"pdf_texts": ["12.0"], "form3_requirement": "LENGTH 12.0"},
        {"pdf_texts": ["NEW DIM"], "form3_requirement": None},
    ]
    assert packet["items"][0]["semantic_callout"]["provenance"]["authority"] == "pdf"
    assert packet["items"][1]["semantic_callout"]["provenance"]["authority"] == "pdf"
    assert packet["items"][1]["semantic_callout"]["status"]["reason_code"] == "not_implemented_in_slice"
