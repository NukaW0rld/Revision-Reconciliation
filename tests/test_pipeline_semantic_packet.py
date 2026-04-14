import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from delta_preservation.evaluation.contracts import GroundTruthPacket
from delta_preservation.io.pdf import TextSpan
from delta_preservation.reconcile.match import GroupedSpan


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


class _PipelineTestCase:
    def __init__(
        self,
        *,
        char_no: int,
        form3_requirement: str,
        revb_text: str,
        expected_status: str,
        expected_reasons,
        expected_semantic_family: str,
        expected_detail: str,
        expected_family_payload: dict | None,
        expected_status_state: str = "parsed",
        expected_reason_code: str | None = None,
    ):
        self.char_no = char_no
        self.form3_requirement = form3_requirement
        self.revb_text = revb_text
        self.expected_status = expected_status
        self.expected_reasons = expected_reasons
        self.expected_semantic_family = expected_semantic_family
        self.expected_detail = expected_detail
        self.expected_family_payload = expected_family_payload
        self.expected_status_state = expected_status_state
        self.expected_reason_code = expected_reason_code


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


def _empty_truth_packet() -> GroundTruthPacket:
    return GroundTruthPacket.model_validate(
        {
            "part_name": "Semantic Packet Fixture",
            "general_notes": "",
            "characteristics": [],
        }
    )


def _run_pipeline_semantic_case(tmp_path, case: _PipelineTestCase):
    from delta_preservation.cli import run_pipeline

    revA = tmp_path / "revA.pdf"
    revB = tmp_path / "revB.pdf"
    form3 = tmp_path / "form3.xlsx"
    revA.write_bytes(b"%PDF-1.4")
    revB.write_bytes(b"%PDF-1.4")
    form3.write_bytes(b"PK")

    anchor = _FakeAnchor(char_no=case.char_no, requirement_raw=case.form3_requirement)
    revb_semantic_span = _span(case.revb_text, block_id=4, line_id=2, span_id=1, x0=40.0, y0=22.0)

    classified_item = _FakeInternalDeltaItem(
        char_no=case.char_no,
        status=case.expected_status,
        confidence=0.88,
        reasons=list(case.expected_reasons),
        component_scores={"location": 0.91, "text": 0.87, "context": 0.83},
        match=_FakeMatch(revb_semantic_span),
    )

    render_stub = np.zeros((40, 40, 3), dtype=np.uint8)

    def fake_extract_text_spans(pdf_path, page_index=0):
        name = Path(pdf_path).name
        if name == "revA.pdf":
            return [_span("FORM3 ANCHOR", block_id=0, line_id=0, span_id=0, x0=12.0, y0=12.0)]
        return [revb_semantic_span]

    with patch("delta_preservation.cli.load_form3", return_value=[SimpleNamespace(char_no=case.char_no, requirement=case.form3_requirement)]), \
         patch("delta_preservation.cli.detect_balloons", return_value=[{"char_no": case.char_no}]), \
         patch("delta_preservation.cli.extract_text_spans", side_effect=fake_extract_text_spans), \
         patch("delta_preservation.cli.build_revA_anchors", return_value=[anchor]), \
         patch("delta_preservation.cli.render_page", return_value=render_stub), \
         patch("delta_preservation.cli.estimate_transform", return_value=_FakeTransform()), \
         patch("delta_preservation.cli.estimate_transform_from_text_spans", return_value=None), \
         patch("delta_preservation.cli.generate_candidates", return_value=[_FakeCandidate(revb_semantic_span)]), \
         patch("delta_preservation.cli.assign_matches", return_value={case.char_no: _FakeMatch(revb_semantic_span)}), \
         patch("delta_preservation.cli.extract_tolerances_for_items", return_value={}), \
         patch("delta_preservation.cli.classify_delta", return_value=classified_item), \
         patch("delta_preservation.cli.detect_added_characteristics", return_value=[]), \
         patch("delta_preservation.cli.export_run_tolerance_debug"), \
         patch("delta_preservation.cli.pdf_to_img_coords", return_value=(0, 0, 10, 10)), \
         patch("delta_preservation.cli.crop_with_padding", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
         patch("delta_preservation.cli.save_snippet", side_effect=["a.png", "b.png"]), \
         patch("delta_preservation.cli.load_ground_truth_packet", return_value=_empty_truth_packet()), \
         patch("delta_preservation.cli.fitz.open", side_effect=[_FakeDoc(), _FakeDoc()]):
        run_dir = run_pipeline(
            revA_pdf=str(revA),
            revB_pdf=str(revB),
            form3_xlsx=str(form3),
            out_dir=str(tmp_path / "out"),
            part_name=f"semantic-packet-{case.char_no}",
        )

    packet = json.loads((run_dir / "delta_packet.json").read_text())
    item = packet["items"][0]
    semantic = item["semantic_callout"]
    return item, semantic


def test_run_pipeline_persists_semantic_callouts_in_delta_packet(tmp_path):
    case = _PipelineTestCase(
        char_no=7,
        form3_requirement="SURFACE FINISH 63 MICROINCH",
        revb_text="⌖ ⌀0.05 M A B",
        expected_status="changed",
        expected_reasons=["semantic GD&T changed: tolerance None → ⌀0.05"],
        expected_semantic_family="gdt",
        expected_detail="parsed feature control frame from PDF spans",
        expected_family_payload={
            "frame_text": "⌖ | ⌀0.05 | M | A | B",
            "control_type": "position",
            "tolerance_text": "⌀0.05",
            "datum_refs": ["A", "B"],
            "modifiers": ["MMC"],
        },
    )

    item, semantic = _run_pipeline_semantic_case(tmp_path, case)

    assert item["char_no"] == 7
    assert item["status"] == "changed"
    assert item["reasons"] == ["semantic GD&T changed: tolerance None → ⌀0.05"]
    assert semantic["provenance"]["authority"] == "pdf"
    assert semantic["provenance"]["source_type"] == "drawing_pdf"
    assert semantic["provenance"]["source_ref"] == "pdf:block:4/line:2/span:1"
    assert semantic["provenance"]["notes"] == [
        "PDF-derived spans are authoritative for semantic extraction.",
        "Form 3 requirement text was retained as secondary advisory context.",
        "PDF authority overrode conflicting Form 3 semantic wording.",
    ]
    assert semantic["status"] == {
        "state": "parsed",
        "parser_family": "gdt",
        "detail": "parsed feature control frame from PDF spans",
    }
    assert semantic["metadata"]["dispatcher"] == "semantic_dispatch"
    assert semantic["metadata"]["authority_source"] == "pdf"
    assert semantic["metadata"]["planned_families"] == "gdt,weld,surface_finish,fit"
    assert semantic["metadata"]["conflict_detected"] == "true"
    assert semantic["metadata"]["form3_context_supplied"] == "true"
    assert semantic["raw_text"] == "⌖ ⌀0.05 M A B"
    assert semantic["normalized_text"] == "⌖ ⌀0.05 M A B"
    assert semantic["gdt"] == {
        "frame_text": "⌖ | ⌀0.05 | M | A | B",
        "control_type": "position",
        "tolerance_text": "⌀0.05",
        "datum_refs": ["A", "B"],
        "modifiers": ["MMC"],
        "compartments": [],
    }
    assert "weld" not in semantic
    assert "surface_finish" not in semantic
    assert "fit" not in semantic


def test_run_pipeline_packet_surfaces_semantic_equivalence_status_and_reason(tmp_path):
    case = _PipelineTestCase(
        char_no=21,
        form3_requirement="⌖ ⌀0.10 M A B C",
        revb_text="⌖    ⌀0.10   M   A  B C",
        expected_status="unchanged",
        expected_reasons=[
            "semantic GD&T match: control type position, tolerance ⌀0.10, datums A/B/C, modifiers MMC",
            "Semantic family agreement overrides formatting-only numeric/text variation",
        ],
        expected_semantic_family="gdt",
        expected_detail="parsed feature control frame from PDF spans",
        expected_family_payload={
            "frame_text": "⌖ | ⌀0.10 | M | A | B | C",
            "control_type": "position",
            "tolerance_text": "⌀0.10",
            "datum_refs": ["A", "B", "C"],
            "modifiers": ["MMC"],
            "compartments": [],
        },
    )

    item, semantic = _run_pipeline_semantic_case(tmp_path, case)

    assert item["status"] == "unchanged"
    assert item["reasons"] == case.expected_reasons
    assert item["reasons"][0].startswith("semantic GD&T match")
    assert "formatting-only" in item["reasons"][1]
    assert semantic["status"]["state"] == case.expected_status_state
    assert semantic["status"]["parser_family"] == case.expected_semantic_family
    assert semantic["status"]["detail"] == case.expected_detail
    assert semantic["gdt"] == case.expected_family_payload
    assert semantic["provenance"]["authority"] == "pdf"
    assert semantic["metadata"]["form3_context_supplied"] == "true"


def test_run_pipeline_packet_surfaces_semantic_delta_status_and_reason(tmp_path):
    case = _PipelineTestCase(
        char_no=22,
        form3_requirement="1/8 FILLET BOTH SIDES ALL AROUND 1.50-3.00 FLUSH TAIL: FIELD",
        revb_text="3/16 FILLET BOTH SIDES ALL AROUND 1.50-3.00 FLUSH TAIL: FIELD",
        expected_status="changed",
        expected_reasons=[
            "semantic weld changed: size 1/8 → 3/16",
            "Semantic family delta indicates a meaningful drawing requirement change",
        ],
        expected_semantic_family="weld",
        expected_detail="parsed bounded weld callout from authoritative semantic text",
        expected_family_payload={
            "process": "fillet",
            "size": "3/16",
            "contour": "flush",
            "side": "both_sides",
            "length": "1.50",
            "pitch": "3.00",
            "tail": "FIELD",
            "all_around": True,
        },
    )

    item, semantic = _run_pipeline_semantic_case(tmp_path, case)

    assert item["status"] == "changed"
    assert item["reasons"] == case.expected_reasons
    assert item["reasons"][0] == "semantic weld changed: size 1/8 → 3/16"
    assert "meaningful drawing requirement change" in item["reasons"][1]
    assert semantic["status"]["state"] == case.expected_status_state
    assert semantic["status"]["parser_family"] == case.expected_semantic_family
    assert semantic["status"]["detail"] == case.expected_detail
    assert semantic["weld"] == case.expected_family_payload
    assert semantic["provenance"]["authority"] == "pdf"
    assert semantic["metadata"]["form3_context_supplied"] == "true"


def test_run_pipeline_packet_surfaces_surface_finish_equivalence_status_and_payload(tmp_path):
    case = _PipelineTestCase(
        char_no=23,
        form3_requirement="Ra 3.2 um",
        revb_text="3.2 Ra",
        expected_status="unchanged",
        expected_reasons=[
            "semantic surface finish match: Ra 3.2 um",
            "Semantic family agreement overrides formatting-only numeric/text variation",
        ],
        expected_semantic_family="surface_finish",
        expected_detail="parsed bounded surface finish callout from authoritative semantic text",
        expected_family_payload={
            "canonical_text": "Ra 3.2 um",
            "roughness_value": "3.2",
            "units": "um",
            "value_micrometers": "3.2",
            "indicator": "Ra",
        },
    )

    item, semantic = _run_pipeline_semantic_case(tmp_path, case)

    assert item["status"] == "unchanged"
    assert item["reasons"] == case.expected_reasons
    assert semantic["status"]["state"] == case.expected_status_state
    assert semantic["status"]["parser_family"] == case.expected_semantic_family
    assert semantic["status"]["detail"] == case.expected_detail
    assert semantic["surface_finish"] == case.expected_family_payload
    assert semantic["provenance"]["authority"] == "pdf"
    assert semantic["metadata"]["form3_context_supplied"] == "true"
    assert semantic["raw_text"] == "3.2 Ra"
    assert semantic["normalized_text"] == "3.2 Ra"
    assert semantic["surface_finish"]["canonical_text"] == "Ra 3.2 um"
    assert semantic["status"].get("reason_code") is None


def test_run_pipeline_packet_surfaces_fit_delta_status_and_payload(tmp_path):
    case = _PipelineTestCase(
        char_no=24,
        form3_requirement="H7/p6",
        revb_text="H7/g6",
        expected_status="changed",
        expected_reasons=[
            "semantic fit changed: fit class H7/p6 → H7/g6",
            "Semantic family delta indicates a meaningful drawing requirement change",
        ],
        expected_semantic_family="fit",
        expected_detail="parsed bounded fit callout from authoritative semantic text",
        expected_family_payload={
            "canonical_text": "H7/g6",
            "fit_class": "H7/g6",
            "hole_class": "H7",
            "shaft_class": "g6",
            "basis": "hole_basis",
            "standard_hint": "iso_limits_and_fits",
        },
    )

    item, semantic = _run_pipeline_semantic_case(tmp_path, case)

    assert item["status"] == "changed"
    assert item["reasons"] == case.expected_reasons
    assert semantic["status"]["state"] == case.expected_status_state
    assert semantic["status"]["parser_family"] == case.expected_semantic_family
    assert semantic["status"]["detail"] == case.expected_detail
    assert semantic["fit"] == case.expected_family_payload
    assert semantic["provenance"]["authority"] == "pdf"
    assert semantic["metadata"]["form3_context_supplied"] == "true"
    assert semantic["status"].get("reason_code") is None


def test_run_pipeline_packet_surfaces_fallback_semantic_status_and_reason_code(tmp_path):
    case = _PipelineTestCase(
        char_no=25,
        form3_requirement="12.0 ± 0.1",
        revb_text="12.0",
        expected_status="unchanged",
        expected_reasons=[
            "semantic comparison fallback: left semantic state empty/surface_finish_no_match",
            "semantic comparison fallback: right semantic state empty/surface_finish_no_match",
            "Primary dimension matches: 12.0",
            "Numeric values match (50% overlap)",
            "High location agreement after global alignment",
        ],
        expected_semantic_family="surface_finish",
        expected_detail="text did not match the bounded GD&T, weld, surface finish, or fit grammar",
        expected_family_payload=None,
        expected_status_state="empty",
        expected_reason_code="surface_finish_no_match",
    )

    item, semantic = _run_pipeline_semantic_case(tmp_path, case)

    assert item["status"] == "unchanged"
    assert item["reasons"] == case.expected_reasons
    assert item["reasons"][0].startswith("semantic comparison fallback")
    assert item["reasons"][1].startswith("semantic comparison fallback")
    assert any("Primary dimension matches" in reason for reason in item["reasons"])
    assert semantic["provenance"]["authority"] == "pdf"
    assert semantic["status"]["state"] == case.expected_status_state
    assert semantic["status"]["parser_family"] == case.expected_semantic_family
    assert semantic["status"]["reason_code"] == case.expected_reason_code
    assert semantic["status"]["detail"] == case.expected_detail
    assert semantic["raw_text"] == "12.0"
    assert semantic["normalized_text"] == "12.0"
    assert "gdt" not in semantic
    assert "weld" not in semantic
    assert "surface_finish" not in semantic
    assert "fit" not in semantic


def test_run_pipeline_uses_pdf_span_inputs_for_matched_and_added_semantics(tmp_path):
    from delta_preservation.cli import run_pipeline

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
         patch("delta_preservation.cli.load_ground_truth_packet", return_value=_empty_truth_packet()), \
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
    assert packet["items"][0]["semantic_callout"]["provenance"]["authority"] == "pdf"
    assert packet["items"][1]["semantic_callout"]["provenance"]["authority"] == "pdf"
    assert packet["items"][0]["semantic_callout"]["raw_text"] == "12.0"
    assert packet["items"][1]["semantic_callout"]["raw_text"] == "NEW DIM"
    assert packet["items"][0]["semantic_callout"]["status"]["parser_family"] == "surface_finish"
    assert packet["items"][0]["semantic_callout"]["status"]["reason_code"] == "surface_finish_no_match"
    assert packet["items"][1]["semantic_callout"]["status"]["parser_family"] == "surface_finish"
    assert packet["items"][1]["semantic_callout"]["status"]["reason_code"] == "surface_finish_no_match"


def test_run_pipeline_persists_surface_finish_semantic_callouts_in_delta_packet(tmp_path):
    from delta_preservation.cli import run_pipeline

    revA = tmp_path / "revA.pdf"
    revB = tmp_path / "revB.pdf"
    form3 = tmp_path / "form3.xlsx"
    revA.write_bytes(b"%PDF-1.4")
    revB.write_bytes(b"%PDF-1.4")
    form3.write_bytes(b"PK")

    anchor = _FakeAnchor(char_no=13, requirement_raw="FIT H7/p6")
    surface_span = _span("Ra 3.2 um", block_id=6, line_id=1, span_id=2, x0=42.0, y0=24.0)

    classified_item = _FakeInternalDeltaItem(
        char_no=13,
        status="changed",
        confidence=0.87,
        reasons=["Matched surface finish callout changed"],
        component_scores={"location": 0.9, "text": 0.86, "context": 0.84},
        match=_FakeMatch(surface_span),
    )

    render_stub = np.zeros((40, 40, 3), dtype=np.uint8)

    def fake_extract_text_spans(pdf_path, page_index=0):
        name = Path(pdf_path).name
        if name == "revA.pdf":
            return [_span("FORM3 ANCHOR", block_id=0, line_id=0, span_id=0, x0=12.0, y0=12.0)]
        return [surface_span]

    with patch("delta_preservation.cli.load_form3", return_value=[SimpleNamespace(char_no=13, requirement="FIT H7/p6")]), \
         patch("delta_preservation.cli.detect_balloons", return_value=[{"char_no": 13}]), \
         patch("delta_preservation.cli.extract_text_spans", side_effect=fake_extract_text_spans), \
         patch("delta_preservation.cli.build_revA_anchors", return_value=[anchor]), \
         patch("delta_preservation.cli.render_page", return_value=render_stub), \
         patch("delta_preservation.cli.estimate_transform", return_value=_FakeTransform()), \
         patch("delta_preservation.cli.estimate_transform_from_text_spans", return_value=None), \
         patch("delta_preservation.cli.generate_candidates", return_value=[_FakeCandidate(surface_span)]), \
         patch("delta_preservation.cli.assign_matches", return_value={13: _FakeMatch(surface_span)}), \
         patch("delta_preservation.cli.extract_tolerances_for_items", return_value={}), \
         patch("delta_preservation.cli.classify_delta", return_value=classified_item), \
         patch("delta_preservation.cli.detect_added_characteristics", return_value=[]), \
         patch("delta_preservation.cli.export_run_tolerance_debug"), \
         patch("delta_preservation.cli.pdf_to_img_coords", return_value=(0, 0, 10, 10)), \
         patch("delta_preservation.cli.crop_with_padding", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
         patch("delta_preservation.cli.save_snippet", side_effect=["a.png", "b.png"]), \
         patch("delta_preservation.cli.load_ground_truth_packet", return_value=_empty_truth_packet()), \
         patch("delta_preservation.cli.fitz.open", side_effect=[_FakeDoc(), _FakeDoc()]):
        run_dir = run_pipeline(
            revA_pdf=str(revA),
            revB_pdf=str(revB),
            form3_xlsx=str(form3),
            out_dir=str(tmp_path / "out"),
            part_name="semantic-packet-surface-finish",
        )

    packet = json.loads((run_dir / "delta_packet.json").read_text())
    item = packet["items"][0]
    semantic = item["semantic_callout"]

    assert item["char_no"] == 13
    assert item["status"] == "changed"
    assert semantic["provenance"]["authority"] == "pdf"
    assert semantic["provenance"]["source_type"] == "drawing_pdf"
    assert semantic["provenance"]["source_ref"] == "pdf:block:6/line:1/span:2"
    assert semantic["provenance"]["notes"] == [
        "PDF-derived spans are authoritative for semantic extraction.",
        "Form 3 requirement text was retained as secondary advisory context.",
        "PDF authority overrode conflicting Form 3 semantic wording.",
    ]
    assert semantic["status"] == {
        "state": "parsed",
        "parser_family": "surface_finish",
        "detail": "parsed bounded surface finish callout from authoritative semantic text",
    }
    assert semantic["metadata"]["dispatcher"] == "semantic_dispatch"
    assert semantic["metadata"]["authority_source"] == "pdf"
    assert semantic["metadata"]["planned_families"] == "gdt,weld,surface_finish,fit"
    assert semantic["metadata"]["conflict_detected"] == "true"
    assert semantic["metadata"]["form3_context_supplied"] == "true"
    assert semantic["raw_text"] == "Ra 3.2 um"
    assert semantic["normalized_text"] == "Ra 3.2 um"
    assert semantic["surface_finish"] == {
        "canonical_text": "Ra 3.2 um",
        "roughness_value": "3.2",
        "units": "um",
        "value_micrometers": "3.2",
        "indicator": "Ra",
    }
    assert "gdt" not in semantic
    assert "weld" not in semantic
    assert "fit" not in semantic


def test_run_pipeline_persists_fit_semantic_callouts_in_delta_packet(tmp_path):
    from delta_preservation.cli import run_pipeline

    revA = tmp_path / "revA.pdf"
    revB = tmp_path / "revB.pdf"
    form3 = tmp_path / "form3.xlsx"
    revA.write_bytes(b"%PDF-1.4")
    revB.write_bytes(b"%PDF-1.4")
    form3.write_bytes(b"PK")

    anchor = _FakeAnchor(char_no=17, requirement_raw="Ra 3.2 um")
    fit_span = _span("H7/p6", block_id=8, line_id=0, span_id=3, x0=46.0, y0=26.0)

    classified_item = _FakeInternalDeltaItem(
        char_no=17,
        status="changed",
        confidence=0.85,
        reasons=["Matched fit callout changed"],
        component_scores={"location": 0.89, "text": 0.84, "context": 0.82},
        match=_FakeMatch(fit_span),
    )

    render_stub = np.zeros((40, 40, 3), dtype=np.uint8)

    def fake_extract_text_spans(pdf_path, page_index=0):
        name = Path(pdf_path).name
        if name == "revA.pdf":
            return [_span("FORM3 ANCHOR", block_id=0, line_id=0, span_id=0, x0=12.0, y0=12.0)]
        return [fit_span]

    with patch("delta_preservation.cli.load_form3", return_value=[SimpleNamespace(char_no=17, requirement="Ra 3.2 um")]), \
         patch("delta_preservation.cli.detect_balloons", return_value=[{"char_no": 17}]), \
         patch("delta_preservation.cli.extract_text_spans", side_effect=fake_extract_text_spans), \
         patch("delta_preservation.cli.build_revA_anchors", return_value=[anchor]), \
         patch("delta_preservation.cli.render_page", return_value=render_stub), \
         patch("delta_preservation.cli.estimate_transform", return_value=_FakeTransform()), \
         patch("delta_preservation.cli.estimate_transform_from_text_spans", return_value=None), \
         patch("delta_preservation.cli.generate_candidates", return_value=[_FakeCandidate(fit_span)]), \
         patch("delta_preservation.cli.assign_matches", return_value={17: _FakeMatch(fit_span)}), \
         patch("delta_preservation.cli.extract_tolerances_for_items", return_value={}), \
         patch("delta_preservation.cli.classify_delta", return_value=classified_item), \
         patch("delta_preservation.cli.detect_added_characteristics", return_value=[]), \
         patch("delta_preservation.cli.export_run_tolerance_debug"), \
         patch("delta_preservation.cli.pdf_to_img_coords", return_value=(0, 0, 10, 10)), \
         patch("delta_preservation.cli.crop_with_padding", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
         patch("delta_preservation.cli.save_snippet", side_effect=["a.png", "b.png"]), \
         patch("delta_preservation.cli.load_ground_truth_packet", return_value=_empty_truth_packet()), \
         patch("delta_preservation.cli.fitz.open", side_effect=[_FakeDoc(), _FakeDoc()]):
        run_dir = run_pipeline(
            revA_pdf=str(revA),
            revB_pdf=str(revB),
            form3_xlsx=str(form3),
            out_dir=str(tmp_path / "out"),
            part_name="semantic-packet-fit",
        )

    packet = json.loads((run_dir / "delta_packet.json").read_text())
    item = packet["items"][0]
    semantic = item["semantic_callout"]

    assert item["char_no"] == 17
    assert item["status"] == "changed"
    assert semantic["provenance"]["authority"] == "pdf"
    assert semantic["provenance"]["source_type"] == "drawing_pdf"
    assert semantic["provenance"]["source_ref"] == "pdf:block:8/line:0/span:3"
    assert semantic["provenance"]["notes"] == [
        "PDF-derived spans are authoritative for semantic extraction.",
        "Form 3 requirement text was retained as secondary advisory context.",
        "PDF authority overrode conflicting Form 3 semantic wording.",
    ]
    assert semantic["status"] == {
        "state": "parsed",
        "parser_family": "fit",
        "detail": "parsed bounded fit callout from authoritative semantic text",
    }
    assert semantic["metadata"]["dispatcher"] == "semantic_dispatch"
    assert semantic["metadata"]["authority_source"] == "pdf"
    assert semantic["metadata"]["planned_families"] == "gdt,weld,surface_finish,fit"
    assert semantic["metadata"]["conflict_detected"] == "true"
    assert semantic["metadata"]["form3_context_supplied"] == "true"
    assert semantic["raw_text"] == "H7/p6"
    assert semantic["normalized_text"] == "H7/p6"
    assert semantic["fit"] == {
        "canonical_text": "H7/p6",
        "fit_class": "H7/p6",
        "hole_class": "H7",
        "shaft_class": "p6",
        "basis": "hole_basis",
        "standard_hint": "iso_limits_and_fits",
    }
    assert "gdt" not in semantic
    assert "weld" not in semantic
    assert "surface_finish" not in semantic


def test_run_pipeline_persists_weld_semantic_callouts_in_delta_packet(tmp_path):
    from delta_preservation.cli import run_pipeline

    revA = tmp_path / "revA.pdf"
    revB = tmp_path / "revB.pdf"
    form3 = tmp_path / "form3.xlsx"
    revA.write_bytes(b"%PDF-1.4")
    revB.write_bytes(b"%PDF-1.4")
    form3.write_bytes(b"PK")

    anchor = _FakeAnchor(char_no=11, requirement_raw="GROOVE WELD PER NOTE")
    weld_span = _span("1/8 FILLET BOTH SIDES ALL AROUND 1.50-3.00 FLUSH TAIL: FIELD", block_id=7, line_id=3, span_id=4, x0=44.0, y0=28.0)

    classified_item = _FakeInternalDeltaItem(
        char_no=11,
        status="changed",
        confidence=0.86,
        reasons=["Matched weld callout changed"],
        component_scores={"location": 0.9, "text": 0.85, "context": 0.82},
        match=_FakeMatch(weld_span),
    )

    render_stub = np.zeros((40, 40, 3), dtype=np.uint8)

    def fake_extract_text_spans(pdf_path, page_index=0):
        name = Path(pdf_path).name
        if name == "revA.pdf":
            return [_span("FORM3 ANCHOR", block_id=0, line_id=0, span_id=0, x0=12.0, y0=12.0)]
        return [weld_span]

    with patch("delta_preservation.cli.load_form3", return_value=[SimpleNamespace(char_no=11, requirement="GROOVE WELD PER NOTE")]), \
         patch("delta_preservation.cli.detect_balloons", return_value=[{"char_no": 11}]), \
         patch("delta_preservation.cli.extract_text_spans", side_effect=fake_extract_text_spans), \
         patch("delta_preservation.cli.build_revA_anchors", return_value=[anchor]), \
         patch("delta_preservation.cli.render_page", return_value=render_stub), \
         patch("delta_preservation.cli.estimate_transform", return_value=_FakeTransform()), \
         patch("delta_preservation.cli.estimate_transform_from_text_spans", return_value=None), \
         patch("delta_preservation.cli.generate_candidates", return_value=[_FakeCandidate(weld_span)]), \
         patch("delta_preservation.cli.assign_matches", return_value={11: _FakeMatch(weld_span)}), \
         patch("delta_preservation.cli.extract_tolerances_for_items", return_value={}), \
         patch("delta_preservation.cli.classify_delta", return_value=classified_item), \
         patch("delta_preservation.cli.detect_added_characteristics", return_value=[]), \
         patch("delta_preservation.cli.export_run_tolerance_debug"), \
         patch("delta_preservation.cli.pdf_to_img_coords", return_value=(0, 0, 10, 10)), \
         patch("delta_preservation.cli.crop_with_padding", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
         patch("delta_preservation.cli.save_snippet", side_effect=["a.png", "b.png"]), \
         patch("delta_preservation.cli.load_ground_truth_packet", return_value=_empty_truth_packet()), \
         patch("delta_preservation.cli.fitz.open", side_effect=[_FakeDoc(), _FakeDoc()]):
        run_dir = run_pipeline(
            revA_pdf=str(revA),
            revB_pdf=str(revB),
            form3_xlsx=str(form3),
            out_dir=str(tmp_path / "out"),
            part_name="semantic-packet-weld",
        )

    packet = json.loads((run_dir / "delta_packet.json").read_text())
    item = packet["items"][0]
    semantic = item["semantic_callout"]

    assert item["char_no"] == 11
    assert item["status"] == "changed"
    assert semantic["provenance"]["authority"] == "pdf"
    assert semantic["provenance"]["source_type"] == "drawing_pdf"
    assert semantic["provenance"]["source_ref"] == "pdf:block:7/line:3/span:4"
    assert semantic["provenance"]["notes"] == [
        "PDF-derived spans are authoritative for semantic extraction.",
        "Form 3 requirement text was retained as secondary advisory context.",
        "PDF authority overrode conflicting Form 3 semantic wording.",
    ]
    assert semantic["status"] == {
        "state": "parsed",
        "parser_family": "weld",
        "detail": "parsed bounded weld callout from authoritative semantic text",
    }
    assert semantic["metadata"]["dispatcher"] == "semantic_dispatch"
    assert semantic["metadata"]["authority_source"] == "pdf"
    assert semantic["metadata"]["planned_families"] == "gdt,weld,surface_finish,fit"
    assert semantic["metadata"]["conflict_detected"] == "true"
    assert semantic["metadata"]["form3_context_supplied"] == "true"
    assert semantic["raw_text"] == "1/8 FILLET BOTH SIDES ALL AROUND 1.50-3.00 FLUSH TAIL: FIELD"
    assert semantic["normalized_text"] == "1/8 FILLET BOTH SIDES ALL AROUND 1.50-3.00 FLUSH TAIL: FIELD"
    assert semantic["weld"] == {
        "process": "fillet",
        "size": "1/8",
        "contour": "flush",
        "side": "both_sides",
        "length": "1.50",
        "pitch": "3.00",
        "tail": "FIELD",
        "all_around": True,
    }
    assert "gdt" not in semantic
    assert "surface_finish" not in semantic
    assert "fit" not in semantic


def test_run_pipeline_packet_keeps_full_part6_semantic_text_when_classification_already_trusted_it(tmp_path):
    from delta_preservation.cli import run_pipeline

    revA = tmp_path / "revA.pdf"
    revB = tmp_path / "revB.pdf"
    form3 = tmp_path / "form3.xlsx"
    revA.write_bytes(b"%PDF-1.4")
    revB.write_bytes(b"%PDF-1.4")
    form3.write_bytes(b"PK")

    anchor = _FakeAnchor(char_no=10, requirement_raw="⌖ ⌀.050 Ⓜ D B C")
    symbol_span = _span("⌖", block_id=12, line_id=0, span_id=0, x0=10.0, y0=10.0)
    tolerance_span = _span("⌀.050", block_id=12, line_id=0, span_id=1, x0=22.0, y0=10.0)
    modifier_span = _span("Ⓜ", block_id=12, line_id=0, span_id=2, x0=46.0, y0=10.0)
    datum_d_span = _span("D", block_id=12, line_id=0, span_id=3, x0=58.0, y0=10.0)
    datum_b_span = _span("B", block_id=12, line_id=0, span_id=4, x0=68.0, y0=10.0)
    datum_c_span = _span("C", block_id=12, line_id=0, span_id=5, x0=78.0, y0=10.0)
    partial_group = GroupedSpan(
        text="⌖ ⌀.050",
        bbox_pdf=(10.0, 10.0, 42.0, 18.0),
        font_size=10.0,
        block_id=12,
        line_id=0,
        span_id=0,
        source_spans=[symbol_span, tolerance_span],
    )

    classified_item = _FakeInternalDeltaItem(
        char_no=10,
        status="unchanged",
        confidence=0.9,
        reasons=[
            "semantic GD&T match: control type position, tolerance ⌀.050, datums D/B/C, modifiers MMC",
            "Semantic family agreement overrides formatting-only numeric/text variation",
        ],
        component_scores={"location": 0.9, "text": 0.9, "context": 0.9},
        match=_FakeMatch(partial_group),
    )

    render_stub = np.zeros((40, 40, 3), dtype=np.uint8)

    def fake_extract_text_spans(pdf_path, page_index=0):
        name = Path(pdf_path).name
        if name == "revA.pdf":
            return [_span("FORM3 ANCHOR", block_id=0, line_id=0, span_id=0, x0=12.0, y0=12.0)]
        return [datum_b_span, datum_c_span, datum_d_span, modifier_span, tolerance_span, symbol_span]

    with patch("delta_preservation.cli.load_form3", return_value=[SimpleNamespace(char_no=10, requirement="⌖ ⌀.050 Ⓜ D B C")]), \
         patch("delta_preservation.cli.detect_balloons", return_value=[{"char_no": 10}]), \
         patch("delta_preservation.cli.extract_text_spans", side_effect=fake_extract_text_spans), \
         patch("delta_preservation.cli.build_revA_anchors", return_value=[anchor]), \
         patch("delta_preservation.cli.render_page", return_value=render_stub), \
         patch("delta_preservation.cli.estimate_transform", return_value=_FakeTransform()), \
         patch("delta_preservation.cli.estimate_transform_from_text_spans", return_value=None), \
         patch("delta_preservation.cli.generate_candidates", return_value=[_FakeCandidate(partial_group)]), \
         patch("delta_preservation.cli.assign_matches", return_value={10: _FakeMatch(partial_group)}), \
         patch("delta_preservation.cli.extract_tolerances_for_items", return_value={}), \
         patch("delta_preservation.cli.classify_delta", return_value=classified_item), \
         patch("delta_preservation.cli.detect_added_characteristics", return_value=[]), \
         patch("delta_preservation.cli.export_run_tolerance_debug"), \
         patch("delta_preservation.cli.pdf_to_img_coords", return_value=(0, 0, 10, 10)), \
         patch("delta_preservation.cli.crop_with_padding", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
         patch("delta_preservation.cli.save_snippet", side_effect=["a.png", "b.png"]), \
         patch("delta_preservation.cli.load_ground_truth_packet", return_value=_empty_truth_packet()), \
         patch("delta_preservation.cli.fitz.open", side_effect=[_FakeDoc(), _FakeDoc()]):
        run_dir = run_pipeline(
            revA_pdf=str(revA),
            revB_pdf=str(revB),
            form3_xlsx=str(form3),
            out_dir=str(tmp_path / "out"),
            part_name="semantic-packet-gdt-closure",
        )

    packet = json.loads((run_dir / "delta_packet.json").read_text())
    item = packet["items"][0]
    semantic = item["semantic_callout"]

    assert item["reasons"][0].startswith("semantic GD&T match")
    assert item["requirement_revB"] == "⌖ ⌀.050 Ⓜ D B C"
    assert semantic["raw_text"] == "⌖ ⌀.050 Ⓜ D B C", (
        "Expected packet semantic text to stay aligned with the full Part 6 FCF that classification already trusted; "
        f"got {semantic['raw_text']}"
    )
