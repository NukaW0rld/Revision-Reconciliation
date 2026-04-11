"""
TDD tests for run_pipeline() stage_callback parameter.

RED phase: these tests verify that stage_callback is accepted and called
with correct (index, name) tuples for all 8 pipeline stages.
"""
import inspect
import pytest
from unittest.mock import MagicMock, patch

from delta_preservation.evaluation.contracts import GroundTruthPacket


def _empty_truth_packet() -> GroundTruthPacket:
    return GroundTruthPacket.model_validate(
        {
            "part_name": "Test Fixture",
            "general_notes": "",
            "characteristics": [],
        }
    )


def test_run_pipeline_signature_has_stage_callback():
    """run_pipeline() must accept stage_callback as an optional keyword argument."""
    from delta_preservation.cli import run_pipeline
    sig = inspect.signature(run_pipeline)
    assert "stage_callback" in sig.parameters, "stage_callback not in run_pipeline signature"
    param = sig.parameters["stage_callback"]
    assert param.default is None, "stage_callback default must be None"


def test_stage_callback_default_is_none():
    """stage_callback defaults to None so existing callers work unchanged."""
    from delta_preservation.cli import run_pipeline
    sig = inspect.signature(run_pipeline)
    param = sig.parameters["stage_callback"]
    assert param.default is None


def test_stage_callback_called_8_times(tmp_path):
    """stage_callback is called exactly 8 times during a pipeline run."""
    from delta_preservation.cli import run_pipeline
    import pathlib

    # Create minimal fake PDFs and xlsx so file validation passes
    revA = tmp_path / "revA.pdf"
    revB = tmp_path / "revB.pdf"
    form3 = tmp_path / "form3.xlsx"
    revA.write_bytes(b"%PDF-1.4 1 0 obj << >> endobj")
    revB.write_bytes(b"%PDF-1.4 1 0 obj << >> endobj")
    form3.write_bytes(b"PK")  # minimal xlsx stub

    cb = MagicMock()

    # Patch all pipeline stage functions to avoid needing real files
    with patch("delta_preservation.cli.load_form3", return_value=[]), \
         patch("delta_preservation.cli.detect_balloons", return_value={}), \
         patch("delta_preservation.cli.extract_text_spans", return_value=[]), \
         patch("delta_preservation.cli.build_revA_anchors", return_value=[]), \
         patch("delta_preservation.cli.render_page", return_value=MagicMock()), \
         patch("delta_preservation.cli.estimate_transform", return_value=MagicMock(inliers=50, inlier_ratio=0.8, H=None)), \
         patch("delta_preservation.cli.estimate_transform_from_text_spans", return_value=None), \
         patch("delta_preservation.cli.generate_candidates", return_value=[]), \
         patch("delta_preservation.cli.assign_matches", return_value={}), \
         patch("delta_preservation.cli.extract_tolerances_for_items", return_value={}), \
         patch("delta_preservation.cli.detect_added_characteristics", return_value=[]), \
         patch("delta_preservation.cli.export_run_tolerance_debug"), \
         patch("delta_preservation.cli.load_ground_truth_packet", return_value=_empty_truth_packet()), \
         patch("delta_preservation.cli.fitz.open", return_value=MagicMock()), \
         patch("delta_preservation.cli.DeltaPacket") as MockPacket:
        MockPacket.return_value.model_dump_json.return_value = "{}"
        run_pipeline(
            revA_pdf=str(revA),
            revB_pdf=str(revB),
            form3_xlsx=str(form3),
            out_dir=str(tmp_path / "out"),
            part_name="test",
            stage_callback=cb,
        )

    assert cb.call_count == 8, f"Expected 8 stage_callback calls, got {cb.call_count}"


def test_stage_callback_called_with_correct_indices_and_names(tmp_path):
    """stage_callback is called with (0..7, stage_name) in order."""
    from delta_preservation.cli import run_pipeline

    revA = tmp_path / "revA.pdf"
    revB = tmp_path / "revB.pdf"
    form3 = tmp_path / "form3.xlsx"
    revA.write_bytes(b"%PDF-1.4 1 0 obj << >> endobj")
    revB.write_bytes(b"%PDF-1.4 1 0 obj << >> endobj")
    form3.write_bytes(b"PK")

    calls = []

    def cb(idx, name):
        calls.append((idx, name))

    expected = [
        (0, "Form 3 parsing"),
        (1, "Balloon detection"),
        (2, "Text extraction"),
        (3, "Anchor building"),
        (4, "Alignment"),
        (5, "Candidate matching"),
        (6, "Classification"),
        (7, "Output"),
    ]

    with patch("delta_preservation.cli.load_form3", return_value=[]), \
         patch("delta_preservation.cli.detect_balloons", return_value={}), \
         patch("delta_preservation.cli.extract_text_spans", return_value=[]), \
         patch("delta_preservation.cli.build_revA_anchors", return_value=[]), \
         patch("delta_preservation.cli.render_page", return_value=MagicMock()), \
         patch("delta_preservation.cli.estimate_transform", return_value=MagicMock(inliers=50, inlier_ratio=0.8, H=None)), \
         patch("delta_preservation.cli.estimate_transform_from_text_spans", return_value=None), \
         patch("delta_preservation.cli.generate_candidates", return_value=[]), \
         patch("delta_preservation.cli.assign_matches", return_value={}), \
         patch("delta_preservation.cli.extract_tolerances_for_items", return_value={}), \
         patch("delta_preservation.cli.detect_added_characteristics", return_value=[]), \
         patch("delta_preservation.cli.export_run_tolerance_debug"), \
         patch("delta_preservation.cli.load_ground_truth_packet", return_value=_empty_truth_packet()), \
         patch("delta_preservation.cli.fitz.open", return_value=MagicMock()), \
         patch("delta_preservation.cli.DeltaPacket") as MockPacket:
        MockPacket.return_value.model_dump_json.return_value = "{}"
        run_pipeline(
            revA_pdf=str(revA),
            revB_pdf=str(revB),
            form3_xlsx=str(form3),
            out_dir=str(tmp_path / "out"),
            part_name="test",
            stage_callback=cb,
        )

    assert calls == expected, f"Callback calls mismatch:\nExpected: {expected}\nGot: {calls}"


def test_stage_callback_none_still_works(tmp_path):
    """When stage_callback=None (default), pipeline runs without error."""
    from delta_preservation.cli import run_pipeline

    revA = tmp_path / "revA.pdf"
    revB = tmp_path / "revB.pdf"
    form3 = tmp_path / "form3.xlsx"
    revA.write_bytes(b"%PDF-1.4 1 0 obj << >> endobj")
    revB.write_bytes(b"%PDF-1.4 1 0 obj << >> endobj")
    form3.write_bytes(b"PK")

    with patch("delta_preservation.cli.load_form3", return_value=[]), \
         patch("delta_preservation.cli.detect_balloons", return_value={}), \
         patch("delta_preservation.cli.extract_text_spans", return_value=[]), \
         patch("delta_preservation.cli.build_revA_anchors", return_value=[]), \
         patch("delta_preservation.cli.render_page", return_value=MagicMock()), \
         patch("delta_preservation.cli.estimate_transform", return_value=MagicMock(inliers=50, inlier_ratio=0.8, H=None)), \
         patch("delta_preservation.cli.estimate_transform_from_text_spans", return_value=None), \
         patch("delta_preservation.cli.generate_candidates", return_value=[]), \
         patch("delta_preservation.cli.assign_matches", return_value={}), \
         patch("delta_preservation.cli.extract_tolerances_for_items", return_value={}), \
         patch("delta_preservation.cli.detect_added_characteristics", return_value=[]), \
         patch("delta_preservation.cli.export_run_tolerance_debug"), \
         patch("delta_preservation.cli.load_ground_truth_packet", return_value=_empty_truth_packet()), \
         patch("delta_preservation.cli.fitz.open", return_value=MagicMock()), \
         patch("delta_preservation.cli.DeltaPacket") as MockPacket:
        MockPacket.return_value.model_dump_json.return_value = "{}"
        # No stage_callback argument — should work exactly as before
        result = run_pipeline(
            revA_pdf=str(revA),
            revB_pdf=str(revB),
            form3_xlsx=str(form3),
            out_dir=str(tmp_path / "out"),
            part_name="test",
        )
    assert result is not None
