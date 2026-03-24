from delta_preservation.io.pdf import TextSpan
from delta_preservation.reconcile.normalize import extract_semantic_callout


def _span(text: str, *, block_id: int = 0, line_id: int = 0, span_id: int = 0, x0: float = 0.0, y0: float = 0.0):
    return TextSpan(
        text=text,
        bbox_pdf=(x0, y0, x0 + 10.0, y0 + 5.0),
        font_size=10.0,
        block_id=block_id,
        line_id=line_id,
        span_id=span_id,
    )


def test_extract_semantic_callout_parses_position_gdt_from_pdf_spans_and_ignores_conflicting_form3_text():
    semantic = extract_semantic_callout(
        pdf_spans=[
            _span("⌖", span_id=0, x0=10.0),
            _span("⌀0.10", span_id=1, x0=20.0),
            _span("M", span_id=2, x0=30.0),
            _span("A", span_id=3, x0=40.0),
            _span("B", span_id=4, x0=50.0),
            _span("C", span_id=5, x0=60.0),
        ],
        form3_requirement="SURFACE FINISH 63 MICROINCH",
    )

    assert semantic.provenance.authority == "pdf"
    assert semantic.provenance.source_type == "drawing_pdf"
    assert semantic.provenance.source_ref == "pdf:block:0/line:0/span:0"
    assert semantic.raw_text == "⌖ ⌀0.10 M A B C"
    assert semantic.normalized_text == "⌖ ⌀0.10 M A B C"
    assert semantic.status.state == "parsed"
    assert semantic.status.parser_family == "gdt"
    assert semantic.status.reason_code is None
    assert semantic.status.detail == "parsed feature control frame from PDF spans"
    assert semantic.metadata["authority_source"] == "pdf"
    assert semantic.metadata["form3_context_supplied"] == "true"
    assert semantic.metadata["conflict_detected"] == "true"
    assert any("secondary advisory context" in note for note in semantic.provenance.notes)
    assert any("overrode conflicting Form 3" in note for note in semantic.provenance.notes)

    assert semantic.gdt is not None
    assert semantic.gdt.frame_text == "⌖ | ⌀0.10 | M | A | B | C"
    assert semantic.gdt.control_type == "position"
    assert semantic.gdt.tolerance_text == "⌀0.10"
    assert semantic.gdt.datum_refs == ["A", "B", "C"]
    assert semantic.gdt.modifiers == ["MMC"]
    assert semantic.weld is None
    assert semantic.surface_finish is None
    assert semantic.fit is None


def test_extract_semantic_callout_parses_profile_gdt_without_datums_when_frame_has_no_references():
    semantic = extract_semantic_callout(
        pdf_spans=[
            _span("⌒", span_id=0, x0=10.0),
            _span("0.20", span_id=1, x0=20.0),
        ],
        form3_requirement="PROFILE PER NOTE",
    )

    assert semantic.provenance.authority == "pdf"
    assert semantic.raw_text == "⌒ 0.20"
    assert semantic.normalized_text == "⌒ 0.20"
    assert semantic.status.state == "parsed"
    assert semantic.status.parser_family == "gdt"
    assert semantic.status.reason_code is None
    assert semantic.status.detail == "parsed feature control frame from PDF spans"

    assert semantic.gdt is not None
    assert semantic.gdt.control_type == "profile_of_a_line"
    assert semantic.gdt.frame_text == "⌒ | 0.20"
    assert semantic.gdt.tolerance_text == "0.20"
    assert semantic.gdt.modifiers == []
    assert semantic.gdt.datum_refs == []
    assert semantic.metadata["authority_source"] == "pdf"


def test_extract_semantic_callout_returns_empty_weld_status_for_unrecognized_semantic_text():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("FLAG NOTE 12", span_id=7, x0=12.0)],
        form3_requirement="FLAG NOTE 12",
    )

    assert semantic.provenance.authority == "pdf"
    assert semantic.raw_text == "FLAG NOTE 12"
    assert semantic.normalized_text == "FLAG NOTE 12"
    assert semantic.status.state == "empty"
    assert semantic.status.parser_family == "weld"
    assert semantic.status.reason_code == "weld_no_match"
    assert semantic.status.detail == "text did not match the bounded weld subset or GD&T frame grammar"
    assert semantic.gdt is None
    assert semantic.weld is None
    assert semantic.surface_finish is None
    assert semantic.fit is None
    assert semantic.metadata["authority_source"] == "pdf"
    assert semantic.metadata["dispatcher"] == "semantic_dispatch"



def test_extract_semantic_callout_returns_error_status_for_malformed_gdt_frame():
    semantic = extract_semantic_callout(
        pdf_spans=[
            _span("⌖", span_id=0, x0=10.0),
            _span("A", span_id=1, x0=20.0),
        ],
        form3_requirement="POSITION TO DATUM A",
    )

    assert semantic.provenance.authority == "pdf"
    assert semantic.raw_text == "⌖ A"
    assert semantic.normalized_text == "⌖ A"
    assert semantic.status.state == "error"
    assert semantic.status.parser_family == "gdt"
    assert semantic.status.reason_code == "gdt_malformed_frame"
    assert semantic.status.detail == "recognized GD&T control symbol but missing tolerance segment"
    assert semantic.gdt is None
    assert semantic.weld is None
    assert semantic.surface_finish is None
    assert semantic.fit is None
    assert semantic.metadata["authority_source"] == "pdf"
    assert semantic.metadata["dispatcher"] == "semantic_dispatch"



def test_extract_semantic_callout_parses_bounded_weld_subset_from_fragmented_pdf_spans():
    semantic = extract_semantic_callout(
        pdf_spans=[
            _span("1/8", span_id=0, x0=10.0),
            _span("FILLET", span_id=1, x0=20.0),
            _span("BOTH", span_id=2, x0=30.0),
            _span("SIDES", span_id=3, x0=40.0),
            _span("ALL", span_id=4, x0=50.0),
            _span("AROUND", span_id=5, x0=60.0),
            _span("1.50-3.00", span_id=6, x0=70.0),
            _span("FLUSH", span_id=7, x0=80.0),
            _span("TAIL:", span_id=8, x0=90.0),
            _span("FIELD", span_id=9, x0=100.0),
        ],
        form3_requirement="GROOVE WELD PER FORM 3",
    )

    assert semantic.provenance.authority == "pdf"
    assert semantic.raw_text == "1/8 FILLET BOTH SIDES ALL AROUND 1.50-3.00 FLUSH TAIL: FIELD"
    assert semantic.normalized_text == semantic.raw_text
    assert semantic.status.state == "parsed"
    assert semantic.status.parser_family == "weld"
    assert semantic.status.reason_code is None
    assert semantic.status.detail == "parsed bounded weld callout from authoritative semantic text"
    assert semantic.metadata["conflict_detected"] == "true"
    assert semantic.gdt is None
    assert semantic.weld is not None
    assert semantic.weld.process == "fillet"
    assert semantic.weld.size == "1/8"
    assert semantic.weld.side == "both_sides"
    assert semantic.weld.all_around is True
    assert semantic.weld.length == "1.50"
    assert semantic.weld.pitch == "3.00"
    assert semantic.weld.contour == "flush"
    assert semantic.weld.tail == "FIELD"
    assert semantic.surface_finish is None
    assert semantic.fit is None
    assert any("overrode conflicting Form 3" in note for note in semantic.provenance.notes)



def test_extract_semantic_callout_returns_error_for_malformed_weld_missing_size():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("FILLET", span_id=0, x0=10.0)],
        form3_requirement="1/8 FILLET",
    )

    assert semantic.provenance.authority == "pdf"
    assert semantic.status.state == "error"
    assert semantic.status.parser_family == "weld"
    assert semantic.status.reason_code == "weld_malformed"
    assert semantic.status.detail == "recognized weld callout is missing a parseable size token before the weld type"
    assert semantic.weld is None
    assert semantic.gdt is None



def test_extract_semantic_callout_returns_error_for_unsupported_weld_segment():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("1/8 FILLET STAGGERED", span_id=0, x0=10.0)],
        form3_requirement="1/8 FILLET",
    )

    assert semantic.provenance.authority == "pdf"
    assert semantic.status.state == "error"
    assert semantic.status.parser_family == "weld"
    assert semantic.status.reason_code == "weld_unsupported"
    assert "STAGGERED" in semantic.status.detail
    assert semantic.weld is None
    assert semantic.gdt is None



def test_extract_semantic_callout_falls_back_to_form3_only_when_pdf_spans_missing():
    semantic = extract_semantic_callout(
        pdf_spans=[],
        form3_requirement="WELD ALL AROUND 1/8 FILLET",
    )

    assert semantic.provenance.authority == "form3"
    assert semantic.provenance.source_type == "form3_requirement"
    assert semantic.provenance.source_ref == "form3:requirement"
    assert semantic.raw_text == "WELD ALL AROUND 1/8 FILLET"
    assert semantic.normalized_text == "WELD ALL AROUND 1/8 FILLET"
    assert semantic.metadata["authority_source"] == "form3"
    assert semantic.metadata["context_alignment"] == "form3_only"
    assert any("fallback source" in note for note in semantic.provenance.notes)
    assert semantic.status.state == "parsed"
    assert semantic.status.parser_family == "weld"
    assert semantic.weld is not None
    assert semantic.weld.process == "fillet"
    assert semantic.weld.size == "1/8"
    assert semantic.weld.all_around is True
