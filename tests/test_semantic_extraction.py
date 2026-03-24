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


def test_extract_semantic_callout_prefers_pdf_spans_over_conflicting_form3_text():
    semantic = extract_semantic_callout(
        pdf_spans=[
            _span("⟂", span_id=0, x0=10.0),
            _span("0.05", span_id=1, x0=20.0),
            _span("A", span_id=2, x0=30.0),
            _span("B", span_id=3, x0=40.0),
        ],
        form3_requirement="SURFACE FINISH 63 MICROINCH",
    )

    assert semantic.provenance.authority == "pdf"
    assert semantic.provenance.source_type == "drawing_pdf"
    assert semantic.provenance.source_ref == "pdf:block:0/line:0/span:0"
    assert semantic.raw_text == "⟂ 0.05 A B"
    assert semantic.normalized_text == "⟂ 0.05 A B"
    assert semantic.status.state == "not_implemented"
    assert semantic.status.parser_family == "semantic_dispatch"
    assert semantic.status.reason_code == "not_implemented_in_slice"
    assert semantic.metadata["authority_source"] == "pdf"
    assert semantic.metadata["form3_context_supplied"] == "true"
    assert semantic.metadata["conflict_detected"] == "true"
    assert any("secondary advisory context" in note for note in semantic.provenance.notes)
    assert any("overrode conflicting Form 3" in note for note in semantic.provenance.notes)



def test_extract_semantic_callout_returns_stub_payloads_for_planned_families():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("H7/g6", span_id=5, x0=15.0)],
        form3_requirement="FIT CLASS RC3",
    )

    assert semantic.gdt is not None
    assert semantic.weld is not None
    assert semantic.surface_finish is not None
    assert semantic.fit is not None
    assert semantic.status.state == "not_implemented"
    assert semantic.status.reason_code == "not_implemented_in_slice"
    assert semantic.metadata["planned_families"] == "gdt,weld,surface_finish,fit"
    assert semantic.metadata["dispatcher"] == "semantic_dispatch"



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
