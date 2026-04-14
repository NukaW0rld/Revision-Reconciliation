from delta_preservation.io.pdf import TextSpan
from delta_preservation.types import (
    FitSemanticPayload,
    GdtSemanticPayload,
    SemanticCallout,
    SemanticParserStatus,
    SemanticProvenance,
    SurfaceFinishSemanticPayload,
    WeldSemanticPayload,
)
from delta_preservation.reconcile.normalize import extract_semantic_callout
from delta_preservation.reconcile.semantic_compare import compare_semantic_callouts


def _span(text: str, *, span_id: int = 0, x0: float = 0.0) -> TextSpan:
    return TextSpan(
        text=text,
        bbox_pdf=(x0, 0.0, x0 + 10.0, 5.0),
        font_size=10.0,
        block_id=0,
        line_id=0,
        span_id=span_id,
    )



def _semantic_callout(
    *,
    family: str,
    status_state: str = "parsed",
    reason_code: str | None = None,
    detail: str | None = None,
    raw_text: str | None = None,
    normalized_text: str | None = None,
    gdt: GdtSemanticPayload | None = None,
    weld: WeldSemanticPayload | None = None,
    surface_finish: SurfaceFinishSemanticPayload | None = None,
    fit: FitSemanticPayload | None = None,
) -> SemanticCallout:
    return SemanticCallout(
        provenance=SemanticProvenance(
            authority="pdf",
            source_type="drawing_pdf",
            source_ref="pdf:block:0/line:0/span:0",
            notes=["test fixture"],
        ),
        status=SemanticParserStatus(
            state=status_state,
            parser_family=family,
            reason_code=reason_code,
            detail=detail,
        ),
        raw_text=raw_text,
        normalized_text=normalized_text,
        gdt=gdt,
        weld=weld,
        surface_finish=surface_finish,
        fit=fit,
        metadata={"authority_source": "pdf"},
    )


# GD&T comparison matrix

def test_compare_semantic_callouts_gdt_unchanged_formatting_only_variant_reports_semantic_match():
    left = _semantic_callout(
        family="gdt",
        raw_text="⌖ ⌀0.10 M A B C",
        normalized_text="⌖ ⌀0.10 M A B C",
        gdt=GdtSemanticPayload(
            frame_text="⌖ | ⌀0.10 | M | A | B | C",
            control_type="position",
            tolerance_text="⌀0.10",
            datum_refs=["A", "B", "C"],
            modifiers=["MMC"],
        ),
    )
    right = _semantic_callout(
        family="gdt",
        raw_text="⌖    ⌀0.10   M   A  B C",
        normalized_text="⌖ ⌀0.10 M A B C",
        gdt=GdtSemanticPayload(
            frame_text="⌖ | ⌀0.10 | M | A | B | C",
            control_type="position",
            tolerance_text="⌀0.10",
            datum_refs=["A", "B", "C"],
            modifiers=["MMC"],
        ),
    )

    result = compare_semantic_callouts(left, right)

    assert result.comparable is True
    assert result.equal is True
    assert result.family == "gdt"
    assert result.mode == "semantic"
    assert result.reason_fragments == ["semantic GD&T match: position ⌀0.10 datums A,B,C modifiers MMC"]


def test_compare_semantic_callouts_gdt_changed_tolerance_reports_meaningful_delta_reason():
    left = _semantic_callout(
        family="gdt",
        normalized_text="⌖ ⌀0.10 M A B C",
        gdt=GdtSemanticPayload(
            frame_text="⌖ | ⌀0.10 | M | A | B | C",
            control_type="position",
            tolerance_text="⌀0.10",
            datum_refs=["A", "B", "C"],
            modifiers=["MMC"],
        ),
    )
    right = _semantic_callout(
        family="gdt",
        normalized_text="⌖ ⌀0.20 M A B C",
        gdt=GdtSemanticPayload(
            frame_text="⌖ | ⌀0.20 | M | A | B | C",
            control_type="position",
            tolerance_text="⌀0.20",
            datum_refs=["A", "B", "C"],
            modifiers=["MMC"],
        ),
    )

    result = compare_semantic_callouts(left, right)

    assert result.comparable is True
    assert result.equal is False
    assert result.reason_fragments == ["semantic GD&T changed: tolerance ⌀0.10 → ⌀0.20"]


# Weld comparison matrix

def test_compare_semantic_callouts_weld_unchanged_text_order_variant_reports_semantic_match():
    left = _semantic_callout(
        family="weld",
        raw_text="1/8 FILLET BOTH SIDES ALL AROUND 1.50-3.00 FLUSH TAIL: FIELD",
        normalized_text="1/8 FILLET BOTH SIDES ALL AROUND 1.50-3.00 FLUSH TAIL: FIELD",
        weld=WeldSemanticPayload(
            process="fillet",
            size="1/8",
            contour="flush",
            side="both_sides",
            length="1.50",
            pitch="3.00",
            tail="FIELD",
            all_around=True,
        ),
    )
    right = _semantic_callout(
        family="weld",
        raw_text="1/8 FILLET ALL AROUND BOTH SIDES 1.50-3.00 FLUSH TAIL: FIELD",
        normalized_text="1/8 FILLET ALL AROUND BOTH SIDES 1.50-3.00 FLUSH TAIL: FIELD",
        weld=WeldSemanticPayload(
            process="fillet",
            size="1/8",
            contour="flush",
            side="both_sides",
            length="1.50",
            pitch="3.00",
            tail="FIELD",
            all_around=True,
        ),
    )

    result = compare_semantic_callouts(left, right)

    assert result.comparable is True
    assert result.equal is True
    assert result.family == "weld"
    assert result.reason_fragments == [
        "semantic weld match: fillet size 1/8 both_sides all-around length 1.50 pitch 3.00 contour flush tail FIELD"
    ]


def test_compare_semantic_callouts_weld_changed_size_reports_meaningful_delta_reason():
    left = _semantic_callout(
        family="weld",
        weld=WeldSemanticPayload(
            process="fillet",
            size="1/8",
            contour="flush",
            side="both_sides",
            length="1.50",
            pitch="3.00",
            tail="FIELD",
            all_around=True,
        ),
    )
    right = _semantic_callout(
        family="weld",
        weld=WeldSemanticPayload(
            process="fillet",
            size="3/16",
            contour="flush",
            side="both_sides",
            length="1.50",
            pitch="3.00",
            tail="FIELD",
            all_around=True,
        ),
    )

    result = compare_semantic_callouts(left, right)

    assert result.comparable is True
    assert result.equal is False
    assert result.reason_fragments == ["semantic weld changed: size 1/8 → 3/16"]


def test_compare_semantic_callouts_weld_error_parser_state_reports_fallback_reason_fragment():
    left = _semantic_callout(
        family="weld",
        status_state="error",
        reason_code="weld_malformed",
        detail="recognized weld callout is missing a parseable size token before the weld type",
    )
    right = _semantic_callout(
        family="weld",
        weld=WeldSemanticPayload(
            process="fillet",
            size="1/8",
            contour=None,
            side=None,
            length=None,
            pitch=None,
            tail=None,
            all_around=None,
        ),
    )

    result = compare_semantic_callouts(left, right)

    assert result.comparable is False
    assert result.equal is None
    assert result.mode == "fallback"
    assert result.reason_fragments == ["semantic comparison fallback: left semantic state error/weld_malformed"]


# Surface finish comparison matrix

def test_compare_semantic_callouts_surface_finish_unchanged_formatting_only_variant_reports_semantic_match():
    left = _semantic_callout(
        family="surface_finish",
        raw_text="Ra 3.2 um",
        normalized_text="Ra 3.2 um",
        surface_finish=SurfaceFinishSemanticPayload(
            canonical_text="Ra 3.2 um",
            roughness_value="3.2",
            units="um",
            value_micrometers="3.2",
            indicator="Ra",
        ),
    )
    right = _semantic_callout(
        family="surface_finish",
        raw_text="3.2 Ra",
        normalized_text="Ra 3.2 um",
        surface_finish=SurfaceFinishSemanticPayload(
            canonical_text="Ra 3.2 um",
            roughness_value="3.2",
            units="um",
            value_micrometers="3.2",
            indicator="Ra",
        ),
    )

    result = compare_semantic_callouts(left, right)

    assert result.comparable is True
    assert result.equal is True
    assert result.family == "surface_finish"
    assert result.reason_fragments == ["semantic surface finish match: Ra 3.2 um"]


def test_compare_semantic_callouts_surface_finish_changed_roughness_reports_meaningful_delta_reason():
    left = _semantic_callout(
        family="surface_finish",
        surface_finish=SurfaceFinishSemanticPayload(
            canonical_text="Ra 3.2 um",
            roughness_value="3.2",
            units="um",
            value_micrometers="3.2",
            indicator="Ra",
        ),
    )
    right = _semantic_callout(
        family="surface_finish",
        surface_finish=SurfaceFinishSemanticPayload(
            canonical_text="Ra 1.6 um",
            roughness_value="1.6",
            units="um",
            value_micrometers="1.6",
            indicator="Ra",
        ),
    )

    result = compare_semantic_callouts(left, right)

    assert result.comparable is True
    assert result.equal is False
    assert result.reason_fragments == ["semantic surface finish changed: roughness Ra 3.2 um → Ra 1.6 um"]


def test_compare_semantic_callouts_surface_finish_empty_parser_state_reports_fallback_reason_fragment():
    left = _semantic_callout(
        family="surface_finish",
        status_state="empty",
        reason_code="surface_finish_no_match",
        detail="text did not match the bounded GD&T, weld, surface finish, or fit grammar",
        raw_text="FLAG NOTE 12",
        normalized_text="FLAG NOTE 12",
    )
    right = _semantic_callout(
        family="surface_finish",
        surface_finish=SurfaceFinishSemanticPayload(
            canonical_text="Ra 3.2 um",
            roughness_value="3.2",
            units="um",
            value_micrometers="3.2",
            indicator="Ra",
        ),
    )

    result = compare_semantic_callouts(left, right)

    assert result.comparable is False
    assert result.equal is None
    assert result.mode == "fallback"
    assert result.reason_fragments == [
        "semantic comparison fallback: left semantic state empty/surface_finish_no_match"
    ]


# Fit comparison matrix

def test_compare_semantic_callouts_fit_unchanged_spacing_variant_reports_semantic_match():
    left = _semantic_callout(
        family="fit",
        raw_text="H7/p6",
        normalized_text="H7/p6",
        fit=FitSemanticPayload(
            canonical_text="H7/p6",
            fit_class="H7/p6",
            hole_class="H7",
            shaft_class="p6",
            basis="hole_basis",
            standard_hint="iso_limits_and_fits",
        ),
    )
    right = _semantic_callout(
        family="fit",
        raw_text="H7 / p6",
        normalized_text="H7/p6",
        fit=FitSemanticPayload(
            canonical_text="H7/p6",
            fit_class="H7/p6",
            hole_class="H7",
            shaft_class="p6",
            basis="hole_basis",
            standard_hint="iso_limits_and_fits",
        ),
    )

    result = compare_semantic_callouts(left, right)

    assert result.comparable is True
    assert result.equal is True
    assert result.family == "fit"
    assert result.reason_fragments == ["semantic fit match: H7/p6 (hole_basis)"]


def test_compare_semantic_callouts_fit_changed_class_reports_meaningful_delta_reason():
    left = _semantic_callout(
        family="fit",
        fit=FitSemanticPayload(
            canonical_text="H7/p6",
            fit_class="H7/p6",
            hole_class="H7",
            shaft_class="p6",
            basis="hole_basis",
            standard_hint="iso_limits_and_fits",
        ),
    )
    right = _semantic_callout(
        family="fit",
        fit=FitSemanticPayload(
            canonical_text="H7/g6",
            fit_class="H7/g6",
            hole_class="H7",
            shaft_class="g6",
            basis="hole_basis",
            standard_hint="iso_limits_and_fits",
        ),
    )

    result = compare_semantic_callouts(left, right)

    assert result.comparable is True
    assert result.equal is False
    assert result.reason_fragments == ["semantic fit changed: fit class H7/p6 → H7/g6"]


# Cross-family and missing/fallback coverage

def test_compare_semantic_callouts_missing_both_semantics_reports_fallback():
    result = compare_semantic_callouts(None, None)

    assert result.comparable is False
    assert result.equal is None
    assert result.mode == "fallback"
    assert result.family is None
    assert result.reason_fragments == ["semantic comparison unavailable: both semantic callouts missing; fall back to numeric/text comparison"]


def test_compare_semantic_callouts_missing_one_semantic_callout_reports_fallback():
    right = _semantic_callout(
        family="fit",
        fit=FitSemanticPayload(
            canonical_text="H7/p6",
            fit_class="H7/p6",
            hole_class="H7",
            shaft_class="p6",
            basis="hole_basis",
            standard_hint="iso_limits_and_fits",
        ),
    )

    result = compare_semantic_callouts(None, right)

    assert result.comparable is False
    assert result.equal is None
    assert result.mode == "fallback"
    assert result.reason_fragments == ["semantic comparison fallback: left semantic callout missing"]


def test_compare_semantic_callouts_mixed_family_reports_incompatible_reason_without_exception():
    left = _semantic_callout(
        family="gdt",
        gdt=GdtSemanticPayload(
            frame_text="⌖ | ⌀0.10 | M | A | B | C",
            control_type="position",
            tolerance_text="⌀0.10",
            datum_refs=["A", "B", "C"],
            modifiers=["MMC"],
        ),
    )
    right = _semantic_callout(
        family="fit",
        fit=FitSemanticPayload(
            canonical_text="H7/p6",
            fit_class="H7/p6",
            hole_class="H7",
            shaft_class="p6",
            basis="hole_basis",
            standard_hint="iso_limits_and_fits",
        ),
    )

    result = compare_semantic_callouts(left, right)

    assert result.comparable is False
    assert result.equal is False
    assert result.mode == "incompatible"


# GD&T word-form vs symbol-form comparison (GDT-02)

def test_compare_semantic_callouts_gdt_word_form_and_symbol_form_report_semantic_match():
    word_form = extract_semantic_callout(
        pdf_spans=[_span("flatness 0.05", span_id=0, x0=10.0)],
    )
    symbol_form = extract_semantic_callout(
        pdf_spans=[_span("⏥", span_id=0, x0=10.0), _span("0.05", span_id=1, x0=20.0)],
    )

    assert word_form.status.state == "parsed", (
        f"word_form parse failed: {word_form.status.reason_code} — {word_form.status.detail}"
    )
    assert symbol_form.status.state == "parsed"

    result = compare_semantic_callouts(word_form, symbol_form)

    assert result.comparable is True
    assert result.equal is True
    assert result.family == "gdt"
    assert result.mode == "semantic"


# GD&T compartment-aware comparison (GDT-03)

def test_compare_semantic_callouts_gdt_compartment_count_mismatch_reports_change():
    from delta_preservation.types import GdtCompartment
    left = _semantic_callout(
        family="gdt",
        gdt=GdtSemanticPayload(
            frame_text="⌓ | 0.05 | D | B | C",
            control_type="profile_of_a_surface",
            tolerance_text="0.05",
            datum_refs=["D", "B", "C"],
            modifiers=[],
            compartments=[
                GdtCompartment(control_type="profile_of_a_surface", tolerance_text="0.05", datum_refs=["D", "B", "C"], modifiers=[]),
                GdtCompartment(control_type="profile_of_a_surface", tolerance_text="0.01", datum_refs=["D"], modifiers=[]),
            ],
        ),
    )
    right = _semantic_callout(
        family="gdt",
        gdt=GdtSemanticPayload(
            frame_text="⌓ | 0.05 | D | B | C",
            control_type="profile_of_a_surface",
            tolerance_text="0.05",
            datum_refs=["D", "B", "C"],
            modifiers=[],
            compartments=[
                GdtCompartment(control_type="profile_of_a_surface", tolerance_text="0.05", datum_refs=["D", "B", "C"], modifiers=[]),
            ],
        ),
    )

    result = compare_semantic_callouts(left, right)

    assert result.comparable is True
    assert result.equal is False
    assert result.family == "gdt"
    assert result.mode == "semantic"
    assert any("semantic GD&T changed: compartments" in f for f in result.reason_fragments)


def test_compare_semantic_callouts_gdt_identical_composite_payloads_report_semantic_match():
    from delta_preservation.types import GdtCompartment
    compartments = [
        GdtCompartment(control_type="profile_of_a_surface", tolerance_text="0.05", datum_refs=["D", "B", "C"], modifiers=[]),
        GdtCompartment(control_type="profile_of_a_surface", tolerance_text="0.01", datum_refs=["D"], modifiers=[]),
    ]
    left = _semantic_callout(
        family="gdt",
        gdt=GdtSemanticPayload(
            frame_text="⌓ | 0.05 | D | B | C",
            control_type="profile_of_a_surface",
            tolerance_text="0.05",
            datum_refs=["D", "B", "C"],
            modifiers=[],
            compartments=compartments,
        ),
    )
    right = _semantic_callout(
        family="gdt",
        gdt=GdtSemanticPayload(
            frame_text="⌓ | 0.05 | D | B | C",
            control_type="profile_of_a_surface",
            tolerance_text="0.05",
            datum_refs=["D", "B", "C"],
            modifiers=[],
            compartments=compartments,
        ),
    )

    result = compare_semantic_callouts(left, right)

    assert result.comparable is True
    assert result.equal is True
    assert result.family == "gdt"
    assert result.mode == "semantic"


def test_compare_semantic_callouts_gdt_single_compartment_unchanged_still_reports_semantic_match():
    # Regression: existing single-compartment equality is unaffected by compartments field
    left = _semantic_callout(
        family="gdt",
        gdt=GdtSemanticPayload(
            frame_text="⌖ | ⌀0.10 | M | A | B | C",
            control_type="position",
            tolerance_text="⌀0.10",
            datum_refs=["A", "B", "C"],
            modifiers=["MMC"],
        ),
    )
    right = _semantic_callout(
        family="gdt",
        gdt=GdtSemanticPayload(
            frame_text="⌖ | ⌀0.10 | M | A | B | C",
            control_type="position",
            tolerance_text="⌀0.10",
            datum_refs=["A", "B", "C"],
            modifiers=["MMC"],
        ),
    )

    result = compare_semantic_callouts(left, right)

    assert result.comparable is True
    assert result.equal is True
    assert result.family == "gdt"
    assert result.mode == "semantic"


# WR-03 regression: GdtCompartment.tolerance_text is Optional[str]; _compare_gdt()
# must not crash when it is None.

def test_compare_semantic_callouts_gdt_compartment_with_null_tolerance_does_not_crash():
    """Schema-valid GdtCompartment(tolerance_text=None) must not raise AttributeError."""
    from delta_preservation.types import GdtCompartment

    left = _semantic_callout(
        family="gdt",
        gdt=GdtSemanticPayload(
            frame_text="⌖ | ⌀0.10 | A",
            control_type="position",
            tolerance_text=None,
            datum_refs=["A"],
            modifiers=[],
            compartments=[
                GdtCompartment(control_type="position", tolerance_text=None, datum_refs=["A"], modifiers=[]),
            ],
        ),
    )
    right = _semantic_callout(
        family="gdt",
        gdt=GdtSemanticPayload(
            frame_text="⌖ | ⌀0.10 | A",
            control_type="position",
            tolerance_text=None,
            datum_refs=["A"],
            modifiers=[],
            compartments=[
                GdtCompartment(control_type="position", tolerance_text=None, datum_refs=["A"], modifiers=[]),
            ],
        ),
    )

    # Must not raise AttributeError: 'NoneType' object has no attribute 'startswith'
    result = compare_semantic_callouts(left, right)

    assert result.comparable is True
    assert result.family == "gdt"
