from delta_preservation.types import (
    FitSemanticPayload,
    GdtSemanticPayload,
    SemanticCallout,
    SemanticParserStatus,
    SemanticProvenance,
    SurfaceFinishSemanticPayload,
    WeldSemanticPayload,
)
from delta_preservation.reconcile.semantic_compare import compare_semantic_callouts



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



def test_compare_semantic_callouts_gdt_treats_formatting_only_variant_as_equal():
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



def test_compare_semantic_callouts_gdt_reports_meaningful_change_reason():
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



def test_compare_semantic_callouts_weld_treats_text_order_variant_as_equal():
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



def test_compare_semantic_callouts_weld_reports_meaningful_change_reason():
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



def test_compare_semantic_callouts_surface_finish_treats_formatting_only_variant_as_equal():
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



def test_compare_semantic_callouts_surface_finish_reports_meaningful_change_reason():
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



def test_compare_semantic_callouts_fit_treats_spacing_variant_as_equal():
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



def test_compare_semantic_callouts_fit_reports_meaningful_change_reason():
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



def test_compare_semantic_callouts_explicitly_reports_missing_semantics_as_fallback():
    result = compare_semantic_callouts(None, None)

    assert result.comparable is False
    assert result.equal is None
    assert result.mode == "fallback"
    assert result.family is None
    assert result.reason_fragments == ["semantic comparison unavailable: both semantic callouts missing; fall back to numeric/text comparison"]



def test_compare_semantic_callouts_explicitly_reports_empty_parser_state_as_fallback():
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



def test_compare_semantic_callouts_explicitly_reports_error_parser_state_as_fallback():
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



def test_compare_semantic_callouts_reports_mixed_family_incompatibility_without_exception():
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
    assert result.reason_fragments == ["semantic families differ: gdt vs fit"]
