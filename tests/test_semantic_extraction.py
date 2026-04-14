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


# GD&T fixture matrix

def test_extract_semantic_callout_gdt_parsed_position_pdf_authority_overrides_conflicting_form3_text():
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


def test_extract_semantic_callout_gdt_parsed_profile_without_datums():
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


def test_extract_semantic_callout_gdt_error_malformed_frame_keeps_pdf_authority():
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


# Surface finish fixture matrix

def test_extract_semantic_callout_surface_finish_parsed_pdf_authority_overrides_conflicting_form3_text():
    semantic = extract_semantic_callout(
        pdf_spans=[
            _span("Ra", span_id=0, x0=10.0),
            _span("3.2", span_id=1, x0=20.0),
            _span("um", span_id=2, x0=30.0),
        ],
        form3_requirement="FIT H7/p6",
    )

    assert semantic.provenance.authority == "pdf"
    assert semantic.raw_text == "Ra 3.2 um"
    assert semantic.status.state == "parsed"
    assert semantic.status.parser_family == "surface_finish"
    assert semantic.status.reason_code is None
    assert semantic.status.detail == "parsed bounded surface finish callout from authoritative semantic text"
    assert semantic.metadata["authority_source"] == "pdf"
    assert semantic.metadata["conflict_detected"] == "true"
    assert semantic.surface_finish is not None
    assert semantic.surface_finish.canonical_text == "Ra 3.2 um"
    assert semantic.surface_finish.roughness_value == "3.2"
    assert semantic.surface_finish.units == "um"
    assert semantic.surface_finish.value_micrometers == "3.2"
    assert semantic.surface_finish.indicator == "Ra"
    assert semantic.gdt is None
    assert semantic.weld is None
    assert semantic.fit is None


def test_extract_semantic_callout_surface_finish_parsed_reversed_order_variant():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("3.2 Ra", span_id=0, x0=10.0)],
        form3_requirement="SURFACE ROUGHNESS",
    )

    assert semantic.status.state == "parsed"
    assert semantic.status.parser_family == "surface_finish"
    assert semantic.surface_finish is not None
    assert semantic.surface_finish.canonical_text == "Ra 3.2 um"
    assert semantic.surface_finish.roughness_value == "3.2"
    assert semantic.surface_finish.units == "um"


def test_extract_semantic_callout_surface_finish_empty_unrecognized_semantic_text():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("FLAG NOTE 12", span_id=7, x0=12.0)],
        form3_requirement="FLAG NOTE 12",
    )

    assert semantic.provenance.authority == "pdf"
    assert semantic.raw_text == "FLAG NOTE 12"
    assert semantic.normalized_text == "FLAG NOTE 12"
    assert semantic.status.state == "empty"
    assert semantic.status.parser_family == "surface_finish"
    assert semantic.status.reason_code == "surface_finish_no_match"
    assert semantic.status.detail == "text did not match the bounded GD&T, weld, surface finish, or fit grammar"
    assert semantic.gdt is None
    assert semantic.weld is None
    assert semantic.surface_finish is None
    assert semantic.fit is None
    assert semantic.metadata["authority_source"] == "pdf"
    assert semantic.metadata["dispatcher"] == "semantic_dispatch"


def test_extract_semantic_callout_surface_finish_error_missing_value():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("Ra", span_id=0, x0=10.0)],
        form3_requirement="Ra 3.2 um",
    )

    assert semantic.status.state == "error"
    assert semantic.status.parser_family == "surface_finish"
    assert semantic.status.reason_code == "surface_finish_malformed"
    assert semantic.status.detail == "recognized surface finish indicator but missing a bounded Ra roughness value"
    assert semantic.surface_finish is None
    assert semantic.fit is None


def test_extract_semantic_callout_surface_finish_error_unsupported_family():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("Rz 12.5", span_id=0, x0=10.0)],
        form3_requirement="Ra 3.2 um",
    )

    assert semantic.status.state == "error"
    assert semantic.status.parser_family == "surface_finish"
    assert semantic.status.reason_code == "surface_finish_unsupported"
    assert semantic.status.detail == "recognized surface finish callout uses unsupported roughness family: RZ"
    assert semantic.surface_finish is None


# Weld fixture matrix

def test_extract_semantic_callout_weld_parsed_fragmented_pdf_spans_with_pdf_authority():
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


def test_extract_semantic_callout_weld_error_missing_size():
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


def test_extract_semantic_callout_weld_error_unsupported_segment():
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


def test_extract_semantic_callout_weld_form3_fallback_when_pdf_spans_missing():
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


# Fit fixture matrix

def test_extract_semantic_callout_fit_parsed_pdf_authority_overrides_conflicting_form3_text():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("H7/p6", span_id=0, x0=10.0)],
        form3_requirement="Ra 3.2 um",
    )

    assert semantic.provenance.authority == "pdf"
    assert semantic.raw_text == "H7/p6"
    assert semantic.status.state == "parsed"
    assert semantic.status.parser_family == "fit"
    assert semantic.status.reason_code is None
    assert semantic.status.detail == "parsed bounded fit callout from authoritative semantic text"
    assert semantic.metadata["authority_source"] == "pdf"
    assert semantic.metadata["conflict_detected"] == "true"
    assert semantic.fit is not None
    assert semantic.fit.canonical_text == "H7/p6"
    assert semantic.fit.fit_class == "H7/p6"
    assert semantic.fit.hole_class == "H7"
    assert semantic.fit.shaft_class == "p6"
    assert semantic.fit.basis == "hole_basis"
    assert semantic.fit.standard_hint == "iso_limits_and_fits"
    assert semantic.gdt is None
    assert semantic.weld is None
    assert semantic.surface_finish is None


def test_extract_semantic_callout_fit_parsed_spacing_variant():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("H7 / p6", span_id=0, x0=10.0)],
        form3_requirement="FIT",
    )

    assert semantic.status.state == "parsed"
    assert semantic.status.parser_family == "fit"
    assert semantic.fit is not None
    assert semantic.fit.canonical_text == "H7/p6"
    assert semantic.fit.hole_class == "H7"
    assert semantic.fit.shaft_class == "p6"


def test_extract_semantic_callout_fit_error_malformed_designation():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("H7/p", span_id=0, x0=10.0)],
        form3_requirement="H7/p6",
    )

    assert semantic.status.state == "error"
    assert semantic.status.parser_family == "fit"
    assert semantic.status.reason_code == "fit_malformed"
    assert semantic.status.detail == "recognized fit designator but missing a complete paired hole/shaft class"


# GD&T word-form control name normalization (GDT-02)

def test_extract_semantic_callout_gdt_parsed_word_name_controls():
    cases = [
        ("circularity .05 A", "circularity", "0.05", ["A"]),
        ("runout 0.10 A", "circular_runout", "0.10", ["A"]),
        ("total runout 0.05 A B", "total_runout", "0.05", ["A", "B"]),
        ("perpendicularity ⌀0.10 A", "perpendicularity", "⌀0.10", ["A"]),
    ]
    for text, expected_control, expected_tol, expected_datums in cases:
        semantic = extract_semantic_callout(
            pdf_spans=[_span(text, span_id=0, x0=10.0)],
        )
        assert semantic.status.state == "parsed", (
            f"Expected parsed for {text!r}, got state={semantic.status.state!r} "
            f"reason_code={semantic.status.reason_code!r} detail={semantic.status.detail!r}"
        )
        assert semantic.status.parser_family == "gdt"
        assert semantic.gdt is not None
        assert semantic.gdt.control_type == expected_control, f"control_type mismatch for {text!r}"
        assert semantic.gdt.tolerance_text == expected_tol, f"tolerance_text mismatch for {text!r}"
        assert semantic.gdt.datum_refs == expected_datums, f"datum_refs mismatch for {text!r}"


# GD&T compact single-token parsing (GDT-01)

def test_extract_semantic_callout_gdt_parsed_compact_token_variants():
    # Compact tokens: control symbol + diameter prefix + tolerance + datum suffix, no whitespace
    compact_cases = [
        ("⌖∅0.35ABC", "position", "∅0.35", ["A", "B", "C"]),
        ("⌓0.5A", "profile_of_a_surface", "0.5", ["A"]),
        ("⏥0.2", "flatness", "0.2", []),
    ]
    for text, expected_control, expected_tol, expected_datums in compact_cases:
        semantic = extract_semantic_callout(
            pdf_spans=[_span(text, span_id=0, x0=10.0)],
        )
        assert semantic.status.state == "parsed", (
            f"Expected parsed for {text!r}, got state={semantic.status.state!r} "
            f"reason_code={semantic.status.reason_code!r} detail={semantic.status.detail!r}"
        )
        assert semantic.status.parser_family == "gdt"
        assert semantic.gdt is not None
        assert semantic.gdt.control_type == expected_control, f"control_type mismatch for {text!r}"
        assert semantic.gdt.tolerance_text == expected_tol, f"tolerance_text mismatch for {text!r}"
        assert semantic.gdt.datum_refs == expected_datums, f"datum_refs mismatch for {text!r}"

    # Regression: whitespace-tokenized path still works
    whitespace = extract_semantic_callout(
        pdf_spans=[
            _span("⌖", span_id=0, x0=10.0),
            _span("⌀0.10", span_id=1, x0=20.0),
            _span("M", span_id=2, x0=30.0),
            _span("A", span_id=3, x0=40.0),
        ],
    )
    assert whitespace.status.state == "parsed"
    assert whitespace.gdt is not None
    assert whitespace.gdt.control_type == "position"
    assert whitespace.gdt.tolerance_text == "⌀0.10"
    assert whitespace.gdt.datum_refs == ["A"]
    assert whitespace.gdt.modifiers == ["MMC"]

    # Regression: true malformed input (control symbol + datum only, no tolerance) still errors
    malformed = extract_semantic_callout(
        pdf_spans=[_span("⌖", span_id=0, x0=10.0), _span("A", span_id=1, x0=20.0)],
    )
    assert malformed.status.state == "error"
    assert malformed.status.reason_code == "gdt_malformed_frame"


# GD&T composite frame capture (GDT-03)

def test_extract_semantic_callout_gdt_parsed_composite_frame_preserves_all_compartments():
    # Standard symbol-form composite: two profile-of-a-surface compartments separated by '/'
    semantic = extract_semantic_callout(
        pdf_spans=[_span("⌓ .05 D B C / ⌓ .01 D", span_id=0, x0=10.0)],
    )
    assert semantic.status.state == "parsed", (
        f"Expected parsed, got {semantic.status.state!r} — {semantic.status.detail!r}"
    )
    assert semantic.status.parser_family == "gdt"
    assert semantic.gdt is not None
    # Primary compartment fields
    assert semantic.gdt.control_type == "profile_of_a_surface"
    assert semantic.gdt.tolerance_text == "0.05"
    assert semantic.gdt.datum_refs == ["D", "B", "C"]
    # Compartments list — both segments captured
    assert len(semantic.gdt.compartments) == 2
    assert semantic.gdt.compartments[0].control_type == "profile_of_a_surface"
    assert semantic.gdt.compartments[0].tolerance_text == "0.05"
    assert semantic.gdt.compartments[0].datum_refs == ["D", "B", "C"]
    assert semantic.gdt.compartments[1].control_type == "profile_of_a_surface"
    assert semantic.gdt.compartments[1].tolerance_text == "0.01"
    assert semantic.gdt.compartments[1].datum_refs == ["D"]


def test_extract_semantic_callout_gdt_parsed_composite_frame_word_normalized_variant():
    # Word-normalized composite: "flatness 0.05 / flatness 0.01" normalizes both segments
    semantic = extract_semantic_callout(
        pdf_spans=[_span("flatness 0.05 / flatness 0.01", span_id=0, x0=10.0)],
    )
    assert semantic.status.state == "parsed", (
        f"Expected parsed, got {semantic.status.state!r} — {semantic.status.detail!r}"
    )
    assert semantic.gdt is not None
    assert semantic.gdt.control_type == "flatness"
    assert len(semantic.gdt.compartments) == 2
    assert semantic.gdt.compartments[1].control_type == "flatness"
    assert semantic.gdt.compartments[1].tolerance_text == "0.01"


# WR-01 regression: compact single-token dashed datum refs must be preserved as
# compound tokens, not split into individual letters.

def test_extract_semantic_callout_gdt_compact_dashed_datum_preserved():
    """Compact "⌖∅0.35A-B" must produce datum_refs ["A-B"], not ["A", "B"]."""
    semantic = extract_semantic_callout(
        pdf_spans=[_span("⌖∅0.35A-B", span_id=0, x0=10.0)],
    )
    assert semantic.status.state == "parsed", (
        f"Expected parsed, got state={semantic.status.state!r} "
        f"reason_code={semantic.status.reason_code!r}"
    )
    assert semantic.gdt is not None
    assert semantic.gdt.datum_refs == ["A-B"], (
        f"Expected ['A-B'], got {semantic.gdt.datum_refs!r}"
    )


def test_extract_semantic_callout_gdt_compact_vs_whitespace_dashed_datum_match():
    """Compact '⌖∅0.35A-B' and whitespace '⌖ ∅0.35 A-B' must produce identical datum_refs."""
    compact = extract_semantic_callout(
        pdf_spans=[_span("⌖∅0.35A-B", span_id=0, x0=10.0)],
    )
    whitespace = extract_semantic_callout(
        pdf_spans=[
            _span("⌖", span_id=0, x0=10.0),
            _span("∅0.35", span_id=1, x0=20.0),
            _span("A-B", span_id=2, x0=30.0),
        ],
    )
    assert compact.status.state == "parsed"
    assert whitespace.status.state == "parsed"
    assert compact.gdt is not None and whitespace.gdt is not None
    assert compact.gdt.datum_refs == whitespace.gdt.datum_refs, (
        f"compact datum_refs {compact.gdt.datum_refs!r} != whitespace {whitespace.gdt.datum_refs!r}"
    )


# WR-02 regression: word-form normalization must not rewrite substrings inside
# unrelated words (e.g. "positioning" contains "position", "flatnessness" is a
# doubled word — neither should produce gdt_malformed_frame).

def test_extract_semantic_callout_positioning_hole_not_gdt_malformed():
    """'positioning hole' contains 'position' as a substring but must not produce a GD&T error."""
    semantic = extract_semantic_callout(
        pdf_spans=[_span("positioning hole", span_id=0, x0=10.0)],
    )
    assert semantic.status.reason_code != "gdt_malformed_frame", (
        f"False positive: 'positioning hole' produced gdt_malformed_frame"
    )


def test_extract_semantic_callout_flatnessness_not_gdt_malformed():
    """Misspelled 'flatnessness 0.1' must not produce gdt_malformed_frame."""
    semantic = extract_semantic_callout(
        pdf_spans=[_span("flatnessness 0.1", span_id=0, x0=10.0)],
    )
    assert semantic.status.reason_code != "gdt_malformed_frame", (
        f"False positive: 'flatnessness 0.1' produced gdt_malformed_frame"
    )


# Slash-family regression guards: weld and fit inputs containing '/' must not be
# captured by the composite GD&T parser path.

def test_extract_semantic_callout_weld_fraction_slash_still_parses_as_weld():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("1/8 FILLET BOTH SIDES", span_id=0, x0=10.0)],
    )
    assert semantic.status.state == "parsed"
    assert semantic.status.parser_family == "weld"
    assert semantic.weld is not None
    assert semantic.weld.process == "fillet"
    assert semantic.gdt is None


def test_extract_semantic_callout_fit_slash_still_parses_as_fit():
    semantic = extract_semantic_callout(
        pdf_spans=[_span("H7/p6", span_id=0, x0=10.0)],
    )
    assert semantic.status.state == "parsed"
    assert semantic.status.parser_family == "fit"
    assert semantic.fit is not None
    assert semantic.fit.hole_class == "H7"
    assert semantic.fit.shaft_class == "p6"
    assert semantic.gdt is None
