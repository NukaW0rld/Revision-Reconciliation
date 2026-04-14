import json

from delta_preservation.types import (
    DeltaItem,
    DeltaPacket,
    SemanticCallout,
    SemanticParserStatus,
    SemanticProvenance,
    FitSemanticPayload,
    GdtSemanticPayload,
    SurfaceFinishSemanticPayload,
    WeldSemanticPayload,
)


def test_delta_packet_serialization_omits_semantic_callout_when_absent():
    item = DeltaItem(
        char_no=7,
        status="unchanged",
        confidence=0.98,
        reasons=["Location and text matched"],
        scores={"location": 0.98, "text": 0.96, "context": 0.94},
        revA=None,
        revB=None,
    )

    packet = DeltaPacket(run_id="run-001", inputs={"dpi": "300"}, items=[item])
    payload = packet.model_dump(exclude_none=True)

    assert payload["items"][0]["char_no"] == 7
    assert payload["items"][0]["status"] == "unchanged"
    assert payload["items"][0]["scores"]["location"] == 0.98
    assert "semantic_callout" not in payload["items"][0]

    reparsed = DeltaPacket.model_validate(payload)
    assert reparsed.items[0].semantic_callout is None
    assert reparsed.items[0].reasons == ["Location and text matched"]



def test_delta_packet_serialization_preserves_parsed_gdt_semantic_callout():
    item = DeltaItem(
        char_no=12,
        status="changed",
        confidence=0.84,
        reasons=["PDF callout changed between revisions"],
        scores={"location": 0.88, "text": 0.7, "context": 0.81},
        semantic_callout=SemanticCallout(
            provenance=SemanticProvenance(
                authority="pdf",
                source_type="drawing_pdf",
                source_ref="page:1/span:42",
                notes=["pdf span selected over conflicting Form 3 wording"],
            ),
            status=SemanticParserStatus(
                state="parsed",
                parser_family="gdt",
                reason_code=None,
                detail="parsed feature control frame from PDF spans",
            ),
            raw_text="⌖ ⌀0.10 M A B C",
            normalized_text="⌖ ⌀0.10 M A B C",
            gdt=GdtSemanticPayload(
                frame_text="⌖ | ⌀0.10 | M | A | B | C",
                control_type="position",
                tolerance_text="⌀0.10",
                datum_refs=["A", "B", "C"],
                modifiers=["MMC"],
            ),
            metadata={"extraction_version": "v2", "authority_source": "pdf"},
        ),
    )

    packet = DeltaPacket(run_id="run-002", inputs={"dpi": "300"}, items=[item])
    payload = packet.model_dump(exclude_none=True)

    semantic = payload["items"][0]["semantic_callout"]
    assert semantic["provenance"]["authority"] == "pdf"
    assert semantic["status"]["state"] == "parsed"
    assert semantic["status"]["parser_family"] == "gdt"
    assert semantic["gdt"]["control_type"] == "position"
    assert semantic["gdt"]["tolerance_text"] == "⌀0.10"
    assert semantic["gdt"]["datum_refs"] == ["A", "B", "C"]
    assert semantic["metadata"] == {"extraction_version": "v2", "authority_source": "pdf"}
    assert "weld" not in semantic
    assert "surface_finish" not in semantic
    assert "fit" not in semantic

    round_trip = DeltaPacket.model_validate_json(packet.model_dump_json(exclude_none=True))
    assert round_trip.items[0].semantic_callout is not None
    assert round_trip.items[0].semantic_callout.provenance.source_ref == "page:1/span:42"
    assert round_trip.items[0].semantic_callout.gdt.datum_refs == ["A", "B", "C"]
    assert round_trip.items[0].semantic_callout.gdt.tolerance_text == "⌀0.10"
    assert round_trip.items[0].scores == {"location": 0.88, "text": 0.7, "context": 0.81}



def test_delta_packet_serialization_preserves_parsed_weld_semantic_callout_only():
    item = DeltaItem(
        char_no=19,
        status="changed",
        confidence=0.79,
        reasons=["Authoritative PDF weld callout changed between revisions"],
        scores={"location": 0.83, "text": 0.71, "context": 0.8},
        semantic_callout=SemanticCallout(
            provenance=SemanticProvenance(
                authority="pdf",
                source_type="drawing_pdf",
                source_ref="pdf:block:2/line:1/span:0",
                notes=["PDF-derived weld text remained authoritative over Form 3 wording"],
            ),
            status=SemanticParserStatus(
                state="parsed",
                parser_family="weld",
                reason_code=None,
                detail="parsed bounded weld callout from authoritative semantic text",
            ),
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
            metadata={"inspection_surface": "delta_packet.json", "authority_source": "pdf"},
        ),
    )

    payload = item.model_dump(exclude_none=True)
    semantic = payload["semantic_callout"]

    assert semantic["status"]["state"] == "parsed"
    assert semantic["status"]["parser_family"] == "weld"
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

    round_trip = DeltaItem.model_validate(payload)
    assert round_trip.semantic_callout is not None
    assert round_trip.semantic_callout.weld is not None
    assert round_trip.semantic_callout.weld.process == "fillet"
    assert round_trip.semantic_callout.weld.length == "1.50"
    assert round_trip.semantic_callout.weld.all_around is True



def test_delta_packet_serialization_preserves_surface_finish_semantic_callout_only():
    item = DeltaItem(
        char_no=24,
        status="changed",
        confidence=0.82,
        reasons=["Authoritative PDF surface finish callout changed between revisions"],
        scores={"location": 0.86, "text": 0.76, "context": 0.81},
        semantic_callout=SemanticCallout(
            provenance=SemanticProvenance(
                authority="pdf",
                source_type="drawing_pdf",
                source_ref="pdf:block:3/line:1/span:2",
                notes=["PDF-derived Ra callout remained authoritative over Form 3 wording"],
            ),
            status=SemanticParserStatus(
                state="parsed",
                parser_family="surface_finish",
                reason_code=None,
                detail="parsed bounded surface finish callout from authoritative semantic text",
            ),
            raw_text="Ra 3.2 um",
            normalized_text="Ra 3.2 um",
            surface_finish=SurfaceFinishSemanticPayload(
                canonical_text="Ra 3.2 um",
                roughness_value="3.2",
                units="um",
                value_micrometers="3.2",
                indicator="Ra",
            ),
            metadata={"inspection_surface": "delta_packet.json", "authority_source": "pdf"},
        ),
    )

    payload = item.model_dump(exclude_none=True)
    semantic = payload["semantic_callout"]

    assert semantic["status"]["state"] == "parsed"
    assert semantic["status"]["parser_family"] == "surface_finish"
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

    round_trip = DeltaItem.model_validate(payload)
    assert round_trip.semantic_callout is not None
    assert round_trip.semantic_callout.surface_finish is not None
    assert round_trip.semantic_callout.surface_finish.canonical_text == "Ra 3.2 um"
    assert round_trip.semantic_callout.surface_finish.value_micrometers == "3.2"



def test_delta_packet_serialization_preserves_fit_semantic_callout_only():
    item = DeltaItem(
        char_no=27,
        status="changed",
        confidence=0.8,
        reasons=["Authoritative PDF fit callout changed between revisions"],
        scores={"location": 0.84, "text": 0.74, "context": 0.79},
        semantic_callout=SemanticCallout(
            provenance=SemanticProvenance(
                authority="pdf",
                source_type="drawing_pdf",
                source_ref="pdf:block:5/line:0/span:1",
                notes=["PDF-derived fit callout remained authoritative over Form 3 wording"],
            ),
            status=SemanticParserStatus(
                state="parsed",
                parser_family="fit",
                reason_code=None,
                detail="parsed bounded fit callout from authoritative semantic text",
            ),
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
            metadata={"inspection_surface": "delta_packet.json", "authority_source": "pdf"},
        ),
    )

    payload = item.model_dump(exclude_none=True)
    semantic = payload["semantic_callout"]

    assert semantic["status"]["state"] == "parsed"
    assert semantic["status"]["parser_family"] == "fit"
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

    round_trip = DeltaItem.model_validate(payload)
    assert round_trip.semantic_callout is not None
    assert round_trip.semantic_callout.fit is not None
    assert round_trip.semantic_callout.fit.canonical_text == "H7/p6"
    assert round_trip.semantic_callout.fit.basis == "hole_basis"



def test_semantic_callout_empty_surface_finish_status_is_explicit_and_json_stable():
    item = DeltaItem(
        char_no=18,
        status="uncertain",
        confidence=0.42,
        reasons=["Semantic parser found no bounded GD&T, weld, surface finish, or fit pattern in the authoritative text"],
        scores={"location": 0.5, "text": 0.2, "context": 0.4},
        semantic_callout=SemanticCallout(
            provenance=SemanticProvenance(
                authority="pdf",
                source_type="drawing_pdf",
                source_ref="pdf:block:8/line:1/span:0",
                notes=["PDF text inspected but no bounded semantic pattern matched"],
            ),
            status=SemanticParserStatus(
                state="empty",
                parser_family="surface_finish",
                reason_code="surface_finish_no_match",
                detail="text did not match the bounded GD&T, weld, surface finish, or fit grammar",
            ),
            raw_text="FLAG NOTE 12",
            normalized_text="FLAG NOTE 12",
            metadata={"inspection_surface": "delta_packet.json", "authority_source": "pdf"},
        ),
    )

    payload_json = item.model_dump_json(exclude_none=True)
    payload = json.loads(payload_json)

    assert payload["semantic_callout"]["status"]["reason_code"] == "surface_finish_no_match"
    assert payload["semantic_callout"]["status"]["parser_family"] == "surface_finish"
    assert payload["semantic_callout"]["provenance"]["notes"] == [
        "PDF text inspected but no bounded semantic pattern matched"
    ]
    assert "gdt" not in payload["semantic_callout"]
    assert "weld" not in payload["semantic_callout"]
    assert "surface_finish" not in payload["semantic_callout"]
    assert "fit" not in payload["semantic_callout"]

    reparsed = DeltaItem.model_validate_json(payload_json)
    assert reparsed.semantic_callout is not None
    assert reparsed.semantic_callout.status.state == "empty"
    assert reparsed.semantic_callout.status.reason_code == "surface_finish_no_match"
    assert reparsed.semantic_callout.metadata["inspection_surface"] == "delta_packet.json"


# GdtSemanticPayload compartments default-factory (GDT-03)

def test_gdt_semantic_payload_compartments_default_factory():
    payload = GdtSemanticPayload(
        frame_text="⌖ | ⌀0.10 | M | A",
        control_type="position",
        tolerance_text="⌀0.10",
        datum_refs=["A"],
        modifiers=["MMC"],
    )
    assert payload.compartments == []
    serialized = payload.model_dump()
    assert serialized["compartments"] == []
