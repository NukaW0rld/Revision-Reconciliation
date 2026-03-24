import json

from delta_preservation.types import (
    DeltaItem,
    DeltaPacket,
    SemanticCallout,
    SemanticParserStatus,
    SemanticProvenance,
    GdtSemanticPayload,
    SurfaceFinishSemanticPayload,
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


def test_delta_packet_serialization_preserves_populated_semantic_callout():
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
                detail="feature control frame normalized",
            ),
            raw_text="⟂ | 0.05 | A | B",
            normalized_text="perpendicularity 0.05 A B",
            gdt=GdtSemanticPayload(
                frame_text="⟂ | 0.05 | A | B",
                control_type="perpendicularity",
                datum_refs=["A", "B"],
                modifiers=["none"],
            ),
            surface_finish=SurfaceFinishSemanticPayload(
                roughness_value="63",
                units="microinch",
            ),
            metadata={"extraction_version": "v1"},
        ),
    )

    packet = DeltaPacket(run_id="run-002", inputs={"dpi": "300"}, items=[item])
    payload = packet.model_dump(exclude_none=True)

    semantic = payload["items"][0]["semantic_callout"]
    assert semantic["provenance"]["authority"] == "pdf"
    assert semantic["status"]["state"] == "parsed"
    assert semantic["status"]["parser_family"] == "gdt"
    assert semantic["gdt"]["control_type"] == "perpendicularity"
    assert semantic["surface_finish"]["units"] == "microinch"
    assert semantic["metadata"] == {"extraction_version": "v1"}

    round_trip = DeltaPacket.model_validate_json(packet.model_dump_json(exclude_none=True))
    assert round_trip.items[0].semantic_callout is not None
    assert round_trip.items[0].semantic_callout.provenance.source_ref == "page:1/span:42"
    assert round_trip.items[0].semantic_callout.gdt.datum_refs == ["A", "B"]
    assert round_trip.items[0].semantic_callout.surface_finish.roughness_value == "63"
    assert round_trip.items[0].scores == {"location": 0.88, "text": 0.7, "context": 0.81}


def test_semantic_callout_empty_status_is_explicit_and_json_stable():
    item = DeltaItem(
        char_no=18,
        status="uncertain",
        confidence=0.42,
        reasons=["Semantic parser deferred for this callout family"],
        scores={"location": 0.5, "text": 0.2, "context": 0.4},
        semantic_callout=SemanticCallout(
            provenance=SemanticProvenance(
                authority="none",
                source_type="none",
                source_ref="form3:char:18",
                notes=["No authoritative drawing semantic span matched"],
            ),
            status=SemanticParserStatus(
                state="not_implemented",
                parser_family="fit",
                reason_code="not_implemented_in_slice",
                detail="Fit parsing is deferred to a later slice",
            ),
            raw_text="FN1",
            normalized_text=None,
            metadata={"inspection_surface": "delta_packet.json"},
        ),
    )

    payload_json = item.model_dump_json(exclude_none=True)
    payload = json.loads(payload_json)

    assert payload["semantic_callout"]["status"]["reason_code"] == "not_implemented_in_slice"
    assert payload["semantic_callout"]["status"]["parser_family"] == "fit"
    assert payload["semantic_callout"]["provenance"]["notes"] == [
        "No authoritative drawing semantic span matched"
    ]
    assert "gdt" not in payload["semantic_callout"]
    assert "weld" not in payload["semantic_callout"]
    assert payload["reasons"] == ["Semantic parser deferred for this callout family"]

    reparsed = DeltaItem.model_validate_json(payload_json)
    assert reparsed.semantic_callout is not None
    assert reparsed.semantic_callout.status.state == "not_implemented"
    assert reparsed.semantic_callout.status.reason_code == "not_implemented_in_slice"
    assert reparsed.semantic_callout.metadata["inspection_surface"] == "delta_packet.json"
