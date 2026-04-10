"""Regression coverage for deterministic snippet-evaluation rules."""

from __future__ import annotations

import pytest

from delta_preservation.evaluation.conformance import evaluate_packet_against_truth
from delta_preservation.evaluation.contracts import GroundTruthPacket
from delta_preservation.types import DeltaItem, Evidence


def _truth_packet(*characteristics: dict[str, object]) -> GroundTruthPacket:
    return GroundTruthPacket.model_validate(
        {
            "part_name": "Snippet Fixture",
            "general_notes": "",
            "characteristics": list(characteristics),
        }
    )


def _evidence(bbox: list[float] | None) -> Evidence | None:
    if bbox is None:
        return None
    return Evidence(page=0, bbox=bbox, image_path="snippet.png")


def _packet_item(
    *,
    char_no: int | None,
    status: str,
    requirement_revB: str | None,
    revA_bbox: list[float] | None,
    revB_bbox: list[float] | None,
    snippet_rule_family: str = "single_callout",
) -> DeltaItem:
    return DeltaItem(
        char_no=char_no,
        status=status,
        confidence=0.9,
        reasons=["snippet fixture"],
        scores={"location": 1.0, "text": 1.0, "context": 1.0},
        revA=_evidence(revA_bbox),
        revB=_evidence(revB_bbox),
        requirement_revB=requirement_revB,
        snippet_rule_family=snippet_rule_family,
    )


def test_single_callout_passes_when_truth_center_is_inside_safe_zone() -> None:
    truth_packet = _truth_packet(
        {
            "char_no": 1,
            "classification": "unchanged",
            "requirement_revB": "LENGTH 12.0",
            "snippet_center_revA": [50.0, 50.0],
            "snippet_center_revB": [110.0, 50.0],
        }
    )

    evaluation = evaluate_packet_against_truth(
        [
            _packet_item(
                char_no=1,
                status="unchanged",
                requirement_revB="LENGTH 12.0",
                revA_bbox=[20.0, 20.0, 80.0, 80.0],
                revB_bbox=[80.0, 20.0, 140.0, 80.0],
            )
        ],
        truth_packet,
    )[0]

    assert evaluation.status == "conforming"
    assert evaluation.snippet_conforms is True
    assert evaluation.mismatches == []


def test_single_callout_fails_when_truth_center_hits_edge_guard() -> None:
    truth_packet = _truth_packet(
        {
            "char_no": 2,
            "classification": "unchanged",
            "requirement_revB": "LENGTH 12.0",
            "snippet_center_revA": [34.0, 50.0],
            "snippet_center_revB": [110.0, 50.0],
        }
    )

    evaluation = evaluate_packet_against_truth(
        [
            _packet_item(
                char_no=2,
                status="unchanged",
                requirement_revB="LENGTH 12.0",
                revA_bbox=[20.0, 20.0, 80.0, 80.0],
                revB_bbox=[80.0, 20.0, 140.0, 80.0],
            )
        ],
        truth_packet,
    )[0]

    assert evaluation.status == "review_needed"
    assert evaluation.snippet_conforms is False
    assert [mismatch.code for mismatch in evaluation.mismatches] == ["snippet_edge_guard_revA"]


def test_grouped_callout_passes_with_looser_contracted_union_rule() -> None:
    truth_packet = _truth_packet(
        {
            "char_no": 3,
            "classification": "unchanged",
            "requirement_revB": "PROFILE 0.2 A B",
            "snippet_center_revA": [27.0, 40.0],
            "snippet_center_revB": [87.0, 40.0],
        }
    )

    evaluation = evaluate_packet_against_truth(
        [
            _packet_item(
                char_no=3,
                status="unchanged",
                requirement_revB="PROFILE 0.2 A B",
                revA_bbox=[20.0, 20.0, 120.0, 60.0],
                revB_bbox=[80.0, 20.0, 180.0, 60.0],
                snippet_rule_family="grouped_callout",
            )
        ],
        truth_packet,
    )[0]

    assert evaluation.status == "conforming"
    assert evaluation.snippet_conforms is True
    assert evaluation.mismatches == []


def test_null_truth_center_exempts_that_side_from_snippet_requirements() -> None:
    truth_packet = _truth_packet(
        {
            "classification": "added",
            "requirement_revB": "NEW SLOT DETAIL",
            "snippet_center_revA": None,
            "snippet_center_revB": [95.0, 40.0],
        }
    )

    evaluation = evaluate_packet_against_truth(
        [
            _packet_item(
                char_no=101,
                status="added",
                requirement_revB="NEW SLOT DETAIL",
                revA_bbox=None,
                revB_bbox=[60.0, 20.0, 130.0, 60.0],
            )
        ],
        truth_packet,
    )[0]

    assert evaluation.status == "conforming"
    assert evaluation.matched_truth_char_no == "added:0"
    assert evaluation.snippet_conforms is True
    assert evaluation.mismatches == []


@pytest.mark.parametrize(
    ("revA_bbox", "revB_bbox", "expected_code"),
    [
        (None, [80.0, 20.0, 140.0, 80.0], "snippet_missing_revA"),
        ([20.0, 20.0, 80.0, 80.0], None, "snippet_missing_revB"),
    ],
)
def test_missing_evidence_emits_side_specific_snippet_mismatch(
    revA_bbox: list[float] | None,
    revB_bbox: list[float] | None,
    expected_code: str,
) -> None:
    truth_packet = _truth_packet(
        {
            "char_no": 4,
            "classification": "unchanged",
            "requirement_revB": "LENGTH 12.0",
            "snippet_center_revA": [50.0, 50.0],
            "snippet_center_revB": [110.0, 50.0],
        }
    )

    evaluation = evaluate_packet_against_truth(
        [
            _packet_item(
                char_no=4,
                status="unchanged",
                requirement_revB="LENGTH 12.0",
                revA_bbox=revA_bbox,
                revB_bbox=revB_bbox,
            )
        ],
        truth_packet,
    )[0]

    assert evaluation.status == "review_needed"
    assert evaluation.snippet_conforms is False
    assert [mismatch.code for mismatch in evaluation.mismatches] == [expected_code]


def test_final_mismatch_order_is_classification_then_requirement_then_snippet() -> None:
    truth_packet = _truth_packet(
        {
            "char_no": 5,
            "classification": "changed",
            "requirement_revB": "LENGTH 10.0",
            "snippet_center_revA": [50.0, 50.0],
            "snippet_center_revB": [110.0, 50.0],
        }
    )

    evaluation = evaluate_packet_against_truth(
        [
            _packet_item(
                char_no=5,
                status="unchanged",
                requirement_revB="LENGTH 8.0",
                revA_bbox=[20.0, 20.0, 80.0, 80.0],
                revB_bbox=None,
            )
        ],
        truth_packet,
    )[0]

    assert evaluation.status == "review_needed"
    assert evaluation.snippet_conforms is False
    assert [mismatch.code for mismatch in evaluation.mismatches] == [
        "classification_mismatch",
        "requirement_mismatch",
        "snippet_missing_revB",
    ]
