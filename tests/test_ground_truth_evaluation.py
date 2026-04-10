"""Regression coverage for packet-side ground-truth evaluation."""

from __future__ import annotations

from delta_preservation.evaluation.conformance import evaluate_packet_against_truth
from delta_preservation.evaluation.contracts import GroundTruthPacket
from delta_preservation.reconcile.normalize import extract_semantic_callout
from delta_preservation.types import DeltaItem


def _truth_packet(*characteristics: dict[str, object]) -> GroundTruthPacket:
    return GroundTruthPacket.model_validate(
        {
            "part_name": "Fixture Part",
            "general_notes": "",
            "characteristics": list(characteristics),
        }
    )


def _packet_item(
    *,
    char_no: int | None,
    status: str,
    requirement_revB: str | None,
    semantic_callout=None,
) -> DeltaItem:
    return DeltaItem(
        char_no=char_no,
        status=status,
        confidence=0.9,
        reasons=["test fixture"],
        scores={"location": 1.0, "text": 1.0, "context": 1.0},
        requirement_revB=requirement_revB,
        semantic_callout=semantic_callout,
    )


def test_matching_unchanged_row_is_pending_snippet_when_classification_and_requirement_align() -> None:
    truth_packet = _truth_packet(
        {
            "char_no": 1,
            "classification": "unchanged",
            "requirement_revB": "LENGTH 12.0",
            "snippet_center_revA": [10.0, 10.0],
            "snippet_center_revB": [12.0, 12.0],
        }
    )

    evaluation = evaluate_packet_against_truth(
        [_packet_item(char_no=1, status="unchanged", requirement_revB="LENGTH 12.0")],
        truth_packet,
    )[0]

    assert evaluation.status == "pending_snippet"
    assert evaluation.matched_truth_char_no == 1
    assert evaluation.classification_conforms is True
    assert evaluation.requirement_conforms is True
    assert evaluation.mismatches == []


def test_classification_mismatch_marks_row_review_needed() -> None:
    truth_packet = _truth_packet(
        {
            "char_no": 8,
            "classification": "changed",
            "requirement_revB": "LENGTH 12.0",
            "snippet_center_revA": [10.0, 10.0],
            "snippet_center_revB": [12.0, 12.0],
        }
    )

    evaluation = evaluate_packet_against_truth(
        [_packet_item(char_no=8, status="unchanged", requirement_revB="LENGTH 12.0")],
        truth_packet,
    )[0]

    assert evaluation.status == "review_needed"
    assert evaluation.classification_conforms is False
    assert evaluation.mismatches[0].code == "classification_mismatch"


def test_removed_rows_allow_null_truth_requirement_without_requirement_mismatch() -> None:
    truth_packet = _truth_packet(
        {
            "char_no": 3,
            "classification": "removed",
            "requirement_revB": None,
            "snippet_center_revA": [24.0, 18.0],
            "snippet_center_revB": None,
        }
    )

    evaluation = evaluate_packet_against_truth(
        [_packet_item(char_no=3, status="removed", requirement_revB=None)],
        truth_packet,
    )[0]

    assert evaluation.status == "pending_snippet"
    assert evaluation.classification_conforms is True
    assert evaluation.requirement_conforms is True
    assert evaluation.mismatches == []


def test_added_truth_rows_are_matched_as_unordered_pool() -> None:
    truth_packet = _truth_packet(
        {
            "classification": "added",
            "requirement_revB": "NEW SLOT DETAIL",
            "snippet_center_revA": None,
            "snippet_center_revB": [10.0, 10.0],
        },
        {
            "classification": "added",
            "requirement_revB": "NEW HOLE PATTERN",
            "snippet_center_revA": None,
            "snippet_center_revB": [20.0, 20.0],
        },
    )

    evaluations = evaluate_packet_against_truth(
        [
            _packet_item(char_no=101, status="added", requirement_revB="NEW HOLE PATTERN"),
            _packet_item(char_no=102, status="added", requirement_revB="NEW SLOT DETAIL"),
        ],
        truth_packet,
    )

    assert [evaluation.matched_truth_char_no for evaluation in evaluations] == ["added:1", "added:0"]
    assert [evaluation.status for evaluation in evaluations] == ["pending_snippet", "pending_snippet"]
    assert all(not evaluation.mismatches for evaluation in evaluations)


def test_semantic_equivalence_precedes_text_fallback_for_requirement_matching() -> None:
    truth_requirement = "H7/p6"
    packet_requirement = "H7 / p6"
    truth_packet = _truth_packet(
        {
            "char_no": 14,
            "classification": "unchanged",
            "requirement_revB": truth_requirement,
            "snippet_center_revA": [10.0, 10.0],
            "snippet_center_revB": [12.0, 12.0],
        }
    )

    evaluation = evaluate_packet_against_truth(
        [
            _packet_item(
                char_no=14,
                status="unchanged",
                requirement_revB=packet_requirement,
                semantic_callout=extract_semantic_callout(pdf_spans=[], form3_requirement=packet_requirement),
            )
        ],
        truth_packet,
    )[0]

    assert evaluation.status == "pending_snippet"
    assert evaluation.requirement_conforms is True
    assert evaluation.mismatches == []


def test_normalized_text_fallback_allows_equivalent_requirement_when_semantics_unavailable() -> None:
    truth_packet = _truth_packet(
        {
            "char_no": 21,
            "classification": "unchanged",
            "requirement_revB": "FLAG NOTE 12",
            "snippet_center_revA": [10.0, 10.0],
            "snippet_center_revB": [12.0, 12.0],
        }
    )

    evaluation = evaluate_packet_against_truth(
        [_packet_item(char_no=21, status="unchanged", requirement_revB=" flag   note 12 ", semantic_callout=None)],
        truth_packet,
    )[0]

    assert evaluation.status == "pending_snippet"
    assert evaluation.requirement_conforms is True
    assert evaluation.mismatches == []


def test_missing_requirement_evidence_emits_requirement_missing_and_blocks_auto_pass() -> None:
    truth_packet = _truth_packet(
        {
            "char_no": 34,
            "classification": "unchanged",
            "requirement_revB": "PROFILE 0.2 A B",
            "snippet_center_revA": [10.0, 10.0],
            "snippet_center_revB": [12.0, 12.0],
        }
    )

    evaluation = evaluate_packet_against_truth(
        [_packet_item(char_no=34, status="unchanged", requirement_revB=None)],
        truth_packet,
    )[0]

    assert evaluation.status == "review_needed"
    assert evaluation.requirement_conforms is False
    assert [mismatch.code for mismatch in evaluation.mismatches] == ["requirement_missing"]
