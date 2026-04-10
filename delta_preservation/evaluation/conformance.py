"""Packet-side conformance checks against immutable ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from delta_preservation.evaluation.contracts import GroundTruthCharacteristic, GroundTruthPacket
from delta_preservation.evaluation.snippet_rules import evaluate_snippet_evidence
from delta_preservation.reconcile.normalize import extract_semantic_callout, parse_requirement
from delta_preservation.reconcile.semantic_compare import compare_semantic_callouts
from delta_preservation.types import DeltaItem, EvaluationMismatch, GroundTruthMatch, ItemEvaluation

REVIEW_NEEDED = "review_needed"
CONFORMING = "conforming"
TRUTH_AMBIGUITY_CODE = "truth_ambiguity"
ADDED_POOL_TOKEN_PREFIX = "added"


@dataclass(frozen=True)
class _TruthSelection:
    """Internal truth-row selection result for a single packet item."""

    truth_row: GroundTruthCharacteristic | None
    truth_index: int | None
    matched_truth_char_no: int | str | None
    ambiguity_message: str | None = None


def _normalize_requirement_text(requirement: str | None) -> str | None:
    """Return a deterministic normalized requirement string or ``None`` when blank."""

    if requirement is None:
        return None

    cleaned = requirement.strip()
    if not cleaned:
        return None

    return parse_requirement(cleaned).norm_text


def _truth_match_token(truth_row: GroundTruthCharacteristic, truth_index: int) -> int | str | None:
    """Return the serialized truth reference for a matched truth row."""

    if truth_row.char_no is not None:
        return truth_row.char_no
    return f"{ADDED_POOL_TOKEN_PREFIX}:{truth_index}"


def select_truth_row_for_item(
    item: DeltaItem,
    truth_rows: Sequence[GroundTruthCharacteristic],
    reserved_added_truth_indexes: set[int],
) -> _TruthSelection:
    """Select the canonical truth row corresponding to a packet item."""

    if item.status != "added":
        if item.char_no is None:
            return _TruthSelection(
                truth_row=None,
                truth_index=None,
                matched_truth_char_no=None,
                ambiguity_message=(
                    "packet row is non-added but has no char_no, so it cannot be matched to canonical truth"
                ),
            )

        matches = [
            (truth_index, truth_row)
            for truth_index, truth_row in enumerate(truth_rows)
            if truth_row.classification != "added" and truth_row.char_no == item.char_no
        ]

        if len(matches) == 1:
            truth_index, truth_row = matches[0]
            return _TruthSelection(
                truth_row=truth_row,
                truth_index=truth_index,
                matched_truth_char_no=_truth_match_token(truth_row, truth_index),
            )

        if not matches:
            return _TruthSelection(
                truth_row=None,
                truth_index=None,
                matched_truth_char_no=None,
                ambiguity_message=f"no canonical truth row found for char_no {item.char_no}",
            )

        return _TruthSelection(
            truth_row=None,
            truth_index=None,
            matched_truth_char_no=None,
            ambiguity_message=f"multiple canonical truth rows found for char_no {item.char_no}",
        )

    added_candidates = [
        (truth_index, truth_row)
        for truth_index, truth_row in enumerate(truth_rows)
        if truth_row.classification == "added" and truth_index not in reserved_added_truth_indexes
    ]

    if not added_candidates:
        return _TruthSelection(
            truth_row=None,
            truth_index=None,
            matched_truth_char_no=None,
            ambiguity_message="no unmatched canonical added truth rows remain for this packet row",
        )

    packet_requirement = _normalize_requirement_text(item.requirement_revB)
    if packet_requirement is None:
        return _TruthSelection(
            truth_row=None,
            truth_index=None,
            matched_truth_char_no=None,
            ambiguity_message=(
                "canonical added matching requires usable requirement text before snippet evidence is available"
            ),
        )

    exact_requirement_matches = [
        (truth_index, truth_row)
        for truth_index, truth_row in added_candidates
        if _normalize_requirement_text(truth_row.requirement_revB) == packet_requirement
    ]

    if len(exact_requirement_matches) == 1:
        truth_index, truth_row = exact_requirement_matches[0]
        return _TruthSelection(
            truth_row=truth_row,
            truth_index=truth_index,
            matched_truth_char_no=_truth_match_token(truth_row, truth_index),
        )

    if not exact_requirement_matches:
        return _TruthSelection(
            truth_row=None,
            truth_index=None,
            matched_truth_char_no=None,
            ambiguity_message=(
                "no canonical added truth row matched the packet requirement text; snippet matching is deferred"
            ),
        )

    return _TruthSelection(
        truth_row=None,
        truth_index=None,
        matched_truth_char_no=None,
        ambiguity_message=(
            "multiple canonical added truth rows share the same normalized requirement text; "
            "snippet matching is deferred"
        ),
    )


def _requirement_conforms(
    item: DeltaItem,
    truth_row: GroundTruthCharacteristic,
    ) -> tuple[bool, list[EvaluationMismatch]]:
    """Compare packet-side requirement evidence against canonical truth."""

    mismatches: list[EvaluationMismatch] = []
    truth_requirement = _normalize_requirement_text(truth_row.requirement_revB)
    if truth_requirement is None:
        return True, mismatches

    packet_requirement_raw = item.requirement_revB.strip() if item.requirement_revB is not None else ""
    packet_requirement = _normalize_requirement_text(packet_requirement_raw)
    if packet_requirement is None:
        mismatches.append(
            EvaluationMismatch(
                code="requirement_missing",
                message="truth expects Rev B requirement evidence but the packet row has none",
            )
        )
        return False, mismatches

    truth_semantic_callout = extract_semantic_callout(pdf_spans=[], form3_requirement=truth_row.requirement_revB)
    packet_semantic_callout = item.semantic_callout or extract_semantic_callout(
        pdf_spans=[],
        form3_requirement=packet_requirement_raw,
    )
    semantic_result = compare_semantic_callouts(truth_semantic_callout, packet_semantic_callout)

    if semantic_result.comparable:
        if semantic_result.equal:
            return True, mismatches

        mismatches.append(
            EvaluationMismatch(
                code="requirement_mismatch",
                message=semantic_result.reason_fragments[0] if semantic_result.reason_fragments else "semantic requirement mismatch",
            )
        )
        return False, mismatches

    if semantic_result.mode == "incompatible":
        mismatches.append(
            EvaluationMismatch(
                code="requirement_mismatch",
                message=semantic_result.reason_fragments[0]
                if semantic_result.reason_fragments
                else "semantic requirement families are incompatible",
            )
        )
        return False, mismatches

    if truth_requirement == packet_requirement:
        return True, mismatches

    mismatch_detail = (
        semantic_result.reason_fragments[0]
        if semantic_result.reason_fragments
        else "semantic comparison unavailable; normalized text still differs"
    )
    mismatches.append(
        EvaluationMismatch(
            code="requirement_mismatch",
            message=f"{mismatch_detail}; normalized requirement mismatch: {truth_requirement} != {packet_requirement}",
        )
    )
    return False, mismatches


def evaluate_item_against_truth(
    item: DeltaItem,
    truth_rows: Sequence[GroundTruthCharacteristic],
    reserved_added_truth_indexes: set[int],
) -> ItemEvaluation:
    """Evaluate one packet row against immutable canonical truth."""

    selection = select_truth_row_for_item(item, truth_rows, reserved_added_truth_indexes)
    if selection.truth_row is None or selection.truth_index is None:
        return ItemEvaluation(
            status=REVIEW_NEEDED,
            matched_truth_char_no=selection.matched_truth_char_no,
            truth_match=None,
            classification_conforms=None,
            requirement_conforms=None,
            snippet_conforms=False,
            mismatches=[
                EvaluationMismatch(
                    code=TRUTH_AMBIGUITY_CODE,
                    message=selection.ambiguity_message or "canonical truth row selection is ambiguous",
                )
            ],
        )

    if item.status == "added":
        reserved_added_truth_indexes.add(selection.truth_index)

    truth_match = GroundTruthMatch(
        truth_index=selection.truth_index,
        matched_truth_char_no=selection.matched_truth_char_no,
        classification=selection.truth_row.classification,
    )

    classification_mismatches: list[EvaluationMismatch] = []

    classification_conforms = item.status == selection.truth_row.classification
    if not classification_conforms:
        classification_mismatches.append(
            EvaluationMismatch(
                code="classification_mismatch",
                message=(
                    f"packet classification {item.status} does not match canonical truth "
                    f"{selection.truth_row.classification}"
                ),
            )
        )

    requirement_conforms, requirement_mismatches = _requirement_conforms(item, selection.truth_row)
    snippet_conforms, snippet_mismatches = evaluate_snippet_evidence(item, selection.truth_row)
    mismatches = classification_mismatches + requirement_mismatches + snippet_mismatches
    status = CONFORMING if classification_conforms and requirement_conforms and snippet_conforms else REVIEW_NEEDED

    return ItemEvaluation(
        status=status,
        matched_truth_char_no=selection.matched_truth_char_no,
        truth_match=truth_match,
        classification_conforms=classification_conforms,
        requirement_conforms=requirement_conforms,
        snippet_conforms=snippet_conforms,
        mismatches=mismatches,
    )


def evaluate_packet_against_truth(
    items: Sequence[DeltaItem],
    truth_packet: GroundTruthPacket,
) -> list[ItemEvaluation]:
    """Evaluate packet rows against the immutable canonical truth packet."""

    reserved_added_truth_indexes: set[int] = set()
    return [
        evaluate_item_against_truth(item, truth_packet.characteristics, reserved_added_truth_indexes)
        for item in items
    ]
