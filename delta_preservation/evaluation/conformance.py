"""Packet-side conformance checks against immutable ground truth."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from delta_preservation.evaluation.contracts import GroundTruthCharacteristic, GroundTruthPacket
from delta_preservation.evaluation.snippet_rules import evaluate_snippet_evidence
from delta_preservation.reconcile.normalize import extract_semantic_callout, parse_requirement
from delta_preservation.reconcile.semantic_compare import compare_semantic_callouts
from delta_preservation.types import (
    AcceptedAlternateHistoryRecord,
    DeltaItem,
    EvaluationMismatch,
    GroundTruthMatch,
    HistoryReference,
    ItemEvaluation,
)

REVIEW_NEEDED = "review_needed"
CONFORMING = "conforming"
TRUTH_AMBIGUITY_CODE = "truth_ambiguity"
ADDED_POOL_TOKEN_PREFIX = "added"

# Maximum distance (in PDF points) from the packet bbox center to a truth
# snippet_center_revB to qualify for the nearest-center fallback tie-break.
# 72 pt ≈ 1 inch; 100 pt provides a comfortable region without risking
# cross-cluster confusion on typical aerospace drawings.
ADDED_TRUTH_TIEBREAK_MAX_DISTANCE_PT: float = 100.0
_CONTROL_SPACING_RE = re.compile(r"([⌖◎⌓⌰⚪⏥⟂↧∅Ø])\s+(?=[∅Ø.\d])")

# Normalizes leading-zero variants in decimal tokens where the integer part is exactly 0,
# e.g. ``0.635`` → ``.635``, ``+0.50`` → ``+.50``, ``-0.05`` → ``-.05``.
# Does NOT affect tokens where the integer part is > 0 (``10.000``, ``1.250``, etc.).
# Applied after all other normalization so the final canonical form is deterministic.
_LEADING_ZERO_RE = re.compile(r"(?<![.\d])([+-]?)0(\.\d+)")


# ---------------------------------------------------------------------------
# Geometry helpers for added-truth tie-break (self-contained; no imports from
# snippet_rules.py private helpers).
# ---------------------------------------------------------------------------

def _coerce_packet_bbox(bbox_raw: object) -> tuple[float, float, float, float] | None:
    """Return a validated 4-element bbox tuple from packet evidence, or None.

    Accepts any sequence of exactly 4 real numbers.  Returns None when the
    input is None, not a sequence, has the wrong length, or contains
    non-numeric values.
    """
    if bbox_raw is None:
        return None
    try:
        items = list(bbox_raw)
    except TypeError:
        return None
    if len(items) != 4:
        return None
    try:
        x0, y0, x1, y1 = float(items[0]), float(items[1]), float(items[2]), float(items[3])
    except (TypeError, ValueError):
        return None
    return (x0, y0, x1, y1)


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the center point of a validated 4-element bbox tuple."""
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _point_inside_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    """Return True when *point* falls strictly inside the packet bbox."""
    px, py = point
    x0, y0, x1, y1 = bbox
    return x0 <= px <= x1 and y0 <= py <= y1


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


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

    normalized = parse_requirement(cleaned).norm_text
    semantic_callout = extract_semantic_callout(pdf_spans=[], form3_requirement=cleaned)
    if semantic_callout is not None and semantic_callout.normalized_text:
        # Semantic fallback may preserve source casing for non-parsed/freeform
        # notes. Canonicalize it to the same uppercase, collapsed-whitespace
        # contract used by parse_requirement so equivalent note text compares
        # equal during ground-truth evaluation.
        normalized = " ".join(semantic_callout.normalized_text.upper().split())

    # Added-row truth selection should not fail solely because one source
    # inserts harmless spacing between a control symbol and its tolerance token.
    normalized = _CONTROL_SPACING_RE.sub(r"\1", normalized)

    # Normalize leading-zero variants in pure-decimal tokens (integer part == 0)
    # so that ``0.635 / 0.615`` and ``.635 / .615`` compare equal.  Tokens where
    # the integer part is non-zero (``10.000``, ``1.250``) are left unchanged.
    return _LEADING_ZERO_RE.sub(r"\1\2", normalized)


def _truth_match_token(truth_row: GroundTruthCharacteristic, truth_index: int) -> int | str | None:
    """Return the serialized truth reference for a matched truth row.

    Added rows always use the canonical ``added:<index>`` token regardless of
    whether the truth row also carries a legacy ``char_no``.  This prevents
    debug-queue accounting from losing the claim when ``_load_missing_added_truth_items``
    checks only for string tokens shaped like ``added:<index>``.
    """

    if truth_row.classification == "added":
        return f"{ADDED_POOL_TOKEN_PREFIX}:{truth_index}"
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

    # ------------------------------------------------------------------
    # Second-stage tie-break using packet-side Rev B bbox evidence.
    # Only reached when exact_requirement_matches has more than one row.
    # ------------------------------------------------------------------

    packet_bbox = _coerce_packet_bbox(item.revB.bbox if item.revB is not None else None)

    if packet_bbox is not None:
        # Stage 1: prefer a truth center that falls inside the packet bbox.
        inside_bbox = [
            (truth_index, truth_row)
            for truth_index, truth_row in exact_requirement_matches
            if truth_row.snippet_center_revB is not None
            and _point_inside_bbox(truth_row.snippet_center_revB, packet_bbox)
        ]
        if len(inside_bbox) == 1:
            truth_index, truth_row = inside_bbox[0]
            return _TruthSelection(
                truth_row=truth_row,
                truth_index=truth_index,
                matched_truth_char_no=_truth_match_token(truth_row, truth_index),
            )

        # Stage 2: if exactly one truth center is the unique nearest within the
        # distance threshold, select it.
        bbox_ctr = _bbox_center(packet_bbox)
        candidates_with_distance = [
            (truth_index, truth_row, _distance(bbox_ctr, truth_row.snippet_center_revB))
            for truth_index, truth_row in exact_requirement_matches
            if truth_row.snippet_center_revB is not None
        ]
        if candidates_with_distance:
            min_dist = min(d for _, _, d in candidates_with_distance)
            if min_dist <= ADDED_TRUTH_TIEBREAK_MAX_DISTANCE_PT:
                nearest = [
                    (ti, tr)
                    for ti, tr, d in candidates_with_distance
                    if d == min_dist
                ]
                if len(nearest) == 1:
                    truth_index, truth_row = nearest[0]
                    return _TruthSelection(
                        truth_row=truth_row,
                        truth_index=truth_index,
                        matched_truth_char_no=_truth_match_token(truth_row, truth_index),
                    )

    # Conservative fallback: multiple canonical added truth rows share the same
    # normalized requirement text and packet evidence does not identify one row
    # uniquely.  Preserve ambiguity rather than guess.
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


def _current_mismatch_codes(evaluation: ItemEvaluation) -> set[str]:
    return {mismatch.code for mismatch in evaluation.mismatches if mismatch.code}


def _history_tokens_match(
    item: DeltaItem,
    evaluation: ItemEvaluation,
    alternate: AcceptedAlternateHistoryRecord,
) -> bool:
    if evaluation.matched_truth_char_no is not None:
        if alternate.matched_truth_char_no is None:
            return False
        return str(evaluation.matched_truth_char_no) == str(alternate.matched_truth_char_no)

    if alternate.matched_truth_char_no is not None:
        return False

    return item.char_no == alternate.char_no


def apply_accepted_alternate_history(
    items: Sequence[DeltaItem],
    evaluations: Sequence[ItemEvaluation],
    approved_alternates: Sequence[AcceptedAlternateHistoryRecord],
) -> list[ItemEvaluation]:
    """Upgrade review-needed rows when an exact accepted alternate fingerprint exists."""

    updated: list[ItemEvaluation] = []
    for item, evaluation in zip(items, evaluations):
        if evaluation.status != REVIEW_NEEDED:
            updated.append(evaluation)
            continue

        current_requirement = _normalize_requirement_text(item.requirement_revB)
        current_mismatch_codes = _current_mismatch_codes(evaluation)
        matched_alternate = None
        for alternate in approved_alternates:
            if item.status != alternate.reviewed_classification:
                continue
            if not _history_tokens_match(item, evaluation, alternate):
                continue

            alternate_requirement = _normalize_requirement_text(alternate.reviewed_requirement_revB)
            if alternate_requirement is not None and current_requirement != alternate_requirement:
                continue

            if current_mismatch_codes != set(alternate.mismatch_codes):
                continue

            matched_alternate = alternate
            break

        if matched_alternate is None:
            updated.append(evaluation)
            continue

        updated.append(
            evaluation.model_copy(
                update={
                    "status": CONFORMING,
                    "conformance_source": "accepted_alternate",
                    "history_reference": HistoryReference(
                        history_id=matched_alternate.history_id,
                        source_run_id=matched_alternate.source_run_id,
                    ),
                }
            )
        )

    return updated
