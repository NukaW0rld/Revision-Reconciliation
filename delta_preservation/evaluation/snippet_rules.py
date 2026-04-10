"""Deterministic snippet-evidence checks against immutable ground truth."""

from __future__ import annotations

from typing import Literal

from delta_preservation.evaluation.contracts import GroundTruthCharacteristic
from delta_preservation.types import DeltaItem, EvaluationMismatch

SINGLE_CALLOUT_RULE = "single_callout"
GROUPED_CALLOUT_RULE = "grouped_callout"
EDGE_GUARD_POINTS = 6.0
GROUPED_CONTRACTION_POINTS = 6.0


def _coerce_bbox(raw_bbox: list[float] | tuple[float, float, float, float] | None) -> tuple[float, float, float, float] | None:
    """Return a validated bbox tuple or ``None`` when evidence is missing."""

    if raw_bbox is None or len(raw_bbox) != 4:
        return None

    x0, y0, x1, y1 = (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _contract_bbox(
    bbox: tuple[float, float, float, float],
    contraction: float,
) -> tuple[float, float, float, float] | None:
    """Contract all bbox edges by the requested margin."""

    x0, y0, x1, y1 = bbox
    contracted = (x0 + contraction, y0 + contraction, x1 - contraction, y1 - contraction)
    if contracted[2] <= contracted[0] or contracted[3] <= contracted[1]:
        return None
    return contracted


def _point_inside_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    """Return True when the point lies inside the bbox, including edges."""

    px, py = point
    x0, y0, x1, y1 = bbox
    return x0 <= px <= x1 and y0 <= py <= y1


def _edge_distance(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> float:
    """Return the minimum point-to-edge distance inside a bbox."""

    px, py = point
    x0, y0, x1, y1 = bbox
    return min(px - x0, x1 - px, py - y0, y1 - py)


def _missing_evidence_mismatch(side: Literal["revA", "revB"]) -> EvaluationMismatch:
    """Return the stable missing-evidence mismatch for one packet side."""

    return EvaluationMismatch(
        code=f"snippet_missing_{side}",
        message=f"truth expects {side} snippet evidence but the packet row has none",
    )


def _outside_mismatch(
    side: Literal["revA", "revB"],
    rule_family: Literal["single_callout", "grouped_callout"],
) -> EvaluationMismatch:
    """Return the stable bbox-containment failure mismatch for one packet side."""

    return EvaluationMismatch(
        code=f"snippet_outside_{side}",
        message=(
            f"{side} truth center falls outside the contracted {rule_family.replace('_', '-')} snippet bbox"
        ),
    )


def _edge_guard_mismatch(side: Literal["revA", "revB"]) -> EvaluationMismatch:
    """Return the stable edge-guard failure mismatch for one packet side."""

    return EvaluationMismatch(
        code=f"snippet_edge_guard_{side}",
        message=f"{side} truth center is too close to the single-callout safe-zone edge",
    )


def _evaluate_side(
    *,
    side: Literal["revA", "revB"],
    truth_center: tuple[float, float] | None,
    evidence_bbox: tuple[float, float, float, float] | None,
    rule_family: Literal["single_callout", "grouped_callout"],
) -> tuple[bool, EvaluationMismatch | None]:
    """Evaluate one packet-side snippet against its canonical truth center."""

    if truth_center is None:
        return True, None

    if evidence_bbox is None:
        return False, _missing_evidence_mismatch(side)

    if rule_family == GROUPED_CALLOUT_RULE:
        contracted_bbox = _contract_bbox(evidence_bbox, GROUPED_CONTRACTION_POINTS)
        if contracted_bbox is None or not _point_inside_bbox(truth_center, contracted_bbox):
            return False, _outside_mismatch(side, rule_family)
        return True, None

    width = evidence_bbox[2] - evidence_bbox[0]
    height = evidence_bbox[3] - evidence_bbox[1]
    contraction = max(12.0, 0.10 * min(width, height))
    contracted_bbox = _contract_bbox(evidence_bbox, contraction)
    if contracted_bbox is None or not _point_inside_bbox(truth_center, contracted_bbox):
        return False, _outside_mismatch(side, rule_family)
    if _edge_distance(truth_center, contracted_bbox) < EDGE_GUARD_POINTS:
        return False, _edge_guard_mismatch(side)
    return True, None


def evaluate_snippet_evidence(
    item: DeltaItem,
    truth_row: GroundTruthCharacteristic,
) -> tuple[bool, list[EvaluationMismatch]]:
    """Evaluate both packet-side snippets using deterministic bbox rules."""

    rule_family = item.snippet_rule_family
    revA_ok, revA_mismatch = _evaluate_side(
        side="revA",
        truth_center=truth_row.snippet_center_revA,
        evidence_bbox=_coerce_bbox(item.revA.bbox if item.revA is not None else None),
        rule_family=rule_family,
    )
    revB_ok, revB_mismatch = _evaluate_side(
        side="revB",
        truth_center=truth_row.snippet_center_revB,
        evidence_bbox=_coerce_bbox(item.revB.bbox if item.revB is not None else None),
        rule_family=rule_family,
    )

    mismatches = [mismatch for mismatch in (revA_mismatch, revB_mismatch) if mismatch is not None]
    return revA_ok and revB_ok, mismatches
