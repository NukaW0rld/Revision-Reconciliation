from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from delta_preservation.types import (
    FitSemanticPayload,
    GdtSemanticPayload,
    SemanticCallout,
    SurfaceFinishSemanticPayload,
    WeldSemanticPayload,
)

SemanticCompareMode = Literal["semantic", "fallback", "incompatible"]
SemanticFamily = Literal["gdt", "weld", "surface_finish", "fit"]


@dataclass(frozen=True)
class SemanticCompareResult:
    comparable: bool
    equal: Optional[bool]
    family: Optional[SemanticFamily]
    mode: SemanticCompareMode
    reason_fragments: List[str]


def compare_semantic_callouts(
    left: Optional[SemanticCallout],
    right: Optional[SemanticCallout],
) -> SemanticCompareResult:
    if left is None and right is None:
        return SemanticCompareResult(
            comparable=False,
            equal=None,
            family=None,
            mode="fallback",
            reason_fragments=[
                "semantic comparison unavailable: both semantic callouts missing; fall back to numeric/text comparison"
            ],
        )

    state_result = _status_fallback_result(left, side="left")
    if state_result is not None:
        return state_result

    state_result = _status_fallback_result(right, side="right")
    if state_result is not None:
        return state_result

    assert left is not None and right is not None
    left_family = left.status.parser_family
    right_family = right.status.parser_family

    if left_family != right_family:
        return SemanticCompareResult(
            comparable=False,
            equal=False,
            family=None,
            mode="incompatible",
            reason_fragments=[f"semantic families differ: {left_family} vs {right_family}"],
        )

    if left_family == "gdt":
        return _compare_gdt(left.gdt, right.gdt)
    if left_family == "weld":
        return _compare_weld(left.weld, right.weld)
    if left_family == "surface_finish":
        return _compare_surface_finish(left.surface_finish, right.surface_finish)
    if left_family == "fit":
        return _compare_fit(left.fit, right.fit)

    return SemanticCompareResult(
        comparable=False,
        equal=None,
        family=None,
        mode="fallback",
        reason_fragments=[
            f"semantic comparison fallback: unsupported parser family {left_family}"
        ],
    )


def _status_fallback_result(
    callout: Optional[SemanticCallout],
    *,
    side: Literal["left", "right"],
) -> Optional[SemanticCompareResult]:
    if callout is None:
        return SemanticCompareResult(
            comparable=False,
            equal=None,
            family=None,
            mode="fallback",
            reason_fragments=[f"semantic comparison fallback: {side} semantic callout missing"],
        )

    if callout.status.state != "parsed":
        detail = callout.status.reason_code or callout.status.parser_family
        return SemanticCompareResult(
            comparable=False,
            equal=None,
            family=None,
            mode="fallback",
            reason_fragments=[
                f"semantic comparison fallback: {side} semantic state {callout.status.state}/{detail}"
            ],
        )

    return None


def _compare_gdt(
    left: Optional[GdtSemanticPayload],
    right: Optional[GdtSemanticPayload],
) -> SemanticCompareResult:
    missing = _missing_payload_result(left, right, family="gdt")
    if missing is not None:
        return missing

    assert left is not None and right is not None
    left_tolerance_normalized = _normalize_gdt_tolerance_text(left.tolerance_text)
    right_tolerance_normalized = _normalize_gdt_tolerance_text(right.tolerance_text)

    def _compartment_key(c) -> tuple:
        return (
            c.control_type,
            _normalize_gdt_tolerance_text(c.tolerance_text),
            tuple(c.datum_refs),
            tuple(c.modifiers),
        )

    left_compartment_keys = [_compartment_key(c) for c in left.compartments]
    right_compartment_keys = [_compartment_key(c) for c in right.compartments]
    compartments_match = (left_compartment_keys == right_compartment_keys)

    if (
        left.control_type == right.control_type
        and left_tolerance_normalized == right_tolerance_normalized
        and left.datum_refs == right.datum_refs
        and left.modifiers == right.modifiers
        and compartments_match
    ):
        return SemanticCompareResult(
            comparable=True,
            equal=True,
            family="gdt",
            mode="semantic",
            reason_fragments=[
                "semantic GD&T match: "
                f"{left.control_type} {left.tolerance_text} datums {_join_tokens(left.datum_refs)} modifiers {_join_tokens(left.modifiers)}"
            ],
        )

    # Compartment count mismatch is the highest-priority compartment signal.
    if len(left.compartments) != len(right.compartments):
        return SemanticCompareResult(
            comparable=True,
            equal=False,
            family="gdt",
            mode="semantic",
            reason_fragments=[
                f"semantic GD&T changed: compartments {len(left.compartments)} → {len(right.compartments)}"
            ],
        )

    # Field-level compartment difference (same count but content differs).
    if not compartments_match and left.compartments:
        for i, (lc, rc) in enumerate(zip(left.compartments, right.compartments)):
            if _compartment_key(lc) != _compartment_key(rc):
                return SemanticCompareResult(
                    comparable=True,
                    equal=False,
                    family="gdt",
                    mode="semantic",
                    reason_fragments=[
                        f"semantic GD&T changed: compartments compartment {i + 1} differs"
                    ],
                )

    for label, left_value, right_value in (
        ("control", left.control_type, right.control_type),
        ("tolerance", left.tolerance_text, right.tolerance_text),
        ("datums", _join_tokens(left.datum_refs), _join_tokens(right.datum_refs)),
        ("modifiers", _join_tokens(left.modifiers), _join_tokens(right.modifiers)),
    ):
        if left_value != right_value:
            return SemanticCompareResult(
                comparable=True,
                equal=False,
                family="gdt",
                mode="semantic",
                reason_fragments=[f"semantic GD&T changed: {label} {left_value} → {right_value}"],
            )

    return _generic_changed_result("gdt", left.frame_text, right.frame_text)


def _compare_weld(
    left: Optional[WeldSemanticPayload],
    right: Optional[WeldSemanticPayload],
) -> SemanticCompareResult:
    missing = _missing_payload_result(left, right, family="weld")
    if missing is not None:
        return missing

    assert left is not None and right is not None
    if left == right:
        qualifiers = []
        if left.side:
            qualifiers.append(left.side)
        if left.all_around:
            qualifiers.append("all-around")
        if left.length:
            qualifiers.append(f"length {left.length}")
        if left.pitch:
            qualifiers.append(f"pitch {left.pitch}")
        if left.contour:
            qualifiers.append(f"contour {left.contour}")
        if left.tail:
            qualifiers.append(f"tail {left.tail}")
        summary = " ".join([f"{left.process} size {left.size}", *qualifiers]).strip()
        return SemanticCompareResult(
            comparable=True,
            equal=True,
            family="weld",
            mode="semantic",
            reason_fragments=[f"semantic weld match: {summary}"],
        )

    for label, left_value, right_value in (
        ("process", left.process, right.process),
        ("size", left.size, right.size),
        ("side", left.side, right.side),
        ("all-around", _bool_text(left.all_around), _bool_text(right.all_around)),
        ("length", left.length, right.length),
        ("pitch", left.pitch, right.pitch),
        ("contour", left.contour, right.contour),
        ("tail", left.tail, right.tail),
    ):
        if left_value != right_value:
            return SemanticCompareResult(
                comparable=True,
                equal=False,
                family="weld",
                mode="semantic",
                reason_fragments=[f"semantic weld changed: {label} {left_value} → {right_value}"],
            )

    return _generic_changed_result("weld", left.process, right.process)


def _compare_surface_finish(
    left: Optional[SurfaceFinishSemanticPayload],
    right: Optional[SurfaceFinishSemanticPayload],
) -> SemanticCompareResult:
    missing = _missing_payload_result(left, right, family="surface_finish")
    if missing is not None:
        return missing

    assert left is not None and right is not None
    if left == right:
        return SemanticCompareResult(
            comparable=True,
            equal=True,
            family="surface_finish",
            mode="semantic",
            reason_fragments=[f"semantic surface finish match: {left.canonical_text}"],
        )

    for label, left_value, right_value in (
        (
            "roughness",
            left.canonical_text,
            right.canonical_text,
        ),
        ("indicator", left.indicator, right.indicator),
        ("units", left.units, right.units),
    ):
        if left_value != right_value:
            return SemanticCompareResult(
                comparable=True,
                equal=False,
                family="surface_finish",
                mode="semantic",
                reason_fragments=[f"semantic surface finish changed: {label} {left_value} → {right_value}"],
            )

    return _generic_changed_result("surface finish", left.canonical_text, right.canonical_text)


def _compare_fit(
    left: Optional[FitSemanticPayload],
    right: Optional[FitSemanticPayload],
) -> SemanticCompareResult:
    missing = _missing_payload_result(left, right, family="fit")
    if missing is not None:
        return missing

    assert left is not None and right is not None
    if left == right:
        return SemanticCompareResult(
            comparable=True,
            equal=True,
            family="fit",
            mode="semantic",
            reason_fragments=[f"semantic fit match: {left.fit_class} ({left.basis})"],
        )

    for label, left_value, right_value in (
        ("fit class", left.fit_class, right.fit_class),
        ("hole class", left.hole_class, right.hole_class),
        ("shaft class", left.shaft_class, right.shaft_class),
        ("basis", left.basis, right.basis),
    ):
        if left_value != right_value:
            return SemanticCompareResult(
                comparable=True,
                equal=False,
                family="fit",
                mode="semantic",
                reason_fragments=[f"semantic fit changed: {label} {left_value} → {right_value}"],
            )

    return _generic_changed_result("fit", left.canonical_text, right.canonical_text)


def _normalize_gdt_tolerance_text(value: str | None) -> str | None:
    if value is None:
        return None
    prefix = ""
    numeric = value
    if value.startswith(("⌀", "∅")):
        prefix = "⌀"
        numeric = value[1:]
    if numeric.startswith("."):
        numeric = f"0{numeric}"
    if "." in numeric:
        numeric = numeric.rstrip("0").rstrip(".")
    return f"{prefix}{numeric}"



def _missing_payload_result(left: object, right: object, *, family: SemanticFamily) -> Optional[SemanticCompareResult]:
    if left is None or right is None:
        missing_side = "left" if left is None else "right"
        return SemanticCompareResult(
            comparable=False,
            equal=None,
            family=family,
            mode="fallback",
            reason_fragments=[
                f"semantic comparison fallback: {missing_side} {family} payload missing"
            ],
        )
    return None


def _generic_changed_result(family: str, left_value: object, right_value: object) -> SemanticCompareResult:
    return SemanticCompareResult(
        comparable=True,
        equal=False,
        family=family,  # type: ignore[arg-type]
        mode="semantic",
        reason_fragments=[f"semantic {family} changed: {left_value} → {right_value}"],
    )


def _join_tokens(tokens: List[str]) -> str:
    return ",".join(tokens) if tokens else "none"


def _bool_text(value: Optional[bool]) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "none"
