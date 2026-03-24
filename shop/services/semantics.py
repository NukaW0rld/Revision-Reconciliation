from __future__ import annotations

from typing import Any

from delta_preservation.types import DeltaItem, SemanticCallout


SEMANTIC_FAMILY_LABELS = {
    "gdt": "GD&T",
    "weld": "Weld",
    "surface_finish": "Surface Finish",
    "fit": "Fit",
    "semantic_dispatch": "Semantic Dispatch",
    "none": "None",
}


def _compact_text(value: str | None) -> str | None:
    if value is None:
        return None
    compact = " ".join(str(value).split())
    return compact or None


def _family_label(family: str | None) -> str | None:
    if family is None:
        return None
    return SEMANTIC_FAMILY_LABELS.get(family, family.replace("_", " ").title())


def _payload_summary(semantic: SemanticCallout) -> str | None:
    family = semantic.status.parser_family

    if family == "gdt" and semantic.gdt is not None:
        bits: list[str] = []
        if semantic.gdt.control_type:
            bits.append(semantic.gdt.control_type)
        if semantic.gdt.tolerance_text:
            bits.append(semantic.gdt.tolerance_text)
        if semantic.gdt.datum_refs:
            bits.append(f"datums {', '.join(semantic.gdt.datum_refs)}")
        if semantic.gdt.modifiers:
            bits.append(f"modifiers {', '.join(semantic.gdt.modifiers)}")
        return " ".join(bits) if bits else _compact_text(semantic.gdt.frame_text)

    if family == "weld" and semantic.weld is not None:
        bits = []
        if semantic.weld.process:
            bits.append(semantic.weld.process)
        if semantic.weld.size:
            bits.append(f"size {semantic.weld.size}")
        if semantic.weld.side:
            bits.append(semantic.weld.side)
        if semantic.weld.all_around:
            bits.append("all-around")
        if semantic.weld.length:
            bits.append(f"length {semantic.weld.length}")
        if semantic.weld.pitch:
            bits.append(f"pitch {semantic.weld.pitch}")
        if semantic.weld.contour:
            bits.append(f"contour {semantic.weld.contour}")
        if semantic.weld.tail:
            bits.append(f"tail {semantic.weld.tail}")
        return " ".join(bits) if bits else _compact_text(semantic.normalized_text or semantic.raw_text)

    if family == "surface_finish" and semantic.surface_finish is not None:
        return _compact_text(
            semantic.surface_finish.canonical_text
            or semantic.normalized_text
            or semantic.raw_text
        )

    if family == "fit" and semantic.fit is not None:
        bits = []
        canonical = _compact_text(semantic.fit.canonical_text or semantic.fit.fit_class)
        if canonical:
            bits.append(canonical)
        if semantic.fit.basis:
            bits.append(f"({semantic.fit.basis})")
        return " ".join(bits) if bits else _compact_text(semantic.normalized_text or semantic.raw_text)

    return _compact_text(semantic.normalized_text or semantic.raw_text)


def shape_semantic_contract(delta: DeltaItem | dict[str, Any]) -> dict[str, Any] | None:
    """Return a normalized semantic summary contract for review/export consumers.

    The returned dict is compact and template-safe: it carries only normalized
    semantic text, parser status/provenance, and filtered rationale fragments.
    Raw packet JSON and filesystem paths stay out of the contract.
    """
    delta_item = delta if isinstance(delta, DeltaItem) else DeltaItem.model_validate(delta)
    semantic = delta_item.semantic_callout
    if semantic is None:
        return None

    family = semantic.status.parser_family
    state = semantic.status.state
    family_label = _family_label(family)
    status_label = state.replace("_", " ")
    summary = _payload_summary(semantic)
    normalized_text = _compact_text(semantic.normalized_text or semantic.raw_text)

    reason_fragments = [
        reason
        for reason in delta_item.reasons
        if isinstance(reason, str) and reason.lower().startswith("semantic ")
    ]

    block_label = None
    if state == "parsed":
        block_label = f"{family_label} parsed" if family_label else "Parsed"
    elif state in {"empty", "error", "skipped", "not_implemented", "not_requested"}:
        if family_label:
            block_label = f"{family_label} {status_label}"
        else:
            block_label = status_label.title()

    return {
        "family": family,
        "family_label": family_label,
        "status": state,
        "status_label": status_label,
        "parser_family": family,
        "parser_family_label": family_label,
        "reason_code": semantic.status.reason_code,
        "detail": _compact_text(semantic.status.detail),
        "provenance": {
            "authority": semantic.provenance.authority,
            "source_type": semantic.provenance.source_type,
            "source_ref": semantic.provenance.source_ref,
            "notes": list(semantic.provenance.notes),
        },
        "summary": summary,
        "normalized_text": normalized_text,
        "reason_fragments": reason_fragments,
        "reason_summary": " | ".join(reason_fragments) if reason_fragments else None,
        "block_label": block_label,
        "is_parsed": state == "parsed",
        "is_fallback": state != "parsed",
        "has_content": any([summary, normalized_text, reason_fragments, semantic.status.detail]),
    }
