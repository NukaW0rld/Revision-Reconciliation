import re
from typing import List, Tuple, Optional, Sequence
from dataclasses import dataclass

from delta_preservation.io.pdf import TextSpan, join_text_spans
from delta_preservation.types import (
    FitSemanticPayload,
    GdtSemanticPayload,
    SemanticCallout,
    SemanticParserStatus,
    SemanticProvenance,
    SurfaceFinishSemanticPayload,
    WeldSemanticPayload,
)


@dataclass
class MatchFingerprint:
    """Deterministic fingerprint for matching requirements across revisions."""
    norm_text: str  # Uppercase, whitespace-collapsed
    count_tokens: List[str]  # e.g., ["2X", "6X"]
    type_tokens: List[str]  # e.g., ["LENGTH", "DIAMETER", "THREAD"]
    symbol_tokens: List[str]  # e.g., ["Ø", "R"]
    numeric_tokens: List[Tuple[float, str]]  # (parsed_value, original_string)
    units: Optional[str]  # "MM", "IN", "DEG", or None
    pattern_class: str  # "note", "hole", "fillet", "dimension", "other"


@dataclass
class ParsedGdtFrame:
    control_type: str
    frame_text: str
    tolerance_text: str
    datum_refs: List[str]
    modifiers: List[str]


def parse_requirement(requirement: str) -> MatchFingerprint:
    """
    Parse a requirement string into a deterministic match fingerprint.

    Args:
        requirement: Raw requirement string from Form 3 data and PDF text spans

    Returns:
        MatchFingerprint with normalized text and extracted tokens
    """
    # Normalize: uppercase and collapse whitespace
    norm_text = " ".join(requirement.upper().split())

    # Extract count tokens (e.g., 2X, 6 X, 12 x)
    # Avoid matching patterns like "10 x 90°" where x is multiplication between two numbers
    # Use negative lookahead to exclude cases where x is followed by a space and digit
    count_tokens = [f"{n}X" for n in re.findall(r'\b(\d+)\s*[Xx](?!\s*\d)', norm_text)]

    # Extract type/prefix tokens
    type_patterns = [
        "DRAWING NOTES", "EDGE RADIUS", "COUNTERBORE", "COUNTERSINK",
        "LENGTH", "DIAMETER", "RADIUS", "DEPTH", "ANGLE", "THREAD", "NOTES",
        "THRU ALL"
    ]
    type_tokens = []
    for pattern in type_patterns:
        if pattern in norm_text:
            type_tokens.append(pattern)

    # Extract symbol tokens
    symbol_tokens = []
    if "Ø" in requirement:
        symbol_tokens.append("Ø")

    # Detect radius symbol when used as R<number> or R.<number>
    if re.search(r'R(?=\d|\.)', norm_text):
        symbol_tokens.append("R")

    # Extract numeric tokens with tolerances
    numeric_tokens = []

    # General numeric values (without word boundary to catch Ø8, R2.5, etc.)
    # Pattern matches:
    #   \d+\.?\d*   — standard decimal numbers (e.g., "0.750", "1.25", "120")
    #   \.\d+       — leading-decimal numbers (e.g., ".750", ".010")
    # Leading-decimal numbers are common in inch-unit engineering drawings
    # (e.g., ".750" = 0.750 inch).  They must be normalized to their float value
    # so that ".750" and "0.750" compare equal.
    numeric_raw = re.findall(r'\d+\.?\d*|\.\d+', norm_text)
    for match in numeric_raw:
        try:
            numeric_tokens.append((float(match), match))
        except ValueError:
            pass

    # Detect units
    units = None
    if re.search(r'\bMM\b', norm_text):
        units = "MM"
    elif re.search(r'\bIN\b', norm_text):
        units = "IN"
    elif re.search(r'\bDEG\b', norm_text):
        units = "DEG"

    # Assign pattern class
    pattern_class = "other"

    if "NOTES" in norm_text or "DRAWING NOTES" in norm_text:
        pattern_class = "note"
    elif "THREAD" in norm_text or re.search(r'[MG]\d+', norm_text):
        pattern_class = "hole"
    elif "RADIUS" in norm_text or "EDGE RADIUS" in norm_text or norm_text.strip().startswith("R "):
        pattern_class = "fillet"
    elif "DIAMETER" in norm_text or "Ø" in requirement or "COUNTERBORE" in norm_text or "COUNTERSINK" in norm_text:
        pattern_class = "hole"
    elif "THRU ALL" in norm_text:
        pattern_class = "hole"
    elif "LENGTH" in norm_text or "DEPTH" in norm_text or "ANGLE" in norm_text:
        pattern_class = "dimension"

    return MatchFingerprint(
        norm_text=norm_text,
        count_tokens=count_tokens,
        type_tokens=type_tokens,
        symbol_tokens=symbol_tokens,
        numeric_tokens=numeric_tokens,
        units=units,
        pattern_class=pattern_class
    )


_GDT_CONTROL_MAP = {
    "⌖": "position",
    "⌒": "profile_of_a_line",
}

_GDT_MODIFIER_MAP = {
    "M": "MMC",
    "MMC": "MMC",
    "L": "LMC",
    "LMC": "LMC",
    "P": "projected_tolerance_zone",
}

_GDT_TOLERANCE_RE = re.compile(r"^(?:⌀)?(?:\d+(?:\.\d+)?|\.\d+)$")
_GDT_DATUM_RE = re.compile(r"^[A-Z](?:-[A-Z])?$")


_SEMANTIC_REASON = "not_implemented_in_slice"


def extract_semantic_callout(
    pdf_spans: Sequence[TextSpan],
    form3_requirement: Optional[str] = None,
) -> SemanticCallout:
    """Build the typed semantic extraction envelope from PDF-first inputs.

    PDF spans are always authoritative when present. Form 3 requirement text is only
    used as advisory context and never becomes the source of truth when drawing spans
    exist. The dispatcher remains bounded: GD&T feature control frames get a narrow
    parser, while other semantic families still return explicit not-implemented states.
    """

    pdf_text = join_text_spans(pdf_spans)
    normalized_pdf = _normalize_semantic_text(pdf_text) if pdf_text else None
    normalized_form3 = _normalize_semantic_text(form3_requirement) if form3_requirement else None

    notes: List[str] = []
    metadata = {
        "dispatcher": "semantic_dispatch",
        "authority_source": "pdf" if normalized_pdf else ("form3" if normalized_form3 else "none"),
    }

    if normalized_pdf:
        authority = "pdf"
        source_type = "drawing_pdf"
        source_ref = _semantic_source_ref(pdf_spans)
        raw_text = pdf_text
        normalized_text = normalized_pdf
        notes.append("PDF-derived spans are authoritative for semantic extraction.")
        if normalized_form3:
            notes.append("Form 3 requirement text was retained as secondary advisory context.")
            if normalized_form3 != normalized_pdf:
                notes.append("PDF authority overrode conflicting Form 3 semantic wording.")
                metadata["conflict_detected"] = "true"
            else:
                metadata["context_alignment"] = "matched"
        else:
            metadata["context_alignment"] = "pdf_only"
    elif normalized_form3:
        authority = "form3"
        source_type = "form3_requirement"
        source_ref = "form3:requirement"
        raw_text = form3_requirement
        normalized_text = normalized_form3
        notes.append("No PDF-derived spans were supplied; Form 3 text is the fallback source.")
        metadata["context_alignment"] = "form3_only"
    else:
        authority = "none"
        source_type = "none"
        source_ref = None
        raw_text = None
        normalized_text = None
        notes.append("No semantic source text was available from PDF spans or Form 3 context.")
        metadata["context_alignment"] = "none"

    metadata["form3_context_supplied"] = "true" if normalized_form3 else "false"
    metadata["planned_families"] = "gdt,weld,surface_finish,fit"

    status, gdt_payload = _extract_gdt_status(normalized_text)

    return SemanticCallout(
        provenance=SemanticProvenance(
            authority=authority,
            source_type=source_type,
            source_ref=source_ref,
            notes=notes,
        ),
        status=status,
        raw_text=raw_text,
        normalized_text=normalized_text,
        gdt=gdt_payload,
        weld=_stub_weld_payload() if status.parser_family != "gdt" else None,
        surface_finish=_stub_surface_finish_payload() if status.parser_family != "gdt" else None,
        fit=_stub_fit_payload() if status.parser_family != "gdt" else None,
        metadata=metadata,
    )


def _normalize_semantic_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = " ".join(text.split())
    return normalized or None


def _semantic_source_ref(pdf_spans: Sequence[TextSpan]) -> Optional[str]:
    if not pdf_spans:
        return None

    first = min(pdf_spans, key=lambda span: (span.block_id, span.line_id, span.span_id))
    return f"pdf:block:{first.block_id}/line:{first.line_id}/span:{first.span_id}"


def _extract_gdt_status(normalized_text: Optional[str]) -> Tuple[SemanticParserStatus, Optional[GdtSemanticPayload]]:
    if not normalized_text:
        return (
            SemanticParserStatus(
                state="not_implemented",
                parser_family="semantic_dispatch",
                reason_code=_SEMANTIC_REASON,
                detail=(
                    "Semantic dispatcher established; family-specific parsers for GD&T, weld, "
                    "surface finish, and fit are deferred to later slices."
                ),
            ),
            _stub_gdt_payload(),
        )

    parse_result = _parse_gdt_frame(normalized_text)
    if parse_result is None:
        return (
            SemanticParserStatus(
                state="empty",
                parser_family="gdt",
                reason_code="gdt_no_match",
                detail="text did not match the bounded GD&T feature control frame grammar",
            ),
            None,
        )

    if isinstance(parse_result, str):
        return (
            SemanticParserStatus(
                state="error",
                parser_family="gdt",
                reason_code="gdt_malformed_frame",
                detail=parse_result,
            ),
            None,
        )

    return (
        SemanticParserStatus(
            state="parsed",
            parser_family="gdt",
            reason_code=None,
            detail="parsed feature control frame from PDF spans",
        ),
        GdtSemanticPayload(
            frame_text=parse_result.frame_text,
            control_type=parse_result.control_type,
            tolerance_text=parse_result.tolerance_text,
            datum_refs=parse_result.datum_refs,
            modifiers=parse_result.modifiers,
        ),
    )


def _parse_gdt_frame(normalized_text: str) -> Optional[ParsedGdtFrame | str]:
    tokens = normalized_text.split()
    if not tokens:
        return None

    control_symbol = tokens[0]
    control_type = _GDT_CONTROL_MAP.get(control_symbol)
    if control_type is None:
        return None

    if len(tokens) < 2:
        return "recognized GD&T control symbol but missing tolerance segment"

    tolerance_token = tokens[1]
    if not _is_gdt_tolerance_token(tolerance_token):
        return "recognized GD&T control symbol but missing tolerance segment"

    modifiers: List[str] = []
    datum_refs: List[str] = []

    for token in tokens[2:]:
        normalized_modifier = _GDT_MODIFIER_MAP.get(token)
        if normalized_modifier:
            modifiers.append(normalized_modifier)
            continue
        if _GDT_DATUM_RE.fullmatch(token):
            datum_refs.append(token)
            continue
        return f"recognized GD&T frame contains unsupported segment: {token}"

    return ParsedGdtFrame(
        control_type=control_type,
        frame_text=" | ".join(tokens),
        tolerance_text=tolerance_token,
        datum_refs=datum_refs,
        modifiers=modifiers,
    )


def _is_gdt_tolerance_token(token: str) -> bool:
    return bool(_GDT_TOLERANCE_RE.fullmatch(token))


def _stub_gdt_payload() -> GdtSemanticPayload:
    return GdtSemanticPayload()


def _stub_weld_payload() -> WeldSemanticPayload:
    return WeldSemanticPayload()


def _stub_surface_finish_payload() -> SurfaceFinishSemanticPayload:
    return SurfaceFinishSemanticPayload()


def _stub_fit_payload() -> FitSemanticPayload:
    return FitSemanticPayload()
