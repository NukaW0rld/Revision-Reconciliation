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


_SEMANTIC_REASON = "not_implemented_in_slice"


def extract_semantic_callout(
    pdf_spans: Sequence[TextSpan],
    form3_requirement: Optional[str] = None,
) -> SemanticCallout:
    """Build the typed semantic extraction envelope from PDF-first inputs.

    PDF spans are always authoritative when present. Form 3 requirement text is only
    used as advisory context and never becomes the source of truth when drawing spans
    exist. In this slice, the dispatcher is intentionally bounded: it returns the
    shared semantic envelope plus explicit not-implemented states for the planned
    parser families instead of attempting family-specific parsing.
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

    return SemanticCallout(
        provenance=SemanticProvenance(
            authority=authority,
            source_type=source_type,
            source_ref=source_ref,
            notes=notes,
        ),
        status=SemanticParserStatus(
            state="not_implemented",
            parser_family="semantic_dispatch",
            reason_code=_SEMANTIC_REASON,
            detail=(
                "Semantic dispatcher established; family-specific parsers for GD&T, weld, "
                "surface finish, and fit are deferred to later slices."
            ),
        ),
        raw_text=raw_text,
        normalized_text=normalized_text,
        gdt=_stub_gdt_payload(),
        weld=_stub_weld_payload(),
        surface_finish=_stub_surface_finish_payload(),
        fit=_stub_fit_payload(),
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


def _stub_gdt_payload() -> GdtSemanticPayload:
    return GdtSemanticPayload()


def _stub_weld_payload() -> WeldSemanticPayload:
    return WeldSemanticPayload()


def _stub_surface_finish_payload() -> SurfaceFinishSemanticPayload:
    return SurfaceFinishSemanticPayload()


def _stub_fit_payload() -> FitSemanticPayload:
    return FitSemanticPayload()
