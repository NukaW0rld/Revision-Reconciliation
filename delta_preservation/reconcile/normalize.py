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


@dataclass
class ParsedWeldCallout:
    process: str
    size: Optional[str]
    contour: Optional[str]
    side: Optional[str]
    length: Optional[str]
    pitch: Optional[str]
    tail: Optional[str]
    all_around: Optional[bool]


@dataclass
class ParsedSurfaceFinishCallout:
    canonical_text: str
    roughness_value: str
    units: str
    value_micrometers: Optional[str]
    indicator: str


@dataclass
class ParsedFitCallout:
    canonical_text: str
    fit_class: str
    hole_class: str
    shaft_class: str
    basis: str
    standard_hint: str


# ---------------------------------------------------------------------------
# Requirement type classification
# ---------------------------------------------------------------------------

# GD&T feature-control-frame anchor symbols (appear as the first token)
_GDT_ANCHOR_RE = re.compile(r"^[⌖⌒⟂⊙⌓⏥∥∠]")

# Thread callout patterns (inch fractional, number-size, metric, pipe, bare class)
_THREAD_CALLOUT_RE = re.compile(
    r"(?:"
    r"\d+/\d+-\d+\s*UN[CEF]*"        # 1/4-20 UNC, 3/8-16 UNF
    r"|#\d+-\d+\s*UN[CEF]*"           # #10-32 UNF
    r"|[Mm]\d+(?:\s*[Xx×]\s*\d+(?:\.\d+)?)?"  # M8 or M8×1.25
    r"|\bNPTF?\b"                     # NPT / NPTF
    r"|\bUN[CEF]+[-\s]\d+[AB]\b"      # UNC-2B, UNF-3A
    r"|\d+-\d+\s*UN[CEF]*"            # 0.250-20 UNC
    r")",
    re.IGNORECASE,
)

# Chamfer callout: <linear_value> X <angle_value>° (the X is multiplication)
# e.g. ".070 X 45.0°", "2 X 45°", ".030 x 45°"
_CHAMFER_CALLOUT_RE = re.compile(
    r"(?:\.\d+|\d+\.?\d*)\s*[Xx×]\s*(?:\d+\.?\d*|\.\d+)\s*°"
)

# Fillet / radius prefix: R immediately followed by a digit or decimal point
_FILLET_PREFIX_RE = re.compile(r"^\s*R[\d.]", re.IGNORECASE)

# Surface-finish Ra indicator
_SURFACE_FINISH_TYPE_RE = re.compile(r"\bRa\b", re.IGNORECASE)

# ISO fit designator: upper-case letter + digit(s) / lower-case letter + digit(s)
_FIT_TYPE_RE = re.compile(r"[A-Z]\d+\s*/\s*[a-z]\d+")


def classify_requirement_type(text: str) -> str:
    """Classify a requirement string into a fine-grained semantic type label.

    Returns one of:
      "gdt"            – GD&T feature control frame (⟂, ⌖, ⏥, …)
      "thread"         – Thread callout (1/4-20 UNC, M8×1.25, …)
      "fillet"         – Radius / fillet (R.250, EDGE RADIUS, …)
      "chamfer"        – Chamfer callout (.070 X 45°, 2 X 45°, …)
      "angle"          – Plain angle (65°, 30 DEG, …) – not a chamfer
      "diameter"       – Plain diameter (Ø.500, …) – not GD&T
      "surface_finish" – Surface roughness (Ra 63, 1000 Ra, …)
      "weld"           – Weld callout (FILLET WELD, GROOVE WELD, …)
      "fit"            – ISO fit (H7/p6, …)
      "other"          – Unrecognised / ambiguous
    """
    norm_upper = " ".join(text.upper().split())

    # GD&T: text starts with a GD&T control symbol
    if _GDT_ANCHOR_RE.match(text.strip()):
        return "gdt"

    # Thread: contains a recognisable thread specification
    if _THREAD_CALLOUT_RE.search(text):
        return "thread"

    # Surface finish: Ra indicator with a numeric roughness value
    if _SURFACE_FINISH_TYPE_RE.search(text) and re.search(r"\d", text):
        return "surface_finish"

    # Weld: weld process keyword present alongside "WELD"
    for weld_pattern, _ in _WELD_PROCESS_PATTERNS:
        if weld_pattern.search(norm_upper) and "WELD" in norm_upper:
            return "weld"

    # Chamfer: <value> X <angle>° multiplication pattern
    if _CHAMFER_CALLOUT_RE.search(text):
        return "chamfer"

    # Fillet / radius: R-prefix or explicit RADIUS / EDGE RADIUS keyword
    if _FILLET_PREFIX_RE.match(text) or "RADIUS" in norm_upper or "EDGE RADIUS" in norm_upper:
        return "fillet"

    # ISO fit designator
    if _FIT_TYPE_RE.search(text):
        return "fit"

    # Diameter: Ø symbol without GD&T context
    if "Ø" in text:
        return "diameter"

    # Angle: degree symbol or DEG keyword (not a chamfer – already caught above)
    if "°" in text or re.search(r"\bDEG\b", norm_upper):
        return "angle"

    return "other"


# Pairs of requirement types that are contextually incompatible.
# Rev A and Rev B requirements belonging to incompatible types should be
# flagged as "uncertain" rather than silently classified as changed/removed.
_INCOMPATIBLE_TYPE_PAIRS: frozenset = frozenset({
    frozenset({"gdt", "thread"}),
    frozenset({"gdt", "fillet"}),
    frozenset({"gdt", "chamfer"}),
    frozenset({"gdt", "surface_finish"}),
    frozenset({"gdt", "weld"}),
    frozenset({"gdt", "angle"}),
    frozenset({"gdt", "diameter"}),
    frozenset({"thread", "fillet"}),
    frozenset({"thread", "angle"}),
    frozenset({"thread", "chamfer"}),
    frozenset({"thread", "surface_finish"}),
    frozenset({"thread", "weld"}),
    frozenset({"thread", "diameter"}),
    frozenset({"fillet", "angle"}),
    frozenset({"fillet", "chamfer"}),
    frozenset({"fillet", "surface_finish"}),
    frozenset({"fillet", "weld"}),
    frozenset({"fillet", "diameter"}),
    frozenset({"angle", "diameter"}),
    frozenset({"angle", "surface_finish"}),
    frozenset({"angle", "weld"}),
    frozenset({"chamfer", "surface_finish"}),
    frozenset({"chamfer", "weld"}),
    frozenset({"weld", "surface_finish"}),
    frozenset({"weld", "diameter"}),
    frozenset({"surface_finish", "diameter"}),
    frozenset({"fit", "gdt"}),
    frozenset({"fit", "weld"}),
    frozenset({"fit", "surface_finish"}),
    frozenset({"fit", "thread"}),
    frozenset({"fit", "angle"}),
    frozenset({"fit", "fillet"}),
    frozenset({"fit", "chamfer"}),
})


def are_requirement_types_incompatible(type_a: str, type_b: str) -> bool:
    """Return True if two requirement type labels are contextually incompatible.

    Two types are incompatible when a Rev A characteristic of one kind (e.g.
    a thread callout) is matched to a Rev B span of a fundamentally different
    kind (e.g. a plain linear dimension or an angle).  Such matches should be
    flagged as "uncertain" so a reviewer can assess whether the characteristic
    was intentionally redesigned.

    Special case: thread callouts are distinctive enough that matching them
    against any unclassified ("other") span is treated as incompatible.
    """
    if type_a == type_b:
        return False
    if "other" in (type_a, type_b):
        # Thread specs are so self-describing that any unclassified match is
        # suspicious (the thread nominal diameter is rarely extracted as a
        # bare decimal; more likely a mismatched span was selected).
        strong = type_a if type_b == "other" else type_b
        return strong == "thread"
    return frozenset({type_a, type_b}) in _INCOMPATIBLE_TYPE_PAIRS


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

    # Build set of count-prefix values to exclude from numeric tokens.
    # e.g., "3X" → exclude 3.0 so the count multiplier doesn't become the
    # "primary" dimension value and cause false mismatch penalties.
    count_prefix_values: set[float] = set()
    for ct in count_tokens:
        m = re.match(r'^(\d+)X$', ct)
        if m:
            try:
                count_prefix_values.add(float(m.group(1)))
            except ValueError:
                pass

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
            val = float(match)
            # Skip count-prefix values (e.g., the "3" in "3X Ø.157")
            if val not in count_prefix_values:
                numeric_tokens.append((val, match))
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

_WELD_PROCESS_PATTERNS = (
    (re.compile(r"\bFILLET\b"), "fillet"),
    (re.compile(r"\bGROOVE\b"), "groove"),
    (re.compile(r"\bPLUG\b"), "plug"),
    (re.compile(r"\bSLOT\b"), "slot"),
    (re.compile(r"\bSPOT\b"), "spot"),
    (re.compile(r"\bSEAM\b"), "seam"),
    (re.compile(r"\bSURFACING\b"), "surfacing"),
    (re.compile(r"\bBACK(?:ING| WELD)?\b"), "back"),
)

_WELD_SIDE_PATTERNS = (
    (re.compile(r"\bBOTH\s+SIDES\b"), "both_sides"),
    (re.compile(r"\bARROW\s+SIDE\b"), "arrow_side"),
    (re.compile(r"\bOTHER\s+SIDE\b"), "other_side"),
)

_WELD_CONTOUR_PATTERNS = (
    (re.compile(r"\bFLUSH\b"), "flush"),
    (re.compile(r"\bCONVEX\b"), "convex"),
    (re.compile(r"\bCONCAVE\b"), "concave"),
)

_WELD_SIZE_RE = re.compile(r"(?<![A-Z0-9])((?:\d+(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?)(?=\s*(?:FILLET|GROOVE|PLUG|SLOT|SPOT|SEAM|SURFACING|BACK(?:ING| WELD)?))")
_WELD_DIMENSION_TOKEN_RE = r"(?:\d+\.\d+|\d+|\.\d+)"
_WELD_LENGTH_PITCH_RE = re.compile(rf"\b(?:LENGTH\s*)?({_WELD_DIMENSION_TOKEN_RE})\s*-\s*({_WELD_DIMENSION_TOKEN_RE})\b")
_WELD_LENGTH_ONLY_RE = re.compile(rf"\bLENGTH\s+({_WELD_DIMENSION_TOKEN_RE})\b")
_WELD_TAIL_RE = re.compile(r"\b(?:TAIL|NOTE|DETAIL)\s*[:\-]?\s*(.+)$")
_WELD_ALLOWED_CONNECTORS = {"ALL", "AROUND", "BOTH", "SIDES", "ARROW", "SIDE", "OTHER", "LENGTH", "TAIL", "NOTE", "DETAIL", "TYP", "WELD"}

_SURFACE_FINISH_ERROR_MALFORMED = "surface_finish_malformed"
_SURFACE_FINISH_ERROR_UNSUPPORTED = "surface_finish_unsupported"
_SURFACE_FINISH_EMPTY_NO_MATCH = "surface_finish_no_match"

_FIT_ERROR_MALFORMED = "fit_malformed"
_FIT_ERROR_UNSUPPORTED = "fit_unsupported"
_FIT_EMPTY_NO_MATCH = "fit_no_match"

_SURFACE_FINISH_INDICATOR_RE = re.compile(r"\bRA\b|(?<![A-Z])RA(?=[0-9.])", re.IGNORECASE)
_SURFACE_FINISH_VALUE_TOKEN_RE = r"(?:\d+(?:\.\d+)?|\.\d+)"
_SURFACE_FINISH_FORWARD_RE = re.compile(
    rf"\bRA\s*({_SURFACE_FINISH_VALUE_TOKEN_RE})\s*(?:([µUM]{1,2}|MICROMETER(?:S)?|MICRON(?:S)?)\b)?",
    re.IGNORECASE,
)
_SURFACE_FINISH_REVERSED_RE = re.compile(
    rf"\b({_SURFACE_FINISH_VALUE_TOKEN_RE})\s*(?:([µUM]{1,2}|MICROMETER(?:S)?|MICRON(?:S)?)\s*)?RA\b",
    re.IGNORECASE,
)
_SURFACE_FINISH_UNSUPPORTED_RE = re.compile(r"\bRZ\b|\bRMS\b|\bCLA\b", re.IGNORECASE)

_FIT_TOKEN_RE = re.compile(r"([A-Z]\d+)\s*/\s*([a-z]\d+)")
_FIT_NEAR_MISS_RE = re.compile(r"\b([A-Z]\d+)\s*/?\s*([A-Za-z])\b")


WELD_ERROR_MALFORMED = "weld_malformed"
WELD_ERROR_UNSUPPORTED = "weld_unsupported"
WELD_EMPTY_NO_MATCH = "weld_no_match"


def extract_semantic_callout(
    pdf_spans: Sequence[TextSpan],
    form3_requirement: Optional[str] = None,
) -> SemanticCallout:
    """Build the typed semantic extraction envelope from PDF-first inputs.

    PDF spans are always authoritative when present. Form 3 requirement text is only
    used as advisory context and never becomes the source of truth when drawing spans
    exist. The dispatcher remains bounded: GD&T feature control frames and a practical
    first-pass weld subset parse into typed payloads, while other semantic families
    still return explicit inspectable empty states.
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

    status, gdt_payload, weld_payload, surface_finish_payload, fit_payload = _extract_semantic_payload(normalized_text)

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
        weld=weld_payload,
        surface_finish=surface_finish_payload,
        fit=fit_payload,
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


def _extract_semantic_payload(
    normalized_text: Optional[str],
) -> Tuple[
    SemanticParserStatus,
    Optional[GdtSemanticPayload],
    Optional[WeldSemanticPayload],
    Optional[SurfaceFinishSemanticPayload],
    Optional[FitSemanticPayload],
]:
    if not normalized_text:
        return (
            SemanticParserStatus(
                state="empty",
                parser_family="surface_finish",
                reason_code=_SURFACE_FINISH_EMPTY_NO_MATCH,
                detail="no semantic source text was available for bounded semantic parsing",
            ),
            None,
            None,
            None,
            None,
        )

    gdt_result = _parse_gdt_frame(normalized_text)
    if isinstance(gdt_result, ParsedGdtFrame):
        return (
            SemanticParserStatus(
                state="parsed",
                parser_family="gdt",
                reason_code=None,
                detail="parsed feature control frame from PDF spans",
            ),
            GdtSemanticPayload(
                frame_text=gdt_result.frame_text,
                control_type=gdt_result.control_type,
                tolerance_text=gdt_result.tolerance_text,
                datum_refs=gdt_result.datum_refs,
                modifiers=gdt_result.modifiers,
            ),
            None,
            None,
            None,
        )
    if isinstance(gdt_result, str):
        return (
            SemanticParserStatus(
                state="error",
                parser_family="gdt",
                reason_code="gdt_malformed_frame",
                detail=gdt_result,
            ),
            None,
            None,
            None,
            None,
        )

    weld_result = _parse_weld_callout(normalized_text)
    if isinstance(weld_result, ParsedWeldCallout):
        return (
            SemanticParserStatus(
                state="parsed",
                parser_family="weld",
                reason_code=None,
                detail="parsed bounded weld callout from authoritative semantic text",
            ),
            None,
            WeldSemanticPayload(
                process=weld_result.process,
                size=weld_result.size,
                contour=weld_result.contour,
                side=weld_result.side,
                length=weld_result.length,
                pitch=weld_result.pitch,
                tail=weld_result.tail,
                all_around=weld_result.all_around,
            ),
            None,
            None,
        )
    if isinstance(weld_result, tuple):
        reason_code, detail = weld_result
        return (
            SemanticParserStatus(
                state="error",
                parser_family="weld",
                reason_code=reason_code,
                detail=detail,
            ),
            None,
            None,
            None,
            None,
        )

    surface_finish_result = _parse_surface_finish_callout(normalized_text)
    if isinstance(surface_finish_result, ParsedSurfaceFinishCallout):
        return (
            SemanticParserStatus(
                state="parsed",
                parser_family="surface_finish",
                reason_code=None,
                detail="parsed bounded surface finish callout from authoritative semantic text",
            ),
            None,
            None,
            SurfaceFinishSemanticPayload(
                canonical_text=surface_finish_result.canonical_text,
                roughness_value=surface_finish_result.roughness_value,
                units=surface_finish_result.units,
                value_micrometers=surface_finish_result.value_micrometers,
                indicator=surface_finish_result.indicator,
            ),
            None,
        )
    if isinstance(surface_finish_result, tuple):
        reason_code, detail = surface_finish_result
        return (
            SemanticParserStatus(
                state="error",
                parser_family="surface_finish",
                reason_code=reason_code,
                detail=detail,
            ),
            None,
            None,
            None,
            None,
        )

    fit_result = _parse_fit_callout(normalized_text)
    if isinstance(fit_result, ParsedFitCallout):
        return (
            SemanticParserStatus(
                state="parsed",
                parser_family="fit",
                reason_code=None,
                detail="parsed bounded fit callout from authoritative semantic text",
            ),
            None,
            None,
            None,
            FitSemanticPayload(
                canonical_text=fit_result.canonical_text,
                fit_class=fit_result.fit_class,
                hole_class=fit_result.hole_class,
                shaft_class=fit_result.shaft_class,
                basis=fit_result.basis,
                standard_hint=fit_result.standard_hint,
            ),
        )
    if isinstance(fit_result, tuple):
        reason_code, detail = fit_result
        return (
            SemanticParserStatus(
                state="error",
                parser_family="fit",
                reason_code=reason_code,
                detail=detail,
            ),
            None,
            None,
            None,
            None,
        )

    return (
        SemanticParserStatus(
            state="empty",
            parser_family="surface_finish",
            reason_code=_SURFACE_FINISH_EMPTY_NO_MATCH,
            detail="text did not match the bounded GD&T, weld, surface finish, or fit grammar",
        ),
        None,
        None,
        None,
        None,
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


def _parse_weld_callout(normalized_text: str) -> Optional[ParsedWeldCallout | Tuple[str, str]]:
    uppercase_text = normalized_text.upper()

    process = None
    for pattern, process_name in _WELD_PROCESS_PATTERNS:
        if pattern.search(uppercase_text):
            process = process_name
            break

    if process is None:
        return None

    if process == "fillet" and not _WELD_SIZE_RE.search(uppercase_text):
        return (
            WELD_ERROR_MALFORMED,
            "recognized weld callout is missing a parseable size token before the weld type",
        )

    unknown_tokens = _find_unsupported_weld_tokens(uppercase_text)
    if unknown_tokens:
        joined = ", ".join(unknown_tokens)
        return (
            WELD_ERROR_UNSUPPORTED,
            f"recognized weld callout contains unsupported segment(s): {joined}",
        )

    size_match = _WELD_SIZE_RE.search(uppercase_text)
    size = size_match.group(1) if size_match else None

    length = None
    pitch = None
    length_pitch_match = _WELD_LENGTH_PITCH_RE.search(uppercase_text)
    if length_pitch_match:
        length = length_pitch_match.group(1)
        pitch = length_pitch_match.group(2)
    else:
        length_only_match = _WELD_LENGTH_ONLY_RE.search(uppercase_text)
        if length_only_match:
            length = length_only_match.group(1)

    side = None
    for pattern, side_value in _WELD_SIDE_PATTERNS:
        if pattern.search(uppercase_text):
            side = side_value
            break

    contour = None
    for pattern, contour_value in _WELD_CONTOUR_PATTERNS:
        if pattern.search(uppercase_text):
            contour = contour_value
            break

    tail_match = _WELD_TAIL_RE.search(uppercase_text)
    tail = tail_match.group(1).strip() if tail_match else None

    return ParsedWeldCallout(
        process=process,
        size=size,
        contour=contour,
        side=side,
        length=length,
        pitch=pitch,
        tail=tail,
        all_around=True if "ALL AROUND" in uppercase_text else None,
    )


def _parse_surface_finish_callout(normalized_text: str) -> Optional[ParsedSurfaceFinishCallout | Tuple[str, str]]:
    uppercase_text = normalized_text.upper()

    if _SURFACE_FINISH_UNSUPPORTED_RE.search(uppercase_text):
        unsupported = _SURFACE_FINISH_UNSUPPORTED_RE.search(uppercase_text)
        assert unsupported is not None
        return (
            _SURFACE_FINISH_ERROR_UNSUPPORTED,
            f"recognized surface finish callout uses unsupported roughness family: {unsupported.group(0).upper()}",
        )

    if not _SURFACE_FINISH_INDICATOR_RE.search(uppercase_text):
        return None

    match = _SURFACE_FINISH_FORWARD_RE.search(normalized_text) or _SURFACE_FINISH_REVERSED_RE.search(normalized_text)
    if match is None:
        return (
            _SURFACE_FINISH_ERROR_MALFORMED,
            "recognized surface finish indicator but missing a bounded Ra roughness value",
        )

    value_text = _normalize_decimal_text(match.group(1))
    units_text = _normalize_surface_finish_units(match.group(2))
    if units_text is None:
        units_text = "um"

    return ParsedSurfaceFinishCallout(
        canonical_text=f"Ra {value_text} {units_text}",
        roughness_value=value_text,
        units=units_text,
        value_micrometers=value_text if units_text == "um" else None,
        indicator="Ra",
    )


def _parse_fit_callout(normalized_text: str) -> Optional[ParsedFitCallout | Tuple[str, str]]:
    match = _FIT_TOKEN_RE.search(normalized_text)
    if match is None:
        near_miss = _FIT_NEAR_MISS_RE.search(normalized_text)
        if near_miss is not None:
            return (
                _FIT_ERROR_MALFORMED,
                "recognized fit designator but missing a complete paired hole/shaft class",
            )
        return None

    hole_class = match.group(1)
    shaft_class = match.group(2)

    canonical_text = f"{hole_class}/{shaft_class}"
    basis = "hole_basis" if hole_class.startswith("H") else "shaft_basis" if shaft_class.startswith("h") else "unspecified_basis"

    return ParsedFitCallout(
        canonical_text=canonical_text,
        fit_class=canonical_text,
        hole_class=hole_class,
        shaft_class=shaft_class,
        basis=basis,
        standard_hint="iso_limits_and_fits",
    )


def _find_unsupported_weld_tokens(uppercase_text: str) -> List[str]:
    cleaned = uppercase_text
    cleaned = _WELD_TAIL_RE.sub(" ", cleaned)
    cleaned = _WELD_LENGTH_PITCH_RE.sub(" ", cleaned)
    cleaned = _WELD_LENGTH_ONLY_RE.sub(" ", cleaned)
    cleaned = _WELD_SIZE_RE.sub(" ", cleaned)

    for pattern, _ in _WELD_PROCESS_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    for pattern, _ in _WELD_SIDE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    for pattern, _ in _WELD_CONTOUR_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)

    cleaned = cleaned.replace("ALL AROUND", " ")
    cleaned = re.sub(r"[^A-Z0-9./\-]+", " ", cleaned)

    leftovers = []
    for token in cleaned.split():
        if token in _WELD_ALLOWED_CONNECTORS:
            continue
        if re.fullmatch(r"(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?", token):
            continue
        leftovers.append(token)
    return leftovers


def _normalize_decimal_text(value_text: str) -> str:
    if value_text.startswith("."):
        value_text = f"0{value_text}"
    if "." in value_text:
        value_text = value_text.rstrip("0").rstrip(".")
    return value_text


def _normalize_surface_finish_units(units_text: Optional[str]) -> Optional[str]:
    if units_text is None:
        return None

    token = units_text.strip().lower()
    if token in {"u", "um", "µm", "μm", "micrometer", "micrometers", "micron", "microns"}:
        return "um"
    return None


def _is_gdt_tolerance_token(token: str) -> bool:
    return bool(_GDT_TOLERANCE_RE.fullmatch(token))
