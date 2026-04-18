from __future__ import annotations

import math
import re
from typing import Optional, Dict, List, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# CLS-01: Adjacency-bleed detection constants
# ---------------------------------------------------------------------------

# Advisory flag stored in DeltaItem.confidence_flags when the classifier
# detects that the Rev B span may have been assembled from multiple adjacent
# balloon annotations merged by the PDF extractor via a "/" separator.
_BLEED_FLAG = "Rev B text may contain adjacent balloon content"

# Regex that splits on whitespace-bounded slashes only (e.g. "A / B") so that
# embedded fractions ("1/4-20"), ratios without spaces ("H7/p6"), and
# drawing-standard callouts are not mis-split.
_BLEED_SPLIT_RE = re.compile(r"\s+/\s+")

# ---------------------------------------------------------------------------
# CLS-03: Asymmetric tolerance shape detection
# ---------------------------------------------------------------------------

# Matches asymmetric (bilateral_stacked / unilateral_stacked) tolerance forms
# in raw requirement text.  Covers:
#   - Standard decimals:  "+0.3 / -0.1",  "+0.3° / −0.1°"
#   - Leading decimals:   "+.005 / -.003"
#   - Unicode minus:      "+0.3 / −0.1" (U+2212)
# The pattern requires an explicit sign on at least the first value so that
# plain dimension ranges like "1.0 / 2.0" are NOT matched.
_ASYMMETRIC_SHAPE_RE = re.compile(
    r"[+\-]\s*\.?\d+(?:\.\d*)?[°]?"    # first signed tolerance value (leading-decimal OK)
    r"\s*(?:/\s*)"                       # separator
    r"[\-\u2212]\s*\.?\d+(?:\.\d*)?[°]?"  # second signed value (negative)
)


def _is_symmetric_to_asymmetric_kind_change(tolerance_comparison: "ToleranceComparison") -> bool:
    """Return True when Rev A has a plus_minus (symmetric) kind and Rev B has a
    bilateral_stacked or unilateral_stacked (asymmetric) kind.

    This is the hallmark of a ±T → +a/−b formatting change that the classifier
    must detect as 'changed' regardless of whether the absolute limits happen to
    agree numerically (tolerances_match=True).
    """
    revA_kind = tolerance_comparison.revA_tolerance.kind
    revB_kind = tolerance_comparison.revB_tolerance.kind
    _symmetric = {"plus_minus"}
    _asymmetric = {"bilateral_stacked", "unilateral_stacked"}
    return revA_kind in _symmetric and revB_kind in _asymmetric

from delta_preservation.reconcile.anchors import Anchor
from delta_preservation.reconcile.match import Match
from delta_preservation.reconcile.normalize import (
    are_requirement_types_incompatible,
    classify_requirement_type,
    extract_semantic_callout,
    parse_requirement,
)
from delta_preservation.reconcile.semantic_compare import compare_semantic_callouts
from delta_preservation.io.pdf import TextSpan
from delta_preservation.types import SemanticCallout
from delta_preservation.reconcile.exclusion import (
    estimate_page_dimensions,
    is_boilerplate_candidate_text,
    span_is_excluded_for_annotation_search,
)

if TYPE_CHECKING:
    from delta_preservation.reconcile.tolerance_pdf import ToleranceComparison


@dataclass
class DeltaItem:
    """Classification result for a single Rev A characteristic."""
    char_no: int
    status: str  # "unchanged", "changed", "removed", "uncertain", "added"
    confidence: float
    reasons: List[str]
    component_scores: Dict[str, float]
    match: Optional[Match] = None
    added_span: Optional[TextSpan] = None  # For added characteristics
    confidence_flags: List[str] = field(default_factory=list)
    # Internal-only CLS-02 metadata: not persisted to the Pydantic packet
    added_requirement_text: Optional[str] = None
    added_bbox: Optional[Tuple[float, float, float, float]] = None
    added_page: Optional[int] = None


@dataclass
class AddedCharacteristic:
    """Represents a newly detected characteristic in Rev B that doesn't exist in Rev A."""
    span: TextSpan
    requirement_text: str
    confidence: float
    reasons: List[str]


_SEMANTIC_GDT_CONTROL_HINTS: tuple[tuple[str, str], ...] = (
    ("⌖✢", "position"),
    ("⌖", "position"),
    ("⌒", "profile_of_a_line"),
    ("⌓", "profile_of_a_surface"),
    ("⟂", "perpendicularity"),
    ("⏥", "flatness"),
    ("∥", "parallelism"),
    ("∠", "angularity"),
    ("⊙", "concentricity"),
    ("◎", "concentricity"),
    ("⚪", "circularity"),
    ("○", "circularity"),
    ("↗", "circular_runout"),
    ("⌿", "circular_runout"),
    ("⌰", "total_runout"),
    ("⟃", "total_runout"),
)
_SEMANTIC_GDT_WORD_HINTS: tuple[tuple[str, str], ...] = (
    ("TOTAL RUNOUT", "total_runout"),
    ("CIRCULAR RUNOUT", "circular_runout"),
    ("RUNOUT", "circular_runout"),
    ("CIRCULARITY", "circularity"),
    ("CONCENTRICITY", "concentricity"),
    ("POSITION", "position"),
    ("FLATNESS", "flatness"),
    ("PERPENDICULARITY", "perpendicularity"),
    ("PARALLELISM", "parallelism"),
    ("ANGULARITY", "angularity"),
    ("PROFILE", "profile_of_a_surface"),
)


def _semantic_identity_from_callout(callout: Optional[SemanticCallout]) -> tuple[str, Optional[str]] | None:
    if callout is None or callout.status.state != "parsed":
        return None
    family = callout.status.parser_family
    if family == "gdt" and callout.gdt is not None:
        return (family, callout.gdt.control_type)
    if family == "weld" and callout.weld is not None:
        return (family, callout.weld.process)
    if family == "surface_finish" and callout.surface_finish is not None:
        return (family, callout.surface_finish.indicator)
    if family == "fit" and callout.fit is not None:
        return (family, callout.fit.basis)
    return (family, None)


def _gdt_control_hint_from_text(text: str) -> Optional[str]:
    normalized = " ".join(text.strip().upper().split())
    for symbol, control_type in _SEMANTIC_GDT_CONTROL_HINTS:
        if symbol in text:
            return control_type
    for token, control_type in _SEMANTIC_GDT_WORD_HINTS:
        if token in normalized:
            return control_type
    return None


def _semantic_identity_from_text(
    text: str,
    *,
    pdf_spans: Optional[List[TextSpan]] = None,
) -> tuple[str, Optional[str]] | None:
    semantic_identity = _semantic_identity_from_callout(
        extract_semantic_callout(pdf_spans=pdf_spans or [], form3_requirement=None if pdf_spans else text)
    )
    if semantic_identity is not None:
        return semantic_identity
    gdt_control_hint = _gdt_control_hint_from_text(text)
    if gdt_control_hint is not None:
        return ("gdt", gdt_control_hint)
    return None


def _semantic_identities_conflict(
    left: tuple[str, Optional[str]] | None,
    right: tuple[str, Optional[str]] | None,
) -> bool:
    if left is None and right is None:
        return False
    if left is None or right is None:
        # CLS-02 uses grouped added evidence. A grouped added string can carry an
        # extra GD&T fragment even when the removed-side anchor is a plain
        # dimension and the numeric/type gates already prove compatibility for
        # the relevant chunk. Treat one-sided semantic identity as informative,
        # not a hard conflict, and only reject when both sides express
        # incompatible semantic families/details.
        return False
    left_family, left_detail = left
    right_family, right_detail = right
    if left_family != right_family:
        return True
    if left_family == "gdt" and left_detail and right_detail and left_detail != right_detail:
        return True
    return False


def _gdt_datum_refs_from_text(
    text: str,
    *,
    pdf_spans: Optional[List[TextSpan]] = None,
) -> tuple[str, ...]:
    semantic_callout = extract_semantic_callout(
        pdf_spans=pdf_spans or [],
        form3_requirement=None if pdf_spans else text,
    )
    if semantic_callout.status.state == "parsed" and semantic_callout.gdt is not None:
        return tuple(semantic_callout.gdt.datum_refs)
    tokens: List[str] = []
    source_texts = [span.text for span in pdf_spans] if pdf_spans else [text]
    for raw_token in source_texts:
        normalized = " ".join(raw_token.strip().upper().split())
        if re.fullmatch(r"[A-HJ-NP-Z](?:-[A-HJ-NP-Z])?", normalized):
            tokens.append(normalized)
    return tuple(tokens)


def _looks_like_adjacency_bleed(span_text: str, anchor_text: str) -> bool:
    """Return True when span_text appears to be a PDF-merged multi-balloon annotation.

    Detection logic:
    1. Split span_text on whitespace-bounded slashes (spaces on both sides).
       Return False immediately if fewer than 2 chunks result — no bleed possible.
    2. Parse anchor_text to extract its primary numeric token and non-stopword keywords.
    3. A chunk is "anchor-bearing" only when it contains:
       - the anchor's primary numeric token, OR
       - at least two normalized non-stopword keywords from the anchor.
    4. A *different* chunk is "foreign-count-bearing" when it:
       - contains a count prefix like "2X" or "4 x", AND
       - does NOT contain the anchor's primary numeric token.
    5. Return True only when BOTH an anchor-bearing chunk and a foreign-count-bearing
       chunk are found in different positions.

    The whitespace-requirement in the split regex deliberately excludes:
    - Embedded fractions: "1/4-20 UNC"
    - Fit-code pairs: "H7/p6"
    - Tolerance/note text without surrounding spaces
    """
    chunks = _BLEED_SPLIT_RE.split(span_text)
    if len(chunks) < 2:
        return False

    anchor_fp = parse_requirement(anchor_text)
    anchor_numerics = {v for v, _ in anchor_fp.numeric_tokens}
    anchor_primary = max(anchor_numerics) if anchor_numerics else None

    # Extract non-stopword keywords from the anchor's normalized text.
    _stop = {"", "A", "AN", "THE", "OF", "IN", "AT", "X", "MM", "IN", "DIAMETER"}
    anchor_kw = {
        w for w in re.sub(r"[()\/\±\+\-]", " ", anchor_fp.norm_text.upper()).split()
        if w not in _stop and not re.fullmatch(r"[\d.]+", w)
    }

    # Count-prefix pattern: "2X", "4 x", "2 X", etc.
    _count_re = re.compile(r"\b\d+\s*[Xx]\b")

    anchor_bearing_idx: Optional[int] = None
    foreign_count_idx: Optional[int] = None

    for idx, chunk in enumerate(chunks):
        chunk_upper = chunk.upper()

        # Check anchor-bearing: primary numeric OR ≥2 anchor keywords present
        is_anchor_bearing = False
        if anchor_primary is not None and str(anchor_primary) in chunk:
            is_anchor_bearing = True
        if not is_anchor_bearing and len(anchor_kw) >= 2:
            chunk_words = set(re.sub(r"[()\/\±\+\-]", " ", chunk_upper).split())
            if len(anchor_kw & chunk_words) >= 2:
                is_anchor_bearing = True
        # Also try numeric match via float comparison for cases like "13.5" in "Ø13.5"
        if not is_anchor_bearing and anchor_primary is not None:
            try:
                chunk_fp = parse_requirement(chunk)
                chunk_nums = {v for v, _ in chunk_fp.numeric_tokens}
                if anchor_primary in chunk_nums:
                    is_anchor_bearing = True
            except Exception:
                pass

        if is_anchor_bearing:
            if anchor_bearing_idx is None:
                anchor_bearing_idx = idx
            continue  # Don't also mark as foreign-count

        # Check foreign-count-bearing: has count prefix AND lacks anchor primary
        has_count_prefix = bool(_count_re.search(chunk))
        lacks_anchor_primary = True
        if anchor_primary is not None:
            try:
                chunk_fp = parse_requirement(chunk)
                chunk_nums = {v for v, _ in chunk_fp.numeric_tokens}
                if anchor_primary in chunk_nums:
                    lacks_anchor_primary = False
            except Exception:
                pass
        if has_count_prefix and lacks_anchor_primary:
            if foreign_count_idx is None:
                foreign_count_idx = idx

    return (
        anchor_bearing_idx is not None
        and foreign_count_idx is not None
        and anchor_bearing_idx != foreign_count_idx
    )


def classify_delta(
    anchor: Anchor,
    match_or_none: Optional[Match],
    location_search_coverage: float = 1.0,
    tolerance_comparison: Optional[ToleranceComparison] = None,
    anchor_semantic_callout: Optional[SemanticCallout] = None,
    matched_semantic_callout: Optional[SemanticCallout] = None,
    revB_text_spans: Optional[List["TextSpan"]] = None,
    matched_span_keys_strong: Optional[Set[Tuple]] = None,
) -> DeltaItem:
    """Classify delta status for a single Rev A anchor.
    
    Classification logic focuses on numeric value matching since PDF spans
    often contain only raw dimension values (e.g., "120") while Form 3
    requirements contain full descriptions (e.g., "Length (120 +/- 0.3 mm)").
    
    Decision rules:
    - If key numeric values from anchor are found in matched span → unchanged
    - If numeric values differ but structural tokens match → changed
    - If no match found → removed
    - Otherwise → uncertain
    
    Args:
        anchor: Rev A anchor with requirement text
        match_or_none: Assigned match or None if no candidate found
        location_search_coverage: Fraction of search window covered (for removed confidence)
        
    Returns:
        DeltaItem with status, confidence, reasons, and component scores
    """
    reasons = []
    
    # Handle removed case
    if match_or_none is None:
        # Pre-compute anchor fingerprint fields used by both last-chance scans so
        # that the keyword-anchor scan at line ~264 can reference them even when
        # revB_text_spans is falsy and the first scan is skipped entirely.
        anchor_fp = parse_requirement(anchor.requirement_raw)
        anchor_numerics = {v for v, _ in anchor_fp.numeric_tokens}
        anchor_primary = max(anchor_numerics) if anchor_numerics else None
        anchor_req_type = classify_requirement_type(anchor.requirement_raw)
        is_notes_anchor = (
            anchor_fp.pattern_class == "note"
            or "NOTES" in anchor.requirement_raw.upper()
        )

        # --- Last-chance identity check ---
        # Before declaring "removed", scan all Rev B spans for the anchor's
        # primary numeric value with a compatible requirement type.  This catches
        # cases where the search window missed the annotation (e.g., because the
        # alignment transform was off due to a new view being added).
        if revB_text_spans:

            if anchor_primary is not None and not is_notes_anchor:
                # Skip the identity check when the anchor's primary value is a
                # common tolerance/small value that appears everywhere on the
                # drawing (e.g., 0.3, 0.005, 0.1).  Only run for distinctive
                # primary dimensions (≥ 1.0) that are unlikely to coincide.
                # Additionally require ≥2 secondary numerics to overlap — a
                # single primary match is too prone to false positives (e.g.,
                # thread callouts like "10-24 UNC" match the number 24 in
                # many unrelated spans).
                anchor_secondary = sorted(anchor_numerics - {anchor_primary})
                is_distinctive = (
                    anchor_primary >= 1.0
                    and len(anchor_numerics) >= 2
                    and anchor_req_type not in ("thread",)
                )
                if is_distinctive:
                    for span in revB_text_spans:
                        span_fp = parse_requirement(span.text)
                        span_numerics = {v for v, _ in span_fp.numeric_tokens}
                        if anchor_primary not in span_numerics:
                            continue
                        # Require the span to be annotation-like (not too many values,
                        # not a tolerance block or notes block)
                        if len(span_numerics) > 5:
                            continue
                        span_text_upper = span.text.strip().upper()
                        if any(kw in span_text_upper for kw in (
                            "UNLESS", "SPECIFIED", "ANGULAR", "FRACTIONAL",
                            "DRAWN", "CHECKED", "TITLE",
                        )):
                            continue
                        # Check requirement type compatibility
                        span_req_type = classify_requirement_type(span.text)
                        if are_requirement_types_incompatible(anchor_req_type, span_req_type):
                            continue
                        # Require at least one secondary numeric overlap to reduce
                        # false positives from coincidental primary matches
                        if anchor_secondary and span_numerics:
                            if not (set(anchor_secondary) & span_numerics):
                                continue
                        # Found a plausible match — reclassify as uncertain
                        return DeltaItem(
                            char_no=anchor.char_no,
                            status="uncertain",
                            confidence=0.45,
                            reasons=[
                                "No candidate in search window, but page-wide scan found "
                                f"primary value {anchor_primary} in Rev B span: \"{span.text.strip()[:80]}\"",
                                "Possible alignment miss — manual review recommended",
                            ],
                            component_scores={
                                "location": 0.0,
                                "text": 0.5,
                                "context": 0.0,
                            },
                            match=None,
                        )

        # --- Keyword-anchor last-chance scan ---
        # When the anchor has no distinctive numeric value (e.g., bare "THRU" or
        # keyword-only requirements), the numeric-based scan above doesn't fire.
        # Instead, look for a Rev B span whose text contains the anchor's requirement
        # as a substring (case-insensitive, Unicode-normalised), excluding title-block
        # and revision-table regions.  This rescues pure-text anchors that were
        # missed because the search window didn't reach the correct annotation.
        if (
            not is_notes_anchor
            and anchor_primary is None
            and len(anchor.requirement_raw.strip()) >= 3
            and revB_text_spans
        ):
            _page_w, _page_h = estimate_page_dimensions(revB_text_spans)
            _anchor_kw = anchor.requirement_raw.strip().upper().replace("\u2212", "-")
            for span in revB_text_spans:
                # Skip title block, revision table, and boilerplate spans using
                # the shared exclusion contract instead of local coordinate checks.
                if span_is_excluded_for_annotation_search(
                    span, page_width=_page_w, page_height=_page_h
                ):
                    continue
                # Skip spans already claimed by another surviving anchor so that
                # keyword anchors (e.g. "THRU") don't wrongly match a grouped
                # characteristic span that belongs to a different char.
                _skw = (span.block_id, span.line_id, span.span_id, span.bbox_pdf)
                if matched_span_keys_strong and _skw in matched_span_keys_strong:
                    continue
                # Note: span_is_excluded_for_annotation_search (called above) already
                # covers boilerplate keywords like UNLESS/SPECIFIED/ANGULAR/etc. via
                # is_boilerplate_candidate_text().  The explicit keyword list that
                # previously lived here has been replaced by the shared contract.
                # Check if anchor keyword is a substring of the span text
                _st_upper = span.text.strip().upper()
                _st_norm = _st_upper.replace("\u2212", "-")
                if _anchor_kw in _st_norm:
                    # When the anchor keyword is the entire span, or is a trailing
                    # suffix (e.g. "THRU" at end of "Ø.266 THRU"), the Rev B span
                    # IS the characteristic annotation → classify as unchanged.
                    # When the keyword is deeply embedded in a longer callout, keep
                    # it as uncertain for reviewer attention.
                    _is_exact_or_suffix = (
                        _st_norm == _anchor_kw
                        or _st_norm.endswith(_anchor_kw)
                        or _st_norm.endswith(" " + _anchor_kw)
                    )
                    _kw_status = "unchanged" if _is_exact_or_suffix else "uncertain"
                    _kw_confidence = 0.65 if _is_exact_or_suffix else 0.45
                    _kw_reason = (
                        "No candidate in search window, but page-wide keyword scan "
                        f"found anchor text as characteristic suffix in Rev B span: \"{span.text.strip()[:80]}\""
                        if _is_exact_or_suffix else
                        "No candidate in search window, but page-wide keyword scan "
                        f"found anchor text in Rev B span: \"{span.text.strip()[:80]}\""
                    )
                    return DeltaItem(
                        char_no=anchor.char_no,
                        status=_kw_status,
                        confidence=_kw_confidence,
                        reasons=[
                            _kw_reason,
                            "Possible alignment miss — manual review recommended",
                        ],
                        component_scores={
                            "location": 0.0,
                            "text": 0.5,
                            "context": 0.0,
                        },
                        match=None,
                    )

        confidence = min(0.9, location_search_coverage)
        return DeltaItem(
            char_no=anchor.char_no,
            status="removed",
            confidence=confidence,
            reasons=["No candidate found within search window"],
            component_scores={
                "location": 0.0,
                "text": 0.0,
                "context": 0.0
            },
            match=None
        )
    
    # Extract component scores from match
    candidate = match_or_none.candidate
    location_score = candidate.location_score
    text_score = candidate.text_score
    context_score = candidate.context_score

    # --- Notes-block content guard (match path) ---
    # If the matched span is notes-block text (numbered general notes like
    # "1. QTY: 2  2. ALL DIMS IN INCHES  3. MATL: ..."), it cannot be a valid
    # Rev B characteristic callout.  Return "removed" immediately.
    # Also catches short "N." patterns that are notes-block item counters
    # (e.g., "1." by itself is the list-item number in a general notes block).
    import re as _re
    _span_text_stripped = candidate.span.text.strip()
    _span_text_upper = _span_text_stripped.upper()
    _notes_keywords = ("QTY", "ALL DIMS", "MATL:", "FINISH NOTED", "BREAK ALL EDGES",
                        "UNLESS OTHERWISE", "TOLERANCES", "PROPRIETARY")
    _is_notes_span = (
        any(kw in _span_text_upper for kw in _notes_keywords)
        or bool(_re.fullmatch(r'\d+\.', _span_text_stripped))  # bare "1.", "2.", etc.
    )
    if _is_notes_span:
        return DeltaItem(
            char_no=anchor.char_no,
            status="removed",
            confidence=min(0.8, location_search_coverage),
            reasons=[
                "Matched span is notes-block content — not a valid Rev B characteristic callout",
                f"Span text: \"{candidate.span.text.strip()[:80]}\"",
            ],
            component_scores={
                "location": location_score,
                "text": 0.0,
                "context": context_score,
            },
            match=None,
        )

    # Parse fingerprints from both anchor requirement and matched span
    anchor_fp = parse_requirement(anchor.requirement_raw)
    matched_fp = parse_requirement(candidate.span.text)

    semantic_result = None
    if anchor_semantic_callout is not None or matched_semantic_callout is not None:
        semantic_result = compare_semantic_callouts(anchor_semantic_callout, matched_semantic_callout)

    if semantic_result is not None and semantic_result.mode == "semantic":
        semantic_reasons = list(semantic_result.reason_fragments)
        if semantic_result.equal:
            return DeltaItem(
                char_no=anchor.char_no,
                status="unchanged",
                confidence=max(0.85, 0.45 * location_score + 0.4 * text_score + 0.15 * context_score),
                reasons=semantic_reasons + ["Semantic family agreement overrides formatting-only numeric/text variation"],
                component_scores={
                    "location": location_score,
                    "text": text_score,
                    "context": context_score,
                },
                match=match_or_none,
            )

        return DeltaItem(
            char_no=anchor.char_no,
            status="changed",
            confidence=max(0.8, 0.35 * location_score + 0.45 * text_score + 0.2 * context_score),
            reasons=semantic_reasons + ["Semantic family delta indicates a meaningful drawing requirement change"],
            component_scores={
                "location": location_score,
                "text": text_score,
                "context": context_score,
            },
            match=match_or_none,
        )

    if semantic_result is not None and semantic_result.mode in {"fallback", "incompatible"}:
        reasons.extend(semantic_result.reason_fragments)

    # --- Requirement-type incompatibility check ---
    # If the Rev A and Rev B requirements belong to contextually incompatible
    # semantic categories (e.g. a fillet radius vs. an angle, a thread callout
    # vs. a plain dimension, a GD&T frame vs. a chamfer callout) the numeric
    # comparison below will produce meaningless results.  Return "uncertain"
    # immediately so a reviewer can decide whether the design intent changed.
    anchor_req_type = classify_requirement_type(anchor.requirement_raw)
    matched_req_type = classify_requirement_type(candidate.span.text)
    if are_requirement_types_incompatible(anchor_req_type, matched_req_type):
        # Special case: anchor text is effectively a substring of the matched span text.
        # This happens when a grouped span combines the anchor's annotation with a
        # preceding feature callout (e.g., "10-24 UNC THRU 120° APART" is a substring
        # of "3X Ø.157 THRU / 10−24 UNC THRU / 120° APART").  The requirement IS
        # present; it was just grouped with a different-type lead annotation.
        # Unicode-normalise hyphens (U+2212 → U+002D) before checking.
        _anchor_norm = anchor.requirement_raw.replace("\u2212", "-").strip()
        _span_norm = candidate.span.text.replace("\u2212", "-")
        if _anchor_norm and _anchor_norm in _span_norm:
            # Anchor text found verbatim in matched span → treat as unchanged
            reasons_compat = reasons + [
                f"Anchor text found verbatim in grouped Rev B span (type mismatch suppressed)",
                f"Rev A: \"{anchor.requirement_raw}\" ⊆ Rev B: \"{candidate.span.text}\"",
            ]
            return DeltaItem(
                char_no=anchor.char_no,
                status="unchanged",
                confidence=max(0.7, 0.6 * location_score + 0.1),
                reasons=reasons_compat,
                component_scores={
                    "location": location_score,
                    "text": 0.8,
                    "context": context_score,
                },
                match=match_or_none,
            )
        # When requirement types are incompatible AND location confidence is low,
        # this is almost certainly a false match — classify as removed.
        if location_score < 0.3:
            return DeltaItem(
                char_no=anchor.char_no,
                status="removed",
                confidence=min(0.75, location_search_coverage),
                reasons=reasons + [
                    f"Requirement type mismatch: Rev A is '{anchor_req_type}', Rev B is '{matched_req_type}'",
                    "Low location confidence + type mismatch → likely removed",
                ],
                component_scores={
                    "location": location_score,
                    "text": 0.0,
                    "context": context_score,
                },
                match=None,
            )
        # Rescue path: if revB spans are available and we have strong-match tracking,
        # scan for an unmatched span that is type-compatible and passes the limits-form
        # check.  This handles the case where the correct span was mis-assigned to
        # another anchor via a weak (limits-form only) path and the true match is
        # accessible as a non-strong-claimed span.
        if revB_text_spans is not None and matched_span_keys_strong is not None:
            _anchor_fp = parse_requirement(anchor.requirement_raw)
            _anchor_numerics = {v for v, _ in _anchor_fp.numeric_tokens}
            _anchor_primary = max(_anchor_numerics) if _anchor_numerics else None
            if _anchor_primary is not None and _anchor_primary > 0:
                _small = [v for v in _anchor_numerics if 0 < v < _anchor_primary]
                _tol_est = min(_small) if _small else None
                _rel_tol = _tol_est / abs(_anchor_primary) if _tol_est else None
                for _rspan in revB_text_spans:
                    _rkey = (_rspan.block_id, _rspan.line_id, _rspan.span_id, _rspan.bbox_pdf)
                    if _rkey in matched_span_keys_strong:
                        continue
                    _rtype = classify_requirement_type(_rspan.text)
                    if are_requirement_types_incompatible(anchor_req_type, _rtype):
                        continue
                    _rfp = parse_requirement(_rspan.text)
                    _rnums = {v for v, _ in _rfp.numeric_tokens}
                    _rprimary = max(_rnums) if _rnums else None
                    if _rprimary is None:
                        continue
                    # Check if anchor primary is in the span (direct match)
                    if _anchor_primary in _rnums:
                        return DeltaItem(
                            char_no=anchor.char_no,
                            status="unchanged",
                            confidence=max(0.65, 0.5 * location_score + 0.2),
                            reasons=reasons + [
                                f"Requirement type mismatch with assigned span suppressed: "
                                f"found compatible Rev B span with primary value {_anchor_primary}",
                                f"Compatible span: \"{_rspan.text.strip()[:80]}\"",
                            ],
                            component_scores={
                                "location": location_score,
                                "text": 0.5,
                                "context": context_score,
                            },
                            match=match_or_none,
                        )
                    # Check limits-form match against the rescue span
                    if _tol_est is not None and _rel_tol is not None and _rel_tol < 0.15:
                        _pdiff = abs(_anchor_primary - _rprimary)
                        if _pdiff <= 1.5 * _tol_est:
                            # Additional stacked-limits verification for rescue spans
                            _rescue_ok = True
                            if len(_rnums) == 2:
                                _rnums_sorted = sorted(_rnums, reverse=True)
                                _rescue_upper = _rnums_sorted[0]
                                _has_zero_pos = 0.0 in _anchor_numerics
                                _exp_upper = _anchor_primary if _has_zero_pos else _anchor_primary + _tol_est
                                if abs(_rescue_upper - _exp_upper) > _tol_est * 0.15 + 1e-9:
                                    _rescue_ok = False
                            if _rescue_ok:
                                return DeltaItem(
                                    char_no=anchor.char_no,
                                    status="unchanged",
                                    confidence=max(0.6, 0.4 * location_score + 0.2),
                                    reasons=reasons + [
                                        f"Requirement type mismatch with assigned span suppressed: "
                                        f"found compatible limits-form Rev B span",
                                        f"Compatible span: \"{_rspan.text.strip()[:80]}\"",
                                        f"Limits-form rescue: anchor {_anchor_primary} ± {_tol_est} ≈ {_rprimary}",
                                    ],
                                    component_scores={
                                        "location": location_score,
                                        "text": 0.4,
                                        "context": context_score,
                                    },
                                    match=match_or_none,
                                )
        return DeltaItem(
            char_no=anchor.char_no,
            status="uncertain",
            confidence=max(0.55, 0.5 * location_score),
            reasons=reasons + [
                f"Requirement type mismatch: Rev A is '{anchor_req_type}', Rev B is '{matched_req_type}'",
                f"Rev A: \"{anchor.requirement_raw}\" — Rev B: \"{candidate.span.text}\"",
                "Contextually incompatible requirement types; manual review required",
            ],
            component_scores={
                "location": location_score,
                "text": 0.0,
                "context": context_score,
            },
            match=match_or_none,
        )

    # Guard: if the matched span is a notes block but the anchor is NOT a notes
    # anchor, the match is a wrong-copy.  Notes block text typically starts with
    # a list-item pattern ("1. QTY:", "2. ALL DIMS", etc.).  A dimension anchor
    # matching such a span has grabbed an unrelated numeric (e.g., "1." in the
    # notes) and should be classified as removed.
    _notes_item_re = re.compile(r'^\d+\.\s+(QTY|ALL DIMS|MATL|FINISH|BREAK|INTERPRET)', re.IGNORECASE)
    _matched_is_notes_block = bool(_notes_item_re.match(" ".join(candidate.span.text.strip().upper().split())))
    if _matched_is_notes_block and anchor_fp.pattern_class != "note" and "NOTES" not in anchor.requirement_raw.upper():
        return DeltaItem(
            char_no=anchor.char_no,
            status="removed",
            confidence=min(0.80, location_search_coverage),
            reasons=reasons + [
                "Matched span is a notes block but anchor is not a notes anchor",
                f"Notes block text: \"{candidate.span.text.strip()[:80]}\"",
                "Wrong-copy match rejected: anchor should be removed",
            ],
            component_scores={
                "location": location_score,
                "text": 0.0,
                "context": context_score,
            },
            match=None,
        )

    # Check if this is a notes-type characteristic
    # Notes blocks should be compared by text pattern, not numeric values
    is_notes_type = anchor_fp.pattern_class == "note" or "NOTES" in anchor.requirement_raw.upper()
    
    if is_notes_type:
        # For notes blocks, check if the matched span contains "NOTES" header
        if "NOTES" in matched_fp.norm_text or "NOTE" in matched_fp.norm_text:
            # If we have Rev B spans, do a content comparison to detect note changes
            # (e.g., QTY change, different material, different finish requirement)
            if revB_text_spans is not None:
                mx0, my0, mx1, my1 = candidate.span.bbox_pdf
                matched_cx = (mx0 + mx1) / 2
                matched_cy = (my0 + my1) / 2
                # Use a wider search for notes blocks: notes content often spans a large
                # rectangular area. Use generous horizontal AND vertical radii.
                _notes_h_radius = 800.0
                _notes_v_radius = 600.0
                nearby_texts = []
                import re as _re_notes_filt
                for nearby_span in revB_text_spans:
                    nx0, ny0, nx1, ny1 = nearby_span.bbox_pdf
                    ncx = (nx0 + nx1) / 2
                    ncy = (ny0 + ny1) / 2
                    if (abs(ncx - matched_cx) <= _notes_h_radius
                            and abs(ncy - matched_cy) <= _notes_v_radius):
                        _st = nearby_span.text.strip()
                        # Only include spans that look like notes content:
                        # - multi-word spans (at least 3 words), OR
                        # - spans containing notes-relevant keywords
                        _words = _st.split()
                        _is_notes_kw = any(kw in _st.upper() for kw in (
                            "NOTES", "INTERPRET", "REFERENCE", "MATERIAL",
                            "CERTIFICATION", "SHIPMENT", "TOLERANCING",
                            "DIMENSIONING", "FINISH", "TITLE", "ASME", "BAG",
                        ))
                        if len(_words) >= 3 or _is_notes_kw:
                            nearby_texts.append(_st)

                if nearby_texts:
                    revB_notes_text = " ".join(nearby_texts)
                    revB_notes_fp = parse_requirement(revB_notes_text)
                    revB_tokens = set(revB_notes_fp.norm_text.split())
                    # Strip HTML entities (e.g. &#10; for newline) from anchor text
                    # before tokenising — these come from Form 3 HTML-encoded requirements
                    # and would cause spurious token mismatches vs plain PDF text.
                    # Also strip leading/trailing parens/punctuation from tokens
                    # (e.g. "(1." → "1.", "SHIPMENT.)" → "SHIPMENT.").
                    import re as _re_ent
                    _anchor_clean = _re_ent.sub(r'&#\d+;|&[a-z]+;', ' ', anchor_fp.norm_text)
                    def _clean_token(t: str) -> str:
                        return t.strip('()')
                    anchor_tokens = {_clean_token(t) for t in _anchor_clean.split() if _clean_token(t)}
                    # Apply same clean_token to revB tokens
                    revB_tokens = {_clean_token(t) for t in revB_tokens if _clean_token(t)}
                    if anchor_tokens and revB_tokens:
                        union_size = len(anchor_tokens | revB_tokens)
                        inter_size = len(anchor_tokens & revB_tokens)
                        notes_similarity = inter_size / union_size if union_size > 0 else 1.0
                        # If similarity is low, try OCR-normalized comparison.
                        # Common OCR errors in standard references: "ZO" → "20", "O" → "0"
                        # in year-like contexts (e.g. "Y14.5-ZO18" → "Y14.5-2018").
                        if notes_similarity < 0.65:
                            import re as _re
                            def _ocr_norm(text: str) -> str:
                                # Replace letter-O with digit-0 in year-like patterns
                                # e.g. "ZO18" → "2018", "Y14.5-ZO18" → "Y14.5-2018"
                                t = _re.sub(r'\bZO(\d{2})\b', r'20\1', text)
                                t = _re.sub(r'(?<=[A-Z\d\.])O(?=\d{2,})', '0', t)
                                t = _re.sub(r'(?<=\d)O(?=[A-Z\d])', '0', t)
                                return t
                            anchor_norm = _ocr_norm(_anchor_clean)
                            revB_norm = _ocr_norm(revB_notes_fp.norm_text)
                            anchor_tokens_norm = {_clean_token(t) for t in parse_requirement(anchor_norm).norm_text.split() if _clean_token(t)}
                            revB_tokens_norm = {_clean_token(t) for t in parse_requirement(revB_norm).norm_text.split() if _clean_token(t)}
                            if anchor_tokens_norm and revB_tokens_norm:
                                union_norm = len(anchor_tokens_norm | revB_tokens_norm)
                                inter_norm = len(anchor_tokens_norm & revB_tokens_norm)
                                notes_similarity_norm = inter_norm / union_norm if union_norm > 0 else 1.0
                                if notes_similarity_norm >= 0.65:
                                    # OCR normalization resolved the apparent difference
                                    return DeltaItem(
                                        char_no=anchor.char_no,
                                        status="unchanged",
                                        confidence=0.80,
                                        reasons=[
                                            "Notes block matched after OCR normalization",
                                            f"Raw similarity {notes_similarity:.2f} < 0.65, normalised similarity {notes_similarity_norm:.2f} ≥ 0.65",
                                        ],
                                        component_scores={
                                            "location": location_score,
                                            "text": notes_similarity_norm,
                                            "context": context_score,
                                        },
                                        match=match_or_none,
                                    )
                        if notes_similarity < 0.65:
                            return DeltaItem(
                                char_no=anchor.char_no,
                                status="changed",
                                confidence=max(0.70, 0.4 * location_score + 0.4 * notes_similarity + 0.2),
                                reasons=[
                                    "Notes block content changed",
                                    f"Token similarity {notes_similarity:.2f} below threshold 0.65",
                                ],
                                component_scores={
                                    "location": location_score,
                                    "text": notes_similarity,
                                    "context": context_score,
                                },
                                match=match_or_none,
                            )

            return DeltaItem(
                char_no=anchor.char_no,
                status="unchanged",
                confidence=0.85,
                reasons=["Notes block header matched", "High location agreement after global alignment"],
                component_scores={
                    "location": location_score,
                    "text": 1.0,
                    "context": context_score
                },
                match=match_or_none
            )
    
    # Extract numeric values (the key comparison for dimensions)
    anchor_numerics = set(val for val, _ in anchor_fp.numeric_tokens)
    matched_numerics = set(val for val, _ in matched_fp.numeric_tokens)
    
    # Identify primary dimension value (largest numeric, typically the main dimension)
    # Tolerances are usually small values like 0.1, 0.3 while dimensions are larger
    anchor_primary = max(anchor_numerics) if anchor_numerics else None
    matched_primary = max(matched_numerics) if matched_numerics else None
    
    # Check if primary dimension matches - this is the key indicator.
    # Use set membership rather than max-equality so that multi-value spans (e.g.,
    # "10 x 90°" for a countersink that encodes both depth=10 and angle=90°) are
    # correctly recognised as containing the anchor's primary value (10) even though
    # the span's overall maximum value (90) is different.
    primary_matches = (
        anchor_primary is not None and
        anchor_primary in matched_numerics
    )
    
    # Extract structural tokens (count patterns like "2X", "6X" and symbols like "Ø", "R")
    anchor_count = set(anchor_fp.count_tokens)
    matched_count = set(matched_fp.count_tokens)
    
    anchor_symbols = set(anchor_fp.symbol_tokens)
    matched_symbols = set(matched_fp.symbol_tokens)
    
    # Compute overlap metrics
    # For numerics: check if anchor's key values appear in matched span
    # Focus on primary value match rather than all tolerances
    if anchor_numerics:
        # If primary matches, give high score even if tolerances not in span
        if primary_matches:
            numeric_overlap = max(0.5, len(anchor_numerics & matched_numerics) / len(anchor_numerics))
        else:
            numeric_overlap = len(anchor_numerics & matched_numerics) / len(anchor_numerics)
    else:
        numeric_overlap = 1.0 if not matched_numerics else 0.0
    
    # For counts: distinguish between count change vs count missing from matched span
    # "2X" → "4X" is a real change, but "2X" → set() is just incomplete span matching
    count_match = anchor_count == matched_count or (not anchor_count and not matched_count)
    count_changed = bool(anchor_count and matched_count and anchor_count != matched_count)
    count_missing = bool(anchor_count and not matched_count)  # Count present in anchor but not in span
    count_added = bool(not anchor_count and matched_count)   # Count added in Rev B (e.g. none → "2X")
    
    # For symbols: check if anchor symbols appear in matched span
    symbol_match = anchor_symbols <= matched_symbols or not anchor_symbols

    # Accumulator for advisory confidence flags (populated by bleed detection etc.)
    result_flags: List[str] = []

    # Classification decision tree
    # Priority 0 (early guard): zero numeric overlap AND primary values diverge
    # significantly → the matched candidate is almost certainly a grid/border label
    # or an unrelated annotation.  Return "removed" before the later elif branches
    # can mis-classify it as "changed".
    # NOTE: this guard must run before the symbol/overlap checks below, because when
    # neither anchor nor span has engineering symbols (symbol_match = True) the
    # `numeric_overlap < 0.5 and symbol_match` branch would otherwise fire first and
    # reach a "changed" verdict via the limits-form path.
    if (
        numeric_overlap == 0.0
        and anchor_numerics
        and matched_numerics
        and anchor_primary is not None
        and matched_primary is not None
        and anchor_primary != 0
        and abs(anchor_primary - matched_primary) / abs(anchor_primary) > 0.15
    ):
        return DeltaItem(
            char_no=anchor.char_no,
            status="removed",
            confidence=min(0.85, location_search_coverage),
            reasons=reasons + [
                "No numeric overlap and primary values differ significantly",
                f"Anchor primary: {anchor_primary}, Matched primary: {matched_primary}",
                "Match candidate likely a grid label or unrelated annotation",
            ],
            component_scores={
                "location": location_score,
                "text": 0.0,
                "context": context_score,
            },
            match=None,
        )

    # Suppress count_added when the anchor is a pure-text modifier (no numerics) and the
    # matched span simply includes both the anchor text and a sibling's count/dimension.
    # E.g. anchor="THRU ALL" or "Depth (THRU ALL)" (no numerics) matched to "4X Ø8 THRU ALL"
    # — the span belongs to an adjacent characteristic; count_added is a false positive.
    if count_added and not anchor_numerics:
        import re as _re_kw
        def _kw_tokens(text: str):
            # Strip punctuation/parens, split, remove noise
            _stop = {'', 'A', 'AN', 'THE', 'OF', 'IN', 'AT', 'X'}
            return {w for w in _re_kw.sub(r'[()\/]', ' ', text.upper()).split() if w not in _stop}
        anchor_kw = _kw_tokens(anchor_fp.norm_text)
        matched_kw = _kw_tokens(matched_fp.norm_text)
        if anchor_kw and matched_kw:
            overlap_ratio = len(anchor_kw & matched_kw) / len(anchor_kw)
            if overlap_ratio >= 0.6:
                # Most anchor keywords are present in matched span → sibling-span capture;
                # the anchor content is present in Rev B, just embedded in a larger span.
                # Return unchanged immediately — no further numeric comparisons make sense
                # since all numerics in the matched span belong to the sibling characteristic.
                return DeltaItem(
                    char_no=anchor.char_no,
                    status="unchanged",
                    confidence=max(0.75, 0.5 * location_score + 0.25),
                    reasons=reasons + [
                        f"Count-added suppressed: {overlap_ratio:.0%} anchor keywords found in matched span "
                        f"(sibling-span capture, no numerics in anchor)",
                        "Anchor content present in Rev B span — unchanged",
                    ],
                    component_scores={
                        "location": location_score,
                        "text": overlap_ratio,
                        "context": context_score,
                    },
                    match=match_or_none,
                )

    # Priority 1: Primary dimension value match is the strongest indicator
    if count_changed or count_added:
        # Count explicitly changed or added (e.g., none → "2X", or "2X" → "4X") → changed
        status = "changed"
        confidence = 0.5 * location_score + 0.3 * numeric_overlap + 0.2
        if count_changed:
            reasons.append(f"Count changed: {anchor_count} → {matched_count}")
        else:
            reasons.append(f"Count added in Rev B: {matched_count} (was absent in Rev A)")
            # CLS-01: Adjacency-bleed suppression — only when count_added (not count_changed).
            # If the Rev B span is a "/" -merged multi-balloon annotation, the foreign count
            # prefix belongs to a neighbour characteristic, not to this anchor.  Suppress the
            # false count_added signal: demote status to "unchanged" and record the bleed flag.
            if _looks_like_adjacency_bleed(candidate.span.text, anchor.requirement_raw):
                status = "unchanged"
                confidence = max(0.55, confidence)
                reasons.append(
                    "Count-added suppressed: Rev B span appears to be a slash-merged "
                    "multi-balloon annotation (adjacency bleed detected)"
                )
                result_flags = [_BLEED_FLAG]
    elif primary_matches:
        # Primary dimension value matches - this is the key indicator of unchanged
        # Even if tolerances differ or aren't visible in span, the main dimension is the same.
        # However, if numeric overlap is low (≤65%), the differing numerics may be
        # tolerance values that genuinely changed (e.g., ±.005 → ±.01).
        tolerance_sized_diff = False
        if numeric_overlap <= 0.70 and anchor_primary is not None and anchor_primary != 0:
            differing_new = matched_numerics - anchor_numerics
            differing_old = anchor_numerics - matched_numerics
            # Only flag when the matched span has a NEW tolerance-sized value AND
            # the anchor has a DIFFERENT tolerance-sized value that's been replaced.
            # Both values must be within 5x of each other (same order of magnitude)
            # to avoid false positives from notation mismatches (e.g., 0.10 vs 0.010).
            tol_new = [v for v in differing_new if 0 < v <= 0.2 * anchor_primary]
            tol_old = [v for v in differing_old if 0 < v <= 0.2 * anchor_primary]
            if tol_new and tol_old:
                # Check that the differing values are within 5x of each other
                for nv in tol_new:
                    for ov in tol_old:
                        ratio = max(nv, ov) / min(nv, ov)
                        if ratio <= 5.0:
                            tolerance_sized_diff = True
                            reasons.append(
                                f"Tolerance-sized numeric differs: "
                                f"anchor {sorted(anchor_numerics)}, matched {sorted(matched_numerics)}"
                            )
                            break
                    if tolerance_sized_diff:
                        break

        if tolerance_sized_diff:
            status = "changed"
            confidence = max(0.65, 0.4 * location_score + 0.3 * numeric_overlap + 0.2)
            reasons.append(f"Primary dimension matches: {anchor_primary}")
            reasons.append("Tolerance value changed despite same primary dimension")
        else:
            status = "unchanged"
            confidence = 0.4 * location_score + 0.5 * numeric_overlap + 0.1
            reasons.append(f"Primary dimension matches: {anchor_primary}")
            if numeric_overlap >= 0.5:
                reasons.append(f"Numeric values match ({int(numeric_overlap*100)}% overlap)")
            if count_match and anchor_count:
                reasons.append(f"Count tokens match: {anchor_count}")
    elif numeric_overlap >= 0.5 and symbol_match:
        # Good numeric overlap with matching symbols.
        # However, the overlap may come purely from matching a shared tolerance value
        # (e.g., both have 0.01) while the primary dimension differs (4.93 vs 5.04).
        # In that case, the primary mismatch dominates and the status should be "changed".
        #
        # We distinguish true primary mismatches from limits-form notation by comparing
        # the primary difference against the anchor's own tolerance band:
        # - Extract the smallest non-zero numeric as the tolerance value.
        # - If primary difference > 1.5 * tolerance, it's a true change.
        # - If within 1.5 * tolerance, it's likely limits-form notation (same characteristic).
        primary_mismatch = False
        if (
            anchor_primary is not None
            and matched_primary is not None
            and not primary_matches
            and anchor_primary != 0
        ):
            primary_diff = abs(anchor_primary - matched_primary)
            # Estimate the tolerance as the smallest numeric in the anchor (> 0)
            small_numerics = [v for v in anchor_numerics if 0 < v < anchor_primary]
            if small_numerics:
                tolerance_estimate = min(small_numerics)
                # True change if primary difference exceeds 1.5x the tolerance band
                primary_mismatch = primary_diff > 1.5 * tolerance_estimate
            else:
                # No tolerance value in anchor — use a 3% relative threshold
                primary_mismatch = primary_diff / abs(anchor_primary) > 0.03

        if primary_mismatch:
            status = "changed"
            confidence = 0.4 * location_score + 0.3 * numeric_overlap + 0.2
            reasons.append(f"Primary dimension changed: {anchor_primary} → {matched_primary}")
            reasons.append(f"Numeric overlap from tolerance only ({int(numeric_overlap*100)}%)")
        else:
            # (count_missing is OK - just means span doesn't include count prefix)
            status = "unchanged"
            confidence = 0.4 * location_score + 0.4 * numeric_overlap + 0.2
            reasons.append(f"Numeric values match ({int(numeric_overlap*100)}% overlap)")
            if count_match and anchor_count:
                reasons.append(f"Count tokens match: {anchor_count}")
    elif numeric_overlap < 0.5 and symbol_match and not count_missing and not primary_matches:
        # Same type (symbol) but different numeric values.
        # First check if this is limits-form notation: the matched primary might be
        # the upper or lower tolerance limit of the anchor's nominal value.
        # e.g., anchor "0.150 ± 0.010" vs matched ".160" (upper limit = 0.150+0.010)
        limits_form_match = False
        if anchor_primary is not None and matched_primary is not None and anchor_primary != 0:
            primary_diff = abs(anchor_primary - matched_primary)
            small_numerics = [v for v in anchor_numerics if 0 < v < anchor_primary]
            if small_numerics:
                tolerance_estimate = min(small_numerics)
                relative_tolerance = tolerance_estimate / abs(anchor_primary)
                # Only apply limits-form check when the tolerance is tight (< 15% of nominal).
                # Tight tolerances (e.g., ±0.010 on 0.150) produce limits like 0.160/0.140
                # that fall within the 1.5x tolerance band.  Loose tolerances (e.g., ±0.3 on
                # 1.570) are too permissive — a genuine change of 0.110 would also pass.
                if relative_tolerance < 0.15:
                    limits_form_match = primary_diff <= 1.5 * tolerance_estimate
            else:
                limits_form_match = primary_diff / abs(anchor_primary) <= 0.03

        sibling_span_rejection = False  # True when stacked-limits check identifies a sibling span
        if limits_form_match and small_numerics:
            # Stacked-limits verification: when the matched span contains exactly two
            # numeric values (a stacked upper/lower limits pair), verify that the
            # anchor's own expected upper limit matches the span's upper value.
            # An anchor with "+0" positive tolerance (0.0 in its numerics) has
            # upper_limit = anchor_primary; otherwise upper_limit = primary + tolerance.
            # Mismatch means this span belongs to a DIFFERENT characteristic whose
            # tolerance band is centred differently (e.g., ±0.2 vs +0/-0.2).
            _tol_est = min(small_numerics)
            if len(matched_numerics) == 2:
                _matched_sorted = sorted(matched_numerics, reverse=True)
                _matched_upper = _matched_sorted[0]
                _has_zero_pos = 0.0 in anchor_numerics
                _expected_upper = anchor_primary if _has_zero_pos else anchor_primary + _tol_est
                if abs(_matched_upper - _expected_upper) > _tol_est * 0.15 + 1e-9:
                    # Span's upper limit doesn't match this anchor's expected upper limit.
                    # The span belongs to a sibling characteristic — reject limits-form.
                    limits_form_match = False
                    sibling_span_rejection = True
                    reasons.append(
                        f"Limits-form rejected: expected upper {_expected_upper:.4g}, "
                        f"span upper {_matched_upper:.4g} — span belongs to sibling char"
                    )

        if limits_form_match:
            # Limits-form notation: matched span is within the anchor's tolerance band
            status = "unchanged"
            confidence = 0.4 * location_score + 0.2 + 0.1  # No numeric overlap bonus
            reasons.append(f"Limits-form match: {anchor_primary} ± tolerance ≈ {matched_primary}")
        elif sibling_span_rejection:
            # The only candidate span belongs to a sibling characteristic (wrong-copy
            # ownership). This anchor has no valid Rev B match → classify as removed.
            status = "removed"
            confidence = min(0.8, location_search_coverage)
            reasons.append("Matched span claimed by sibling characteristic — anchor has no valid Rev B span")
        else:
            # Genuinely different numeric values → changed
            status = "changed"
            confidence = 0.4 * location_score + 0.3 * numeric_overlap + 0.2
            reasons.append(f"Numeric values changed (only {int(numeric_overlap*100)}% overlap)")
            reasons.append(f"Anchor: {sorted(anchor_numerics)}, Matched: {sorted(matched_numerics)}")
    elif location_score > 0.6:
        # Good location match but text doesn't align well → likely unchanged
        status = "unchanged"
        confidence = 0.6 * location_score + 0.2 * numeric_overlap
        reasons.append("Location strongly suggests same requirement")
        reasons.append(f"Numeric overlap: {int(numeric_overlap*100)}%")
    else:
        status = "uncertain"
        confidence = 0.5 * location_score + 0.3 * numeric_overlap
        reasons.append("Weak match confidence")
    
    # Add location-based reason
    if location_score > 0.7:
        reasons.append("High location agreement after global alignment")

    # --- Tolerance refinement ---
    if tolerance_comparison is not None:
        # CLS-03: Kind-transition pre-check — must run BEFORE tolerances_match boost.
        # When Rev A uses symmetric (±T) notation and Rev B switches to asymmetric
        # (+a/−b) notation, the tolerance KIND changed even if the absolute numeric
        # limits happen to agree (e.g. ±1 == +1/−1).  Promote to 'changed' and record
        # a kind-transition reason so reviewers can see the formatting change.
        # This guard fires first so the tolerances_match branch below cannot override it.
        # NOTE: the reason is appended regardless of current status to ensure traceability;
        # we only promote status when it is not already 'changed'.
        if _is_symmetric_to_asymmetric_kind_change(tolerance_comparison):
            _kind_reason = (
                f"Tolerance kind changed: {tolerance_comparison.revA_tolerance.kind} "
                f"→ {tolerance_comparison.revB_tolerance.kind} "
                f"(symmetric ± → asymmetric +/− kind transition)"
            )
            if status in ("unchanged", "uncertain"):
                status = "changed"
                confidence = max(confidence, 0.70)
            # Always append the kind-transition reason for traceability,
            # even when status was already 'changed' by an earlier branch.
            reasons.append(_kind_reason)
            reasons.extend(tolerance_comparison.reasons)
            # Do NOT fall through to tolerances_match — the kind change dominates.
        elif tolerance_comparison.tolerances_match:
            if status == "unchanged":
                confidence += 0.05
                reasons.extend(tolerance_comparison.reasons)
            elif status == "uncertain":
                status = "unchanged"
                confidence += 0.1
                reasons.append("Tolerance agreement resolves uncertainty")
                reasons.extend(tolerance_comparison.reasons)
        elif tolerance_comparison.tolerances_differ:
            if status == "unchanged":
                status = "changed"
                reasons.append("Tolerance changed despite same dimension")
                reasons.extend(tolerance_comparison.reasons)
            elif status == "uncertain":
                status = "changed"
                reasons.append("Tolerance difference resolves uncertainty")
                reasons.extend(tolerance_comparison.reasons)
    else:
        # CLS-03 fallback: when no tolerance_comparison is available, compare raw
        # requirement text for a symmetric→asymmetric kind transition.
        # Rev A is symmetric if it contains ± (or +/-); Rev B is asymmetric if it
        # matches _ASYMMETRIC_SHAPE_RE (signed pair separated by /).
        # Only promote 'unchanged' or 'uncertain' — never downgrade 'changed'.
        if status in ("unchanged", "uncertain") and candidate is not None:
            _anchor_raw = anchor.requirement_raw
            _matched_raw = candidate.span.text
            _revA_symmetric = bool(re.search(r"[±]|\+/-|\+/−", _anchor_raw))
            _revB_asymmetric = bool(_ASYMMETRIC_SHAPE_RE.search(_matched_raw))
            if _revA_symmetric and _revB_asymmetric:
                status = "changed"
                confidence = max(confidence, 0.65)
                reasons.append(
                    "Tolerance kind changed: symmetric ± (Rev A) → asymmetric +/− (Rev B) "
                    "detected via raw text fallback (no tolerance_comparison available)"
                )

    # Clip confidence
    confidence = max(0.0, min(1.0, confidence))
    
    return DeltaItem(
        char_no=anchor.char_no,
        status=status,
        confidence=confidence,
        reasons=reasons,
        component_scores={
            "location": location_score,
            "text": numeric_overlap,  # Report numeric overlap as text score
            "context": context_score
        },
        match=match_or_none,
        confidence_flags=result_flags,
    )


def detect_added_characteristics(
    revB_spans: List[TextSpan],
    matches: Dict[int, Match],
    next_char_no: int,
    page_width: float = 612.0,
    page_height: float = 792.0,
    revA_spans: Optional[List[TextSpan]] = None,
) -> List[DeltaItem]:
    """Detect new characteristics in Rev B that don't exist in Rev A.

    Looks for unmatched Rev B text spans that appear to be dimension requirements.
    Detects three categories:
    1. Standard annotated dimensions: symbol (Ø, R) or count prefix (2X, 4X)
    2. Angle-style callouts: degree marker with multiple numerics
    3. Leading-decimal single values: ".900", ".625" — engineering-drawing notation
       for plain length/depth dimensions without prefix symbols
    4. Stacked tolerance-limit pairs: two numeric-only spans at the same x position
       and adjacent y positions (e.g., ".635" above ".615") representing an upper/lower
       tolerance pair (equivalent to 0.625 ± 0.010)

    Args:
        revB_spans: All text spans from Rev B PDF
        matches: Dict of char_no to Match objects (matched spans)
        next_char_no: Starting char_no for added items (typically max_existing + 1)
        page_width: Width of the page in PDF points (used for exclusion zones)
        page_height: Height of the page in PDF points (used for exclusion zones)

    Returns:
        List of DeltaItems with status="added" for detected new characteristics
    """
    # Build set of matched span keys
    matched_span_keys: Set[Tuple] = set()
    matched_span_bboxes: List[Tuple[float, float, float, float]] = []
    for match in matches.values():
        span = match.candidate.span
        key = (span.block_id, span.line_id, span.span_id, span.bbox_pdf)
        matched_span_keys.add(key)
        matched_span_bboxes.append(span.bbox_pdf)
        source_spans = getattr(span, "source_spans", None)
        if source_spans:
            for source_span in source_spans:
                matched_span_keys.add((source_span.block_id, source_span.line_id, source_span.span_id, source_span.bbox_pdf))
                matched_span_bboxes.append(source_span.bbox_pdf)

    # Build a lookup of Rev A span content for cross-referencing.
    # Rev B spans whose text appears at the same position in Rev A are pre-existing
    # annotations (not new characteristics) and should not be flagged as "added".
    revA_span_signatures: Set[Tuple[str, int, int]] = set()
    if revA_spans is not None:
        for s in revA_spans:
            sx0, sy0, sx1, sy1 = s.bbox_pdf
            scx = round((sx0 + sx1) / 2)
            scy = round((sy0 + sy1) / 2)
            revA_span_signatures.add((s.text.strip(), scx, scy))

    def is_in_revA(span: TextSpan, position_tolerance: float = 15.0) -> bool:
        """Return True if this span exists in Rev A at the same position."""
        if not revA_span_signatures:
            return False
        x0, y0, x1, y1 = span.bbox_pdf
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        text = span.text.strip()
        tol = int(position_tolerance)
        for dx in range(-tol, tol + 1, 5):
            for dy in range(-tol, tol + 1, 5):
                if (text, round(cx) + dx, round(cy) + dy) in revA_span_signatures:
                    return True
        return False

    def is_in_exclusion_zone(bbox: tuple) -> bool:
        """Thin wrapper: delegates to the shared span_is_excluded_for_annotation_search
        helper so that title-block, revision-table, and boilerplate-text exclusion is
        consistent across all annotation-search callers."""
        from delta_preservation.io.pdf import TextSpan as _TS
        x0, y0, x1, y1 = bbox
        # Synthesise a minimal TextSpan with an empty text (geometry-only check);
        # boilerplate-text checks are handled separately in the span loop below via
        # span_is_excluded_for_annotation_search(span, ...) which also inspects text.
        _dummy = _TS(text="", bbox_pdf=bbox, font_size=0.0, block_id=0, line_id=0, span_id=0)
        return span_is_excluded_for_annotation_search(
            _dummy, page_width=page_width, page_height=page_height
        )

    def is_inside_matched_group(bbox: tuple, tolerance: float = 8.0) -> bool:
        x0, y0, x1, y1 = bbox
        for mx0, my0, mx1, my1 in matched_span_bboxes:
            if x0 >= mx0 - tolerance and y0 >= my0 - tolerance and x1 <= mx1 + tolerance and y1 <= my1 + tolerance:
                return True
        return False

    # GD&T feature control frame anchor symbols — defined early so the
    # unmatched-span pre-filter can exempt single-character GD&T symbols.
    # Includes all GD&T control and modifier symbols that appear as single-char
    # seed spans in feature control frames.
    GDT_ANCHOR_SYMBOLS = {
        '⌖', '⌒', '⟂', '⊙', '⌓', '⏥', '∥', '∠', '⌖∅',
        '◎',  # Circularity (◎ U+25CE)
        '⌰',  # Circular runout (⌰ U+2330)
        '↗',  # Angularity (↗ U+2197 used as angularity in some drawing systems)
        '⏤',  # Straightness (⏤ U+23E4)
        '⌴',  # Counterbore (⌴ U+2334, also used as depth symbol in some FCFs)
        '↧',  # Depth (↧ U+21A7)
        '⌖✢', # Position with projected tolerance zone
        '⌵',  # Countersink (⌵ U+2375) — identifies countersink callouts such as
               # "⌵ Ø.531 X 82.0°"; treating it as a GDT anchor lets Pass 0 seed
               # from the symbol span and group the angle/diameter companions.
    }
    GDT_COMPANION_CHARS = GDT_ANCHOR_SYMBOLS | {'Ø', '⌀', 'R', 'Ⓜ', 'Ⓛ', 'Ⓟ'}
    GDT_MODIFIER_TOKENS = {"M", "L", "P", "MMC", "LMC", "RFS"}
    ANNOTATION_ANCHOR_HINTS = (
        ('⌖✢', 'position'),
        ('⌖', 'position'),
        ('⌒', 'profile_of_a_line'),
        ('⌓', 'profile_of_a_surface'),
        ('⟂', 'perpendicularity'),
        ('⏥', 'flatness'),
        ('∥', 'parallelism'),
        ('∠', 'angularity'),
        ('⊙', 'concentricity'),
        ('◎', 'concentricity'),
        ('⚪', 'circularity'),
        ('○', 'circularity'),
        ('↗', 'circular_runout'),
        ('⌿', 'circular_runout'),
        ('⌰', 'total_runout'),
        ('⟃', 'total_runout'),
        ('↧', 'depth'),
        ('⌴', 'counterbore'),
        ('⏤', 'straightness'),
    )

    def _normalize_span_text(text: str) -> str:
        return " ".join(text.strip().upper().split())

    def _span_has_numeric_payload(text: str) -> bool:
        return bool(re.search(r"\d|[+\-±]", text))

    def _is_stacked_numeric_fragment(text: str) -> bool:
        normalized = _normalize_span_text(text)
        return bool(re.fullmatch(r"(?:[+\-]?(?:\d+(?:\.\d+)?|\.\d+))", normalized))

    def _looks_like_gdt_fragment(text: str) -> bool:
        normalized = _normalize_span_text(text)
        if not normalized:
            return False
        if any(ch in text for ch in GDT_COMPANION_CHARS):
            return True
        if normalized in GDT_MODIFIER_TOKENS:
            return True
        if re.fullmatch(r"[A-HJ-NP-Z]", normalized):
            return True
        return False

    def _annotation_anchor_hint_from_text(text: str) -> Optional[str]:
        for symbol, control_type in ANNOTATION_ANCHOR_HINTS:
            if symbol in text:
                return control_type
        return None

    def _horizontal_gap(a: TextSpan, b: TextSpan) -> float:
        ax0, _ay0, ax1, _ay1 = a.bbox_pdf
        bx0, _by0, bx1, _by1 = b.bbox_pdf
        if bx0 > ax1:
            return bx0 - ax1
        if ax0 > bx1:
            return ax0 - bx1
        return 0.0

    def _spans_are_annotation_companions(base: TextSpan, other: TextSpan) -> bool:
        bx0, by0, bx1, by1 = base.bbox_pdf
        ox0, oy0, ox1, oy1 = other.bbox_pdf
        bcx = (bx0 + bx1) / 2
        bcy = (by0 + by1) / 2
        ocx = (ox0 + ox1) / 2
        ocy = (oy0 + oy1) / 2
        base_h = max(1.0, by1 - by0)
        other_h = max(1.0, oy1 - oy0)

        same_row = abs(ocy - bcy) <= max(10.0, 0.75 * max(base_h, other_h))
        horiz_gap = _horizontal_gap(base, other)
        center_dx = abs(ocx - bcx)
        center_dy = abs(ocy - bcy)
        pair_is_gdt = _looks_like_gdt_fragment(base.text) or _looks_like_gdt_fragment(other.text)
        either_has_numeric_payload = _span_has_numeric_payload(base.text) or _span_has_numeric_payload(other.text)
        base_anchor_hint = _annotation_anchor_hint_from_text(base.text)
        other_anchor_hint = _annotation_anchor_hint_from_text(other.text)
        incompatible_anchor_pair = (
            base_anchor_hint is not None
            and other_anchor_hint is not None
            and base_anchor_hint != other_anchor_hint
        )

        close_horiz_limit = 28.0 if pair_is_gdt else 20.0
        center_dx_limit = 32.0 if pair_is_gdt else 28.0
        stacked_dx_limit = 18.0 if pair_is_gdt else 14.0
        stacked_dy_limit = 18.0 if pair_is_gdt else 14.0

        if same_row and (horiz_gap <= close_horiz_limit or center_dx <= center_dx_limit):
            if incompatible_anchor_pair:
                return False
            if pair_is_gdt or either_has_numeric_payload:
                return True

        if (
            abs(ocx - bcx) <= stacked_dx_limit
            and 0.0 < center_dy <= stacked_dy_limit
            and (pair_is_gdt or either_has_numeric_payload)
        ):
            if incompatible_anchor_pair:
                return False
            if base_anchor_hint or other_anchor_hint:
                return base_anchor_hint is not None and base_anchor_hint == other_anchor_hint
            return _is_stacked_numeric_fragment(base.text) and _is_stacked_numeric_fragment(other.text)

        return False

    # Collect unmatched candidate spans (filtered for drawing content area)
    unmatched_spans: List[TextSpan] = []
    for span in revB_spans:
        key = (span.block_id, span.line_id, span.span_id, span.bbox_pdf)
        if key in matched_span_keys:
            continue
        if is_inside_matched_group(span.bbox_pdf):
            continue
        # Use the shared exclusion contract (covers geometry + boilerplate text).
        if span_is_excluded_for_annotation_search(
            span, page_width=page_width, page_height=page_height
        ):
            continue
        text = span.text.strip()
        if len(text) < 2 and text not in GDT_ANCHOR_SYMBOLS:
            continue
        unmatched_spans.append(span)

    # Set of span keys consumed by stacked-pair detection (to avoid double-counting)
    stacked_pair_keys: Set[Tuple] = set()

    # Set of span keys consumed by GD&T group detection
    gdt_consumed_keys: Set[Tuple] = set()

    added_items: List[DeltaItem] = []
    current_char_no = next_char_no

    # Build set of matched span centre positions for proximity checks below.
    # Used to suppress unmatched spans that are companion tolerance limits of
    # already-matched annotations (e.g., the ".140" lower-limit span sitting
    # 12 pts below the matched ".160" upper-limit span).
    matched_centers: List[Tuple[float, float]] = []
    for match in matches.values():
        s = match.candidate.span
        sx0, sy0, sx1, sy1 = s.bbox_pdf
        matched_centers.append(((sx0 + sx1) / 2, (sy0 + sy1) / 2))

    # Build set of numeric float values from matched spans.
    # Used in stacked-pair detection to suppress pairs whose values are already
    # covered by an existing matched characteristic.  Example: when char 1 is
    # matched to a "35.2" span (limits-form upper limit of Ø35 ±0.2), a second
    # unmatched "35.2" / "34.8" stacked pair in another view represents the same
    # characteristic — not a new one — and should be suppressed.
    matched_span_float_values: set = set()
    matched_numeric_pairs: set[tuple[float, float]] = set()
    for match in matches.values():
        span = match.candidate.span
        texts = [span.text.strip()]
        for source_span in getattr(span, "source_spans", []) or []:
            texts.append(source_span.text.strip())
        numeric_values = []
        for text in texts:
            try:
                numeric_values.append(float(text))
                matched_span_float_values.add(float(text))
            except ValueError:
                continue
        uniq = sorted(set(numeric_values))
        if len(uniq) >= 2:
            matched_numeric_pairs.add((uniq[-2], uniq[-1]))

    def is_near_matched_span(bbox: tuple, threshold: float = 25.0) -> bool:
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for mx, my in matched_centers:
            if math.sqrt((cx - mx) ** 2 + (cy - my) ** 2) <= threshold:
                return True
        return False

    # --- Pass 0: GD&T feature control frame detection ---
    # Feature control frames: a GD&T symbol span (⟂, ⌖, ⏥, etc.) followed by
    # companion spans (tolerance value, datum references) on the same row.
    # These groups must be detected together; individual spans don't pass the
    # standard filters (symbol spans lack numerics; companion spans lack symbols).

    for span in unmatched_spans:
        key = (span.block_id, span.line_id, span.span_id, span.bbox_pdf)
        if key in gdt_consumed_keys:
            continue

        text = span.text.strip()
        has_gdt = any(sym in text for sym in GDT_ANCHOR_SYMBOLS)
        if not has_gdt:
            continue

        if is_in_revA(span):
            continue

        # If the GD&T anchor span is on the exact same row as an already-matched span
        # (within 12 pt — tight enough for same-annotation-row companions), it is a
        # second-row compartment of an existing matched FCF, not a new characteristic.
        # NOTE: the broader explained-by-match suppressor at the end of this function
        # handles cases where a GD&T frame is nearby but NOT semantically explained —
        # those must survive even if they are within a larger proximity radius.
        if is_near_matched_span(span.bbox_pdf, threshold=12.0):
            gdt_consumed_keys.add(key)
            continue

        group_spans = [span]
        group_keys = [key]
        seen_group_keys = {key}
        queue: List[TextSpan] = [span]
        # Walk local companion geometry to a fixed point so the FCF keeps its
        # own modifier/datum fragments without absorbing distant same-row content.
        while queue:
            current = queue.pop(0)
            for other in revB_spans:
                other_key = (other.block_id, other.line_id, other.span_id, other.bbox_pdf)
                if other_key in seen_group_keys or other_key in gdt_consumed_keys:
                    continue
                if not _spans_are_annotation_companions(current, other):
                    continue
                group_spans.append(other)
                seen_group_keys.add(other_key)
                queue.append(other)
                # Only mark unmatched spans as consumed; matched spans remain available
                if other_key not in matched_span_keys:
                    group_keys.append(other_key)

        # Sort group spans left-to-right and concatenate text
        group_spans.sort(key=lambda s: s.bbox_pdf[0])
        group_text = ' '.join(s.text.strip() for s in group_spans)

        # Mark all unmatched group spans consumed so Pass 1 and Pass 2 skip them
        for k in group_keys:
            gdt_consumed_keys.add(k)

        # If any companion span (non-anchor) that contains numeric content is
        # already matched to an existing Rev A characteristic, this GD&T frame
        # belongs to an existing characteristic — not a new one.  Skip it.
        # Examples prevented:
        #   • "⏥ 0.2" where "0.2" was already matched by char 9 (double-detection)
        #   • "−0.1 ⌓ 0.5 A" where "0.5" was matched by char 11 via global fallback
        anchor_span_key = key  # Key of the GD&T anchor symbol span
        has_matched_numeric_companion = any(
            (s.block_id, s.line_id, s.span_id, s.bbox_pdf) in matched_span_keys
            and (s.block_id, s.line_id, s.span_id, s.bbox_pdf) != anchor_span_key
            and any(c.isdigit() for c in s.text)
            for s in group_spans
        )
        if has_matched_numeric_companion:
            continue

        # If any span in the assembled group shares a span with an already-matched
        # characteristic (matched numeric companion) — already handled above — or if
        # a non-anchor companion in the group sits within 12 pt of a matched span center,
        # this FCF belongs to an existing characteristic.
        # NOTE: The post-pass content-aware suppressor handles broader proximity cases.
        group_near_match = any(
            is_near_matched_span(s.bbox_pdf, threshold=12.0)
            for s in group_spans
            if (s.block_id, s.line_id, s.span_id, s.bbox_pdf) != key  # exclude anchor itself
        )
        if group_near_match:
            continue

        reasons_gdt = [
            f"New GD&T feature control frame in Rev B: \"{group_text}\"",
            "GD&T control symbol detected",
        ]
        # Compute union bbox of the entire group for CLS-02 distance gating
        _gdt_union_bbox = group_spans[0].bbox_pdf
        for _gs in group_spans[1:]:
            _gdt_union_bbox = (
                min(_gdt_union_bbox[0], _gs.bbox_pdf[0]),
                min(_gdt_union_bbox[1], _gs.bbox_pdf[1]),
                max(_gdt_union_bbox[2], _gs.bbox_pdf[2]),
                max(_gdt_union_bbox[3], _gs.bbox_pdf[3]),
            )
        _gdt_item = DeltaItem(
            char_no=current_char_no,
            status="added",
            confidence=0.75,
            reasons=reasons_gdt,
            component_scores={"location": 0.0, "text": 1.0, "context": 0.0},
            match=None,
            added_span=group_spans[0],
            added_requirement_text=group_text,
            added_bbox=_gdt_union_bbox,
            added_page=0,
        )
        added_items.append(_gdt_item)
        current_char_no += 1

    # --- Pass 1: detect stacked tolerance-limit pairs ---
    # Stacked pairs: two numeric-only spans at the same horizontal position and
    # adjacent vertical positions that together express a tolerance band.
    # Example: ".635" (upper) stacked above ".615" (lower) → 0.625 ± 0.010
    # Criteria:
    #   - Both spans are numeric-only (no symbol, no count)
    #   - Not a trailing-dot list number (e.g., "1.", "2.", "3.")
    #   - Horizontal centres within 10 pts of each other (same column)
    #   - Vertical separation ≤ 30 pts (adjacent lines)
    #   - Upper value > lower value (ordered pair)
    #   - Difference ≤ 20% of mean (tight tolerance, not two unrelated dims)
    #   - Neither span is close to an already-matched span (would be a companion
    #     tolerance limit of an existing characteristic, not a new one)
    leading_decimal_re = re.compile(r'^\.\d+$')
    trailing_dot_re = re.compile(r'^\d+\.$')  # List item numbers: "1.", "2.", "3."
    numeric_only_spans = [
        s for s in unmatched_spans
        if (leading_decimal_re.match(s.text.strip()) or
            re.match(r'^\d+\.?\d*$', s.text.strip()))
        and not trailing_dot_re.match(s.text.strip())
        and not is_near_matched_span(s.bbox_pdf)
    ]

    for i, span_a in enumerate(numeric_only_spans):
        key_a = (span_a.block_id, span_a.line_id, span_a.span_id, span_a.bbox_pdf)
        if key_a in stacked_pair_keys:
            continue
        ax0, ay0, ax1, ay1 = span_a.bbox_pdf
        acx = (ax0 + ax1) / 2
        acy = (ay0 + ay1) / 2
        val_a_str = span_a.text.strip()
        try:
            val_a = float(val_a_str)
        except ValueError:
            continue

        for span_b in numeric_only_spans[i + 1:]:
            key_b = (span_b.block_id, span_b.line_id, span_b.span_id, span_b.bbox_pdf)
            if key_b in stacked_pair_keys:
                continue
            bx0, by0, bx1, by1 = span_b.bbox_pdf
            bcx = (bx0 + bx1) / 2
            bcy = (by0 + by1) / 2
            val_b_str = span_b.text.strip()
            try:
                val_b = float(val_b_str)
            except ValueError:
                continue

            # Horizontal alignment check
            if abs(acx - bcx) > 10.0:
                continue
            # Vertical proximity check
            vertical_gap = abs(acy - bcy)
            if vertical_gap > 30.0:
                continue
            # Must be an ordered pair (upper > lower)
            upper_val = max(val_a, val_b)
            lower_val = min(val_a, val_b)
            if upper_val <= lower_val:
                continue
            # Tolerance check: difference must be tight (≤ 20% of mean)
            # A tighter threshold rejects numeric pairs that are two unrelated
            # dimension values happening to be adjacent (e.g., "3." and "4." note
            # numbers have a ratio of ~29%, which tighter threshold excludes).
            mean_val = (upper_val + lower_val) / 2.0
            if mean_val == 0:
                continue
            if (upper_val - lower_val) / mean_val > 0.20:
                continue

            # Valid stacked pair found
            nominal = mean_val
            half_tol = (upper_val - lower_val) / 2.0
            pair_text = f"{upper_val} / {lower_val}"
            ordered_pair = (lower_val, upper_val)
            if ordered_pair in matched_numeric_pairs:
                stacked_pair_keys.add(key_a)
                stacked_pair_keys.add(key_b)
                break
            # Use the upper span as the representative span
            upper_span = span_a if val_a >= val_b else span_b
            lower_span = span_b if val_a >= val_b else span_a
            reasons = [
                f"New stacked-limits dimension in Rev B: {pair_text}",
                f"Interpreted as {nominal:.4g} \u00b1 {half_tol:.4g}",
                "Leading-decimal stacked tolerance limits detected",
            ]
            confidence = 0.65
            # Union bbox of the pair for CLS-02 distance gating
            _pair_union_bbox = (
                min(upper_span.bbox_pdf[0], lower_span.bbox_pdf[0]),
                min(upper_span.bbox_pdf[1], lower_span.bbox_pdf[1]),
                max(upper_span.bbox_pdf[2], lower_span.bbox_pdf[2]),
                max(upper_span.bbox_pdf[3], lower_span.bbox_pdf[3]),
            )

            added_item = DeltaItem(
                char_no=current_char_no,
                status="added",
                confidence=confidence,
                reasons=reasons,
                component_scores={
                    "location": 0.0,
                    "text": 1.0,
                    "context": 0.0
                },
                match=None,
                added_span=upper_span,
                added_requirement_text=pair_text,
                added_bbox=_pair_union_bbox,
                added_page=0,
            )
            added_items.append(added_item)
            current_char_no += 1
            stacked_pair_keys.add(key_a)
            stacked_pair_keys.add(key_b)
            break  # Each span_a pairs with at most one span_b

    # ---------------------------------------------------------------------------
    # Helper: expand a standard unmatched seed span into its companion annotation
    # spans (same row, horizontally close), returning grouped text and union bbox.
    # This mirrors the grouping logic used in Pass 0 for GD&T anchor frames.
    # ---------------------------------------------------------------------------
    def _expand_standard_added_span(
        seed: TextSpan,
        consumed_keys: Set[Tuple],
        is_surface_finish_seed: bool = False,
    ) -> Tuple[str, Tuple[float, float, float, float], List[TextSpan]]:
        """Group companion spans for a standard added seed span.

        Walk local geometric companions to a fixed point so nearby fragments stay
        together without sweeping in distant same-row annotations.

        For surface-finish seeds (Ra indicator), also sweep same-block adjacent-line
        spans that are within a 25 pt vertical window.  This handles multi-line
        surface-finish callouts such as:

            FINISH TURN   ← block 47, line 0
            1000 Ra       ← block 47, line 1  (seed span)

        where the prefix row has no numeric payload and would not pass the normal
        companion geometry check.

        Returns:
            (grouped_text, union_bbox, all_companion_spans_including_seed)
        """
        seed_key = (seed.block_id, seed.line_id, seed.span_id, seed.bbox_pdf)

        group_spans = [seed]
        seen_group_keys: Set[Tuple] = {seed_key}
        queue: List[TextSpan] = [seed]

        while queue:
            current = queue.pop(0)
            for other in revB_spans:
                other_key = (other.block_id, other.line_id, other.span_id, other.bbox_pdf)
                if other_key in seen_group_keys or other_key in consumed_keys:
                    continue
                # Only group unmatched companions; matched spans may be included for
                # text purposes but should not be re-consumed
                if other_key in matched_span_keys:
                    continue
                if span_is_excluded_for_annotation_search(
                    other, page_width=page_width, page_height=page_height
                ):
                    continue

                # Surface-finish same-block sweep: include same-block adjacent-line
                # spans within 25 pt vertically that are not excluded or matched.
                # This captures prefix labels such as "FINISH TURN" that share the
                # annotation block but carry no numeric payload of their own.
                if is_surface_finish_seed and other.block_id == seed.block_id:
                    oy0, oy1 = other.bbox_pdf[1], other.bbox_pdf[3]
                    sy0, sy1 = seed.bbox_pdf[1], seed.bbox_pdf[3]
                    vert_gap = max(0.0, max(oy0, sy0) - min(oy1, sy1))
                    if vert_gap <= 25.0:
                        group_spans.append(other)
                        seen_group_keys.add(other_key)
                        queue.append(other)
                        continue

                if not _spans_are_annotation_companions(current, other):
                    continue
                group_spans.append(other)
                seen_group_keys.add(other_key)
                queue.append(other)

        # Sort left-to-right then top-to-bottom so prefix rows come before value rows
        group_spans.sort(key=lambda s: (s.bbox_pdf[1], s.bbox_pdf[0]))

        # Build grouped text (representative seed's text + companion texts)
        grouped_text = " ".join(s.text.strip() for s in group_spans if s.text.strip())

        # Build union bbox
        union = group_spans[0].bbox_pdf
        for gs in group_spans[1:]:
            union = (
                min(union[0], gs.bbox_pdf[0]),
                min(union[1], gs.bbox_pdf[1]),
                max(union[2], gs.bbox_pdf[2]),
                max(union[3], gs.bbox_pdf[3]),
            )

        return grouped_text, union, group_spans

    # Track keys consumed by the standard pass (to avoid double-counting when
    # two spans belong to the same grouped callout)
    standard_consumed_keys: Set[Tuple] = set()

    # Deduplicate added candidates by grouped evidence identity (grouped_text + bbox).
    # Prevents the same callout from generating multiple added items when the seed
    # span for a group is encountered multiple times (e.g., companion spans that
    # are individually dimension-like).
    seen_grouped_evidence: Set[Tuple[str, Tuple[int, int, int, int]]] = set()

    # --- Pass 2: standard span-by-span detection ---
    for span in unmatched_spans:
        key = (span.block_id, span.line_id, span.span_id, span.bbox_pdf)
        # Skip spans already consumed by GD&T group, stacked-pair detection, or
        # the standard pass grouping from a previous iteration
        if key in gdt_consumed_keys or key in stacked_pair_keys or key in standard_consumed_keys:
            continue

        # Skip spans that appear verbatim in Rev A at the same position —
        # these are pre-existing annotations not tracked by Form 3, not new features.
        if is_in_revA(span):
            continue

        text = span.text.strip()

        # Parse the span to check if it looks like a dimension requirement
        fp = parse_requirement(text)

        # Angle callouts (e.g., "2 X 5 X 45°") often lack Ø/R symbols and explicit count
        # tokens, so treat a degree marker plus multiple numeric values as dimension-like.
        has_angle_token = "°" in text or "DEG" in fp.norm_text or "ANGLE" in fp.norm_text
        has_angle_dimension = has_angle_token and len(fp.numeric_tokens) >= 2

        # Leading-decimal format: ".900", ".625" are engineering-drawing shorthand for
        # dimension values without a leading zero.  These may be plain length/depth
        # annotations without a prefix symbol (Ø, R) or count (2X).
        # Only match the pure leading-decimal form (exactly ".<digits>") to avoid
        # false-positives from mid-sentence decimal numbers.
        is_leading_decimal_single = bool(
            leading_decimal_re.match(text)
            and len(fp.numeric_tokens) == 1
        )

        # Plain decimal format: "1.250", "3.750" — dimension values with a leading
        # integer digit and explicit decimal point.  Treated similarly to leading-decimal
        # annotations; requires exactly one numeric token to avoid matching multi-value
        # spans that already fail other filters.
        is_plain_decimal_single = bool(
            re.match(r'^\d+\.\d+$', text)
            and len(fp.numeric_tokens) == 1
        )

        # Plain integer dimension: standalone integers ≥ 100 such as ordinate
        # dimensions (155, 225, 450) or general lengths (800) added in a new view.
        # Zone labels and note numbers are typically < 100; title-block entries are
        # caught by the exclusion zone check above.
        is_plain_integer_dimension = bool(
            re.match(r'^\d+$', text)
            and len(fp.numeric_tokens) == 1
            and fp.numeric_tokens[0][0] >= 100
        )

        # Surface finish callouts: "Ra <value>" or "<value> Ra" (e.g., "1000 Ra",
        # "FINISH TURN 1000 Ra").  The "Ra" indicator alone with a numeric value is
        # sufficient to identify a surface finish annotation.
        has_surface_finish = bool(
            re.search(r'\bRa\b', text, re.IGNORECASE) and fp.numeric_tokens
        )

        # Filter for spans that look like dimension requirements:
        # - Must have numeric tokens (dimension values)
        # - Must have symbols (Ø, R) OR count patterns (2X, 4X) OR angle-style callouts
        #   OR be a pure leading-decimal single value (engineering notation for length)
        #   OR be a surface finish annotation (Ra indicator)
        if not fp.numeric_tokens:
            continue

        if not (fp.symbol_tokens or fp.count_tokens or has_angle_dimension or is_leading_decimal_single or is_plain_decimal_single or has_surface_finish or is_plain_integer_dimension):
            continue

        # Skip spans that look like revision notes (contain "ADDED", "NEW", etc.)
        if any(word in fp.norm_text for word in ["ADDED", "NEW", "REVISED", "DELETED"]):
            continue

        # Skip non-leading-decimal, non-surface-finish single numeric values
        if len(fp.numeric_tokens) == 1 and not fp.symbol_tokens and not fp.count_tokens and not is_leading_decimal_single and not is_plain_decimal_single and not has_surface_finish and not is_plain_integer_dimension:
            continue

        # Suppress spans that sit close to an already-matched span — they are likely
        # companion parts of an existing annotation (e.g., "Ø.531 X 82.0°" adjacent to
        # a matched "Ø.266 THRU" countersink callout, or tolerance limit companions).
        if is_near_matched_span(span.bbox_pdf):
            continue

        # Suppress plain-integer candidates that sit in the shared boilerplate
        # zone, or that are direct same-row companions of a boilerplate phrase.
        # The previous 120 pt omni-directional proximity sweep was too broad —
        # it suppressed real added callouts (e.g. Part 5 truth_index 16 "800")
        # merely because "UNLESS OTHERWISE SPECIFIED" happened to sit within
        # 120 pt of the span centre even though the tolerance block itself is
        # confined to the bottom-centre notes zone.  Symbol-bearing and
        # count-bearing spans remain exempt (handled by the outer guard).
        if is_plain_integer_dimension and not fp.symbol_tokens and not fp.count_tokens:
            if span_is_excluded_for_annotation_search(
                span,
                page_width=page_width,
                page_height=page_height,
            ):
                continue
            sx0, sy0, sx1, sy1 = span.bbox_pdf
            scx, scy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
            _is_same_row_boilerplate_companion = False
            for _other in revB_spans:
                if not is_boilerplate_candidate_text(_other.text):
                    continue
                _ox0, _oy0, _ox1, _oy1 = _other.bbox_pdf
                _ocx, _ocy = (_ox0 + _ox1) / 2, (_oy0 + _oy1) / 2
                if abs(_ocy - scy) <= 10.0 and math.hypot(_ocx - scx, _ocy - scy) <= 40.0:
                    _is_same_row_boilerplate_companion = True
                    break
            if _is_same_row_boilerplate_companion:
                continue

        # This looks like a new characteristic.
        # Expand the seed span into its companion annotation spans (same row)
        # to build the authoritative grouped text and union bbox.
        all_consumed_for_expansion = gdt_consumed_keys | stacked_pair_keys | standard_consumed_keys
        grouped_text, grouped_union_bbox, group_spans = _expand_standard_added_span(
            span,
            consumed_keys=all_consumed_for_expansion,
            is_surface_finish_seed=has_surface_finish,
        )

        # Deduplicate by grouped evidence identity: if the same grouped callout
        # (same text + same bounding region) was already emitted from a companion
        # span in an earlier iteration, skip this seed.
        _rounded_group_bbox = tuple(int(round(v)) for v in grouped_union_bbox)
        _evidence_key = (grouped_text.strip(), _rounded_group_bbox)
        if _evidence_key in seen_grouped_evidence:
            # Mark this span consumed so it doesn't generate a duplicate
            standard_consumed_keys.add(key)
            continue
        seen_grouped_evidence.add(_evidence_key)

        # Mark all companion spans consumed so subsequent iterations skip them
        for _gs in group_spans:
            _gk = (_gs.block_id, _gs.line_id, _gs.span_id, _gs.bbox_pdf)
            standard_consumed_keys.add(_gk)

        reasons = [
            f"New requirement detected in Rev B: \"{grouped_text}\"",
            f"Symbols: {fp.symbol_tokens}, Counts: {fp.count_tokens}",
            f"Grouped from {len(group_spans)} span(s) on the same annotation row",
        ]
        if has_angle_dimension and not fp.symbol_tokens and not fp.count_tokens:
            reasons.append("Angle-style callout detected")
        if is_leading_decimal_single:
            reasons.append("Leading-decimal dimension annotation detected")
        if is_plain_decimal_single:
            reasons.append("Plain decimal dimension annotation detected")
        if has_surface_finish:
            reasons.append("Surface finish annotation detected (Ra indicator)")
        if is_plain_integer_dimension:
            reasons.append("Plain integer dimension annotation detected (≥100)")

        # Calculate confidence based on how "dimension-like" the span is
        confidence = 0.6
        if fp.symbol_tokens:
            confidence += 0.15
        if fp.count_tokens:
            confidence += 0.15
        if fp.pattern_class in ["hole", "dimension", "fillet"]:
            confidence += 0.1
        if has_surface_finish:
            confidence += 0.10
        if is_leading_decimal_single or is_plain_decimal_single:
            confidence = 0.55  # Slightly lower confidence for bare decimal values
        if is_plain_integer_dimension:
            confidence = 0.55  # Slightly lower confidence for bare integers
        confidence = min(confidence, 0.95)

        added_item = DeltaItem(
            char_no=current_char_no,
            status="added",
            confidence=confidence,
            reasons=reasons,
            component_scores={
                "location": 0.0,
                "text": 1.0,
                "context": 0.0
            },
            match=None,
            added_span=span,                         # representative seed span (provenance)
            added_requirement_text=grouped_text,     # canonical grouped annotation text
            added_bbox=grouped_union_bbox,           # canonical union bbox of all group spans
            added_page=0,
        )
        added_items.append(added_item)
        current_char_no += 1

    # ---------------------------------------------------------------------------
    # Explained-by-match suppressor (ADD-02)
    # Remove added candidates whose grouped text content is already fully owned by
    # an existing matched characteristic's annotation.
    #
    # Suppression requires BOTH conditions to be true:
    #   1. Content ownership: the candidate's grouped text is a normalized-text
    #      subset (after stripping GD&T punctuation and whitespace) of a matched
    #      annotation's grouped text.
    #   2. Bbox ownership: the candidate's grouped bbox sits inside or strongly
    #      overlaps (IoU > 0.3 or containment ratio > 0.5) the matched annotation's
    #      grouped bbox.
    #
    # Proximity alone is NOT sufficient — a geometrically close but semantically
    # unrelated annotation must survive.
    # ---------------------------------------------------------------------------

    def _normalize_for_suppression(text: str) -> str:
        """Normalize annotation text for ownership checking.

        Strips GD&T punctuation, whitespace, and case-irrelevant separators so
        that '∅.045 A' and '◎ ∅.045 A' can be compared as subsets.
        """
        # Remove spaces and common separators; lowercase; keep digits, letters,
        # GD&T symbols, and dot/decimal.
        import unicodedata
        t = text.strip()
        # Collapse multiple whitespace to single space
        t = re.sub(r'\s+', ' ', t)
        return t.upper()

    def _is_content_subset(candidate_text: str, owner_text: str) -> bool:
        """Return True when candidate_text content is already present in owner_text.

        Uses two checks:
        1. Normalized substring: candidate_text (normalized) is a substring of
           owner_text (normalized).
        2. Token subset: every normalized token in candidate_text is also present
           in owner_text (handles reordering).
        """
        if not candidate_text or not owner_text:
            return False
        cand_norm = _normalize_for_suppression(candidate_text)
        owner_norm = _normalize_for_suppression(owner_text)
        # Direct substring check
        if cand_norm in owner_norm:
            return True
        # Token subset check: every whitespace-delimited token in candidate
        # must appear in owner
        cand_tokens = set(cand_norm.split())
        owner_tokens = set(owner_norm.split())
        # Need at least 2 tokens to avoid single-char false positives (e.g. 'A' in any text)
        if len(cand_tokens) >= 2 and cand_tokens <= owner_tokens:
            return True
        return False

    def _bbox_overlap_ratio(bbox_a: tuple, bbox_b: tuple) -> float:
        """Return the containment ratio of bbox_a within bbox_b.

        Returns the fraction of bbox_a that is covered by bbox_b (intersection
        area / bbox_a area).  Values near 1.0 mean bbox_a is inside bbox_b.
        """
        ax0, ay0, ax1, ay1 = bbox_a
        bx0, by0, bx1, by1 = bbox_b
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        intersection = (ix1 - ix0) * (iy1 - iy0)
        area_a = (ax1 - ax0) * (ay1 - ay0)
        if area_a <= 0:
            return 0.0
        return intersection / area_a

    # Build matched annotation signatures: grouped text + union bbox for every match
    # Search all revB_spans for companion spans to each matched span (same row,
    # horizontally close) so that the suppressor has the full callout context.
    matched_annotation_signatures: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for match in matches.values():
        mspan = match.candidate.span
        msource_spans = getattr(mspan, "source_spans", None) or [mspan]
        # Compute the union bbox of source spans
        m_union_bbox = msource_spans[0].bbox_pdf
        for _ms in msource_spans[1:]:
            m_union_bbox = (
                min(m_union_bbox[0], _ms.bbox_pdf[0]),
                min(m_union_bbox[1], _ms.bbox_pdf[1]),
                max(m_union_bbox[2], _ms.bbox_pdf[2]),
                max(m_union_bbox[3], _ms.bbox_pdf[3]),
            )
        # Collect companion spans that are annotation companions to the matched spans.
        # Only include unmatched spans that satisfy _spans_are_annotation_companions
        # when walked from the matched source spans to a fixed point.  The old ±200 pt
        # same-row horizontal window is intentionally removed: it was too broad and
        # absorbed unrelated same-row annotations (e.g. the Part 9 flatness frame) into
        # a matched owner's synthetic signature, causing false suppression.
        full_group_spans = list(msource_spans)
        newly_added = True
        while newly_added:
            newly_added = False
            for other in revB_spans:
                other_key = (other.block_id, other.line_id, other.span_id, other.bbox_pdf)
                if other_key in matched_span_keys:
                    continue
                # Skip spans already in the group
                if any(
                    (s.block_id, s.line_id, s.span_id, s.bbox_pdf) == other_key
                    for s in full_group_spans
                ):
                    continue
                # Accept only if this span is a companion to at least one span already in
                # the group (fixed-point walk so transitively adjacent companions are included)
                if any(_spans_are_annotation_companions(base, other) for base in full_group_spans):
                    full_group_spans.append(other)
                    newly_added = True
        full_group_spans.sort(key=lambda s: s.bbox_pdf[0])
        full_group_text = " ".join(s.text.strip() for s in full_group_spans if s.text.strip())
        # Extend the union bbox to include companion spans
        full_union_bbox = m_union_bbox
        for _gs in full_group_spans:
            full_union_bbox = (
                min(full_union_bbox[0], _gs.bbox_pdf[0]),
                min(full_union_bbox[1], _gs.bbox_pdf[1]),
                max(full_union_bbox[2], _gs.bbox_pdf[2]),
                max(full_union_bbox[3], _gs.bbox_pdf[3]),
            )
        # Also use the matched span text directly for the signature
        mspan_text = mspan.text.strip()
        matched_annotation_signatures.append((full_group_text or mspan_text, full_union_bbox))
        if full_group_text != mspan_text:
            matched_annotation_signatures.append((mspan_text, m_union_bbox))

    # Apply the suppressor to the collected added_items
    surviving_added_items: List[DeltaItem] = []
    for candidate in added_items:
        cand_text = candidate.added_requirement_text or (
            candidate.added_span.text if candidate.added_span else ""
        )
        cand_bbox = candidate.added_bbox or (
            candidate.added_span.bbox_pdf if candidate.added_span else None
        )

        suppressed = False
        suppression_reason = ""
        for match_text, match_bbox in matched_annotation_signatures:
            # Gate 1: content ownership
            if not _is_content_subset(cand_text, match_text):
                continue
            # Gate 2: bbox ownership
            if cand_bbox is not None:
                overlap = _bbox_overlap_ratio(cand_bbox, match_bbox)
                if overlap < 0.3:
                    continue
            suppressed = True
            suppression_reason = (
                f"explained by an existing matched characteristic: "
                f"candidate grouped text '{cand_text}' is a content subset of "
                f"matched annotation '{match_text[:80]}' "
                f"(bbox containment={_bbox_overlap_ratio(cand_bbox, match_bbox):.2f})"
            )
            break

        if suppressed:
            # Record suppression in debug output but do not include in results
            # (the suppression reason is available for inspection)
            _ = suppression_reason  # available for future debug export
            continue

        surviving_added_items.append(candidate)

    return surviving_added_items


# ---------------------------------------------------------------------------
# CLS-02: Removed+Added reconciliation post-pass
# ---------------------------------------------------------------------------

# Maximum Euclidean distance (in PDF points) between removed-side and added-side
# centroids for the pair to be eligible for reconciliation.
CLS02_MAX_DISTANCE_PT = 150.0


def reconcile_removed_added_pairs(
    items: List[DeltaItem],
    anchors: List[Anchor],
) -> List[DeltaItem]:
    """Post-pass that collapses close-proximity removed+added pairs into 'changed' rows.

    Contract:
    - Build anchor_by_char_no from the supplied anchors list.
    - Only consider removed items that have a matching anchor.
    - Only consider added items that have added_bbox and added_page set.
    - Removed-side point: centroid of anchor.req_bbox if present,
      otherwise centroid of anchor.balloon_bbox.
    - Added-side point: centroid of added_bbox.
    - Page gate: anchor.page must equal added_page.
    - Distance gate: Euclidean distance <= CLS02_MAX_DISTANCE_PT.
    - Type gate: classify_requirement_type on both sides must be compatible
      (not incompatible per are_requirement_types_incompatible).
    - Resolve one-to-one by selecting the nearest compatible added item.
    - Rewrite the removed item in place: status='changed', attach added_span,
      append reconciliation reasons, raise confidence to at least 0.70.
    - Return a new list that omits consumed added items.
    """
    anchor_by_char_no: Dict[int, Anchor] = {a.char_no: a for a in anchors}

    # Partition items
    # Use getattr with None fallback to tolerate legacy/fake DeltaItem objects
    # (e.g., _FakeInternalDeltaItem in tests) that pre-date the CLS-02 metadata fields.
    removed_items = [it for it in items if it.status == "removed"]
    added_candidates = [
        it for it in items
        if it.status == "added"
        and getattr(it, "added_bbox", None) is not None
        and getattr(it, "added_page", None) is not None
    ]

    consumed_added_char_nos: set = set()

    for removed in removed_items:
        anchor = anchor_by_char_no.get(removed.char_no)
        if anchor is None:
            continue
        removed_semantic_identity = _semantic_identity_from_text(anchor.requirement_raw)
        removed_gdt_datums = _gdt_datum_refs_from_text(anchor.requirement_raw)

        # Removed-side centroid: req_bbox preferred, balloon_bbox fallback
        if anchor.req_bbox is not None:
            rx0, ry0, rx1, ry1 = anchor.req_bbox
        else:
            rx0, ry0, rx1, ry1 = anchor.balloon_bbox
        removed_cx = (rx0 + rx1) / 2.0
        removed_cy = (ry0 + ry1) / 2.0

        removed_type = classify_requirement_type(anchor.requirement_raw)

        best_added = None
        best_dist = float("inf")

        for added in added_candidates:
            if added.char_no in consumed_added_char_nos:
                continue
            # Page gate
            if getattr(added, "added_page", None) != anchor.page:
                continue
            # Distance gate
            ax0, ay0, ax1, ay1 = getattr(added, "added_bbox")
            added_cx = (ax0 + ax1) / 2.0
            added_cy = (ay0 + ay1) / 2.0
            dist = math.sqrt((removed_cx - added_cx) ** 2 + (removed_cy - added_cy) ** 2)
            if dist > CLS02_MAX_DISTANCE_PT:
                continue
            # Type gate
            added_text = getattr(added, "added_requirement_text", None) or (added.added_span.text if added.added_span else "")
            added_type = classify_requirement_type(added_text)
            if are_requirement_types_incompatible(removed_type, added_type):
                continue
            added_semantic_identity = _semantic_identity_from_text(
                added_text,
                pdf_spans=getattr(added.added_span, "source_spans", None),
            )
            if _semantic_identities_conflict(removed_semantic_identity, added_semantic_identity):
                continue
            added_gdt_datums = _gdt_datum_refs_from_text(
                added_text,
                pdf_spans=getattr(added.added_span, "source_spans", None),
            )
            if (
                removed_semantic_identity is not None
                and removed_semantic_identity[0] == "gdt"
                and removed_gdt_datums
                and added_gdt_datums
                and removed_gdt_datums != added_gdt_datums
            ):
                continue
            removed_fp = parse_requirement(anchor.requirement_raw)
            added_fp = parse_requirement(added_text)
            removed_numerics = {value for value, _ in removed_fp.numeric_tokens}
            added_numerics = {value for value, _ in added_fp.numeric_tokens}
            removed_primary = max(removed_numerics) if removed_numerics else None
            added_primary = max(added_numerics) if added_numerics else None
            if (
                removed_primary is not None
                and added_primary is not None
                and removed_primary != 0
                and removed_primary != added_primary
                and abs(removed_primary - added_primary) / abs(removed_primary) > 0.15
            ):
                continue
            # Closest-wins
            if dist < best_dist:
                best_dist = dist
                best_added = added

        if best_added is None:
            continue

        # Rewrite the removed item in place
        removed.status = "changed"
        removed.confidence = max(removed.confidence, 0.70)
        removed.added_span = best_added.added_span
        removed.reasons.append(
            f"reconciled removed+added pair: char_no={removed.char_no} merged with "
            f"added char_no={best_added.char_no} "
            f"(distance={best_dist:.1f} pt, removed_type={removed_type}, "
            f"added_type={classify_requirement_type(best_added.added_requirement_text or '')})"
        )
        consumed_added_char_nos.add(best_added.char_no)

    # Return new list omitting consumed added items
    return [it for it in items if not (it.status == "added" and it.char_no in consumed_added_char_nos)]
