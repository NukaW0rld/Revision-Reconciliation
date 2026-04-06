"""PDF drawing tolerance parsing.

This module implements tolerance extraction from PDF drawing text spans.

Scope (Milestone 1):
- Parse tolerance from a provided *annotation group* of :class:`~delta_preservation.io.pdf.TextSpan`.
- Support these representation kinds:
  - none
  - plus_minus (inline ± or +/-)
  - bilateral_stacked (stacked + / - values)
  - unilateral_stacked (stacked same-sign values)
  - limits_stacked (stacked absolute max/min limits)

All extracted tolerances are normalized to **absolute limits**:
- ``upper_limit``: maximum permissible value
- ``lower_limit``: minimum permissible value

Later milestones will build annotation groups from raw spans and attach the parsed
results to pipeline objects (anchors/candidates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import re
import json
from pathlib import Path

from delta_preservation.io.pdf import TextSpan

try:
    # Imported only for typing / structured access.
    from delta_preservation.reconcile.anchors import Anchor
    from delta_preservation.reconcile.classify import DeltaItem as DeltaItemInternal
except Exception:  # pragma: no cover
    Anchor = Any  # type: ignore
    DeltaItemInternal = Any  # type: ignore


PdfToleranceKind = Literal[
    "none",
    "plus_minus",
    "bilateral_stacked",
    "unilateral_stacked",
    "limits_stacked",
]


_NUM_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")


@dataclass(frozen=True)
class PdfTolerance:
    """Represents tolerance extracted from PDF drawing annotation spans.

    Attributes:
        kind: The detected formatting kind for this tolerance.
        nominal_value: The base dimension value (if inferred). For limits-only
            stacked format, this is computed as the midpoint of the limits.
        upper_limit: Maximum permissible value (absolute limit).
        lower_limit: Minimum permissible value (absolute limit).
        confidence: Parser confidence in [0, 1].
        source_spans: Identifiers for spans used as evidence.
        reasons: Human-readable parsing notes.
    """

    kind: PdfToleranceKind
    nominal_value: Optional[float]
    upper_limit: Optional[float]
    lower_limit: Optional[float]
    confidence: float
    source_spans: List[Tuple[int, int, int, Tuple[float, float, float, float]]]
    reasons: List[str]

    def to_dict(self) -> Dict:
        return {
            "kind": self.kind,
            "nominal_value": self.nominal_value,
            "upper_limit": self.upper_limit,
            "lower_limit": self.lower_limit,
            "confidence": self.confidence,
            "source_spans": [
                {
                    "block_id": b,
                    "line_id": l,
                    "span_id": s,
                    "bbox_pdf": list(bbox),
                }
                for (b, l, s, bbox) in self.source_spans
            ],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ToleranceComparison:
    """Result of comparing Rev A and Rev B tolerances for a single characteristic."""

    char_no: int
    revA_tolerance: PdfTolerance
    revB_tolerance: PdfTolerance
    tolerances_match: bool
    tolerances_differ: bool
    has_tolerance: bool
    reasons: List[str]


def compare_tolerances(
    revA_tol: PdfTolerance,
    revB_tol: PdfTolerance,
    epsilon: float = 1e-6,
) -> Tuple[bool, bool, bool, List[str]]:
    """Compare two PdfTolerance objects by absolute limits.

    Returns:
        (tolerances_match, tolerances_differ, has_tolerance, reasons)

        - tolerances_match: True only when both sides have a real tolerance and limits agree
        - tolerances_differ: True only when both sides have a real tolerance and limits disagree
        - has_tolerance: True when both sides have a non-"none" tolerance
        - reasons: human-readable notes
    """
    a_has = revA_tol.kind != "none" and revA_tol.upper_limit is not None
    b_has = revB_tol.kind != "none" and revB_tol.upper_limit is not None

    if not a_has and not b_has:
        return False, False, False, ["neither rev has tolerance"]

    if a_has and not b_has:
        return False, False, False, ["only Rev A has tolerance"]

    if not a_has and b_has:
        return False, False, False, ["only Rev B has tolerance"]

    # Both have tolerance — compare limits
    upper_ok = abs((revA_tol.upper_limit or 0) - (revB_tol.upper_limit or 0)) < epsilon
    lower_ok = abs((revA_tol.lower_limit or 0) - (revB_tol.lower_limit or 0)) < epsilon

    if upper_ok and lower_ok:
        reasons = [
            f"tolerances match: upper={revA_tol.upper_limit}, lower={revA_tol.lower_limit}"
        ]
        return True, False, True, reasons
    else:
        reasons = [
            f"tolerances differ: revA=[{revA_tol.lower_limit}, {revA_tol.upper_limit}] "
            f"vs revB=[{revB_tol.lower_limit}, {revB_tol.upper_limit}]"
        ]
        return False, True, True, reasons


def _span_key(span: TextSpan) -> Tuple[int, int, int, Tuple[float, float, float, float]]:
    return (span.block_id, span.line_id, span.span_id, span.bbox_pdf)


def _extract_numbers(text: str) -> List[float]:
    """Extract signed numeric tokens from text.

    Supports leading-dot formats like ``.160`` and signed values like ``-0.1``.
    """

    nums: List[float] = []
    for m in _NUM_RE.findall(text.replace("\x00", "")):
        try:
            nums.append(float(m))
        except ValueError:
            continue
    return nums


def _normalized_join(spans: Sequence[TextSpan]) -> str:
    """Concatenate span texts in a stable reading-ish order."""

    def sort_key(s: TextSpan) -> Tuple[float, float]:
        x0, y0, x1, y1 = s.bbox_pdf
        return (y0, x0)

    return " ".join(s.text.strip() for s in sorted(spans, key=sort_key) if s.text.strip())


def _parse_inline_plus_minus(text: str) -> Optional[Tuple[float, float, float]]:
    """Parse inline plus/minus tolerances.

    Returns:
        (nominal, upper_limit, lower_limit)
    """

    marker = re.search(r"±|\+/-", text)
    if marker is None:
        return None

    left_numbers = _extract_numbers(text[: marker.start()])
    right_numbers = _extract_numbers(text[marker.end() :])
    if not left_numbers or not right_numbers:
        return None

    # Use the last numeric before the tolerance marker as the nominal so
    # prefixes like "2X" or "Ø" do not steal the parse from leading-decimal
    # dimensions such as "Ø.438 ±.005".
    nominal = left_numbers[-1]
    tol = abs(right_numbers[0])
    return (nominal, nominal + tol, nominal - tol)


def _parse_inline_signed_pair(text: str) -> Optional[Tuple[float, float, float, PdfToleranceKind]]:
    """Parse inline signed tolerance pairs like ``8.0 +0.3 -0.1``.

    This covers cases where the tolerance values are present inline (not stacked)
    with either same-sign (unilateral) or opposite-sign (bilateral).

    Returns:
        (nominal, upper_limit, lower_limit, kind)
    """

    # Example matches:
    # - 8.0 +0.3 -0.1
    # - Ø8.0 +0.3/+0.1
    # - 12 +0.3 / +0.1
    pair_re = re.compile(
        r"[^0-9+-]*"
        r"(?P<nom>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
        r"(?P<t1>[+-](?:\d+(?:\.\d*)?|\.\d+))\s*(?:/)?\s*"
        r"(?P<t2>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    )
    m = pair_re.search(text)
    if not m:
        return None

    nominal = float(m.group("nom"))
    t1 = float(m.group("t1"))
    raw_t2 = m.group("t2")

    # If t2 lacks a sign, assume same sign as t1 (common: "+0.3 0")
    if raw_t2.startswith(("+", "-")):
        t2 = float(raw_t2)
    else:
        t2 = float(raw_t2)
        if t1 < 0:
            t2 = -abs(t2)
        else:
            t2 = abs(t2)

    upper_tol = max(t1, t2)
    lower_tol = min(t1, t2)

    # Classify kind using sign relationship.
    if upper_tol >= 0 and lower_tol >= 0:
        kind: PdfToleranceKind = "unilateral_stacked"  # same semantic, even if inline
    elif upper_tol <= 0 and lower_tol <= 0:
        kind = "unilateral_stacked"
    else:
        kind = "bilateral_stacked"  # opposite-sign bilateral, even if inline

    return (nominal, nominal + upper_tol, nominal + lower_tol, kind)


def _x_overlap_ratio(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    denom = max(1e-6, min(ax1 - ax0, bx1 - bx0))
    return inter / denom


def _center_y(bbox: Tuple[float, float, float, float]) -> float:
    return (bbox[1] + bbox[3]) / 2


def _height(bbox: Tuple[float, float, float, float]) -> float:
    return bbox[3] - bbox[1]


def _detect_best_stacked_pair(spans: Sequence[TextSpan]) -> Optional[Tuple[TextSpan, TextSpan, float]]:
    """Find the best vertical stacked pair among spans.

    Returns:
        (upper_span, lower_span, score)
    """

    numeric_spans = [s for s in spans if _NUM_RE.search(s.text.strip())]
    if len(numeric_spans) < 2:
        return None

    best: Optional[Tuple[TextSpan, TextSpan, float]] = None

    for i in range(len(numeric_spans)):
        for j in range(i + 1, len(numeric_spans)):
            s1 = numeric_spans[i]
            s2 = numeric_spans[j]
            r = _x_overlap_ratio(s1.bbox_pdf, s2.bbox_pdf)
            if r < 0.6:
                continue

            cy1 = _center_y(s1.bbox_pdf)
            cy2 = _center_y(s2.bbox_pdf)
            if abs(cy1 - cy2) < 1e-3:
                continue

            upper, lower = (s1, s2) if cy1 < cy2 else (s2, s1)
            gap = abs(_center_y(upper.bbox_pdf) - _center_y(lower.bbox_pdf))
            h = max(_height(upper.bbox_pdf), _height(lower.bbox_pdf), 1.0)
            gap_norm = gap / h

            # Prefer tighter stacks.
            if gap_norm > 2.0:
                continue

            score = r * (1.0 / (1.0 + gap_norm))
            if best is None or score > best[2]:
                best = (upper, lower, score)

    return best


def _is_likely_tolerance_content(spans: Sequence[TextSpan]) -> Tuple[bool, List[str]]:
    """Semantic filtering to reject balloon numbers and non-tolerance content.
    
    Returns:
        (is_likely_tolerance, rejection_reasons)
    """
    if not spans:
        return False, ["empty spans"]
    
    combined_text = " ".join(s.text.strip() for s in spans)
    rejection_reasons = []
    
    # Reject if it looks like just balloon numbers (1-2 digits only)
    if re.match(r'^\s*\d{1,2}\s*$', combined_text):
        rejection_reasons.append("appears to be balloon number only")
        return False, rejection_reasons
    
    # Reject if it contains only dimension prefixes without tolerance indicators
    # e.g., "R12", "Ø8.5", "12.0" without any ±, +, - signs
    dimension_only_pattern = r'^\s*[RØ]?\s*\d+\.?\d*\s*$'
    if re.match(dimension_only_pattern, combined_text):
        rejection_reasons.append("appears to be dimension without tolerance")
        return False, rejection_reasons
    
    # Reject if it contains hole pattern or engineering symbols without tolerance context
    engineering_symbols = ['⌴', '↧', 'THRU ALL', 'M4', 'M6', 'G1/2', '-6H']
    has_engineering_symbols = any(symbol in combined_text for symbol in engineering_symbols)
    
    if has_engineering_symbols:
        # Allow if it also has explicit tolerance indicators
        has_explicit_tolerance = bool(re.search(r'[±+\-]\s*\d', combined_text))
        if not has_explicit_tolerance:
            rejection_reasons.append("contains engineering symbols without tolerance indicators")
            return False, rejection_reasons
    
    # For stacked detection, require tolerance indicators (±, +, -, or decimal precision)
    has_tolerance_indicators = bool(re.search(r'[±+\-]\s*\d', combined_text))
    has_decimal_precision = bool(re.search(r'\d+\.\d{2,}', combined_text))  # .125, .001 etc
    
    if not has_tolerance_indicators and not has_decimal_precision:
        # Check if we have multiple numeric values that could be limits
        numeric_matches = re.findall(r'\d+\.?\d*', combined_text)
        if len(numeric_matches) < 2:
            rejection_reasons.append("no tolerance indicators and insufficient numeric values")
            return False, rejection_reasons
    
    return True, []


def parse_pdf_tolerance(annotation_spans: Sequence[TextSpan]) -> PdfTolerance:
    """Parse tolerance from a PDF annotation group.

    The caller is responsible for providing spans that belong to a single drawing
    annotation (future milestone will build such groups).

    Returns:
        PdfTolerance normalized to absolute limits.
    """

    spans = [s for s in annotation_spans if s.text and s.text.strip()]
    if not spans:
        return PdfTolerance(
            kind="none",
            nominal_value=None,
            upper_limit=None,
            lower_limit=None,
            confidence=0.0,
            source_spans=[],
            reasons=["empty span group"],
        )
    
    # Apply semantic filtering to reject obvious non-tolerance content
    is_likely_tolerance, semantic_reasons = _is_likely_tolerance_content(spans)
    if not is_likely_tolerance:
        return PdfTolerance(
            kind="none",
            nominal_value=None,
            upper_limit=None,
            lower_limit=None,
            confidence=0.0,
            source_spans=[_span_key(s) for s in spans],
            reasons=["semantic filter rejected"] + semantic_reasons,
        )

    text = _normalized_join(spans)
    reasons: List[str] = [f"group_text='{text}'"]

    # Inline ± first (most explicit)
    pm = _parse_inline_plus_minus(text)
    if pm is not None:
        nominal, upper, lower = pm
        return PdfTolerance(
            kind="plus_minus",
            nominal_value=nominal,
            upper_limit=upper,
            lower_limit=lower,
            confidence=0.95,
            source_spans=[_span_key(s) for s in spans],
            reasons=reasons + ["parsed inline ±/+/−"],
        )

    # Inline signed pair (unilateral or uneven bilateral)
    pair = _parse_inline_signed_pair(text)
    if pair is not None:
        nominal, upper, lower, kind = pair
        conf = 0.9
        return PdfTolerance(
            kind=kind,
            nominal_value=nominal,
            upper_limit=upper,
            lower_limit=lower,
            confidence=conf,
            source_spans=[_span_key(s) for s in spans],
            reasons=reasons + ["parsed inline signed tolerance pair"],
        )

    # Stacked detection (geometry)
    stacked = _detect_best_stacked_pair(spans)
    if stacked is not None:
        upper_span, lower_span, score = stacked
        upper_nums = _extract_numbers(upper_span.text)
        lower_nums = _extract_numbers(lower_span.text)

        if upper_nums and lower_nums:
            upper_val = upper_nums[0]
            lower_val = lower_nums[0]

            upper_has_sign = upper_span.text.strip().startswith(("+", "-"))
            lower_has_sign = lower_span.text.strip().startswith(("+", "-"))
            any_sign = upper_has_sign or lower_has_sign

            other_spans = [s for s in spans if s not in (upper_span, lower_span)]
            other_numbers: List[float] = []
            for s in other_spans:
                other_numbers.extend(_extract_numbers(s.text))

            nominal: Optional[float]
            if other_numbers:
                # Use the largest magnitude as the base dimension value.
                nominal = max(other_numbers, key=lambda v: abs(v))
                reasons.append(f"nominal_from_other_spans={nominal}")
            else:
                nominal = None

            # Interpret stacked values.
            if not any_sign and nominal is None:
                # Limits-only stacked format (.160 over .140)
                u = upper_val
                l = lower_val
                nom = (u + l) / 2
                return PdfTolerance(
                    kind="limits_stacked",
                    nominal_value=nom,
                    upper_limit=u,
                    lower_limit=l,
                    confidence=min(0.9, 0.6 + score),
                    source_spans=[_span_key(upper_span), _span_key(lower_span)],
                    reasons=reasons + ["parsed unsigned stacked limits"],
                )

            if nominal is None:
                return PdfTolerance(
                    kind="none",
                    nominal_value=None,
                    upper_limit=None,
                    lower_limit=None,
                    confidence=0.2,
                    source_spans=[_span_key(upper_span), _span_key(lower_span)],
                    reasons=reasons + ["stacked pair found but nominal missing"],
                )

            # If one value lacks sign, assume same sign as the other.
            if any_sign and not upper_has_sign and lower_has_sign:
                upper_val = abs(upper_val) if lower_val >= 0 else -abs(upper_val)
            if any_sign and not lower_has_sign and upper_has_sign:
                lower_val = abs(lower_val) if upper_val >= 0 else -abs(lower_val)

            # If both are signed values, treat as tolerances.
            if any_sign:
                upper_tol = upper_val
                lower_tol = lower_val

                if upper_tol >= 0 and lower_tol >= 0:
                    kind2: PdfToleranceKind = "unilateral_stacked"
                elif upper_tol <= 0 and lower_tol <= 0:
                    kind2 = "unilateral_stacked"
                else:
                    kind2 = "bilateral_stacked"

                return PdfTolerance(
                    kind=kind2,
                    nominal_value=nominal,
                    upper_limit=nominal + upper_tol,
                    lower_limit=nominal + lower_tol,
                    confidence=min(0.9, 0.55 + score),
                    source_spans=[_span_key(upper_span), _span_key(lower_span)],
                    reasons=reasons + ["parsed signed stacked tolerance values"],
                )

            # Otherwise, treat as absolute limits adjacent to nominal (rare).
            return PdfTolerance(
                kind="limits_stacked",
                nominal_value=(upper_val + lower_val) / 2,
                upper_limit=upper_val,
                lower_limit=lower_val,
                confidence=min(0.7, 0.4 + score),
                source_spans=[_span_key(upper_span), _span_key(lower_span)],
                reasons=reasons + ["parsed stacked values as limits (nominal present)"],
            )

    return PdfTolerance(
        kind="none",
        nominal_value=None,
        upper_limit=None,
        lower_limit=None,
        confidence=0.3,
        source_spans=[_span_key(s) for s in spans],
        reasons=reasons + ["no tolerance pattern matched"],
    )


def _expand_bbox(
    bbox: Tuple[float, float, float, float],
    pad_x: float,
    pad_y: float,
) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)


def _bbox_contains_point(
    bbox: Tuple[float, float, float, float],
    x: float,
    y: float,
) -> bool:
    x0, y0, x1, y1 = bbox
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def _collect_spans_near_bbox(
    all_spans: Sequence[TextSpan],
    seed_bbox: Tuple[float, float, float, float],
    *,
    page_index: Optional[int],
    pad_x: float = 80.0,
    pad_y: float = 50.0,
    balloon_exclusion_locations: Optional[List[Tuple[float, float]]] = None,
    balloon_exclusion_radius: float = 25.0,
) -> List[TextSpan]:
    """Collect spans in a neighborhood around a seed bbox.

    This is intentionally conservative and used only for Milestone 1 debug exports.
    Later milestones will implement a proper annotation grouping algorithm.
    
    Args:
        balloon_exclusion_locations: List of (x, y) balloon centers to exclude spans near
        balloon_exclusion_radius: Minimum distance from balloon centers to exclude spans
    """

    region = _expand_bbox(seed_bbox, pad_x=pad_x, pad_y=pad_y)
    spans: List[TextSpan] = []
    for s in all_spans:
        # Note: page_index filtering removed since extract_text_spans() already
        # filters by page, and block_id is the text block index, not page index
        x0, y0, x1, y1 = s.bbox_pdf
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        
        # Skip if not in the collection region
        if not _bbox_contains_point(region, cx, cy):
            continue
            
        # Skip if too close to any balloon center (likely balloon number)
        if balloon_exclusion_locations:
            too_close_to_balloon = any(
                ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5 < balloon_exclusion_radius
                for bx, by in balloon_exclusion_locations
            )
            if too_close_to_balloon:
                continue
                
        spans.append(s)

    # Stable order for debug readability.
    spans.sort(key=lambda s: (s.bbox_pdf[1], s.bbox_pdf[0]))
    return spans


def _check_duplicate_tolerance(
    char_no: int,
    tolerance: PdfTolerance,
    seed_bbox: Tuple[float, float, float, float],
    detected_tolerances: List[Tuple[int, PdfTolerance, Tuple[float, float, float, float]]],
    min_distance: float = 100.0,
) -> bool:
    """Check if a tolerance has already been assigned to a nearby characteristic.
    
    This prevents spatially aligned characteristics (e.g., chars 31-32 above char 30)
    from incorrectly detecting the same tolerance.
    
    Args:
        char_no: Current characteristic number being processed
        tolerance: Detected tolerance to check
        seed_bbox: Seed bbox of current characteristic
        detected_tolerances: List of (char_no, tolerance, bbox) tuples already processed
        min_distance: Minimum distance (in PDF points) to consider as separate tolerances
    
    Returns:
        True if this appears to be a duplicate detection
    """
    # Get center of current seed bbox
    curr_x0, curr_y0, curr_x1, curr_y1 = seed_bbox
    curr_cx = (curr_x0 + curr_x1) / 2
    curr_cy = (curr_y0 + curr_y1) / 2
    
    for prev_char_no, prev_tol, prev_bbox in detected_tolerances:
        # Skip if same characteristic
        if prev_char_no == char_no:
            continue
            
        # Skip if previous tolerance was "none"
        if prev_tol.kind == "none":
            continue
        
        # Check if tolerance values match (same nominal, upper, lower)
        values_match = (
            abs((tolerance.nominal_value or 0) - (prev_tol.nominal_value or 0)) < 0.01 and
            abs((tolerance.upper_limit or 0) - (prev_tol.upper_limit or 0)) < 0.01 and
            abs((tolerance.lower_limit or 0) - (prev_tol.lower_limit or 0)) < 0.01
        )
        
        if not values_match:
            continue
        
        # Check spatial proximity
        prev_x0, prev_y0, prev_x1, prev_y1 = prev_bbox
        prev_cx = (prev_x0 + prev_x1) / 2
        prev_cy = (prev_y0 + prev_y1) / 2
        
        distance = ((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2) ** 0.5
        
        # If characteristics are close and have identical tolerance values, it's likely a duplicate
        if distance < min_distance:
            return True
    
    return False


def _is_closest_balloon(
    current_balloon_center: Tuple[float, float],
    source_spans: List[Tuple[int, int, int, Tuple[float, float, float, float]]],
    all_balloon_centers: List[Tuple[float, float]],
    all_balloon_char_nos: List[int],
    current_char_no: int,
    group_spans: Sequence[TextSpan],
) -> Tuple[bool, Optional[int]]:
    """Check whether the current balloon is the closest to the tolerance source spans.

    Args:
        current_balloon_center: (x, y) center of the balloon being evaluated.
        source_spans: PdfTolerance.source_spans identifying the tolerance text.
        all_balloon_centers: Centers of every balloon in the drawing.
        all_balloon_char_nos: Corresponding char_no for each center.
        current_char_no: char_no of the balloon being evaluated.
        group_spans: The TextSpan objects collected for this annotation group
            (used to identify which source_spans carry tolerance notation).

    Returns:
        (is_closest, closer_char_no) — True if no other balloon is strictly closer;
        otherwise False and the char_no of the closer balloon.
    """
    if not source_spans or not all_balloon_centers:
        return True, None

    # Build a map from span key → text so we can filter to tolerance-bearing spans
    _TOL_MARKER_RE = re.compile(r"[±]|[+]/-|[+-]\s*\d")
    span_text_map: Dict[Tuple[int, int, int, Tuple[float, float, float, float]], str] = {}
    for s in group_spans:
        span_text_map[_span_key(s)] = s.text

    # Filter source_spans to only those containing tolerance markers (±, +/-, signed numbers)
    tol_spans = [
        sp for sp in source_spans
        if sp in span_text_map and _TOL_MARKER_RE.search(span_text_map[sp])
    ]

    # If no spans matched the filter, fall back to all source_spans
    target_spans = tol_spans if tol_spans else source_spans

    # Average center of tolerance-bearing span bboxes
    sum_x = 0.0
    sum_y = 0.0
    for _b, _l, _s, bbox in target_spans:
        sum_x += (bbox[0] + bbox[2]) / 2
        sum_y += (bbox[1] + bbox[3]) / 2
    tol_cx = sum_x / len(target_spans)
    tol_cy = sum_y / len(target_spans)

    current_dist = (
        (current_balloon_center[0] - tol_cx) ** 2
        + (current_balloon_center[1] - tol_cy) ** 2
    ) ** 0.5

    for center, cno in zip(all_balloon_centers, all_balloon_char_nos):
        if cno == current_char_no:
            continue
        dist = ((center[0] - tol_cx) ** 2 + (center[1] - tol_cy) ** 2) ** 0.5
        if dist < current_dist:
            return False, cno

    return True, None


def _parse_rev_tolerance(
    *,
    seed_bbox: Optional[Tuple[float, float, float, float]],
    all_spans: Sequence[TextSpan],
    page_index: Optional[int],
    is_notes_type: bool,
    balloon_exclusion_locations: Optional[List[Tuple[float, float]]],
    char_no: int,
    detected_tolerances: List[Tuple[int, PdfTolerance, Tuple[float, float, float, float]]],
    default_reason: str,
    current_balloon_center: Optional[Tuple[float, float]] = None,
    all_balloon_centers: Optional[List[Tuple[float, float]]] = None,
    all_balloon_char_nos: Optional[List[int]] = None,
) -> Tuple[PdfTolerance, List[TextSpan]]:
    """Parse tolerance for one revision side (Rev A or Rev B).

    Shared logic extracted from the duplicated blocks in export_run_tolerance_debug().

    Returns:
        (tolerance, group_spans) — the parsed tolerance and the collected nearby spans
        (group_spans is needed by debug export for JSON output).
    """
    group_spans: List[TextSpan] = []

    if is_notes_type:
        tol = PdfTolerance(
            kind="none",
            nominal_value=None,
            upper_limit=None,
            lower_limit=None,
            confidence=0.0,
            source_spans=[],
            reasons=["NOTES type characteristic - tolerance parsing skipped"],
        )
        return tol, group_spans

    if seed_bbox is None:
        tol = PdfTolerance(
            kind="none",
            nominal_value=None,
            upper_limit=None,
            lower_limit=None,
            confidence=0.0,
            source_spans=[],
            reasons=[default_reason],
        )
        return tol, group_spans

    group_spans = _collect_spans_near_bbox(
        all_spans,
        seed_bbox,
        page_index=page_index,
        balloon_exclusion_locations=balloon_exclusion_locations,
        balloon_exclusion_radius=25.0,
    )
    tol = parse_pdf_tolerance(group_spans)

    # Ownership check: is this balloon the closest to the tolerance source spans?
    if (
        tol.kind != "none"
        and current_balloon_center is not None
        and all_balloon_centers is not None
        and all_balloon_char_nos is not None
    ):
        is_closest, closer_char = _is_closest_balloon(
            current_balloon_center=current_balloon_center,
            source_spans=tol.source_spans,
            all_balloon_centers=all_balloon_centers,
            all_balloon_char_nos=all_balloon_char_nos,
            current_char_no=char_no,
            group_spans=group_spans,
        )
        if not is_closest:
            tol = PdfTolerance(
                kind="none",
                nominal_value=None,
                upper_limit=None,
                lower_limit=None,
                confidence=0.0,
                source_spans=tol.source_spans,
                reasons=[
                    f"tolerance source closer to another balloon (char {closer_char})"
                ]
                + tol.reasons,
            )
            # Do NOT register in detected_tolerances so the rightful owner can claim it
            return tol, group_spans

    # Check for duplicate detection with previously processed characteristics
    if tol.kind != "none":
        is_duplicate = _check_duplicate_tolerance(
            char_no=char_no,
            tolerance=tol,
            seed_bbox=seed_bbox,
            detected_tolerances=detected_tolerances,
            min_distance=100.0,
        )
        if is_duplicate:
            tol = PdfTolerance(
                kind="none",
                nominal_value=None,
                upper_limit=None,
                lower_limit=None,
                confidence=0.0,
                source_spans=tol.source_spans,
                reasons=["duplicate detection - tolerance already assigned to nearby characteristic"]
                + tol.reasons,
            )
        else:
            # Register this tolerance as detected
            detected_tolerances.append((char_no, tol, seed_bbox))

    return tol, group_spans


def _span_to_debug_dict(span: TextSpan) -> Dict[str, Any]:
    return {
        "text": span.text,
        "bbox_pdf": list(span.bbox_pdf),
        "block_id": getattr(span, "block_id", None),
        "line_id": getattr(span, "line_id", None),
        "span_id": getattr(span, "span_id", None),
        "font_size": getattr(span, "font_size", None),
    }


def extract_tolerances_for_items(
    *,
    anchors: Sequence[Anchor],
    matches: Dict[int, Any],
    revA_spans: Sequence[TextSpan],
    revB_spans: Sequence[TextSpan],
) -> Dict[int, ToleranceComparison]:
    """Extract and compare tolerances for all anchors, callable BEFORE classification.

    Args:
        anchors: Rev A anchors from Stage 4.
        matches: Dict mapping char_no → Match from Stage 6.
        revA_spans: All text spans from Rev A PDF.
        revB_spans: All text spans from Rev B PDF.

    Returns:
        Dict mapping char_no → ToleranceComparison for each anchor.
    """
    # Build balloon exclusion centers and closest-balloon lookup lists
    balloon_exclusion_locations: List[Tuple[float, float]] = []
    all_balloon_centers: List[Tuple[float, float]] = []
    all_balloon_char_nos: List[int] = []
    for anchor in anchors:
        if hasattr(anchor, "balloon_bbox") and anchor.balloon_bbox:
            bx0, by0, bx1, by1 = anchor.balloon_bbox
            center = ((bx0 + bx1) / 2, (by0 + by1) / 2)
            balloon_exclusion_locations.append(center)
            all_balloon_centers.append(center)
            all_balloon_char_nos.append(int(anchor.char_no))

    detected_tolerances: List[Tuple[int, PdfTolerance, Tuple[float, float, float, float]]] = []
    result: Dict[int, ToleranceComparison] = {}

    for anchor in anchors:
        char_no = int(anchor.char_no)
        is_notes_type = "NOTES" in anchor.requirement_raw.upper()

        # Rev A seed bbox
        revA_seed_bbox = getattr(anchor, "req_bbox", None) or getattr(anchor, "balloon_bbox", None)
        revA_page = int(getattr(anchor, "page", 0))

        # Compute current balloon center for ownership check
        current_balloon_center: Optional[Tuple[float, float]] = None
        if hasattr(anchor, "balloon_bbox") and anchor.balloon_bbox:
            bx0, by0, bx1, by1 = anchor.balloon_bbox
            current_balloon_center = ((bx0 + bx1) / 2, (by0 + by1) / 2)

        revA_tol, _ = _parse_rev_tolerance(
            seed_bbox=revA_seed_bbox,
            all_spans=revA_spans,
            page_index=revA_page,
            is_notes_type=is_notes_type,
            balloon_exclusion_locations=balloon_exclusion_locations,
            char_no=char_no,
            detected_tolerances=detected_tolerances,
            default_reason="no Rev A seed bbox",
            current_balloon_center=current_balloon_center,
            all_balloon_centers=all_balloon_centers,
            all_balloon_char_nos=all_balloon_char_nos,
        )

        # Rev B seed bbox from match
        revB_seed_bbox: Optional[Tuple[float, float, float, float]] = None
        match = matches.get(char_no)
        if match is not None and getattr(match, "candidate", None) is not None:
            revB_seed_bbox = match.candidate.span.bbox_pdf

        revB_tol, _ = _parse_rev_tolerance(
            seed_bbox=revB_seed_bbox,
            all_spans=revB_spans,
            page_index=0,
            is_notes_type=is_notes_type,
            balloon_exclusion_locations=balloon_exclusion_locations,
            char_no=char_no,
            detected_tolerances=detected_tolerances,
            default_reason="no Rev B seed span (removed/unmatched)",
            current_balloon_center=current_balloon_center,
            all_balloon_centers=all_balloon_centers,
            all_balloon_char_nos=all_balloon_char_nos,
        )

        tol_match, tol_differ, has_tol, reasons = compare_tolerances(revA_tol, revB_tol)
        result[char_no] = ToleranceComparison(
            char_no=char_no,
            revA_tolerance=revA_tol,
            revB_tolerance=revB_tol,
            tolerances_match=tol_match,
            tolerances_differ=tol_differ,
            has_tolerance=has_tol,
            reasons=reasons,
        )

    return result


def export_run_tolerance_debug(
    *,
    debug_dir: Path,
    run_id: str,
    revA_pdf_path: Path,
    revB_pdf_path: Path,
    items: Sequence[DeltaItemInternal],
    anchors: Sequence[Anchor],
    revA_spans: Sequence[TextSpan],
    revB_spans: Sequence[TextSpan],
) -> Path:
    """Export per-run tolerance parsing debug for manual inspection.

    Writes a JSON file into the current run's debug directory:
    ``out/<run_id>/debug/tolerance_parsing_tests.json``.

    The output contains only tolerance parsing data per characteristic:
    - group_spans
    - group_text
    - tolerance (parsed result)
    for both Rev A and Rev B.
    """

    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / "tolerance_parsing_tests.json"

    anchors_by_char: Dict[int, Anchor] = {int(a.char_no): a for a in anchors}

    # Extract balloon centers for exclusion filtering and closest-balloon lookup
    balloon_exclusion_locations: List[Tuple[float, float]] = []
    all_balloon_centers: List[Tuple[float, float]] = []
    all_balloon_char_nos: List[int] = []
    for anchor in anchors:
        if hasattr(anchor, 'balloon_bbox') and anchor.balloon_bbox:
            bx0, by0, bx1, by1 = anchor.balloon_bbox
            center = ((bx0 + bx1) / 2, (by0 + by1) / 2)
            balloon_exclusion_locations.append(center)
            all_balloon_centers.append(center)
            all_balloon_char_nos.append(int(anchor.char_no))

    records: List[Dict[str, Any]] = []
    errors: List[str] = []

    # Track detected tolerances to prevent duplicates for spatially aligned characteristics
    detected_tolerances: List[Tuple[int, PdfTolerance, Tuple[float, float, float, float]]] = []

    for item in items:
        try:
            char_no = int(getattr(item, "char_no"))
            anchor = anchors_by_char.get(char_no)

            # Rev A seed bbox preference: req_bbox, else balloon bbox.
            revA_seed_bbox: Optional[Tuple[float, float, float, float]] = None
            revA_page: Optional[int] = None
            if anchor is not None:
                revA_page = int(getattr(anchor, "page", 0))
                revA_seed_bbox = getattr(anchor, "req_bbox", None) or getattr(anchor, "balloon_bbox", None)

            is_notes_type = False
            if anchor is not None:
                is_notes_type = "NOTES" in anchor.requirement_raw.upper()

            # Compute current balloon center for ownership check
            current_balloon_center: Optional[Tuple[float, float]] = None
            if anchor is not None and hasattr(anchor, "balloon_bbox") and anchor.balloon_bbox:
                bx0, by0, bx1, by1 = anchor.balloon_bbox
                current_balloon_center = ((bx0 + bx1) / 2, (by0 + by1) / 2)

            revA_tol, revA_group_spans = _parse_rev_tolerance(
                seed_bbox=revA_seed_bbox,
                all_spans=revA_spans,
                page_index=revA_page,
                is_notes_type=is_notes_type,
                balloon_exclusion_locations=balloon_exclusion_locations,
                char_no=char_no,
                detected_tolerances=detected_tolerances,
                default_reason="no Rev A seed bbox",
                current_balloon_center=current_balloon_center,
                all_balloon_centers=all_balloon_centers,
                all_balloon_char_nos=all_balloon_char_nos,
            )

            # Rev B seed bbox from match span or added span.
            match = getattr(item, "match", None)
            added_span = getattr(item, "added_span", None)
            revB_seed_bbox: Optional[Tuple[float, float, float, float]] = None
            if match is not None and getattr(match, "candidate", None) is not None:
                revB_seed_bbox = match.candidate.span.bbox_pdf
            elif added_span is not None:
                revB_seed_bbox = added_span.bbox_pdf

            revB_tol, revB_group_spans = _parse_rev_tolerance(
                seed_bbox=revB_seed_bbox,
                all_spans=revB_spans,
                page_index=0,
                is_notes_type=is_notes_type,
                balloon_exclusion_locations=balloon_exclusion_locations,
                char_no=char_no,
                detected_tolerances=detected_tolerances,
                default_reason="no Rev B seed span (removed/unmatched)",
                current_balloon_center=current_balloon_center,
                all_balloon_centers=all_balloon_centers,
                all_balloon_char_nos=all_balloon_char_nos,
            )

            records.append(
                {
                    "char_no": char_no,
                    "revA": {
                        "group_spans": [_span_to_debug_dict(s) for s in revA_group_spans],
                        "group_text": _normalized_join(revA_group_spans) if revA_group_spans else None,
                        "tolerance": revA_tol.to_dict(),
                    },
                    "revB": {
                        "group_spans": [_span_to_debug_dict(s) for s in revB_group_spans],
                        "group_text": _normalized_join(revB_group_spans) if revB_group_spans else None,
                        "tolerance": revB_tol.to_dict(),
                    },
                }
            )
        except Exception as e:
            errors.append(f"char_no={getattr(item, 'char_no', None)} error={type(e).__name__}: {e}")

    payload: Dict[str, Any] = {
        "items": records,
        "errors": errors,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_path
