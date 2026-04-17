"""
Shared exclusion helpers for annotation-search filtering across all reconcile callers.

This module centralizes the title-block, revision-table, and boilerplate-text
exclusion logic that was previously duplicated across match.py, anchors.py, and
classify.py.  Every annotation-search caller (anchor lookup, candidate generation,
rescue scans, and added detection) must import from here so the exclusion contract
stays in one place and cannot drift between modules.

Public API
----------
estimate_page_dimensions(spans, *, min_width=612.0, min_height=792.0)
    -> (float, float)

is_boilerplate_candidate_text(text: str) -> bool

span_is_excluded_for_annotation_search(span, *, page_width: float, page_height: float)
    -> bool
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from delta_preservation.io.pdf import TextSpan


# ---------------------------------------------------------------------------
# Boilerplate text registry
# ---------------------------------------------------------------------------

_TITLE_BLOCK_EXACT = frozenset({
    "DATE", "TITLE", "NAME", "DRAWN", "CHECKED", "DESIGNED", "APPROVED",
    "REV", "REV.", "REVISION", "MATERIAL", "MATERIALS", "FINISH",
    "SCALE", "SHEET", "SIZE", "DWG NO.", "DWG NO", "WEIGHT",
    "THIRD ANGLE PROJECTION", "FIRST ANGLE PROJECTION",
    "DO NOT SCALE DRAWING", "DO NOT SCALE",
    "BREAK ALL SHARP EDGES AND", "BREAK ALL SHARP EDGES", "REMOVE BURRS",
    "DIMENSIONS ARE IN INCHES", "DIMENSIONS ARE IN MM",
    "DIMENSIONS ARE IN MILLIMETERS",
    "SURFACE FINISH", "PROPRIETARY", "CONFIDENTIAL",
    "UNLESS OTHERWISE SPECIFIED", "UNLESS OTHERWISE SPECIFIED,",
    "INTERPRET DRAWING PER", "INTERPRET DIMENSIONS PER",
    "THIRD ANGLE", "DRAWING NO", "DRAWING NO.",
    "MACHINING",
})

_TITLE_BLOCK_CONTAINS = (
    "UNLESS OTHERWISE SPECIFIED",
    "INTERPRET DIMENSIONING AND TOLERANCING",
    "INTERPRET DRAWING PER",
    "DO NOT SCALE",
    "THIRD ANGLE",
    "FIRST ANGLE",
    "BREAK ALL SHARP",
    "REMOVE BURRS",
    "DIMENSIONS ARE IN",
)

_TITLE_BLOCK_PATTERNS = [
    re.compile(r"^\s*\.X+\s*=\s*"),           # .X = ±.050, .XX = ±.020
    re.compile(r"^\s*\d+\s+of\s+\d+\s*$"),    # "1 of 1", "2 of 3"
    re.compile(r"^(ANGULAR|FRACTIONAL)\s*="),  # ANGULAR = ±0.3°
    re.compile(r"^\d+\.\s+(QTY|ALL DIMS|MATL|FINISH|BREAK|INTERPRET)"),  # Notes block items
]


def is_boilerplate_candidate_text(text: str) -> bool:
    """Return True for general tolerance/title-block boilerplate text.

    These spans often contain engineering numerics/symbols (e.g. "ANGULAR = ±0.3°")
    but are not characteristic annotations and should never be matched or detected as
    added characteristics.

    Normalizes text with ``" ".join(text.strip().upper().split())`` before checks
    to handle whitespace variations consistently.
    """
    upper = " ".join(text.strip().upper().split())
    if not upper:
        return False
    if upper in _TITLE_BLOCK_EXACT:
        return True
    for phrase in _TITLE_BLOCK_CONTAINS:
        if phrase in upper:
            return True
    for pat in _TITLE_BLOCK_PATTERNS:
        if pat.match(upper):
            return True
    return False


# ---------------------------------------------------------------------------
# Page-dimension estimation
# ---------------------------------------------------------------------------

def estimate_page_dimensions(
    spans: Sequence[TextSpan],
    *,
    min_width: float = 612.0,
    min_height: float = 792.0,
) -> Tuple[float, float]:
    """Estimate page width and height from the bounding boxes of text spans.

    The minimum floors (default 612 × 792 pt = US letter) ensure exclusion zones
    remain meaningful even when the span list is very small (e.g., unit-test fixtures
    with only a few fabricated spans).  Real engineering drawings always have text
    extents larger than the floor.

    Returns:
        (width, height) where each is at least the specified floor value.
    """
    if not spans:
        return min_width, min_height

    max_x1 = max(s.bbox_pdf[2] for s in spans)
    max_y1 = max(s.bbox_pdf[3] for s in spans)

    return max(max_x1, min_width), max(max_y1, min_height)


# ---------------------------------------------------------------------------
# Geometric exclusion
# ---------------------------------------------------------------------------

def span_is_excluded_for_annotation_search(
    span: TextSpan,
    *,
    page_width: float,
    page_height: float,
) -> bool:
    """Return True when a span lies in a non-annotation boilerplate region.

    Exclusion zones (all defined using span *centres*, not raw x0/y0):

    1. Bottom-right title block:
       centre-x > 70 % width  AND  centre-y > 85 % height

    2. Upper-right revision table:
       centre-x > 75 % width  AND  centre-y < 15 % height

    3. Bottom-centre general tolerance / defaults block
       (78 % – 92 % height, 25 % – 72 % width):
       only when text is also flagged as boilerplate by
       ``is_boilerplate_candidate_text``

    4. Any span anywhere whose text is recognised as boilerplate by
       ``is_boilerplate_candidate_text``

    Using span centres (not raw corners) means a span that merely *extends*
    into a boilerplate zone is not incorrectly rejected — the span's visual
    centre must lie in the zone.
    """
    sx0, sy0, sx1, sy1 = span.bbox_pdf
    span_cx = (sx0 + sx1) / 2
    span_cy = (sy0 + sy1) / 2

    # Zone 1: Bottom-right title block
    if span_cx > page_width * 0.70 and span_cy > page_height * 0.85:
        return True

    # Zone 2: Upper-right revision table
    if span_cx > page_width * 0.75 and span_cy < page_height * 0.15:
        return True

    # Zone 3: Bottom-centre tolerance / defaults block
    if (
        page_height * 0.78 <= span_cy <= page_height * 0.92
        and page_width * 0.25 <= span_cx <= page_width * 0.72
        and is_boilerplate_candidate_text(span.text)
    ):
        return True

    # Zone 4: Any boilerplate text regardless of position
    if is_boilerplate_candidate_text(span.text):
        return True

    return False
