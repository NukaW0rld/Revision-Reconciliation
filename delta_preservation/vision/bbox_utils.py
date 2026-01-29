"""
Bounding box utilities for snippet generation.

This module provides functions for combining, expanding, and normalizing
bounding boxes to ensure snippets capture complete annotation regions
including symbols, prefixes, and suffixes.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

from delta_preservation.io.pdf import TextSpan


@dataclass
class ExpandedBbox:
    """
    Result of expanding a bbox to include adjacent annotation spans.
    
    Attributes:
        bbox: The expanded bounding box (x0, y0, x1, y1) in PDF coordinates
        included_spans: List of spans that were merged into this bbox
        expansion_reason: Human-readable description of what was expanded
    """
    bbox: Tuple[float, float, float, float]
    included_spans: List[TextSpan]
    expansion_reason: str


def union_bbox(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """
    Compute the union (minimum enclosing box) of two bounding boxes.
    
    Args:
        bbox1: First bounding box (x0, y0, x1, y1)
        bbox2: Second bounding box (x0, y0, x1, y1)
    
    Returns:
        Union bounding box containing both inputs
    """
    x0 = min(bbox1[0], bbox2[0])
    y0 = min(bbox1[1], bbox2[1])
    x1 = max(bbox1[2], bbox2[2])
    y1 = max(bbox1[3], bbox2[3])
    return (x0, y0, x1, y1)


def compute_combined_evidence_bbox(
    balloon_bbox: Tuple[float, float, float, float],
    req_bbox: Optional[Tuple[float, float, float, float]],
    min_width: float = 50.0,
    min_height: float = 30.0
) -> Tuple[float, float, float, float]:
    """
    Compute a combined bbox that includes both the balloon and the annotation.
    
    For Rev A snippets, we want to show both the balloon (circled number) and
    the characteristic annotation it points to. This function creates a bbox
    that encompasses both elements.
    
    Args:
        balloon_bbox: Bounding box of the balloon in PDF coordinates
        req_bbox: Bounding box of the requirement annotation (may be None)
        min_width: Minimum width of the result in PDF points
        min_height: Minimum height of the result in PDF points
    
    Returns:
        Combined bounding box in PDF coordinates
    """
    if req_bbox is None:
        # Fallback: expand balloon bbox to meet minimum dimensions
        bx0, by0, bx1, by1 = balloon_bbox
        width = bx1 - bx0
        height = by1 - by0
        
        # Ensure minimum dimensions
        if width < min_width:
            expand = (min_width - width) / 2
            bx0 -= expand
            bx1 += expand
        
        if height < min_height:
            expand = (min_height - height) / 2
            by0 -= expand
            by1 += expand
        
        return (bx0, by0, bx1, by1)
    
    # Union of balloon and annotation
    combined = union_bbox(balloon_bbox, req_bbox)
    
    # Ensure minimum dimensions
    x0, y0, x1, y1 = combined
    width = x1 - x0
    height = y1 - y0
    
    if width < min_width:
        expand = (min_width - width) / 2
        x0 -= expand
        x1 += expand
    
    if height < min_height:
        expand = (min_height - height) / 2
        y0 -= expand
        y1 += expand
    
    return (x0, y0, x1, y1)


def expand_bbox_with_adjacent_spans(
    center_bbox: Tuple[float, float, float, float],
    all_spans: List[TextSpan],
    horizontal_tolerance: float = 15.0,
    vertical_tolerance: float = 5.0,
    max_horizontal_expansion: float = 100.0
) -> ExpandedBbox:
    """
    Expand a bbox to include adjacent spans that are part of the same annotation.
    
    Engineering drawing annotations often consist of multiple text spans that
    should be treated as a single unit (e.g., "⌴ Ø13.5" may be split into
    ["⌴", "Ø13.5"] or "Ø12 +0.3/+0.1" may be split into ["Ø12", "+0.3/+0.1"]).
    
    This function looks for spans that are:
    - On approximately the same vertical line (within vertical_tolerance)
    - Close horizontally (within horizontal_tolerance of the bbox edges)
    
    IMPORTANT: Only expands horizontally on the SAME LINE. Does not expand to
    include spans on different lines, as those are typically separate characteristics.
    
    Args:
        center_bbox: The starting bounding box to expand
        all_spans: All text spans on the page
        horizontal_tolerance: Max horizontal gap to bridge when merging spans
        vertical_tolerance: Max vertical offset for spans to be considered same line
        max_horizontal_expansion: Maximum total horizontal expansion allowed
    
    Returns:
        ExpandedBbox with the expanded bbox and included spans
    """
    x0, y0, x1, y1 = center_bbox
    center_y = (y0 + y1) / 2
    bbox_height = y1 - y0
    
    included_spans = []
    reasons = []
    
    # Find spans on the same horizontal line (strict vertical tolerance)
    # Only merge spans that are truly on the same line
    for span in all_spans:
        sx0, sy0, sx1, sy1 = span.bbox_pdf
        span_center_y = (sy0 + sy1) / 2
        span_height = sy1 - sy0
        
        # Check if on same line - use stricter criteria:
        # 1. Center y must be close
        # 2. Heights should be similar (within 50% of each other)
        y_diff = abs(span_center_y - center_y)
        height_ratio = min(span_height, bbox_height) / max(span_height, bbox_height, 1)
        
        if y_diff > vertical_tolerance or height_ratio < 0.5:
            continue
        
        # Check if horizontally adjacent (to the left or right)
        # Left adjacency: span ends near our start
        left_gap = x0 - sx1
        # Right adjacency: span starts near our end
        right_gap = sx0 - x1
        
        if 0 <= left_gap <= horizontal_tolerance:
            # Span is to the left, include it
            included_spans.append(span)
            x0 = min(x0, sx0)
            reasons.append(f"left: '{span.text.strip()}'")
        elif 0 <= right_gap <= horizontal_tolerance:
            # Span is to the right, include it
            included_spans.append(span)
            x1 = max(x1, sx1)
            reasons.append(f"right: '{span.text.strip()}'")
    
    # Enforce max expansion limit
    original_width = center_bbox[2] - center_bbox[0]
    new_width = x1 - x0
    if new_width - original_width > max_horizontal_expansion:
        # Revert to modest expansion
        x0 = center_bbox[0] - max_horizontal_expansion / 2
        x1 = center_bbox[2] + max_horizontal_expansion / 2
        reasons = ["limited expansion"]
    
    # Expand vertically ONLY to include subscripts/superscripts of already-included spans
    # NOT to include spans on different lines
    # This is a conservative expansion for tolerance values like "+0.3/+0.1"
    for span in included_spans:
        sx0, sy0, sx1, sy1 = span.bbox_pdf
        # Only expand if the span overlaps horizontally with our current bbox
        if sx1 >= x0 and sx0 <= x1:
            if sy0 < y0:
                y0 = sy0
            if sy1 > y1:
                y1 = sy1
    
    reason_str = ", ".join(reasons) if reasons else "no expansion needed"
    
    return ExpandedBbox(
        bbox=(x0, y0, x1, y1),
        included_spans=included_spans,
        expansion_reason=reason_str
    )


def normalize_snippet_size(
    revA_bbox: Tuple[float, float, float, float],
    revB_bbox: Tuple[float, float, float, float]
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    """
    Normalize two bboxes to have the same dimensions while preserving centers.
    
    For paired snippets (revA and revB), we want them to have consistent sizes
    for easier visual comparison. This function takes the maximum dimensions
    from both boxes and applies them to each, centered on the original centers.
    
    Args:
        revA_bbox: Bounding box for Rev A snippet
        revB_bbox: Bounding box for Rev B snippet
    
    Returns:
        Tuple of (normalized_revA_bbox, normalized_revB_bbox) with same dimensions
    """
    # Calculate dimensions
    widthA = revA_bbox[2] - revA_bbox[0]
    heightA = revA_bbox[3] - revA_bbox[1]
    widthB = revB_bbox[2] - revB_bbox[0]
    heightB = revB_bbox[3] - revB_bbox[1]
    
    # Take maximum dimensions
    max_width = max(widthA, widthB)
    max_height = max(heightA, heightB)
    
    # Calculate centers
    centerA_x = (revA_bbox[0] + revA_bbox[2]) / 2
    centerA_y = (revA_bbox[1] + revA_bbox[3]) / 2
    centerB_x = (revB_bbox[0] + revB_bbox[2]) / 2
    centerB_y = (revB_bbox[1] + revB_bbox[3]) / 2
    
    # Create normalized bboxes centered on original centers
    norm_revA = (
        centerA_x - max_width / 2,
        centerA_y - max_height / 2,
        centerA_x + max_width / 2,
        centerA_y + max_height / 2
    )
    
    norm_revB = (
        centerB_x - max_width / 2,
        centerB_y - max_height / 2,
        centerB_x + max_width / 2,
        centerB_y + max_height / 2
    )
    
    return (norm_revA, norm_revB)


def find_best_span_for_requirement(
    requirement_text: str,
    spans: List[TextSpan],
    reference_point: Tuple[float, float],
    search_radius: float = 200.0
) -> Optional[TextSpan]:
    """
    Find the best matching span for a requirement text near a reference point.
    
    This is used as a fallback when the anchor's req_bbox is None. It searches
    for spans that contain key numeric values from the requirement.
    
    Args:
        requirement_text: The requirement text to match
        spans: All spans to search through
        reference_point: (x, y) point to search around (typically balloon center)
        search_radius: Maximum distance from reference point
    
    Returns:
        Best matching TextSpan or None if no good match found
    """
    import re
    
    # Extract numeric values from requirement
    req_numerics = set(re.findall(r'\d+\.?\d*', requirement_text))
    if not req_numerics:
        return None
    
    ref_x, ref_y = reference_point
    best_span = None
    best_score = 0.0
    
    for span in spans:
        # Calculate distance
        sx0, sy0, sx1, sy1 = span.bbox_pdf
        span_cx = (sx0 + sx1) / 2
        span_cy = (sy0 + sy1) / 2
        dist = ((span_cx - ref_x)**2 + (span_cy - ref_y)**2)**0.5
        
        if dist > search_radius:
            continue
        
        # Check numeric overlap
        span_numerics = set(re.findall(r'\d+\.?\d*', span.text))
        overlap = len(req_numerics & span_numerics)
        
        if overlap == 0:
            continue
        
        # Score: overlap / distance (with minimum distance to avoid division issues)
        score = overlap / max(dist, 10.0)
        
        if score > best_score:
            best_score = score
            best_span = span
    
    return best_span
