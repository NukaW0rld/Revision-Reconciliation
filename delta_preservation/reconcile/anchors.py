"""
Anchor building module for linking Form 3 requirements to Rev A PDF spatial locations.

This module creates "anchors" that establish stable connections between AS9102 Form 3 
inspection characteristics and their corresponding spatial locations in Rev A drawings.
Each anchor serves as a reference point for finding the same characteristic in Rev B,
even when layout changes occur.

The anchoring process combines balloon detection with text matching to create robust
spatial-semantic associations that can withstand drawing revisions.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import math

from delta_preservation.io.pdf import TextSpan
from delta_preservation.vision.balloons import Balloon
from delta_preservation.reconcile.normalize import parse_requirement


@dataclass
class Anchor:
    """
    Stable spatial-semantic anchor linking Form 3 requirements to Rev A PDF locations.

    An Anchor represents the authoritative association between an inspection characteristic
    from the AS9102 Form 3 and its physical manifestation in the Rev A drawing. This
    includes both the balloon marker and the requirement text annotation.

    Attributes:
        char_no: Characteristic number from Form 3 (e.g., 1, 2, 17)
        page: Zero-based page index in Rev A PDF where characteristic is found
        balloon_bbox: Bounding box of the balloon marker in PDF coordinates
        req_bbox: Bounding box of the requirement text annotation (may be None if not found)
        requirement_raw: Original requirement text from Form 3
        requirement_norm: Normalized/parsed version of requirement text for matching
        local_context: Nearby text spans that provide spatial context for matching
        zone_bbox: Optional bounding box of the drawing grid zone from Form 3 Field 6
                   (e.g. "Zone B.2"). Used to constrain Rev A text search and as a
                   fallback anchor point when req_bbox is None. None when no zone is
                   specified or the grid could not be inferred.

    Notes:
        - Balloon bbox is always available (from balloon detection)
        - Req bbox may be None if text matching fails
        - Local context includes spans within ~150 PDF points of the characteristic
        - PDF coordinates are in points (72 DPI standard)
    """
    char_no: int
    page: int
    balloon_bbox: tuple[float, float, float, float]
    req_bbox: Optional[tuple[float, float, float, float]]
    requirement_raw: str
    requirement_norm: str
    local_context: List[TextSpan]
    zone_bbox: Optional[tuple[float, float, float, float]] = None


def build_revA_anchors(
    form3_chars: Dict[int, str],
    balloons: Dict[int, Balloon],
    revA_text_spans: List[TextSpan],
    zone_bboxes: Optional[Dict[int, tuple[float, float, float, float]]] = None,
) -> List[Anchor]:
    """
    Build spatial-semantic anchors linking Form 3 requirements to Rev A PDF locations.

    This function creates the foundation for characteristic identity preservation by
    establishing robust connections between inspection requirements and their physical
    manifestations in the drawing. Each anchor combines:
    1. Balloon marker location (from computer vision detection)
    2. Requirement text location (from intelligent text matching)
    3. Local spatial context (nearby spans for disambiguation)

    The anchoring process handles various challenges:
    - Requirements may not have exact text matches in the PDF
    - Multiple similar annotations may exist near balloons
    - Title blocks and revision tables must be avoided
    - Text spans may be fragmented or incomplete

    Args:
        form3_chars: Dictionary mapping characteristic numbers to requirement text strings
                    from AS9102 Form 3 (e.g., {1: "Ø6.0 +0.1/-0.0", 2: "R3.5 ± 0.2"})
        balloons: Dictionary mapping characteristic numbers to detected Balloon objects
                 containing spatial locations and confidence metrics
        revA_text_spans: Complete list of text spans extracted from Rev A PDF with
                        bounding boxes and content
        zone_bboxes: Optional dict mapping char_no to a grid zone bounding box
                    ``(x0, y0, x1, y1)`` in PDF points derived from Form 3 Field 6
                    reference locations and the drawing grid. When provided for a
                    characteristic, the requirement-text candidate search is restricted
                    to spans within that zone, and the zone bbox is stored on the
                    resulting Anchor for use in Rev B candidate generation.

    Returns:
        List of Anchor objects representing successfully linked characteristics.
        Only characteristics with both Form 3 entries and detected balloons are included.

    Notes:
        - Uses weighted scoring combining token overlap, distance, and size heuristics
        - Excludes title block regions (bottom 15%, right 20% of page)
        - Filters out very long spans likely to be headers or notes
        - Builds local context within 150 PDF points for spatial disambiguation
        - Falls back gracefully when requirement text cannot be precisely located
    """
    if zone_bboxes is None:
        zone_bboxes = {}

    anchors = []

    for char_no, requirement in form3_chars.items():
        if char_no not in balloons:
            continue

        balloon = balloons[char_no]
        page = balloon.page_index
        balloon_bbox = balloon.bbox_pdf
        balloon_cx, balloon_cy = balloon.center_pdf

        zone_bbox = zone_bboxes.get(char_no)

        # Filter text spans to the same page as the balloon
        # Note: block_id in TextSpan corresponds to page index from PDF extraction
        page_spans = [s for s in revA_text_spans if s.block_id == page]

        # Estimate page dimensions from text span extents for exclusion zones
        # This approximation helps identify title blocks and revision tables
        page_height = max(s.bbox_pdf[3] for s in page_spans) if page_spans else 1000
        page_width = max(s.bbox_pdf[2] for s in page_spans) if page_spans else 1000

        # Filter candidate spans for requirement text matching.
        # When a zone bbox is available, restrict candidates to spans within that
        # zone — this narrows the search to the correct region of the drawing and
        # avoids false matches from similar annotations elsewhere on the page.
        candidates = []
        for span in page_spans:
            x0, y0, x1, y1 = span.bbox_pdf

            # Skip title block regions (bottom 15% and right 20% of page)
            # These areas typically contain part numbers, revision history, etc.
            if y0 > page_height * 0.85 or x0 > page_width * 0.80:
                continue

            # Skip very long spans (likely headers, notes, or revision text)
            # Requirement annotations are typically concise (e.g., "Ø6.0 ±0.1")
            if len(span.text) > 100:
                continue

            # Skip empty or whitespace-only spans
            if not span.text.strip():
                continue

            # Zone-based pre-filter: when a zone bbox is known, only consider
            # spans whose center falls within that zone.
            if zone_bbox is not None:
                zx0, zy0, zx1, zy1 = zone_bbox
                span_cx = (x0 + x1) / 2
                span_cy = (y0 + y1) / 2
                if not (zx0 <= span_cx <= zx1 and zy0 <= span_cy <= zy1):
                    continue

            candidates.append(span)
        
        # Parse and normalize the Form 3 requirement text for token-based matching
        # This extracts symbols (Ø, R), numeric values, and normalized text
        req_norm = parse_requirement(requirement)
        req_tokens = set(req_norm.norm_text.split())
        
        # Find the best matching text span for this requirement
        best_span = None
        best_score = 0.2  # Minimum score threshold to accept a match
        
        for span in candidates:
            # Parse the candidate span text using the same normalization
            span_norm = parse_requirement(span.text)
            span_tokens = set(span_norm.norm_text.split())
            
            # Token overlap score (Jaccard similarity)
            # Measures semantic similarity between Form 3 text and PDF text
            overlap = len(req_tokens & span_tokens)
            total = len(req_tokens | span_tokens)
            token_score = overlap / total if total > 0 else 0
            
            # Distance score (spatial proximity to balloon)
            # Closer spans are more likely to be the correct annotation
            sx0, sy0, sx1, sy1 = span.bbox_pdf
            span_cx = (sx0 + sx1) / 2
            span_cy = (sy0 + sy1) / 2
            distance = math.sqrt((span_cx - balloon_cx)**2 + (span_cy - balloon_cy)**2)
            dist_score = 1.0 / (1.0 + distance / 100)  # Normalize by typical balloon-to-text spacing
            
            # Size penalty to discourage matching to large text blocks
            # Requirement annotations should be relatively compact
            span_area = (sx1 - sx0) * (sy1 - sy0)
            size_penalty = min(span_area / 5000, 0.3)  # Cap penalty at 30%
            
            # Weighted combination prioritizing semantic match with spatial validation
            score = 0.5 * token_score + 0.4 * dist_score - 0.1 * size_penalty
            
            if score > best_score:
                best_score = score
                best_span = span
        
        # Store the requirement text bounding box (None if no good match found)
        req_bbox = best_span.bbox_pdf if best_span else None
        
        # Build local spatial context for this characteristic
        # Context includes nearby text spans that can help with Rev B matching
        # Use requirement text center if found, otherwise fall back to balloon center
        context_cx = (best_span.bbox_pdf[0] + best_span.bbox_pdf[2]) / 2 if best_span else balloon_cx
        context_cy = (best_span.bbox_pdf[1] + best_span.bbox_pdf[3]) / 2 if best_span else balloon_cy
        context_radius = 150  # Context radius in PDF points (~2 inches at 72 DPI)
        
        # Collect all spans within the context radius
        # These provide spatial fingerprints that help match the same region in Rev B
        local_context = []
        for span in page_spans:
            sx0, sy0, sx1, sy1 = span.bbox_pdf
            span_cx = (sx0 + sx1) / 2
            span_cy = (sy0 + sy1) / 2
            dist = math.sqrt((span_cx - context_cx)**2 + (span_cy - context_cy)**2)
            
            if dist <= context_radius:
                local_context.append(span)
        
        # Create the anchor linking Form 3 requirement to Rev A spatial location
        # This anchor serves as the foundation for finding the same characteristic in Rev B
        anchors.append(Anchor(
            char_no=char_no,
            page=page,
            balloon_bbox=balloon_bbox,
            req_bbox=req_bbox,
            requirement_raw=requirement,
            requirement_norm=req_norm.norm_text,
            local_context=local_context,
            zone_bbox=zone_bbox,
        ))
    
    return anchors
