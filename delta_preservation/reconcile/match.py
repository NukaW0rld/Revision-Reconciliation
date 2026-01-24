from typing import List, Tuple, Optional
from dataclasses import dataclass
import math
import numpy as np
import cv2

from delta_preservation.reconcile.anchors import Anchor
from delta_preservation.io.pdf import TextSpan
from delta_preservation.reconcile.normalize import parse_requirement
from delta_preservation.vision.alignment import Transform


@dataclass
class Candidate:
    """A potential Rev B match for a Rev A requirement.
    
    Attributes:
        span: The Rev B text span
        total_score: Combined weighted score
        location_score: Distance-based score (0-1)
        text_score: Token overlap score (0-1)
        context_score: Neighboring span similarity (0-1)
        reasons: Human-readable explanation of score components
    """
    span: TextSpan
    total_score: float
    location_score: float
    text_score: float
    context_score: float
    reasons: List[str]


def generate_candidates(
    anchor: Anchor,
    revB_spans: List[TextSpan],
    transform: Transform,
    top_k: int = 5
) -> List[Candidate]:
    """Generate ranked candidate matches for a Rev A anchor in Rev B.
    
    Args:
        anchor: Rev A anchor with requirement and bbox
        revB_spans: All text spans from Rev B PDF
        transform: Homography from Rev A to Rev B
        top_k: Maximum number of candidates to return
        
    Returns:
        List of top-K candidates sorted by score descending
    """
    # Fixed radius (~1.5 inches = 108 points)
    SEARCH_RADIUS = 108.0
    
    # Get predicted Rev B center
    if anchor.req_bbox is not None:
        x0, y0, x1, y1 = anchor.req_bbox
        center_a = np.array([[(x0 + x1) / 2, (y0 + y1) / 2]], dtype=np.float32)
    else:
        # Fall back to balloon center
        center_a = np.array([[(anchor.balloon_bbox[0] + anchor.balloon_bbox[2]) / 2,
                             (anchor.balloon_bbox[1] + anchor.balloon_bbox[3]) / 2]], dtype=np.float32)
    
    center_a = center_a.reshape(-1, 1, 2)
    center_b = cv2.perspectiveTransform(center_a, transform.H)[0][0]
    pred_x, pred_y = center_b
    
    # Filter spans within radius
    candidate_pool = []
    for span in revB_spans:
        sx0, sy0, sx1, sy1 = span.bbox_pdf
        span_cx = (sx0 + sx1) / 2
        span_cy = (sy0 + sy1) / 2
        dist = math.sqrt((span_cx - pred_x)**2 + (span_cy - pred_y)**2)
        
        if dist <= SEARCH_RADIUS:
            candidate_pool.append((span, dist, span_cx, span_cy))
    
    # Score each candidate
    candidates = []
    for span, dist, cx, cy in candidate_pool:
        candidate = score_candidate(
            anchor, span, dist, SEARCH_RADIUS,
            revB_spans, cx, cy
        )
        candidates.append(candidate)
    
    # Sort by total score descending and return top K
    candidates.sort(key=lambda c: c.total_score, reverse=True)
    return candidates[:top_k]


def score_candidate(
    anchor: Anchor,
    span: TextSpan,
    dist: float,
    radius: float,
    all_spans: List[TextSpan],
    span_cx: float,
    span_cy: float
) -> Candidate:
    """Compute multi-component score for a candidate span.
    
    Args:
        anchor: Rev A anchor
        span: Candidate Rev B span
        dist: Distance from predicted center
        radius: Search radius
        all_spans: All Rev B spans for context
        span_cx: Span center x
        span_cy: Span center y
        
    Returns:
        Candidate with component scores and reasons
    """
    reasons = []
    
    # (1) Location score
    location_score = 1.0 - min(dist / radius, 1.0)
    if location_score > 0:
        dist_mm = dist * 25.4 / 72.0  # Convert points to mm
        reasons.append(f"within {dist_mm:.1f} mm of predicted location")
    
    # (2) Text score - weighted token overlap
    anchor_tokens = set(anchor.requirement_norm.split())
    span_parsed = parse_requirement(span.text)
    span_tokens = set(span_parsed.norm_text.split())
    
    # Categorize tokens
    prefix_symbols = {'R', 'Ø', 'DRAWING', 'NOTES', 'EDGE', 'RADIUS', 'COUNTERBORE', 
                      'COUNTERSINK', 'LENGTH', 'DIAMETER', 'DEPTH', 'ANGLE', 'THREAD'}
    count_patterns = {t for t in anchor_tokens | span_tokens if t.endswith('X') and t[:-1].isdigit()}
    
    # Weight tokens
    matched_tokens = anchor_tokens & span_tokens
    prefix_matches = matched_tokens & (prefix_symbols | count_patterns)
    numeric_matches = matched_tokens - prefix_matches
    
    # Weighted scoring
    prefix_weight = 1.0
    numeric_weight = 0.5
    
    total_weight = len(prefix_matches) * prefix_weight + len(numeric_matches) * numeric_weight
    max_weight = len(anchor_tokens) * prefix_weight  # Conservative denominator
    text_score = min(total_weight / max_weight, 1.0) if max_weight > 0 else 0.0
    
    if matched_tokens:
        reasons.append(f"matched tokens: {', '.join(sorted(matched_tokens))}")
    
    # (3) Context score - Jaccard of neighboring spans
    CONTEXT_WINDOW = 50.0  # Points
    
    # Get anchor context tokens
    anchor_context_tokens = set()
    for ctx_span in anchor.local_context:
        ctx_parsed = parse_requirement(ctx_span.text)
        anchor_context_tokens.update(ctx_parsed.norm_text.split())
    
    # Get candidate context tokens
    candidate_context_tokens = set()
    for other_span in all_spans:
        if other_span.block_id != span.block_id:
            continue
        ox0, oy0, ox1, oy1 = other_span.bbox_pdf
        other_cx = (ox0 + ox1) / 2
        other_cy = (oy0 + oy1) / 2
        other_dist = math.sqrt((other_cx - span_cx)**2 + (other_cy - span_cy)**2)
        
        if other_dist <= CONTEXT_WINDOW and other_dist > 0:
            other_parsed = parse_requirement(other_span.text)
            candidate_context_tokens.update(other_parsed.norm_text.split())
    
    # Jaccard similarity
    if anchor_context_tokens or candidate_context_tokens:
        intersection = len(anchor_context_tokens & candidate_context_tokens)
        union = len(anchor_context_tokens | candidate_context_tokens)
        context_score = intersection / union if union > 0 else 0.0
        if context_score > 0:
            reasons.append(f"context similarity: {context_score:.2f}")
    else:
        context_score = 0.0
    
    # Combine scores
    LOCATION_WEIGHT = 0.5
    TEXT_WEIGHT = 0.35
    CONTEXT_WEIGHT = 0.15
    
    total_score = (
        LOCATION_WEIGHT * location_score +
        TEXT_WEIGHT * text_score +
        CONTEXT_WEIGHT * context_score
    )
    
    return Candidate(
        span=span,
        total_score=total_score,
        location_score=location_score,
        text_score=text_score,
        context_score=context_score,
        reasons=reasons
    )
