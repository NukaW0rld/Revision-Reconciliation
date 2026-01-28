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


@dataclass
class Match:
    """A confirmed assignment of a Rev A characteristic to a Rev B span.
    
    Attributes:
        char_no: The characteristic number from Rev A
        candidate: The chosen candidate from Rev B
        pred_center_b: Predicted center in Rev B coordinates (x, y)
    """
    char_no: int
    candidate: Candidate
    pred_center_b: Optional[Tuple[float, float]] = None


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
    
    # (2) Text score - semantic matching using parsed fingerprints
    # This compares symbols (Ø, R), count patterns (2X, 4X), and numeric values
    anchor_fp = parse_requirement(anchor.requirement_raw)
    span_fp = parse_requirement(span.text)
    
    # Compare symbols (e.g., Ø for diameter, R for radius)
    anchor_symbols = set(anchor_fp.symbol_tokens)
    span_symbols = set(span_fp.symbol_tokens)
    symbol_match = len(anchor_symbols & span_symbols) / max(len(anchor_symbols), 1) if anchor_symbols else 0.0
    
    # Compare count patterns (e.g., 2X, 4X, 6X)
    anchor_counts = set(anchor_fp.count_tokens)
    span_counts = set(span_fp.count_tokens)
    # Count patterns present in both is a strong signal, even if values differ
    has_count_pattern = bool(anchor_counts and span_counts)
    count_exact_match = anchor_counts == span_counts if anchor_counts else True
    
    # Compare numeric values
    anchor_numerics = set(val for val, _ in anchor_fp.numeric_tokens)
    span_numerics = set(val for val, _ in span_fp.numeric_tokens)
    if anchor_numerics:
        numeric_overlap = len(anchor_numerics & span_numerics) / len(anchor_numerics)
    else:
        numeric_overlap = 1.0 if not span_numerics else 0.0
    
    # Compare pattern class (hole, dimension, note, etc.)
    class_match = 1.0 if anchor_fp.pattern_class == span_fp.pattern_class else 0.0
    
    # Weighted combination for text score
    # Symbols are most important (Ø indicates diameter requirement)
    # Count patterns indicate same type of requirement
    # Numeric overlap indicates matching values
    text_score = (
        0.4 * symbol_match +
        0.2 * (1.0 if has_count_pattern else 0.0) +
        0.2 * numeric_overlap +
        0.2 * class_match
    )
    
    matched_parts = []
    if symbol_match > 0:
        matched_parts.append(f"symbols: {anchor_symbols & span_symbols}")
    if has_count_pattern:
        matched_parts.append(f"counts: {anchor_counts} vs {span_counts}")
    if numeric_overlap > 0:
        matched_parts.append(f"numerics: {int(numeric_overlap*100)}%")
    if matched_parts:
        reasons.append(f"matched: {', '.join(matched_parts)}")
    
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


def assign_matches(
    anchors: List[Anchor],
    candidates_by_anchor: dict[int, List[Candidate]]
) -> dict[int, Match]:
    """Greedily assign anchors to Rev B spans using a global matching strategy.
    
    This function implements a greedy bipartite matching algorithm that ensures:
    1. Each characteristic (char_no) is assigned at most once
    2. Each Rev B span is matched to at most one characteristic
    
    The algorithm flattens all candidate edges, sorts by score, and greedily
    accepts edges that don't conflict with previous assignments.
    
    Args:
        anchors: List of Rev A anchors with characteristic numbers
        candidates_by_anchor: Dict mapping char_no to list of candidates
        
    Returns:
        Dict mapping assigned char_no to its Match object
    """
    # Build list of all edges: (total_score, char_no, span_key, candidate)
    edges = []
    for anchor in anchors:
        char_no = anchor.char_no
        candidates = candidates_by_anchor.get(char_no, [])
        
        for candidate in candidates:
            # Create unique span key using TextSpan attributes
            span = candidate.span
            span_key = (
                span.block_id,
                span.line_id,
                span.span_id,
                span.bbox_pdf
            )
            
            edges.append((
                candidate.total_score,
                char_no,
                span_key,
                candidate
            ))
    
    # Sort edges by total_score descending
    edges.sort(key=lambda e: e[0], reverse=True)
    
    # Greedy assignment
    assigned_chars = set()
    used_spans = set()
    matches = {}
    
    for total_score, char_no, span_key, candidate in edges:
        # Accept edge only if both char_no and span are available
        if char_no not in assigned_chars and span_key not in used_spans:
            matches[char_no] = Match(
                char_no=char_no,
                candidate=candidate,
                pred_center_b=None  # Can be populated by caller if needed
            )
            assigned_chars.add(char_no)
            used_spans.add(span_key)
    
    return matches
