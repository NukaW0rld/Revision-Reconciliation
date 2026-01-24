from typing import List, Dict, Optional
from dataclasses import dataclass
import math

from delta_preservation.io.pdf import TextSpan
from delta_preservation.vision.balloons import Balloon
from delta_preservation.reconcile.normalize import parse_requirement


@dataclass
class Anchor:
    """Stable anchor linking Form3 requirement to Rev A PDF location."""
    char_no: int
    page: int
    balloon_bbox: tuple[float, float, float, float]
    req_bbox: Optional[tuple[float, float, float, float]]
    requirement_raw: str
    requirement_norm: str
    local_context: List[TextSpan]


def build_revA_anchors(
    form3_chars: Dict[int, str],
    balloons: Dict[int, Balloon],
    revA_text_spans: List[TextSpan]
) -> List[Anchor]:
    """Build anchors linking Form3 requirements to Rev A PDF locations.
    
    Args:
        form3_chars: Mapping from char_no to requirement text
        balloons: Mapping from char_no to detected Balloon
        revA_text_spans: All text spans extracted from Rev A PDF
    
    Returns:
        List of Anchor objects with matched requirement locations
    """
    anchors = []
    
    for char_no, requirement in form3_chars.items():
        if char_no not in balloons:
            continue
        
        balloon = balloons[char_no]
        page = balloon.page_index
        balloon_bbox = balloon.bbox_pdf
        balloon_cx, balloon_cy = balloon.center_pdf
        
        # Filter spans to same page
        page_spans = [s for s in revA_text_spans if s.block_id == page]
        
        # Get page dimensions from balloon bbox (approximate)
        # Title block heuristic: exclude bottom 15% and right 20%
        page_height = max(s.bbox_pdf[3] for s in page_spans) if page_spans else 1000
        page_width = max(s.bbox_pdf[2] for s in page_spans) if page_spans else 1000
        
        # Filter candidates
        candidates = []
        for span in page_spans:
            x0, y0, x1, y1 = span.bbox_pdf
            
            # Skip title block regions
            if y0 > page_height * 0.85 or x0 > page_width * 0.80:
                continue
            
            # Skip very long spans (likely headers/titles)
            if len(span.text) > 100:
                continue
            
            # Skip empty or whitespace
            if not span.text.strip():
                continue
            
            candidates.append(span)
        
        # Normalize requirement
        req_norm = parse_requirement(requirement)
        req_tokens = set(req_norm.norm_text.split())
        
        # Score candidates
        best_span = None
        best_score = 0.2  # Threshold
        
        for span in candidates:
            span_norm = parse_requirement(span.text)
            span_tokens = set(span_norm.norm_text.split())
            
            # Token overlap score (primary)
            overlap = len(req_tokens & span_tokens)
            total = len(req_tokens | span_tokens)
            token_score = overlap / total if total > 0 else 0
            
            # Distance score (strong prior)
            sx0, sy0, sx1, sy1 = span.bbox_pdf
            span_cx = (sx0 + sx1) / 2
            span_cy = (sy0 + sy1) / 2
            distance = math.sqrt((span_cx - balloon_cx)**2 + (span_cy - balloon_cy)**2)
            dist_score = 1.0 / (1.0 + distance / 100)  # Normalize by typical spacing
            
            # Size penalty (small penalty for unusually large spans)
            span_area = (sx1 - sx0) * (sy1 - sy0)
            size_penalty = min(span_area / 5000, 0.3)  # Cap at 0.3
            
            # Weighted combination
            score = 0.5 * token_score + 0.4 * dist_score - 0.1 * size_penalty
            
            if score > best_score:
                best_score = score
                best_span = span
        
        # Set req_bbox
        req_bbox = best_span.bbox_pdf if best_span else None
        
        # Build local context
        context_cx = (best_span.bbox_pdf[0] + best_span.bbox_pdf[2]) / 2 if best_span else balloon_cx
        context_cy = (best_span.bbox_pdf[1] + best_span.bbox_pdf[3]) / 2 if best_span else balloon_cy
        context_radius = 150  # Fixed radius in PDF points
        
        local_context = []
        for span in page_spans:
            sx0, sy0, sx1, sy1 = span.bbox_pdf
            span_cx = (sx0 + sx1) / 2
            span_cy = (sy0 + sy1) / 2
            dist = math.sqrt((span_cx - context_cx)**2 + (span_cy - context_cy)**2)
            
            if dist <= context_radius:
                local_context.append(span)
        
        anchors.append(Anchor(
            char_no=char_no,
            page=page,
            balloon_bbox=balloon_bbox,
            req_bbox=req_bbox,
            requirement_raw=requirement,
            requirement_norm=req_norm.norm_text,
            local_context=local_context
        ))
    
    return anchors
