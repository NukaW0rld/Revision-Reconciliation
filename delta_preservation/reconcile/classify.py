from typing import Optional, Dict, List
from dataclasses import dataclass

from delta_preservation.reconcile.anchors import Anchor
from delta_preservation.reconcile.match import Match
from delta_preservation.reconcile.normalize import parse_requirement


@dataclass
class DeltaItem:
    """Classification result for a single Rev A characteristic."""
    char_no: int
    status: str  # "unchanged", "changed", "removed", "uncertain"
    confidence: float
    reasons: List[str]
    component_scores: Dict[str, float]
    match: Optional[Match] = None


def classify_delta(
    anchor: Anchor,
    match_or_none: Optional[Match],
    location_search_coverage: float = 1.0
) -> DeltaItem:
    """Classify delta status for a single Rev A anchor.
    
    Args:
        anchor: Rev A anchor
        match_or_none: Assigned match or None if no candidate found
        location_search_coverage: Fraction of search window covered (for removed confidence)
        
    Returns:
        DeltaItem with status, confidence, reasons, and component scores
    """
    reasons = []
    
    # Handle removed case
    if match_or_none is None:
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
    
    # Parse tokens
    anchor_fingerprint = parse_requirement(anchor.requirement_norm)
    matched_fingerprint = parse_requirement(candidate.span.text)
    
    # Extract key tokens
    prefix_symbols = {'R', 'Ø', 'DRAWING', 'NOTES', 'EDGE', 'RADIUS', 
                      'COUNTERBORE', 'COUNTERSINK', 'LENGTH', 'DIAMETER', 
                      'DEPTH', 'ANGLE', 'THREAD'}
    
    anchor_prefix = set(anchor_fingerprint.symbol_tokens + anchor_fingerprint.type_tokens + 
                       anchor_fingerprint.count_tokens) & prefix_symbols
    matched_prefix = set(matched_fingerprint.symbol_tokens + matched_fingerprint.type_tokens + 
                        matched_fingerprint.count_tokens) & prefix_symbols
    
    anchor_numeric = set(val for val, _ in anchor_fingerprint.numeric_tokens)
    matched_numeric = set(val for val, _ in matched_fingerprint.numeric_tokens)
    
    # Apply classification rules
    prefix_match = anchor_prefix == matched_prefix
    numeric_match = anchor_numeric == matched_numeric
    
    if prefix_match and numeric_match:
        status = "unchanged"
        confidence = 0.5 * location_score + 0.5 * text_score
        reasons.append("All tokens match exactly")
    elif prefix_match and not numeric_match:
        status = "changed"
        confidence = 0.5 * location_score + 0.5 * text_score
        reasons.append("Text tokens match except numeric value")
    else:
        status = "uncertain"
        confidence = 0.5 * location_score + 0.5 * text_score
        reasons.append("Token mismatch or ambiguous match")
    
    # Add location-based reason
    if location_score > 0.7:
        reasons.append("High location agreement after global alignment")
    
    # Clip confidence
    confidence = max(0.0, min(1.0, confidence))
    
    return DeltaItem(
        char_no=anchor.char_no,
        status=status,
        confidence=confidence,
        reasons=reasons,
        component_scores={
            "location": location_score,
            "text": text_score,
            "context": context_score
        },
        match=match_or_none
    )
