from typing import Optional, Dict, List, Set, Tuple
from dataclasses import dataclass

from delta_preservation.reconcile.anchors import Anchor
from delta_preservation.reconcile.match import Match
from delta_preservation.reconcile.normalize import parse_requirement
from delta_preservation.io.pdf import TextSpan


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


@dataclass
class AddedCharacteristic:
    """Represents a newly detected characteristic in Rev B that doesn't exist in Rev A."""
    span: TextSpan
    requirement_text: str
    confidence: float
    reasons: List[str]


def classify_delta(
    anchor: Anchor,
    match_or_none: Optional[Match],
    location_search_coverage: float = 1.0
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
    
    # Parse fingerprints from both anchor requirement and matched span
    anchor_fp = parse_requirement(anchor.requirement_raw)
    matched_fp = parse_requirement(candidate.span.text)
    
    # Extract numeric values (the key comparison for dimensions)
    anchor_numerics = set(val for val, _ in anchor_fp.numeric_tokens)
    matched_numerics = set(val for val, _ in matched_fp.numeric_tokens)
    
    # Extract structural tokens (count patterns like "2X", "6X" and symbols like "Ø", "R")
    anchor_count = set(anchor_fp.count_tokens)
    matched_count = set(matched_fp.count_tokens)
    
    anchor_symbols = set(anchor_fp.symbol_tokens)
    matched_symbols = set(matched_fp.symbol_tokens)
    
    # Compute overlap metrics
    # For numerics: check if anchor's key values appear in matched span
    # (matched span may have additional values, that's OK)
    if anchor_numerics:
        numeric_overlap = len(anchor_numerics & matched_numerics) / len(anchor_numerics)
    else:
        numeric_overlap = 1.0 if not matched_numerics else 0.0
    
    # For counts: distinguish between count change vs count missing from matched span
    # "2X" → "4X" is a real change, but "2X" → set() is just incomplete span matching
    count_match = anchor_count == matched_count or (not anchor_count and not matched_count)
    count_changed = bool(anchor_count and matched_count and anchor_count != matched_count)
    count_missing = bool(anchor_count and not matched_count)  # Count present in anchor but not in span
    
    # For symbols: check if anchor symbols appear in matched span
    symbol_match = anchor_symbols <= matched_symbols or not anchor_symbols
    
    # Classification decision tree
    if count_changed:
        # Count explicitly changed (e.g., "2 x Ø8" → "4 x Ø8") → changed
        status = "changed"
        confidence = 0.5 * location_score + 0.3 * numeric_overlap + 0.2
        reasons.append(f"Count changed: {anchor_count} → {matched_count}")
    elif numeric_overlap >= 0.5 and symbol_match:
        # Good numeric overlap with matching symbols → unchanged
        # (count_missing is OK - just means span doesn't include count prefix)
        status = "unchanged"
        confidence = 0.4 * location_score + 0.4 * numeric_overlap + 0.2
        reasons.append(f"Numeric values match ({int(numeric_overlap*100)}% overlap)")
        if count_match and anchor_count:
            reasons.append(f"Count tokens match: {anchor_count}")
    elif numeric_overlap < 0.5 and symbol_match and not count_missing:
        # Same type (symbol) but different numeric values → changed
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
        match=match_or_none
    )


def detect_added_characteristics(
    revB_spans: List[TextSpan],
    matches: Dict[int, Match],
    next_char_no: int
) -> List[DeltaItem]:
    """Detect new characteristics in Rev B that don't exist in Rev A.
    
    Looks for unmatched Rev B text spans that appear to be dimension requirements
    (have numeric tokens and symbols/count patterns).
    
    Args:
        revB_spans: All text spans from Rev B PDF
        matches: Dict of char_no to Match objects (matched spans)
        next_char_no: Starting char_no for added items (typically max_existing + 1)
        
    Returns:
        List of DeltaItems with status="added" for detected new characteristics
    """
    # Build set of matched span keys
    matched_span_keys: Set[Tuple] = set()
    for match in matches.values():
        span = match.candidate.span
        key = (span.block_id, span.line_id, span.span_id, span.bbox_pdf)
        matched_span_keys.add(key)
    
    added_items: List[DeltaItem] = []
    current_char_no = next_char_no
    
    for span in revB_spans:
        # Skip already matched spans
        key = (span.block_id, span.line_id, span.span_id, span.bbox_pdf)
        if key in matched_span_keys:
            continue
        
        text = span.text.strip()
        if len(text) < 3:
            continue
        
        # Parse the span to check if it looks like a dimension requirement
        fp = parse_requirement(text)
        
        # Filter for spans that look like dimension requirements:
        # - Must have numeric tokens (dimension values)
        # - Must have symbols (Ø, R) OR count patterns (2X, 4X)
        # - Skip spans that look like revision notes or title block text
        if not fp.numeric_tokens:
            continue
        
        if not (fp.symbol_tokens or fp.count_tokens):
            continue
        
        # Skip spans that look like revision notes (contain "ADDED", "NEW", etc.)
        if any(word in fp.norm_text for word in ["ADDED", "NEW", "REVISED", "DELETED"]):
            continue
        
        # Skip single numeric values (likely just dimension labels)
        if len(fp.numeric_tokens) == 1 and not fp.symbol_tokens and not fp.count_tokens:
            continue
        
        # This looks like a new characteristic
        reasons = [
            f"New requirement detected in Rev B: \"{text}\"",
            f"Symbols: {fp.symbol_tokens}, Counts: {fp.count_tokens}"
        ]
        
        # Calculate confidence based on how "dimension-like" the span is
        confidence = 0.6
        if fp.symbol_tokens:
            confidence += 0.15
        if fp.count_tokens:
            confidence += 0.15
        if fp.pattern_class in ["hole", "dimension", "fillet"]:
            confidence += 0.1
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
            added_span=span
        )
        added_items.append(added_item)
        current_char_no += 1
    
    return added_items
