import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MatchFingerprint:
    """Deterministic fingerprint for matching requirements across revisions."""
    norm_text: str  # Uppercase, whitespace-collapsed
    count_tokens: List[str]  # e.g., ["2X", "6X"]
    type_tokens: List[str]  # e.g., ["LENGTH", "DIAMETER", "THREAD"]
    symbol_tokens: List[str]  # e.g., ["Ø", "R"]
    numeric_tokens: List[Tuple[float, str]]  # (parsed_value, original_string)
    units: Optional[str]  # "MM", "IN", "DEG", or None
    pattern_class: str  # "note", "hole", "fillet", "dimension", "other"


def parse_requirement(requirement: str) -> MatchFingerprint:
    """
    Parse a requirement string into a deterministic match fingerprint.
    
    Args:
        requirement: Raw requirement string from Form 3 data and PDF text spans
        
    Returns:
        MatchFingerprint with normalized text and extracted tokens
    """
    # Normalize: uppercase and collapse whitespace
    norm_text = " ".join(requirement.upper().split())
    
    # Extract count tokens (e.g., 2X, 6 X, 12 x)
    count_tokens = [f"{n}X" for n in re.findall(r'\b(\d+)\s*[Xx]\b', norm_text)]
    
    # Extract type/prefix tokens
    type_patterns = [
        "DRAWING NOTES", "EDGE RADIUS", "COUNTERBORE", "COUNTERSINK",
        "LENGTH", "DIAMETER", "RADIUS", "DEPTH", "ANGLE", "THREAD", "NOTES",
        "THRU ALL"
    ]
    type_tokens = []
    for pattern in type_patterns:
        if pattern in norm_text:
            type_tokens.append(pattern)
    
    # Extract symbol tokens
    symbol_tokens = []
    if "Ø" in requirement:
        symbol_tokens.append("Ø")
        
    # Detect radius symbol when used as R<number> or R.<number>
    if re.search(r'R(?=\d|\.)', norm_text):
        symbol_tokens.append("R")
    
    # Extract numeric tokens with tolerances
    numeric_tokens = []
    
    # General numeric values (without word boundary to catch Ø8, R2.5, etc.)
    numeric_matches = re.findall(r'\d+\.?\d*', norm_text)
    for match in numeric_matches:
        try:
            numeric_tokens.append((float(match), match))
        except ValueError:
            pass
    
    # Detect units
    units = None
    if re.search(r'\bMM\b', norm_text):
        units = "MM"
    elif re.search(r'\bIN\b', norm_text):
        units = "IN"
    elif re.search(r'\bDEG\b', norm_text):
        units = "DEG"
    
    # Assign pattern class
    pattern_class = "other"
    
    if "NOTES" in norm_text or "DRAWING NOTES" in norm_text:
        pattern_class = "note"
    elif "THREAD" in norm_text or re.search(r'[MG]\d+', norm_text):
        pattern_class = "hole"
    elif "RADIUS" in norm_text or "EDGE RADIUS" in norm_text or norm_text.strip().startswith("R "):
        pattern_class = "fillet"
    elif "DIAMETER" in norm_text or "Ø" in requirement or "COUNTERBORE" in norm_text or "COUNTERSINK" in norm_text:
        pattern_class = "hole"
    elif "THRU ALL" in norm_text:
        pattern_class = "hole"
    elif "LENGTH" in norm_text or "DEPTH" in norm_text or "ANGLE" in norm_text:
        pattern_class = "dimension"
    
    return MatchFingerprint(
        norm_text=norm_text,
        count_tokens=count_tokens,
        type_tokens=type_tokens,
        symbol_tokens=symbol_tokens,
        numeric_tokens=numeric_tokens,
        units=units,
        pattern_class=pattern_class
    )
