"""
Reconciliation module for characteristic matching across drawing revisions.

This module implements the core logic for preserving and reconciling inspection 
characteristic identity across drawing revisions. It handles:

- Anchor building: Linking Form 3 requirements to Rev A PDF spatial locations
- Candidate matching: Finding potential Rev B matches for each Rev A characteristic  
- Classification: Determining whether characteristics are unchanged/changed/removed/added
- Text normalization: Parsing and comparing engineering requirement text

The reconciliation process maintains characteristic identity even when PDF layout 
changes, projections shift, or annotations move between revisions. This enables
incremental revision updates rather than complete re-FAIRs.
"""

from .anchors import build_revA_anchors, Anchor
from .match import generate_candidates, assign_matches, refine_match_display_text, Candidate, Match
from .classify import classify_delta, detect_added_characteristics, DeltaItem
from .normalize import parse_requirement
from .semantic_compare import compare_semantic_callouts, SemanticCompareResult

__all__ = [
    "build_revA_anchors",
    "Anchor", 
    "generate_candidates",
    "assign_matches",
    "refine_match_display_text",
    "Candidate",
    "Match",
    "classify_delta", 
    "detect_added_characteristics",
    "DeltaItem",
    "parse_requirement",
    "compare_semantic_callouts",
    "SemanticCompareResult",
]