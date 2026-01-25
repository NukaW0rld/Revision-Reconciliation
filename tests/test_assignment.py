"""
Tests for assign_matches greedy bipartite matching algorithm.

These tests verify the one-to-one assignment constraints:
- Each characteristic (char_no) assigned at most once
- Each Rev B span matched to at most one characteristic
- Greedy selection respects score ordering and prior assignments
"""

import pytest
from delta_preservation.reconcile.match import assign_matches, Match, Candidate
from delta_preservation.reconcile.anchors import Anchor
from delta_preservation.io.pdf import TextSpan


def make_span(text: str, bbox: tuple = (0, 0, 10, 10), block_id: int = 0, 
              line_id: int = 0, span_id: int = 0, font_size: float = 10.0) -> TextSpan:
    """Helper to create a TextSpan with unique identity."""
    return TextSpan(
        text=text,
        bbox_pdf=bbox,
        font_size=font_size,
        block_id=block_id,
        line_id=line_id,
        span_id=span_id
    )


def make_candidate(span: TextSpan, total_score: float) -> Candidate:
    """Helper to create a Candidate with specified score."""
    return Candidate(
        span=span,
        location_score=total_score * 0.5,
        text_score=total_score * 0.3,
        context_score=total_score * 0.2,
        total_score=total_score,
        reasons=[]
    )


def make_anchor(char_no: int, requirement: str = "TEST") -> Anchor:
    """Helper to create an Anchor."""
    return Anchor(
        char_no=char_no,
        page=0,
        balloon_bbox=(0, 0, 10, 10),
        requirement_raw=requirement,
        requirement_norm=requirement.upper(),
        req_bbox=None,
        local_context=[]
    )


class TestAssignMatches:
    
    def test_collision_one_to_one_constraint(self):
        """
        Test greedy assignment with forced collision:
        - Anchor 1 and Anchor 2 both have shared_span as top candidate
        - Each anchor also has a unique second-best candidate
        - Verify one-to-one constraints and greedy fallback
        """
        # Create spans with unique identities
        shared_span = make_span("SHARED", bbox=(0, 0, 10, 10), block_id=0, line_id=0, span_id=0)
        unique_span_1 = make_span("UNIQUE1", bbox=(10, 10, 20, 20), block_id=0, line_id=0, span_id=1)
        unique_span_2 = make_span("UNIQUE2", bbox=(20, 20, 30, 30), block_id=0, line_id=0, span_id=2)
        
        # Create anchors
        anchor_1 = make_anchor(char_no=1, requirement="REQ1")
        anchor_2 = make_anchor(char_no=2, requirement="REQ2")
        
        # Build candidates with collision on shared_span
        # Anchor 1: shared_span (score 0.9) > unique_span_1 (score 0.7)
        # Anchor 2: shared_span (score 0.8) > unique_span_2 (score 0.6)
        candidates_by_anchor = {
            1: [
                make_candidate(shared_span, total_score=0.9),
                make_candidate(unique_span_1, total_score=0.7)
            ],
            2: [
                make_candidate(shared_span, total_score=0.8),
                make_candidate(unique_span_2, total_score=0.6)
            ]
        }
        
        # Run assignment
        matches = assign_matches([anchor_1, anchor_2], candidates_by_anchor)
        
        # (1) No span_key used more than once
        used_span_keys = set()
        for match in matches.values():
            span = match.candidate.span
            span_key = (span.block_id, span.line_id, span.span_id, span.bbox_pdf)
            assert span_key not in used_span_keys, f"Span key {span_key} used multiple times"
            used_span_keys.add(span_key)
        
        # (2) Each char_no appears at most once
        assert len(matches) <= 2
        assert all(char_no in [1, 2] for char_no in matches.keys())
        
        # (3) Only one anchor gets shared_span, other gets unique or unassigned
        shared_span_key = (shared_span.block_id, shared_span.line_id, 
                          shared_span.span_id, shared_span.bbox_pdf)
        
        assignments = {
            char_no: (match.candidate.span.block_id, match.candidate.span.line_id,
                     match.candidate.span.span_id, match.candidate.span.bbox_pdf)
            for char_no, match in matches.items()
        }
        
        shared_span_count = sum(1 for key in assignments.values() if key == shared_span_key)
        assert shared_span_count == 1, "Shared span should be assigned to exactly one anchor"
        
        # (4) Greedy selection: anchor 1 should get shared_span (higher score 0.9)
        # anchor 2 should get unique_span_2 (fallback)
        assert 1 in matches, "Anchor 1 should be assigned"
        assert matches[1].candidate.span.text == "SHARED"
        assert matches[1].candidate.total_score == 0.9
        
        assert 2 in matches, "Anchor 2 should be assigned"
        assert matches[2].candidate.span.text == "UNIQUE2"
        assert matches[2].candidate.total_score == 0.6
    
    def test_multiple_collisions_greedy_order(self):
        """
        Test with three anchors competing for two spans:
        - Anchor 1: span_a (0.95), span_b (0.85)
        - Anchor 2: span_a (0.90), span_b (0.80)
        - Anchor 3: span_b (0.75)
        
        Expected: 1→span_a, 2→span_b, 3→unassigned
        """
        span_a = make_span("SPAN_A", block_id=0, line_id=0, span_id=0)
        span_b = make_span("SPAN_B", block_id=0, line_id=0, span_id=1)
        
        anchor_1 = make_anchor(char_no=1)
        anchor_2 = make_anchor(char_no=2)
        anchor_3 = make_anchor(char_no=3)
        
        candidates_by_anchor = {
            1: [
                make_candidate(span_a, total_score=0.95),
                make_candidate(span_b, total_score=0.85)
            ],
            2: [
                make_candidate(span_a, total_score=0.90),
                make_candidate(span_b, total_score=0.80)
            ],
            3: [
                make_candidate(span_b, total_score=0.75)
            ]
        }
        
        matches = assign_matches([anchor_1, anchor_2, anchor_3], candidates_by_anchor)
        
        # Verify one-to-one constraints
        assert len(matches) <= 3
        used_spans = set()
        for match in matches.values():
            span_key = (match.candidate.span.block_id, match.candidate.span.line_id,
                       match.candidate.span.span_id, match.candidate.span.bbox_pdf)
            assert span_key not in used_spans
            used_spans.add(span_key)
        
        # Verify greedy assignment order
        assert 1 in matches
        assert matches[1].candidate.span.text == "SPAN_A"
        
        assert 2 in matches
        assert matches[2].candidate.span.text == "SPAN_B"
        
        # Anchor 3 has no remaining candidates
        assert 3 not in matches
    
    def test_no_collisions_all_assigned(self):
        """Test case where each anchor has unique top candidate - all should be assigned."""
        span_1 = make_span("SPAN1", block_id=0, line_id=0, span_id=0)
        span_2 = make_span("SPAN2", block_id=0, line_id=0, span_id=1)
        
        anchor_1 = make_anchor(char_no=1)
        anchor_2 = make_anchor(char_no=2)
        
        candidates_by_anchor = {
            1: [make_candidate(span_1, total_score=0.9)],
            2: [make_candidate(span_2, total_score=0.8)]
        }
        
        matches = assign_matches([anchor_1, anchor_2], candidates_by_anchor)
        
        assert len(matches) == 2
        assert 1 in matches
        assert 2 in matches
        assert matches[1].candidate.span.text == "SPAN1"
        assert matches[2].candidate.span.text == "SPAN2"
    
    def test_empty_candidates(self):
        """Test anchors with no candidates remain unassigned."""
        anchor_1 = make_anchor(char_no=1)
        anchor_2 = make_anchor(char_no=2)
        
        candidates_by_anchor = {
            1: [],
            2: []
        }
        
        matches = assign_matches([anchor_1, anchor_2], candidates_by_anchor)
        
        assert len(matches) == 0
