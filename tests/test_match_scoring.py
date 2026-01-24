"""
Tests for match.py scoring and candidate generation.

These tests use synthetic fixtures to deterministically verify scoring behavior
without requiring actual PDFs.
"""

import numpy as np
import pytest

from delta_preservation.reconcile.match import generate_candidates, score_candidate, Candidate
from delta_preservation.reconcile.anchors import Anchor
from delta_preservation.io.pdf import TextSpan
from delta_preservation.vision.alignment import Transform


def make_anchor(
    requirement_norm: str,
    req_bbox: tuple[float, float, float, float],
    local_context: list[TextSpan] = None
) -> Anchor:
    """
    Create a synthetic Anchor for testing.
    
    Args:
        requirement_norm: Normalized requirement text (e.g., "2X R 0.125")
        req_bbox: Requirement bounding box in PDF coordinates
        local_context: List of nearby text spans for context scoring
        
    Returns:
        Anchor object with minimal required fields
    """
    return Anchor(
        char_no=1,
        page=0,
        balloon_bbox=(0, 0, 10, 10),
        req_bbox=req_bbox,
        requirement_raw=requirement_norm.lower(),
        requirement_norm=requirement_norm,
        local_context=local_context or []
    )


def make_span(
    text: str,
    center: tuple[float, float],
    block_id: int = 0
) -> TextSpan:
    """
    Create a synthetic TextSpan for testing.
    
    Args:
        text: Text content
        center: (cx, cy) center coordinates in PDF space
        block_id: Block identifier (used for context grouping)
        
    Returns:
        TextSpan with bbox centered at the given coordinates
    """
    cx, cy = center
    # Create a small bbox around the center
    bbox = (cx - 5, cy - 5, cx + 5, cy + 5)
    return TextSpan(
        text=text,
        bbox_pdf=bbox,
        font_size=10.0,
        block_id=block_id,
        line_id=0,
        span_id=0
    )


def make_identity_transform() -> Transform:
    """
    Create an identity transform (no transformation).
    
    Returns:
        Transform with identity homography matrix
    """
    return Transform(
        H=np.eye(3, dtype=np.float32),
        inliers=100,
        inlier_ratio=0.9,
        quality_ok=True
    )


class TestCandidateScoring:
    """Test individual candidate scoring components."""
    
    def test_location_score_at_center(self):
        """Verify location score is 1.0 when span is at predicted center."""
        anchor = make_anchor("2X R 0.125", req_bbox=(0, 0, 10, 10))
        span = make_span("2X R 0.125", center=(5, 5))
        
        # Distance = 0, so location_score should be 1.0
        candidate = score_candidate(
            anchor=anchor,
            span=span,
            dist=0.0,
            radius=108.0,
            all_spans=[span],
            span_cx=5.0,
            span_cy=5.0
        )
        
        assert candidate.location_score == 1.0
        assert "within 0.0 mm" in candidate.reasons[0]
    
    def test_location_score_at_radius(self):
        """Verify location score is 0.0 when span is at search radius."""
        anchor = make_anchor("2X R 0.125", req_bbox=(0, 0, 10, 10))
        span = make_span("2X R 0.125", center=(113, 5))
        
        # Distance = 108, so location_score should be ~0.0
        candidate = score_candidate(
            anchor=anchor,
            span=span,
            dist=108.0,
            radius=108.0,
            all_spans=[span],
            span_cx=113.0,
            span_cy=5.0
        )
        
        assert candidate.location_score == pytest.approx(0.0, abs=1e-6)
    
    def test_text_score_exact_match(self):
        """Verify text score with exact token match."""
        anchor = make_anchor("2X R 0.125", req_bbox=(0, 0, 10, 10))
        span = make_span("2X R 0.125", center=(5, 5))
        
        candidate = score_candidate(
            anchor=anchor,
            span=span,
            dist=0.0,
            radius=108.0,
            all_spans=[span],
            span_cx=5.0,
            span_cy=5.0
        )
        
        # Should have high text score with all tokens matched
        assert candidate.text_score > 0.8
        assert "matched tokens: 0.125, 2X, R" in candidate.reasons[1]
    
    def test_text_score_prefix_only(self):
        """Verify text score when only prefix tokens match (no numeric match)."""
        anchor = make_anchor("2X R 0.125", req_bbox=(0, 0, 10, 10))
        span = make_span("2X R 0.250", center=(5, 5))  # Different numeric value
        
        candidate = score_candidate(
            anchor=anchor,
            span=span,
            dist=0.0,
            radius=108.0,
            all_spans=[span],
            span_cx=5.0,
            span_cy=5.0
        )
        
        # Should have non-zero text score (prefix tokens match)
        # This proves "changed dimension" behavior
        assert candidate.text_score > 0.0
        assert candidate.text_score < 1.0
        assert "matched tokens: 2X, R" in candidate.reasons[1]
    
    def test_text_score_no_match(self):
        """Verify text score is 0.0 when no tokens match."""
        anchor = make_anchor("2X R 0.125", req_bbox=(0, 0, 10, 10))
        span = make_span("DRAWING NOTES", center=(5, 5))
        
        candidate = score_candidate(
            anchor=anchor,
            span=span,
            dist=0.0,
            radius=108.0,
            all_spans=[span],
            span_cx=5.0,
            span_cy=5.0
        )
        
        assert candidate.text_score == 0.0
        # No "matched tokens" in reasons
        assert len(candidate.reasons) == 1  # Only location reason
    
    def test_context_score_with_matching_context(self):
        """Verify context score when nearby spans share tokens."""
        # Create anchor with local context
        context_spans = [
            make_span("HOLE DEPTH 10", center=(20, 20)),
            make_span("THREAD M8", center=(25, 25))
        ]
        anchor = make_anchor("2X R 0.125", req_bbox=(0, 0, 10, 10), local_context=context_spans)
        
        # Create candidate span with similar nearby spans
        candidate_span = make_span("2X R 0.125", center=(5, 5), block_id=0)
        nearby_spans = [
            candidate_span,
            make_span("HOLE DEPTH 10", center=(10, 10), block_id=0),  # Matches anchor context
            make_span("THREAD M8", center=(15, 15), block_id=0)  # Matches anchor context
        ]
        
        candidate = score_candidate(
            anchor=anchor,
            span=candidate_span,
            dist=0.0,
            radius=108.0,
            all_spans=nearby_spans,
            span_cx=5.0,
            span_cy=5.0
        )
        
        # Should have non-zero context score
        assert candidate.context_score > 0.0
        assert any("context similarity" in r for r in candidate.reasons)
    
    def test_context_score_no_match(self):
        """Verify context score is 0.0 when nearby spans don't match."""
        # Create anchor with local context
        context_spans = [
            make_span("HOLE DEPTH 10", center=(20, 20)),
        ]
        anchor = make_anchor("2X R 0.125", req_bbox=(0, 0, 10, 10), local_context=context_spans)
        
        # Create candidate span with different nearby spans
        candidate_span = make_span("2X R 0.125", center=(5, 5), block_id=0)
        nearby_spans = [
            candidate_span,
            make_span("DRAWING NOTES", center=(10, 10), block_id=0),  # No match
        ]
        
        candidate = score_candidate(
            anchor=anchor,
            span=candidate_span,
            dist=0.0,
            radius=108.0,
            all_spans=nearby_spans,
            span_cx=5.0,
            span_cy=5.0
        )
        
        # Context score should be low or zero
        assert candidate.context_score < 0.5


class TestGenerateCandidates:
    """Test candidate generation and ranking."""
    
    def test_radius_filtering(self):
        """Verify only spans within search radius are returned."""
        anchor = make_anchor("2X R 0.125", req_bbox=(0, 0, 10, 10))
        transform = make_identity_transform()
        
        # Create spans at various distances from predicted center (5, 5)
        spans = [
            make_span("CLOSE", center=(5, 5)),      # dist = 0
            make_span("NEAR", center=(50, 5)),      # dist = 45
            make_span("FAR", center=(200, 5)),      # dist = 195 > 108 (radius)
            make_span("VERY FAR", center=(500, 5))  # dist = 495 > 108 (radius)
        ]
        
        candidates = generate_candidates(anchor, spans, transform, top_k=10)
        
        # Only first two spans should be within radius
        assert len(candidates) == 2
        assert candidates[0].span.text in ["CLOSE", "NEAR"]
        assert candidates[1].span.text in ["CLOSE", "NEAR"]
    
    def test_sorting_by_total_score(self):
        """Verify candidates are sorted by total score descending."""
        anchor = make_anchor("2X R 0.125", req_bbox=(0, 0, 10, 10))
        transform = make_identity_transform()
        
        # Create spans with different scoring characteristics
        spans = [
            # (a) Very close + matching prefix + different numeric (should rank #1)
            make_span("2X R 0.250", center=(5, 5)),
            
            # (b) Very close + no shared tokens (location only, should rank below (a))
            make_span("DRAWING NOTES", center=(6, 6)),
            
            # (c) Far but strong text overlap (should rank below close matches)
            make_span("2X R 0.125", center=(100, 5)),
            
            # (d) Both far and text-mismatched (should rank last)
            make_span("UNRELATED", center=(105, 5))
        ]
        
        candidates = generate_candidates(anchor, spans, transform, top_k=10)
        
        # Verify strict descending order
        for i in range(len(candidates) - 1):
            assert candidates[i].total_score >= candidates[i + 1].total_score
        
        # Verify expected ranking
        assert candidates[0].span.text == "2X R 0.250"  # Close + prefix match
        assert candidates[-1].span.text == "UNRELATED"  # Far + no match
    
    def test_top_candidate_has_expected_tokens(self):
        """Verify top candidate contains expected matched tokens in reasons."""
        anchor = make_anchor("2X R 0.125", req_bbox=(0, 0, 10, 10))
        transform = make_identity_transform()
        
        spans = [
            make_span("2X R 0.250", center=(5, 5)),  # Close + prefix match
            make_span("OTHER TEXT", center=(10, 10))
        ]
        
        candidates = generate_candidates(anchor, spans, transform, top_k=10)
        
        # Top candidate should have matched tokens in reasons
        assert len(candidates) > 0
        top_candidate = candidates[0]
        assert any("matched tokens" in r for r in top_candidate.reasons)
        assert any("2X" in r and "R" in r for r in top_candidate.reasons)
    
    def test_numeric_difference_non_zero_score(self):
        """Verify spans with different numeric values still get non-zero text score."""
        anchor = make_anchor("2X R 0.125", req_bbox=(0, 0, 10, 10))
        transform = make_identity_transform()
        
        # Same prefix tokens, different numeric value
        spans = [make_span("2X R 0.250", center=(5, 5))]
        
        candidates = generate_candidates(anchor, spans, transform, top_k=10)
        
        assert len(candidates) == 1
        # Text score should be non-zero (prefix tokens match)
        assert candidates[0].text_score > 0.0
        # But not perfect (numeric doesn't match)
        assert candidates[0].text_score < 1.0
    
    def test_context_score_influence_on_ranking(self):
        """Verify context score influences ranking when two spans compete spatially."""
        # Create anchor with context
        context_spans = [
            make_span("HOLE DEPTH 10", center=(20, 20)),
        ]
        anchor = make_anchor("2X R 0.125", req_bbox=(0, 0, 10, 10), local_context=context_spans)
        transform = make_identity_transform()
        
        # Two spans at similar distance, but only one has matching context
        span_with_context = make_span("2X R 0.125", center=(5, 5), block_id=0)
        span_without_context = make_span("2X R 0.125", center=(6, 6), block_id=1)
        
        # Add nearby spans for context scoring (place far enough to not interfere)
        all_spans = [
            span_with_context,
            span_without_context,
            make_span("HOLE DEPTH 10", center=(10, 10), block_id=0),  # Near span_with_context
        ]
        
        candidates = generate_candidates(anchor, all_spans, transform, top_k=10)
        
        # All spans within radius should be present (including context span)
        assert len(candidates) >= 2
        
        # Find the two main candidates (with text "2X R 0.125")
        main_candidates = [c for c in candidates if c.span.text == "2X R 0.125"]
        assert len(main_candidates) == 2
        
        candidate_with_ctx = next(c for c in main_candidates if c.span.block_id == 0)
        candidate_without_ctx = next(c for c in main_candidates if c.span.block_id == 1)
        
        # Candidate with matching context should have higher context score
        assert candidate_with_ctx.context_score > candidate_without_ctx.context_score
    
    def test_top_k_limit(self):
        """Verify top_k parameter limits number of returned candidates."""
        anchor = make_anchor("2X R 0.125", req_bbox=(0, 0, 10, 10))
        transform = make_identity_transform()
        
        # Create many spans within radius
        spans = [make_span(f"SPAN_{i}", center=(5 + i, 5)) for i in range(20)]
        
        candidates = generate_candidates(anchor, spans, transform, top_k=5)
        
        assert len(candidates) == 5
    
    def test_comprehensive_ranking_scenario(self):
        """
        Comprehensive test with all scoring dimensions:
        (a) Close + prefix match + different numeric (rank #1)
        (b) Close + no tokens (rank #2-3)
        (c) Far + strong text overlap (rank #2-3)
        (d) Far + no match (filtered or rank last)
        (e) Two competing spans, one with context (context influences rank)
        """
        # Create anchor with context
        context_spans = [
            make_span("THREAD M8", center=(20, 20)),
        ]
        anchor = make_anchor("2X R 0.125", req_bbox=(0, 0, 10, 10), local_context=context_spans)
        transform = make_identity_transform()
        
        # Create candidate spans
        span_a = make_span("2X R 0.250", center=(5, 5), block_id=0)  # Close + prefix
        span_b = make_span("DRAWING NOTES", center=(6, 6), block_id=1)  # Close + no tokens
        span_c = make_span("2X R 0.125", center=(100, 5), block_id=2)  # Far + exact match
        span_d = make_span("UNRELATED", center=(107, 5), block_id=3)  # Far + no match
        span_e1 = make_span("2X R 0.125", center=(7, 7), block_id=4)  # Competing with e2
        span_e2 = make_span("2X R 0.125", center=(8, 8), block_id=5)  # Competing with e1
        
        # Add context near e2
        all_spans = [
            span_a, span_b, span_c, span_d, span_e1, span_e2,
            make_span("THREAD M8", center=(10, 10), block_id=5),  # Near e2, matches anchor context
        ]
        
        candidates = generate_candidates(anchor, all_spans, transform, top_k=10)
        
        # Verify all within radius are returned (including context span)
        assert len(candidates) >= 6
        
        # Verify strict descending order
        for i in range(len(candidates) - 1):
            assert candidates[i].total_score >= candidates[i + 1].total_score
        
        # Find span_a (should rank high due to close + prefix match)
        span_a_candidate = next(c for c in candidates if c.span.block_id == 0)
        # Should have non-zero text score despite numeric difference
        assert span_a_candidate.text_score > 0.0
        assert "2X" in str(span_a_candidate.reasons)
        assert "R" in str(span_a_candidate.reasons)
        
        # Verify context influence on e1 vs e2
        e1_candidate = next(c for c in candidates if c.span.block_id == 4)
        e2_candidate = next(c for c in candidates if c.span.block_id == 5)
        assert e2_candidate.context_score > e1_candidate.context_score
        # e2 should rank higher than e1 due to context
        e1_idx = candidates.index(e1_candidate)
        e2_idx = candidates.index(e2_candidate)
        assert e2_idx < e1_idx
        
        # Far + no match should rank last (if present)
        if any(c.span.text == "UNRELATED" for c in candidates):
            assert candidates[-1].span.text == "UNRELATED"
