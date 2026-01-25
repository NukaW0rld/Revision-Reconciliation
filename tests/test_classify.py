"""
Unit tests for delta_preservation.reconcile.classify module.

These tests feed the classifier small, fully controlled anchor/match inputs
(no PDFs) and assert only the rule outcomes, not the upstream matching.
"""

import pytest
from delta_preservation.reconcile.classify import classify_delta, DeltaItem
from delta_preservation.reconcile.anchors import Anchor
from delta_preservation.reconcile.match import Match, Candidate
from delta_preservation.io.pdf import TextSpan


# Test fixtures for creating controlled test data

def make_anchor(
    char_no: int = 1,
    requirement_norm: str = "Ø 8.0 +0.1 -0.0"
) -> Anchor:
    """Create a minimal Anchor for testing."""
    return Anchor(
        char_no=char_no,
        page=0,
        balloon_bbox=(100.0, 100.0, 120.0, 120.0),
        req_bbox=(150.0, 100.0, 200.0, 110.0),
        requirement_raw=requirement_norm,
        requirement_norm=requirement_norm,
        local_context=[]
    )


def make_text_span(
    text: str,
    bbox: tuple = (150.0, 100.0, 200.0, 110.0)
) -> TextSpan:
    """Create a minimal TextSpan for testing."""
    return TextSpan(
        text=text,
        bbox_pdf=bbox,
        font_size=10.0,
        block_id=0,
        line_id=0,
        span_id=0
    )


def make_candidate(
    text: str,
    location_score: float = 0.8,
    text_score: float = 0.9,
    context_score: float = 0.5
) -> Candidate:
    """Create a Candidate with controlled scores."""
    span = make_text_span(text)
    total_score = 0.5 * location_score + 0.35 * text_score + 0.15 * context_score
    
    return Candidate(
        span=span,
        total_score=total_score,
        location_score=location_score,
        text_score=text_score,
        context_score=context_score,
        reasons=["test candidate"]
    )


def make_match(
    char_no: int,
    text: str,
    location_score: float = 0.8,
    text_score: float = 0.9,
    context_score: float = 0.5
) -> Match:
    """Create a Match with controlled candidate scores."""
    candidate = make_candidate(text, location_score, text_score, context_score)
    return Match(
        char_no=char_no,
        candidate=candidate,
        pred_center_b=(150.0, 105.0)
    )


# Test 1: Removed status when match_or_none=None

def test_removed_status_no_candidate():
    """
    Test removed status: pass match_or_none=None and assert status=="removed",
    confidence is high (≈0.9), and reasons contains a "no candidate" explanation.
    """
    anchor = make_anchor(char_no=1, requirement_norm="Ø 8.0 +0.1 -0.0")
    
    result = classify_delta(anchor, match_or_none=None, location_search_coverage=1.0)
    
    assert isinstance(result, DeltaItem)
    assert result.char_no == 1
    assert result.status == "removed"
    assert result.confidence == pytest.approx(0.9, abs=0.01)
    assert result.match is None
    assert any("no candidate" in reason.lower() for reason in result.reasons)
    assert result.component_scores["location"] == 0.0
    assert result.component_scores["text"] == 0.0
    assert result.component_scores["context"] == 0.0


def test_removed_status_partial_coverage():
    """Test removed status with partial search coverage affects confidence."""
    anchor = make_anchor(char_no=2)
    
    result = classify_delta(anchor, match_or_none=None, location_search_coverage=0.7)
    
    assert result.status == "removed"
    assert result.confidence == pytest.approx(0.7, abs=0.01)
    assert result.match is None


# Test 2: Unchanged status with identical tokens

def test_unchanged_status_identical_tokens():
    """
    Test unchanged status: give identical normalized numeric tokens and identical
    prefix tokens with a high location_score and assert status=="unchanged",
    confidence > changed confidence, and a reason indicating strong location/text agreement.
    """
    # Create anchor with specific requirement
    anchor = make_anchor(char_no=3, requirement_norm="Ø 8.0 +0.1 -0.0")
    
    # Create match with identical text and high scores
    match = make_match(
        char_no=3,
        text="Ø 8.0 +0.1 -0.0",
        location_score=0.85,
        text_score=0.95,
        context_score=0.6
    )
    
    result = classify_delta(anchor, match_or_none=match)
    
    assert result.status == "unchanged"
    assert result.confidence > 0.8  # High confidence for unchanged
    assert any("all tokens match" in reason.lower() for reason in result.reasons)
    assert any("location agreement" in reason.lower() for reason in result.reasons)
    assert result.match == match


def test_unchanged_with_radius_symbol():
    """Test unchanged status with radius symbol (R) prefix."""
    anchor = make_anchor(char_no=4, requirement_norm="R 2.5")
    match = make_match(char_no=4, text="R 2.5", location_score=0.9, text_score=0.95)
    
    result = classify_delta(anchor, match_or_none=match)
    
    assert result.status == "unchanged"
    assert result.confidence > 0.85


# Test 3: Changed status with differing numeric tokens

def test_changed_status_numeric_difference():
    """
    Test changed status: give matching prefix tokens but at least one differing
    numeric token and assert status=="changed", confidence slightly lower than
    unchanged, and a reason mentioning numeric difference.
    """
    # Same prefix (Ø) but different numeric value
    anchor = make_anchor(char_no=5, requirement_norm="Ø 8.0 +0.1 -0.0")
    match = make_match(
        char_no=5,
        text="Ø 8.5 +0.1 -0.0",  # Changed from 8.0 to 8.5
        location_score=0.8,
        text_score=0.85,
        context_score=0.5
    )
    
    result = classify_delta(anchor, match_or_none=match)
    
    assert result.status == "changed"
    # Confidence formula: 0.5 * location + 0.5 * text
    expected_confidence = 0.5 * 0.8 + 0.5 * 0.85
    assert result.confidence == pytest.approx(expected_confidence, abs=0.01)
    assert any("numeric" in reason.lower() for reason in result.reasons)


def test_changed_status_tolerance_change():
    """Test changed status when tolerance values change."""
    anchor = make_anchor(char_no=6, requirement_norm="Ø 10.0 +0.2 -0.1")
    match = make_match(
        char_no=6,
        text="Ø 10.0 +0.3 -0.1",  # Tolerance changed
        location_score=0.75,
        text_score=0.8
    )
    
    result = classify_delta(anchor, match_or_none=match)
    
    assert result.status == "changed"
    assert any("numeric" in reason.lower() for reason in result.reasons)


# Test 4: Uncertain status with weak text similarity

def test_uncertain_status_weak_text_high_location():
    """
    Test uncertain status: give weak text similarity but high location score
    and assert status=="uncertain" with mid-range confidence and a reason
    indicating ambiguity.
    """
    # Different prefix and different numeric values
    anchor = make_anchor(char_no=7, requirement_norm="Ø 8.0")
    match = make_match(
        char_no=7,
        text="R 12.5",  # Completely different requirement type
        location_score=0.9,  # High location score
        text_score=0.3,      # Low text score
        context_score=0.4
    )
    
    result = classify_delta(anchor, match_or_none=match)
    
    assert result.status == "uncertain"
    # Confidence: 0.5 * 0.9 + 0.5 * 0.3 = 0.6
    expected_confidence = 0.5 * 0.9 + 0.5 * 0.3
    assert result.confidence == pytest.approx(expected_confidence, abs=0.01)
    assert any("mismatch" in reason.lower() or "ambiguous" in reason.lower() 
               for reason in result.reasons)


def test_uncertain_status_prefix_mismatch():
    """Test uncertain status when prefix symbols don't match."""
    anchor = make_anchor(char_no=8, requirement_norm="DRAWING NOTES")
    match = make_match(
        char_no=8,
        text="EDGE RADIUS",
        location_score=0.85,
        text_score=0.5
    )
    
    result = classify_delta(anchor, match_or_none=match)
    
    assert result.status == "uncertain"


# Test 5: Confidence math and clipping

def test_confidence_clipping_upper_bound():
    """Test that confidence is clipped to [0, 1] - upper bound."""
    anchor = make_anchor(char_no=9, requirement_norm="Ø 5.0")
    
    # Create match with artificially high scores that would exceed 1.0
    match = make_match(
        char_no=9,
        text="Ø 5.0",
        location_score=1.0,
        text_score=1.0,
        context_score=1.0
    )
    
    result = classify_delta(anchor, match_or_none=match)
    
    # Confidence should be clipped to 1.0
    assert result.confidence <= 1.0
    assert result.confidence >= 0.0


def test_confidence_clipping_lower_bound():
    """Test that confidence is clipped to [0, 1] - lower bound."""
    anchor = make_anchor(char_no=10, requirement_norm="Ø 5.0")
    
    # Create match with very low scores
    match = make_match(
        char_no=10,
        text="completely different text",
        location_score=0.0,
        text_score=0.0,
        context_score=0.0
    )
    
    result = classify_delta(anchor, match_or_none=match)
    
    # Confidence should be clipped to 0.0
    assert result.confidence >= 0.0
    assert result.confidence <= 1.0


def test_confidence_weighted_sum():
    """Test that confidence is computed as weighted sum of component scores."""
    anchor = make_anchor(char_no=11, requirement_norm="Ø 6.0")
    
    location_score = 0.7
    text_score = 0.8
    
    match = make_match(
        char_no=11,
        text="Ø 6.0",
        location_score=location_score,
        text_score=text_score,
        context_score=0.5
    )
    
    result = classify_delta(anchor, match_or_none=match)
    
    # For unchanged status: confidence = 0.5 * location + 0.5 * text
    expected = 0.5 * location_score + 0.5 * text_score
    assert result.confidence == pytest.approx(expected, abs=0.01)


# Test 6: Reason emission determinism

def test_reason_determinism_unchanged():
    """
    Test that reasons are deterministic given the same inputs (same order/content),
    so regressions in rule firing are caught.
    """
    anchor = make_anchor(char_no=12, requirement_norm="Ø 7.0")
    match = make_match(
        char_no=12,
        text="Ø 7.0",
        location_score=0.85,
        text_score=0.9,
        context_score=0.6
    )
    
    # Run classification multiple times
    result1 = classify_delta(anchor, match_or_none=match)
    result2 = classify_delta(anchor, match_or_none=match)
    result3 = classify_delta(anchor, match_or_none=match)
    
    # Reasons should be identical across runs
    assert result1.reasons == result2.reasons
    assert result2.reasons == result3.reasons
    
    # Verify expected reasons are present
    assert len(result1.reasons) >= 1
    assert any("all tokens match" in reason.lower() for reason in result1.reasons)


def test_reason_determinism_changed():
    """Test reason determinism for changed status."""
    anchor = make_anchor(char_no=13, requirement_norm="R 3.0")
    match = make_match(
        char_no=13,
        text="R 3.5",
        location_score=0.75,
        text_score=0.8,
        context_score=0.5
    )
    
    result1 = classify_delta(anchor, match_or_none=match)
    result2 = classify_delta(anchor, match_or_none=match)
    
    assert result1.reasons == result2.reasons
    assert result1.status == "changed"
    assert any("numeric" in reason.lower() for reason in result1.reasons)


def test_reason_determinism_uncertain():
    """Test reason determinism for uncertain status."""
    anchor = make_anchor(char_no=14, requirement_norm="Ø 9.0")
    match = make_match(
        char_no=14,
        text="LENGTH 15.0",
        location_score=0.8,
        text_score=0.4,
        context_score=0.3
    )
    
    result1 = classify_delta(anchor, match_or_none=match)
    result2 = classify_delta(anchor, match_or_none=match)
    
    assert result1.reasons == result2.reasons
    assert result1.status == "uncertain"


def test_reason_determinism_removed():
    """Test reason determinism for removed status."""
    anchor = make_anchor(char_no=15)
    
    result1 = classify_delta(anchor, match_or_none=None, location_search_coverage=1.0)
    result2 = classify_delta(anchor, match_or_none=None, location_search_coverage=1.0)
    
    assert result1.reasons == result2.reasons
    assert len(result1.reasons) == 1
    assert "no candidate" in result1.reasons[0].lower()


# Additional edge cases

def test_component_scores_preserved():
    """Test that component scores from match are preserved in DeltaItem."""
    anchor = make_anchor(char_no=16, requirement_norm="Ø 4.0")
    
    loc_score = 0.72
    txt_score = 0.88
    ctx_score = 0.55
    
    match = make_match(
        char_no=16,
        text="Ø 4.0",
        location_score=loc_score,
        text_score=txt_score,
        context_score=ctx_score
    )
    
    result = classify_delta(anchor, match_or_none=match)
    
    assert result.component_scores["location"] == pytest.approx(loc_score, abs=0.01)
    assert result.component_scores["text"] == pytest.approx(txt_score, abs=0.01)
    assert result.component_scores["context"] == pytest.approx(ctx_score, abs=0.01)


def test_high_location_score_adds_reason():
    """Test that high location score (>0.7) adds specific reason."""
    anchor = make_anchor(char_no=17, requirement_norm="Ø 11.0")
    match = make_match(
        char_no=17,
        text="Ø 11.0",
        location_score=0.95,  # High location score
        text_score=0.9
    )
    
    result = classify_delta(anchor, match_or_none=match)
    
    assert any("location agreement" in reason.lower() for reason in result.reasons)


def test_low_location_score_no_extra_reason():
    """Test that low location score (<0.7) doesn't add location reason."""
    anchor = make_anchor(char_no=18, requirement_norm="Ø 12.0")
    match = make_match(
        char_no=18,
        text="Ø 12.0",
        location_score=0.5,  # Low location score
        text_score=0.9
    )
    
    result = classify_delta(anchor, match_or_none=match)
    
    # Should not have location agreement reason
    location_reasons = [r for r in result.reasons if "location agreement" in r.lower()]
    assert len(location_reasons) == 0
