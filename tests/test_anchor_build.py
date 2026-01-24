import pytest
import math
from typing import Dict, List

from delta_preservation.io.pdf import TextSpan
from delta_preservation.vision.balloons import Balloon, DetectionMethod
from delta_preservation.reconcile.anchors import build_revA_anchors, Anchor


@pytest.fixture
def small_revA_text_spans() -> List[TextSpan]:
    """
    Small fixture of PDF text spans on page 0.
    
    Simulates a typical engineering drawing with:
    - Title block text (bottom-right, should be filtered)
    - Requirement text near balloons
    - Random text elsewhere on page
    """
    return [
        # Requirement text near balloon 1 (page center-left)
        TextSpan(
            text="Ø8 +/- 0.03 mm",
            bbox_pdf=(100.0, 200.0, 180.0, 215.0),
            font_size=10.0,
            block_id=0,
            line_id=0,
            span_id=0
        ),
        # Requirement text near balloon 2 (page upper-right)
        TextSpan(
            text="Thread M4-6H",
            bbox_pdf=(450.0, 150.0, 520.0, 165.0),
            font_size=10.0,
            block_id=0,
            line_id=1,
            span_id=0
        ),
        # Requirement text near balloon 3 (page lower-left)
        TextSpan(
            text="Edge radius R2 mm",
            bbox_pdf=(120.0, 500.0, 220.0, 515.0),
            font_size=10.0,
            block_id=0,
            line_id=2,
            span_id=0
        ),
        # Random text far from balloons (should not be matched)
        TextSpan(
            text="Random text",
            bbox_pdf=(300.0, 50.0, 380.0, 65.0),
            font_size=10.0,
            block_id=0,
            line_id=3,
            span_id=0
        ),
        # Title block text (bottom-right, should be filtered out)
        TextSpan(
            text="TITLE BLOCK INFO",
            bbox_pdf=(520.0, 750.0, 600.0, 765.0),
            font_size=12.0,
            block_id=0,
            line_id=4,
            span_id=0
        ),
        # Additional context text near balloon 1
        TextSpan(
            text="hole depth 10mm",
            bbox_pdf=(95.0, 220.0, 175.0, 235.0),
            font_size=9.0,
            block_id=0,
            line_id=5,
            span_id=0
        ),
        # Additional context text near balloon 2
        TextSpan(
            text="2X threads",
            bbox_pdf=(445.0, 170.0, 510.0, 185.0),
            font_size=9.0,
            block_id=0,
            line_id=6,
            span_id=0
        ),
        # Additional context text near balloon 3
        TextSpan(
            text="all edges",
            bbox_pdf=(115.0, 520.0, 175.0, 535.0),
            font_size=9.0,
            block_id=0,
            line_id=7,
            span_id=0
        ),
        # Requirement text near balloon 4 (page center)
        TextSpan(
            text="Length 25 mm",
            bbox_pdf=(340.0, 400.0, 410.0, 415.0),
            font_size=10.0,
            block_id=0,
            line_id=8,
            span_id=0
        ),
        # Requirement text near balloon 5 (page center-lower)
        TextSpan(
            text="Countersink Ø12 x 90 deg",
            bbox_pdf=(195.0, 295.0, 310.0, 310.0),
            font_size=10.0,
            block_id=0,
            line_id=9,
            span_id=0
        ),
        # Random text far from all balloons (for spatial proximity test)
        TextSpan(
            text="see detail A",
            bbox_pdf=(550.0, 100.0, 620.0, 115.0),
            font_size=10.0,
            block_id=0,
            line_id=10,
            span_id=0
        ),
        # Very long span (should be filtered as likely header)
        TextSpan(
            text="This is a very long header text that should be filtered out because it exceeds the length threshold for candidate spans",
            bbox_pdf=(50.0, 30.0, 550.0, 45.0),
            font_size=14.0,
            block_id=0,
            line_id=9,
            span_id=0
        ),
    ]


@pytest.fixture
def form3_chars() -> Dict[int, str]:
    """
    Form3 characteristic numbers mapped to requirement text.
    
    Includes 5 characteristics with varying complexity.
    """
    return {
        1: "Ø8 +/- 0.03 mm",
        2: "Thread M4-6H",
        3: "Edge radius R2 mm",
        4: "Length 25 mm",
        5: "Countersink Ø12 x 90 deg",
    }


@pytest.fixture
def balloons() -> Dict[int, Balloon]:
    """
    Detected balloons for characteristics 1-5.
    
    Balloons are positioned to be near their corresponding requirement text
    in the text spans fixture.
    """
    return {
        1: Balloon(
            char_no=1,
            page_index=0,
            bbox_pdf=(85.0, 195.0, 105.0, 215.0),
            center_pdf=(95.0, 205.0),
            method=DetectionMethod.PDF_TEXT,
            confidence=0.9
        ),
        2: Balloon(
            char_no=2,
            page_index=0,
            bbox_pdf=(435.0, 145.0, 455.0, 165.0),
            center_pdf=(445.0, 155.0),
            method=DetectionMethod.PDF_TEXT,
            confidence=0.9
        ),
        3: Balloon(
            char_no=3,
            page_index=0,
            bbox_pdf=(105.0, 495.0, 125.0, 515.0),
            center_pdf=(115.0, 505.0),
            method=DetectionMethod.PDF_TEXT,
            confidence=0.9
        ),
        4: Balloon(
            char_no=4,
            page_index=0,
            bbox_pdf=(335.0, 395.0, 355.0, 415.0),
            center_pdf=(345.0, 405.0),
            method=DetectionMethod.PDF_TEXT,
            confidence=0.9
        ),
        5: Balloon(
            char_no=5,
            page_index=0,
            bbox_pdf=(200.0, 300.0, 220.0, 320.0),
            center_pdf=(210.0, 310.0),
            method=DetectionMethod.CV,
            confidence=0.7
        ),
    }


class TestAnchorBuildIntegration:
    """
    Integration tests for build_revA_anchors that verify end-to-end behavior
    with realistic fixtures.
    """
    
    def test_anchor_produced_for_every_detected_balloon(
        self,
        form3_chars: Dict[int, str],
        balloons: Dict[int, Balloon],
        small_revA_text_spans: List[TextSpan]
    ):
        """
        Assertion (1): An anchor is produced for every Form3 char_no that has
        a detected balloon.
        """
        anchors = build_revA_anchors(form3_chars, balloons, small_revA_text_spans)
        
        # Extract char_nos from anchors
        anchor_char_nos = {anchor.char_no for anchor in anchors}
        
        # All balloon char_nos should have anchors
        balloon_char_nos = set(balloons.keys())
        assert anchor_char_nos == balloon_char_nos, (
            f"Expected anchors for {balloon_char_nos}, got {anchor_char_nos}"
        )
        
        # Should have exactly 5 anchors (one per balloon)
        assert len(anchors) == 5
    
    def test_anchor_page_matches_balloon_page(
        self,
        form3_chars: Dict[int, str],
        balloons: Dict[int, Balloon],
        small_revA_text_spans: List[TextSpan]
    ):
        """
        Assertion (2): Each anchor's page matches the balloon's page.
        """
        anchors = build_revA_anchors(form3_chars, balloons, small_revA_text_spans)
        
        for anchor in anchors:
            balloon = balloons[anchor.char_no]
            assert anchor.page == balloon.page_index, (
                f"Anchor {anchor.char_no} page {anchor.page} != "
                f"balloon page {balloon.page_index}"
            )
    
    def test_req_bbox_coverage_and_spatial_proximity(
        self,
        form3_chars: Dict[int, str],
        balloons: Dict[int, Balloon],
        small_revA_text_spans: List[TextSpan]
    ):
        """
        Assertion (3): req_bbox is not None for a clear majority of characteristics
        (>= 70%) and is spatially closer to the balloon center than a random span
        on the same page.
        """
        anchors = build_revA_anchors(form3_chars, balloons, small_revA_text_spans)
        
        # Count anchors with req_bbox
        anchors_with_bbox = [a for a in anchors if a.req_bbox is not None]
        coverage_rate = len(anchors_with_bbox) / len(anchors)
        
        assert coverage_rate >= 0.70, (
            f"req_bbox coverage {coverage_rate:.1%} < 70%"
        )
        
        # For each anchor with req_bbox, verify it's closer than a random span
        for anchor in anchors_with_bbox:
            balloon = balloons[anchor.char_no]
            balloon_cx, balloon_cy = balloon.center_pdf
            
            # Calculate distance from req_bbox to balloon center
            req_x0, req_y0, req_x1, req_y1 = anchor.req_bbox
            req_cx = (req_x0 + req_x1) / 2
            req_cy = (req_y0 + req_y1) / 2
            req_distance = math.sqrt(
                (req_cx - balloon_cx)**2 + (req_cy - balloon_cy)**2
            )
            
            # Find a random span on the same page (use the "Random text" span)
            random_span = next(
                s for s in small_revA_text_spans
                if s.text == "Random text" and s.block_id == anchor.page
            )
            rand_x0, rand_y0, rand_x1, rand_y1 = random_span.bbox_pdf
            rand_cx = (rand_x0 + rand_x1) / 2
            rand_cy = (rand_y0 + rand_y1) / 2
            rand_distance = math.sqrt(
                (rand_cx - balloon_cx)**2 + (rand_cy - balloon_cy)**2
            )
            
            assert req_distance < rand_distance, (
                f"Anchor {anchor.char_no}: req_bbox distance {req_distance:.1f} "
                f">= random span distance {rand_distance:.1f}"
            )
    
    def test_req_bbox_contains_normalized_token_from_requirement(
        self,
        form3_chars: Dict[int, str],
        balloons: Dict[int, Balloon],
        small_revA_text_spans: List[TextSpan]
    ):
        """
        Assertion (4): The chosen req_bbox span contains at least one normalized
        token from the Form3 requirement, ensuring token overlap drove the match.
        """
        anchors = build_revA_anchors(form3_chars, balloons, small_revA_text_spans)
        
        for anchor in anchors:
            if anchor.req_bbox is None:
                # Skip anchors without req_bbox (allowed by assertion 3)
                continue
            
            # Get normalized tokens from the requirement
            req_tokens = set(anchor.requirement_norm.split())
            
            # Find the span that matches req_bbox
            matched_span = None
            for span in small_revA_text_spans:
                if span.bbox_pdf == anchor.req_bbox:
                    matched_span = span
                    break
            
            assert matched_span is not None, (
                f"Anchor {anchor.char_no}: req_bbox not found in text spans"
            )
            
            # Normalize the span text
            from delta_preservation.reconcile.normalize import parse_requirement
            span_norm = parse_requirement(matched_span.text)
            span_tokens = set(span_norm.norm_text.split())
            
            # Check for token overlap
            overlap = req_tokens & span_tokens
            assert len(overlap) > 0, (
                f"Anchor {anchor.char_no}: no token overlap between "
                f"requirement '{anchor.requirement_norm}' and "
                f"span '{span_norm.norm_text}'"
            )
    
    def test_local_context_non_empty_and_within_radius(
        self,
        form3_chars: Dict[int, str],
        balloons: Dict[int, Balloon],
        small_revA_text_spans: List[TextSpan]
    ):
        """
        Assertion (5): local_context is non-empty and all its spans lie within
        the configured radius of the chosen req_bbox (or balloon center when
        req_bbox is missing), guaranteeing anchors are geometrically coherent
        and usable for downstream matching.
        """
        anchors = build_revA_anchors(form3_chars, balloons, small_revA_text_spans)
        
        # Context radius is hardcoded in anchors.py
        CONTEXT_RADIUS = 150.0
        
        for anchor in anchors:
            # local_context should be non-empty
            assert len(anchor.local_context) > 0, (
                f"Anchor {anchor.char_no}: local_context is empty"
            )
            
            # Determine the center point for context
            if anchor.req_bbox is not None:
                cx = (anchor.req_bbox[0] + anchor.req_bbox[2]) / 2
                cy = (anchor.req_bbox[1] + anchor.req_bbox[3]) / 2
            else:
                balloon = balloons[anchor.char_no]
                cx, cy = balloon.center_pdf
            
            # Verify all context spans are within radius
            for span in anchor.local_context:
                sx0, sy0, sx1, sy1 = span.bbox_pdf
                span_cx = (sx0 + sx1) / 2
                span_cy = (sy0 + sy1) / 2
                distance = math.sqrt((span_cx - cx)**2 + (span_cy - cy)**2)
                
                assert distance <= CONTEXT_RADIUS, (
                    f"Anchor {anchor.char_no}: context span '{span.text}' "
                    f"at distance {distance:.1f} > radius {CONTEXT_RADIUS}"
                )
    
    def test_anchor_fields_populated_correctly(
        self,
        form3_chars: Dict[int, str],
        balloons: Dict[int, Balloon],
        small_revA_text_spans: List[TextSpan]
    ):
        """
        Additional test: Verify all anchor fields are populated correctly.
        """
        anchors = build_revA_anchors(form3_chars, balloons, small_revA_text_spans)
        
        for anchor in anchors:
            # char_no should be valid
            assert anchor.char_no in form3_chars
            assert anchor.char_no in balloons
            
            # page should be non-negative
            assert anchor.page >= 0
            
            # balloon_bbox should match the balloon
            balloon = balloons[anchor.char_no]
            assert anchor.balloon_bbox == balloon.bbox_pdf
            
            # requirement_raw should match form3_chars
            assert anchor.requirement_raw == form3_chars[anchor.char_no]
            
            # requirement_norm should be uppercase and non-empty
            assert anchor.requirement_norm.isupper()
            assert len(anchor.requirement_norm) > 0
            
            # local_context should contain TextSpan objects
            assert all(isinstance(span, TextSpan) for span in anchor.local_context)
    
    def test_no_anchors_for_characteristics_without_balloons(
        self,
        small_revA_text_spans: List[TextSpan]
    ):
        """
        Test that characteristics without detected balloons do not produce anchors.
        """
        # Form3 chars with no corresponding balloons
        form3_chars_no_balloons = {
            10: "Ø10 mm",
            20: "Thread M6-6H",
        }
        balloons_empty = {}
        
        anchors = build_revA_anchors(
            form3_chars_no_balloons,
            balloons_empty,
            small_revA_text_spans
        )
        
        assert len(anchors) == 0, "Should produce no anchors when no balloons exist"
    
    def test_anchor_with_no_req_bbox_still_has_context(
        self,
        small_revA_text_spans: List[TextSpan]
    ):
        """
        Test that anchors without req_bbox still have local_context centered
        on the balloon.
        """
        # Create a characteristic with text that won't match any spans
        form3_chars_no_match = {
            99: "Completely unmatched requirement text xyz123",
        }
        balloons_no_match = {
            99: Balloon(
                char_no=99,
                page_index=0,
                bbox_pdf=(650.0, 650.0, 670.0, 670.0),
                center_pdf=(660.0, 660.0),
                method=DetectionMethod.PDF_TEXT,
                confidence=0.9
            ),
        }
        
        anchors = build_revA_anchors(
            form3_chars_no_match,
            balloons_no_match,
            small_revA_text_spans
        )
        
        assert len(anchors) == 1
        anchor = anchors[0]
        
        # req_bbox should be None (no good match)
        assert anchor.req_bbox is None
        
        # But local_context should still be populated (centered on balloon)
        assert len(anchor.local_context) > 0
        
        # Verify context is centered on balloon
        balloon = balloons_no_match[99]
        balloon_cx, balloon_cy = balloon.center_pdf
        CONTEXT_RADIUS = 150.0
        
        for span in anchor.local_context:
            sx0, sy0, sx1, sy1 = span.bbox_pdf
            span_cx = (sx0 + sx1) / 2
            span_cy = (sy0 + sy1) / 2
            distance = math.sqrt((span_cx - balloon_cx)**2 + (span_cy - balloon_cy)**2)
            
            assert distance <= CONTEXT_RADIUS
