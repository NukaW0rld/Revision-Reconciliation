from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import fitz

from delta_preservation.io.pdf import (
    extract_text_spans,
    render_page,
    pdf_to_img_coords,
    TextSpan,
)


class DetectionMethod(Enum):
    PDF_TEXT = "pdf_text"
    CV = "cv"


@dataclass
class Balloon:
    """Represents a detected balloon with its characteristic number."""
    char_no: int
    page_index: int
    bbox_pdf: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    center_pdf: Tuple[float, float]  # (cx, cy)
    method: DetectionMethod
    confidence: float


def detect_balloons(pdf_path: Path, dpi: int = 300) -> Dict[int, Balloon]:
    """
    Detect balloons in a PDF using text extraction with CV fallback.
    
    Args:
        pdf_path: Path to PDF file
        dpi: DPI for rendering (used in CV fallback)
    
    Returns:
        Dictionary mapping char_no to Balloon object
    """
    doc = fitz.open(pdf_path)
    all_detections: List[Balloon] = []
    
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        
        # Try PDF text-based detection first
        text_balloons = _detect_balloons_from_text(pdf_path, page, page_idx, dpi)
        
        # Always keep text detections
        all_detections.extend(text_balloons)
        
        # If insufficient detections, add CV fallback detections
        if len(text_balloons) < 3:  # Heuristic threshold
            cv_balloons = _detect_balloons_cv(pdf_path, page, page_idx, dpi)
            all_detections.extend(cv_balloons)
    
    doc.close()
    
    # Deduplicate by keeping highest confidence per char_no
    result: Dict[int, Balloon] = {}
    for balloon in all_detections:
        if balloon.char_no not in result or balloon.confidence > result[balloon.char_no].confidence:
            result[balloon.char_no] = balloon
    
    return result


def _detect_balloons_from_text(
    pdf_path: Path,
    page: fitz.Page,
    page_idx: int,
    dpi: int
) -> List[Balloon]:
    """Extract balloons using PDF text spans with circle validation.
    
    Handles both single-number spans and combined spans like "24 25" where
    the PDF text extraction merges adjacent balloon numbers.
    """
    spans = extract_text_spans(pdf_path, page_idx)
    balloons = []
    
    # Render page once for all validations
    img = render_page(pdf_path, page_idx, dpi=dpi)
    
    for span in spans:
        text = span.text.strip()
        x0, y0, x1, y1 = span.bbox_pdf
        width = x1 - x0
        height = y1 - y0
        
        # Handle combined spans (e.g., "24 25") by splitting on whitespace
        tokens = text.split()
        num_tokens = len(tokens)
        
        for token_idx, token in enumerate(tokens):
            # Check if token is a candidate integer 1-200
            if not token.isdigit():
                continue
            
            char_no = int(token)
            if char_no < 1 or char_no > 200:
                continue
            
            # Calculate sub-bbox for this token (approximate equal division)
            if num_tokens > 1:
                # Divide the span bbox among tokens
                token_width = width / num_tokens
                sub_x0 = x0 + token_idx * token_width
                sub_x1 = sub_x0 + token_width
                sub_bbox = (sub_x0, y0, sub_x1, y1)
                sub_width = token_width
            else:
                sub_bbox = (x0, y0, x1, y1)
                sub_width = width
            
            # Check bbox is small and square-ish (balloon-like)
            if sub_width < 5 or height < 5 or sub_width > 50 or height > 50:
                continue
            
            # Relax aspect ratio for single digits (which can be narrow, ~0.4)
            # Two-digit numbers tend to be wider (~0.8+)
            aspect_ratio = sub_width / height
            if aspect_ratio < 0.3 or aspect_ratio > 2.0:
                continue
            
            # Create a pseudo-span for circle validation
            class PseudoSpan:
                def __init__(self, bbox):
                    self.bbox_pdf = bbox
            
            pseudo_span = PseudoSpan(sub_bbox)
            
            # Validate with circle detection on tight crop
            if _validate_circle_around_span(img, page, pseudo_span, dpi):
                cx = (sub_bbox[0] + sub_bbox[2]) / 2
                cy = (sub_bbox[1] + sub_bbox[3]) / 2
                confidence = 0.85 if num_tokens > 1 else 0.9  # Slightly lower for split spans
                
                balloons.append(Balloon(
                    char_no=char_no,
                    page_index=page_idx,
                    bbox_pdf=sub_bbox,
                    center_pdf=(cx, cy),
                    method=DetectionMethod.PDF_TEXT,
                    confidence=confidence
                ))
    
    return balloons


def _validate_circle_around_span(
    img: np.ndarray,
    page: fitz.Page,
    span: TextSpan,
    dpi: int
) -> bool:
    """Validate circle presence around text span using edge detection."""
    # Convert span bbox to image coordinates with padding
    x0_pdf, y0_pdf, x1_pdf, y1_pdf = span.bbox_pdf
    pad = max(x1_pdf - x0_pdf, y1_pdf - y0_pdf) * 0.5
    padded_bbox = (x0_pdf - pad, y0_pdf - pad, x1_pdf + pad, y1_pdf + pad)
    
    x0, y0, x1, y1 = pdf_to_img_coords(padded_bbox, page, dpi)
    
    # Clamp to image bounds
    h, w = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    
    if x1 <= x0 or y1 <= y0:
        return False
    
    crop = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Detect circles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=int(min(crop.shape[:2]) * 0.3),
        maxRadius=int(max(crop.shape[:2]) * 0.6)
    )
    
    return circles is not None and len(circles[0]) > 0


def _detect_balloons_cv(
    pdf_path: Path,
    page: fitz.Page,
    page_idx: int,
    dpi: int
) -> List[Balloon]:
    """Detect balloons using computer vision on rendered page."""
    img = render_page(pdf_path, page_idx, dpi=dpi)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect circles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )
    
    if circles is None:
        return []
    
    balloons = []
    scale = dpi / 72.0
    
    for circle in circles[0]:
        cx_img, cy_img, radius = circle
        
        # Crop circle region
        x0_img = int(cx_img - radius)
        y0_img = int(cy_img - radius)
        x1_img = int(cx_img + radius)
        y1_img = int(cy_img + radius)
        
        # Clamp to image bounds
        h, w = gray.shape
        x0_img = max(0, x0_img)
        y0_img = max(0, y0_img)
        x1_img = min(w, x1_img)
        y1_img = min(h, y1_img)
        
        if x1_img <= x0_img or y1_img <= y0_img:
            continue
        
        crop = gray[y0_img:y1_img, x0_img:x1_img]
        
        # Recognize digit
        char_no, confidence = _recognize_digit_template(crop)
        
        if char_no is None or char_no < 1 or char_no > 200:
            continue
        
        # Convert back to PDF coordinates
        cx_pdf = cx_img / scale
        cy_pdf = cy_img / scale
        x0_pdf = x0_img / scale
        y0_pdf = y0_img / scale
        x1_pdf = x1_img / scale
        y1_pdf = y1_img / scale
        
        balloons.append(Balloon(
            char_no=char_no,
            page_index=page_idx,
            bbox_pdf=(x0_pdf, y0_pdf, x1_pdf, y1_pdf),
            center_pdf=(cx_pdf, cy_pdf),
            method=DetectionMethod.CV,
            confidence=confidence
        ))
    
    return balloons


def _recognize_digit_template(crop: np.ndarray) -> Tuple[Optional[int], float]:
    """Recognize 1-2 digit number using template matching."""
    # Binarize
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours to isolate digit region
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, 0.0
    
    # Get bounding box of largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    if w < 3 or h < 3:
        return None, 0.0
    
    digit_crop = binary[y:y+h, x:x+w]
    
    # Resize to standard size for template matching
    digit_resized = cv2.resize(digit_crop, (20, 30))
    
    # Simple template matching against 0-9
    best_match = None
    best_score = 0.0
    
    for digit in range(10):
        template = _get_digit_template(digit)
        result = cv2.matchTemplate(digit_resized, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_match = digit
    
    # Check for two-digit number by looking for multiple components
    if best_match is not None and best_score > 0.5:
        # Simple heuristic: if crop is wide, might be 2 digits
        aspect = w / h
        if aspect > 1.5:
            # Try to split and recognize second digit
            mid = w // 2
            left_crop = binary[y:y+h, x:x+mid]
            right_crop = binary[y:y+h, x+mid:x+w]
            
            left_digit, left_score = _match_single_digit(left_crop)
            right_digit, right_score = _match_single_digit(right_crop)
            
            if left_digit is not None and right_digit is not None:
                two_digit = left_digit * 10 + right_digit
                avg_score = (left_score + right_score) / 2
                if avg_score > 0.4:
                    return two_digit, avg_score
        
        return best_match, best_score
    
    return None, 0.0


def _match_single_digit(crop: np.ndarray) -> Tuple[Optional[int], float]:
    """Match a single digit crop against templates."""
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return None, 0.0
    
    resized = cv2.resize(crop, (20, 30))
    
    best_match = None
    best_score = 0.0
    
    for digit in range(10):
        template = _get_digit_template(digit)
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_match = digit
    
    return best_match, best_score


def _get_digit_template(digit: int) -> np.ndarray:
    """Generate simple template for digit 0-9."""
    # Create 20x30 template with digit drawn
    template = np.zeros((30, 20), dtype=np.uint8)
    
    # Use OpenCV's putText to draw digit
    cv2.putText(
        template,
        str(digit),
        (2, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255,
        2
    )
    
    return template
