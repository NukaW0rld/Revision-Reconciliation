"""
PDF rendering and text extraction primitives using PyMuPDF (fitz).

This module provides deterministic primitives for extracting both visual and textual
information from PDF pages with precise coordinate mapping between PDF space and
rendered image space.
"""

from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

import fitz  # PyMuPDF
import numpy as np


@dataclass
class TextSpan:
    """
    Represents a single text span extracted from a PDF page.
    
    Attributes:
        text: The actual text content of the span
        bbox_pdf: Bounding box in PDF coordinates (x0, y0, x1, y1)
                  where (x0, y0) is top-left and (x1, y1) is bottom-right
        font_size: Font size in points (if available)
        block_id: Index of the block containing this span
        line_id: Index of the line within the block containing this span
        span_id: Index of the span within the line
    """
    text: str
    bbox_pdf: Tuple[float, float, float, float]
    font_size: float
    block_id: int
    line_id: int
    span_id: int


def render_page(pdf_path: Path, page_index: int, dpi: int = 300) -> np.ndarray:
    """
    Render a PDF page to a NumPy image array at the specified DPI.
    
    This function opens the PDF, renders the requested page at the given DPI,
    and returns the result as a NumPy array in BGR format (OpenCV compatible).
    
    Args:
        pdf_path: Path to the PDF file
        page_index: Zero-based page index to render
        dpi: Dots per inch for rendering (default: 300)
    
    Returns:
        NumPy array of shape (height, width, 3) in BGR format
    
    Raises:
        FileNotFoundError: If pdf_path does not exist
        IndexError: If page_index is out of range
    
    Notes:
        - PyMuPDF uses a scale factor relative to 72 DPI (PDF standard)
        - The resulting image uses top-left origin (0,0) at upper-left corner
        - Color channels are converted from RGB to BGR for OpenCV compatibility
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Open PDF document
    doc = fitz.open(pdf_path)
    
    page_count = len(doc)
    if page_index < 0 or page_index >= page_count:
        doc.close()
        raise IndexError(f"Page index {page_index} out of range (0-{page_count-1})")
    
    # Load the requested page
    page = doc.load_page(page_index)
    
    # Calculate zoom factor: PyMuPDF's default is 72 DPI
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    
    # Render page to pixmap
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Convert pixmap to NumPy array
    # PyMuPDF pixmap.samples gives us raw bytes in RGB format
    img_data = np.frombuffer(pix.samples, dtype=np.uint8)
    img_rgb = img_data.reshape(pix.height, pix.width, pix.n)
    
    # Convert RGB to BGR for OpenCV compatibility
    if pix.n == 3:  # RGB
        img_bgr = img_rgb[:, :, ::-1].copy()
    else:
        # Fallback for grayscale or other formats
        img_bgr = img_rgb.copy()
    
    doc.close()
    
    return img_bgr


def extract_text_spans(pdf_path: Path, page_index: int) -> List[TextSpan]:
    """
    Extract all text spans from a PDF page with precise bounding boxes.
    
    This function uses PyMuPDF's structured text extraction to walk through
    blocks, lines, and spans, capturing each span's text content, bounding box
    in PDF coordinates, and metadata.
    
    Args:
        pdf_path: Path to the PDF file
        page_index: Zero-based page index to extract text from
    
    Returns:
        List of TextSpan objects, each containing text and positioning metadata
    
    Raises:
        FileNotFoundError: If pdf_path does not exist
        IndexError: If page_index is out of range
    
    Notes:
        - Bounding boxes are in PDF coordinate space (points, 72 DPI)
        - PyMuPDF's get_text("dict") returns coordinates with top-left origin for
          consistency with rendering
        - Empty spans (whitespace only) are included if present in PDF structure
        - block_id, line_id, span_id are sequential indices from iteration order
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Open PDF document
    doc = fitz.open(pdf_path)
    
    page_count = len(doc)
    if page_index < 0 or page_index >= page_count:
        doc.close()
        raise IndexError(f"Page index {page_index} out of range (0-{page_count-1})")
    
    # Load the requested page
    page = doc.load_page(page_index)
    
    # Extract text with detailed structure
    # "dict" mode returns hierarchical structure: blocks -> lines -> spans
    text_dict = page.get_text("dict")
    
    spans = []
    
    # Walk through blocks
    for block_idx, block in enumerate(text_dict.get("blocks", [])):
        # Skip image blocks (only process text blocks)
        if block.get("type") != 0:
            continue
        
        # Walk through lines in this block
        for line_idx, line in enumerate(block.get("lines", [])):
            # Walk through spans in this line
            for span_idx, span in enumerate(line.get("spans", [])):
                text = span.get("text", "")
                bbox = span.get("bbox", (0, 0, 0, 0))  # (x0, y0, x1, y1)
                font_size = span.get("size", 0.0)
                
                # Create TextSpan object
                text_span = TextSpan(
                    text=text,
                    bbox_pdf=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    font_size=font_size,
                    block_id=block_idx,
                    line_id=line_idx,
                    span_id=span_idx
                )
                
                spans.append(text_span)
    
    doc.close()
    
    return spans


def pdf_to_img_coords(
    bbox_pdf: Tuple[float, float, float, float],
    page: fitz.Page,
    dpi: int
) -> Tuple[int, int, int, int]:
    """
    Convert PDF-space bounding box to image pixel coordinates.
    
    This function transforms a bounding box from PDF coordinate space (points at 72 DPI)
    to pixel coordinates in the rendered image at the specified DPI. The conversion
    accounts for the scale factor and ensures consistency with how render_page()
    produces images.
    
    Args:
        bbox_pdf: Bounding box in PDF coordinates (x0, y0, x1, y1)
                  PyMuPDF uses top-left origin for consistency
        page: PyMuPDF page object (used for validation, if needed)
        dpi: The DPI used when rendering the page
    
    Returns:
        Tuple of (x0, y0, x1, y1) in integer pixel coordinates suitable for
        cropping the rendered image array
    
    Notes:
        - Scale factor is dpi / 72.0 (PDF standard DPI)
        - PyMuPDF's coordinate system has (0, 0) at top-left for both PDF
          coordinates and rendered pixmaps, so no origin flip is needed
        - Result is rounded to integers for pixel-perfect cropping
        - Coordinates are clamped to ensure they're within valid image bounds
    """
    x0_pdf, y0_pdf, x1_pdf, y1_pdf = bbox_pdf
    
    # Calculate scale factor
    scale = dpi / 72.0
    
    # Apply scale to convert PDF points to pixels
    x0_img = x0_pdf * scale
    y0_img = y0_pdf * scale
    x1_img = x1_pdf * scale
    y1_img = y1_pdf * scale
    
    # Round to integer pixel coordinates
    # Use floor for top-left and ceil for bottom-right to ensure coverage
    x0_px = int(np.floor(x0_img))
    y0_px = int(np.floor(y0_img))
    x1_px = int(np.ceil(x1_img))
    y1_px = int(np.ceil(y1_img))
    
    return (x0_px, y0_px, x1_px, y1_px)
