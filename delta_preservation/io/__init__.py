"""
Input/Output module for handling PDF and Excel file operations.

This module provides functionality for:
- PDF rendering and text extraction with precise coordinate mapping
- AS9102 Form 3 parsing from Excel files

The io module serves as the foundation for extracting structured data from
engineering documents used in the revision reconciliation pipeline.
"""

from .pdf import render_page, extract_text_spans, pdf_to_img_coords, TextSpan, join_text_spans
from .xlsx import load_form3, Characteristic

__all__ = [
    "render_page",
    "extract_text_spans", 
    "pdf_to_img_coords",
    "TextSpan",
    "join_text_spans",
    "load_form3",
    "Characteristic"
]