"""
Computer Vision module for engineering drawing analysis.

This module provides computer vision functionality for:
- Balloon detection in engineering drawings using both PDF text extraction and OpenCV
- Image alignment and transformation estimation between drawing revisions
- Visual snippet extraction and processing for evidence generation
- Bounding box utilities for spatial analysis

The vision module combines traditional OCR approaches with modern computer vision
techniques to provide robust detection capabilities for engineering documents.
"""

from .balloons import detect_balloons, Balloon, DetectionMethod
from .alignment import estimate_transform, apply_transform_bbox, Transform, AlignmentError

__all__ = [
    "detect_balloons",
    "Balloon", 
    "DetectionMethod",
    "estimate_transform",
    "apply_transform_bbox",
    "Transform",
    "AlignmentError"
]