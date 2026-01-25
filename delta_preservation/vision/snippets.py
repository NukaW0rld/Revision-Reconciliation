from pathlib import Path
from typing import Tuple

import numpy as np
import cv2


def crop_with_padding(
    img: np.ndarray,
    bbox_px: Tuple[int, int, int, int],
    pad_px: int
) -> np.ndarray:
    """
    Crop image region with padding, clamped to image bounds.
    
    Args:
        img: Input image as NumPy array (H, W, C) in BGR format
        bbox_px: Bounding box in pixel coordinates (x0, y0, x1, y1)
        pad_px: Padding to add on all sides in pixels
    
    Returns:
        Cropped image region as NumPy array
    
    Raises:
        ValueError: If bbox is invalid or results in empty crop
    """
    x0, y0, x1, y1 = bbox_px
    
    # Validate input bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid bbox: x1 must be > x0 and y1 must be > y0, got {bbox_px}")
    
    # Expand bbox with padding
    x0_pad = x0 - pad_px
    y0_pad = y0 - pad_px
    x1_pad = x1 + pad_px
    y1_pad = y1 + pad_px
    
    # Clamp to image bounds
    h, w = img.shape[:2]
    x0_clamp = max(0, x0_pad)
    y0_clamp = max(0, y0_pad)
    x1_clamp = min(w, x1_pad)
    y1_clamp = min(h, y1_pad)
    
    # Validate clamped bbox
    if x1_clamp <= x0_clamp or y1_clamp <= y0_clamp:
        raise ValueError(
            f"Clamped bbox is empty: original={bbox_px}, "
            f"padded=({x0_pad},{y0_pad},{x1_pad},{y1_pad}), "
            f"clamped=({x0_clamp},{y0_clamp},{x1_clamp},{y1_clamp}), "
            f"image_shape=({h},{w})"
        )
    
    # Crop and return
    crop = img[y0_clamp:y1_clamp, x0_clamp:x1_clamp]
    
    if crop.size == 0:
        raise ValueError(f"Crop resulted in empty array for bbox {bbox_px}")
    
    return crop


def save_snippet(
    img: np.ndarray,
    out_dir: Path,
    char_no: int,
    rev_label: str,
    page_index: int
) -> str:
    """
    Save image snippet with deterministic filename.
    
    Args:
        img: Image to save as NumPy array
        out_dir: Output directory path
        char_no: Characteristic number (0-padded to 3 digits)
        rev_label: Revision label (e.g., "revA", "revB")
        page_index: Page index
    
    Returns:
        Relative path string of saved file
    
    Raises:
        ValueError: If image is empty or invalid
        OSError: If directory creation or file write fails
    """
    if img.size == 0:
        raise ValueError("Cannot save empty image")
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate deterministic filename
    filename = f"char_{char_no:03d}_{rev_label}_p{page_index}.png"
    out_path = out_dir / filename
    
    # Write image
    success = cv2.imwrite(str(out_path), img)
    if not success:
        raise OSError(f"Failed to write image to {out_path}")
    
    return filename
