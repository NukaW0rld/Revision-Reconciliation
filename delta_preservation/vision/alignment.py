from pathlib import Path
from typing import Tuple
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Transform:
    """Homography transform from Rev A to Rev B with quality metrics."""
    H: np.ndarray  # 3x3 homography matrix
    inliers: int
    inlier_ratio: float
    quality_ok: bool


class AlignmentError(Exception):
    """Raised when image alignment fails quality thresholds."""
    pass


def estimate_transform(imgA: np.ndarray, imgB: np.ndarray) -> Transform:
    """
    Estimate homography transform from Rev A to Rev B using ORB features.
    
    Args:
        imgA: Rev A rendered page image (BGR)
        imgB: Rev B rendered page image (BGR)
    
    Returns:
        Transform object with homography and quality metrics
    
    Raises:
        AlignmentError: If alignment fails quality thresholds
    """
    # Convert to grayscale
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    
    # Detect and compute ORB features
    orb = cv2.ORB_create(nfeatures=4000)
    kpA, descA = orb.detectAndCompute(grayA, None)
    kpB, descB = orb.detectAndCompute(grayB, None)
    
    if descA is None or descB is None:
        raise AlignmentError("Failed to extract features from one or both images")
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descA, descB)
    
    if len(matches) < 4:
        raise AlignmentError(f"Insufficient matches found: {len(matches)} < 4")
    
    # Sort by distance and keep best matches
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:min(300, len(matches))]
    
    # Extract matched point arrays
    ptsA = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find homography with RANSAC
    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold=3.0)
    
    if H is None:
        raise AlignmentError("Failed to compute homography")
    
    # Calculate quality metrics
    inliers = int(np.sum(mask))
    inlier_ratio = inliers / len(matches)
    quality_ok = inliers >= 40 and inlier_ratio >= 0.15
    
    if not quality_ok:
        raise AlignmentError(
            f"Alignment quality too low: inliers={inliers}, ratio={inlier_ratio:.3f}"
        )
    
    return Transform(
        H=H,
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        quality_ok=quality_ok
    )


def apply_transform_bbox(bbox_xyxy: Tuple[float, float, float, float], H: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Transform axis-aligned bbox from Rev A to Rev B coordinates.
    
    Args:
        bbox_xyxy: Bounding box (x0, y0, x1, y1) in Rev A coordinates
        H: 3x3 homography matrix
    
    Returns:
        Transformed axis-aligned bbox (x0, y0, x1, y1) in Rev B coordinates
    """
    x0, y0, x1, y1 = bbox_xyxy
    
    # Four corners of bbox
    corners = np.float32([
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1]
    ]).reshape(-1, 1, 2)
    
    # Transform corners
    transformed = cv2.perspectiveTransform(corners, H)
    
    # Get axis-aligned bbox from transformed corners
    x_coords = transformed[:, 0, 0]
    y_coords = transformed[:, 0, 1]
    
    return (
        float(np.min(x_coords)),
        float(np.min(y_coords)),
        float(np.max(x_coords)),
        float(np.max(y_coords))
    )


def render_debug_overlay(
    imgB: np.ndarray,
    pointsA: np.ndarray,
    H: np.ndarray,
    out_path: Path
) -> None:
    """
    Render debug overlay showing transformed anchor points on Rev B.
    
    Args:
        imgB: Rev B rendered page image (BGR)
        pointsA: Nx2 array of anchor centers from Rev A
        H: 3x3 homography matrix
        out_path: Path to save debug image
    """
    overlay = imgB.copy()
    
    # Transform points from A to B
    pointsA_reshaped = pointsA.reshape(-1, 1, 2).astype(np.float32)
    pointsB = cv2.perspectiveTransform(pointsA_reshaped, H)
    
    # Draw circles at transformed locations
    for pt in pointsB:
        x, y = pt[0]
        cv2.circle(overlay, (int(x), int(y)), 5, (0, 255, 0), 2)
    
    # Save overlay
    cv2.imwrite(str(out_path), overlay)
