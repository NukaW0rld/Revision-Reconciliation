"""
Image alignment module for engineering drawing revision comparison.

This module provides robust image alignment capabilities for comparing different 
revisions of engineering drawings. It uses ORB feature detection and homography 
estimation to establish geometric correspondence between drawing versions, even 
when there are layout changes, scaling differences, or perspective variations.

The alignment process is critical for the revision reconciliation pipeline as it 
enables accurate spatial matching of characteristics between Rev A and Rev B drawings.
"""

from pathlib import Path
from typing import Tuple
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Transform:
    """
    Represents a geometric transformation between two drawing revisions with quality metrics.
    
    The Transform encapsulates a homography matrix that maps coordinates from Rev A 
    to Rev B coordinate space, along with quality indicators to assess the reliability
    of the alignment.
    
    Attributes:
        H: 3x3 homography matrix for perspective transformation
        inliers: Number of feature point correspondences that support the transform
        inlier_ratio: Fraction of matches that are geometrically consistent (0.0 to 1.0)
        quality_ok: Boolean indicating if the alignment meets minimum quality thresholds
    """
    H: np.ndarray  # 3x3 homography matrix
    inliers: int
    inlier_ratio: float
    quality_ok: bool


class AlignmentError(Exception):
    """
    Exception raised when image alignment fails to meet quality thresholds.
    
    This error indicates that the automatic alignment between drawing revisions
    could not establish a reliable geometric transformation. This may occur with
    severely distorted images, drawings with insufficient common features, or
    when the layout changes are too dramatic for feature-based alignment.
    """
    pass


def estimate_transform(imgA: np.ndarray, imgB: np.ndarray) -> Transform:
    """
    Estimate homography transformation from Rev A to Rev B using ORB feature matching.
    
    This function implements a robust feature-based alignment pipeline:
    1. Extract ORB features from both images 
    2. Match features using brute-force matcher with cross-checking
    3. Estimate homography using RANSAC to filter outliers
    4. Validate alignment quality using inlier metrics
    
    The homography enables mapping of coordinates and bounding boxes from Rev A 
    coordinate space to Rev B coordinate space, which is essential for characteristic
    matching across drawing revisions.
    
    Args:
        imgA: Rev A rendered page image in BGR format
        imgB: Rev B rendered page image in BGR format
    
    Returns:
        Transform object containing the 3x3 homography matrix and quality metrics
    
    Raises:
        AlignmentError: If insufficient features found, homography estimation fails,
                       or the resulting alignment doesn't meet quality thresholds
                       (minimum 40 inliers and 15% inlier ratio)
    
    Notes:
        - Uses ORB features for rotation and scale invariance
        - RANSAC robustly handles outliers in feature matches
        - Quality thresholds ensure reliable geometric transformation
        - Higher inlier counts and ratios indicate better alignment confidence
    """
    # Convert to grayscale
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    
    # Detect and compute ORB features with generous feature count
    # Higher feature count improves robustness for drawings with sparse detail
    orb = cv2.ORB_create(nfeatures=4000)
    kpA, descA = orb.detectAndCompute(grayA, None)
    kpB, descB = orb.detectAndCompute(grayB, None)
    
    if descA is None or descB is None:
        raise AlignmentError("Failed to extract features from one or both images")
    
    # Match features using brute-force matcher with cross-checking
    # Cross-checking ensures bidirectional consistency of matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descA, descB)
    
    if len(matches) < 4:
        raise AlignmentError(f"Insufficient matches found: {len(matches)} < 4")
    
    # Sort by match distance (lower is better) and limit to best matches
    # This filtering improves RANSAC performance by reducing outlier noise
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:min(300, len(matches))]
    
    # Extract corresponding point coordinates for homography estimation
    ptsA = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Estimate homography using RANSAC for outlier rejection
    # 3.0 pixel threshold balances precision with robustness
    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold=3.0)
    
    if H is None:
        raise AlignmentError("Failed to compute homography")
    
    # Evaluate alignment quality using inlier statistics
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
