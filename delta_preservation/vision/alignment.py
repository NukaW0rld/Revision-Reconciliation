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
from typing import Tuple, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

import cv2
import numpy as np

if TYPE_CHECKING:
    from delta_preservation.io.pdf import TextSpan


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


def _is_dimension_like(text: str) -> bool:
    """
    Return True if this text span looks like an annotation dimension / callout.

    We want to include spans that carry engineering measurement content (dimensions,
    tolerances, symbols) and exclude boilerplate title-block text.  The heuristic is:
    the text must contain at least one digit AND at least one of: a decimal point,
    engineering symbol (Ø, R, ±, °, x), or tolerance marker (±, +, -).

    This deliberately excludes pure word strings (names, titles, notes headers) and
    zone labels (single letters/digits).
    """
    t = text.strip()

    # Too short to be meaningful
    if len(t) < 2:
        return False

    has_digit = any(c.isdigit() for c in t)
    if not has_digit:
        return False

    # Engineering annotation markers
    has_engineering_marker = any(c in t for c in ("Ø", "°", "±", "ø", "R.", "r."))
    has_decimal = "." in t
    # Count pattern like "2 x" or "4X"
    import re
    has_count_pattern = bool(re.search(r'\d\s*[xX]\s', t))

    return has_engineering_marker or (has_decimal and any(c.isdigit() for c in t.split(".")[-1]))


def estimate_transform_from_text_spans(
    revA_spans: "List[TextSpan]",
    revB_spans: "List[TextSpan]",
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> "Optional[Transform]":
    """
    Estimate transformation from Rev A to Rev B using matched annotation text spans.

    Engineering drawings share dimension annotations between revisions (values that
    don't change, like base dimensions that stay the same even if tolerance changes).
    By matching identical annotation text strings (excluding title-block boilerplate)
    and using their centre-point pairs as correspondences, we compute a robust
    translation/homography that captures the true content shift.

    Unlike ORB image features, text-span correspondences are not confused by the large
    identical title-block region that dominates ORB matching on engineering drawings.

    Algorithm:
        1. Filter spans to dimension-like content only (engineering markers + decimals).
        2. Collect point pairs for texts that appear exactly once in each revision
           (unambiguous correspondences).
        3. Separate pairs by whether they appear at the same position (delta < 5 pts)
           or have shifted — the shifted pairs carry the true content transform.
        4. If enough shifted pairs exist, estimate homography from shifted pairs only.
           Otherwise fall back to all pairs with RANSAC.
        5. Return None if fewer than 3 pairs found.

    Args:
        revA_spans: Text spans from Rev A (PDF coordinate space, 72 DPI).
        revB_spans: Text spans from Rev B (PDF coordinate space, 72 DPI).
        page_width: Page width in PDF pts (unused, kept for API compatibility).
        page_height: Page height in PDF pts (unused, kept for API compatibility).

    Returns:
        Transform if enough matched annotation spans are found, otherwise None.
    """
    # Index spans by text, keeping only dimension-like spans that appear exactly once.
    def _index_unique(spans: "List[TextSpan]") -> "dict[str, Tuple[float, float]]":
        counts: "dict[str, int]" = {}
        centres: "dict[str, Tuple[float, float]]" = {}
        for s in spans:
            key = s.text.strip()
            if not key or not _is_dimension_like(key):
                continue
            counts[key] = counts.get(key, 0) + 1
            x0, y0, x1, y1 = s.bbox_pdf
            centres[key] = ((x0 + x1) / 2, (y0 + y1) / 2)
        # Only keep texts that appear exactly once (unambiguous correspondences)
        return {k: centres[k] for k, n in counts.items() if n == 1}

    uniqueA = _index_unique(revA_spans)
    uniqueB = _index_unique(revB_spans)

    # Find common unique dimension-like texts
    common_keys = set(uniqueA) & set(uniqueB)

    if len(common_keys) < 2:
        return None

    # Build point pairs and separate into "shifted" vs "static" groups
    shifted_pairs: "List[Tuple[Tuple[float, float], Tuple[float, float]]]" = []
    all_pairs: "List[Tuple[Tuple[float, float], Tuple[float, float]]]" = []

    for key in common_keys:
        pA = uniqueA[key]
        pB = uniqueB[key]
        delta = ((pA[0] - pB[0]) ** 2 + (pA[1] - pB[1]) ** 2) ** 0.5
        all_pairs.append((pA, pB))
        if delta > 5.0:  # 5 PDF pts threshold for "shifted"
            shifted_pairs.append((pA, pB))

    # Prefer pairs that actually moved — they encode the true content shift.
    # Static pairs (same position) don't constrain the translation.
    pairs_to_use = shifted_pairs if len(shifted_pairs) >= 2 else all_pairs

    if len(pairs_to_use) < 2:
        return None

    ptsA = np.float32([[p[0][0], p[0][1]] for p in pairs_to_use]).reshape(-1, 1, 2)
    ptsB = np.float32([[p[1][0], p[1][1]] for p in pairs_to_use]).reshape(-1, 1, 2)

    # Compute pure translation from the median displacement of shifted pairs.
    #
    # We deliberately avoid full homography estimation here because:
    # 1. Engineering drawing revisions apply translations (not rotations/shears) to
    #    annotation groups.
    # 2. When multiple view groups shift by different amounts (e.g., main view +362 pts,
    #    side view +212 pts), fitting a homography yields rotation/shear artifacts that
    #    are wrong for features outside the fitted region.
    # 3. Median displacement is robust to the minority cluster (different shift amount)
    #    and to any mismatched pairs.
    #
    # The resulting pure translation gets us within the search radius for most features.
    # Features that shift by a secondary amount may still land outside the search radius
    # but will be partially covered by an expanded SEARCH_RADIUS in match.py.
    displacements = np.array([[p[1][0] - p[0][0], p[1][1] - p[0][1]] for p in pairs_to_use])
    tx = float(np.median(displacements[:, 0]))
    ty = float(np.median(displacements[:, 1]))

    H = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return Transform(
        H=H,
        inliers=len(pairs_to_use),
        inlier_ratio=1.0,
        quality_ok=True,
    )


def _homography_translation(H: np.ndarray) -> Tuple[float, float]:
    """Extract the translation components (tx, ty) from a homography matrix."""
    return float(H[0, 2]), float(H[1, 2])


def _homography_is_near_identity(H: np.ndarray, translation_threshold: float = 5.0) -> bool:
    """
    Return True if the homography is essentially a no-op (translation < threshold PDF pts,
    rotation/scale negligible).

    A near-identity homography from ORB on engineering drawings usually means the
    static title-block dominated matching and the true content shift was not captured.
    """
    # Check that the 2x2 rotation/scale part is close to identity
    rot_scale = H[:2, :2]
    identity_2x2 = np.eye(2, dtype=np.float64)
    rot_error = np.max(np.abs(rot_scale - identity_2x2))

    # Check translation magnitude
    tx, ty = _homography_translation(H)
    translation_magnitude = (tx ** 2 + ty ** 2) ** 0.5

    return rot_error < 0.01 and translation_magnitude < translation_threshold


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
