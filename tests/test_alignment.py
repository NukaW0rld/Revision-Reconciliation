"""
Deterministic, outcome-focused tests for alignment module.

Tests validate transform recovery, identity mapping, failure signaling,
and bbox correctness using synthetic fixtures with known transformations.
"""

import numpy as np
import cv2
import pytest
from delta_preservation.vision.alignment import (
    estimate_transform,
    apply_transform_bbox,
    Transform,
    AlignmentError,
)


def create_test_image(width: int = 800, height: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Create a synthetic document-like image with text-like features.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        seed: Random seed for reproducibility
    
    Returns:
        BGR image with synthetic text-like patterns
    """
    np.random.seed(seed)
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add horizontal lines (simulating text lines)
    for y in range(100, height - 100, 40):
        x_start = 100
        x_end = width - 100
        cv2.rectangle(img, (x_start, y), (x_end, y + 15), (0, 0, 0), -1)
    
    # Add vertical structure (simulating margins/columns)
    cv2.rectangle(img, (80, 80), (90, height - 80), (0, 0, 0), -1)
    cv2.rectangle(img, (width - 90, 80), (width - 80, height - 80), (0, 0, 0), -1)
    
    # Add some distinctive corner markers for feature matching
    cv2.circle(img, (150, 150), 20, (0, 0, 0), -1)
    cv2.circle(img, (width - 150, 150), 20, (0, 0, 0), -1)
    cv2.circle(img, (150, height - 150), 20, (0, 0, 0), -1)
    cv2.circle(img, (width - 150, height - 150), 20, (0, 0, 0), -1)
    
    # Add some random text-like blobs for feature richness
    for _ in range(50):
        x = np.random.randint(120, width - 120)
        y = np.random.randint(120, height - 120)
        w = np.random.randint(10, 30)
        h = np.random.randint(5, 15)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
    
    return img


def apply_known_transform(img: np.ndarray, tx: float, ty: float, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a known translation and scale transform to an image.
    
    Args:
        img: Source image
        tx: Translation in x (pixels)
        ty: Translation in y (pixels)
        scale: Scale factor
    
    Returns:
        Tuple of (transformed_image, ground_truth_homography)
    """
    height, width = img.shape[:2]
    
    # Create homography matrix for translation + scale
    # H = [[scale, 0, tx], [0, scale, ty], [0, 0, 1]]
    H = np.array([
        [scale, 0, tx],
        [0, scale, ty],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Apply transform
    transformed = cv2.warpPerspective(img, H, (width, height), borderValue=(255, 255, 255))
    
    return transformed, H


class TestTransformRecovery:
    """Test successful transform recovery with known transformations."""
    
    def test_translation_only(self):
        """Test recovery of pure translation transform."""
        imgA = create_test_image(seed=42)
        imgB, H_gt = apply_known_transform(imgA, tx=50, ty=30, scale=1.0)
        
        result = estimate_transform(imgA, imgB)
        
        # Validate transform object structure
        assert isinstance(result, Transform)
        assert result.H is not None
        assert result.H.shape == (3, 3)
        assert result.quality_ok is True
        assert result.inliers >= 40
        assert result.inlier_ratio >= 0.15
    
    def test_translation_with_scale(self):
        """Test recovery of translation + scale transform."""
        imgA = create_test_image(seed=42)
        imgB, H_gt = apply_known_transform(imgA, tx=20, ty=-15, scale=1.05)
        
        result = estimate_transform(imgA, imgB)
        
        assert result.quality_ok is True
        assert result.inliers >= 40
        assert result.inlier_ratio >= 0.15
    
    def test_negative_translation(self):
        """Test recovery with negative translation."""
        imgA = create_test_image(seed=42)
        imgB, H_gt = apply_known_transform(imgA, tx=-30, ty=-40, scale=1.0)
        
        result = estimate_transform(imgA, imgB)
        
        assert result.quality_ok is True
        assert result.inliers >= 40
        assert result.inlier_ratio >= 0.15
    
    def test_small_scale_change(self):
        """Test recovery with small scale variation."""
        imgA = create_test_image(seed=42)
        imgB, H_gt = apply_known_transform(imgA, tx=10, ty=10, scale=0.98)
        
        result = estimate_transform(imgA, imgB)
        
        assert result.quality_ok is True
        assert result.inliers >= 40


class TestIdentityMapping:
    """Test transform accuracy using fixed reference points."""
    
    def get_reference_points(self) -> np.ndarray:
        """
        Define fixed reference points corresponding to known features.
        
        Returns:
            Nx2 array of (x, y) coordinates in Rev A
        """
        # Points at known locations from create_test_image:
        # - Corner circles at (150, 150), (650, 150), (150, 850), (650, 850)
        # - Margin lines at x=85, x=715
        # - Center points of text lines
        return np.array([
            [150, 150],   # Top-left corner circle
            [650, 150],   # Top-right corner circle (width=800, so 800-150=650)
            [150, 850],   # Bottom-left corner circle (height=1000, so 1000-150=850)
            [650, 850],   # Bottom-right corner circle
            [400, 500],   # Center of image
            [85, 500],    # Left margin center
            [715, 500],   # Right margin center (800-85=715)
            [400, 200],   # Upper center
            [400, 800],   # Lower center
        ], dtype=np.float32)
    
    def transform_points(self, points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Transform points using homography.
        
        Args:
            points: Nx2 array of points
            H: 3x3 homography matrix
        
        Returns:
            Nx2 array of transformed points
        """
        points_reshaped = points.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points_reshaped, H)
        return transformed.reshape(-1, 2)
    
    def test_translation_point_accuracy(self):
        """Test point transformation accuracy for pure translation."""
        imgA = create_test_image(seed=42)
        tx, ty = 50.0, 30.0
        imgB, H_gt = apply_known_transform(imgA, tx=tx, ty=ty, scale=1.0)
        
        result = estimate_transform(imgA, imgB)
        
        # Get reference points and expected transformed locations
        ref_points = self.get_reference_points()
        expected_points = ref_points + np.array([tx, ty])
        
        # Transform using estimated homography
        transformed_points = self.transform_points(ref_points, result.H)
        
        # Calculate distances
        distances = np.linalg.norm(transformed_points - expected_points, axis=1)
        
        # Assert all points are within tolerance (20 pixels at 300 dpi ~ 1.7mm)
        max_distance = np.max(distances)
        mean_distance = np.mean(distances)
        
        assert max_distance < 20.0, f"Max distance {max_distance:.2f} exceeds 20px tolerance"
        assert mean_distance < 10.0, f"Mean distance {mean_distance:.2f} exceeds 10px tolerance"
    
    def test_scale_point_accuracy(self):
        """Test point transformation accuracy with scale change."""
        imgA = create_test_image(seed=42)
        tx, ty, scale = 20.0, -15.0, 1.05
        imgB, H_gt = apply_known_transform(imgA, tx=tx, ty=ty, scale=scale)
        
        result = estimate_transform(imgA, imgB)
        
        # Get reference points and expected transformed locations
        ref_points = self.get_reference_points()
        expected_points = ref_points * scale + np.array([tx, ty])
        
        # Transform using estimated homography
        transformed_points = self.transform_points(ref_points, result.H)
        
        # Calculate distances
        distances = np.linalg.norm(transformed_points - expected_points, axis=1)
        max_distance = np.max(distances)
        
        # With scale, allow slightly larger tolerance
        assert max_distance < 25.0, f"Max distance {max_distance:.2f} exceeds 25px tolerance"
    
    def test_negative_translation_point_accuracy(self):
        """Test point transformation accuracy with negative translation."""
        imgA = create_test_image(seed=42)
        tx, ty = -30.0, -40.0
        imgB, H_gt = apply_known_transform(imgA, tx=tx, ty=ty, scale=1.0)
        
        result = estimate_transform(imgA, imgB)
        
        ref_points = self.get_reference_points()
        expected_points = ref_points + np.array([tx, ty])
        transformed_points = self.transform_points(ref_points, result.H)
        
        distances = np.linalg.norm(transformed_points - expected_points, axis=1)
        max_distance = np.max(distances)
        
        assert max_distance < 20.0, f"Max distance {max_distance:.2f} exceeds 20px tolerance"


class TestFailureSignaling:
    """Test that alignment properly fails on mismatched or invalid inputs."""
    
    def test_different_content_images(self):
        """Test failure when images have completely different content."""
        imgA = create_test_image(seed=42)
        
        # Create completely different content (vertical lines instead of horizontal)
        imgB = np.ones((1000, 800, 3), dtype=np.uint8) * 255
        for x in range(100, 700, 40):
            cv2.rectangle(imgB, (x, 100), (x + 15, 900), (0, 0, 0), -1)
        
        with pytest.raises(AlignmentError) as exc_info:
            estimate_transform(imgA, imgB)
        
        # Should fail with quality or matching error
        assert "quality too low" in str(exc_info.value).lower() or \
               "insufficient matches" in str(exc_info.value).lower() or \
               "failed" in str(exc_info.value).lower()
    
    def test_random_noise_images(self):
        """Test failure when one image is random noise."""
        imgA = create_test_image(seed=42)
        
        # Create random noise image
        np.random.seed(123)
        imgB = np.random.randint(0, 256, (1000, 800, 3), dtype=np.uint8)
        
        with pytest.raises(AlignmentError) as exc_info:
            estimate_transform(imgA, imgB)
        
        # Accept any alignment failure message
        assert "quality too low" in str(exc_info.value).lower() or \
               "failed" in str(exc_info.value).lower() or \
               "insufficient" in str(exc_info.value).lower()
    
    def test_blank_image(self):
        """Test failure when one image is blank."""
        imgA = create_test_image(seed=42)
        imgB = np.ones((1000, 800, 3), dtype=np.uint8) * 255  # Blank white
        
        with pytest.raises(AlignmentError) as exc_info:
            estimate_transform(imgA, imgB)
        
        assert "failed" in str(exc_info.value).lower()
    
    def test_extreme_transformation(self):
        """Test failure when transformation is too extreme to recover."""
        imgA = create_test_image(seed=42)
        
        # Apply extreme rotation + scale that should break alignment
        height, width = imgA.shape[:2]
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, 90, 0.3)  # 90° rotation + 0.3x scale
        imgB = cv2.warpAffine(imgA, M, (width, height), borderValue=(255, 255, 255))
        
        # This should either raise AlignmentError or return quality_ok=False
        # Since the implementation raises on quality_ok=False, we expect an exception
        with pytest.raises(AlignmentError) as exc_info:
            estimate_transform(imgA, imgB)
        
        assert "quality too low" in str(exc_info.value).lower() or \
               "failed" in str(exc_info.value).lower()


class TestBboxCorrectness:
    """Test bbox transformation correctness."""
    
    def test_bbox_encloses_transformed_corners(self):
        """Test that transformed bbox encloses all transformed corners."""
        # Define a test bbox
        bbox = (200.0, 300.0, 400.0, 500.0)  # x0, y0, x1, y1
        
        # Create a simple translation homography
        H = np.array([
            [1.0, 0.0, 50.0],
            [0.0, 1.0, 30.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Transform bbox
        result_bbox = apply_transform_bbox(bbox, H)
        x0, y0, x1, y1 = result_bbox
        
        # Manually transform corners
        corners = np.array([
            [bbox[0], bbox[1]],  # Top-left
            [bbox[2], bbox[1]],  # Top-right
            [bbox[2], bbox[3]],  # Bottom-right
            [bbox[0], bbox[3]],  # Bottom-left
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        transformed_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        
        # Check that result bbox encloses all corners
        for corner in transformed_corners:
            cx, cy = corner
            assert x0 <= cx <= x1, f"Corner x={cx} not in [{x0}, {x1}]"
            assert y0 <= cy <= y1, f"Corner y={cy} not in [{y0}, {y1}]"
    
    def test_bbox_min_max_logic(self):
        """Test that bbox uses correct min/max logic for corners."""
        bbox = (100.0, 200.0, 300.0, 400.0)
        
        # Create homography with scale
        H = np.array([
            [1.2, 0.0, 10.0],
            [0.0, 1.2, 20.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        result_bbox = apply_transform_bbox(bbox, H)
        x0, y0, x1, y1 = result_bbox
        
        # Expected transformation
        corners = np.array([
            [100.0, 200.0],
            [300.0, 200.0],
            [300.0, 400.0],
            [100.0, 400.0],
        ], dtype=np.float32)
        
        # Apply H manually
        expected_corners = []
        for cx, cy in corners:
            new_x = 1.2 * cx + 10.0
            new_y = 1.2 * cy + 20.0
            expected_corners.append([new_x, new_y])
        
        expected_corners = np.array(expected_corners)
        expected_x0 = np.min(expected_corners[:, 0])
        expected_y0 = np.min(expected_corners[:, 1])
        expected_x1 = np.max(expected_corners[:, 0])
        expected_y1 = np.max(expected_corners[:, 1])
        
        assert abs(x0 - expected_x0) < 1e-3
        assert abs(y0 - expected_y0) < 1e-3
        assert abs(x1 - expected_x1) < 1e-3
        assert abs(y1 - expected_y1) < 1e-3
    
    def test_bbox_non_negative_area(self):
        """Test that transformed bbox has non-negative area."""
        bbox = (150.0, 250.0, 350.0, 450.0)
        
        H = np.array([
            [0.9, 0.0, -20.0],
            [0.0, 0.9, -30.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        result_bbox = apply_transform_bbox(bbox, H)
        x0, y0, x1, y1 = result_bbox
        
        # Check valid bbox
        assert x1 > x0, f"Invalid bbox: x1={x1} <= x0={x0}"
        assert y1 > y0, f"Invalid bbox: y1={y1} <= y0={y0}"
        
        # Check area is positive
        area = (x1 - x0) * (y1 - y0)
        assert area > 0, f"Bbox area {area} is not positive"
    
    def test_bbox_with_estimated_transform(self):
        """Test bbox transformation with real estimated transform."""
        imgA = create_test_image(seed=42)
        tx, ty = 50.0, 30.0
        imgB, H_gt = apply_known_transform(imgA, tx=tx, ty=ty, scale=1.0)
        
        result = estimate_transform(imgA, imgB)
        
        # Define a bbox in Rev A
        bbox_a = (200.0, 300.0, 400.0, 500.0)
        
        # Transform to Rev B
        bbox_b = apply_transform_bbox(bbox_a, result.H)
        x0, y0, x1, y1 = bbox_b
        
        # Expected bbox (approximately)
        expected_x0 = bbox_a[0] + tx
        expected_y0 = bbox_a[1] + ty
        expected_x1 = bbox_a[2] + tx
        expected_y1 = bbox_a[3] + ty
        
        # Allow some tolerance due to estimation error
        assert abs(x0 - expected_x0) < 20.0
        assert abs(y0 - expected_y0) < 20.0
        assert abs(x1 - expected_x1) < 20.0
        assert abs(y1 - expected_y1) < 20.0
        
        # Ensure valid bbox
        assert x1 > x0
        assert y1 > y0
