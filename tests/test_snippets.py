"""
Test module for delta_preservation.vision.snippets.

Tests focus on deterministic pixel behavior:
- Exact crop dimensions after padding and clamping
- Edge clamping behavior (no negative indices or overflow)
- Invalid/degenerate bbox handling
- Filename convention compliance
- Pixel fidelity (no color conversion or artifacts)
"""

import pytest
import numpy as np
from pathlib import Path
import cv2
import tempfile
import shutil

from delta_preservation.vision.snippets import crop_with_padding, save_snippet


class TestCropWithPadding:
    """Test crop_with_padding for deterministic pixel behavior."""
    
    def test_exact_dimensions_with_padding_no_clamping(self):
        """
        Test (1): Given synthetic page with known dimensions, valid bbox, and fixed pad_px,
        assert output crop has exact expected width and height after expansion (no clamping).
        """
        # Create synthetic 1000×800 image (H=800, W=1000)
        img = np.zeros((800, 1000, 3), dtype=np.uint8)
        
        # Define bbox well within bounds: (100, 100, 200, 150)
        # Width = 100, Height = 50
        bbox = (100, 100, 200, 150)
        pad_px = 20
        
        # Expected padded bbox: (80, 80, 220, 170)
        # Expected dimensions: width = 220-80 = 140, height = 170-80 = 90
        crop = crop_with_padding(img, bbox, pad_px)
        
        assert crop.shape[0] == 90, f"Expected height 90, got {crop.shape[0]}"
        assert crop.shape[1] == 140, f"Expected width 140, got {crop.shape[1]}"
        assert crop.shape[2] == 3, f"Expected 3 channels, got {crop.shape[2]}"
    
    def test_clamping_top_left_edge(self):
        """
        Test (2a): Verify clamping at top-left edge.
        Bbox near origin should clamp to (0, 0) without negative indices.
        """
        # Create 1000×800 image
        img = np.zeros((800, 1000, 3), dtype=np.uint8)
        
        # Bbox near top-left: (10, 10, 60, 60)
        bbox = (10, 10, 60, 60)
        pad_px = 30
        
        # Padded would be: (-20, -20, 90, 90)
        # Clamped should be: (0, 0, 90, 90)
        # Expected dimensions: width = 90, height = 90
        crop = crop_with_padding(img, bbox, pad_px)
        
        assert crop.shape[0] == 90, f"Expected height 90, got {crop.shape[0]}"
        assert crop.shape[1] == 90, f"Expected width 90, got {crop.shape[1]}"
        
        # Verify no exception was raised (implicit by reaching this point)
    
    def test_clamping_bottom_right_edge(self):
        """
        Test (2b): Verify clamping at bottom-right edge.
        Bbox near bottom-right should clamp to image bounds without overflow.
        """
        # Create 1000×800 image (H=800, W=1000)
        img = np.zeros((800, 1000, 3), dtype=np.uint8)
        
        # Bbox near bottom-right: (940, 740, 990, 790)
        bbox = (940, 740, 990, 790)
        pad_px = 30
        
        # Padded would be: (910, 710, 1020, 820)
        # Clamped should be: (910, 710, 1000, 800)
        # Expected dimensions: width = 1000-910 = 90, height = 800-710 = 90
        crop = crop_with_padding(img, bbox, pad_px)
        
        assert crop.shape[0] == 90, f"Expected height 90, got {crop.shape[0]}"
        assert crop.shape[1] == 90, f"Expected width 90, got {crop.shape[1]}"
    
    def test_clamping_all_edges(self):
        """
        Test (2c): Verify clamping when bbox extends beyond all edges.
        """
        # Create small 100×100 image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Bbox that covers entire image with padding extending beyond
        bbox = (20, 20, 80, 80)
        pad_px = 50
        
        # Padded would be: (-30, -30, 130, 130)
        # Clamped should be: (0, 0, 100, 100)
        # Expected dimensions: width = 100, height = 100
        crop = crop_with_padding(img, bbox, pad_px)
        
        assert crop.shape[0] == 100, f"Expected height 100, got {crop.shape[0]}"
        assert crop.shape[1] == 100, f"Expected width 100, got {crop.shape[1]}"
    
    def test_invalid_bbox_x_coordinates(self):
        """
        Test (3a): Assert invalid bbox with x1 <= x0 raises clear exception.
        """
        img = np.zeros((800, 1000, 3), dtype=np.uint8)
        
        # Invalid: x1 <= x0
        bbox = (200, 100, 200, 150)
        pad_px = 10
        
        with pytest.raises(ValueError, match="Invalid bbox.*x1 must be > x0"):
            crop_with_padding(img, bbox, pad_px)
    
    def test_invalid_bbox_y_coordinates(self):
        """
        Test (3b): Assert invalid bbox with y1 <= y0 raises clear exception.
        """
        img = np.zeros((800, 1000, 3), dtype=np.uint8)
        
        # Invalid: y1 <= y0
        bbox = (100, 150, 200, 150)
        pad_px = 10
        
        with pytest.raises(ValueError, match="Invalid bbox.*y1 must be > y0"):
            crop_with_padding(img, bbox, pad_px)
    
    def test_degenerate_bbox_after_clamping(self):
        """
        Test (3c): Assert degenerate bbox (zero/negative area after clamping) raises exception.
        """
        # Create 1000×800 image
        img = np.zeros((800, 1000, 3), dtype=np.uint8)
        
        # Bbox with very small dimensions that becomes degenerate after clamping
        # Place bbox at (0, 0, 1, 1) with negative padding
        bbox = (0, 0, 1, 1)
        pad_px = -5  # This will cause clamped bbox to be invalid
        
        # Padded would be: (5, 5, -4, -4)
        # After clamping: (5, 5, 0, 0) which is invalid (x1 <= x0)
        with pytest.raises(ValueError, match="Clamped bbox is empty"):
            crop_with_padding(img, bbox, pad_px)
    
    def test_pixel_fidelity_no_color_conversion(self):
        """
        Test (5): Ensure pixel fidelity by embedding known colored rectangle
        and asserting cropped image contains that color unchanged.
        """
        # Create 1000×800 black image
        img = np.zeros((800, 1000, 3), dtype=np.uint8)
        
        # Define bbox: (100, 100, 200, 150)
        bbox = (100, 100, 200, 150)
        
        # Fill bbox region with known color: BGR (255, 128, 64) - bright blue-ish
        img[100:150, 100:200] = [255, 128, 64]
        
        # Crop with no padding to get exact region
        pad_px = 0
        crop = crop_with_padding(img, bbox, pad_px)
        
        # Assert all pixels in crop match the known color
        expected_color = np.array([255, 128, 64], dtype=np.uint8)
        assert crop.shape == (50, 100, 3), f"Expected shape (50, 100, 3), got {crop.shape}"
        
        # Check every pixel matches
        assert np.all(crop == expected_color), "Cropped pixels do not match expected color"
    
    def test_pixel_fidelity_with_padding(self):
        """
        Test (5b): Verify pixel fidelity with padding - colored region should be preserved.
        """
        # Create 1000×800 image with gray background
        img = np.full((800, 1000, 3), 50, dtype=np.uint8)
        
        # Define bbox and fill with distinct color
        bbox = (200, 200, 300, 250)
        img[200:250, 200:300] = [0, 255, 0]  # Green
        
        # Crop with padding
        pad_px = 10
        crop = crop_with_padding(img, bbox, pad_px)
        
        # Expected dimensions: (250+10)-(200-10) = 70 height, (300+10)-(200-10) = 120 width
        assert crop.shape == (70, 120, 3), f"Expected shape (70, 120, 3), got {crop.shape}"
        
        # The center region (after accounting for 10px padding) should be green
        # Center region in crop: [10:60, 10:110] (original bbox region)
        center = crop[10:60, 10:110]
        expected_green = np.array([0, 255, 0], dtype=np.uint8)
        assert np.all(center == expected_green), "Center region color not preserved"
        
        # Padding region should be gray background
        expected_gray = np.array([50, 50, 50], dtype=np.uint8)
        # Check top padding row
        assert np.all(crop[0, :] == expected_gray), "Top padding color not preserved"


class TestSaveSnippet:
    """Test save_snippet for filename conventions and file I/O."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_filename_convention_basic(self, temp_dir):
        """
        Test (4a): Confirm saved filename follows exact naming convention.
        Format: char_{char_no:03d}_{rev_label}_p{page_index}.png
        """
        # Create simple test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Save with specific parameters
        filename = save_snippet(img, temp_dir, char_no=12, rev_label="revA", page_index=1)
        
        # Assert exact filename format
        assert filename == "char_012_revA_p1.png", f"Expected 'char_012_revA_p1.png', got '{filename}'"
        
        # Verify file exists on disk
        expected_path = temp_dir / filename
        assert expected_path.exists(), f"File not found at {expected_path}"
    
    def test_filename_convention_various_inputs(self, temp_dir):
        """
        Test (4b): Test filename convention with various char numbers and labels.
        """
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        
        test_cases = [
            (0, "revA", 0, "char_000_revA_p0.png"),
            (1, "revB", 2, "char_001_revB_p2.png"),
            (999, "revC", 10, "char_999_revC_p10.png"),
            (42, "baseline", 5, "char_042_baseline_p5.png"),
        ]
        
        for char_no, rev_label, page_index, expected_filename in test_cases:
            filename = save_snippet(img, temp_dir, char_no, rev_label, page_index)
            assert filename == expected_filename, f"Expected '{expected_filename}', got '{filename}'"
            assert (temp_dir / filename).exists(), f"File not found: {filename}"
    
    def test_returned_path_matches_written_file(self, temp_dir):
        """
        Test (4c): Assert returned path matches what was written to disk.
        """
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = [100, 150, 200]  # Add some content
        
        filename = save_snippet(img, temp_dir, char_no=5, rev_label="revA", page_index=1)
        
        # Construct expected path
        expected_path = temp_dir / filename
        
        # Verify file exists
        assert expected_path.exists(), f"File not found at {expected_path}"
        
        # Read back and verify it's the same image
        loaded_img = cv2.imread(str(expected_path))
        assert loaded_img is not None, "Failed to load saved image"
        assert np.array_equal(loaded_img, img), "Loaded image does not match saved image"
    
    def test_empty_image_raises_exception(self, temp_dir):
        """
        Test (3d): Assert empty image raises clear exception.
        """
        # Create empty image
        img = np.array([], dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Cannot save empty image"):
            save_snippet(img, temp_dir, char_no=1, rev_label="revA", page_index=1)
    
    def test_directory_creation(self, temp_dir):
        """
        Test that save_snippet creates nested directories if they don't exist.
        """
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        
        # Use nested directory that doesn't exist
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        
        filename = save_snippet(img, nested_dir, char_no=1, rev_label="revA", page_index=1)
        
        # Verify directory was created
        assert nested_dir.exists(), "Nested directory was not created"
        assert (nested_dir / filename).exists(), "File not found in nested directory"
    
    def test_pixel_fidelity_after_save_load(self, temp_dir):
        """
        Test (5c): Verify pixel fidelity through save/load cycle.
        Ensure no color conversion or artifacts introduced during I/O.
        """
        # Create image with specific color pattern
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create distinct colored regions
        img[0:50, 0:50] = [255, 0, 0]      # Blue (BGR)
        img[0:50, 50:100] = [0, 255, 0]    # Green
        img[50:100, 0:50] = [0, 0, 255]    # Red
        img[50:100, 50:100] = [128, 128, 128]  # Gray
        
        # Save
        filename = save_snippet(img, temp_dir, char_no=99, rev_label="test", page_index=0)
        
        # Load back
        loaded_img = cv2.imread(str(temp_dir / filename))
        
        # Assert exact pixel match
        assert loaded_img.shape == img.shape, f"Shape mismatch: {loaded_img.shape} vs {img.shape}"
        assert np.array_equal(loaded_img, img), "Pixel values changed after save/load cycle"
        
        # Verify specific regions
        assert np.all(loaded_img[0:50, 0:50] == [255, 0, 0]), "Blue region corrupted"
        assert np.all(loaded_img[0:50, 50:100] == [0, 255, 0]), "Green region corrupted"
        assert np.all(loaded_img[50:100, 0:50] == [0, 0, 255]), "Red region corrupted"
        assert np.all(loaded_img[50:100, 50:100] == [128, 128, 128]), "Gray region corrupted"


class TestIntegration:
    """Integration tests combining crop_with_padding and save_snippet."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_crop_and_save_pipeline(self, temp_dir):
        """
        Test full pipeline: create synthetic page, crop with padding, save, and verify.
        """
        # Create synthetic 1000×800 page with colored characteristic region
        page_img = np.full((800, 1000, 3), 200, dtype=np.uint8)  # Light gray background
        
        # Define characteristic bbox and fill with distinct color
        char_bbox = (400, 300, 500, 350)
        page_img[300:350, 400:500] = [0, 165, 255]  # Orange (BGR)
        
        # Crop with padding
        pad_px = 15
        crop = crop_with_padding(page_img, char_bbox, pad_px)
        
        # Expected dimensions: height = (350+15)-(300-15) = 80, width = (500+15)-(400-15) = 130
        assert crop.shape == (80, 130, 3), f"Expected (80, 130, 3), got {crop.shape}"
        
        # Save the crop
        filename = save_snippet(crop, temp_dir, char_no=17, rev_label="revA", page_index=1)
        
        # Verify filename
        assert filename == "char_017_revA_p1.png"
        
        # Load and verify pixel fidelity
        loaded = cv2.imread(str(temp_dir / filename))
        assert np.array_equal(loaded, crop), "Saved crop does not match original"
        
        # Verify the characteristic region is preserved in center
        # Center region in crop: [15:65, 15:115] (original bbox after padding offset)
        center_region = loaded[15:65, 15:115]
        expected_orange = np.array([0, 165, 255], dtype=np.uint8)
        assert np.all(center_region == expected_orange), "Characteristic color not preserved"
