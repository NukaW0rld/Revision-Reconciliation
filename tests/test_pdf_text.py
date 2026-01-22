import pytest
import numpy as np
from pathlib import Path
import fitz

from delta_preservation.io.pdf import (
    render_page,
    extract_text_spans,
    pdf_to_img_coords,
    TextSpan,
)


FIXTURE_PDF = Path(__file__).parent.parent / "assets" / "part1" / "revA.pdf"


class TestRenderPage:
    def test_raises_filenotfounderror_for_missing_path(self):
        missing_path = Path("/nonexistent/path/to/file.pdf")
        with pytest.raises(FileNotFoundError):
            render_page(missing_path, 0)

    def test_raises_indexerror_for_out_of_range_page_index(self):
        with pytest.raises(IndexError):
            render_page(FIXTURE_PDF, 999)
        
        with pytest.raises(IndexError):
            render_page(FIXTURE_PDF, -1)

    def test_returns_valid_numpy_array_for_real_pdf_page(self):
        img = render_page(FIXTURE_PDF, 0, dpi=150)
        
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.shape[0] > 0
        assert img.shape[1] > 0

    def test_pixel_dimensions_scale_correctly_with_dpi(self):
        img_72 = render_page(FIXTURE_PDF, 0, dpi=72)
        img_144 = render_page(FIXTURE_PDF, 0, dpi=144)
        
        height_72, width_72 = img_72.shape[:2]
        height_144, width_144 = img_144.shape[:2]
        
        width_ratio = width_144 / width_72
        height_ratio = height_144 / height_72
        
        tolerance = 0.05
        assert abs(width_ratio - 2.0) < tolerance
        assert abs(height_ratio - 2.0) < tolerance


class TestExtractTextSpans:
    def test_raises_filenotfounderror_for_missing_path(self):
        missing_path = Path("/nonexistent/path/to/file.pdf")
        with pytest.raises(FileNotFoundError):
            extract_text_spans(missing_path, 0)

    def test_raises_indexerror_for_out_of_range_page_index(self):
        with pytest.raises(IndexError):
            extract_text_spans(FIXTURE_PDF, 999)
        
        with pytest.raises(IndexError):
            extract_text_spans(FIXTURE_PDF, -1)

    def test_returns_list_of_textspan_with_correct_types(self):
        spans = extract_text_spans(FIXTURE_PDF, 0)
        
        assert isinstance(spans, list)
        assert len(spans) > 0
        
        for span in spans:
            assert isinstance(span, TextSpan)
            assert isinstance(span.text, str)
            assert isinstance(span.bbox_pdf, tuple)
            assert len(span.bbox_pdf) == 4
            assert all(isinstance(coord, float) for coord in span.bbox_pdf)
            assert isinstance(span.font_size, float)
            assert isinstance(span.block_id, int)
            assert isinstance(span.line_id, int)
            assert isinstance(span.span_id, int)

    def test_returns_nontrivial_spans_with_expected_content(self):
        spans = extract_text_spans(FIXTURE_PDF, 0)
        
        assert len(spans) > 10
        
        all_text = " ".join(span.text for span in spans)
        
        assert any(
            token in all_text.upper()
            for token in ["PART", "REV", "DRAWING", "TITLE", "SHEET", "DATE"]
        )


class TestPdfToImgCoords:
    def test_preserves_geometry_under_scaling(self):
        doc = fitz.open(FIXTURE_PDF)
        page = doc.load_page(0)
        
        original_bbox = (100.0, 150.0, 300.0, 250.0)
        dpi = 144
        
        pixel_coords = pdf_to_img_coords(original_bbox, page, dpi)
        x0_px, y0_px, x1_px, y1_px = pixel_coords
        
        scale = dpi / 72.0
        recovered_x0 = x0_px / scale
        recovered_y0 = y0_px / scale
        recovered_x1 = x1_px / scale
        recovered_y1 = y1_px / scale
        
        tolerance = 1.0
        assert abs(recovered_x0 - original_bbox[0]) <= tolerance
        assert abs(recovered_y0 - original_bbox[1]) <= tolerance
        assert abs(recovered_x1 - original_bbox[2]) <= tolerance
        assert abs(recovered_y1 - original_bbox[3]) <= tolerance
        
        doc.close()

    def test_produces_valid_bbox_for_cropping(self):
        doc = fitz.open(FIXTURE_PDF)
        page = doc.load_page(0)
        dpi = 150
        
        img = render_page(FIXTURE_PDF, 0, dpi=dpi)
        img_height, img_width = img.shape[:2]
        
        bbox_pdf = (50.0, 50.0, 200.0, 200.0)
        x0, y0, x1, y1 = pdf_to_img_coords(bbox_pdf, page, dpi)
        
        assert x0 < x1
        assert y0 < y1
        
        x0_clamped = max(0, min(x0, img_width))
        y0_clamped = max(0, min(y0, img_height))
        x1_clamped = max(0, min(x1, img_width))
        y1_clamped = max(0, min(y1, img_height))
        
        assert 0 <= x0_clamped <= img_width
        assert 0 <= y0_clamped <= img_height
        assert 0 <= x1_clamped <= img_width
        assert 0 <= y1_clamped <= img_height
        
        doc.close()


class TestEndToEndMappingSanity:
    def test_span_bbox_to_image_crop_contains_ink(self):
        dpi = 150
        
        spans = extract_text_spans(FIXTURE_PDF, 0)
        assert len(spans) > 0
        
        non_empty_span = None
        for span in spans:
            if len(span.text.strip()) > 0:
                x0, y0, x1, y1 = span.bbox_pdf
                if x1 - x0 > 5 and y1 - y0 > 5:
                    non_empty_span = span
                    break
        
        assert non_empty_span is not None, "No suitable text span found"
        
        img = render_page(FIXTURE_PDF, 0, dpi=dpi)
        
        doc = fitz.open(FIXTURE_PDF)
        page = doc.load_page(0)
        
        x0, y0, x1, y1 = pdf_to_img_coords(non_empty_span.bbox_pdf, page, dpi)
        doc.close()
        
        img_height, img_width = img.shape[:2]
        x0 = max(0, min(x0, img_width - 1))
        y0 = max(0, min(y0, img_height - 1))
        x1 = max(x0 + 1, min(x1, img_width))
        y1 = max(y0 + 1, min(y1, img_height))
        
        crop = img[y0:y1, x0:x1]
        
        assert crop.size > 0, "Crop is empty"
        assert crop.shape[0] > 0 and crop.shape[1] > 0
        
        mean_intensity = crop.mean()
        
        white_patch = np.full((10, 10, 3), 255, dtype=np.uint8)
        white_mean = white_patch.mean()
        
        intensity_diff = abs(mean_intensity - white_mean)
        assert intensity_diff > 5, f"Crop appears blank (mean={mean_intensity:.1f})"
