# Testing Patterns

**Analysis Date:** 2026-02-25

## Test Framework

**Runner:**
- pytest 8.x (declared in `pyproject.toml` as `pytest>=8,<9`)
- No pytest configuration file present (`pytest.ini`, `setup.cfg` with pytest section, or `pyproject.toml [tool.pytest]` not found)
- Default pytest behavior: discovers test files matching `test_*.py` or `*_test.py`

**Assertion Library:**
- Standard Python `assert` statements (no external assertion library imported)
- Will use pytest's native assertion introspection for failure messages

**Run Commands:**
```bash
# Run all tests (pytest discovers in current directory)
pytest

# Watch mode (requires pytest-watch plugin, not configured)
# Not set up - would require: pip install pytest-watch && ptw

# Coverage (requires pytest-cov, not configured)
# Not set up - would require: pip install pytest-cov && pytest --cov=delta_preservation
```

## Test File Organization

**Location:**
- No test files found in codebase (search for `test_*.py`, `*_test.py`, `*_spec.py` returned no results)
- `.pytest_cache/` directory exists indicating pytest has been run, but no test files present
- Implication: Testing is not yet implemented for this prototype pipeline

**Naming Convention (for future tests):**
- Should follow pytest discovery: `test_[module_name].py` for test modules
- Test functions: `test_[functionality_under_test]()`
- Test classes: `Test[ClassName]` (pytest convention)

**Structure (if tests existed):**
```
# Proposed structure (not currently implemented):
tests/
├── test_io/
│   ├── test_pdf.py          # Tests for extract_text_spans, render_page, pdf_to_img_coords
│   └── test_xlsx.py         # Tests for load_form3, parse_tolerance
├── test_vision/
│   ├── test_balloons.py     # Tests for detect_balloons, digit recognition
│   ├── test_alignment.py    # Tests for estimate_transform
│   └── test_snippets.py     # Tests for crop_with_padding, save_snippet
├── test_reconcile/
│   ├── test_normalize.py    # Tests for parse_requirement, MatchFingerprint
│   ├── test_anchors.py      # Tests for build_revA_anchors
│   ├── test_match.py        # Tests for generate_candidates, assign_matches
│   └── test_classify.py     # Tests for classify_delta, detect_added_characteristics
└── test_cli.py              # Integration tests for run_pipeline
```

## Test Structure (Proposed Based on Code Patterns)

**Suite Organization:**
```python
# Example: tests/test_reconcile/test_normalize.py
import pytest
from delta_preservation.reconcile.normalize import parse_requirement, MatchFingerprint

class TestParseRequirement:
    """Test requirement string parsing into fingerprints."""

    def test_parse_simple_diameter(self):
        """Parse simple diameter requirement."""
        fp = parse_requirement("Ø6.0 +0.1/-0.0")
        assert fp.pattern_class == "hole"
        assert "Ø" in fp.symbol_tokens
        assert (6.0, "6.0") in fp.numeric_tokens

    def test_parse_count_pattern(self):
        """Extract count tokens from '2X' patterns."""
        fp = parse_requirement("2X Ø5.0 ± 0.1")
        assert "2X" in fp.count_tokens
        assert len(fp.count_tokens) == 1

    def test_parse_notes_block(self):
        """Identify notes-type requirements."""
        fp = parse_requirement("DRAWING NOTES - Break all sharp edges")
        assert fp.pattern_class == "note"
```

**Patterns (inferred from code organization):**
- Class-based test suites for related functionality
- One test class per function/feature
- Descriptive test names with `test_` prefix describing the behavior
- Docstrings explaining what is being tested

## Mocking

**Framework:** No mocking library imported (would need `unittest.mock` from standard library or `pytest-mock`)

**Patterns (inferred needs based on code):**
```python
# Example mocking pattern for PDF operations
from unittest.mock import Mock, patch, MagicMock
import pytest

@patch('delta_preservation.io.pdf.fitz.open')
def test_extract_text_spans_missing_file(mock_fitz_open):
    """Verify FileNotFoundError when PDF doesn't exist."""
    from pathlib import Path
    mock_fitz_open.side_effect = FileNotFoundError("Test file")

    with pytest.raises(FileNotFoundError):
        extract_text_spans(Path("nonexistent.pdf"), 0)

@patch('delta_preservation.vision.balloons.cv2.HoughCircles')
def test_detect_balloons_cv_fallback(mock_hough):
    """Test fallback to OpenCV when text extraction fails."""
    mock_hough.return_value = None  # No circles detected
    # Implementation would assert graceful handling
```

**What to Mock:**
- External file I/O (`fitz.open()`, PDF rendering, image read/write)
- Computer vision operations (`cv2.HoughCircles`, `cv2.ORB_create`, feature matching)
- Excel workbook loading (`openpyxl.load_workbook`)
- Any subprocess or network calls (none present currently)

**What NOT to Mock:**
- Data parsing logic (`parse_requirement()`, `parse_tolerance()`, tolerance parsing)
- Numeric computations (bounding box operations, coordinate transforms)
- Type/dataclass instantiation
- Data validation and transformation logic in reconciliation
- Exclude actual PDF file I/O mocking for integration tests - use fixture PDFs instead

## Fixtures and Factories

**Test Data (would be defined in `conftest.py`):**
```python
# Example: tests/conftest.py
import pytest
from pathlib import Path
from delta_preservation.types import DeltaItem, Evidence
from delta_preservation.io.pdf import TextSpan

@pytest.fixture
def sample_text_span():
    """Create a TextSpan for testing."""
    return TextSpan(
        text="Ø6.0 ± 0.1",
        bbox_pdf=(100.0, 200.0, 150.0, 220.0),
        font_size=12.0,
        block_id=0,
        line_id=0,
        span_id=0
    )

@pytest.fixture
def sample_delta_item():
    """Create a DeltaItem for testing."""
    return DeltaItem(
        char_no=1,
        status="unchanged",
        confidence=0.95,
        reasons=["Primary dimension matches"],
        scores={"location": 0.9, "text": 0.95, "context": 0.8},
        revA=Evidence(page=0, bbox=[100.0, 200.0, 150.0, 220.0], image_path=None),
        revB=Evidence(page=0, bbox=[100.0, 200.0, 150.0, 220.0], image_path=None)
    )
```

**Location:**
- Should live in `tests/conftest.py` (pytest auto-discovers and loads fixtures)
- Domain-specific fixtures in sub-package `conftest.py` files: `tests/test_reconcile/conftest.py` for reconciliation fixtures

**Factory Pattern (for complex objects):**
```python
# Example factory for creating Anchor objects with defaults
class AnchorFactory:
    """Factory for creating test Anchor instances."""

    @staticmethod
    def create(char_no=1, requirement="Ø6.0 ± 0.1", **kwargs):
        from delta_preservation.reconcile.anchors import Anchor
        defaults = {
            "page": 0,
            "balloon_bbox": (100.0, 200.0, 150.0, 220.0),
            "req_bbox": (120.0, 190.0, 180.0, 210.0),
            "requirement_raw": requirement,
            "requirement_norm": requirement.upper(),
            "local_context": []
        }
        defaults.update(kwargs)
        return Anchor(char_no=char_no, **defaults)
```

## Coverage

**Requirements:** No coverage targets enforced (no `pytest.ini` or `pyproject.toml` coverage settings)

**View Coverage (if configured):**
```bash
# Would require: pip install pytest-cov
# Then run:
pytest --cov=delta_preservation --cov-report=html
# Coverage report: htmlcov/index.html
```

**Current State:** No coverage measurement configured; project is pre-testing phase

## Test Types

**Unit Tests:**
- Scope: Individual functions in isolation
- Approach: Mock external dependencies (files, PDFs, images)
- Example targets: `parse_requirement()`, `parse_tolerance()`, `union_bbox()`, tolerance parsing logic
- These test correctness of parsing, matching, and classification logic without I/O

**Integration Tests:**
- Scope: Multiple functions working together
- Approach: Use fixture PDFs if available; test reconciliation pipeline stages
- Example: `test_build_revA_anchors()` combines form3 parsing, balloon detection, and text extraction
- Would exercise: anchor building → candidate generation → matching pipeline

**E2E Tests:**
- Framework: Not implemented
- Approach (if added): Could use pytest with fixture drawing files (small test PDFs)
- Example: `test_run_pipeline_end_to_end()` would run complete 8-stage pipeline on test files
- Not critical for prototype but valuable for regression testing of pipeline changes

## Common Patterns

**Async Testing:**
- Not applicable - codebase has no async/await operations
- All operations are synchronous I/O and computation

**Error Testing:**
```python
# Example error cases to test (based on existing error handling)

def test_extract_text_spans_file_not_found():
    """Verify FileNotFoundError when PDF doesn't exist."""
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        extract_text_spans(Path("nonexistent.pdf"), 0)

def test_extract_text_spans_invalid_page():
    """Verify IndexError for out-of-range page index."""
    # Would require fixture PDF with known page count
    with pytest.raises(IndexError, match="out of range"):
        extract_text_spans(fixture_pdf_path, page_index=999)

def test_load_form3_missing_columns():
    """Verify ValueError when required columns missing."""
    with pytest.raises(ValueError, match="Missing required columns"):
        load_form3(fixture_xlsx_missing_cols, Path("debug"))

def test_alignment_insufficient_features():
    """Verify AlignmentError when images lack common features."""
    from delta_preservation.vision.alignment import AlignmentError
    # Create blank or unrelated images
    with pytest.raises(AlignmentError, match="Insufficient matches"):
        estimate_transform(blank_img1, blank_img2)
```

**Fixture PDFs and Data:**
- Small test PDFs should be committed to `tests/fixtures/pdfs/`
- Test Excel files in `tests/fixtures/xlsx/`
- Example variants: complete form3.xlsx, missing-columns.xlsx, empty.pdf, single-page.pdf

---

*Testing analysis: 2026-02-25*
