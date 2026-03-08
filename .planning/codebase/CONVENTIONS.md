# Coding Conventions

**Analysis Date:** 2026-02-25

## Naming Patterns

**Files:**
- Snake case with descriptive names: `delta_preservation/reconcile/normalize.py`, `delta_preservation/vision/alignment.py`
- Module names match their primary responsibility domain
- Organization by domain subdirectory (`io/`, `vision/`, `reconcile/`) not by file type

**Functions:**
- Snake case: `extract_text_spans()`, `parse_requirement()`, `build_revA_anchors()`
- Descriptive verb-noun patterns: `estimate_transform()`, `generate_candidates()`, `assign_matches()`
- Private functions prefixed with underscore: `_detect_balloons_from_text()`, `_parse_inline_plus_minus()`, `_collect_spans_near_bbox()`

**Variables and Parameters:**
- Snake case for all variables: `form3_chars`, `revA_pdf`, `text_spans`, `anchor_bbox`
- Abbreviations used consistently: `revA`/`revB` for revision versions, `imgA`/`imgB` for rendered images, `bbox_pdf` for coordinate space
- Coordinate tuples named with context: `bbox_pdf`, `bbox_img_a`, `balloon_bbox` (never ambiguous like `bbox`)

**Types and Classes:**
- PascalCase for classes: `DeltaItem`, `Characteristic`, `Evidence`, `TextSpan`, `Transform`, `Balloon`
- Dataclass names follow noun patterns: `Anchor`, `Candidate`, `Match`, `MatchFingerprint`
- Exception classes end with "Error": `AlignmentError`

## Code Style

**Formatting:**
- No auto-formatter detected (no `.black`, `.ruff`, `.prettierrc` configuration)
- Apparent style follows PEP 8: 4-space indentation, max line length ~100-120 characters
- String formatting uses f-strings throughout (e.g., `f"Detected {len(balloons)} balloons"`)

**Linting:**
- No linting configuration files present (no `.pylintrc`, `.flake8`, `ruff.toml`)
- Manual PEP 8 adherence observed in codebase
- Type hints are optional but used sporadically (see imports with `from typing import ...`)

## Import Organization

**Order:**
1. Standard library imports (`pathlib`, `typing`, `re`, `json`, `hashlib`, `argparse`)
2. Third-party library imports (`numpy`, `cv2`, `fitz`, `openpyxl`, `pydantic`)
3. Local relative imports (`from delta_preservation.io.pdf import ...`)

**Path Aliases:**
- No path aliases configured; all imports use absolute paths from project root
- Common pattern: `from delta_preservation.[module].[submodule] import [Class/function]`
- TYPE_CHECKING guards used for circular dependency prevention: `if TYPE_CHECKING: from delta_preservation.reconcile.tolerance_pdf import ToleranceComparison`

**Import Examples:**
```python
# cli.py: Mixed standard/third-party/local pattern
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import cv2
import fitz

from delta_preservation.io.pdf import render_page, extract_text_spans, pdf_to_img_coords
```

## Error Handling

**Patterns:**
- Explicit exception types: `FileNotFoundError`, `ValueError`, `IndexError`, `OSError`
- Custom exception for domain: `AlignmentError` in `vision/alignment.py`
- Try-except blocks limited to specific, recoverable errors (e.g., number parsing in `normalize.py`)
- Validation errors raised with descriptive messages: `f"Revision A must be a PDF file, got: {revA_path.suffix}"`

**Input Validation:**
- File existence checks before processing: `if not revA_path.exists(): raise FileNotFoundError(...)`
- File type checks by extension: `if not revA_path.suffix.lower() == ".pdf": raise ValueError(...)`
- Page index bounds validation: `if page_index < 0 or page_index >= page_count: raise IndexError(...)`
- Bounding box validity checks: `raise ValueError(f"Invalid bbox: x1 must be > x0 and y1 must be > y0, got {bbox_px}")`

**Error Messages:**
- Descriptive and context-specific: `f"Page index {page_index} out of range (0-{page_count-1})"`
- Include expected vs. actual values
- No silent failures; all errors explicitly raised or logged

**Recovery:**
- Graceful fallback in anchor building when exact text match fails (uses balloon bbox as fallback)
- Try-catch for numeric conversions with explicit None returns on failure
- Image crop failures caught and recorded in evidence without image path: `image_path=None`

## Logging

**Framework:** `print()` statements (no logging library imported)

**Patterns:**
- Progress indicators in pipeline orchestration: `print("[1/8] Loading Form 3...")`
- Status updates with counts: `print(f"  Loaded {len(form3_chars)} characteristics")`
- Final output path reported: `print(f"Review delta packet: {packet_path}")`

**When to Log:**
- Major pipeline stage entry points (8 stages in cli.py)
- Processing summary counts (features detected, matches found, items classified)
- Output directory and file paths for user reference
- Alignment quality metrics (inliers, ratio)

**Logging Examples from `cli.py`:**
```python
print(f"Run ID: {run_id}")
print(f"Output: {run_dir.absolute()}")
print("[1/8] Loading Form 3...")
print(f"  Detected {len(balloons)} balloons")
print(f"  Alignment: {transform.inliers} inliers, ratio={transform.inlier_ratio:.2f}")
```

## Comments

**When to Comment:**
- Module docstrings mandatory (all modules have """ """ headers)
- Function/method docstrings with Args, Returns, Raises sections: `extract_text_spans()` has full docstring
- Complexity explanation: "Avoid matching patterns like '10 x 90°' where x is multiplication between two numbers"
- Business logic reasoning in decision trees: `# Notes blocks should be compared by text pattern, not numeric values`
- Coordinate space clarifications: `# PDF coordinates use standard 72 DPI point units`

**JSDoc/TSDoc:**
- Python docstrings follow NumPy/SciPy style with sections:
  ```python
  """Brief description.

  Extended description.

  Args:
      param_name: Description.

  Returns:
      Description of return value.

  Raises:
      ExceptionType: When raised.

  Notes:
      - Implementation detail 1
      - Implementation detail 2
  """
  ```
- Used consistently in: `run_pipeline()`, `estimate_transform()`, `parse_requirement()`, `classify_delta()`

## Function Design

**Size:** Functions range 20-150 lines; larger functions (400+ lines in `cli.py:run_pipeline()`) are pipeline orchestration stage sequences

**Parameters:**
- Explicit over implicit: all parameters named, no *args/**kwargs patterns observed
- Type hints in signatures: `extract_text_spans(pdf_path: Path, page_index: int) -> List[TextSpan]`
- Optional parameters have defaults: `dpi: int = 300`, `page_index: int = 0`
- Coordinate space made explicit in parameter names: `bbox_pdf`, `bbox_img_coords` (never ambiguous `bbox`)

**Return Values:**
- Single returns preferred (no multiple assignment unpacking)
- Complex returns use dataclasses/Pydantic models: `DeltaItem`, `Transform`, `Evidence`
- None returns only for optional/missing data: `Optional[TextSpan]`, `Optional[Match]`
- List returns for collections: `List[TextSpan]`, `List[Anchor]`
- Dict returns for mappings: `Dict[int, Balloon]`, `Dict[str, float]`

**Return Value Examples:**
```python
# Simple scalar
def pdf_to_img_coords(...) -> Tuple[float, float, float, float]:
    return (x0, y0, x1, y1)

# Complex dataclass
def classify_delta(...) -> DeltaItem:
    return DeltaItem(char_no=..., status=..., confidence=..., ...)

# Collections
def extract_text_spans(...) -> List[TextSpan]:
    return spans

def build_revA_anchors(...) -> List[Anchor]:
    return anchors
```

## Module Design

**Exports:**
- All public functions/classes defined at module level (no export lists)
- Private module functions prefixed with underscore and not documented in main docstring
- Clear module responsibility with focused exports: `io/pdf.py` exports PDF primitives, `reconcile/normalize.py` exports parsing utilities

**Barrel Files:**
- Not used; imports are direct from submodules: `from delta_preservation.io.pdf import extract_text_spans`
- `__init__.py` files present but empty (no re-exports)

**Module Cohesion Examples:**
- `io/xlsx.py`: Form 3 parsing utilities only (tolerance parsing, characteristic loading)
- `reconcile/`: All matching/classification logic (anchors, matching, classification, normalization)
- `vision/`: Computer vision operations (balloons, alignment, snippets, bbox utilities)

---

*Convention analysis: 2026-02-25*
