# Coding Conventions

**Analysis Date:** 2026-04-09

## Language Style

**Python:**
- Indentation: 4 spaces per level (standard PEP 8)
- String quotes: Double quotes (`"`) are primary
- Semicolons: Not used (standard Python)
- Type hints: Required for function signatures and class attributes
  - Example: `def render_page(pdf_path: Path, page_index: int, dpi: int = 300) -> np.ndarray:`
  - Dataclass fields use mapped types: `page: int = Field(..., description="...")`
- Line length: Appears flexible, generally follows PEP 8 (~100-120 chars)
- Docstrings: Triple-quoted, comprehensive module/class/function docstrings with Args/Returns sections

**SQLAlchemy Models:**
- Use SQLAlchemy 2.0+ mapping style with `Mapped` type hints
- Example from `shop/models.py`: `id: Mapped[int] = mapped_column(Integer, primary_key=True)`
- Relationships defined with `Mapped[list["ClassName"]]` for collections

## Naming Patterns

**Files:**
- `snake_case.py` for module files (e.g., `bbox_utils.py`, `semantic_compare.py`)
- `test_<module>.py` for test files (e.g., `test_grid.py`, `test_auth.py`)
- `__init__.py` for package markers

**Functions:**
- `snake_case` for public functions: `detect_balloons()`, `render_page()`, `extract_text_spans()`
- `_snake_case` prefix for internal/private functions: `_span_key()`, `_dedupe_spans()`, `_normalize_span_text()`
- Single leading underscore indicates module-private scope

**Variables:**
- `snake_case` for all variable names (local, module, class attributes)
- `ALL_CAPS_CONST` for module-level constants: `DEFAULT_DPI`, `MIN_ALIGNMENT_INLIERS`, `FORM3_DEFAULT_COLUMNS`
- Abbreviated names avoided except for well-established domain terms: `bbox` (bounding box), `dpi` (dots per inch), `revA`/`revB` (revision A/B)

**Classes:**
- `PascalCase` for all class names
- Dataclasses with `@dataclass` decorator: `TextSpan`, `Balloon`, `DeltaItem`, `Candidate`, `Match`
- Pydantic models inherit from `BaseModel`: `DeltaPacket`, `Evidence`, `SemanticCallout`
- Frozen dataclasses used when immutability needed: `@dataclass(frozen=True)` in `GroupedSpan`

**Type Names:**
- Type hints use standard Python typing: `Optional[T]`, `List[T]`, `Dict[K, V]`, `Tuple[...]`
- Literal types for fixed string values: `Literal["unchanged", "changed", "removed", "added", "uncertain"]`
- Enum classes for method families: `DetectionMethod.PDF_TEXT`, `DetectionMethod.CV`

**Domain-Specific Names:**
- Revision references: `revA`, `revB` (standardized across codebase)
- Characteristic number: `char_no` (from AS9102 Form 3)
- Bounding box: `bbox`, `bbox_pdf` (PDF coordinates), `bbox_img` (image coordinates)
- Coordinate tuples: `(x0, y0, x1, y1)` for bboxes, `(x, y)` for points, `(cx, cy)` for centers

## Code Style

**Formatting:**
- No automatic formatter configured (no `.prettierrc`, `.black`, or `pyproject.toml` [tool.black] section)
- Manual formatting follows PEP 8 with exceptions noted above
- Imports organized in standard Python style: stdlib, third-party, local

**Linting:**
- No linter config files detected (no `.eslintrc`, `.flake8`, `pylint.rc`)
- Linting appears to be implicit compliance rather than enforced

**Imports Organization:**
```python
# 1. Standard library imports (sys, os, pathlib, etc.)
from pathlib import Path
from typing import Dict, List, Literal, Optional
from datetime import datetime

# 2. Third-party imports (numpy, pandas, fastapi, sqlalchemy, etc.)
import numpy as np
import cv2
import fitz
from fastapi import FastAPI
from pydantic import BaseModel, Field

# 3. Local imports (delta_preservation, shop modules)
from delta_preservation.io.pdf import TextSpan
from delta_preservation.reconcile.anchors import Anchor
```

**Path Aliases:**
- No path aliases configured in `pyproject.toml`
- Absolute imports from package root: `from delta_preservation.types import DeltaItem`

## Error Handling

**Patterns:**
- Explicit exception raising for validation failures: `raise FileNotFoundError()`, `raise IndexError()`
- Error messages include context: `f"PDF not found: {pdf_path}"`, `f"Page index {page_index} out of range (0-{page_count-1})"`
- No global error handlers; exceptions propagate to caller for handling

**Design approach:**
- Functions validate inputs at entry points (path existence, index bounds)
- Optional match results: return `None` instead of raising (e.g., `match_or_none: Optional[Match]`)
- Classification uses status strings rather than exceptions: `status: str # "unchanged", "changed", "removed", "added", "uncertain"`

**Dataclass defaults:**
- Pydantic `Field()` used for validation and description: `Field(..., ge=0.0, le=1.0)` for ranges
- Optional fields use `Optional[T]` with `default=None` in Field

## Comments

**When to Comment:**
- Module-level docstrings explain purpose and key concepts: See `delta_preservation/types.py`, `delta_preservation/cli.py`
- Function docstrings describe parameters, return values, raises, and notes
- Inline comments explain non-obvious logic or complex transformations
- Examples from codebase:
  - `# Notes blocks should preserve full row-major content, not just the matched header span.`
  - `# Convert RGB to BGR for OpenCV compatibility`
  - `# Last-chance identity check` in classify_delta

**JSDoc/TSDoc:**
- Python docstrings follow Google/NumPy style with Args/Returns sections
- Pydantic Field descriptions serve as inline documentation: `description="Bounding box coordinates [x0, y0, x1, y1] in PDF points"`

## Function Design

**Size:**
- Functions vary in length; longer functions have clear logical sections marked by comments
- Example: `run_pipeline()` in `cli.py` is ~300 lines with 8 marked pipeline stages

**Parameters:**
- Type hints required on all parameters
- Default values used for configuration: `dpi: int = 300`, `search_radius: float = 144.0`
- Callback parameters accepted: `stage_callback: Optional[Callable[[int, str], None]] = None`

**Return Values:**
- Explicit return types required
- Multiple returns: use dataclass or tuple
- Example: `Candidate` dataclass contains `span`, `total_score`, `location_score`, `text_score`, `context_score`, `reasons`
- `None` used for optional results

## Module Design

**Exports:**
- Public API defined by module-level imports and direct function/class definitions
- No explicit `__all__` lists observed (imports handle visibility)
- Internal functions prefixed with `_` excluded from intended public API

**Barrel Files:**
- No barrel export files observed (e.g., no index.ts)
- `__init__.py` files exist but appear minimal/empty
- Imports use direct module paths: `from delta_preservation.io.pdf import TextSpan`

## Pydantic/SQLAlchemy Specifics

**Pydantic Models:**
- Used for structured data validation and serialization
- Example from `delta_preservation/types.py`: `DeltaPacket` with nested models
- Field validation: `confidence: float = Field(..., ge=0.0, le=1.0, description="...")`

**SQLAlchemy:**
- Modern declarative syntax with `Mapped` type hints
- Foreign key relationships with `ForeignKey` and `relationship()`
- Cascade options used: `cascade="all, delete-orphan"`
- Example from `shop/models.py`: `sessions: Mapped[list["UserSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")`

## Anti-patterns to Avoid

- Ambiguous single-letter variable names (except loop counters in tight scopes)
- Silent failures; exceptions preferred over returning incorrect values
- Circular dependencies between modules
- Mixing camelCase and snake_case in same scope
- Global mutable state (use dependency injection via `SessionLocal` or passed parameters)

---

*Convention analysis: 2026-04-09*
