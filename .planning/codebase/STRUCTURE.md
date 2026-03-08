# Codebase Structure

**Analysis Date:** 2026-02-25

## Directory Layout

```
delta-preservation/
├── delta_preservation/           # Main package
│   ├── __init__.py               # Empty marker file
│   ├── cli.py                    # Pipeline orchestration and CLI entry point
│   ├── config.py                 # Centralized configuration and thresholds
│   ├── types.py                  # Pydantic models: DeltaPacket, DeltaItem, Evidence
│   ├── io/                       # Document I/O layer
│   │   ├── __init__.py
│   │   ├── pdf.py                # PDF rendering and text extraction (PyMuPDF)
│   │   └── xlsx.py               # AS9102 Form 3 Excel parsing
│   ├── vision/                   # Computer vision layer
│   │   ├── __init__.py
│   │   ├── balloons.py           # Balloon detection (hybrid PDF text + OpenCV)
│   │   ├── alignment.py          # Image alignment and homography estimation
│   │   ├── snippets.py           # Evidence image cropping and storage
│   │   └── bbox_utils.py         # Bounding box utilities (expansion, normalization)
│   └── reconcile/                # Reconciliation and matching layer
│       ├── __init__.py
│       ├── anchors.py            # Anchor building (link Form 3 to Rev A)
│       ├── match.py              # Candidate generation and bipartite assignment
│       ├── classify.py           # Delta classification (unchanged/changed/removed/added)
│       ├── normalize.py          # Requirement text parsing and fingerprinting
│       └── tolerance_pdf.py      # Tolerance extraction and comparison
├── assets/                       # Test/reference data
│   ├── part1/                    # Part 1 test data
│   │   ├── revA.pdf              # Revision A drawing
│   │   ├── revB.pdf              # Revision B drawing
│   │   └── FAIR.xlsx             # AS9102 Form 3
│   └── part2/                    # Part 2 test data
├── out/                          # Pipeline output directory
│   └── part2_2026-02-24T16-29-52_95484c30/  # Run output (timestamped, hashed)
│       ├── delta_packet.json     # Main results (DeltaPacket serialized)
│       ├── snippets/             # Evidence images (named per char_no and revision)
│       │   ├── char_1_revA_p0.png
│       │   ├── char_1_revB_p0.png
│       │   └── ...
│       └── debug/                # Debug outputs
│           ├── form3_chars.json  # Parsed Form 3 characteristics
│           └── tolerance_parsing_tests.json  # Tolerance extraction debug data
├── .planning/                    # GSD planning artifacts
│   └── codebase/                 # Codebase analysis documents
│       ├── ARCHITECTURE.md       # This file's companion
│       └── STRUCTURE.md          # This file
├── .claude/                      # Claude-specific resources
│   └── AS9102C.md                # Full AS9102 Rev C standard reference
├── run.py                        # Thin wrapper CLI entry point
├── pyproject.toml                # Python project metadata and dependencies
├── uv.lock                       # Lock file for uv package manager
├── CLAUDE.md                     # Project instructions for Claude
└── README.md                     # User-facing project documentation
```

## Directory Purposes

**`delta_preservation/`:**
- Purpose: Main Python package containing all pipeline logic
- Contains: CLI, configuration, type definitions, and three processing layers (IO, Vision, Reconcile)
- Key files: `cli.py` (entry point), `config.py` (thresholds), `types.py` (output contracts)

**`delta_preservation/io/`:**
- Purpose: Read engineering drawings and inspection forms
- Contains: PDF text extraction with PyMuPDF, Excel parsing with openpyxl, coordinate conversion utilities
- Key files: `pdf.py` (TextSpan extraction, page rendering), `xlsx.py` (Form 3 parsing)

**`delta_preservation/vision/`:**
- Purpose: Computer vision processing for spatial information
- Contains: Balloon detection, image alignment, image cropping, bbox utilities
- Key files: `balloons.py` (hybrid PDF+OpenCV detection), `alignment.py` (ORB+RANSAC homography), `bbox_utils.py` (expansion/normalization)

**`delta_preservation/reconcile/`:**
- Purpose: Link requirements to locations and classify deltas
- Contains: Anchor building, candidate matching, delta classification, text normalization, tolerance handling
- Key files: `anchors.py` (weighted scoring), `match.py` (multi-component ranking), `classify.py` (decision tree), `normalize.py` (fingerprints)

**`assets/`:**
- Purpose: Reference PDFs and Form 3 files for testing
- Contains: Two parts (part1, part2), each with revA.pdf, revB.pdf, FAIR.xlsx
- Generated: No; committed to repository

**`out/`:**
- Purpose: Runtime output storage
- Contains: Timestamped run directories with delta packets, evidence snippets, debug JSON
- Generated: Yes, by `run_pipeline()`; not committed

**`.planning/codebase/`:**
- Purpose: GSD (Goal-Setting Document) analysis artifacts
- Contains: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md (as created)
- Generated: No; created by codebase mapper agent

**`.claude/`:**
- Purpose: Claude-specific resources and context
- Contains: AS9102 standard reference document, project instructions
- Generated: No; curated manually

## Key File Locations

**Entry Points:**
- `run.py`: Thin wrapper parsing `--part_name`, `--dpi` arguments and calling `delta_preservation.cli.run_pipeline()`
- `delta_preservation/cli.py:main()`: Full argparse CLI with 8-stage pipeline orchestration
- `delta_preservation/cli.py:run_pipeline()`: Core pipeline function (reusable from tests/automation)

**Configuration:**
- `delta_preservation/config.py`: All numeric thresholds (search radii, quality thresholds, column mappings)
- `pyproject.toml`: Python dependencies (fitz, cv2, numpy, openpyxl, pydantic)
- `.claude/AS9102C.md`: External reference (AS9102 standard) loaded into Claude memory

**Core Logic:**

*IO Layer:*
- `delta_preservation/io/pdf.py:render_page()`: Render PDF page to NumPy BGR array at specified DPI
- `delta_preservation/io/pdf.py:extract_text_spans()`: Extract TextSpan list from PDF page
- `delta_preservation/io/pdf.py:pdf_to_img_coords()`: Convert PDF coords to image coords given DPI and page rect
- `delta_preservation/io/xlsx.py:load_form3()`: Parse Excel Form 3 with auto-detected columns

*Vision Layer:*
- `delta_preservation/vision/balloons.py:detect_balloons()`: Hybrid PDF text + OpenCV circle detection
- `delta_preservation/vision/alignment.py:estimate_transform()`: ORB + RANSAC homography
- `delta_preservation/vision/snippets.py:crop_with_padding()`: Crop image to bbox with padding
- `delta_preservation/vision/bbox_utils.py:expand_bbox_with_adjacent_spans()`: Grow bbox to include nearby text symbols

*Reconcile Layer:*
- `delta_preservation/reconcile/anchors.py:build_revA_anchors()`: Link Form 3 to Rev A via weighted scoring
- `delta_preservation/reconcile/match.py:generate_candidates()`: Score candidates using homography prediction + multi-component matching
- `delta_preservation/reconcile/match.py:assign_matches()`: Greedy bipartite assignment
- `delta_preservation/reconcile/classify.py:classify_delta()`: Decision tree classification (unchanged/changed/removed/added/uncertain)
- `delta_preservation/reconcile/normalize.py:parse_requirement()`: Extract MatchFingerprint (symbols, numerics, pattern class)

**Testing:**
- `pytest` runs all tests in repository (if test files exist; none found in glob)
- Config: `pyproject.toml` pytest section (if present; check via read)

## Naming Conventions

**Files:**
- Snake case: `pdf.py`, `balloons.py`, `bbox_utils.py`
- Descriptive: Function/module name matches primary responsibility (e.g., `alignment.py` → `estimate_transform()`)
- No abbreviations except domain-standard: PDF, DPI, ORB, RANSAC, CSV, XLSX

**Directories:**
- Snake case: `delta_preservation/`, `vision/`, `reconcile/`
- Domain-organized: `io/`, `vision/`, `reconcile/` map to processing layers in pipeline

**Functions:**
- Snake case: `render_page()`, `extract_text_spans()`, `build_revA_anchors()`, `detect_balloons()`
- Verb-first: `extract_`, `build_`, `detect_`, `estimate_`, `classify_`
- No abbreviations: `revA`, `revB` acceptable for domain context

**Classes & Dataclasses:**
- PascalCase: `TextSpan`, `Anchor`, `Candidate`, `Match`, `Transform`, `Balloon`, `Characteristic`, `MatchFingerprint`, `Evidence`, `DeltaItem`, `DeltaPacket`
- Descriptive: Class name reflects entity it represents

**Variables:**
- Snake case: `revA_text_spans`, `balloon_bbox`, `form3_chars`, `candidates_by_anchor`
- Mnemonic: `H` for homography matrix (standard in CV), `bbox_pdf`, `center_b` (B=Rev B coordinate space)
- Suffixes: `_pdf` for PDF coordinates, `_img` for image coordinates, `_b`/`_a` for Rev B/Rev A

**Constants:**
- UPPER_CASE: `DEFAULT_DPI`, `DEFAULT_SEARCH_RADIUS`, `MIN_ALIGNMENT_INLIERS`, `SEARCH_RADIUS` (local scope)
- Organized in `config.py` for global configuration

## Where to Add New Code

**New Feature (e.g., new type of characteristic detection):**
- Primary code: `delta_preservation/reconcile/` module matching feature domain
- Tests: `tests/test_[module_name].py` if test structure exists
- Example: New tolerance format → update `delta_preservation/reconcile/tolerance_pdf.py` + `parse_pdf_tolerance()` function

**New Computer Vision Module (e.g., OCR enhancement):**
- Implementation: `delta_preservation/vision/[feature_name].py`
- Entry point: Function called from `cli.py` stage where it fits
- Example: OCR for balloons → new `vision/ocr.py` with `ocr_balloons()` function, called before/after `detect_balloons()` in stage 2

**Shared Utilities (e.g., new bbox operation):**
- Helpers: `delta_preservation/vision/bbox_utils.py` (geometry) or `delta_preservation/reconcile/normalize.py` (text parsing)
- Pattern: Standalone function taking minimal context, returning typed result

**New Data Model:**
- Output models: `delta_preservation/types.py` as Pydantic BaseModel (enables JSON serialization, validation)
- Internal models: Dataclass in relevant module (e.g., `Transform` in `alignment.py`, `Anchor` in `anchors.py`)

**Configuration Updates:**
- Numeric thresholds: `delta_preservation/config.py` (centralized, imported via `from delta_preservation.config import DEFAULT_X`)
- Form 3 column mapping: `FORM3_DEFAULT_COLUMNS` and `FORM3_HEADER_KEYWORDS` in `config.py`

## Special Directories

**`out/`:**
- Purpose: Runtime outputs
- Generated: Yes, by `run_pipeline()` at line 99 in `cli.py`
- Committed: No; in `.gitignore`
- Structure: `out/<part_name>_<timestamp>_<hash>/` with `delta_packet.json`, `snippets/`, `debug/`

**`.planning/codebase/`:**
- Purpose: GSD analysis documents
- Generated: No; created by codebase mapper agent (`/gsd:map-codebase`)
- Committed: Yes; guides future implementation phases
- Contents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md

**`.claude/`:**
- Purpose: Claude-specific context and resources
- Generated: No; curated manually
- Committed: Yes
- Key file: `AS9102C.md` (full standard reference, auto-loaded in Claude sessions)

**`assets/`:**
- Purpose: Reference test data
- Generated: No; checked in as repository fixtures
- Committed: Yes
- Structure: Organized by part (part1, part2), each containing revA.pdf, revB.pdf, FAIR.xlsx

**`.venv/` (virtualenv):**
- Purpose: Python virtual environment
- Generated: Yes, by `uv sync`
- Committed: No; in `.gitignore`

## Coordinate System Mappings

When adding new code that manipulates coordinates, follow these conventions:

**PDF Space** (used throughout reconciliation):
- Units: Points (1/72 inch)
- Origin: Top-left (0, 0)
- Representation: `(x0, y0, x1, y1)` tuples (xyxy format)
- Convention: Suffix bbox variables with `_pdf` → `balloon_bbox_pdf`

**Image Space** (used in vision layer):
- Units: Pixels at render DPI (default 300)
- Origin: Top-left (0, 0)
- Representation: Same xyxy format as PDF space
- Convention: Suffix with `_img` → `bbox_img_a`

**Conversion:**
```python
# PDF → Image
from delta_preservation.io.pdf import pdf_to_img_coords
bbox_img = pdf_to_img_coords(bbox_pdf, page, dpi=300)

# Spatial transform (Rev A → Rev B)
import cv2
center_a = np.array([[(x0 + x1) / 2, (y0 + y1) / 2]], dtype=np.float32).reshape(-1, 1, 2)
center_b = cv2.perspectiveTransform(center_a, transform.H)[0][0]
```

---

*Structure analysis: 2026-02-25*
