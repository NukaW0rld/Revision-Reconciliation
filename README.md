# Delta Preservation

**Advanced characteristic identity preservation across engineering drawing revisions**

This repository contains a prototype that solves the critical problem of maintaining inspection characteristic identity when engineering drawings undergo revision changes. The system enables incremental AS9102 FAIR updates rather than complete re-inspections, saving significant time and cost in aerospace and defense manufacturing.

## Problem Statement

When engineering drawings transition from Rev A → Rev B, quality teams must determine which inspection characteristics are:
- **Unchanged** (carry forward existing inspection data)
- **Changed** (require re-inspection) 
- **Removed** (no longer applicable)
- **Added** (new inspection requirements)

Traditional approaches fail when layout changes, projections shift, or annotations move, which forces complete re-FAIRs even for minor revisions.

## Solution

Our **8-stage reconciliation pipeline** combines computer vision, spatial transformation, and semantic analysis to preserve characteristic identity across revisions:

1. **Form 3 Parsing** - Extract inspection characteristics from AS9102 Excel documents
2. **Balloon Detection** - Hybrid PDF text + computer vision approach for characteristic markers
3. **Text Extraction** - Precise coordinate mapping of requirement annotations
4. **Anchor Building** - Link Form 3 requirements to Rev A spatial locations
5. **Alignment Estimation** - ORB feature matching + RANSAC homography; text-span fallback for near-identity transforms
6. **Candidate Matching** - Multi-component scoring (location + semantic + context)
7. **Delta Classification** - Intelligent unchanged/changed/removed/added decisions
8. **Evidence Generation** - Visual snippets for human review and audit compliance

---

## Requirements

- **Python 3.10+**
- **uv** for dependency management (recommended) or pip
- **System Dependencies:**
  - OpenCV (computer vision)
  - PyMuPDF (PDF processing)
  - OpenPyXL (Excel parsing)

## Installation

```bash
# Clone repository
git clone https://github.com/NukaW0rld/Revision-Reconciliation.git
cd delta-preservation

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

---

## Repository Structure

```
delta-preservation/
├── run.py                       # Convenience entry point (uv run python run.py part1)
├── delta_preservation/           # Main package
│   ├── io/                      # PDF and Excel I/O modules
│   ├── vision/                  # Computer vision (balloons, alignment)
│   ├── reconcile/               # Core matching and classification logic
│   ├── cli.py                   # Pipeline orchestration and CLI
│   ├── types.py                 # Pydantic data models
│   └── config.py                # Configuration constants
├── tests/                       # Comprehensive test suite
├── assets/                      # Test fixtures and sample data
│   ├── part1/                   # Stable layout test case
│   └── part2/                   # Major layout shift test case
└── README.md                    # This file
```

---

## Quick Start

Run the complete pipeline on provided test data:

```bash
# Short form (recommended)
uv run python run.py part1
uv run python run.py part2
uv run python run.py part1 --dpi 150

# Long-form (useful for custom asset paths)
python -m delta_preservation.cli \
  --revA_pdf assets/part1/revA.pdf \
  --revB_pdf assets/part1/revB.pdf \
  --form3_xlsx assets/part1/FAIR.xlsx \
  --part_name part1
```

### Command-Line Arguments

**`run.py` (short form):**
- `part_name` - Part name matching `assets/` subdirectory (e.g. `part1`, `part2`)
- `--dpi` - Rendering DPI for image processing (default: `300`)
- `--out_dir` - Output directory (default: `<repo_root>/out`)

**`delta_preservation.cli` (long form):**
- `--revA_pdf` - Path to Rev A PDF (ballooned with characteristic numbers)
- `--revB_pdf` - Path to Rev B PDF (may be unballooned)
- `--form3_xlsx` - Path to AS9102 Form 3 Excel file
- `--out_dir` - Output directory (default: `./out`)
- `--dpi` - Rendering DPI for image processing (default: `300`)
- `--part_name` - Part identifier for run naming (default: `part`)

---

## Output Structure

Each execution creates a timestamped run directory:

```
out/<part_name>_<timestamp>_<hash>/
├── delta_packet.json           # Complete analysis results
├── snippets/                   # Visual evidence images
│   ├── char_001_revA_p0.png   # Rev A evidence snippets
│   ├── char_001_revB_p0.png   # Rev B evidence snippets
│   └── ...
└── debug/                      # Debug artifacts
    ├── form3_chars.json        # Parsed Form 3 data
    └── tolerance_parsing_tests.json  # Per-characteristic tolerance debug data
```

### Delta Packet Structure

The `delta_packet.json` contains structured results with full traceability:

```json
{
  "run_id": "part1_2024-01-29T16-12-59_95484c30",
  "inputs": {
    "revA_pdf": "/path/to/revA.pdf",
    "revB_pdf": "/path/to/revB.pdf", 
    "form3_xlsx": "/path/to/FAIR.xlsx",
    "dpi": "300"
  },
  "items": [
    {
      "char_no": 17,
      "status": "unchanged",
      "confidence": 0.92,
      "reasons": [
        "Primary dimension matches: 6.0",
        "Numeric values match (100% overlap)",
        "High location agreement after global alignment"
      ],
      "scores": {
        "location": 0.88,
        "text": 0.95, 
        "context": 0.42
      },
      "revA": {
        "page": 0,
        "bbox": [284.2, 156.8, 324.1, 178.3],
        "image_path": "snippets/char_017_revA_p0.png"
      },
      "revB": {
        "page": 0, 
        "bbox": [291.5, 162.1, 331.4, 183.6],
        "image_path": "snippets/char_017_revB_p0.png"
      }
    }
  ]
}
```

---

## Key Features

### Robust Characteristic Detection
- **Hybrid balloon detection** using PDF text extraction + computer vision fallbacks
- **Handles merged text spans** (e.g., "24 25" balloon clusters)
- **Circle validation** to reduce false positives from non-balloon text

### Advanced Spatial Matching
- **Hybrid alignment** using ORB feature matching + RANSAC homography, with text-span-based fallback when ORB produces a near-identity transform
- **Multi-component candidate scoring** (location + semantic + spatial context)
- **Intelligent search radius adaptation** for major layout variations

### Semantic Classification
- **Engineering-aware text parsing** (symbols: Ø, R, ±; patterns: 2X, 4X; numerics)
- **Primary dimension matching** as key discriminator for unchanged characteristics
- **Context-sensitive decisions** distinguishing real changes from PDF formatting differences

### Human-Reviewable Output
- **Visual evidence snippets** for every classification decision
- **Detailed reasoning** with confidence scores and component breakdowns
- **Audit-ready traceability** linking decisions back to spatial and semantic evidence

---

## Architecture

The system follows a **clean, modular architecture** with clear separation of concerns:

- **`io/`** - PDF and Excel I/O with coordinate precision
- **`vision/`** - Computer vision for detection and alignment  
- **`reconcile/`** - Core matching and classification intelligence
- **`types.py`** - Validated data models with Pydantic
- **`config.py`** - Centralized configuration management

Each module includes comprehensive docstrings, type hints, and extensive comments explaining the domain-specific algorithms and engineering decisions.

---

## Limitations & Future Work

### Current Limitations
- **Single-page support** (multi-page drawings require manual processing)
- **Limited revision table detection** (may miss some title block variations)
- **English-only text parsing** (symbols and patterns are language-agnostic)

### Planned Improvements
- **Multi-page drawing support** with cross-page characteristic tracking
- **Advanced revision table parsing** for ECO number extraction
- **Machine learning enhancements** for improved classification accuracy
- **Web interface** for quality team review and approval workflows

---

## Support

For questions, issues, or contributions, please email james@handymechanics.com.
