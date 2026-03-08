# Technology Stack

**Analysis Date:** 2026-02-25

## Languages

**Primary:**
- Python 3.10–3.13 - Entire application codebase

## Runtime

**Environment:**
- CPython (Python 3.10+)
- Tested on Python 3.12.3

**Package Manager:**
- uv - Dependency management and build orchestration
- Lockfile: `uv.lock` (present, pinned to specific versions)

## Frameworks

**Core:**
- Pydantic 2.12.5 - Data validation and JSON serialization for output models in `delta_preservation/types.py`

**Computer Vision:**
- OpenCV (opencv-python) 4.13.0.90 - Image processing for balloon detection and alignment; used in `delta_preservation/vision/balloons.py` and `delta_preservation/vision/alignment.py`
- NumPy 2.2.6–2.4.1 - Array operations and homography transforms (version depends on Python <3.11 vs ≥3.11)

**Document Processing:**
- PyMuPDF (pymupdf) 1.26.7 - PDF rendering and text extraction in `delta_preservation/io/pdf.py`
- OpenPyXL (openpyxl) 3.1.5 - AS9102 Form 3 Excel parsing in `delta_preservation/io/xlsx.py`

**Testing:**
- pytest 8.4.2 - Test runner configured via implicit pytest.ini or pyproject.toml discovery
- Run with: `pytest` (see CLAUDE.md)

**Build/Dev:**
- None detected beyond uv

## Key Dependencies

**Critical:**
- PyMuPDF (pymupdf) 1.26.7 - PDF coordinate precision and rendering at variable DPI (72–300 DPI)
- OpenCV (opencv-python) 4.13.0.90 - ORB feature detection + RANSAC homography for cross-revision alignment; fallback balloon detection via HoughCircles
- NumPy 2.2.6–2.4.1 - Homography matrix operations and coordinate transforms
- OpenPyXL 3.1.5 - Robust Excel parsing with keyword-driven column detection

**Infrastructure:**
- Pydantic 2.12.5 - Validates and serializes `DeltaPacket`, `DeltaItem`, `Evidence` output models; ensures JSON compliance
- RapidFuzz 3.14.3 - Listed as a dependency in `pyproject.toml` but not currently imported or used anywhere in the codebase; token matching uses plain Python set operations instead

## Configuration

**Environment:**
- Command-line arguments only (no .env file required)
- Key parameters: `--revA_pdf`, `--revB_pdf`, `--form3_xlsx`, `--dpi` (default 300), `--out_dir` (default `./out`), `--part_name`
- Config centralized in `delta_preservation/config.py`: `DEFAULT_DPI=300`, `DEFAULT_SEARCH_RADIUS=144.0` (PDF points), `MIN_ALIGNMENT_INLIERS=40`, `MIN_ALIGNMENT_RATIO=0.15`

**Build:**
- `pyproject.toml` - Single file: declares Python ≥3.10,<3.13, lists 7 dependencies, specifies package structure
- No separate build config (setuptools integrated via pyproject.toml)

## Platform Requirements

**Development:**
- Python 3.10+ with pip or uv
- OpenCV system dependencies (libsm6, libxext6 on Linux; may require `apt-get install` or Homebrew on macOS)
- uv recommended for fast, reproducible builds

**Production:**
- Self-contained CLI: all pipeline stages orchestrated in `delta_preservation/cli.py`
- Input: PDF files (Rev A and Rev B) + Excel Form 3
- Output: JSON delta packet + PNG evidence snippets to local filesystem
- No server, no containerization detected

---

*Stack analysis: 2026-02-25*
