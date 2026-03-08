# Architecture

**Analysis Date:** 2026-02-25

## Pattern Overview

**Overall:** 8-stage data transformation pipeline with separation of concerns across IO, Vision, and Reconciliation layers

**Key Characteristics:**
- Sequential pipeline orchestration with deterministic output (run_id = timestamp + hash)
- Domain-separated modules: document I/O, computer vision, semantic reconciliation, classification
- Spatial coordinate spaces (PDF points, image pixels) with explicit transformation functions
- Pydantic models for structured output serialization
- Gradient matching from strict (spatial location) to semantic (text content) signals

## Layers

**Input/Output Layer (`delta_preservation/io/`):**
- Purpose: Read engineering drawings and inspection forms; produce structured evidence snapshots
- Location: `delta_preservation/io/pdf.py`, `delta_preservation/io/xlsx.py`
- Contains: PDF rendering/text extraction (`TextSpan`), Excel parsing (`Characteristic`), coordinate conversion utilities
- Depends on: PyMuPDF (fitz), openpyxl
- Used by: All stages; forms foundation for text spans and characteristics

**Vision Layer (`delta_preservation/vision/`):**
- Purpose: Extract spatial information from documents via computer vision
- Location: `delta_preservation/vision/balloons.py`, `delta_preservation/vision/alignment.py`, `delta_preservation/vision/snippets.py`, `delta_preservation/vision/bbox_utils.py`
- Contains: Balloon detection (PDF text + OpenCV fallback), homography-based alignment, image cropping, bbox utilities
- Depends on: OpenCV (cv2), NumPy
- Used by: CLI pipeline stages; produces `Balloon` objects and `Transform` matrices

**Reconciliation Layer (`delta_preservation/reconcile/`):**
- Purpose: Link Form 3 requirements to drawing locations; match characteristics across revisions; classify deltas
- Location: `delta_preservation/reconcile/anchors.py`, `delta_preservation/reconcile/match.py`, `delta_preservation/reconcile/classify.py`, `delta_preservation/reconcile/normalize.py`, `delta_preservation/reconcile/tolerance_pdf.py`
- Contains: Anchor building (weighted token/distance scoring), candidate generation (location/text/context scoring), delta classification (numeric/token matching), tolerance parsing
- Depends on: Alignment transforms, text spans, normalized requirement fingerprints
- Used by: CLI pipeline stages 4-7

**Configuration & Types Layer:**
- Purpose: Centralize numeric thresholds, provide validated data contracts
- Location: `delta_preservation/config.py`, `delta_preservation/types.py`
- Contains: Search radii, quality thresholds, Form 3 column mapping, Pydantic models (`DeltaPacket`, `DeltaItem`, `Evidence`)
- Depends on: Pydantic
- Used by: All other layers for configuration and serialization

## Data Flow

**Rev A → Rev B Characteristic Matching (Primary Flow):**

1. **Stage 1 - Form 3 Load** (`io/xlsx.py:load_form3`)
   - Input: AS9102 Form 3 Excel file
   - Process: Parse spreadsheet, auto-detect column headers via keyword matching, extract `char_no` → `requirement` map
   - Output: `Dict[int, str]` mapping characteristic numbers to requirement text

2. **Stage 2 - Balloon Detection** (`vision/balloons.py:detect_balloons`)
   - Input: Rev A PDF
   - Process: Hybrid approach — attempt PDF text extraction for circular text annotations, fallback to OpenCV HoughCircles with Tesseract OCR for digit recognition
   - Output: `Dict[int, Balloon]` mapping char_no to spatial locations with confidence scores

3. **Stage 3 - Text Extraction (Rev A)** (`io/pdf.py:extract_text_spans`)
   - Input: Rev A PDF (page 0 by default)
   - Process: PyMuPDF block/line/span iteration; extract text, font size, bounding box (PDF points)
   - Output: `List[TextSpan]` with `bbox_pdf`, `block_id` (page index), `text` content

4. **Stage 4 - Anchor Building** (`reconcile/anchors.py:build_revA_anchors`)
   - Input: Form 3 requirements, balloons, Rev A text spans
   - Process: For each characteristic, link Form 3 requirement to Rev A location via weighted scoring (token overlap 50%, distance 40%, size penalty -10%); build local spatial context within 150 PDF points
   - Output: `List[Anchor]` with balloon location, requirement text bbox (if found), normalized requirement fingerprint, local context spans

5. **Stage 5 - Text Extraction + Alignment (Rev B)** (`io/pdf.py`, `vision/alignment.py:estimate_transform`)
   - Input: Rev B PDF; rendered Rev A & Rev B pages (300 DPI default)
   - Process: Extract Rev B text spans; run ORB feature detection + cross-checked BFMatcher + RANSAC homography on page images
   - Output: `List[TextSpan]` (Rev B), `Transform` with homography matrix H and quality metrics (≥40 inliers, ≥15% inlier ratio)

6. **Stage 6 - Candidate Generation & Assignment** (`reconcile/match.py`)
   - Input: Anchors, Rev B spans, homography transform
   - Process: For each anchor, apply homography to predict Rev B location; search within 144 PDF points; score candidates via location (35% weight), text/numeric overlap (50% weight), context Jaccard (15% weight); greedy bipartite assignment to maximize total score without span reuse
   - Output: `Dict[int, Match]` mapping char_no to selected `Candidate` with scores and matched span

7. **Stage 7 - Classification + Evidence** (`reconcile/classify.py`, `vision/snippets.py`, `reconcile/tolerance_pdf.py`)
   - Input: Anchors, matches, tolerance comparisons, page images
   - Process: For each anchor, classify status (unchanged/changed/removed/added/uncertain) via decision tree on numeric value match, count tokens, numeric overlap; detect unmatched Rev B spans as "added" characteristics; extract evidence snippets (center on requirement bbox, expand to include adjacent symbols/tolerances)
   - Output: `List[DeltaItem]` with status, confidence, reasons, component scores, Rev A/Rev B `Evidence` (page, bbox, image path)

8. **Stage 8 - Output Serialization** (`types.py`, `cli.py`)
   - Input: Classified delta items, evidence snippets, run metadata
   - Process: Wrap in `DeltaPacket` Pydantic model; serialize to JSON with indentation
   - Output: `delta_packet.json` at `out/<run_id>/delta_packet.json`

**State Management:**
- **Transient State:** Homography matrix stored in memory for coordinate transformation
- **Persistent State:** Rev A PDF (source of truth for balloons and text locations)
- **Output State:** DeltaPacket JSON, evidence PNG snippets, debug tolerance JSON (optional)

## Key Abstractions

**Anchor:**
- Purpose: Immutable spatial-semantic link between Form 3 requirement and Rev A drawing
- Examples: `delta_preservation/reconcile/anchors.py:Anchor` (dataclass)
- Pattern: Combines balloon detection (computer vision confidence) + text matching (semantic scoring) to establish identity before cross-revision matching; local context included to support spatial fingerprinting in Rev B

**Candidate:**
- Purpose: Scored Rev B text span as potential match for Rev A characteristic
- Examples: `delta_preservation/reconcile/match.py:Candidate` (dataclass)
- Pattern: Multi-component scoring (location 35%, text 50%, context 15%) with transparent reasons; enables human review of matching decisions before delta classification

**MatchFingerprint:**
- Purpose: Deterministic representation of requirement text for structured matching
- Examples: `delta_preservation/reconcile/normalize.py:MatchFingerprint` (dataclass)
- Pattern: Extracts symbols (Ø, R, ±), count patterns (2X, 6X), numeric values, and pattern class to enable semantic matching robust to formatting variations and PDF text fragmentation

**Transform:**
- Purpose: Quality-assessed geometric transformation between drawing revisions
- Examples: `delta_preservation/vision/alignment.py:Transform` (dataclass)
- Pattern: Wraps homography matrix with inlier metrics; enables spatial coordinate transformation and quality validation (≥40 inliers, ≥15% ratio as hard thresholds)

**Evidence:**
- Purpose: Visual and spatial proof of characteristic location for human review
- Examples: `delta_preservation/types.py:Evidence` (Pydantic model)
- Pattern: Page index + PDF bbox + relative image path; enables traceability without embedding full images; bbox enables navigation back to original PDF

**DeltaPacket:**
- Purpose: Complete audit trail of revision comparison results
- Examples: `delta_preservation/types.py:DeltaPacket` (Pydantic model)
- Pattern: Wraps run_id (timestamp + file hash), input metadata, and list of DeltaItems; JSON-serializable for downstream quality management systems

## Entry Points

**Command-Line Interface:**
- Location: `delta_preservation/cli.py:main()`
- Triggers: `uv run python run.py part1` (wrapper in `run.py`) or direct `delta_preservation/cli.py` execution
- Responsibilities: Parse arguments (PDF paths, Form 3 path, DPI, part name), validate files, call `run_pipeline()`, print progress

**Pipeline Orchestration:**
- Location: `delta_preservation/cli.py:run_pipeline()`
- Triggers: Called by `main()` or directly by test/automation scripts
- Responsibilities: Sequence all 8 stages, manage output directory structure, aggregate results, serialize to DeltaPacket JSON

## Error Handling

**Strategy:** Fail fast with informative error messages; graceful degradation in matching/classification stages

**Patterns:**

- **File Validation:** Check existence and extension before processing; raise `FileNotFoundError`, `ValueError` with clear messages (`cli.py` lines 73-86)

- **Alignment Failures:** `AlignmentError` when ORB features insufficient or RANSAC homography estimation fails; includes inlier count and ratio in message for debugging (`vision/alignment.py` lines 76-131)

- **Snippet Generation Failures:** Try-except blocks around image cropping; record Evidence with bbox but null image_path when generation fails (`cli.py` lines 380-414)

- **Text Extraction Edge Cases:** Graceful handling when:
  - No balloons detected (anchor list empty, max_char_no defaults to 0)
  - No requirement text found in Rev A (req_bbox = None, uses balloon fallback for spatial context)
  - No candidates within search radius (status = "removed")
  - PDF text parsing succeeds but text spans are fragmented (matched across multiple spans via local context)

- **Coordinate Transformation Errors:** Clamp transformed bbox to page boundaries when perspective transforms corners outside page dimensions

## Cross-Cutting Concerns

**Logging:**
- Simple print statements to stdout with progress markers `[N/8]` and result counts
- Debug output files in `out/<run_id>/debug/` (Form 3 parsing, tolerance comparisons)
- No structured logging framework; suitable for CLI-only execution

**Validation:**
- PDF coordinate bounds checking when rendering snippets
- Inlier thresholds enforced in alignment quality assessment (`MIN_ALIGNMENT_INLIERS=40`, `MIN_ALIGNMENT_RATIO=0.15`)
- Characteristic number presence check (Form 3 → balloons → anchors filtering)

**Authentication:**
- Not applicable; reads local files only, no remote API calls

**Coordinate Spaces:**
- **PDF space:** Points at 72 DPI (PyMuPDF standard); origin top-left; used throughout IO and reconciliation layers
- **Image space:** Pixels at render DPI (default 300); origin top-left; used in vision layer
- **Conversion:** `pdf_to_img_coords(bbox_pdf, page, dpi)` and implicit homography application in `alignment.py`
- Critical mapping: `zoom = dpi / 72.0` in `io/pdf.py:render_page()` lines 77-78

**Tolerance Handling:**
- Extracted from PDF annotation groups via pattern matching (inline ±, signed pairs, stacked bilateral/unilateral)
- Normalized to `(upper_limit, lower_limit)` floats
- Compared between Rev A and Rev B to inform "changed" status (numeric tolerance variation)
- Debug JSON exported to `out/<run_id>/debug/tolerance_parsing_tests.json`

---

*Architecture analysis: 2026-02-25*
