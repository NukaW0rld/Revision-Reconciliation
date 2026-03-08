# Codebase Concerns

**Analysis Date:** 2026-02-25

## Tech Debt

### Complex Tolerance Parsing Logic
- **Issue:** Tolerance parsing in `reconcile/tolerance_pdf.py` (1080 lines) is monolithic with multiple regex patterns and heuristic-based span detection. The file handles inline, stacked bilateral, stacked unilateral, and limits-stacked formats with substantial duplication across parsing functions.
- **Files:** `delta_preservation/reconcile/tolerance_pdf.py`
- **Impact:** Hard to maintain, test, and extend. Edge cases in tolerance detection could silently produce incorrect limits, affecting delta classification accuracy. High cyclomatic complexity in stacked pair detection (`_detect_best_stacked_pair`, `_parse_stacked_tolerance_v2`).
- **Fix approach:** Refactor into smaller, composable tolerance parsers with dedicated unit tests. Consider grammar-based parsing instead of regex chains. Extract span grouping logic into separate module.

### Hardcoded Search Radii and Thresholds
- **Issue:** Multiple constant values scattered throughout codebase: `SEARCH_RADIUS=144.0` in `match.py`, `CONTEXT_WINDOW=50.0`, balloon detection heuristics (`< 3 balloons`), alignment quality thresholds (`MIN_ALIGNMENT_INLIERS=40`, `MIN_ALIGNMENT_RATIO=0.15`).
- **Files:** `config.py`, `reconcile/match.py`, `reconcile/anchors.py`, `vision/balloons.py`, `vision/alignment.py`
- **Impact:** Values are calibrated for specific drawing sizes/resolutions. Changing DPI or page dimensions breaks assumptions. No systematic way to tune for different drawing types (schematics, PCB, mechanical). Hard to validate if constants are optimal.
- **Fix approach:** Create configuration profiles by drawing type (mechanical, schematic, PCB). Add validation layer that warns if DPI/page size differs significantly from calibration baseline. Document empirical derivation of each threshold.

### Fragile Coordinate System Conversions
- **Issue:** Multiple coordinate transformations (PDF points → image pixels → transformed space) scattered across `io/pdf.py`, `vision/alignment.py`, `vision/bbox_utils.py`, and `cli.py`. Scale factor calculation (`dpi / 72.0`) appears in multiple places. No centralized coordinate mapping layer.
- **Files:** `io/pdf.py`, `vision/alignment.py`, `vision/bbox_utils.py`, `cli.py` (lines 379, 399)
- **Impact:** Easy to forget about coordinate space in new code. Off-by-one or scale factor errors silently corrupt bbox positions, leading to incorrect snippets or failed matches. Current approach requires developers to track coordinates manually.
- **Fix approach:** Create `CoordinateSpace` enum and `Transform` classes (extend existing `Transform` in alignment.py). Require explicit conversions with type hints. Add validation that bbox values are within expected ranges.

### Classification Decision Tree Heuristics
- **Issue:** Delta classification in `reconcile/classify.py` uses empirically-tuned decision tree with hard-coded weights (0.4, 0.5, 0.1 for location/numeric/padding scores). Rules are applied sequentially with no clear priority ordering. Tolerance comparison is bolted-on at the end rather than integrated into primary logic.
- **Files:** `delta_preservation/reconcile/classify.py` (lines 155-221)
- **Impact:** Adding new signals (e.g., image-based similarity) requires careful rebalancing. Small changes to score weights can flip classification results. No way to validate decisions against ground truth. Confidence scores are partially arbitrary (e.g., `0.85` for notes blocks).
- **Fix approach:** Extract weights to configuration. Build unit test suite with known examples (unchanged, changed, removed, uncertain). Consider tree-based classifier (scikit-learn DecisionTreeClassifier) with training on annotated data. Make tolerance comparison a first-class signal, not post-hoc.

## Known Bugs

### Incomplete Error Handling in Pipeline Stages
- **Symptoms:** Pipeline can fail silently at stages 5-6 (alignment, candidate generation). If `estimate_transform()` raises `AlignmentError`, the error propagates with minimal context about which revisions/pages failed. No retry or fallback behavior.
- **Files:** `cli.py` (lines 136), `vision/alignment.py` (lines 97-131)
- **Trigger:** Run pipeline on drawings with low feature overlap (blank pages, heavily modified regions, or low-contrast PDFs).
- **Workaround:** Manually check `transform.inliers` and `transform.inlier_ratio` after stage 5. Re-render at higher DPI if alignment fails.
- **Fix approach:** Catch `AlignmentError` in `run_pipeline()`, attempt re-alignment at higher DPI. If still fails, switch to fallback (identity transform with warning). Log detailed diagnostics to debug output.

### Excluded Characteristics Not Tracked
- **Symptoms:** Characteristics in Form 3 that don't have corresponding balloons are silently dropped. No warning or report of which characteristics were excluded, making it unclear if Form 3 is incomplete or balloons were missed.
- **Files:** `reconcile/anchors.py` (line 98-99)
- **Trigger:** Run pipeline with Form 3 containing additional characteristics not ballooned in Rev A.
- **Workaround:** Compare Form 3 char_no set against detected balloons manually to identify missing characteristics.
- **Fix approach:** Log dropped characteristics to debug output. Generate report of matched/unmatched characteristics in delta packet. Provide confidence threshold for inclusion (e.g., characteristics with confidence < 0.3 are logged but not processed).

### Span Matching Gaps in Text Extraction
- **Symptoms:** Some requirement text may not be found in PDF text spans (rare symbols, custom fonts, embedded images). When `find_best_span_for_requirement()` fails to locate annotation, system falls back to balloon bbox alone, producing oversized/poorly-framed snippets.
- **Files:** `cli.py` (lines 218-229), `vision/bbox_utils.py`
- **Trigger:** PDFs with custom symbols (GD&T callouts, proprietary notations), or text rendered as image overlays rather than selectable text.
- **Workaround:** Manually crop snippet regions or increase search radius in `find_best_span_for_requirement()`.
- **Fix approach:** Add OCR fallback (Tesseract) for text spans that cannot be located via PDF extraction. Log which characteristics required OCR to enable manual review. Track OCR confidence to adjust delta classification confidence.

## Security Considerations

### No Input Validation for Untrusted PDFs
- **Risk:** Pipeline accepts any PDF file without validation. Malformed PDFs could cause `fitz.open()` to hang, crash, or consume excessive memory. No timeouts on PDF operations.
- **Files:** `io/pdf.py` (line 67), `vision/balloons.py` (line 91), `cli.py` (lines 73-86)
- **Current mitigation:** File extension checks (`*.pdf`), path existence checks. No file size limits or PDF sanitization.
- **Recommendations:**
  - Add file size limits (e.g., max 500 MB) before attempting to open PDF.
  - Implement timeout wrapper around `fitz.open()` and page rendering operations.
  - Validate PDF structure before processing (check magic bytes, page count).
  - Consider sandboxing PDF operations in separate process to isolate crashes.

### Output Directory Traversal Risk
- **Risk:** Output directory path is user-supplied. If attacker controls `out_dir` parameter, could write outside intended directory via path traversal (e.g., `../../etc/passwd`).
- **Files:** `cli.py` (lines 95-101)
- **Current mitigation:** Path is normalized via `Path()` constructor, which provides basic traversal protection. Directory is created with `mkdir(parents=True)`.
- **Recommendations:**
  - Validate that `out_dir` is within expected root directory. Reject paths containing `..` or absolute paths outside allowed base.
  - Use `Path.resolve()` and check that resulting path is within allowed base directory.

### No Validation of Excel File Contents
- **Risk:** `load_form3()` accepts any XLSX file. Malicious Excel could contain formulas or macros, or extremely large sheets causing denial of service.
- **Files:** `io/xlsx.py` (line 15, openpyxl usage)
- **Current mitigation:** `openpyxl` does not execute macros by default. No file size limits.
- **Recommendations:**
  - Add file size limit before parsing Excel.
  - Validate sheet count and cell count limits.
  - Reject sheets with external links or formulas (set `openpyxl.load_workbook(..., data_only=True)`).

## Performance Bottlenecks

### Image Rendering at High DPI
- **Problem:** Pages are rendered at full DPI (default 300) for alignment and candidate generation. Large drawings (20+ MB at 300 DPI) cause memory spikes and slow down alignment feature detection.
- **Files:** `io/pdf.py` (lines 78-82), `cli.py` (lines 134-135, 196-197)
- **Cause:** ORB feature detection works on full-resolution images. For typical A1/A2 drawings, 300 DPI produces 5000x7000 pixel images (~75 MB in memory).
- **Improvement path:**
  - Implement adaptive DPI: detect page size and scale DPI accordingly (e.g., 150 DPI for large drawings).
  - Add downsampling pass for alignment (use 150 DPI for `estimate_transform()`, then use 300 DPI for snippet generation).
  - Cache rendered images per run_id to avoid re-rendering if processing same pair multiple times.
  - Profile memory usage and log warnings if per-page rendering exceeds 50 MB.

### Candidate Scoring Loops
- **Problem:** `score_candidate()` is called for every span within search radius for every characteristic. For dense drawings (1000+ spans), this produces O(N*M) comparisons (N characteristics × M spans per search radius). Each score call includes text parsing, fingerprint generation, Jaccard similarity computation.
- **Files:** `reconcile/match.py` (lines 146-157, 160-338)
- **Cause:** No spatial indexing. Linear search for candidate pool within `SEARCH_RADIUS`. Multiple tokenization passes per span.
- **Improvement path:**
  - Build spatial index (KD-tree or R-tree) for text spans keyed by (x, y). Query index for candidates instead of linear search.
  - Cache `parse_requirement()` results per span across multiple anchor comparisons.
  - Batch fingerprint generation for all spans upfront, reuse during scoring.
  - Consider approximate nearest neighbor (ANN) search for very dense drawings.

### Tolerance Parsing Overhead
- **Problem:** `extract_tolerances_for_items()` calls `parse_pdf_tolerance()` for each characteristic, which in turn calls multiple regex patterns and span grouping functions. No caching of parsed tolerances across multiple uses.
- **Files:** `reconcile/tolerance_pdf.py`
- **Cause:** Tolerance parsing is not amortized. Stacked pair detection re-scans all nearby spans multiple times.
- **Improvement path:**
  - Cache `parse_pdf_tolerance()` results per span key (block_id, line_id, span_id).
  - Batch all tolerance parsing upfront during stage 1 or 3, attach to Anchor objects.
  - Profile regex performance; consider compiled regex patterns or state machine approach.

## Fragile Areas

### Balloon Detection Hybrid Strategy
- **Files:** `vision/balloons.py` (lines 65-119)
- **Why fragile:** System switches to CV fallback if text-based detection finds < 3 balloons. This heuristic assumes all drawings have ≥ 3 balloons, which fails for single-characteristic drawings or test documents. CV fallback uses HoughCircles with fixed radius range and Tesseract OCR, which may have high false positive rates on drawings with circular annotations (dimension leaders, repeat patterns, hole callouts).
- **Safe modification:** Add configuration for minimum balloon threshold (default 3, but allow override). Log which detection method was used per balloon for transparency. Add confidence filtering to CV detections to reduce false positives. Test fallback on variety of drawing types before raising threshold.
- **Test coverage:** No unit tests for detection methods. Manual verification on 2-3 test cases only. Edge cases unknown (blank drawings, highly stylized balloons, overlapping balloons).

### Anchor Building Text Matching
- **Files:** `reconcile/anchors.py` (lines 55-211)
- **Why fragile:** Anchor building combines balloon detection with fuzzy text matching to locate requirement text. Scoring uses token overlap + distance + size heuristics. If text normalization changes, or if similar requirement text exists near balloon, wrong span may be selected as annotation.
- **Safe modification:** Add confidence threshold to anchor building. Reject anchors with req_bbox confidence < 0.5, fall back to balloon-only anchoring. Log which anchors have low confidence. Test on drawings with repeated geometry (e.g., "6X Ø3.5") where multiple identical annotations exist.
- **Test coverage:** No unit tests. Relies on end-to-end testing with specific test PDFs.

### Tolerance Comparison Logic
- **Files:** `reconcile/tolerance_pdf.py` (lines 109-150), `reconcile/classify.py` (lines 201-220)
- **Why fragile:** Tolerance parsing relies on specific formatting. A single extra space, decimal point variation (`.1` vs `0.1`), or unit abbreviation change breaks parsing. The post-hoc tolerance refinement in `classify_delta()` can flip classification from "unchanged" to "changed" based on epsilon threshold (0.01 by default), which may not be appropriate for all unit systems (metric vs imperial).
- **Safe modification:** Increase epsilon tolerance to account for rounding in Excel exports. Log tolerance parsing failures for manual review. Add unit information to `PdfTolerance` to compute epsilon appropriately. Test on Form 3 files from different sources and software packages.
- **Test coverage:** Tolerance parsing has debug JSON output (`tolerance_parsing_tests.json`), but no automated validation of results.

## Scaling Limits

### Single-Page Assumption
- **Current capacity:** Pipeline processes page 0 only. Multi-page drawings are not supported.
- **Limit:** If drawing spans multiple pages, only first page is analyzed. Characteristics on pages 2+ are missed entirely.
- **Scaling path:**
  - Extend pipeline to loop over all pages (lines 120, 130 in `cli.py` hardcode `page_index=0`).
  - Modify anchor building to handle page-specific balloon/text searches.
  - Update coordinate systems to include page numbers.
  - Test on multi-page drawing sets.

### Memory Usage with Large Drawings
- **Current capacity:** Tested with single-page A1 drawings (~2000 text spans, 20-50 balloons).
- **Limit:** Very large multi-page technical manuals (100+ pages) or very dense PCB schematics (1000+ spans per page) may cause memory exhaustion during text extraction or alignment.
- **Scaling path:**
  - Implement streaming text extraction (extract one page at a time, process, discard).
  - Use memory profiling to identify peak usage. Consider generators instead of loading all spans upfront.
  - For multi-page support, process pages in parallel (one process per page) to parallelize I/O and rendering.

### Runtime Performance
- **Current capacity:** Typical 2-page comparison completes in 10-30 seconds (8 cores, 16 GB RAM).
- **Limit:** If scaling to 100+ page comparisons or batch processing many drawing pairs, pipeline may timeout or become bottleneck in larger automation system.
- **Scaling path:**
  - Implement page-level parallelization.
  - Cache alignment homography per PDF pair (avoid re-aligning if run again).
  - Profile hotspots and optimize loops in `score_candidate()` and `parse_pdf_tolerance()`.

## Dependencies at Risk

### PyMuPDF (fitz) Stability
- **Risk:** PyMuPDF is a wrapper around MuPDF C library. Updates occasionally have breaking API changes. Currently pinned to specific version range in `pyproject.toml` but no version lock.
- **Impact:** If fitz API changes (e.g., return types for pixmap data), text extraction breaks.
- **Migration plan:** Monitor PyMuPDF releases. Create integration tests that verify pixmap data format and text extraction output. Consider pdfplumber as alternative (pure Python, more stable API).

### Pydantic Version Constraint
- **Risk:** Pinned to `pydantic>=2.5,<3.0`. Pydantic 3.0 is in development and will have breaking changes.
- **Impact:** When Pydantic 3.0 releases, serialization/validation may break (e.g., `model_dump_json()` method signature, validation hooks).
- **Migration plan:** Test with Pydantic 3.0 beta releases. Create compatibility layer for model serialization. Plan upgrade when Pydantic 3.0 is stable.

### OpenCV Behavior Changes
- **Risk:** OpenCV ORB feature detector parameters (`nfeatures=4000`, `ransacReprojThreshold=3.0`) are sensitive to image quality. Version updates or platform differences (Linux vs Windows) may produce different feature counts.
- **Impact:** Alignment quality may vary across platforms/versions, making alignment thresholds brittle.
- **Migration plan:** Document empirical validation of alignment parameters on test set. Lock OpenCV version. Consider deterministic feature detector (e.g., SIFT if license is obtained) for reproducibility.

## Missing Critical Features

### No Multi-Page Support
- **Problem:** Only first page of drawing is processed. Drawings that span multiple pages cannot be fully analyzed.
- **Blocks:** Any drawing revision comparison requiring analysis of pages 2-10+ (common for complex assemblies).
- **Priority:** High - limits applicability to real engineering drawings.

### No Manual Annotation/Correction Interface
- **Problem:** If system mismatches characteristics (uncertain classification, missed balloons), there is no way to manually correct results before finalizing delta packet. System outputs are final.
- **Blocks:** Quality assurance workflows that require human review and correction.
- **Priority:** Medium - impacts accuracy but system can still provide useful preliminary results.

### No Batch Processing
- **Problem:** Pipeline is designed for single-pair comparisons. To process 100 drawing revisions (50 pairs), requires 50 manual CLI invocations.
- **Blocks:** Automated end-to-end FAI report generation for production runs.
- **Priority:** Medium - adds operational friction for large-scale use.

### No Baseline Comparison
- **Problem:** System always compares Rev A vs Rev B. Cannot compare multiple revisions to see evolution of characteristics across versions (Rev A → Rev B → Rev C).
- **Blocks:** Tracking characteristic stability over multiple revisions.
- **Priority:** Low - not required for basic FAIR compliance but useful for analysis.

## Test Coverage Gaps

### No Unit Tests for Core Matching Logic
- **What's not tested:** `score_candidate()`, `generate_candidates()`, `assign_matches()` matching algorithm. No test fixtures with known good/bad matches. Scoring weights are not validated against ground truth.
- **Files:** `reconcile/match.py`
- **Risk:** Changes to scoring formula or weights can silently break matching accuracy. No automated validation that new code maintains or improves match quality.
- **Priority:** High - core algorithm should have regression tests.

### No Unit Tests for Tolerance Parsing
- **What's not tested:** `parse_pdf_tolerance()` and regex patterns for various tolerance formats. No test cases for edge cases (missing units, stacked with line gaps, implicit tolerance signs).
- **Files:** `reconcile/tolerance_pdf.py`
- **Risk:** Tolerance parsing errors silently affect delta classification. Bug fixes or feature additions may break existing tolerance detection on real Form 3 files.
- **Priority:** High - tolerance parsing drives classification decisions.

### No Unit Tests for Text Normalization
- **What's not tested:** `parse_requirement()` fingerprint generation. No validation that similar requirements produce similar fingerprints, or that different requirements are distinguished.
- **Files:** `reconcile/normalize.py`
- **Risk:** Fingerprint changes affect matching and classification. No way to validate fingerprints are stable across code changes.
- **Priority:** Medium - indirectly affects matching quality.

### No Integration Tests with Real PDFs
- **What's not tested:** End-to-end pipeline with actual engineering drawings. Current manual testing uses 2-3 internal test cases only. No validation on variety of drawing styles (mechanical, electrical, PCB, schematic).
- **Files:** `cli.py`
- **Risk:** Pipeline may fail on unfamiliar drawing formats discovered in production.
- **Priority:** High - real-world applicability depends on robustness across drawing types.

### No Regression Tests for Coordinate Transformations
- **What's not tested:** Coordinate system conversions (PDF→image, image→homography, homography→PDF) across different DPI values and page sizes. No validation that transformations are invertible or preserve distances.
- **Files:** `io/pdf.py`, `vision/alignment.py`, `vision/bbox_utils.py`
- **Risk:** Coordinate bugs produce incorrect snippet locations or failed matches. Hard to debug because transformations are scattered across modules.
- **Priority:** Medium - affects snippet accuracy and visual evidence quality.

---

*Concerns audit: 2026-02-25*
