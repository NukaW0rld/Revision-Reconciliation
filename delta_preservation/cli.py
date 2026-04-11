"""
Command-line interface and main pipeline orchestration for delta preservation.

This module provides the primary entry point for the revision reconciliation system,
orchestrating all pipeline stages from input validation through final output generation.
The main() function implements an 8-stage pipeline that processes engineering drawing
revisions and produces structured delta analysis results.

The CLI handles file validation, output directory management, and progress reporting
while delegating specialized processing to domain-specific modules.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import cv2

from delta_preservation.io.pdf import render_page, extract_text_spans, pdf_to_img_coords, TextSpan, join_text_spans
import fitz
from delta_preservation.evaluation import load_ground_truth_packet
from delta_preservation.evaluation.conformance import (
    apply_accepted_alternate_history,
    evaluate_packet_against_truth,
)
from delta_preservation.io.xlsx import load_form3
from delta_preservation.vision.grid import infer_grid, parse_zone_from_reference_location
from delta_preservation.vision.balloons import detect_balloons
from delta_preservation.vision.alignment import (
    estimate_transform,
    estimate_transform_from_text_spans,
    estimate_transform_candidates_from_text_spans,
    _homography_is_near_identity,
)
from delta_preservation.vision.snippets import crop_with_padding, save_snippet
from delta_preservation.vision.bbox_utils import (
    compute_combined_evidence_bbox,
    expand_bbox_with_adjacent_spans,
    normalize_snippet_size,
    expand_bbox_coverage,
    union_bbox,
    find_best_span_for_requirement,
    expand_notes_block
)
from delta_preservation.reconcile.anchors import build_revA_anchors
from delta_preservation.reconcile.match import _group_candidate_spans, generate_candidates, assign_matches, refine_match_display_text
from delta_preservation.reconcile.classify import classify_delta, detect_added_characteristics, DeltaItem as DeltaItemInternal
from delta_preservation.reconcile.tolerance_pdf import export_run_tolerance_debug, extract_tolerances_for_items
from delta_preservation.reconcile.normalize import extract_semantic_callout
from delta_preservation.types import (
    AcceptedAlternateHistoryRecord,
    DeltaPacket,
    DeltaItem,
    Evidence,
    SemanticCallout,
)


_REORDER_SYMBOLS = {"↧", "⌴", "⌵", "⏥", "⌓", "⟂", "⌖", "⌒", "∥", "∠", "⊙", "Ø", "R"}
_GDT_SYMBOLS = {"⏥", "⌓", "⟂", "⌖", "⌒", "∥", "∠", "⊙"}


def _span_key(span: TextSpan) -> tuple:
    return (span.block_id, span.line_id, span.span_id, span.bbox_pdf)


def _dedupe_spans(spans: List[TextSpan]) -> List[TextSpan]:
    unique = []
    seen = set()
    for span in spans:
        key = _span_key(span)
        if key in seen:
            continue
        seen.add(key)
        unique.append(span)
    return unique


def _expand_grouped_annotation_spans(base_spans: List[TextSpan], all_spans: List[TextSpan]) -> List[TextSpan]:
    """Rebuild local companion closure for reviewer-facing packet assembly.

    The matcher normally emits GroupedSpan.source_spans, but downstream tests may
    inject a partial grouped span to prove packet assembly can still recover the
    full local annotation from page spans. Walk closure from each base span and
    return a deduped set in deterministic order.
    """
    expanded: List[TextSpan] = []
    for span in base_spans:
        grouped = _group_candidate_spans(span, all_spans)
        expanded.extend(getattr(grouped, "source_spans", None) or [span])

    ordered = _dedupe_spans(expanded)
    ordered.sort(key=lambda span: (span.bbox_pdf[1], span.bbox_pdf[0], span.block_id, span.line_id, span.span_id))
    return ordered


def _spans_union_bbox(spans: List[TextSpan]) -> Optional[tuple[float, float, float, float]]:
    if not spans:
        return None
    bbox = spans[0].bbox_pdf
    for span in spans[1:]:
        bbox = union_bbox(bbox, span.bbox_pdf)
    return bbox


def _spans_in_bbox(all_spans: List[TextSpan], bbox: tuple[float, float, float, float], tolerance: float = 2.0) -> List[TextSpan]:
    x0, y0, x1, y1 = bbox
    selected = []
    for span in all_spans:
        sx0, sy0, sx1, sy1 = span.bbox_pdf
        if sx0 >= x0 - tolerance and sy0 >= y0 - tolerance and sx1 <= x1 + tolerance and sy1 <= y1 + tolerance:
            selected.append(span)
    return _dedupe_spans(selected)


def _format_annotation_text(spans: List[TextSpan], fallback_text: Optional[str] = None) -> Optional[str]:
    """Render a compact, human-readable annotation string from PDF spans."""
    spans = [span for span in _dedupe_spans(spans) if span.text and span.text.strip()]
    if not spans:
        return fallback_text

    ordered = sorted(spans, key=lambda span: (span.bbox_pdf[1], span.bbox_pdf[0], span.block_id, span.line_id, span.span_id))
    texts = [span.text.strip() for span in ordered]

    # Notes blocks should preserve full row-major content, not just the matched header span.
    if any("NOTE" in text.upper() for text in texts):
        return join_text_spans(ordered)

    # Group spans into visual rows.
    rows: List[List[TextSpan]] = []
    for span in ordered:
        cy = (span.bbox_pdf[1] + span.bbox_pdf[3]) / 2
        placed = False
        for row in rows:
            row_cy = sum((s.bbox_pdf[1] + s.bbox_pdf[3]) / 2 for s in row) / len(row)
            if abs(cy - row_cy) <= 8.0:
                row.append(span)
                placed = True
                break
        if not placed:
            rows.append([span])
    for row in rows:
        row.sort(key=lambda span: (span.bbox_pdf[0], span.block_id, span.line_id, span.span_id))

    # Vertical diameter / limit stack: Ø above/below numeric limits.
    flat_texts = [span.text.strip() for row in rows for span in row]
    pure_numeric = [text for text in flat_texts if re.fullmatch(r"[+−\-]?\d+(?:\.\d+)?", text)]
    if any(text == "Ø" for text in flat_texts) and len(pure_numeric) >= 2:
        numeric_rows = [row for row in rows if any(re.fullmatch(r"[+−\-]?\d+(?:\.\d+)?", s.text.strip()) for s in row)]
        if len(numeric_rows) >= 2:
            top_num = next((s.text.strip() for s in numeric_rows[0] if re.fullmatch(r"[+−\-]?\d+(?:\.\d+)?", s.text.strip())), None)
            bottom_num = next((s.text.strip() for s in numeric_rows[1] if re.fullmatch(r"[+−\-]?\d+(?:\.\d+)?", s.text.strip())), None)
            prefix = next((text for text in flat_texts if re.fullmatch(r"\d+X", text.upper())), None)
            if top_num and bottom_num:
                body = f"Ø{top_num} / Ø{bottom_num}"
                return f"{prefix} {body}" if prefix else body

    row_texts: List[str] = []
    for row in rows:
        row_tokens = [span.text.strip() for span in row if span.text.strip()]
        if not row_tokens:
            continue

        prefix_tokens = [token for token in row_tokens if re.fullmatch(r"\d+X", token.upper())]
        prefix = " ".join(prefix_tokens).strip()
        core_tokens = [token for token in row_tokens if token not in prefix_tokens]

        leading_symbols = [token for token in core_tokens if token in _REORDER_SYMBOLS or any(c in token for c in _REORDER_SYMBOLS)]
        trailing_tokens = [token for token in core_tokens if token not in leading_symbols]

        if any(any(c in token for c in _GDT_SYMBOLS) for token in core_tokens):
            gdt = [token for token in core_tokens if any(c in token for c in _GDT_SYMBOLS)]
            trailing_tokens = [token for token in core_tokens if not any(c in token for c in _GDT_SYMBOLS)]
            row_text = " ".join(gdt + trailing_tokens)
        elif leading_symbols and trailing_tokens:
            row_text = " ".join(leading_symbols + trailing_tokens)
        else:
            row_text = " ".join(core_tokens)

        if prefix:
            row_text = f"{prefix} {row_text}" if row_text else prefix
        row_texts.append(row_text.strip())

    combined = " / ".join(text for text in row_texts if text)
    return combined or fallback_text or join_text_spans(ordered)


def run_pipeline(
    revA_pdf,
    revB_pdf,
    form3_xlsx,
    out_dir="./out",
    dpi=300,
    part_name="part",
    accepted_alternates: Optional[List[AcceptedAlternateHistoryRecord]] = None,
    stage_callback: Optional[Callable[[int, str], None]] = None,
) -> Path:
    """
    Run the complete delta preservation pipeline.

    Orchestrates all 8 pipeline stages from input validation through final output:
    1. Load and parse AS9102 Form 3 inspection characteristics
    2. Detect characteristic balloons in Rev A drawing
    3. Extract text spans from Rev A for spatial anchoring
    4. Build spatial-semantic anchors linking Form 3 to Rev A locations
    5. Extract Rev B text spans and estimate geometric alignment
    6. Generate candidate matches and assign Rev A characteristics to Rev B spans
    7. Classify changes and detect new characteristics in Rev B
    8. Generate visual evidence snippets and output structured delta packet

    Returns:
        Path to the run output directory.
    """
    # Validate input files
    revA_path = Path(revA_pdf)
    revB_path = Path(revB_pdf)
    form3_path = Path(form3_xlsx)

    if not revA_path.exists():
        raise FileNotFoundError(f"Revision A PDF not found: {revA_path}")
    if not revA_path.suffix.lower() == ".pdf":
        raise ValueError(f"Revision A must be a PDF file, got: {revA_path.suffix}")

    if not revB_path.exists():
        raise FileNotFoundError(f"Revision B PDF not found: {revB_path}")
    if not revB_path.suffix.lower() == ".pdf":
        raise ValueError(f"Revision B must be a PDF file, got: {revB_path.suffix}")

    if not form3_path.exists():
        raise FileNotFoundError(f"Form 3 XLSX not found: {form3_path}")
    if not form3_path.suffix.lower() == ".xlsx":
        raise ValueError(f"Form 3 must be an XLSX file, got: {form3_path.suffix}")

    # Generate deterministic run_id
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    hash_input = f"{revA_path.absolute()}{revB_path.absolute()}{form3_path.absolute()}"
    short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    run_id = f"{part_name}_{timestamp}_{short_hash}"

    # Create output directory structure
    run_dir = Path(out_dir) / run_id
    snippets_dir = run_dir / "snippets"
    debug_dir = run_dir / "debug"

    run_dir.mkdir(parents=True, exist_ok=False)
    snippets_dir.mkdir(parents=True, exist_ok=False)
    debug_dir.mkdir(parents=True, exist_ok=False)

    print(f"Run ID: {run_id}")
    print(f"Output: {run_dir.absolute()}")
    print()

    # Stage 1: Load Form 3
    if stage_callback:
        stage_callback(0, "Form 3 parsing")
    print("[1/8] Loading Form 3...")
    form3_chars_list = load_form3(form3_path, debug_dir)
    form3_chars = {char.char_no: char.requirement for char in form3_chars_list}
    print(f"  Loaded {len(form3_chars)} characteristics")

    # Infer drawing grid from Rev A border labels and build per-characteristic
    # zone bboxes from Form 3 Field 6 reference locations.  Falls back to None
    # (no zone info) when the grid cannot be detected or when a characteristic's
    # reference location does not contain a zone coordinate.
    drawing_grid = None
    if revA_path.exists() and revA_path.stat().st_size > 1024:
        drawing_grid = infer_grid(revA_path, page_index=0)
    if drawing_grid is not None:
        col_desc = " ".join(f"{lbl}" for lbl, _ in reversed(drawing_grid.col_labels))
        row_desc = " ".join(f"{lbl}" for lbl, _ in drawing_grid.row_labels)
        print(f"  Drawing grid: columns [{col_desc}] rows [{row_desc}]")
    else:
        print("  Drawing grid: not detected (zone-based search disabled)")

    zone_bboxes: Dict[int, tuple] = {}
    if drawing_grid is not None:
        for char in form3_chars_list:
            zone_str = parse_zone_from_reference_location(char.reference_location)
            if zone_str is not None:
                bbox = drawing_grid.zone_to_bbox(zone_str)
                if bbox is not None:
                    zone_bboxes[char.char_no] = bbox
    if zone_bboxes:
        print(f"  Zone bboxes resolved for {len(zone_bboxes)}/{len(form3_chars)} characteristics")

    # Stage 2: Detect balloons in Rev A
    if stage_callback:
        stage_callback(1, "Balloon detection")
    print("[2/8] Detecting balloons in Rev A...")
    balloons = detect_balloons(revA_path, dpi=dpi)
    print(f"  Detected {len(balloons)} balloons")

    # Stage 3: Extract text from Rev A
    if stage_callback:
        stage_callback(2, "Text extraction")
    print("[3/8] Extracting text from Rev A...")
    revA_text_spans = extract_text_spans(revA_path, page_index=0)
    print(f"  Extracted {len(revA_text_spans)} text spans")

    # Stage 4: Build Rev A anchors
    if stage_callback:
        stage_callback(3, "Anchor building")
    print("[4/8] Building Rev A anchors...")
    anchors = build_revA_anchors(form3_chars, balloons, revA_text_spans, zone_bboxes=zone_bboxes)
    print(f"  Built {len(anchors)} anchors")

    # Stage 5: Extract text from Rev B and estimate alignment
    if stage_callback:
        stage_callback(4, "Alignment")
    print("[5/8] Extracting text from Rev B and aligning...")
    revB_text_spans = extract_text_spans(revB_path, page_index=0)
    print(f"  Extracted {len(revB_text_spans)} text spans")

    # Render pages for alignment (use first page for now)
    imgA = render_page(revA_path, page_index=0, dpi=dpi)
    imgB = render_page(revB_path, page_index=0, dpi=dpi)
    orb_transform = estimate_transform(imgA, imgB)
    print(f"  ORB alignment: {orb_transform.inliers} inliers, ratio={orb_transform.inlier_ratio:.2f}")

    # Text-span alignment: use matching annotation texts as point correspondences.
    # This is more reliable than ORB for engineering drawings where the static title
    # block dominates feature matching and can produce a spurious near-identity
    # transform even when annotation content has shifted significantly.
    text_transforms = estimate_transform_candidates_from_text_spans(revA_text_spans, revB_text_spans)
    text_transform = text_transforms[0] if text_transforms else None

    if text_transform is not None:
        print(f"  Text-span alignment: {text_transform.inliers} inliers, ratio={text_transform.inlier_ratio:.2f}")
        tx = text_transform.H[0, 2]
        ty = text_transform.H[1, 2]
        print(f"  Text-span translation: tx={tx:.1f} ty={ty:.1f} PDF pts")
        if len(text_transforms) > 1:
            alt_parts = []
            for idx, alt_transform in enumerate(text_transforms[1:], start=2):
                alt_parts.append(
                    f"T{idx}(tx={alt_transform.H[0, 2]:.1f}, ty={alt_transform.H[1, 2]:.1f}, inliers={alt_transform.inliers})"
                )
            print("  Alternate text-span transforms: " + "; ".join(alt_parts))

        # Use the text-span transform when the ORB transform is near-identity but
        # the text-span transform indicates significant content shift (> 20 PDF pts).
        orb_near_identity = _homography_is_near_identity(orb_transform.H, translation_threshold=20.0)
        text_has_shift = (tx ** 2 + ty ** 2) ** 0.5 > 20.0

        if orb_near_identity and text_has_shift:
            transform = text_transform
            print("  Using text-span alignment (ORB produced near-identity transform)")
        else:
            transform = orb_transform
            print("  Using ORB alignment")
    else:
        transform = orb_transform
        print("  Text-span alignment: insufficient common spans, using ORB alignment")

    def _transform_shift(tf):
        H = getattr(tf, "H", None)
        if H is None:
            return None
        return float(H[0, 2]), float(H[1, 2])

    candidate_transforms: List = [transform]
    for extra_transform in [orb_transform, *text_transforms]:
        extra_shift = _transform_shift(extra_transform)
        if extra_shift is None:
            if extra_transform not in candidate_transforms:
                candidate_transforms.append(extra_transform)
            continue
        if all(
            (_transform_shift(existing) is None)
            or abs(_transform_shift(existing)[0] - extra_shift[0]) > 5.0
            or abs(_transform_shift(existing)[1] - extra_shift[1]) > 5.0
            for existing in candidate_transforms
        ):
            candidate_transforms.append(extra_transform)

    print(f"  Final transform: {transform.inliers} inliers, ratio={transform.inlier_ratio:.2f}")

    # Stage 6: Generate candidates and assign matches
    if stage_callback:
        stage_callback(5, "Candidate matching")
    print("[6/8] Generating candidates and assigning matches...")

    anchor_semantic_callouts: Dict[int, SemanticCallout] = {}
    for anchor in anchors:
        anchor_spans = []
        if anchor.req_bbox is not None:
            anchor_spans = [
                span for span in revA_text_spans
                if span.block_id == anchor.page and span.bbox_pdf == anchor.req_bbox
            ]
        anchor_semantic_callouts[anchor.char_no] = extract_semantic_callout(
            pdf_spans=anchor_spans,
            form3_requirement=anchor.requirement_raw,
        )

    revB_semantic_callouts_by_span_key: Dict[tuple, SemanticCallout] = {}
    for span in revB_text_spans:
        span_key = (span.block_id, span.line_id, span.span_id, span.bbox_pdf)
        revB_semantic_callouts_by_span_key[span_key] = extract_semantic_callout(
            pdf_spans=[span],
            form3_requirement=None,
        )

    candidates_by_anchor = {}
    for anchor in anchors:
        candidates = generate_candidates(
            anchor,
            revB_text_spans,
            candidate_transforms,
            top_k=5,
            anchor_semantic_callout=anchor_semantic_callouts.get(anchor.char_no),
            revB_semantic_callouts_by_span_key=revB_semantic_callouts_by_span_key,
        )
        candidates_by_anchor[anchor.char_no] = candidates

    matches = assign_matches(anchors, candidates_by_anchor)
    refine_match_display_text(anchors, matches)
    print(f"  Assigned {len(matches)} matches")

    # Stage 7: Classify deltas and save snippets
    if stage_callback:
        stage_callback(6, "Classification")
    print("[7/8] Classifying deltas and saving evidence snippets...")

    tolerance_comparisons = extract_tolerances_for_items(
        anchors=anchors, matches=matches,
        revA_spans=revA_text_spans, revB_spans=revB_text_spans,
    )

    delta_items_internal: List[DeltaItemInternal] = []

    # Build a set of span keys for "strong" matches — matches where the anchor's
    # primary numeric value is present in the matched span's numerics.  These spans
    # are legitimately claimed and should NOT be reused by the rescue path in
    # classify_delta.  Limits-form-only matches (text_score≈0, primary not in span)
    # are intentionally excluded so that a sibling anchor can find the correct span
    # via the rescue scan.
    from delta_preservation.reconcile.normalize import parse_requirement as _pr
    _matched_span_keys_strong: set = set()
    for _char_no, _match in matches.items():
        _anchor_obj = next((a for a in anchors if a.char_no == _char_no), None)
        if _anchor_obj is None:
            continue
        _afp = _pr(_anchor_obj.requirement_raw)
        _aprimary = max((v for v, _ in _afp.numeric_tokens), default=None)
        if _aprimary is None:
            continue
        _mspan = _match.candidate.span
        _mfp = _pr(_mspan.text)
        _mnums = {v for v, _ in _mfp.numeric_tokens}
        if _aprimary in _mnums:
            # Anchor primary is present in matched span → strong match
            _mk = (_mspan.block_id, _mspan.line_id, _mspan.span_id, _mspan.bbox_pdf)
            _matched_span_keys_strong.add(_mk)
            for _src_span in getattr(_mspan, "source_spans", []) or []:
                _matched_span_keys_strong.add(
                    (_src_span.block_id, _src_span.line_id, _src_span.span_id, _src_span.bbox_pdf)
                )

    for anchor in anchors:
        match_or_none = matches.get(anchor.char_no)
        tol_cmp = tolerance_comparisons.get(anchor.char_no)
        matched_semantic_callout = None
        if match_or_none is not None:
            matched_span = match_or_none.candidate.span
            # Use source_spans from GroupedSpan if available — individual PDF
            # spans preserve proper text ordering for GD&T symbol recognition.
            semantic_spans = getattr(matched_span, "source_spans", None) or [matched_span]
            matched_semantic_callout = extract_semantic_callout(
                pdf_spans=semantic_spans,
                form3_requirement=anchor.requirement_raw,
            )
            revB_semantic_callouts_by_span_key[
                (matched_span.block_id, matched_span.line_id, matched_span.span_id, matched_span.bbox_pdf)
            ] = matched_semantic_callout
        delta_item_internal = classify_delta(
            anchor,
            match_or_none,
            location_search_coverage=1.0,
            tolerance_comparison=tol_cmp,
            anchor_semantic_callout=anchor_semantic_callouts.get(anchor.char_no),
            matched_semantic_callout=matched_semantic_callout,
            revB_text_spans=revB_text_spans,
            matched_span_keys_strong=_matched_span_keys_strong,
        )
        delta_items_internal.append(delta_item_internal)

    # Open PDF documents for coordinate conversion and page dimensions
    docA = fitz.open(revA_path)
    docB = fitz.open(revB_path)
    pageB_rect = docB.load_page(0).rect
    page_width_b = pageB_rect.width
    page_height_b = pageB_rect.height

    # Detect added characteristics (new in Rev B, not in Rev A)
    max_char_no = max(a.char_no for a in anchors) if anchors else 0
    added_items = detect_added_characteristics(
        revB_text_spans, matches, next_char_no=max_char_no + 1,
        page_width=page_width_b, page_height=page_height_b,
        revA_spans=revA_text_spans,
    )
    delta_items_internal.extend(added_items)
    print(f"  Found {len(added_items)} added characteristics in Rev B")

    export_run_tolerance_debug(
        debug_dir=debug_dir,
        run_id=run_id,
        revA_pdf_path=revA_path,
        revB_pdf_path=revB_path,
        items=delta_items_internal,
        anchors=anchors,
        revA_spans=revA_text_spans,
        revB_spans=revB_text_spans,
    )

    # Convert internal DeltaItems to Pydantic models with Evidence
    delta_items_pydantic: List[DeltaItem] = []

    # Render pages once for all snippets
    imgA_page = render_page(revA_path, page_index=0, dpi=dpi)
    imgB_page = render_page(revB_path, page_index=0, dpi=dpi)
    pageA = docA.load_page(0)
    pageB = docB.load_page(0)

    for delta_internal in delta_items_internal:
        # Find corresponding anchor (may not exist for added items)
        anchor = next((a for a in anchors if a.char_no == delta_internal.char_no), None)

        # Initialize evidence objects
        revA_evidence = None
        revB_evidence = None
        revA_annotation_spans: List[TextSpan] = []

        # Compute bboxes for both revA and revB to enable consistent sizing
        revA_bbox_pdf = None
        revB_bbox_pdf = None

        # --- Compute Rev A bbox (centered on annotation, expanded to include balloon) ---
        if anchor is not None:
            # Try to get annotation bbox - first from anchor, then search as fallback
            annotation_bbox = anchor.req_bbox

            # Fallback: search for annotation span near balloon
            if annotation_bbox is None:
                bx0, by0, bx1, by1 = anchor.balloon_bbox
                balloon_center = ((bx0 + bx1) / 2, (by0 + by1) / 2)
                found_span = find_best_span_for_requirement(
                    requirement_text=anchor.requirement_raw,
                    spans=revA_text_spans,
                    reference_point=balloon_center,
                    search_radius=200.0
                )
                if found_span is not None:
                    annotation_bbox = found_span.bbox_pdf

            if annotation_bbox is not None:
                # Center on the characteristic annotation, like Rev B
                base_bbox_a = annotation_bbox

                # Check if this is a notes-type characteristic
                is_notes_type = "NOTES" in anchor.requirement_raw.upper()

                if is_notes_type:
                    # For notes blocks, expand vertically to include all numbered items
                    expanded_bbox = expand_notes_block(
                        notes_header_bbox=base_bbox_a,
                        all_spans=revA_text_spans,
                        max_vertical_expansion=150.0
                    )
                    revA_annotation_spans = _spans_in_bbox(revA_text_spans, expanded_bbox)
                    # Create a simple wrapper to match the expected structure
                    class ExpandedWrapper:
                        def __init__(self, bbox):
                            self.bbox = bbox
                    expanded = ExpandedWrapper(expanded_bbox)
                else:
                    # Expand bbox to include adjacent spans (symbols like ⌴, ↧, tolerances)
                    expanded = expand_bbox_with_adjacent_spans(
                        center_bbox=base_bbox_a,
                        all_spans=revA_text_spans,
                        horizontal_tolerance=20.0,  # ~0.28 inch gap tolerance
                        vertical_tolerance=8.0,     # ~0.11 inch vertical tolerance
                        max_horizontal_expansion=150.0  # ~2 inch max expansion
                    )
                    revA_annotation_spans = _spans_in_bbox(revA_text_spans, expanded.bbox)

                # Compute annotation center (this will be the snippet center)
                ann_x0, ann_y0, ann_x1, ann_y1 = expanded.bbox
                ann_cx = (ann_x0 + ann_x1) / 2
                ann_cy = (ann_y0 + ann_y1) / 2

                # Get balloon bbox for inclusion in snippet
                bx0, by0, bx1, by1 = anchor.balloon_bbox

                # Distance from annotation center to each edge needed
                half_width = max(
                    ann_cx - ann_x0,  # left edge of annotation
                    ann_x1 - ann_cx,  # right edge of annotation
                    ann_cx - bx0,     # left edge of balloon
                    bx1 - ann_cx,     # right edge of balloon
                    120.0             # minimum half-width
                )
                half_height = max(
                    ann_cy - ann_y0,  # top edge of annotation
                    ann_y1 - ann_cy,  # bottom edge of annotation
                    ann_cy - by0,     # top edge of balloon
                    by1 - ann_cy,     # bottom edge of balloon
                    120.0             # minimum half-height
                )

                # Build bbox centered on annotation
                revA_bbox_pdf = (
                    ann_cx - half_width,
                    ann_cy - half_height,
                    ann_cx + half_width,
                    ann_cy + half_height
                )
            else:
                # Last resort fallback: expand balloon bbox when no annotation found
                bx0, by0, bx1, by1 = anchor.balloon_bbox
                width = bx1 - bx0
                height = by1 - by0
                min_width, min_height = 240.0, 240.0

                # Ensure minimum dimensions
                if width < min_width:
                    expand = (min_width - width) / 2
                    bx0 -= expand
                    bx1 += expand

                if height < min_height:
                    expand = (min_height - height) / 2
                    by0 -= expand
                    by1 += expand

                revA_bbox_pdf = (bx0, by0, bx1, by1)

        # --- Compute Rev B bbox (expand to include adjacent spans) ---
        # Check if this is a notes-type characteristic
        is_notes_type_b = anchor is not None and "NOTES" in anchor.requirement_raw.upper()

        # revB_annotation_spans: all spans that make up the complete annotation text
        # (primary matched span + companion spans such as ↧, ⌴, ⌵, tolerance suffixes).
        # Populated below so that requirement_revB can report the full annotation, not
        # just the single matched span.
        revB_annotation_spans: List = []

        if delta_internal.match is not None:
            span = delta_internal.match.candidate.span
            base_spans = getattr(span, "source_spans", None) or [span]
            if not is_notes_type_b:
                base_spans = _expand_grouped_annotation_spans(list(base_spans), revB_text_spans)
            base_bbox_b = _spans_union_bbox(base_spans) or span.bbox_pdf

            if is_notes_type_b:
                # For notes blocks, expand vertically to include all numbered items
                revB_bbox_pdf = expand_notes_block(
                    notes_header_bbox=base_bbox_b,
                    all_spans=revB_text_spans,
                    max_vertical_expansion=150.0
                )
                revB_annotation_spans = _spans_in_bbox(revB_text_spans, revB_bbox_pdf)
            else:
                # Expand bbox to include adjacent spans (symbols like ⌴, ↧, tolerances)
                expanded = expand_bbox_with_adjacent_spans(
                    center_bbox=base_bbox_b,
                    all_spans=revB_text_spans,
                    horizontal_tolerance=20.0,  # ~0.28 inch gap tolerance
                    vertical_tolerance=8.0,     # ~0.11 inch vertical tolerance
                    max_horizontal_expansion=150.0  # ~2 inch max expansion
                )
                revB_bbox_pdf = expanded.bbox
                # Use source_spans from GroupedSpan if available — individual PDF
                # spans preserve font/position info needed for proper text ordering
                # (e.g., GD&T symbol span before numeric span).
                base_bboxes = {s.bbox_pdf for s in base_spans}
                revB_annotation_spans = list(base_spans) + [
                    s for s in expanded.included_spans
                    if s.bbox_pdf not in base_bboxes
                ]

        elif delta_internal.added_span is not None:
            span = delta_internal.added_span
            base_bbox_b = span.bbox_pdf

            # Expand for added items too
            expanded = expand_bbox_with_adjacent_spans(
                center_bbox=base_bbox_b,
                all_spans=revB_text_spans,
                horizontal_tolerance=20.0,
                vertical_tolerance=8.0,
                max_horizontal_expansion=150.0
            )
            exp_bbox = expanded.bbox
            revB_annotation_spans = [span] + [
                s for s in expanded.included_spans
                if s.bbox_pdf != span.bbox_pdf
            ]

            # Ensure minimum size for added characteristics (120x120 in PDF points)
            min_size = 120.0
            ex0, ey0, ex1, ey1 = exp_bbox
            width = ex1 - ex0
            height = ey1 - ey0
            cx, cy = (ex0 + ex1) / 2, (ey0 + ey1) / 2

            if width < min_size:
                ex0 = cx - min_size / 2
                ex1 = cx + min_size / 2
            if height < min_size:
                ey0 = cy - min_size / 2
                ey1 = cy + min_size / 2

            revB_bbox_pdf = (ex0, ey0, ex1, ey1)

        elif delta_internal.status == "removed" and anchor is not None and revA_bbox_pdf is not None:
            try:
                from delta_preservation.vision.alignment import apply_transform_bbox
                revB_bbox_pdf = apply_transform_bbox(revA_bbox_pdf, transform.H)
            except Exception:
                revB_bbox_pdf = None  # Fallback: "Not found in Rev B" placeholder shown in review card

        # Added characteristics have no Rev A evidence counterpart. Leave revA_evidence
        # empty rather than projecting the Rev B bbox backward into a misleading blank area.

        # --- Normalize sizes for consistent paired snippets ---
        if revA_bbox_pdf is not None and revB_bbox_pdf is not None:
            revA_bbox_pdf, revB_bbox_pdf = normalize_snippet_size(revA_bbox_pdf, revB_bbox_pdf)

        # --- Expand snippet bboxes to 2x coverage for more context ---
        if revA_bbox_pdf is not None:
            revA_bbox_pdf = expand_bbox_coverage(revA_bbox_pdf, scale_factor=2.0)
        if revB_bbox_pdf is not None:
            revB_bbox_pdf = expand_bbox_coverage(revB_bbox_pdf, scale_factor=2.0)

        # --- Generate Rev A snippet ---
        if revA_bbox_pdf is not None:
            page_a = anchor.page if anchor is not None else 0
            bbox_img_a = pdf_to_img_coords(revA_bbox_pdf, pageA, dpi=dpi)
            try:
                crop_a = crop_with_padding(imgA_page, bbox_img_a, pad_px=10)
                filename_a = save_snippet(crop_a, snippets_dir, delta_internal.char_no, "revA", page_a)
                revA_evidence = Evidence(
                    page=page_a,
                    bbox=list(revA_bbox_pdf),
                    image_path=f"snippets/{filename_a}"
                )
            except ValueError as e:
                # Bbox invalid, record evidence without image
                revA_evidence = Evidence(
                    page=page_a,
                    bbox=list(revA_bbox_pdf),
                    image_path=None
                )

        # --- Generate Rev B snippet ---
        if revB_bbox_pdf is not None:
            page_b = 0
            bbox_img_b = pdf_to_img_coords(revB_bbox_pdf, pageB, dpi=dpi)
            try:
                crop_b = crop_with_padding(imgB_page, bbox_img_b, pad_px=10)
                filename_b = save_snippet(crop_b, snippets_dir, delta_internal.char_no, "revB", page_b)
                revB_evidence = Evidence(
                    page=page_b,
                    bbox=list(revB_bbox_pdf),
                    image_path=f"snippets/{filename_b}"
                )
            except ValueError as e:
                # Bbox invalid, record evidence without image
                revB_evidence = Evidence(
                    page=page_b,
                    bbox=list(revB_bbox_pdf),
                    image_path=None
                )

        semantic_callout = None
        if delta_internal.match is not None:
            matched_span = delta_internal.match.candidate.span
            semantic_callout = revB_semantic_callouts_by_span_key.get(
                (matched_span.block_id, matched_span.line_id, matched_span.span_id, matched_span.bbox_pdf)
            )
        elif delta_internal.added_span is not None:
            semantic_callout = revB_semantic_callouts_by_span_key.get(
                (
                    delta_internal.added_span.block_id,
                    delta_internal.added_span.line_id,
                    delta_internal.added_span.span_id,
                    delta_internal.added_span.bbox_pdf,
                )
            )
        elif anchor is not None:
            semantic_callout = anchor_semantic_callouts.get(anchor.char_no)

        if semantic_callout is None:
            semantic_pdf_spans = []
            form3_requirement = anchor.requirement_raw if anchor is not None else None

            if revB_annotation_spans:
                semantic_pdf_spans = revB_annotation_spans
            elif delta_internal.match is not None:
                semantic_pdf_spans = [delta_internal.match.candidate.span]
            elif delta_internal.added_span is not None:
                semantic_pdf_spans = [delta_internal.added_span]

            semantic_callout = extract_semantic_callout(
                pdf_spans=semantic_pdf_spans,
                form3_requirement=form3_requirement,
            )
        elif revB_annotation_spans and delta_internal.match is not None:
            semantic_callout = extract_semantic_callout(
                pdf_spans=revB_annotation_spans,
                form3_requirement=anchor.requirement_raw if anchor is not None else None,
            )

        # Extract Rev B annotation text.
        # Combine the primary matched/added span with all companion spans that were
        # absorbed during bbox expansion (e.g., the ↧ depth symbol beside "8.5",
        # the ⌴ counterbore symbol beside "Ø13.5", or a stacked tolerance suffix).
        # Sort left-to-right so the text reads in natural drawing order.
        requirement_revB = None
        if revB_annotation_spans:
            requirement_revB = _format_annotation_text(
                revB_annotation_spans,
                fallback_text=(delta_internal.match.candidate.span.text if delta_internal.match is not None else None),
            )
        if requirement_revB is None:
            if delta_internal.match is not None:
                requirement_revB = delta_internal.match.candidate.span.text
            elif delta_internal.added_span is not None:
                requirement_revB = _format_annotation_text([delta_internal.added_span], fallback_text=delta_internal.added_span.text)

        snippet_rule_family = "single_callout"
        if is_notes_type_b or len(_dedupe_spans(revA_annotation_spans)) > 1 or len(_dedupe_spans(revB_annotation_spans)) > 1:
            snippet_rule_family = "grouped_callout"

        # Create Pydantic DeltaItem
        delta_pydantic = DeltaItem(
            char_no=delta_internal.char_no,
            status=delta_internal.status,
            confidence=delta_internal.confidence,
            reasons=delta_internal.reasons,
            scores=delta_internal.component_scores,
            revA=revA_evidence,
            revB=revB_evidence,
            requirement_revB=requirement_revB,
            semantic_callout=semantic_callout,
            snippet_rule_family=snippet_rule_family,
        )
        delta_items_pydantic.append(delta_pydantic)

    # Close PDF documents
    docA.close()
    docB.close()

    print(f"  Classified {len(delta_items_pydantic)} delta items")
    print(f"  Saved evidence snippets to {snippets_dir}")

    # Stage 8: Construct and write DeltaPacket
    if stage_callback:
        stage_callback(7, "Output")
    print("[8/8] Writing delta packet...")
    truth_fixture_key = part_name
    truth_packet = load_ground_truth_packet(part_name)

    evaluations = evaluate_packet_against_truth(delta_items_pydantic, truth_packet)
    if accepted_alternates:
        scoped_alternates = [
            alternate for alternate in accepted_alternates
            if alternate.part_number == part_name
        ]
        evaluations = apply_accepted_alternate_history(
            delta_items_pydantic,
            evaluations,
            scoped_alternates,
        )

    evaluated_items: List[DeltaItem] = []
    for delta_item, evaluation in zip(delta_items_pydantic, evaluations):
        evaluated_items.append(
            DeltaItem(
                **delta_item.model_dump(exclude={"evaluation"}),
                evaluation=evaluation,
            )
        )
    delta_items_pydantic = evaluated_items

    packet = DeltaPacket(
        run_id=run_id,
        inputs={
            "revA_pdf": str(revA_path.absolute()),
            "revB_pdf": str(revB_path.absolute()),
            "form3_xlsx": str(form3_path.absolute()),
            "dpi": str(dpi),
            "truth_fixture_key": truth_fixture_key,
        },
        items=delta_items_pydantic
    )

    packet_path = run_dir / "delta_packet.json"
    with open(packet_path, "w") as f:
        f.write(packet.model_dump_json(indent=2, exclude_none=True))

    print(f"  Written to {packet_path}")
    print()
    print("Pipeline complete!")
    print(f"Review delta packet: {packet_path}")

    return run_dir


def main():
    """
    Main command-line entry point for the delta preservation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Delta preservation pipeline for engineering drawings"
    )
    parser.add_argument("--revA_pdf", required=True, help="Path to Revision A PDF")
    parser.add_argument("--revB_pdf", required=True, help="Path to Revision B PDF")
    parser.add_argument("--form3_xlsx", required=True, help="Path to Form 3 XLSX")
    parser.add_argument("--out_dir", default="./out", help="Output directory (default: ./out)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rendering (default: 300)")
    parser.add_argument("--part_name", default="part", help="Part name (default: part)")

    args = parser.parse_args()
    run_pipeline(**vars(args))


if __name__ == "__main__":
    main()
