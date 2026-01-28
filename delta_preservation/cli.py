import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import cv2

from delta_preservation.io.pdf import render_page, extract_text_spans, pdf_to_img_coords
import fitz
from delta_preservation.io.xlsx import load_form3
from delta_preservation.vision.balloons import detect_balloons
from delta_preservation.vision.alignment import estimate_transform
from delta_preservation.vision.snippets import crop_with_padding, save_snippet
from delta_preservation.vision.bbox_utils import (
    compute_combined_evidence_bbox,
    expand_bbox_with_adjacent_spans,
    normalize_snippet_size,
)
from delta_preservation.reconcile.anchors import build_revA_anchors
from delta_preservation.reconcile.match import generate_candidates, assign_matches
from delta_preservation.reconcile.classify import classify_delta, detect_added_characteristics, DeltaItem as DeltaItemInternal
from delta_preservation.types import DeltaPacket, DeltaItem, Evidence


def main():
    """
    Command-line entrypoint for the delta preservation prototype.

    Given:
    - Rev A PDF (ballooned)
    - Rev B PDF (unballooned)
    - Rev A AS9102 Form 3 (xlsx, sheet "Form3")

    Creates:
    out/<run_id>/
    delta_packet.json        # stub for now; later filled with change queue + evidence paths
    intermediate/            # debug artifacts per stage
    snippets/                # evidence snippet images per item
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
    
    # Validate input files
    revA_path = Path(args.revA_pdf)
    revB_path = Path(args.revB_pdf)
    form3_path = Path(args.form3_xlsx)
    
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
    run_id = f"{args.part_name}_{timestamp}_{short_hash}"
    
    # Create output directory structure
    run_dir = Path(args.out_dir) / run_id
    snippets_dir = run_dir / "snippets"
    intermediate_dir = run_dir / "intermediate"
    
    run_dir.mkdir(parents=True, exist_ok=False)
    snippets_dir.mkdir(parents=True, exist_ok=False)
    intermediate_dir.mkdir(parents=True, exist_ok=False)
    
    print(f"Run ID: {run_id}")
    print(f"Output: {run_dir.absolute()}")
    print()
    
    # Stage 1: Load Form 3
    print("[1/8] Loading Form 3...")
    form3_chars_list = load_form3(form3_path, intermediate_dir)
    form3_chars = {char.char_no: char.requirement for char in form3_chars_list}
    print(f"  Loaded {len(form3_chars)} characteristics")
    
    # Stage 2: Detect balloons in Rev A
    print("[2/8] Detecting balloons in Rev A...")
    balloons = detect_balloons(revA_path, dpi=args.dpi)
    print(f"  Detected {len(balloons)} balloons")
    
    # Stage 3: Extract text from Rev A
    print("[3/8] Extracting text from Rev A...")
    revA_text_spans = extract_text_spans(revA_path, page_index=0)
    print(f"  Extracted {len(revA_text_spans)} text spans")
    
    # Stage 4: Build Rev A anchors
    print("[4/8] Building Rev A anchors...")
    anchors = build_revA_anchors(form3_chars, balloons, revA_text_spans)
    print(f"  Built {len(anchors)} anchors")
    
    # Stage 5: Extract text from Rev B and estimate alignment
    print("[5/8] Extracting text from Rev B and aligning...")
    revB_text_spans = extract_text_spans(revB_path, page_index=0)
    print(f"  Extracted {len(revB_text_spans)} text spans")
    
    # Render pages for alignment (use first page for now)
    imgA = render_page(revA_path, page_index=0, dpi=args.dpi)
    imgB = render_page(revB_path, page_index=0, dpi=args.dpi)
    transform = estimate_transform(imgA, imgB)
    print(f"  Alignment: {transform.inliers} inliers, ratio={transform.inlier_ratio:.2f}")
    
    # Stage 6: Generate candidates and assign matches
    print("[6/8] Generating candidates and assigning matches...")
    candidates_by_anchor = {}
    for anchor in anchors:
        candidates = generate_candidates(anchor, revB_text_spans, transform, top_k=5)
        candidates_by_anchor[anchor.char_no] = candidates
    
    matches = assign_matches(anchors, candidates_by_anchor)
    print(f"  Assigned {len(matches)} matches")
    
    # Stage 7: Classify deltas and save snippets
    print("[7/8] Classifying deltas and saving evidence snippets...")
    delta_items_internal: List[DeltaItemInternal] = []
    
    for anchor in anchors:
        match_or_none = matches.get(anchor.char_no)
        delta_item_internal = classify_delta(anchor, match_or_none, location_search_coverage=1.0)
        delta_items_internal.append(delta_item_internal)
    
    # Detect added characteristics (new in Rev B, not in Rev A)
    max_char_no = max(a.char_no for a in anchors) if anchors else 0
    added_items = detect_added_characteristics(revB_text_spans, matches, next_char_no=max_char_no + 1)
    delta_items_internal.extend(added_items)
    print(f"  Found {len(added_items)} added characteristics in Rev B")
    
    # Open PDF documents for coordinate conversion
    docA = fitz.open(revA_path)
    docB = fitz.open(revB_path)
    
    # Convert internal DeltaItems to Pydantic models with Evidence
    delta_items_pydantic: List[DeltaItem] = []
    
    # Render pages once for all snippets
    imgA_page = render_page(revA_path, page_index=0, dpi=args.dpi)
    imgB_page = render_page(revB_path, page_index=0, dpi=args.dpi)
    pageA = docA.load_page(0)
    pageB = docB.load_page(0)
    
    for delta_internal in delta_items_internal:
        # Find corresponding anchor (may not exist for added items)
        anchor = next((a for a in anchors if a.char_no == delta_internal.char_no), None)
        
        # Initialize evidence objects
        revA_evidence = None
        revB_evidence = None
        
        # Compute bboxes for both revA and revB to enable consistent sizing
        revA_bbox_pdf = None
        revB_bbox_pdf = None
        
        # --- Compute Rev A bbox (balloon + annotation combined) ---
        if anchor is not None:
            # Create combined bbox including balloon and annotation
            revA_bbox_pdf = compute_combined_evidence_bbox(
                balloon_bbox=anchor.balloon_bbox,
                req_bbox=anchor.req_bbox,
                min_width=60.0,  # ~0.8 inch minimum
                min_height=30.0
            )
        
        # --- Compute Rev B bbox (expand to include adjacent spans) ---
        if delta_internal.match is not None:
            span = delta_internal.match.candidate.span
            base_bbox_b = span.bbox_pdf
            
            # Expand bbox to include adjacent spans (symbols like ⌴, ↧, tolerances)
            expanded = expand_bbox_with_adjacent_spans(
                center_bbox=base_bbox_b,
                all_spans=revB_text_spans,
                horizontal_tolerance=20.0,  # ~0.28 inch gap tolerance
                vertical_tolerance=8.0,     # ~0.11 inch vertical tolerance
                max_horizontal_expansion=150.0  # ~2 inch max expansion
            )
            revB_bbox_pdf = expanded.bbox
            
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
            revB_bbox_pdf = expanded.bbox
        
        # --- Normalize sizes for consistent paired snippets ---
        if revA_bbox_pdf is not None and revB_bbox_pdf is not None:
            revA_bbox_pdf, revB_bbox_pdf = normalize_snippet_size(revA_bbox_pdf, revB_bbox_pdf)
        
        # --- Generate Rev A snippet ---
        if anchor is not None and revA_bbox_pdf is not None:
            bbox_img_a = pdf_to_img_coords(revA_bbox_pdf, pageA, dpi=args.dpi)
            try:
                crop_a = crop_with_padding(imgA_page, bbox_img_a, pad_px=10)
                filename_a = save_snippet(crop_a, snippets_dir, delta_internal.char_no, "revA", anchor.page)
                revA_evidence = Evidence(
                    page=anchor.page,
                    bbox=list(revA_bbox_pdf),
                    image_path=f"snippets/{filename_a}"
                )
            except ValueError as e:
                # Bbox invalid, record evidence without image
                revA_evidence = Evidence(
                    page=anchor.page,
                    bbox=list(revA_bbox_pdf),
                    image_path=None
                )
        
        # --- Generate Rev B snippet ---
        if revB_bbox_pdf is not None:
            page_b = 0
            bbox_img_b = pdf_to_img_coords(revB_bbox_pdf, pageB, dpi=args.dpi)
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
        
        # Create Pydantic DeltaItem
        delta_pydantic = DeltaItem(
            char_no=delta_internal.char_no,
            status=delta_internal.status,
            confidence=delta_internal.confidence,
            reasons=delta_internal.reasons,
            scores=delta_internal.component_scores,
            revA=revA_evidence,
            revB=revB_evidence
        )
        delta_items_pydantic.append(delta_pydantic)
    
    # Close PDF documents
    docA.close()
    docB.close()
    
    print(f"  Classified {len(delta_items_pydantic)} delta items")
    print(f"  Saved evidence snippets to {snippets_dir}")
    
    # Stage 8: Construct and write DeltaPacket
    print("[8/8] Writing delta packet...")
    packet = DeltaPacket(
        run_id=run_id,
        inputs={
            "revA_pdf": str(revA_path.absolute()),
            "revB_pdf": str(revB_path.absolute()),
            "form3_xlsx": str(form3_path.absolute()),
            "dpi": str(args.dpi)
        },
        items=delta_items_pydantic
    )
    
    packet_path = run_dir / "delta_packet.json"
    with open(packet_path, "w") as f:
        f.write(packet.model_dump_json(indent=2))
    
    print(f"  Written to {packet_path}")
    print()
    print("Pipeline complete!")
    print(f"Review delta packet: {packet_path}")


if __name__ == "__main__":
    main()
