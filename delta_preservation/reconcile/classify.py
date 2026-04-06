from __future__ import annotations

import math
import re
from typing import Optional, Dict, List, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from delta_preservation.reconcile.anchors import Anchor
from delta_preservation.reconcile.match import Match
from delta_preservation.reconcile.normalize import (
    are_requirement_types_incompatible,
    classify_requirement_type,
    parse_requirement,
)
from delta_preservation.reconcile.semantic_compare import compare_semantic_callouts
from delta_preservation.io.pdf import TextSpan
from delta_preservation.types import SemanticCallout

if TYPE_CHECKING:
    from delta_preservation.reconcile.tolerance_pdf import ToleranceComparison


@dataclass
class DeltaItem:
    """Classification result for a single Rev A characteristic."""
    char_no: int
    status: str  # "unchanged", "changed", "removed", "uncertain", "added"
    confidence: float
    reasons: List[str]
    component_scores: Dict[str, float]
    match: Optional[Match] = None
    added_span: Optional[TextSpan] = None  # For added characteristics


@dataclass
class AddedCharacteristic:
    """Represents a newly detected characteristic in Rev B that doesn't exist in Rev A."""
    span: TextSpan
    requirement_text: str
    confidence: float
    reasons: List[str]


def classify_delta(
    anchor: Anchor,
    match_or_none: Optional[Match],
    location_search_coverage: float = 1.0,
    tolerance_comparison: Optional[ToleranceComparison] = None,
    anchor_semantic_callout: Optional[SemanticCallout] = None,
    matched_semantic_callout: Optional[SemanticCallout] = None,
    revB_text_spans: Optional[List["TextSpan"]] = None,
) -> DeltaItem:
    """Classify delta status for a single Rev A anchor.
    
    Classification logic focuses on numeric value matching since PDF spans
    often contain only raw dimension values (e.g., "120") while Form 3
    requirements contain full descriptions (e.g., "Length (120 +/- 0.3 mm)").
    
    Decision rules:
    - If key numeric values from anchor are found in matched span → unchanged
    - If numeric values differ but structural tokens match → changed
    - If no match found → removed
    - Otherwise → uncertain
    
    Args:
        anchor: Rev A anchor with requirement text
        match_or_none: Assigned match or None if no candidate found
        location_search_coverage: Fraction of search window covered (for removed confidence)
        
    Returns:
        DeltaItem with status, confidence, reasons, and component scores
    """
    reasons = []
    
    # Handle removed case
    if match_or_none is None:
        # --- Last-chance identity check ---
        # Before declaring "removed", scan all Rev B spans for the anchor's
        # primary numeric value with a compatible requirement type.  This catches
        # cases where the search window missed the annotation (e.g., because the
        # alignment transform was off due to a new view being added).
        if revB_text_spans:
            anchor_fp = parse_requirement(anchor.requirement_raw)
            anchor_numerics = {v for v, _ in anchor_fp.numeric_tokens}
            anchor_primary = max(anchor_numerics) if anchor_numerics else None
            anchor_req_type = classify_requirement_type(anchor.requirement_raw)
            is_notes_anchor = (
                anchor_fp.pattern_class == "note"
                or "NOTES" in anchor.requirement_raw.upper()
            )

            if anchor_primary is not None and not is_notes_anchor:
                # Skip the identity check when the anchor's primary value is a
                # common tolerance/small value that appears everywhere on the
                # drawing (e.g., 0.3, 0.005, 0.1).  Only run for distinctive
                # primary dimensions (≥ 1.0) that are unlikely to coincide.
                # Additionally require ≥2 secondary numerics to overlap — a
                # single primary match is too prone to false positives (e.g.,
                # thread callouts like "10-24 UNC" match the number 24 in
                # many unrelated spans).
                anchor_secondary = sorted(anchor_numerics - {anchor_primary})
                is_distinctive = (
                    anchor_primary >= 1.0
                    and len(anchor_numerics) >= 2
                    and anchor_req_type not in ("thread",)
                )
                if is_distinctive:
                    for span in revB_text_spans:
                        span_fp = parse_requirement(span.text)
                        span_numerics = {v for v, _ in span_fp.numeric_tokens}
                        if anchor_primary not in span_numerics:
                            continue
                        # Require the span to be annotation-like (not too many values,
                        # not a tolerance block or notes block)
                        if len(span_numerics) > 5:
                            continue
                        span_text_upper = span.text.strip().upper()
                        if any(kw in span_text_upper for kw in (
                            "UNLESS", "SPECIFIED", "ANGULAR", "FRACTIONAL",
                            "DRAWN", "CHECKED", "TITLE",
                        )):
                            continue
                        # Check requirement type compatibility
                        span_req_type = classify_requirement_type(span.text)
                        if are_requirement_types_incompatible(anchor_req_type, span_req_type):
                            continue
                        # Require at least one secondary numeric overlap to reduce
                        # false positives from coincidental primary matches
                        if anchor_secondary and span_numerics:
                            if not (set(anchor_secondary) & span_numerics):
                                continue
                        # Found a plausible match — reclassify as uncertain
                        return DeltaItem(
                            char_no=anchor.char_no,
                            status="uncertain",
                            confidence=0.45,
                            reasons=[
                                "No candidate in search window, but page-wide scan found "
                                f"primary value {anchor_primary} in Rev B span: \"{span.text.strip()[:80]}\"",
                                "Possible alignment miss — manual review recommended",
                            ],
                            component_scores={
                                "location": 0.0,
                                "text": 0.5,
                                "context": 0.0,
                            },
                            match=None,
                        )

        confidence = min(0.9, location_search_coverage)
        return DeltaItem(
            char_no=anchor.char_no,
            status="removed",
            confidence=confidence,
            reasons=["No candidate found within search window"],
            component_scores={
                "location": 0.0,
                "text": 0.0,
                "context": 0.0
            },
            match=None
        )
    
    # Extract component scores from match
    candidate = match_or_none.candidate
    location_score = candidate.location_score
    text_score = candidate.text_score
    context_score = candidate.context_score
    
    # Parse fingerprints from both anchor requirement and matched span
    anchor_fp = parse_requirement(anchor.requirement_raw)
    matched_fp = parse_requirement(candidate.span.text)

    semantic_result = None
    if anchor_semantic_callout is not None or matched_semantic_callout is not None:
        semantic_result = compare_semantic_callouts(anchor_semantic_callout, matched_semantic_callout)

    if semantic_result is not None and semantic_result.mode == "semantic":
        semantic_reasons = list(semantic_result.reason_fragments)
        if semantic_result.equal:
            return DeltaItem(
                char_no=anchor.char_no,
                status="unchanged",
                confidence=max(0.85, 0.45 * location_score + 0.4 * text_score + 0.15 * context_score),
                reasons=semantic_reasons + ["Semantic family agreement overrides formatting-only numeric/text variation"],
                component_scores={
                    "location": location_score,
                    "text": text_score,
                    "context": context_score,
                },
                match=match_or_none,
            )

        return DeltaItem(
            char_no=anchor.char_no,
            status="changed",
            confidence=max(0.8, 0.35 * location_score + 0.45 * text_score + 0.2 * context_score),
            reasons=semantic_reasons + ["Semantic family delta indicates a meaningful drawing requirement change"],
            component_scores={
                "location": location_score,
                "text": text_score,
                "context": context_score,
            },
            match=match_or_none,
        )

    if semantic_result is not None and semantic_result.mode in {"fallback", "incompatible"}:
        reasons.extend(semantic_result.reason_fragments)

    # --- Requirement-type incompatibility check ---
    # If the Rev A and Rev B requirements belong to contextually incompatible
    # semantic categories (e.g. a fillet radius vs. an angle, a thread callout
    # vs. a plain dimension, a GD&T frame vs. a chamfer callout) the numeric
    # comparison below will produce meaningless results.  Return "uncertain"
    # immediately so a reviewer can decide whether the design intent changed.
    anchor_req_type = classify_requirement_type(anchor.requirement_raw)
    matched_req_type = classify_requirement_type(candidate.span.text)
    if are_requirement_types_incompatible(anchor_req_type, matched_req_type):
        # When requirement types are incompatible AND location confidence is low,
        # this is almost certainly a false match — classify as removed.
        if location_score < 0.3:
            return DeltaItem(
                char_no=anchor.char_no,
                status="removed",
                confidence=min(0.75, location_search_coverage),
                reasons=reasons + [
                    f"Requirement type mismatch: Rev A is '{anchor_req_type}', Rev B is '{matched_req_type}'",
                    "Low location confidence + type mismatch → likely removed",
                ],
                component_scores={
                    "location": location_score,
                    "text": 0.0,
                    "context": context_score,
                },
                match=None,
            )
        return DeltaItem(
            char_no=anchor.char_no,
            status="uncertain",
            confidence=max(0.55, 0.5 * location_score),
            reasons=reasons + [
                f"Requirement type mismatch: Rev A is '{anchor_req_type}', Rev B is '{matched_req_type}'",
                f"Rev A: \"{anchor.requirement_raw}\" — Rev B: \"{candidate.span.text}\"",
                "Contextually incompatible requirement types; manual review required",
            ],
            component_scores={
                "location": location_score,
                "text": 0.0,
                "context": context_score,
            },
            match=match_or_none,
        )

    # Check if this is a notes-type characteristic
    # Notes blocks should be compared by text pattern, not numeric values
    is_notes_type = anchor_fp.pattern_class == "note" or "NOTES" in anchor.requirement_raw.upper()
    
    if is_notes_type:
        # For notes blocks, check if the matched span contains "NOTES" header
        if "NOTES" in matched_fp.norm_text or "NOTE" in matched_fp.norm_text:
            # If we have Rev B spans, do a content comparison to detect note changes
            # (e.g., QTY change, different material, different finish requirement)
            if revB_text_spans is not None:
                matched_cx = (candidate.span.bbox_pdf[0] + candidate.span.bbox_pdf[2]) / 2
                matched_cy = (candidate.span.bbox_pdf[1] + candidate.span.bbox_pdf[3]) / 2
                nearby_texts = []
                for nearby_span in revB_text_spans:
                    nx0, ny0, nx1, ny1 = nearby_span.bbox_pdf
                    ncx = (nx0 + nx1) / 2
                    ncy = (ny0 + ny1) / 2
                    dist = math.sqrt((ncx - matched_cx) ** 2 + (ncy - matched_cy) ** 2)
                    if dist <= 200.0 and abs(ncx - matched_cx) <= 200.0:
                        nearby_texts.append(nearby_span.text)

                if nearby_texts:
                    revB_notes_text = " ".join(nearby_texts)
                    revB_notes_fp = parse_requirement(revB_notes_text)
                    revB_tokens = set(revB_notes_fp.norm_text.split())
                    anchor_tokens = set(anchor_fp.norm_text.split())
                    if anchor_tokens and revB_tokens:
                        union_size = len(anchor_tokens | revB_tokens)
                        inter_size = len(anchor_tokens & revB_tokens)
                        notes_similarity = inter_size / union_size if union_size > 0 else 1.0
                        if notes_similarity < 0.65:
                            return DeltaItem(
                                char_no=anchor.char_no,
                                status="changed",
                                confidence=max(0.70, 0.4 * location_score + 0.4 * notes_similarity + 0.2),
                                reasons=[
                                    "Notes block content changed",
                                    f"Token similarity {notes_similarity:.2f} below threshold 0.65",
                                ],
                                component_scores={
                                    "location": location_score,
                                    "text": notes_similarity,
                                    "context": context_score,
                                },
                                match=match_or_none,
                            )

            return DeltaItem(
                char_no=anchor.char_no,
                status="unchanged",
                confidence=0.85,
                reasons=["Notes block header matched", "High location agreement after global alignment"],
                component_scores={
                    "location": location_score,
                    "text": 1.0,
                    "context": context_score
                },
                match=match_or_none
            )
    
    # Extract numeric values (the key comparison for dimensions)
    anchor_numerics = set(val for val, _ in anchor_fp.numeric_tokens)
    matched_numerics = set(val for val, _ in matched_fp.numeric_tokens)
    
    # Identify primary dimension value (largest numeric, typically the main dimension)
    # Tolerances are usually small values like 0.1, 0.3 while dimensions are larger
    anchor_primary = max(anchor_numerics) if anchor_numerics else None
    matched_primary = max(matched_numerics) if matched_numerics else None
    
    # Check if primary dimension matches - this is the key indicator.
    # Use set membership rather than max-equality so that multi-value spans (e.g.,
    # "10 x 90°" for a countersink that encodes both depth=10 and angle=90°) are
    # correctly recognised as containing the anchor's primary value (10) even though
    # the span's overall maximum value (90) is different.
    primary_matches = (
        anchor_primary is not None and
        anchor_primary in matched_numerics
    )
    
    # Extract structural tokens (count patterns like "2X", "6X" and symbols like "Ø", "R")
    anchor_count = set(anchor_fp.count_tokens)
    matched_count = set(matched_fp.count_tokens)
    
    anchor_symbols = set(anchor_fp.symbol_tokens)
    matched_symbols = set(matched_fp.symbol_tokens)
    
    # Compute overlap metrics
    # For numerics: check if anchor's key values appear in matched span
    # Focus on primary value match rather than all tolerances
    if anchor_numerics:
        # If primary matches, give high score even if tolerances not in span
        if primary_matches:
            numeric_overlap = max(0.5, len(anchor_numerics & matched_numerics) / len(anchor_numerics))
        else:
            numeric_overlap = len(anchor_numerics & matched_numerics) / len(anchor_numerics)
    else:
        numeric_overlap = 1.0 if not matched_numerics else 0.0
    
    # For counts: distinguish between count change vs count missing from matched span
    # "2X" → "4X" is a real change, but "2X" → set() is just incomplete span matching
    count_match = anchor_count == matched_count or (not anchor_count and not matched_count)
    count_changed = bool(anchor_count and matched_count and anchor_count != matched_count)
    count_missing = bool(anchor_count and not matched_count)  # Count present in anchor but not in span
    count_added = bool(not anchor_count and matched_count)   # Count added in Rev B (e.g. none → "2X")
    
    # For symbols: check if anchor symbols appear in matched span
    symbol_match = anchor_symbols <= matched_symbols or not anchor_symbols
    
    # Classification decision tree
    # Priority 0 (early guard): zero numeric overlap AND primary values diverge
    # significantly → the matched candidate is almost certainly a grid/border label
    # or an unrelated annotation.  Return "removed" before the later elif branches
    # can mis-classify it as "changed".
    # NOTE: this guard must run before the symbol/overlap checks below, because when
    # neither anchor nor span has engineering symbols (symbol_match = True) the
    # `numeric_overlap < 0.5 and symbol_match` branch would otherwise fire first and
    # reach a "changed" verdict via the limits-form path.
    if (
        numeric_overlap == 0.0
        and anchor_numerics
        and matched_numerics
        and anchor_primary is not None
        and matched_primary is not None
        and anchor_primary != 0
        and abs(anchor_primary - matched_primary) / abs(anchor_primary) > 0.15
    ):
        return DeltaItem(
            char_no=anchor.char_no,
            status="removed",
            confidence=min(0.85, location_search_coverage),
            reasons=reasons + [
                "No numeric overlap and primary values differ significantly",
                f"Anchor primary: {anchor_primary}, Matched primary: {matched_primary}",
                "Match candidate likely a grid label or unrelated annotation",
            ],
            component_scores={
                "location": location_score,
                "text": 0.0,
                "context": context_score,
            },
            match=None,
        )

    # Priority 1: Primary dimension value match is the strongest indicator
    if count_changed or count_added:
        # Count explicitly changed or added (e.g., none → "2X", or "2X" → "4X") → changed
        status = "changed"
        confidence = 0.5 * location_score + 0.3 * numeric_overlap + 0.2
        if count_changed:
            reasons.append(f"Count changed: {anchor_count} → {matched_count}")
        else:
            reasons.append(f"Count added in Rev B: {matched_count} (was absent in Rev A)")
    elif primary_matches:
        # Primary dimension value matches - this is the key indicator of unchanged
        # Even if tolerances differ or aren't visible in span, the main dimension is the same.
        # However, if numeric overlap is low (≤65%), the differing numerics may be
        # tolerance values that genuinely changed (e.g., ±.005 → ±.01).
        tolerance_sized_diff = False
        if numeric_overlap <= 0.65 and anchor_primary is not None and anchor_primary != 0:
            differing_new = matched_numerics - anchor_numerics
            differing_old = anchor_numerics - matched_numerics
            # Only flag when the matched span has a NEW tolerance-sized value AND
            # the anchor has a DIFFERENT tolerance-sized value that's been replaced.
            # Both values must be within 5x of each other (same order of magnitude)
            # to avoid false positives from notation mismatches (e.g., 0.10 vs 0.010).
            tol_new = [v for v in differing_new if 0 < v <= 0.2 * anchor_primary]
            tol_old = [v for v in differing_old if 0 < v <= 0.2 * anchor_primary]
            if tol_new and tol_old:
                # Check that the differing values are within 5x of each other
                for nv in tol_new:
                    for ov in tol_old:
                        ratio = max(nv, ov) / min(nv, ov)
                        if ratio <= 5.0:
                            tolerance_sized_diff = True
                            reasons.append(
                                f"Tolerance-sized numeric differs: "
                                f"anchor {sorted(anchor_numerics)}, matched {sorted(matched_numerics)}"
                            )
                            break
                    if tolerance_sized_diff:
                        break

        if tolerance_sized_diff:
            status = "changed"
            confidence = max(0.65, 0.4 * location_score + 0.3 * numeric_overlap + 0.2)
            reasons.append(f"Primary dimension matches: {anchor_primary}")
            reasons.append("Tolerance value changed despite same primary dimension")
        else:
            status = "unchanged"
            confidence = 0.4 * location_score + 0.5 * numeric_overlap + 0.1
            reasons.append(f"Primary dimension matches: {anchor_primary}")
            if numeric_overlap >= 0.5:
                reasons.append(f"Numeric values match ({int(numeric_overlap*100)}% overlap)")
            if count_match and anchor_count:
                reasons.append(f"Count tokens match: {anchor_count}")
    elif numeric_overlap >= 0.5 and symbol_match:
        # Good numeric overlap with matching symbols.
        # However, the overlap may come purely from matching a shared tolerance value
        # (e.g., both have 0.01) while the primary dimension differs (4.93 vs 5.04).
        # In that case, the primary mismatch dominates and the status should be "changed".
        #
        # We distinguish true primary mismatches from limits-form notation by comparing
        # the primary difference against the anchor's own tolerance band:
        # - Extract the smallest non-zero numeric as the tolerance value.
        # - If primary difference > 1.5 * tolerance, it's a true change.
        # - If within 1.5 * tolerance, it's likely limits-form notation (same characteristic).
        primary_mismatch = False
        if (
            anchor_primary is not None
            and matched_primary is not None
            and not primary_matches
            and anchor_primary != 0
        ):
            primary_diff = abs(anchor_primary - matched_primary)
            # Estimate the tolerance as the smallest numeric in the anchor (> 0)
            small_numerics = [v for v in anchor_numerics if 0 < v < anchor_primary]
            if small_numerics:
                tolerance_estimate = min(small_numerics)
                # True change if primary difference exceeds 1.5x the tolerance band
                primary_mismatch = primary_diff > 1.5 * tolerance_estimate
            else:
                # No tolerance value in anchor — use a 3% relative threshold
                primary_mismatch = primary_diff / abs(anchor_primary) > 0.03

        if primary_mismatch:
            status = "changed"
            confidence = 0.4 * location_score + 0.3 * numeric_overlap + 0.2
            reasons.append(f"Primary dimension changed: {anchor_primary} → {matched_primary}")
            reasons.append(f"Numeric overlap from tolerance only ({int(numeric_overlap*100)}%)")
        else:
            # (count_missing is OK - just means span doesn't include count prefix)
            status = "unchanged"
            confidence = 0.4 * location_score + 0.4 * numeric_overlap + 0.2
            reasons.append(f"Numeric values match ({int(numeric_overlap*100)}% overlap)")
            if count_match and anchor_count:
                reasons.append(f"Count tokens match: {anchor_count}")
    elif numeric_overlap < 0.5 and symbol_match and not count_missing and not primary_matches:
        # Same type (symbol) but different numeric values.
        # First check if this is limits-form notation: the matched primary might be
        # the upper or lower tolerance limit of the anchor's nominal value.
        # e.g., anchor "0.150 ± 0.010" vs matched ".160" (upper limit = 0.150+0.010)
        limits_form_match = False
        if anchor_primary is not None and matched_primary is not None and anchor_primary != 0:
            primary_diff = abs(anchor_primary - matched_primary)
            small_numerics = [v for v in anchor_numerics if 0 < v < anchor_primary]
            if small_numerics:
                tolerance_estimate = min(small_numerics)
                relative_tolerance = tolerance_estimate / abs(anchor_primary)
                # Only apply limits-form check when the tolerance is tight (< 15% of nominal).
                # Tight tolerances (e.g., ±0.010 on 0.150) produce limits like 0.160/0.140
                # that fall within the 1.5x tolerance band.  Loose tolerances (e.g., ±0.3 on
                # 1.570) are too permissive — a genuine change of 0.110 would also pass.
                if relative_tolerance < 0.15:
                    limits_form_match = primary_diff <= 1.5 * tolerance_estimate
            else:
                limits_form_match = primary_diff / abs(anchor_primary) <= 0.03

        if limits_form_match:
            # Limits-form notation: matched span is within the anchor's tolerance band
            status = "unchanged"
            confidence = 0.4 * location_score + 0.2 + 0.1  # No numeric overlap bonus
            reasons.append(f"Limits-form match: {anchor_primary} ± tolerance ≈ {matched_primary}")
        else:
            # Genuinely different numeric values → changed
            status = "changed"
            confidence = 0.4 * location_score + 0.3 * numeric_overlap + 0.2
            reasons.append(f"Numeric values changed (only {int(numeric_overlap*100)}% overlap)")
            reasons.append(f"Anchor: {sorted(anchor_numerics)}, Matched: {sorted(matched_numerics)}")
    elif location_score > 0.6:
        # Good location match but text doesn't align well → likely unchanged
        status = "unchanged"
        confidence = 0.6 * location_score + 0.2 * numeric_overlap
        reasons.append("Location strongly suggests same requirement")
        reasons.append(f"Numeric overlap: {int(numeric_overlap*100)}%")
    else:
        status = "uncertain"
        confidence = 0.5 * location_score + 0.3 * numeric_overlap
        reasons.append("Weak match confidence")
    
    # Add location-based reason
    if location_score > 0.7:
        reasons.append("High location agreement after global alignment")

    # --- Tolerance refinement ---
    if tolerance_comparison is not None:
        if tolerance_comparison.tolerances_match:
            if status == "unchanged":
                confidence += 0.05
                reasons.extend(tolerance_comparison.reasons)
            elif status == "uncertain":
                status = "unchanged"
                confidence += 0.1
                reasons.append("Tolerance agreement resolves uncertainty")
                reasons.extend(tolerance_comparison.reasons)
        elif tolerance_comparison.tolerances_differ:
            if status == "unchanged":
                status = "changed"
                reasons.append("Tolerance changed despite same dimension")
                reasons.extend(tolerance_comparison.reasons)
            elif status == "uncertain":
                status = "changed"
                reasons.append("Tolerance difference resolves uncertainty")
                reasons.extend(tolerance_comparison.reasons)

    # Clip confidence
    confidence = max(0.0, min(1.0, confidence))
    
    return DeltaItem(
        char_no=anchor.char_no,
        status=status,
        confidence=confidence,
        reasons=reasons,
        component_scores={
            "location": location_score,
            "text": numeric_overlap,  # Report numeric overlap as text score
            "context": context_score
        },
        match=match_or_none
    )


def detect_added_characteristics(
    revB_spans: List[TextSpan],
    matches: Dict[int, Match],
    next_char_no: int,
    page_width: float = 612.0,
    page_height: float = 792.0,
    revA_spans: Optional[List[TextSpan]] = None,
) -> List[DeltaItem]:
    """Detect new characteristics in Rev B that don't exist in Rev A.

    Looks for unmatched Rev B text spans that appear to be dimension requirements.
    Detects three categories:
    1. Standard annotated dimensions: symbol (Ø, R) or count prefix (2X, 4X)
    2. Angle-style callouts: degree marker with multiple numerics
    3. Leading-decimal single values: ".900", ".625" — engineering-drawing notation
       for plain length/depth dimensions without prefix symbols
    4. Stacked tolerance-limit pairs: two numeric-only spans at the same x position
       and adjacent y positions (e.g., ".635" above ".615") representing an upper/lower
       tolerance pair (equivalent to 0.625 ± 0.010)

    Args:
        revB_spans: All text spans from Rev B PDF
        matches: Dict of char_no to Match objects (matched spans)
        next_char_no: Starting char_no for added items (typically max_existing + 1)
        page_width: Width of the page in PDF points (used for exclusion zones)
        page_height: Height of the page in PDF points (used for exclusion zones)

    Returns:
        List of DeltaItems with status="added" for detected new characteristics
    """
    # Build set of matched span keys
    matched_span_keys: Set[Tuple] = set()
    matched_span_bboxes: List[Tuple[float, float, float, float]] = []
    for match in matches.values():
        span = match.candidate.span
        key = (span.block_id, span.line_id, span.span_id, span.bbox_pdf)
        matched_span_keys.add(key)
        matched_span_bboxes.append(span.bbox_pdf)
        source_spans = getattr(span, "source_spans", None)
        if source_spans:
            for source_span in source_spans:
                matched_span_keys.add((source_span.block_id, source_span.line_id, source_span.span_id, source_span.bbox_pdf))
                matched_span_bboxes.append(source_span.bbox_pdf)

    # Build a lookup of Rev A span content for cross-referencing.
    # Rev B spans whose text appears at the same position in Rev A are pre-existing
    # annotations (not new characteristics) and should not be flagged as "added".
    revA_span_signatures: Set[Tuple[str, int, int]] = set()
    if revA_spans is not None:
        for s in revA_spans:
            sx0, sy0, sx1, sy1 = s.bbox_pdf
            scx = round((sx0 + sx1) / 2)
            scy = round((sy0 + sy1) / 2)
            revA_span_signatures.add((s.text.strip(), scx, scy))

    def is_in_revA(span: TextSpan, position_tolerance: float = 15.0) -> bool:
        """Return True if this span exists in Rev A at the same position."""
        if not revA_span_signatures:
            return False
        x0, y0, x1, y1 = span.bbox_pdf
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        text = span.text.strip()
        tol = int(position_tolerance)
        for dx in range(-tol, tol + 1, 5):
            for dy in range(-tol, tol + 1, 5):
                if (text, round(cx) + dx, round(cy) + dy) in revA_span_signatures:
                    return True
        return False

    # Define exclusion zones (revision table, title block)
    # Revision table: typically top-right corner (x > 80% width, y < 15% height)
    # Title block: typically bottom-right corner (x > 70% width, y > 85% height)
    def is_in_exclusion_zone(bbox: tuple) -> bool:
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        # Revision table: top-right corner
        if cx > page_width * 0.75 and cy < page_height * 0.15:
            return True

        # Title block: bottom-right corner
        if cx > page_width * 0.70 and cy > page_height * 0.85:
            return True

        return False

    def is_inside_matched_group(bbox: tuple, tolerance: float = 8.0) -> bool:
        x0, y0, x1, y1 = bbox
        for mx0, my0, mx1, my1 in matched_span_bboxes:
            if x0 >= mx0 - tolerance and y0 >= my0 - tolerance and x1 <= mx1 + tolerance and y1 <= my1 + tolerance:
                return True
        return False

    # GD&T feature control frame anchor symbols — defined early so the
    # unmatched-span pre-filter can exempt single-character GD&T symbols.
    GDT_ANCHOR_SYMBOLS = {'⌖', '⌒', '⟂', '⊙', '⌓', '⏥', '∥', '∠', '⌖∅'}

    # Collect unmatched candidate spans (filtered for drawing content area)
    unmatched_spans: List[TextSpan] = []
    for span in revB_spans:
        key = (span.block_id, span.line_id, span.span_id, span.bbox_pdf)
        if key in matched_span_keys:
            continue
        if is_inside_matched_group(span.bbox_pdf):
            continue
        if is_in_exclusion_zone(span.bbox_pdf):
            continue
        text = span.text.strip()
        if len(text) < 2 and text not in GDT_ANCHOR_SYMBOLS:
            continue
        unmatched_spans.append(span)

    # Set of span keys consumed by stacked-pair detection (to avoid double-counting)
    stacked_pair_keys: Set[Tuple] = set()

    # Set of span keys consumed by GD&T group detection
    gdt_consumed_keys: Set[Tuple] = set()

    added_items: List[DeltaItem] = []
    current_char_no = next_char_no

    # Build set of matched span centre positions for proximity checks below.
    # Used to suppress unmatched spans that are companion tolerance limits of
    # already-matched annotations (e.g., the ".140" lower-limit span sitting
    # 12 pts below the matched ".160" upper-limit span).
    matched_centers: List[Tuple[float, float]] = []
    for match in matches.values():
        s = match.candidate.span
        sx0, sy0, sx1, sy1 = s.bbox_pdf
        matched_centers.append(((sx0 + sx1) / 2, (sy0 + sy1) / 2))

    # Build set of numeric float values from matched spans.
    # Used in stacked-pair detection to suppress pairs whose values are already
    # covered by an existing matched characteristic.  Example: when char 1 is
    # matched to a "35.2" span (limits-form upper limit of Ø35 ±0.2), a second
    # unmatched "35.2" / "34.8" stacked pair in another view represents the same
    # characteristic — not a new one — and should be suppressed.
    matched_span_float_values: set = set()
    matched_numeric_pairs: set[tuple[float, float]] = set()
    for match in matches.values():
        span = match.candidate.span
        texts = [span.text.strip()]
        for source_span in getattr(span, "source_spans", []) or []:
            texts.append(source_span.text.strip())
        numeric_values = []
        for text in texts:
            try:
                numeric_values.append(float(text))
                matched_span_float_values.add(float(text))
            except ValueError:
                continue
        uniq = sorted(set(numeric_values))
        if len(uniq) >= 2:
            matched_numeric_pairs.add((uniq[-2], uniq[-1]))

    def is_near_matched_span(bbox: tuple, threshold: float = 25.0) -> bool:
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for mx, my in matched_centers:
            if math.sqrt((cx - mx) ** 2 + (cy - my) ** 2) <= threshold:
                return True
        return False

    # --- Pass 0: GD&T feature control frame detection ---
    # Feature control frames: a GD&T symbol span (⟂, ⌖, ⏥, etc.) followed by
    # companion spans (tolerance value, datum references) on the same row.
    # These groups must be detected together; individual spans don't pass the
    # standard filters (symbol spans lack numerics; companion spans lack symbols).

    for span in unmatched_spans:
        key = (span.block_id, span.line_id, span.span_id, span.bbox_pdf)
        if key in gdt_consumed_keys:
            continue

        text = span.text.strip()
        has_gdt = any(sym in text for sym in GDT_ANCHOR_SYMBOLS)
        if not has_gdt:
            continue

        if is_in_revA(span):
            continue

        x0, y0, x1, y1 = span.bbox_pdf
        sy_center = (y0 + y1) / 2

        # Collect companion spans on the same row (within 10 pt vertically,
        # within 300 pt horizontally).
        # Search ALL revB_spans (not just unmatched_spans) so that tolerance values
        # and datum references that were already consumed by the main matching phase
        # are still included in the FCF text.  Only unmatched companions are added
        # to gdt_consumed_keys (to prevent Pass 1/2 from re-detecting them); matched
        # companions are included for text purposes only.
        group_spans = [span]
        group_keys = [key]
        # Traverse all revB_spans (not just unmatched_spans) for companion text so that
        # tolerance values and datum refs already consumed by the main match phase are
        # still included in the FCF group text.
        for other in revB_spans:
            other_key = (other.block_id, other.line_id, other.span_id, other.bbox_pdf)
            if other_key == key or other_key in gdt_consumed_keys:
                continue
            ox0, oy0, ox1, oy1 = other.bbox_pdf
            oy_center = (oy0 + oy1) / 2
            if abs(oy_center - sy_center) > 10.0:
                continue
            # Must be horizontally close (within 300 pt)
            if ox0 > x1 + 300.0 or ox1 < x0 - 300.0:
                continue
            group_spans.append(other)
            # Only mark unmatched spans as consumed; matched spans remain available
            if other_key not in matched_span_keys:
                group_keys.append(other_key)

        # Sort group spans left-to-right and concatenate text
        group_spans.sort(key=lambda s: s.bbox_pdf[0])
        group_text = ' '.join(s.text.strip() for s in group_spans)

        # Mark all unmatched group spans consumed so Pass 1 and Pass 2 skip them
        for k in group_keys:
            gdt_consumed_keys.add(k)

        # If any companion span (non-anchor) that contains numeric content is
        # already matched to an existing Rev A characteristic, this GD&T frame
        # belongs to an existing characteristic — not a new one.  Skip it.
        # Examples prevented:
        #   • "⏥ 0.2" where "0.2" was already matched by char 9 (double-detection)
        #   • "−0.1 ⌓ 0.5 A" where "0.5" was matched by char 11 via global fallback
        anchor_span_key = key  # Key of the GD&T anchor symbol span
        has_matched_numeric_companion = any(
            (s.block_id, s.line_id, s.span_id, s.bbox_pdf) in matched_span_keys
            and (s.block_id, s.line_id, s.span_id, s.bbox_pdf) != anchor_span_key
            and any(c.isdigit() for c in s.text)
            for s in group_spans
        )
        if has_matched_numeric_companion:
            continue

        reasons_gdt = [
            f"New GD&T feature control frame in Rev B: \"{group_text}\"",
            "GD&T control symbol detected",
        ]
        added_items.append(DeltaItem(
            char_no=current_char_no,
            status="added",
            confidence=0.75,
            reasons=reasons_gdt,
            component_scores={"location": 0.0, "text": 1.0, "context": 0.0},
            match=None,
            added_span=group_spans[0],
        ))
        current_char_no += 1

    # --- Pass 1: detect stacked tolerance-limit pairs ---
    # Stacked pairs: two numeric-only spans at the same horizontal position and
    # adjacent vertical positions that together express a tolerance band.
    # Example: ".635" (upper) stacked above ".615" (lower) → 0.625 ± 0.010
    # Criteria:
    #   - Both spans are numeric-only (no symbol, no count)
    #   - Not a trailing-dot list number (e.g., "1.", "2.", "3.")
    #   - Horizontal centres within 10 pts of each other (same column)
    #   - Vertical separation ≤ 30 pts (adjacent lines)
    #   - Upper value > lower value (ordered pair)
    #   - Difference ≤ 20% of mean (tight tolerance, not two unrelated dims)
    #   - Neither span is close to an already-matched span (would be a companion
    #     tolerance limit of an existing characteristic, not a new one)
    leading_decimal_re = re.compile(r'^\.\d+$')
    trailing_dot_re = re.compile(r'^\d+\.$')  # List item numbers: "1.", "2.", "3."
    numeric_only_spans = [
        s for s in unmatched_spans
        if (leading_decimal_re.match(s.text.strip()) or
            re.match(r'^\d+\.?\d*$', s.text.strip()))
        and not trailing_dot_re.match(s.text.strip())
        and not is_near_matched_span(s.bbox_pdf)
    ]

    for i, span_a in enumerate(numeric_only_spans):
        key_a = (span_a.block_id, span_a.line_id, span_a.span_id, span_a.bbox_pdf)
        if key_a in stacked_pair_keys:
            continue
        ax0, ay0, ax1, ay1 = span_a.bbox_pdf
        acx = (ax0 + ax1) / 2
        acy = (ay0 + ay1) / 2
        val_a_str = span_a.text.strip()
        try:
            val_a = float(val_a_str)
        except ValueError:
            continue

        for span_b in numeric_only_spans[i + 1:]:
            key_b = (span_b.block_id, span_b.line_id, span_b.span_id, span_b.bbox_pdf)
            if key_b in stacked_pair_keys:
                continue
            bx0, by0, bx1, by1 = span_b.bbox_pdf
            bcx = (bx0 + bx1) / 2
            bcy = (by0 + by1) / 2
            val_b_str = span_b.text.strip()
            try:
                val_b = float(val_b_str)
            except ValueError:
                continue

            # Horizontal alignment check
            if abs(acx - bcx) > 10.0:
                continue
            # Vertical proximity check
            vertical_gap = abs(acy - bcy)
            if vertical_gap > 30.0:
                continue
            # Must be an ordered pair (upper > lower)
            upper_val = max(val_a, val_b)
            lower_val = min(val_a, val_b)
            if upper_val <= lower_val:
                continue
            # Tolerance check: difference must be tight (≤ 20% of mean)
            # A tighter threshold rejects numeric pairs that are two unrelated
            # dimension values happening to be adjacent (e.g., "3." and "4." note
            # numbers have a ratio of ~29%, which tighter threshold excludes).
            mean_val = (upper_val + lower_val) / 2.0
            if mean_val == 0:
                continue
            if (upper_val - lower_val) / mean_val > 0.20:
                continue

            # Valid stacked pair found
            nominal = mean_val
            half_tol = (upper_val - lower_val) / 2.0
            pair_text = f"{upper_val} / {lower_val}"
            ordered_pair = (lower_val, upper_val)
            if ordered_pair in matched_numeric_pairs:
                stacked_pair_keys.add(key_a)
                stacked_pair_keys.add(key_b)
                break
            # Use the upper span as the representative span
            upper_span = span_a if val_a >= val_b else span_b
            reasons = [
                f"New stacked-limits dimension in Rev B: {pair_text}",
                f"Interpreted as {nominal:.4g} \u00b1 {half_tol:.4g}",
                "Leading-decimal stacked tolerance limits detected",
            ]
            confidence = 0.65

            added_item = DeltaItem(
                char_no=current_char_no,
                status="added",
                confidence=confidence,
                reasons=reasons,
                component_scores={
                    "location": 0.0,
                    "text": 1.0,
                    "context": 0.0
                },
                match=None,
                added_span=upper_span
            )
            added_items.append(added_item)
            current_char_no += 1
            stacked_pair_keys.add(key_a)
            stacked_pair_keys.add(key_b)
            break  # Each span_a pairs with at most one span_b

    # --- Pass 2: standard span-by-span detection ---
    for span in unmatched_spans:
        key = (span.block_id, span.line_id, span.span_id, span.bbox_pdf)
        # Skip spans already consumed by GD&T group or stacked-pair detection
        if key in gdt_consumed_keys or key in stacked_pair_keys:
            continue

        # Skip spans that appear verbatim in Rev A at the same position —
        # these are pre-existing annotations not tracked by Form 3, not new features.
        if is_in_revA(span):
            continue

        text = span.text.strip()

        # Parse the span to check if it looks like a dimension requirement
        fp = parse_requirement(text)

        # Angle callouts (e.g., "2 X 5 X 45°") often lack Ø/R symbols and explicit count
        # tokens, so treat a degree marker plus multiple numeric values as dimension-like.
        has_angle_token = "°" in text or "DEG" in fp.norm_text or "ANGLE" in fp.norm_text
        has_angle_dimension = has_angle_token and len(fp.numeric_tokens) >= 2

        # Leading-decimal format: ".900", ".625" are engineering-drawing shorthand for
        # dimension values without a leading zero.  These may be plain length/depth
        # annotations without a prefix symbol (Ø, R) or count (2X).
        # Only match the pure leading-decimal form (exactly ".<digits>") to avoid
        # false-positives from mid-sentence decimal numbers.
        is_leading_decimal_single = bool(
            leading_decimal_re.match(text)
            and len(fp.numeric_tokens) == 1
        )

        # Plain decimal format: "1.250", "3.750" — dimension values with a leading
        # integer digit and explicit decimal point.  Treated similarly to leading-decimal
        # annotations; requires exactly one numeric token to avoid matching multi-value
        # spans that already fail other filters.
        is_plain_decimal_single = bool(
            re.match(r'^\d+\.\d+$', text)
            and len(fp.numeric_tokens) == 1
        )

        # Plain integer dimension: standalone integers ≥ 100 such as ordinate
        # dimensions (155, 225, 450) or general lengths (800) added in a new view.
        # Zone labels and note numbers are typically < 100; title-block entries are
        # caught by the exclusion zone check above.
        is_plain_integer_dimension = bool(
            re.match(r'^\d+$', text)
            and len(fp.numeric_tokens) == 1
            and fp.numeric_tokens[0][0] >= 100
        )

        # Surface finish callouts: "Ra <value>" or "<value> Ra" (e.g., "1000 Ra",
        # "FINISH TURN 1000 Ra").  The "Ra" indicator alone with a numeric value is
        # sufficient to identify a surface finish annotation.
        has_surface_finish = bool(
            re.search(r'\bRa\b', text, re.IGNORECASE) and fp.numeric_tokens
        )

        # Filter for spans that look like dimension requirements:
        # - Must have numeric tokens (dimension values)
        # - Must have symbols (Ø, R) OR count patterns (2X, 4X) OR angle-style callouts
        #   OR be a pure leading-decimal single value (engineering notation for length)
        #   OR be a surface finish annotation (Ra indicator)
        if not fp.numeric_tokens:
            continue

        if not (fp.symbol_tokens or fp.count_tokens or has_angle_dimension or is_leading_decimal_single or is_plain_decimal_single or has_surface_finish or is_plain_integer_dimension):
            continue

        # Skip spans that look like revision notes (contain "ADDED", "NEW", etc.)
        if any(word in fp.norm_text for word in ["ADDED", "NEW", "REVISED", "DELETED"]):
            continue

        # Skip non-leading-decimal, non-surface-finish single numeric values
        if len(fp.numeric_tokens) == 1 and not fp.symbol_tokens and not fp.count_tokens and not is_leading_decimal_single and not is_plain_decimal_single and not has_surface_finish and not is_plain_integer_dimension:
            continue

        # Suppress spans that sit close to an already-matched span — they are likely
        # companion parts of an existing annotation (e.g., "Ø.531 X 82.0°" adjacent to
        # a matched "Ø.266 THRU" countersink callout, or tolerance limit companions).
        if is_near_matched_span(span.bbox_pdf):
            continue

        # This looks like a new characteristic
        reasons = [
            f"New requirement detected in Rev B: \"{text}\"",
            f"Symbols: {fp.symbol_tokens}, Counts: {fp.count_tokens}"
        ]
        if has_angle_dimension and not fp.symbol_tokens and not fp.count_tokens:
            reasons.append("Angle-style callout detected")
        if is_leading_decimal_single:
            reasons.append("Leading-decimal dimension annotation detected")
        if is_plain_decimal_single:
            reasons.append("Plain decimal dimension annotation detected")
        if has_surface_finish:
            reasons.append("Surface finish annotation detected (Ra indicator)")
        if is_plain_integer_dimension:
            reasons.append("Plain integer dimension annotation detected (≥100)")

        # Calculate confidence based on how "dimension-like" the span is
        confidence = 0.6
        if fp.symbol_tokens:
            confidence += 0.15
        if fp.count_tokens:
            confidence += 0.15
        if fp.pattern_class in ["hole", "dimension", "fillet"]:
            confidence += 0.1
        if has_surface_finish:
            confidence += 0.10
        if is_leading_decimal_single or is_plain_decimal_single:
            confidence = 0.55  # Slightly lower confidence for bare decimal values
        if is_plain_integer_dimension:
            confidence = 0.55  # Slightly lower confidence for bare integers
        confidence = min(confidence, 0.95)

        added_item = DeltaItem(
            char_no=current_char_no,
            status="added",
            confidence=confidence,
            reasons=reasons,
            component_scores={
                "location": 0.0,
                "text": 1.0,
                "context": 0.0
            },
            match=None,
            added_span=span
        )
        added_items.append(added_item)
        current_char_no += 1

    return added_items
