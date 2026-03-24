"""
Core data types for the delta preservation pipeline.

This module defines the primary data structures used throughout the revision
reconciliation process, particularly for the output delta packet that contains
the results of characteristic matching and classification.

These types provide structured, validated data models for JSON serialization
and ensure consistent interfaces between pipeline stages.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """
    Visual evidence snippet for a characteristic in a specific drawing revision.

    Evidence provides spatial and visual context for characteristic matching
    decisions, enabling human review and validation of automated classifications.
    Each Evidence object points to both the PDF coordinate location and an
    extracted image snippet for visual inspection.

    Attributes:
        page: Zero-based page index where the characteristic was found
        bbox: Bounding box in PDF coordinates [x0, y0, x1, y1] in points
        image_path: Relative path to evidence snippet image file (None if generation failed)

    Notes:
        - PDF coordinates use standard 72 DPI point units
        - Image paths are relative to the output directory for portability
        - bbox may extend beyond actual content due to padding for context
    """

    page: int = Field(..., description="Zero-based page index in PDF")
    bbox: List[float] = Field(..., description="Bounding box coordinates [x0, y0, x1, y1] in PDF points")
    image_path: Optional[str] = Field(None, description="Relative path to evidence snippet image")


class SemanticProvenance(BaseModel):
    """Source metadata describing where semantic callout data came from."""

    authority: Literal["pdf", "form3", "combined", "none"] = Field(
        ..., description="Source selected as authoritative for semantic extraction"
    )
    source_type: Literal["drawing_pdf", "form3_requirement", "combined", "none"] = Field(
        ..., description="Origin family for the semantic source"
    )
    source_ref: Optional[str] = Field(
        None,
        description="Redacted source locator such as page/span or Form 3 row reference",
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Human-readable provenance notes that avoid secret-bearing paths",
    )


class SemanticParserStatus(BaseModel):
    """Structured parser state so empty and deferred outcomes stay inspectable."""

    state: Literal["not_requested", "parsed", "empty", "not_implemented", "skipped", "error"] = Field(
        ..., description="High-level parser outcome for this semantic envelope"
    )
    parser_family: Literal["semantic_dispatch", "gdt", "weld", "surface_finish", "fit", "none"] = Field(
        ..., description="Parser family responsible for this semantic result"
    )
    reason_code: Optional[str] = Field(
        None,
        description="Machine-friendly reason for empty/skipped/not-implemented/error outcomes",
    )
    detail: Optional[str] = Field(
        None,
        description="Human-readable explanation of the parser outcome",
    )


class GdtSemanticPayload(BaseModel):
    """GD&T-specific semantic payload slot for future parser output."""

    frame_text: Optional[str] = Field(None, description="Normalized feature control frame text")
    control_type: Optional[str] = Field(None, description="Primary GD&T control type")
    datum_refs: List[str] = Field(default_factory=list, description="Referenced datums in order")
    modifiers: List[str] = Field(default_factory=list, description="Applied GD&T modifiers")


class WeldSemanticPayload(BaseModel):
    """Weld-callout semantic payload slot for future parser output."""

    process: Optional[str] = Field(None, description="Normalized weld process or symbol family")
    size: Optional[str] = Field(None, description="Weld size or leg specification")
    contour: Optional[str] = Field(None, description="Contour or finish instruction")
    side: Optional[str] = Field(None, description="Arrow side / other side / both sides")


class SurfaceFinishSemanticPayload(BaseModel):
    """Surface-finish semantic payload slot for future parser output."""

    roughness_value: Optional[str] = Field(None, description="Normalized roughness value")
    units: Optional[str] = Field(None, description="Surface-finish units such as microinch or µm")
    process_note: Optional[str] = Field(None, description="Supplemental process note")
    sampling_length: Optional[str] = Field(None, description="Sampling length or related modifier")


class FitSemanticPayload(BaseModel):
    """Fit/class semantic payload slot for future parser output."""

    fit_class: Optional[str] = Field(None, description="Fit class such as H7/g6")
    basis: Optional[str] = Field(None, description="Hole-basis or shaft-basis fit system")
    allowance: Optional[str] = Field(None, description="Allowance or fit note text")
    tolerance_note: Optional[str] = Field(None, description="Supplemental fit/tolerance note")


class SemanticCallout(BaseModel):
    """Shared semantic envelope attached additively to packet items."""

    provenance: SemanticProvenance = Field(..., description="Semantic source metadata")
    status: SemanticParserStatus = Field(..., description="Structured parser execution status")
    raw_text: Optional[str] = Field(None, description="Normalized raw semantic source text")
    normalized_text: Optional[str] = Field(None, description="Canonicalized semantic text for future comparisons")
    gdt: Optional[GdtSemanticPayload] = Field(None, description="GD&T semantic payload slot")
    weld: Optional[WeldSemanticPayload] = Field(None, description="Weld semantic payload slot")
    surface_finish: Optional[SurfaceFinishSemanticPayload] = Field(
        None, description="Surface-finish semantic payload slot"
    )
    fit: Optional[FitSemanticPayload] = Field(None, description="Fit semantic payload slot")
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Shared semantic metadata for additive future expansion",
    )


class DeltaItem(BaseModel):
    """
    Classification result for a single characteristic in the revision comparison.

    A DeltaItem represents the outcome of comparing a Rev A characteristic with
    its potential match in Rev B. It includes the classification decision
    (unchanged/changed/removed/added), confidence metrics, and supporting evidence
    for human review and audit purposes.

    Attributes:
        char_no: Characteristic number from Form 3 (None for system-assigned added items)
        status: Classification result - "unchanged", "changed", "removed", "added", or "uncertain"
        confidence: Classification confidence score between 0.0 and 1.0
        reasons: Human-readable explanations for the classification decision
        scores: Component scores breakdown for transparency (location, text, context)
        revA: Visual evidence from Rev A PDF (None for added characteristics)
        revB: Visual evidence from Rev B PDF (None for removed characteristics)
        semantic_callout: Optional semantic envelope carrying typed drawing semantics

    Notes:
        - Added characteristics receive new char_no values starting from max existing + 1
        - Confidence combines spatial, semantic, and contextual matching signals
        - Reasons provide audit trail for regulatory compliance
        - Evidence enables visual verification of automated decisions
        - semantic_callout is additive and optional for backward compatibility
    """

    char_no: Optional[int] = Field(None, description="Characteristic number from AS9102 Form 3")
    status: str = Field(..., description="Classification: unchanged/changed/removed/added/uncertain")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")
    reasons: List[str] = Field(..., description="Human-readable explanations for classification")
    scores: Dict[str, float] = Field(..., description="Component scores (location, text, context)")
    revA: Optional[Evidence] = Field(None, description="Rev A visual evidence")
    revB: Optional[Evidence] = Field(None, description="Rev B visual evidence")
    semantic_callout: Optional[SemanticCallout] = Field(
        None,
        description="Optional typed semantic callout envelope sourced from drawing/Form 3 text",
    )


class DeltaPacket(BaseModel):
    """
    Complete output package containing all revision comparison results.

    The DeltaPacket is the primary deliverable of the delta preservation pipeline,
    containing structured results that enable human review and approval of revision
    changes. It packages classification results with metadata and evidence for
    audit compliance and quality assurance workflows.

    Attributes:
        run_id: Unique identifier for this pipeline execution (timestamp + hash)
        inputs: Input file paths and configuration parameters used
        items: List of all characteristic classification results

    Notes:
        - run_id enables result traceability and prevents overwrites
        - inputs provide full provenance for audit requirements
        - items are sorted by char_no for consistent presentation
        - JSON serialization compatible with quality management systems
    """

    run_id: str = Field(..., description="Unique execution identifier")
    inputs: Dict[str, str] = Field(..., description="Input files and parameters")
    items: List[DeltaItem] = Field(..., description="All characteristic classification results")
