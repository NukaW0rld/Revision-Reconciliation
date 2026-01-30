"""
Core data types for the delta preservation pipeline.

This module defines the primary data structures used throughout the revision
reconciliation process, particularly for the output delta packet that contains
the results of characteristic matching and classification.

These types provide structured, validated data models for JSON serialization
and ensure consistent interfaces between pipeline stages.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


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
        
    Notes:
        - Added characteristics receive new char_no values starting from max existing + 1
        - Confidence combines spatial, semantic, and contextual matching signals
        - Reasons provide audit trail for regulatory compliance
        - Evidence enables visual verification of automated decisions
    """
    char_no: Optional[int] = Field(None, description="Characteristic number from AS9102 Form 3")
    status: str = Field(..., description="Classification: unchanged/changed/removed/added/uncertain")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")
    reasons: List[str] = Field(..., description="Human-readable explanations for classification")
    scores: Dict[str, float] = Field(..., description="Component scores (location, text, context)")
    revA: Optional[Evidence] = Field(None, description="Rev A visual evidence")
    revB: Optional[Evidence] = Field(None, description="Rev B visual evidence")


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
