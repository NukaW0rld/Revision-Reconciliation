"""
Configuration management for the delta preservation pipeline.

This module provides centralized configuration management for the revision
reconciliation system. It handles default parameters, validation, and
configuration file loading to ensure consistent behavior across different
execution environments.

Currently the system uses command-line arguments for configuration, but this
module provides a foundation for more sophisticated configuration management
as the system evolves.
"""

# Default pipeline parameters
DEFAULT_DPI = 300
DEFAULT_OUTPUT_DIR = "./out" 
DEFAULT_PART_NAME = "part"

# Search and matching parameters
DEFAULT_SEARCH_RADIUS = 144.0  # PDF points (~2 inches at 72 DPI)
DEFAULT_CONTEXT_RADIUS = 150.0  # PDF points for local context
DEFAULT_TOP_K_CANDIDATES = 5

# Quality thresholds
MIN_ALIGNMENT_INLIERS = 40
MIN_ALIGNMENT_RATIO = 0.15
MIN_MATCH_SCORE_THRESHOLD = 0.2

# File validation
SUPPORTED_PDF_EXTENSIONS = {'.pdf'}
SUPPORTED_EXCEL_EXTENSIONS = {'.xlsx', '.xls'}

# AS9102 Form 3 column mapping
FORM3_DEFAULT_COLUMNS = {
    "char_no": 1,           # Column A: Characteristic Number
    "reference_location": 2, # Column B: Reference Location 
    "characteristic_designator": 3,  # Column C: Characteristic Designator
    "requirement": 4         # Column D: Requirement
}

# Expected Form 3 header keywords for intelligent column detection
FORM3_HEADER_KEYWORDS = {
    "char_no": ["char", "no", "number"],
    "reference_location": ["reference", "location", "ref"],
    "characteristic_designator": ["characteristic", "designator"],
    "requirement": ["requirement", "req"]
}