import logging
from io import BytesIO
from typing import Any

from openpyxl import load_workbook

from delta_preservation.config import FORM3_HEADER_KEYWORDS

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ["char_no", "requirement", "reference_location"]


def detect_column_mapping(headers: list[str | None]) -> dict[str, int]:
    """
    Auto-detect column indices for required fields using FORM3_HEADER_KEYWORDS.

    Returns {field_name: col_index} for confident matches.
    Unmatched fields are omitted from the result (caller highlights them as amber).
    Uses existing delta_preservation FORM3_HEADER_KEYWORDS — detection logic is not
    duplicated here.
    """
    detected: dict[str, int] = {}
    for field, keywords in FORM3_HEADER_KEYWORDS.items():
        if field not in REQUIRED_FIELDS:
            continue
        for col_idx, header in enumerate(headers):
            if header is None:
                continue
            header_lower = str(header).lower()
            if any(kw.lower() in header_lower for kw in keywords):
                detected[field] = col_idx
                break
    return detected


def parse_excel_preview(
    file_bytes: bytes,
) -> tuple[list[str | None], list[tuple[Any, ...]], dict[str, int]]:
    """
    Parse Excel bytes for column mapping preview and auto-detection.

    Returns: (headers, preview_rows[:5], detected_mapping)
    Raises ValueError with a descriptive message on fatal errors (empty file,
    unreadable bytes, sheet with no rows).
    Non-fatal issues (columns not auto-detected) result in those fields being
    omitted from detected_mapping — not raised as errors.
    """
    if not file_bytes:
        raise ValueError("File is empty — please upload a non-empty Excel file.")

    try:
        wb = load_workbook(BytesIO(file_bytes), read_only=True, data_only=True)
    except Exception as exc:
        raise ValueError(f"Cannot read file: {exc}") from exc

    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    if not rows:
        raise ValueError("Sheet has no rows — please upload a file with data.")

    headers = list(rows[0])
    preview_rows = rows[1:6]  # Up to 5 data rows for preview table
    detected = detect_column_mapping(headers)
    return headers, list(preview_rows), detected
