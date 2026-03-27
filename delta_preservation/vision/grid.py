"""
Drawing grid inference from engineering drawing PDF borders.

Engineering drawings follow a standard grid coordinate system with labels on all four
borders (top, bottom, left, right). Columns are labeled along the top and bottom;
rows are labeled along the left and right. Labels are single uppercase letters or
single non-zero digits. The ordering may be ascending or descending.

This module:
1. Extracts border text from a PDF page using PyMuPDF
2. Identifies grid labels by requiring them to appear on BOTH opposite borders
   (top+bottom for columns, left+right for rows) at the same position
3. Computes zone cell boundaries as midpoints between consecutive label positions
4. Provides zone-string → PDF bbox conversion for use in characteristic matching
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz


@dataclass
class DrawingGrid:
    """
    Represents the zone grid inferred from an engineering drawing's border labels.

    Attributes:
        col_labels: List of (label, x_center) tuples sorted left to right.
                    Labels are the column identifiers (e.g. '8', '7', ..., '1').
        row_labels: List of (label, y_center) tuples sorted top to bottom.
                    Labels are the row identifiers (e.g. 'H', 'G', ..., 'A').
        page_width:  PDF page width in points.
        page_height: PDF page height in points.
    """

    col_labels: List[Tuple[str, float]]  # (label, x_center), sorted left → right
    row_labels: List[Tuple[str, float]]  # (label, y_center), sorted top → bottom
    page_width: float
    page_height: float

    def zone_to_bbox(self, zone_str: str) -> Optional[Tuple[float, float, float, float]]:
        """
        Convert a zone label string (e.g. ``"B.2"``) to a PDF bounding box.

        The zone string consists of two components separated by an optional ``.``:
        one column label and one row label (order may vary).  The function checks
        the grid's known column and row labels to determine which component is which.

        Args:
            zone_str: Zone label such as ``"B.2"``, ``"B2"``, ``"2.B"``, or ``"2B"``.

        Returns:
            ``(x0, y0, x1, y1)`` in PDF points, or ``None`` if the zone cannot be
            resolved (unknown labels or unparseable string).
        """
        parts = _parse_zone_parts(zone_str)
        if parts is None:
            return None
        p1, p2 = parts

        col_map: Dict[str, float] = {lbl: x for lbl, x in self.col_labels}
        row_map: Dict[str, float] = {lbl: y for lbl, y in self.row_labels}

        col_label: Optional[str] = None
        row_label: Optional[str] = None

        for part in (p1, p2):
            if col_label is None and part in col_map:
                col_label = part
            elif row_label is None and part in row_map:
                row_label = part

        if col_label is None or row_label is None:
            return None

        # --- Column x-boundaries ---
        col_xs = [x for _, x in self.col_labels]
        col_x = col_map[col_label]
        col_idx = col_xs.index(col_x)

        x0 = 0.0 if col_idx == 0 else (col_xs[col_idx - 1] + col_x) / 2
        x1 = self.page_width if col_idx == len(col_xs) - 1 else (col_x + col_xs[col_idx + 1]) / 2

        # --- Row y-boundaries ---
        row_ys = [y for _, y in self.row_labels]
        row_y = row_map[row_label]
        row_idx = row_ys.index(row_y)

        y0 = 0.0 if row_idx == 0 else (row_ys[row_idx - 1] + row_y) / 2
        y1 = self.page_height if row_idx == len(row_ys) - 1 else (row_y + row_ys[row_idx + 1]) / 2

        return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def parse_zone_from_reference_location(ref_loc: str) -> Optional[str]:
    """
    Extract a zone label from a Form 3 Reference Location string.

    Handles formats such as:

    * ``"revA pg.1, Zone B.2"`` → ``"B.2"``
    * ``"Zone B2"``             → ``"B2"``
    * ``"B.2"``                 → ``"B.2"``  (bare zone, no prefix)

    Args:
        ref_loc: Raw Reference Location string from AS9102 Form 3 Field 6.

    Returns:
        Zone string (e.g. ``"B.2"``) or ``None`` if no zone is detected.
    """
    if not ref_loc:
        return None

    # "Zone B.2" or "Zone B2" (case-insensitive)
    m = re.search(r'[Zz]one\s+([A-Za-z0-9]+\.?[A-Za-z0-9]*)', ref_loc)
    if m:
        return m.group(1).strip()

    # Bare zone at start or end: e.g. "B.2" or "2.B" or "B2"
    m = re.match(
        r'^\s*([A-Za-z]\d+|\d+[A-Za-z]|[A-Za-z]\.\d+|\d+\.[A-Za-z])\s*$',
        ref_loc,
    )
    if m:
        return m.group(1).strip()

    return None


def infer_grid(pdf_path: Path, page_index: int = 0) -> Optional["DrawingGrid"]:
    """
    Infer the drawing zone grid from an engineering drawing PDF page.

    Algorithm:
    1. Extract all text spans from the page using PyMuPDF.
    2. Filter spans to those very close to one of the four borders (within
       ``BORDER_THRESHOLD`` points) whose text is a single uppercase letter or
       single non-zero digit (the canonical form of a grid label).
    3. For columns: find labels that appear on **both** the top and bottom borders
       at approximately the same x-position.
    4. For rows: find labels that appear on **both** the left and right borders at
       approximately the same y-position.
       Requiring bi-border agreement suppresses noise (balloon numbers, revision
       letters, title-block content) that appears on only one border.
    5. Sort matched column labels by x (left → right) and row labels by y
       (top → bottom).

    Args:
        pdf_path:   Path to the PDF file.
        page_index: Zero-based page index (default 0).

    Returns:
        :class:`DrawingGrid` if at least 2 column and 2 row labels are found,
        otherwise ``None``.
    """
    doc = fitz.open(pdf_path)
    if page_index < 0 or page_index >= len(doc):
        doc.close()
        return None

    page = doc.load_page(page_index)
    pw: float = page.rect.width
    ph: float = page.rect.height
    text_dict = page.get_text("dict")
    doc.close()

    BORDER_THRESHOLD = 60.0

    # Accumulate candidate grid labels per border.
    # Columns: (label, x_center); Rows: (label, y_center)
    top_cands: List[Tuple[str, float]] = []
    bot_cands: List[Tuple[str, float]] = []
    left_cands: List[Tuple[str, float]] = []
    right_cands: List[Tuple[str, float]] = []

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text: str = span["text"].strip()
                if not _is_grid_label(text):
                    continue

                x0, y0, x1, y1 = span["bbox"]
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2

                is_top = y0 < BORDER_THRESHOLD
                is_bot = y1 > ph - BORDER_THRESHOLD
                is_left = x0 < BORDER_THRESHOLD
                is_right = x1 > pw - BORDER_THRESHOLD

                # Assign to exactly one border; corner spans are excluded to
                # avoid double-counting the same label as both row and column.
                if is_top and not is_left and not is_right:
                    top_cands.append((text, cx))
                if is_bot and not is_left and not is_right:
                    bot_cands.append((text, cx))
                if is_left and not is_top and not is_bot:
                    left_cands.append((text, cy))
                if is_right and not is_top and not is_bot:
                    right_cands.append((text, cy))

    # Require labels to appear on BOTH opposite borders at the same position.
    col_labels = _match_opposite_borders(top_cands, bot_cands, position_tolerance=25.0)
    row_labels = _match_opposite_borders(left_cands, right_cands, position_tolerance=25.0)

    if len(col_labels) < 2 or len(row_labels) < 2:
        return None

    return DrawingGrid(
        col_labels=sorted(col_labels, key=lambda lx: lx[1]),
        row_labels=sorted(row_labels, key=lambda ly: ly[1]),
        page_width=pw,
        page_height=ph,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_grid_label(text: str) -> bool:
    """Return True for a single uppercase letter or single non-zero digit."""
    if len(text) != 1:
        return False
    if text.isalpha():
        return text.upper() == text  # uppercase only
    if text.isdigit():
        return text != "0"
    return False


def _parse_zone_parts(zone_str: str) -> Optional[Tuple[str, str]]:
    """
    Split a zone string into its two label components.

    Supported formats:
    * ``"B.2"`` or ``"2.B"`` (dot separator)
    * ``"B2"``  or ``"2B"``  (concatenated, letter + digit)

    Returns a 2-tuple ``(part1, part2)`` with both parts uppercased, or ``None``
    if the string cannot be parsed.
    """
    z = zone_str.strip()

    # Dot-separated: letter.digit or digit.letter
    m = re.match(r'^([A-Za-z])\.(\d+)$', z)
    if m:
        return m.group(1).upper(), m.group(2)

    m = re.match(r'^(\d+)\.([A-Za-z])$', z)
    if m:
        return m.group(1), m.group(2).upper()

    # Concatenated: letter+digits or digits+letter
    m = re.match(r'^([A-Za-z])(\d+)$', z)
    if m:
        return m.group(1).upper(), m.group(2)

    m = re.match(r'^(\d+)([A-Za-z])$', z)
    if m:
        return m.group(1), m.group(2).upper()

    return None


def _match_opposite_borders(
    side_a: List[Tuple[str, float]],
    side_b: List[Tuple[str, float]],
    position_tolerance: float = 25.0,
) -> List[Tuple[str, float]]:
    """
    Return labels that appear on both ``side_a`` and ``side_b`` at approximately
    the same perpendicular position.

    Each matched pair is collapsed to a single ``(label, avg_position)`` entry.
    """
    matched: List[Tuple[str, float]] = []
    used_b: set = set()

    for label_a, pos_a in side_a:
        for i, (label_b, pos_b) in enumerate(side_b):
            if i in used_b:
                continue
            if label_a == label_b and abs(pos_a - pos_b) <= position_tolerance:
                matched.append((label_a, (pos_a + pos_b) / 2.0))
                used_b.add(i)
                break

    return matched
