# Form 3 parsing

from pathlib import Path
from typing import List
import json
from openpyxl import load_workbook


class Characteristic:
    """
    Represents a single row from AS9102 Form 3.
    
    Attributes:
        char_no: Integer characteristic number (e.g. 1, 2, 3...)
        reference_location: Reference location string from Form 3
        characteristic_designator: Characteristic designator string
        requirement: Requirement text string
    """
    def __init__(self, char_no: int, reference_location: str, 
                 characteristic_designator: str, requirement: str):
        self.char_no = char_no
        self.reference_location = reference_location
        self.characteristic_designator = characteristic_designator
        self.requirement = requirement
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "char_no": self.char_no,
            "reference_location": self.reference_location,
            "characteristic_designator": self.characteristic_designator,
            "requirement": self.requirement
        }


def load_form3(xlsx_path: Path, intermediate_dir: Path) -> List[Characteristic]:
    """
    Parse AS9102 Form 3 from an Excel file and extract characteristic rows.
    
    This function:
    1. Opens the workbook and selects the "Form3" or "F3" worksheet
    2. Scans header row 6 to identify column positions for:
       - Char no
       - Reference location
       - Characteristic Designator
       - Requirement
    3. Falls back to a fixed column map if headers are not found:
       - Char no: column A (index 1)
       - Reference location: column B (index 2)
       - Characteristic Designator: column C (index 3)
       - Requirement: column D (index 4)
    4. Iterates from row 7 downward until Char no cell is empty
    5. Parses each row into a Characteristic object with minimal cleanup:
       - Char no: parsed as integer
       - Other fields: stripped strings (empty string if blank/None)
    6. Writes raw parsed data to intermediate/form3_chars.json for debugging
    7. Returns the list of Characteristic objects
    
    Args:
        xlsx_path: Path to the Form 3 Excel file
        intermediate_dir: Directory for debug output (form3_chars.json)
    
    Returns:
        List of Characteristic objects, one per valid row starting at row 7
    
    Raises:
        FileNotFoundError: If xlsx_path does not exist
        ValueError: If neither "Form3" nor "F3" worksheet is found
    """
    # Load workbook and select Form3 sheet
    wb = load_workbook(xlsx_path, data_only=True)
    
    # Look for sheet containing "Form3" or "F3"
    sheet_name = None
    for name in wb.sheetnames:
        if "Form3" in name or "F3" in name:
            sheet_name = name
            break
    
    if sheet_name is None:
        raise ValueError(f"Neither 'Form3' nor 'F3' worksheet found in {xlsx_path}")
    
    ws = wb[sheet_name]
    
    # Default column indices
    # Notes: When using ws.cell() method, column indices are 1-based
    #        But when using ws.columns or ws.rows, column indices are 0-based
    FALLBACK_COLUMNS = {
        "char_no": 1,           # Column A
        "reference_location": 2, # Column B
        "characteristic_designator": 3,  # Column C
        "requirement": 4         # Column D
    }
    
    # Attempt to find columns by scanning header row 6
    column_map = {}
    header_keywords = {
        "char_no": ["char", "no", "number"],
        "reference_location": ["reference", "location", "ref"],
        "characteristic_designator": ["characteristic", "designator"],
        "requirement": ["requirement", "req"]
    }
    
    # Scan row 6 for headers
    for col_idx in range(1, ws.max_column + 1):
        cell_value = ws.cell(row=6, column=col_idx).value
        if cell_value is None:
            continue
        
        cell_text = str(cell_value).lower().strip()
        
        # Check each field we're looking for
        for field_name, keywords in header_keywords.items():
            if field_name not in column_map:
                # Check if all keywords appear in cell text
                if all(kw in cell_text for kw in keywords):
                    column_map[field_name] = col_idx
    
    # Use fallback for any missing columns
    for field_name, fallback_col in FALLBACK_COLUMNS.items():
        if field_name not in column_map:
            column_map[field_name] = fallback_col
    
    # Parse characteristics starting from row 7
    characteristics = []
    row_idx = 7
    
    while True:
        # Read Char no cell
        char_no_cell = ws.cell(row=row_idx, column=column_map["char_no"]).value
        
        # Stop if Char no is empty/None
        if char_no_cell is None or str(char_no_cell).strip() == "":
            break
        
        # Parse Char no as integer
        try:
            char_no = int(char_no_cell)
        except (ValueError, TypeError):
            # Skip rows where Char no cannot be parsed as integer
            row_idx += 1
            continue
        
        # Read other fields and coerce to stripped strings
        ref_loc_cell = ws.cell(row=row_idx, column=column_map["reference_location"]).value
        reference_location = str(ref_loc_cell).strip() if ref_loc_cell is not None else ""
        
        char_des_cell = ws.cell(row=row_idx, column=column_map["characteristic_designator"]).value
        characteristic_designator = str(char_des_cell).strip() if char_des_cell is not None else ""
        
        req_cell = ws.cell(row=row_idx, column=column_map["requirement"]).value
        requirement = str(req_cell).strip() if req_cell is not None else ""
        
        # Create Characteristic object
        char = Characteristic(
            char_no=char_no,
            reference_location=reference_location,
            characteristic_designator=characteristic_designator,
            requirement=requirement
        )
        characteristics.append(char)
        
        row_idx += 1
    
    # Serialize to JSON for debugging
    debug_output = [char.to_dict() for char in characteristics]
    debug_path = intermediate_dir / "form3_chars.json"
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_output, f, indent=2, ensure_ascii=False)
    
    return characteristics
