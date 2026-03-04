import os
import uuid
from pathlib import Path
from typing import Optional

import fitz

from shop.models import Run
from shop.services.form3 import parse_excel_preview

_default_uploads_dir = os.environ.get("UPLOADS_DIR", "/app/data/uploads")

# When running outside Docker (dev or tests), /app/data may not exist.
# Fall back to a local path that can always be created.
if not Path(_default_uploads_dir).parent.exists():
    _default_uploads_dir = str(Path(__file__).parent.parent.parent / "uploads")

UPLOADS_DIR = _default_uploads_dir


def save_upload(file_bytes: bytes, suffix: str, run_uuid: str) -> Path:
    """Save file bytes to {UPLOADS_DIR}/{run_uuid}/{uuid}{suffix} and return Path."""
    dest_dir = Path(UPLOADS_DIR) / run_uuid
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}{suffix}"
    dest = dest_dir / filename
    dest.write_bytes(file_bytes)
    return dest


def validate_pdf_bytes(content: bytes) -> dict:
    """Return {"is_raster": bool, "page_count": int}.

    is_raster is True when the first page has no extractable text AND
    image coverage >= 95% of the page area.
    """
    doc = fitz.open(stream=content, filetype="pdf")
    page = doc.load_page(0)
    page_count = doc.page_count

    # Primary check: if text is extractable the PDF is vector, not raster
    if page.get_text().strip():
        doc.close()
        return {"is_raster": False, "page_count": page_count}

    page_area = abs(page.rect)
    if page_area == 0:
        doc.close()
        return {"is_raster": False, "page_count": page_count}

    images = page.get_images(full=True)
    total_image_area = sum(
        abs(rect)
        for img in images
        for rect in page.get_image_rects(img[0])
    )
    doc.close()
    is_raster = bool(images) and (total_image_area / page_area) >= 0.95
    return {"is_raster": is_raster, "page_count": page_count}


def validate_excel_bytes(content: bytes) -> None:
    """Raises ValueError if Excel is unreadable or empty. Returns None on success."""
    parse_excel_preview(content)  # raises ValueError on fatal errors


def create_run(
    db,
    *,
    part_number: str,
    rev_a_label: str,
    rev_b_label: str,
    customer: str,
    job_number: str,
    revA_path,
    revB_path,
    form3_path,
    revA_page: int = 0,
    revB_page: int = 0,
    reviewer_id: Optional[int] = None,
) -> Run:
    """Create a Run record with status='queued', commit, and return the Run."""
    run = Run(
        part_number=part_number,
        rev_a_label=rev_a_label,
        rev_b_label=rev_b_label,
        customer=customer,
        job_number=job_number,
        revA_path=str(revA_path),
        revB_path=str(revB_path),
        form3_path=str(form3_path),
        revA_page=revA_page,
        revB_page=revB_page,
        reviewer_id=reviewer_id,
        status="queued",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run
