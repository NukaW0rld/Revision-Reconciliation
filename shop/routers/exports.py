"""Audit packet and work order download routes."""
import io
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from shop.dependencies import get_db, get_current_user
from shop.models import Run, User

router = APIRouter()


def _get_signed_run(run_id: int, db: Session) -> Run:
    """Return run if signed_off; raise 404/403 otherwise."""
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != "signed_off":
        raise HTTPException(status_code=403, detail="Run is not signed off")
    return run


@router.get("/{run_id}/audit-packet.pdf")
def download_audit_packet_pdf(
    run_id: int,
    version: int = 1,  # which version to download (1-based)
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = _get_signed_run(run_id, db)

    # Try to serve from stored path for the requested version
    versions = run.packet_versions or []
    entry = next((v for v in versions if v.get("version") == version), None)
    if entry:
        stored_path = Path(entry["path"])
        if stored_path.exists():
            return FileResponse(
                path=str(stored_path),
                media_type="application/pdf",
                filename=f"audit_packet_{run_id}_v{version}.pdf",
            )

    # Fallback: re-render (handles test environments without real output_dir)
    from shop.services.exports import render_audit_packet_pdf
    from shop.models import ShopConfig
    shop_config = db.query(ShopConfig).filter(ShopConfig.id == 1).first()
    pdf_bytes = render_audit_packet_pdf(db, run, shop_config)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="audit_packet_{run_id}_v{version}.pdf"'},
    )


@router.get("/{run_id}/audit-packet.csv")
def download_audit_packet_csv(
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = _get_signed_run(run_id, db)
    from shop.services.exports import generate_audit_packet_csv
    buf = generate_audit_packet_csv(db, run)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="audit_packet_{run_id}.csv"'},
    )


@router.get("/{run_id}/work-order.pdf")
def download_work_order_pdf(
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = _get_signed_run(run_id, db)
    from shop.services.exports import generate_work_order_pdf
    pdf_bytes = generate_work_order_pdf(db, run)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="work_order_{run_id}.pdf"'},
    )


@router.get("/{run_id}/work-order.csv")
def download_work_order_csv(
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = _get_signed_run(run_id, db)
    from shop.services.exports import generate_work_order_csv
    buf = generate_work_order_csv(db, run)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="work_order_{run_id}.csv"'},
    )
