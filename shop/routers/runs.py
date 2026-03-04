import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from shop.app import templates
from shop.dependencies import get_db, get_current_user
from shop.models import User, Run, RunAlert, ShopConfig

router = APIRouter(redirect_slashes=False)
logger = logging.getLogger(__name__)


def _get_nav_context(db: Session, user: User) -> dict:
    """Return common nav bar context variables for authenticated routes."""
    unread_count = db.query(RunAlert).filter(
        RunAlert.user_id == user.id, RunAlert.is_read == False  # noqa: E712
    ).count()
    config = db.query(ShopConfig).filter(ShopConfig.id == 1).first()
    shop_name = config.shop_name if config else "Delta Preservation"
    return {"unread_alert_count": unread_count, "shop_name": shop_name}


@router.get("", response_class=HTMLResponse)
def list_runs(
    request: Request,
    part_number: str = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    query = db.query(Run).order_by(Run.submitted_at.desc())
    if part_number:
        query = query.filter(Run.part_number.ilike(f"%{part_number}%"))
    runs = query.all()
    nav = _get_nav_context(db, user)
    return templates.TemplateResponse(request, "runs/list.html", {
        "user": user,
        "runs": runs,
        "part_number": part_number or "",
        **nav,
    })


@router.get("/new", response_class=HTMLResponse)
def new_run_form(
    request: Request,
    user: User = Depends(get_current_user),
):
    return templates.TemplateResponse(request, "runs/new.html", {"user": user})


@router.post("/validate-pdf", response_class=HTMLResponse)
async def validate_pdf(
    request: Request,
    file: UploadFile = File(...),
    field: str = Form(...),
    user: User = Depends(get_current_user),
):
    from shop.services.runs import validate_pdf_bytes

    content = await file.read()
    result = validate_pdf_bytes(content)
    if result["is_raster"]:
        return templates.TemplateResponse(
            request,
            "runs/_pdf_error.html",
            {
                "error": "This PDF appears to be a scanned image. Please upload a vector PDF.",
                "field": field,
            },
        )
    if result["page_count"] > 1:
        return templates.TemplateResponse(
            request,
            "runs/_page_selector.html",
            {"page_count": result["page_count"], "field": field},
        )
    return HTMLResponse("")  # Single-page valid PDF — clear any prior validation message


@router.post("/new")
async def submit_run(
    request: Request,
    revA_pdf: UploadFile = File(...),
    revB_pdf: UploadFile = File(...),
    form3_xlsx: UploadFile = File(...),
    part_number: str = Form(...),
    rev_a_label: str = Form(...),
    rev_b_label: str = Form(...),
    customer: str = Form(...),
    job_number: str = Form(...),
    revA_page: int = Form(0),
    revB_page: int = Form(0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    from shop.services.runs import validate_pdf_bytes, validate_excel_bytes, save_upload, create_run
    from shop.tasks import run_pipeline_task

    run_uuid = uuid.uuid4().hex

    revA_bytes = await revA_pdf.read()
    revB_bytes = await revB_pdf.read()
    form3_bytes = await form3_xlsx.read()

    # Validate Rev A PDF
    revA_info = validate_pdf_bytes(revA_bytes)
    if revA_info["is_raster"]:
        return templates.TemplateResponse(
            request,
            "runs/new.html",
            {"user": user, "error": "Rev A PDF appears to be a scanned image. Please upload a vector PDF."},
            status_code=422,
        )

    # Validate Rev B PDF
    revB_info = validate_pdf_bytes(revB_bytes)
    if revB_info["is_raster"]:
        return templates.TemplateResponse(
            request,
            "runs/new.html",
            {"user": user, "error": "Rev B PDF appears to be a scanned image. Please upload a vector PDF."},
            status_code=422,
        )

    # Validate Excel
    try:
        validate_excel_bytes(form3_bytes)
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "runs/new.html",
            {"user": user, "error": f"Form 3 Excel error: {exc}"},
            status_code=422,
        )

    # Save files to disk
    revA_path = save_upload(revA_bytes, ".pdf", run_uuid)
    revB_path = save_upload(revB_bytes, ".pdf", run_uuid)
    form3_path = save_upload(form3_bytes, ".xlsx", run_uuid)

    # Create Run record — submitter is the default reviewer
    run = create_run(
        db,
        part_number=part_number,
        rev_a_label=rev_a_label,
        rev_b_label=rev_b_label,
        customer=customer,
        job_number=job_number,
        revA_path=revA_path,
        revB_path=revB_path,
        form3_path=form3_path,
        revA_page=revA_page,
        revB_page=revB_page,
        reviewer_id=user.id,
    )

    # Enqueue pipeline task (runs synchronously in immediate/test mode via huey_immediate fixture)
    run_pipeline_task(run.id, str(revA_path), str(revB_path), str(form3_path), part_number)

    return RedirectResponse(f"/runs/{run.id}", status_code=302)


@router.post("/{run_id}/acknowledge-warning")
def acknowledge_warning(
    run_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Acknowledge a revB_balloon warning. Phase 3 will implement the review queue.
    For now, redirect to a placeholder review path."""
    return RedirectResponse(f"/review/{run_id}", status_code=302)
