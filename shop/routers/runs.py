import asyncio
import json as _json
import logging
import uuid
from collections.abc import AsyncIterable

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from sqlalchemy.orm import Session

from shop.app import templates
from shop.dependencies import get_db, get_current_user
from shop.models import User, Run, RunAlert, ShopConfig

router = APIRouter(redirect_slashes=False)
logger = logging.getLogger(__name__)

STAGE_NAMES = [
    "Form 3 parsing",
    "Balloon detection",
    "Text extraction",
    "Anchor building",
    "Alignment",
    "Candidate matching",
    "Classification",
    "Output",
]


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
    user: User = Depends(get_current_user),
):
    """Validate an uploaded PDF for raster content and page count.

    Accepts any multipart/form-data upload.  The file field name is flexible:
    HTMX sends the triggering <input name="revA_pdf"> as "revA_pdf", while
    direct API calls (e.g. tests) may send it as "file".  We look up the
    specific field first (e.g. "revB_pdf" when field="revB") to avoid picking
    up a sibling file input that HTMX also includes in the full-form submission.
    """
    from shop.services.runs import validate_pdf_bytes

    form = await request.form()
    field = str(form.get("field", ""))

    # Prefer the field-specific key (revA_pdf / revB_pdf) so that when HTMX
    # submits the whole form on change, a previously-uploaded sibling file
    # (e.g. revA_pdf already selected) doesn't get validated instead.
    target_key = f"{field}_pdf" if field else None
    uploaded_file = form.get(target_key) if target_key else None
    if not hasattr(uploaded_file, "read"):
        # Fallback: tests send as "file"; direct API calls may omit field name.
        uploaded_file = None
        for key, value in form.multi_items():
            if hasattr(value, "read"):
                uploaded_file = value
                break

    if uploaded_file is None:
        return HTMLResponse("")  # No file uploaded — clear any prior validation message

    content = await uploaded_file.read()
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
    # Single-page valid PDF — restore the hidden page input so the form field is
    # always present regardless of whether the user previously uploaded a multi-page
    # PDF (which would have replaced this input with a <select>).
    return HTMLResponse(
        f'<input type="hidden" name="{field}_page" value="0">'
    )


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
    print("HUEY TASK: enqueuing run_id=", run.id)
    logger.info("submit_run: enqueuing pipeline task for run_id=%d", run.id)
    run_pipeline_task(run.id, str(revA_path), str(revB_path), str(form3_path), part_number)

    return RedirectResponse(f"/runs/{run.id}", status_code=302)


@router.get("/{run_id}", response_class=HTMLResponse)
def run_status(
    run_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    nav = _get_nav_context(db, user)
    stages = [{"index": i, "name": n} for i, n in enumerate(STAGE_NAMES)]
    engineers = db.query(User).filter(User.is_active == True, User.role == "engineer").all()  # noqa: E712
    signed = request.query_params.get("signed") == "1"
    return templates.TemplateResponse(request, "runs/status.html", {
        "user": user,
        "run": run,
        "stages": stages,
        "engineers": engineers,
        "signed": signed,
        **nav,
    })


@router.get("/{run_id}/sse", response_class=EventSourceResponse)
async def run_sse(
    run_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AsyncIterable[ServerSentEvent]:
    """Stream run stage progress as Server-Sent Events.

    Polls the DB every 1 second. Closes the stream when the run reaches a
    terminal state (completed/failed/warning), preventing browser reconnect loops.

    Two bugs fixed here:
    1. SSE data was double-encoded: ServerSentEvent(data=json_str) JSON-encodes the
       string a second time on the wire, so JS JSON.parse(e.data) received a string
       instead of an object. Fix: use raw_data= so the pre-serialised JSON string is
       sent verbatim.
    2. SQLAlchemy identity-map caching: db.query(Run) inside the loop returned the
       same cached instance every iteration, so run.status never changed. Fix: call
       db.expire(run) + db.refresh(run) before each poll to force a fresh DB read.
    """
    terminal = {"completed", "failed", "warning"}
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        return
    while True:
        if await request.is_disconnected():
            break
        # Expire the cached instance so the next attribute access re-queries the DB.
        db.expire(run)
        db.refresh(run)
        payload = _json.dumps({
            "status": run.status,
            "current_stage_index": run.current_stage_index,
            "current_stage": run.current_stage,
            "failure_stage": run.failure_stage,
            "failure_message": run.failure_message,
            "warning_type": run.warning_type,
            "confidence_summary": run.confidence_summary,
        })
        yield ServerSentEvent(raw_data=payload, event="stage_update")
        if run.status in terminal:
            yield ServerSentEvent(event="close", raw_data="done")
            break
        await asyncio.sleep(1.0)


@router.post("/{run_id}/abort")
def abort_run(
    run_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status == "warning" and run.warning_type == "low_confidence":
        run.status = "failed"
        run.failure_stage = "Alignment"
        run.failure_message = "Run aborted by engineer due to low alignment confidence."
        db.commit()
    return RedirectResponse(f"/runs/{run_id}", status_code=302)


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
