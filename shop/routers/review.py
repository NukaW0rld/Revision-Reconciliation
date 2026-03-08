import asyncio
import json as _json
from collections.abc import AsyncIterable
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from sqlalchemy.orm import Session
from shop.app import templates
from shop.dependencies import get_db, get_current_user
from shop.models import User, Run, ReviewItem
from shop.services.review import open_review_queue, attempt_sign_off
from shop.routers.runs import _get_nav_context

router = APIRouter(redirect_slashes=False)


@router.get("/{run_id}", response_class=HTMLResponse)
def review_queue(
    run_id: int,
    request: Request,
    status_filter: str = "all",
    classification_filter: str = "all",
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404)
    # Only allow review of completed/warning/reviewing/signing_off/signed_off runs
    if run.status not in ("completed", "warning", "reviewing", "signing_off", "signed_off"):
        return RedirectResponse(f"/runs/{run_id}", status_code=302)

    all_items = open_review_queue(db, run)

    # Counts (unfiltered — sign-off gate counts all regardless of filter)
    pending = sum(1 for i in all_items if i.reviewer_decision is None)
    approved = sum(1 for i in all_items if i.reviewer_decision == "approved")
    overridden = sum(1 for i in all_items if i.reviewer_decision == "overridden")
    total = len(all_items)

    # Apply filters
    visible_items = all_items
    if status_filter == "pending":
        visible_items = [i for i in all_items if i.reviewer_decision is None]
    elif status_filter == "approved":
        visible_items = [i for i in all_items if i.reviewer_decision == "approved"]
    elif status_filter == "overridden":
        visible_items = [i for i in all_items if i.reviewer_decision == "overridden"]

    if classification_filter != "all":
        visible_items = [i for i in visible_items if i.pipeline_classification == classification_filter]

    read_only = run.status == "signed_off"
    error = request.query_params.get("error")
    nav = _get_nav_context(db, user)
    return templates.TemplateResponse(request, "review/queue.html", {
        "run": run,
        "run_id": run.id,
        "items": visible_items,
        "user": user,
        "pending": pending,
        "approved": approved,
        "overridden": overridden,
        "total": total,
        "status_filter": status_filter,
        "classification_filter": classification_filter,
        "error": error,
        "read_only": read_only,
        **nav,
    })


def _item_counts(db: Session, run_id: int):
    all_items = db.query(ReviewItem).filter(ReviewItem.run_id == run_id).all()
    pending = sum(1 for i in all_items if i.reviewer_decision is None)
    approved = sum(1 for i in all_items if i.reviewer_decision == "approved")
    overridden = sum(1 for i in all_items if i.reviewer_decision == "overridden")
    return all_items, pending, approved, overridden


@router.post("/{run_id}/items/{char_no}/approve", response_class=HTMLResponse)
def approve_item(
    run_id: int,
    char_no: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404)
    if run.status == "signed_off":
        raise HTTPException(status_code=409, detail="Run is already signed off")
    item = db.query(ReviewItem).filter(
        ReviewItem.run_id == run_id, ReviewItem.char_no == char_no
    ).first()
    if item is None:
        raise HTTPException(status_code=404)
    item.reviewer_decision = "approved"
    item.reviewed_by_id = user.id
    item.reviewed_at = datetime.utcnow()
    db.commit()
    db.refresh(item)
    all_items, pending, approved, overridden = _item_counts(db, run_id)
    return templates.TemplateResponse(request, "review/_item_card.html", {
        "item": item,
        "run": run,
        "run_id": run_id,
        "pending": pending,
        "approved": approved,
        "overridden": overridden,
        "total": len(all_items),
        "oob_update": True,
    })


VALID_CLASSIFICATIONS = {"unchanged", "changed", "removed", "added", "uncertain"}


@router.post("/{run_id}/items/{char_no}/override", response_class=HTMLResponse)
def override_item(
    run_id: int,
    char_no: int,
    request: Request,
    override_note: str = Form(""),
    override_classification: str = Form("changed"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404)
    if run.status == "signed_off":
        raise HTTPException(status_code=409, detail="Run is already signed off")
    item = db.query(ReviewItem).filter(
        ReviewItem.run_id == run_id, ReviewItem.char_no == char_no
    ).first()
    if item is None:
        raise HTTPException(status_code=404)
    if not override_note or not override_note.strip():
        return templates.TemplateResponse(
            request,
            "review/_item_card.html",
            {"item": item, "run_id": run_id, "error": "Override note is required."},
            status_code=422,
        )
    if override_classification not in VALID_CLASSIFICATIONS:
        override_classification = "changed"
    item.reviewer_decision = "overridden"
    item.override_classification = override_classification
    item.override_note = override_note.strip()
    item.reviewed_by_id = user.id
    item.reviewed_at = datetime.utcnow()
    db.commit()
    db.refresh(item)
    all_items, pending, approved, overridden = _item_counts(db, run_id)
    return templates.TemplateResponse(request, "review/_item_card.html", {
        "item": item,
        "run": run,
        "run_id": run_id,
        "pending": pending,
        "approved": approved,
        "overridden": overridden,
        "total": len(all_items),
        "oob_update": True,
    })


@router.post("/{run_id}/sign-off/confirm")
def sign_off_confirm(
    run_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404)
    if run.status == "signed_off":
        return RedirectResponse(f"/runs/{run_id}", status_code=302)
    pending = (
        db.query(ReviewItem)
        .filter(ReviewItem.run_id == run_id, ReviewItem.reviewer_decision == None)  # noqa: E711
        .count()
    )
    if pending > 0:
        return RedirectResponse(f"/review/{run_id}?error=pending_items", status_code=302)
    success = attempt_sign_off(db, run, user.id)
    if success:
        return RedirectResponse(f"/review/{run_id}/generating", status_code=302)
    else:
        return RedirectResponse(f"/review/{run_id}?error=sign_off_failed", status_code=302)


@router.post("/{run_id}/reassign")
def reassign_run(
    run_id: int,
    request: Request,
    reviewer_id: int = Form(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404)
    if run.status == "signed_off":
        raise HTTPException(status_code=409, detail="Cannot reassign a signed-off run")
    new_reviewer = db.query(User).filter(User.id == reviewer_id, User.is_active == True).first()  # noqa: E712
    if new_reviewer is None:
        raise HTTPException(status_code=400, detail="Invalid reviewer")
    run.reviewer_id = reviewer_id
    db.commit()
    return RedirectResponse(f"/runs/{run_id}", status_code=302)


@router.get("/{run_id}/snippets/{filename}")
def serve_snippet(
    run_id: int,
    filename: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    from pathlib import Path as _Path
    if ".." in filename or "/" in filename or not filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None or run.output_dir is None:
        raise HTTPException(status_code=404)
    snippet_path = _Path(run.output_dir) / "snippets" / filename
    if not snippet_path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(str(snippet_path), media_type="image/png")


@router.post("/{run_id}/amend")
def create_amendment_route(
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Create an amendment for a signed-off run. Redirects to the amendment's review queue."""
    from shop.services.amendments import create_amendment
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != "signed_off":
        raise HTTPException(status_code=403, detail="Only signed-off runs can be amended")
    amendment = create_amendment(db, run, user.id)
    return RedirectResponse(f"/review/{amendment.id}", status_code=303)


@router.get("/{run_id}/generating", response_class=HTMLResponse)
def review_generating(
    run_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404)
    # If already terminal, redirect immediately
    if run.status == "signed_off":
        return RedirectResponse(f"/runs/{run_id}", status_code=302)
    if run.status == "reviewing":
        return RedirectResponse(f"/review/{run_id}?error=sign_off_failed", status_code=302)
    nav = _get_nav_context(db, user)
    return templates.TemplateResponse(request, "review/generating.html", {
        "run": run, "user": user, **nav,
    })


@router.get("/{run_id}/sign-off/sse")
async def sign_off_sse(
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    async def generator() -> AsyncIterable[ServerSentEvent]:
        terminal = {"signed_off", "reviewing"}
        run = db.query(Run).filter(Run.id == run_id).first()
        if run is None:
            yield ServerSentEvent(event="close", raw_data="notfound")
            return
        while True:
            db.expire(run)   # bypass SQLAlchemy identity map cache
            db.refresh(run)
            payload = _json.dumps({"status": run.status, "run_id": run_id})
            yield ServerSentEvent(raw_data=payload, event="status_update")
            if run.status in terminal:
                yield ServerSentEvent(event="close", raw_data="done")
                break
            await asyncio.sleep(2.0)
    return EventSourceResponse(generator())
