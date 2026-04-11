import asyncio
import json as _json
from collections.abc import AsyncIterable
from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from sqlalchemy.orm import Session
from shop.utils import utcnow
from shop.app import templates
from shop.dependencies import get_db, get_current_user
from shop.models import User, Run, ReviewItem
from shop.services.alternate_history import sync_accepted_alternate_history
from shop.services.review import (
    DebugVerdictValidationError,
    assemble_debug_report_payload,
    attempt_sign_off,
    build_debug_queue_state,
    build_run_debug_summary,
    debug_internals_by_char,
    debug_internals_by_item_id,
    debug_verdict_state,
    load_debug_notes,
    load_debug_verdicts_for_render,
    open_review_queue,
    save_debug_notes,
    save_debug_verdict,
    semantic_contracts_by_char,
    semantic_contracts_by_item_id,
    validate_debug_verdict_payload,
)
from shop.routers.runs import _get_nav_context

router = APIRouter(redirect_slashes=False)


def _debug_queue_progress(run: Run, items: list[ReviewItem]) -> dict:
    debug_verdicts = load_debug_verdicts_for_render(run)
    return debug_verdict_state(items, debug_verdicts)


@router.get("/{run_id}", response_class=HTMLResponse)
def review_queue(
    run_id: int,
    request: Request,
    status_filter: str = "all",
    classification_filter: str = "all",
    debug: bool = False,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404)
    # Only allow review of completed/warning/reviewing/signing_off/signed_off runs
    if run.status not in ("completed", "warning", "reviewing", "signing_off", "signed_off"):
        return RedirectResponse(f"/runs/{run_id}", status_code=302)

    # Debug mode: only admins may access
    if debug and user.role != "admin":
        return RedirectResponse("/dashboard", status_code=302)

    debug_queue_state = build_debug_queue_state(db, run) if debug else None
    all_items = debug_queue_state["all_items"] if debug_queue_state else open_review_queue(db, run)
    exception_items = debug_queue_state["exception_items"] if debug_queue_state else []
    if debug and not exception_items and (
        all_items or debug_queue_state.get("packet_declares_items")
    ):
        return RedirectResponse(f"/runs/{run_id}", status_code=302)
    semantic_contracts = semantic_contracts_by_char(run)
    debug_semantic_contracts = semantic_contracts_by_item_id(db, run) if debug else {}
    debug_internals = debug_internals_by_item_id(db, run) if debug else {}
    debug_progress = (
        _debug_queue_progress(run, exception_items)
        if debug
        else {"by_item_id": {}, "submitted": 0, "total": len(all_items)}
    )
    debug_notes = load_debug_notes(run) if debug else ""

    # Counts (unfiltered — sign-off gate counts all regardless of filter)
    pending = sum(1 for i in all_items if i.reviewer_decision is None)
    approved = sum(1 for i in all_items if i.reviewer_decision == "approved")
    overridden = sum(1 for i in all_items if i.reviewer_decision == "overridden")
    total = len(all_items)

    # Apply filters
    visible_items = exception_items if debug else all_items
    if status_filter == "pending":
        visible_items = [i for i in visible_items if i.reviewer_decision is None]
    elif status_filter == "approved":
        visible_items = [i for i in visible_items if i.reviewer_decision == "approved"]
    elif status_filter == "overridden":
        visible_items = [i for i in visible_items if i.reviewer_decision == "overridden"]

    if classification_filter != "all":
        visible_items = [i for i in visible_items if i.pipeline_classification == classification_filter]

    read_only = run.status == "signed_off"
    is_amendment = bool(run.parent_run_id)
    error = request.query_params.get("error")
    nav = _get_nav_context(db, user)
    return templates.TemplateResponse(request, "review/queue.html", {
        "run": run,
        "run_id": run.id,
        "items": visible_items,
        "semantic_contracts": semantic_contracts,
        "debug_semantic_contracts": debug_semantic_contracts,
        "user": user,
        "pending": pending,
        "approved": approved,
        "overridden": overridden,
        "total": total,
        "status_filter": status_filter,
        "classification_filter": classification_filter,
        "error": error,
        "read_only": read_only,
        "is_amendment": is_amendment,
        "debug_mode": debug,
        "debug_internals": debug_internals,
        "debug_verdicts": debug_progress["by_item_id"],
        "debug_submitted": debug_progress["submitted"],
        "debug_total": debug_progress["total"],
        "debug_notes": debug_notes,
        **nav,
    })


def _item_counts(db: Session, run_id: int):
    all_items = db.query(ReviewItem).filter(ReviewItem.run_id == run_id).all()
    pending = sum(1 for i in all_items if i.reviewer_decision is None)
    approved = sum(1 for i in all_items if i.reviewer_decision == "approved")
    overridden = sum(1 for i in all_items if i.reviewer_decision == "overridden")
    return all_items, pending, approved, overridden


def _render_debug_item_card(
    *,
    request: Request,
    run: Run,
    item: ReviewItem,
    db: Session,
    user: User,
    error: str | None = None,
    status_code: int = 200,
    form_values: dict | None = None,
    oob_update: bool = True,
):
    all_items, pending, approved, overridden = _item_counts(db, run.id)
    debug_queue_state = build_debug_queue_state(db, run, activate_review=False)
    debug_progress = _debug_queue_progress(run, debug_queue_state["exception_items"])
    semantic_contract = semantic_contracts_by_item_id(db, run).get(item.id)
    debug_internal = debug_internals_by_item_id(db, run).get(item.id)
    saved_verdict = debug_progress["by_item_id"].get(item.id)
    debug_form = {**(saved_verdict or {}), **(form_values or {})}
    return templates.TemplateResponse(
        request,
        "review/_item_card_debug.html",
        {
            "item": item,
            "run": run,
            "run_id": run.id,
            "user": user,
            "semantic_contract": semantic_contract,
            "debug_internal": debug_internal,
            "debug_verdict": saved_verdict,
            "debug_form": debug_form,
            "debug_submitted": debug_progress["submitted"],
            "debug_total": debug_progress["total"],
            "pending": pending,
            "approved": approved,
            "overridden": overridden,
            "total": len(all_items),
            "read_only": run.status == "signed_off",
            "is_amendment": bool(run.parent_run_id),
            "error": error,
            "debug_mode": True,
            "oob_update": oob_update,
        },
        status_code=status_code,
    )


@router.post("/{run_id}/debug/items/{item_id}/verdict", response_class=HTMLResponse)
def save_debug_item_verdict(
    run_id: int,
    item_id: int,
    request: Request,
    verdict: str = Form(""),
    corrected_classification: str = Form(""),
    corrected_requirement_revA: str = Form(""),
    corrected_requirement_revB: str = Form(""),
    explanation: str = Form(""),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404)

    item = db.query(ReviewItem).filter(ReviewItem.id == item_id).first()
    if item is None or item.run_id != run_id:
        raise HTTPException(status_code=404)

    try:
        payload = validate_debug_verdict_payload(
            verdict=verdict,
            corrected_classification=corrected_classification,
            corrected_requirement_revA=corrected_requirement_revA,
            corrected_requirement_revB=corrected_requirement_revB,
            explanation=explanation,
        )
    except DebugVerdictValidationError as exc:
        return _render_debug_item_card(
            request=request,
            run=run,
            item=item,
            db=db,
            user=user,
            error=str(exc),
            status_code=422,
            form_values={
                "verdict": verdict.strip(),
                "corrected_classification": corrected_classification.strip(),
                "corrected_requirement_revA": corrected_requirement_revA,
                "corrected_requirement_revB": corrected_requirement_revB,
                "explanation": explanation,
            },
        )

    save_debug_verdict(run, item, payload)
    sync_accepted_alternate_history(db, run, item, payload)
    db.refresh(item)
    return _render_debug_item_card(request=request, run=run, item=item, db=db, user=user)


@router.get("/{run_id}/debug-report.json")
def download_debug_report(
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    summary = build_run_debug_summary(db, run)
    if not summary["debug_report_ready"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "Debug export is incomplete: "
                f"{summary['resolved_exception_count']} of {len(summary['exception_rows'])} "
                "exception rows are resolved."
            ),
        )

    try:
        payload = assemble_debug_report_payload(db, run)
    except DebugVerdictValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=_json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="debug_report.json"'},
    )


@router.post("/{run_id}/debug/notes", response_class=HTMLResponse)
def save_debug_notes_route(
    run_id: int,
    request: Request,
    notes: str = Form(""),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404)

    try:
        save_debug_notes(run, notes.strip())
    except Exception:
        return HTMLResponse(
            '<span id="debug-notes-status" class="text-error text-xs font-mono">Save failed.</span>',
            status_code=500,
        )

    return HTMLResponse(
        '<span id="debug-notes-status" class="text-success text-xs font-mono">Saved.</span>'
    )


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
    item.reviewed_at = utcnow()
    semantic_contract = semantic_contracts_by_char(run).get(item.char_no)
    db.commit()
    db.refresh(item)
    all_items, pending, approved, overridden = _item_counts(db, run_id)
    return templates.TemplateResponse(request, "review/_item_card.html", {
        "item": item,
        "run": run,
        "run_id": run_id,
        "semantic_contract": semantic_contract,
        "pending": pending,
        "approved": approved,
        "overridden": overridden,
        "total": len(all_items),
        "oob_update": True,
        "is_amendment": bool(run.parent_run_id),
    })


@router.post("/{run_id}/items/{char_no}/reset", response_class=HTMLResponse)
def reset_item(
    run_id: int,
    char_no: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Reset a previously-decided review item back to pending (undecided) state.

    Only valid before sign-off. Clears reviewer_decision, override_classification,
    override_note, reviewed_by_id, and reviewed_at so the item can be re-reviewed.
    Returns the updated item card HTML with OOB progress bar and signoff footer.
    """
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
    item.reviewer_decision = None
    item.override_classification = None
    item.override_note = None
    item.reviewed_by_id = None
    item.reviewed_at = None
    semantic_contract = semantic_contracts_by_char(run).get(item.char_no)
    db.commit()
    db.refresh(item)
    all_items, pending, approved, overridden = _item_counts(db, run_id)
    return templates.TemplateResponse(request, "review/_item_card.html", {
        "item": item,
        "run": run,
        "run_id": run_id,
        "semantic_contract": semantic_contract,
        "pending": pending,
        "approved": approved,
        "overridden": overridden,
        "total": len(all_items),
        "oob_update": True,
        "is_amendment": bool(run.parent_run_id),
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
    semantic_contract = semantic_contracts_by_char(run).get(char_no)
    if item is None:
        raise HTTPException(status_code=404)
    if not override_note or not override_note.strip():
        return templates.TemplateResponse(
            request,
            "review/_item_card.html",
            {
                "item": item,
                "run": run,
                "run_id": run_id,
                "semantic_contract": semantic_contract,
                "error": "Override note is required.",
                "is_amendment": bool(run.parent_run_id),
            },
            status_code=422,
        )
    if override_classification not in VALID_CLASSIFICATIONS:
        override_classification = "changed"
    item.reviewer_decision = "overridden"
    item.override_classification = override_classification
    item.override_note = override_note.strip()
    item.reviewed_by_id = user.id
    item.reviewed_at = utcnow()
    db.commit()
    db.refresh(item)
    all_items, pending, approved, overridden = _item_counts(db, run_id)
    return templates.TemplateResponse(request, "review/_item_card.html", {
        "item": item,
        "run": run,
        "run_id": run_id,
        "semantic_contract": semantic_contract,
        "pending": pending,
        "approved": approved,
        "overridden": overridden,
        "total": len(all_items),
        "oob_update": True,
        "is_amendment": bool(run.parent_run_id),
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
