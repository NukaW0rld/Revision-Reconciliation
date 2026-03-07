from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from shop.app import templates
from shop.dependencies import get_db, get_current_user
from shop.models import User, Run, ReviewItem
from shop.services.review import open_review_queue
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
    # Only allow review of completed/warning/reviewing/signing_off runs
    if run.status not in ("completed", "warning", "reviewing", "signing_off"):
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

    error = request.query_params.get("error")
    nav = _get_nav_context(db, user)
    return templates.TemplateResponse(request, "review/queue.html", {
        "run": run,
        "items": visible_items,
        "user": user,
        "pending": pending,
        "approved": approved,
        "overridden": overridden,
        "total": total,
        "status_filter": status_filter,
        "classification_filter": classification_filter,
        "error": error,
        **nav,
    })
