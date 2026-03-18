import logging
from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from shop.app import templates
from shop.dependencies import get_db, get_current_user
from shop.models import User, Run, RunAlert, ShopConfig
from shop.services.auth import verify_password, create_session, delete_session

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_class=RedirectResponse)
def root():
    return RedirectResponse("/login", status_code=302)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request, "auth/login.html")


@router.post("/login")
def login(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(
            request,
            "auth/login.html",
            {"error": "Invalid username or password"},
            status_code=200,
        )
    if not user.is_active:
        return templates.TemplateResponse(
            request,
            "auth/login.html",
            {"error": "Account is deactivated"},
            status_code=200,
        )
    token = create_session(db, user)
    redirect = RedirectResponse("/dashboard", status_code=302)
    redirect.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,   # False for HTTP; set True only behind HTTPS terminator
        max_age=8 * 3600,
    )
    return redirect


@router.post("/logout")
def logout(
    request: Request,
    db: Session = Depends(get_db),
):
    token = request.cookies.get("session_token")
    if token:
        delete_session(db, token)
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("session_token")
    return response


@router.post("/alerts/dismiss/{alert_id}", response_class=HTMLResponse)
def dismiss_alert(
    alert_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Mark a RunAlert as read and return empty HTML so HTMX swaps the banner out."""
    alert = db.query(RunAlert).filter(RunAlert.id == alert_id).first()
    if alert and alert.user_id == user.id:
        alert.is_read = True
        db.commit()
    return HTMLResponse("")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    recent_runs = db.query(Run).order_by(Run.submitted_at.desc()).limit(10).all()
    unread_alerts = db.query(RunAlert).filter(
        RunAlert.user_id == user.id, RunAlert.is_read == False  # noqa: E712
    ).order_by(RunAlert.created_at.desc()).limit(5).all()
    unread_count = len(unread_alerts)
    config = db.query(ShopConfig).filter(ShopConfig.id == 1).first()
    shop_name = config.shop_name if config else "Delta Preservation"
    return templates.TemplateResponse(request, "dashboard.html", {
        "user": user,
        "recent_runs": recent_runs,
        "unread_alerts": unread_alerts,
        "unread_alert_count": unread_count,
        "shop_name": shop_name,
    })
