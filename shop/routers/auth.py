import logging
from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from shop.app import templates
from shop.dependencies import get_db, get_current_user
from shop.models import User
from shop.services.auth import verify_password, create_session, delete_session

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request, "auth/login.html")


@router.post("/login")
def login(
    request: Request,
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(
            request,
            "auth/login.html",
            {"error": "Invalid email or password"},
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


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    user: User = Depends(get_current_user),
):
    return templates.TemplateResponse(request, "dashboard.html", {"user": user})
