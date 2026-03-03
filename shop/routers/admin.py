import logging
from fastapi import APIRouter, Depends, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from shop.app import templates
from shop.dependencies import get_db, require_admin
from shop.models import User
from shop.services.auth import hash_password

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/users", response_class=HTMLResponse)
def list_users(
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    users = db.query(User).order_by(User.created_at).all()
    return templates.TemplateResponse(
        request,
        "admin/users.html",
        {"users": users, "current_user": admin},
    )


@router.post("/users", response_class=HTMLResponse)
def create_user(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(default="engineer"),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    new_user = User(
        email=email,
        hashed_password=hash_password(password),
        role=role,
        is_active=True,
    )
    db.add(new_user)
    try:
        db.commit()
        db.refresh(new_user)
    except IntegrityError:
        db.rollback()
        return templates.TemplateResponse(
            request,
            "admin/users_row.html",
            {"error": f"Email '{email}' already exists"},
            status_code=200,
        )
    return templates.TemplateResponse(
        request,
        "admin/users_row.html",
        {"user": new_user},
        status_code=200,
    )


@router.post("/users/{user_id}/deactivate", response_class=HTMLResponse)
def deactivate_user(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = False
    db.commit()
    return templates.TemplateResponse(
        request,
        "admin/users_row.html",
        {"user": user},
        status_code=200,
    )
