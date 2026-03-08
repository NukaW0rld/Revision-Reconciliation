import logging
from fastapi import APIRouter, Depends, File, Form, Request, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from shop.app import templates
from shop.dependencies import get_db, require_admin
from shop.models import ShopConfig, User
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


@router.get("/settings", response_class=HTMLResponse)
def settings_page(
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    config = db.query(ShopConfig).filter(ShopConfig.id == 1).first()
    return templates.TemplateResponse(
        request,
        "admin/settings.html",
        {"config": config, "current_user": admin},
    )


@router.post("/settings/name")
def settings_update_name(
    request: Request,
    shop_name: str = Form(...),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    config = db.query(ShopConfig).filter(ShopConfig.id == 1).first()
    if config:
        config.shop_name = shop_name.strip()
        db.commit()
    return RedirectResponse("/admin/settings", status_code=302)


@router.post("/settings/upload", response_class=HTMLResponse)
async def settings_upload(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    from shop.services.form3 import parse_excel_preview, REQUIRED_FIELDS

    content = await file.read()
    try:
        headers, preview_rows, detected = parse_excel_preview(content)
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "setup/step4_error.html",
            {"error": str(exc)},
            status_code=200,
        )
    return templates.TemplateResponse(
        request,
        "setup/step4_mapping_partial.html",
        {
            "headers": headers,
            "preview_rows": preview_rows,
            "detected": detected,
            "required_fields": REQUIRED_FIELDS,
            "form_action": "/admin/settings/save",
        },
        status_code=200,
    )


@router.post("/settings/retention")
def settings_update_retention(
    request: Request,
    retention_days: int = Form(...),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    config = db.query(ShopConfig).filter(ShopConfig.id == 1).first()
    if config:
        # Clamp to reasonable range: 1 day minimum, 3650 days (~10 years) maximum
        config.retention_days = max(1, min(3650, retention_days))
        db.commit()
    return RedirectResponse("/admin/settings", status_code=302)


@router.post("/settings/save")
async def settings_save(
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    from shop.services.form3 import REQUIRED_FIELDS

    form_data = await request.form()
    mapping: dict[str, int] = {}
    for key, val in form_data.multi_items():
        if key.startswith("col_") and val and val != "ignore" and val in REQUIRED_FIELDS:
            try:
                mapping[val] = int(key[4:])
            except ValueError:
                pass
    config = db.query(ShopConfig).filter(ShopConfig.id == 1).first()
    if config and mapping:
        config.column_mapping = mapping
        db.commit()
    return RedirectResponse("/admin/settings", status_code=302)
