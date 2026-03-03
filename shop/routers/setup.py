import logging
from io import BytesIO
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from shop.app import templates
from shop.dependencies import get_db
from shop.models import ShopConfig, User
from shop.services.auth import hash_password
from shop.services.form3 import parse_excel_preview, REQUIRED_FIELDS

router = APIRouter()
logger = logging.getLogger(__name__)

ADMIN_DEFAULT_PASSWORD = "changeme"  # Matches ADMIN_DEFAULT_PASSWORD env var default


def _get_or_create_config(db: Session) -> ShopConfig:
    """Get singleton ShopConfig row, creating it if missing."""
    config = db.query(ShopConfig).filter(ShopConfig.id == 1).first()
    if not config:
        config = ShopConfig(id=1)
        db.add(config)
        db.commit()
        db.refresh(config)
    return config


def _get_admin(db: Session) -> User | None:
    return db.query(User).filter(User.role == "admin").first()


def _step_redirect(step: int) -> RedirectResponse:
    return RedirectResponse(f"/setup/step{step}", status_code=302)


@router.get("/", response_class=HTMLResponse)
def setup_root(db: Session = Depends(get_db)):
    config = _get_or_create_config(db)
    next_step = min(config.wizard_step + 1, 4)
    return RedirectResponse(f"/setup/step{next_step}", status_code=302)


@router.get("/step1", response_class=HTMLResponse)
def step1_get(request: Request, db: Session = Depends(get_db)):
    config = _get_or_create_config(db)
    return templates.TemplateResponse(
        request,
        "setup/step1_shop_name.html",
        {"config": config, "current_step": 1},
    )


@router.post("/step1")
def step1_post(
    shop_name: str = Form(...),
    db: Session = Depends(get_db),
):
    config = _get_or_create_config(db)
    config.shop_name = shop_name.strip()
    if config.wizard_step < 1:
        config.wizard_step = 1
    db.commit()
    return _step_redirect(2)


@router.get("/step2", response_class=HTMLResponse)
def step2_get(request: Request, db: Session = Depends(get_db)):
    config = _get_or_create_config(db)
    if config.wizard_step < 1:
        return _step_redirect(1)
    return templates.TemplateResponse(
        request,
        "setup/step2_password.html",
        {"config": config, "current_step": 2},
    )


@router.post("/step2", response_class=HTMLResponse)
def step2_post(
    request: Request,
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
):
    config = _get_or_create_config(db)
    if config.wizard_step < 1:
        return _step_redirect(1)
    if new_password != confirm_password:
        return templates.TemplateResponse(
            request,
            "setup/step2_password.html",
            {
                "config": config,
                "current_step": 2,
                "error": "Passwords do not match",
            },
        )
    if new_password == ADMIN_DEFAULT_PASSWORD:
        return templates.TemplateResponse(
            request,
            "setup/step2_password.html",
            {
                "config": config,
                "current_step": 2,
                "error": "Cannot reuse the default password",
            },
        )
    admin = _get_admin(db)
    if admin:
        admin.hashed_password = hash_password(new_password)
        db.commit()
    if config.wizard_step < 2:
        config.wizard_step = 2
        db.commit()
    return _step_redirect(3)


@router.get("/step3", response_class=HTMLResponse)
def step3_get(request: Request, db: Session = Depends(get_db)):
    config = _get_or_create_config(db)
    if config.wizard_step < 2:
        return _step_redirect(config.wizard_step + 1)
    return templates.TemplateResponse(
        request,
        "setup/step3_engineer.html",
        {"config": config, "current_step": 3},
    )


@router.post("/step3", response_class=HTMLResponse)
def step3_post(
    request: Request,
    engineer_email: str = Form(...),
    engineer_password: str = Form(...),
    db: Session = Depends(get_db),
):
    config = _get_or_create_config(db)
    if config.wizard_step < 2:
        return _step_redirect(config.wizard_step + 1)
    new_engineer = User(
        email=engineer_email,
        hashed_password=hash_password(engineer_password),
        role="engineer",
        is_active=True,
    )
    db.add(new_engineer)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return templates.TemplateResponse(
            request,
            "setup/step3_engineer.html",
            {
                "config": config,
                "current_step": 3,
                "error": f"Email '{engineer_email}' already exists",
            },
        )
    if config.wizard_step < 3:
        config.wizard_step = 3
        db.commit()
    return _step_redirect(4)


@router.get("/step4", response_class=HTMLResponse)
def step4_get(request: Request, db: Session = Depends(get_db)):
    config = _get_or_create_config(db)
    if config.wizard_step < 3:
        return _step_redirect(config.wizard_step + 1)
    return templates.TemplateResponse(
        request,
        "setup/step4_column_mapping.html",
        {"config": config, "current_step": 4},
    )


@router.post("/step4/upload", response_class=HTMLResponse)
async def step4_upload(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
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
            "form_action": "/setup/step4/save",
        },
        status_code=200,
    )


@router.post("/step4/save")
async def step4_save(
    request: Request,
    db: Session = Depends(get_db),
):
    form_data = await request.form()
    mapping: dict[str, int] = {}
    for field in REQUIRED_FIELDS:
        val = form_data.get(field)
        if val and val != "ignore":
            try:
                mapping[field] = int(val)
            except ValueError:
                pass
    # Validate minimum required fields
    if "char_no" not in mapping or "requirement" not in mapping:
        config = _get_or_create_config(db)
        return templates.TemplateResponse(
            request,
            "setup/step4_column_mapping.html",
            {
                "config": config,
                "current_step": 4,
                "error": "char_no and requirement columns must be mapped before saving.",
            },
        )
    config = _get_or_create_config(db)
    config.column_mapping = mapping
    config.wizard_step = 4
    config.setup_complete = True
    db.commit()
    return RedirectResponse("/dashboard", status_code=302)
