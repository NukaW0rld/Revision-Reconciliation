import logging
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from shop.app import templates
from shop.dependencies import get_db
from shop.models import ShopConfig, User
from shop.services.auth import hash_password

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
