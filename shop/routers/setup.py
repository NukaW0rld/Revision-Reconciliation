import logging
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
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
    next_step = min(config.wizard_step + 1, 2)
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
    admin = _get_admin(db)
    admin_email = admin.email if admin else "admin@shop.local"
    return templates.TemplateResponse(
        request,
        "setup/step2_password.html",
        {"config": config, "current_step": 2, "admin_email": admin_email},
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
    admin = _get_admin(db)
    admin_email = admin.email if admin else "admin@shop.local"
    if new_password != confirm_password:
        return templates.TemplateResponse(
            request,
            "setup/step2_password.html",
            {
                "config": config,
                "current_step": 2,
                "admin_email": admin_email,
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
                "admin_email": admin_email,
                "error": "Cannot reuse the default password",
            },
        )
    if admin:
        admin.hashed_password = hash_password(new_password)
        db.commit()
    else:
        # No admin seeded yet (supervisord starts uvicorn directly without seed_admin).
        # Create the admin user now with the chosen password.
        admin_email_env = "admin@shop.local"
        new_admin = User(
            email=admin_email_env,
            hashed_password=hash_password(new_password),
            role="admin",
            is_active=True,
        )
        db.add(new_admin)
        db.commit()
        logger.info("Created admin user %s during setup wizard step 2", admin_email_env)
    if config.wizard_step < 2:
        config.wizard_step = 2
    config.setup_complete = True
    db.commit()
    logger.info("Setup wizard complete: admin password set, setup_complete=True")
    return RedirectResponse("/login", status_code=302)


@router.get("/step3")
def step3_redirect(db: Session = Depends(get_db)):
    config = _get_or_create_config(db)
    if config.setup_complete:
        return RedirectResponse("/login", status_code=302)
    return RedirectResponse("/setup/", status_code=302)


@router.post("/step3")
def step3_post_redirect(db: Session = Depends(get_db)):
    config = _get_or_create_config(db)
    if config.setup_complete:
        return RedirectResponse("/login", status_code=302)
    return RedirectResponse("/setup/", status_code=302)


@router.get("/step4/{path:path}")
def step4_redirect(path: str, db: Session = Depends(get_db)):
    config = _get_or_create_config(db)
    if config.setup_complete:
        return RedirectResponse("/login", status_code=302)
    return RedirectResponse("/setup/", status_code=302)


@router.post("/step4/{path:path}")
async def step4_post_redirect(path: str, db: Session = Depends(get_db)):
    config = _get_or_create_config(db)
    if config.setup_complete:
        return RedirectResponse("/login", status_code=302)
    return RedirectResponse("/setup/", status_code=302)
