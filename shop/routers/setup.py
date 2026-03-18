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
    next_step = min(config.wizard_step + 1, 3)
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
        "setup/step2_username.html",
        {"config": config, "current_step": 2},
    )


@router.post("/step2", response_class=HTMLResponse)
def step2_post(
    request: Request,
    admin_username: str = Form(...),
    db: Session = Depends(get_db),
):
    config = _get_or_create_config(db)
    if config.wizard_step < 1:
        return _step_redirect(1)

    admin_username = admin_username.strip()
    if not admin_username:
        return templates.TemplateResponse(
            request,
            "setup/step2_username.html",
            {
                "config": config,
                "current_step": 2,
                "error": "Username cannot be empty",
            },
        )
    # Check uniqueness (edge case: re-running setup on existing DB)
    existing = db.query(User).filter(User.username == admin_username).first()
    if existing and existing.role != "admin":
        return templates.TemplateResponse(
            request,
            "setup/step2_username.html",
            {
                "config": config,
                "current_step": 2,
                "error": f"Username '{admin_username}' is already taken",
            },
        )

    # Store chosen username in session via config temporarily — use a separate
    # ShopConfig field is overkill; pass via hidden form in step3 instead.
    # We store the pending admin username in the session store here by writing it
    # to a transient config field we read in step3.
    config.pending_admin_username = admin_username  # type: ignore[attr-defined]
    # Use wizard_step to track progress; store username in session via cookie is complex.
    # Simpler: redirect to step3 with username as a query param — it's not a secret yet.
    if config.wizard_step < 2:
        config.wizard_step = 2
    db.commit()
    return RedirectResponse(f"/setup/step3?u={admin_username}", status_code=302)


@router.get("/step3", response_class=HTMLResponse)
def step3_get(request: Request, db: Session = Depends(get_db)):
    config = _get_or_create_config(db)
    if config.setup_complete:
        return RedirectResponse("/login", status_code=302)
    if config.wizard_step < 2:
        return _step_redirect(2)
    admin_username = request.query_params.get("u", "")
    return templates.TemplateResponse(
        request,
        "setup/step3_password.html",
        {"config": config, "current_step": 3, "admin_username": admin_username},
    )


@router.post("/step3", response_class=HTMLResponse)
def step3_post(
    request: Request,
    admin_username: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
):
    config = _get_or_create_config(db)
    if config.wizard_step < 2:
        return _step_redirect(2)

    if new_password != confirm_password:
        return templates.TemplateResponse(
            request,
            "setup/step3_password.html",
            {
                "config": config,
                "current_step": 3,
                "admin_username": admin_username,
                "error": "Passwords do not match",
            },
        )
    if new_password == ADMIN_DEFAULT_PASSWORD:
        return templates.TemplateResponse(
            request,
            "setup/step3_password.html",
            {
                "config": config,
                "current_step": 3,
                "admin_username": admin_username,
                "error": "Cannot reuse the default password",
            },
        )

    admin = _get_admin(db)
    if admin:
        admin.username = admin_username
        admin.hashed_password = hash_password(new_password)
        db.commit()
    else:
        new_admin = User(
            username=admin_username,
            hashed_password=hash_password(new_password),
            role="admin",
            is_active=True,
        )
        db.add(new_admin)
        db.commit()
        logger.info("Created admin user '%s' during setup wizard step 3", admin_username)

    if config.wizard_step < 3:
        config.wizard_step = 3
    config.setup_complete = True
    db.commit()
    logger.info("Setup wizard complete: admin '%s' created, setup_complete=True", admin_username)
    return RedirectResponse("/login", status_code=302)


# ── Catch-all redirects for removed / old steps ──────────────────────────────

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
