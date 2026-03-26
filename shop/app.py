import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from shop.database import Base, engine
from shop.middleware.setup_guard import SetupGuardMiddleware

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

templates = Jinja2Templates(directory="shop/templates")


def _status_badge_class(status: str) -> str:
    return {
        "completed": "success",
        "failed": "error",
        "warning": "warning",
        "running": "info",
        "queued": "ghost",
        "reviewing": "info",
        "signing_off": "warning",
        "signed_off": "success",
    }.get(status, "ghost")


def _semantic_state_tone(state: str | None) -> str:
    return {
        "parsed": "success",
        "error": "warning",
        "empty": "ghost",
        "skipped": "ghost",
        "not_implemented": "ghost",
        "not_requested": "ghost",
    }.get(state or "", "ghost")


def _semantic_detail_lines(contract: dict | None) -> list[str]:
    if not contract:
        return []

    lines: list[str] = []
    summary = contract.get("summary")
    normalized_text = contract.get("normalized_text")
    detail = contract.get("detail")

    if summary:
        lines.append(summary)
    if normalized_text and normalized_text != summary:
        lines.append(f"Normalized: {normalized_text}")
    if detail:
        lines.append(detail)

    return lines


def _classification_badge_class(classification: str) -> str:
    return {
        "unchanged": "bg-base-300 text-base-content/80",
        "changed": "bg-warning/20 text-warning",
        "removed": "bg-error/20 text-error",
        "added": "bg-success/20 text-success",
        "uncertain": "bg-info/20 text-info",
    }.get(classification, "bg-base-300 text-base-content/80")


templates.env.filters["status_badge_class"] = _status_badge_class
templates.env.filters["classification_badge_class"] = _classification_badge_class
templates.env.filters["semantic_state_tone"] = _semantic_state_tone
templates.env.filters["semantic_detail_lines"] = _semantic_detail_lines

import os as _os
templates.env.filters["basename"] = lambda p: _os.path.basename(p) if p else ""


def create_app(session_factory=None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        session_factory: Optional SQLAlchemy session factory for test isolation.
            When provided, SetupGuardMiddleware checks this DB instead of the
            global SessionLocal (shop.db).
    """
    app = FastAPI(title="Delta Preservation")
    # Create all DB tables on startup
    Base.metadata.create_all(bind=engine)
    from shop.database import run_schema_migrations
    run_schema_migrations(engine)
    # Middleware (outermost registered = executes last)
    app.add_middleware(SetupGuardMiddleware, session_factory=session_factory)
    # Static files
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except RuntimeError:
        pass  # static/ dir may not exist in tests
    # Routers
    from shop.routers import auth, admin, setup, runs, review, exports
    app.include_router(auth.router)
    app.include_router(admin.router, prefix="/admin")
    app.include_router(setup.router, prefix="/setup")
    app.include_router(runs.router, prefix="/runs")
    app.include_router(review.router, prefix="/review")
    app.include_router(exports.router, prefix="/exports")
    return app


app = create_app()
