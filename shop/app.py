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


templates.env.filters["status_badge_class"] = _status_badge_class

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
    # Middleware (outermost registered = executes last)
    app.add_middleware(SetupGuardMiddleware, session_factory=session_factory)
    # Static files
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except RuntimeError:
        pass  # static/ dir may not exist in tests
    # Routers
    from shop.routers import auth, admin, setup, runs, review
    app.include_router(auth.router)
    app.include_router(admin.router, prefix="/admin")
    app.include_router(setup.router, prefix="/setup")
    app.include_router(runs.router, prefix="/runs")
    app.include_router(review.router, prefix="/review")
    return app


app = create_app()
