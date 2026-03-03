import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from shop.database import SessionLocal
from shop.models import ShopConfig

logger = logging.getLogger(__name__)

# Paths that bypass the setup wizard guard entirely
SETUP_EXEMPT_PATHS = {"/setup", "/login", "/logout", "/static", "/favicon.ico"}


def _is_setup_complete(session_factory=None) -> bool:
    """Check shop_config table for setup_complete flag.

    Returns False if no row exists (treat as setup incomplete).
    Accepts an optional session_factory for test isolation.
    """
    factory = session_factory if session_factory is not None else SessionLocal
    db = factory()
    try:
        config = db.query(ShopConfig).filter(ShopConfig.id == 1).first()
        if config is None:
            return False
        return bool(config.setup_complete)
    except Exception:
        logger.exception("Error checking setup_complete in middleware")
        return False
    finally:
        db.close()


class SetupGuardMiddleware(BaseHTTPMiddleware):
    """Redirect all requests to /setup/step1 until setup_complete is True in shop_config.

    Handles HTMX requests differently from plain browser navigations:
    - HTMX: return HTTP 200 with HX-Redirect header (302 would cause partial DOM swap)
    - Browser: return 302 RedirectResponse

    Args:
        app: The ASGI application.
        session_factory: Optional SQLAlchemy session factory. When provided (e.g. in
            tests), the middleware checks the provided DB instead of the global
            SessionLocal (which connects to shop.db).
    """

    def __init__(self, app, session_factory=None):
        super().__init__(app)
        self._session_factory = session_factory

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow exempt paths (setup pages, static files, auth endpoints)
        if any(path.startswith(exempt) for exempt in SETUP_EXEMPT_PATHS):
            return await call_next(request)

        # Check if setup is complete (pass factory for test isolation)
        if not _is_setup_complete(self._session_factory):
            redirect_url = "/setup/step1"
            is_htmx = request.headers.get("HX-Request") == "true"
            if is_htmx:
                # HTMX requires 200 + HX-Redirect header; 302 causes partial swap
                return Response(
                    content="",
                    status_code=200,
                    headers={"HX-Redirect": redirect_url},
                )
            return RedirectResponse(url=redirect_url, status_code=302)

        return await call_next(request)
