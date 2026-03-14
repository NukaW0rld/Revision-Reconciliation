# Phase 1: Foundation - Research

**Researched:** 2026-03-02
**Domain:** FastAPI web layer, HTMX, SQLite auth/sessions, Docker multi-stage build
**Confidence:** HIGH (core stack locked by user decisions; verified via official docs and PyPI)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Frontend stack:**
- Jinja2 + HTMX: FastAPI renders HTML templates; HTMX handles partial page swaps without full reloads
- No separate JS build toolchain at runtime
- Tailwind CSS built at Docker image build time via Tailwind CLI (`npx tailwindcss`); Node.js only in the build stage, not at runtime
- DaisyUI as the Tailwind component library (`btn`, `card`, `badge`, `modal`) for pre-built semantic classes
- Server-Sent Events (SSE) via HTMX `sse` extension for live pipeline stage progress updates in Phase 2 (`hx-ext="sse"` + FastAPI `EventSourceResponse`)

**Auth session model:**
- Server-side sessions: session token in an HttpOnly cookie; session data in a `sessions` SQLite table (`id`, `user_id`, `created_at`, `expires_at`)
- 8-hour sliding window: each request resets the expiry timer; session expires after 8 hours of inactivity
- Engineer accessing an admin-only route → silent redirect to engineer dashboard (no 403 page, no error message)
- Passwords hashed with bcrypt (cost factor 12); no complexity rules enforced — admin creates all accounts, no self-registration

**Setup wizard flow:**
- FastAPI middleware intercepts every request and redirects to `/setup/*` until a `setup_complete` flag is true in the DB
- Linear step-by-step: 4 screens in order — (1) shop name, (2) admin password change, (3) first engineer account creation, (4) Form 3 column mapping
- Can go back to a completed step but cannot skip forward
- Mid-wizard resume: each completed step persisted to DB immediately; next visit resumes from first incomplete step
- Default admin credentials seeded via docker-compose.yml env vars (`ADMIN_EMAIL`, `ADMIN_DEFAULT_PASSWORD`); step 2 forces a password change and blocks the default password from being reused

**Form 3 column mapping UI:**
- Table preview: show first 5 rows of the uploaded Excel sheet
- Dropdowns above each column to assign: `char_no` / `requirement` / `reference_location` / `(ignore)`
- Auto-detect (existing `xlsx.py` keyword matching) pre-fills dropdowns; unmatched columns highlighted in amber
- Fatal errors caught before mapping UI; column-level issues surface in the mapping UI itself
- Column mapping re-configurable after setup via admin settings panel; changes apply to new runs only
- Shop-wide config persisted in a `shop_config` SQLite table (not a file on disk)

### Claude's Discretion
- Exact DaisyUI theme/color palette
- SQLite schema details (indexes, migration approach)
- FastAPI app structure (routers, dependency injection patterns)
- Specific Jinja2 template organization

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| AUTH-01 | User can create an account with email and password (admin-initiated; no self-registration) | SQLAlchemy `users` table + bcrypt via pwdlib; admin-only POST endpoint |
| AUTH-02 | User can log in and maintain a session across browser refresh | Server-side `sessions` table + HttpOnly cookie; sliding 8-hour expiry; session middleware |
| AUTH-03 | User can log out from any page | DELETE session row + expire cookie; available on any page via HTMX or standard form |
| AUTH-04 | Admin can create, view, and deactivate engineer accounts | Admin router with RBAC dependency; `is_active` flag in `users` table |
| AUTH-05 | Admin role can access shop configuration; engineer role cannot | RBAC via FastAPI `Depends()` on role field; silent redirect for unauthorized access |
| AUTH-06 | Reviewer name and timestamp permanently attached to every approve, override, sign-off | `reviewed_by` + `reviewed_at` columns on action tables; set from session at write time |
| SETUP-01 | First admin login triggers undismissable setup wizard | FastAPI middleware checks `setup_complete` in `shop_config` table; redirects all requests to `/setup/` |
| SETUP-02 | Admin can upload Form 3 Excel and map columns via UI | `UploadFile` + openpyxl BytesIO; auto-detect calls existing `xlsx.py`; HTMX partial renders mapping table |
| SETUP-03 | Fatal upload errors before mapping UI; minor issues in mapping UI | Two-phase validation: file-level before render, column-level as inline feedback |
| SETUP-04 | System accepts non-contiguous or non-standard char numbering | Existing `xlsx.py` parses char_no as-is; no normalization; store raw values |
| DEPLOY-01 | Application runs via `docker compose up` from single `docker-compose.yml` | Multi-stage Dockerfile: Node stage for Tailwind, Python stage with uv; docker-compose.yml with env vars |
| DEPLOY-02 | All pipeline and web dependencies bundled in Docker image | `uv sync --locked` installs all deps; no pip install at runtime |
| DEPLOY-03 | Functions in air-gapped shop network with no outbound internet | No external calls at runtime; all assets served from container; no CDN links |
</phase_requirements>

---

## Summary

Phase 1 builds the web application shell around the existing `delta_preservation/` computation core. The stack is fully locked: FastAPI with Jinja2 templates, HTMX for partial page updates, DaisyUI + Tailwind CSS compiled at build time, server-side sessions via SQLite, pwdlib for password hashing, and SQLAlchemy 2.0 for ORM. No new technology decisions are required — research confirms all locked choices are sound and currently maintained.

The most important implementation insight is the **HTMX redirect pattern**: because HTMX intercepts 302 responses and performs a partial swap instead of a full page navigation, redirects from middleware must return HTTP 200 with an `HX-Redirect` response header for HTMX requests, or return a plain 302 for direct browser navigations. The setup wizard middleware needs to distinguish these two request types. The second key insight is **DaisyUI version**: the current release is v5 (not v4), which has breaking changes to component class names and the color system compared to v4. Planner must target DaisyUI v5 with Tailwind CSS v4.

The `pyproject.toml` already includes `pytest>=8,<9` but there is no `tests/` directory. Wave 0 of Phase 1 must create the test infrastructure from scratch, including a `conftest.py` with an in-memory SQLite engine override and FastAPI `TestClient` fixture.

**Primary recommendation:** Build the web layer as a `shop/` package at the project root, separate from `delta_preservation/`, with FastAPI routers organized by domain (auth, setup, admin). Use synchronous SQLAlchemy with SQLite for simplicity (no async complexity for a single-shop deployment).

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| fastapi | 0.135.x | Web framework, routing, DI, SSE | Locked decision; async-capable, Pydantic-native, HTMX-friendly |
| uvicorn | latest | ASGI server | Standard FastAPI runner; supports `--workers` flag |
| jinja2 | 3.x | HTML templating | FastAPI's built-in template engine via `Jinja2Templates` |
| python-multipart | 0.0.x | Form + file upload parsing | Required by FastAPI for `UploadFile` and `Form()` fields |
| sqlalchemy | 2.0.41 | ORM, schema, session management | Locked decision; 2.0 API with type annotations |
| pwdlib[bcrypt] | 0.3.0 | Password hashing | Locked decision; modern passlib replacement |
| bcrypt | 4.3.0 | Bcrypt backend for pwdlib | Must pin to <5.0.0; bcrypt 5.0.0 removed `__about__` attr, breaking pwdlib |
| itsdangerous | 2.x | Session token signing | Secure HMAC signing for session IDs stored in cookies |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| openpyxl | 3.x | Excel file parsing | Already in project; used for Form 3 upload in SETUP-02 |
| python-dotenv | 1.x | Load `.env` for dev | Docker passes env vars; dotenv for local dev only |
| httpx | 0.x | Async test client | pytest-compatible async FastAPI testing |

### Frontend (build-time only)

| Tool | Version | Purpose | Notes |
|------|---------|---------|-------|
| tailwindcss | 4.x | Utility CSS | Compiled in Docker Node stage; not in Python deps |
| daisyui | 5.5.x | Component classes | Plugin in Tailwind config; MUST use v5, not v4 |
| htmx | 2.x | Partial page swaps | Served as a static file from `/static/`; no CDN |
| htmx/sse extension | 2.x | SSE swap attribute support | Bundled with HTMX; enables `hx-ext="sse"` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| SQLAlchemy ORM | Raw SQLite via `sqlite3` | ORM chosen; cleaner schema evolution, Pydantic integration |
| pwdlib[bcrypt] | argon2 | Argon2 is stronger; bcrypt is fine for this threat model and already chosen |
| server-side sessions | JWT tokens | JWTs cannot be revoked server-side; sessions allow forced logout |
| Tailwind CLI (npm) | Tailwind standalone binary | CLI requires Node but is standard; standalone binary is v3 only |

**Installation (Python):**
```bash
uv add fastapi uvicorn jinja2 python-multipart sqlalchemy "pwdlib[bcrypt]" "bcrypt<5.0.0" itsdangerous
```

**Installation (Tailwind + DaisyUI — Docker build stage only):**
```bash
npm install -D tailwindcss@latest daisyui@latest
npx tailwindcss -i ./static/src/input.css -o ./static/dist/output.css --minify
```

---

## Architecture Patterns

### Recommended Project Structure

```
shop/                        # New web layer package (beside delta_preservation/)
├── __init__.py
├── app.py                   # FastAPI app factory, middleware registration
├── database.py              # SQLAlchemy engine, SessionLocal, Base
├── models.py                # ORM models: User, Session, ShopConfig, WizardProgress
├── dependencies.py          # get_db(), get_current_user(), require_admin()
├── middleware/
│   └── setup_guard.py       # Wizard redirect middleware
├── routers/
│   ├── auth.py              # /login, /logout
│   ├── setup.py             # /setup/* wizard steps
│   └── admin.py             # /admin/* user management, settings
├── services/
│   ├── auth.py              # hash_password(), verify_password(), create_session()
│   └── form3.py             # wrap xlsx.py auto-detect for web layer
└── templates/
    ├── base.html             # Base layout with HTMX script, Tailwind output CSS
    ├── auth/
    │   └── login.html
    ├── setup/
    │   ├── step1_shop_name.html
    │   ├── step2_password.html
    │   ├── step3_engineer.html
    │   └── step4_column_mapping.html
    └── admin/
        ├── users.html
        └── settings.html
static/
├── src/
│   └── input.css            # Tailwind CSS entry point with @import and @plugin
├── dist/
│   └── output.css           # Compiled Tailwind output (committed or build-time artifact)
└── js/
    ├── htmx.min.js          # HTMX 2.x bundled (no CDN at runtime)
    └── htmx-sse.js          # SSE extension bundled
docker/
├── Dockerfile
└── docker-compose.yml
```

### Pattern 1: FastAPI App Factory with Middleware

**What:** Create the `FastAPI` instance in a factory function; register middleware and routers in one place.

**When to use:** Always — allows testing to create a fresh app instance without global state.

```python
# Source: https://fastapi.tiangolo.com/tutorial/middleware/
# shop/app.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from shop.middleware.setup_guard import SetupGuardMiddleware
from shop.routers import auth, setup, admin

def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(SetupGuardMiddleware)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.include_router(auth.router)
    app.include_router(setup.router, prefix="/setup")
    app.include_router(admin.router, prefix="/admin")
    return app

app = create_app()
```

### Pattern 2: Setup Wizard Middleware (redirect guard)

**What:** Middleware that intercepts every request, checks `setup_complete` in `shop_config` table, and redirects to `/setup/` until setup is done. Must handle HTMX requests differently from browser navigations.

**Critical finding:** HTMX intercepts 302 responses and performs a partial DOM swap instead of a full page redirect. For HTMX requests, return HTTP 200 with `HX-Redirect` header. For normal browser requests, return 302.

```python
# Source: https://htmx.org/headers/hx-redirect/ + https://fastapi.tiangolo.com/tutorial/middleware/
# shop/middleware/setup_guard.py
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

SETUP_EXEMPT_PATHS = {"/setup", "/static", "/favicon.ico"}

class SetupGuardMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip static files and setup routes
        if any(request.url.path.startswith(p) for p in SETUP_EXEMPT_PATHS):
            return await call_next(request)

        db = next(get_db())  # Get DB session for check
        setup_complete = check_setup_complete(db)

        if not setup_complete:
            redirect_url = "/setup/step1"
            is_htmx = request.headers.get("HX-Request") == "true"
            if is_htmx:
                return Response(
                    status_code=200,
                    headers={"HX-Redirect": redirect_url}
                )
            return RedirectResponse(url=redirect_url, status_code=302)

        return await call_next(request)
```

### Pattern 3: SQLAlchemy 2.0 Session Dependency

**What:** Yield-based dependency for per-request DB sessions. Uses synchronous SQLAlchemy with SQLite (no async needed for a shop with O(10) users).

```python
# Source: https://fastapi.tiangolo.com/tutorial/sql-databases/
# shop/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

SQLALCHEMY_DATABASE_URL = "sqlite:///./shop.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

# shop/dependencies.py
from typing import Generator
from sqlalchemy.orm import Session
from shop.database import SessionLocal

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Pattern 4: RBAC Dependency — Role Enforcement

**What:** FastAPI `Depends()` chain: extract session token from cookie → look up session → look up user → check role. Silent redirect for unauthorized access (per locked decision: no 403 page).

```python
# shop/dependencies.py
from fastapi import Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from shop.models import User, Session as UserSession

def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get("session_token")
    if not token:
        raise NeedsLoginRedirect()
    session = db.query(UserSession).filter(
        UserSession.id == token,
        UserSession.expires_at > datetime.utcnow()
    ).first()
    if not session:
        raise NeedsLoginRedirect()
    # Sliding window: reset expiry on each request
    session.expires_at = datetime.utcnow() + timedelta(hours=8)
    db.commit()
    return session.user

def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        # Silent redirect — no 403, no error message
        raise HTTPException(status_code=307, headers={"Location": "/dashboard"})
    return user
```

### Pattern 5: Password Hashing with pwdlib

**What:** Use `pwdlib[bcrypt]` at cost factor 12 (per locked decision). Pin bcrypt<5.0.0 to avoid compatibility breakage.

```python
# Source: https://pypi.org/project/pwdlib/ + https://github.com/frankie567/pwdlib
# shop/services/auth.py
from pwdlib import PasswordHash
from pwdlib.hashers.bcrypt import BcryptHasher

# Cost factor 12 as decided
password_hash = PasswordHash([BcryptHasher(rounds=12)])

def hash_password(plain: str) -> str:
    return password_hash.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return password_hash.verify(plain, hashed)
```

### Pattern 6: Excel Upload → In-Memory Parse → HTMX Partial

**What:** Receive `UploadFile`, read into BytesIO, call existing `xlsx.py` logic server-side, return HTMX partial with pre-filled mapping table.

```python
# Source: https://fastapi.tiangolo.com/tutorial/request-files/
# shop/routers/setup.py
from io import BytesIO
from fastapi import UploadFile, File, Depends
from openpyxl import load_workbook
from delta_preservation.io.xlsx import load_form3
from delta_preservation.config import FORM3_HEADER_KEYWORDS

@router.post("/step4/upload", response_class=HTMLResponse)
async def upload_form3(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Fatal error check first
    content = await file.read()
    if not content:
        return templates.TemplateResponse("setup/step4_error.html",
            {"request": request, "error": "File is empty"})
    try:
        wb = load_workbook(BytesIO(content))
    except Exception as e:
        return templates.TemplateResponse("setup/step4_error.html",
            {"request": request, "error": f"Cannot read file: {e}"})

    ws = wb.active
    headers = [cell.value for cell in next(ws.iter_rows(max_row=1))]
    preview_rows = list(ws.iter_rows(min_row=2, max_row=6, values_only=True))

    # Auto-detect using existing config keywords
    detected = detect_column_mapping(headers, FORM3_HEADER_KEYWORDS)

    return templates.TemplateResponse("setup/step4_mapping.html", {
        "request": request,
        "headers": headers,
        "preview_rows": preview_rows,
        "detected": detected,
    })
```

### Pattern 7: SSE Endpoint for Pipeline Progress (Phase 2 placeholder)

**What:** FastAPI native SSE via `EventSourceResponse` from `fastapi.sse`. Template architecture accommodates this now so Phase 2 needs no rewrite.

```python
# Source: https://fastapi.tiangolo.com/tutorial/server-sent-events/
from fastapi.sse import EventSourceResponse, ServerSentEvent

@router.get("/runs/{run_id}/stream", response_class=EventSourceResponse)
async def stream_run_progress(run_id: str):
    async def event_generator():
        # Phase 2 will populate this from a job queue
        yield ServerSentEvent(data="connected", event="status")
    return EventSourceResponse(event_generator())
```

### Pattern 8: Docker Multi-Stage Build with uv + Node

**What:** Three-stage Dockerfile: (1) Node stage for Tailwind CSS compilation, (2) Python/uv builder stage, (3) minimal Python runtime stage.

```dockerfile
# Source: https://docs.astral.sh/uv/guides/integration/docker/
# Stage 1: Build Tailwind CSS
FROM node:22-alpine AS tailwind-builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY static/src/ ./static/src/
COPY tailwind.config.js ./
RUN npx tailwindcss -i ./static/src/input.css -o ./static/dist/output.css --minify

# Stage 2: Python dependency builder
FROM python:3.11-slim AS python-builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-editable

# Stage 3: Runtime
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=python-builder /app/.venv /app/.venv
COPY --from=tailwind-builder /app/static/dist/ ./static/dist/
COPY delta_preservation/ ./delta_preservation/
COPY shop/ ./shop/
COPY static/js/ ./static/js/
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "shop.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Pattern 9: SQLite Schema — Users, Sessions, ShopConfig

**What:** Core tables needed for Phase 1. All persisted in `shop.db` SQLite file mounted via Docker volume.

```python
# shop/models.py
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from shop.database import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)  # "admin" | "engineer"
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    sessions: Mapped[list["UserSession"]] = relationship(back_populates="user")

class UserSession(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String, primary_key=True)  # signed token
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    user: Mapped["User"] = relationship(back_populates="sessions")

class ShopConfig(Base):
    __tablename__ = "shop_config"
    id: Mapped[int] = mapped_column(primary_key=True)
    shop_name: Mapped[str] = mapped_column(String, default="")
    setup_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    column_mapping: Mapped[dict] = mapped_column(JSON, default=dict)
    # wizard_step tracks last completed step (1-4); 0 = not started
    wizard_step: Mapped[int] = mapped_column(default=0)
```

### Anti-Patterns to Avoid

- **Storing session data in the cookie itself:** Session data belongs in the `sessions` table, not in a signed cookie payload. Only the session token ID goes in the cookie.
- **Using 302 redirect for HTMX requests:** HTMX treats 302 as a partial swap target, not a navigation. Use `HX-Redirect` header with 200 status for HTMX; 302 only for plain browser navigations.
- **Calling `xlsx.py` logic from within middleware:** Import the detection logic into the setup router only; middleware must stay lightweight (only reads `setup_complete` flag).
- **Async SQLAlchemy with SQLite:** Async SQLAlchemy requires `aiosqlite` and adds complexity. Sync is correct for this single-shop deployment.
- **Tailwind `@apply` in Jinja2 templates:** Write Tailwind utility classes directly in templates. Reserve `@apply` only for truly reusable component definitions in `input.css`.
- **Mounting HTMX from a CDN:** Air-gapped requirement (DEPLOY-03) means HTMX must be served from `/static/js/`. No CDN links in `base.html`.
- **bcrypt pinning omission:** Not pinning `bcrypt<5.0.0` causes `HasherNotAvailable` error at runtime when pwdlib 0.3.0 tries to use bcrypt 5.x.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Session token signing | Custom HMAC signing | `itsdangerous.URLSafeTimedSerializer` | Timing-safe comparison, replay protection |
| Password hashing | Custom bcrypt wrapper | `pwdlib[bcrypt]` | Handles salt generation, timing-safe verify |
| HTMX redirect detection | Parse headers manually | Check `request.headers.get("HX-Request") == "true"` | HTMX always sends this header |
| Excel in-memory parsing | Save to disk then read | `openpyxl.load_workbook(BytesIO(content))` | Avoids temp file management |
| Column keyword matching | Rewrite detection logic | Reuse `delta_preservation/config.py:FORM3_HEADER_KEYWORDS` | Already tested and working |
| CSS component styling | Custom CSS classes | DaisyUI v5 classes (`btn`, `card`, `badge`, `modal`) | Consistent, maintained, accessible |
| DB session cleanup | Background task | SQLAlchemy `delete().where(expires_at < now())` on login | Simple enough; cron-style cleanup on login |

**Key insight:** The existing `delta_preservation/` package already contains the Excel keyword detection vocabulary (`FORM3_HEADER_KEYWORDS`) and the column parsing logic (`xlsx.py:load_form3()`). The web layer should call or adapt these rather than rebuilding the detection.

---

## Common Pitfalls

### Pitfall 1: HTMX 302 Redirect Swaps Instead of Navigates

**What goes wrong:** Middleware or login handler returns HTTP 302. HTMX intercepts it, fetches the redirect target, and swaps the result into the current page's `hx-target` — the user sees the login page embedded in the middle of the current page instead of navigating away.

**Why it happens:** HTMX follows redirects silently and swaps the response body into the target element, rather than navigating the browser.

**How to avoid:** For HTMX requests (`HX-Request: true`), return 200 with `HX-Redirect: /destination` header. For plain browser requests, return standard 302. Detect with: `request.headers.get("HX-Request") == "true"`.

**Warning signs:** Login page content appears inside an existing page's layout. Browser URL doesn't change after a redirect-triggering action.

### Pitfall 2: bcrypt 5.0.0 Breaks pwdlib at Runtime

**What goes wrong:** `HasherNotAvailable` error when calling `password_hash.hash()`. Application starts but crashes on first login attempt.

**Why it happens:** bcrypt 5.0.0 removed the `__about__` module attribute that pwdlib 0.3.0 uses for backend detection.

**How to avoid:** Pin `bcrypt<5.0.0` in `pyproject.toml` (e.g., `"bcrypt>=4.0.0,<5.0.0"`). Confirmed working version: 4.3.0.

**Warning signs:** Error message contains `HasherNotAvailable` or `bcrypt hash algorithm is not available`.

### Pitfall 3: DaisyUI v4 vs v5 Class Name Changes

**What goes wrong:** Component classes from documentation or tutorials do not render correctly; elements appear unstyled.

**Why it happens:** DaisyUI v5 (current) renames and removes many v4 class names. Installing `daisyui@latest` gives v5; using v4 class names silently produces no output.

**How to avoid:** Target DaisyUI v5 (5.5.x) from the start. Use the v5 upgrade guide at `https://daisyui.com/docs/upgrade/` as a reference. Notable changes: `artboard` removed, avatar classes renamed, `menu-disabled`/`menu-active` added, color variables changed for Tailwind v4 CSS variables.

**Warning signs:** Buttons/cards appear completely unstyled despite DaisyUI being installed.

### Pitfall 4: Setup Middleware Intercepts Static Files and Setup Routes

**What goes wrong:** The middleware redirects requests for `/static/output.css` and `/setup/step1` itself to `/setup/step1`, causing an infinite redirect loop. The browser loads a blank page or spins indefinitely.

**Why it happens:** Middleware runs for every request including static files and the `/setup/` routes.

**How to avoid:** Explicitly exempt paths in the middleware: `/static`, `/favicon.ico`, and all `/setup/` prefixes from the redirect check. See Pattern 2 above.

**Warning signs:** Browser shows "too many redirects" error when navigating to `/setup/`.

### Pitfall 5: Session Expiry Not Sliding (Resets on Each Request)

**What goes wrong:** Sessions expire after 8 hours from creation rather than 8 hours from last activity.

**Why it happens:** The `expires_at` column is set once at creation and never updated on subsequent requests.

**How to avoid:** In the `get_current_user()` dependency, update `session.expires_at = datetime.utcnow() + timedelta(hours=8)` and call `db.commit()` before returning the user. This runs on every authenticated request.

**Warning signs:** Users report being logged out despite active use.

### Pitfall 6: Wizard Forward-Skip via Direct URL

**What goes wrong:** Admin navigates directly to `/setup/step4` before completing steps 1-3, bypassing required configuration.

**Why it happens:** If step routing only checks `setup_complete` (via middleware) and not per-step completion state, any step URL is accessible once the app starts.

**How to avoid:** Each step's GET handler checks `wizard_step` from `shop_config` — if `wizard_step < step_number - 1`, redirect to the first incomplete step. The middleware only guards against bypassing setup entirely; step ordering is enforced in the router.

**Warning signs:** Admin can access step 4 without completing step 1.

### Pitfall 7: Python Version Constraint

**What goes wrong:** `uv sync` fails in Docker because the base image Python version doesn't satisfy `requires-python = ">=3.10,<3.13"` in `pyproject.toml`.

**Why it happens:** The existing `pyproject.toml` pins `<3.13`. The existing `.venv` uses Python 3.11 (confirmed by `find` output). Using Python 3.13 base image will fail.

**How to avoid:** Use `python:3.11-slim` as the runtime base image. Do not use `python:3.13`.

**Warning signs:** Docker build fails with `No interpreter found for constraint >=3.10,<3.13`.

---

## Code Examples

Verified patterns from official sources:

### Setting an HttpOnly Session Cookie

```python
# Source: https://fastapi.tiangolo.com/advanced/response-cookies/ + Starlette docs
from fastapi import Response

def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,          # Not accessible via JavaScript
        samesite="lax",         # CSRF protection
        secure=False,           # True only for HTTPS; False for local HTTP shop
        max_age=8 * 3600,       # 8 hours in seconds
    )

def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key="session_token")
```

### FastAPI TestClient with In-Memory SQLite Override

```python
# Source: https://fastapi.tiangolo.com/how-to/testing-database/
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, StaticPool
from sqlalchemy.orm import sessionmaker
from shop.app import create_app
from shop.database import Base
from shop.dependencies import get_db

@pytest.fixture(scope="function")
def db_engine():
    engine = create_engine(
        "sqlite://",  # In-memory
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="function")
def client(db_engine):
    TestingSessionLocal = sessionmaker(bind=db_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
```

### Jinja2Templates Setup in FastAPI

```python
# Source: https://fastapi.tiangolo.com/advanced/templates/
from fastapi import Request
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="shop/templates")

@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("auth/login.html", {"request": request})
```

### HTMX Base Template (air-gapped, no CDN)

```html
<!-- shop/templates/base.html -->
<!DOCTYPE html>
<html data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/static/dist/output.css">
    <script src="/static/js/htmx.min.js"></script>
    <script src="/static/js/htmx-sse.js"></script>
</head>
<body>
    {% block content %}{% endblock %}
</body>
</html>
```

### Tailwind Input CSS Entry Point (v4 + DaisyUI v5)

```css
/* static/src/input.css */
@import "tailwindcss";
@plugin "daisyui";
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `passlib[bcrypt]` | `pwdlib[bcrypt]` | passlib unmaintained since 2020; breaks Python 3.13+ | Use pwdlib; pin bcrypt<5.0.0 |
| DaisyUI v4 (`@tailwindcss/...`) | DaisyUI v5 (`@plugin "daisyui"`) | DaisyUI 5.0 released 2025 | CSS import syntax and class names changed |
| Tailwind v3 config JS | Tailwind v4 CSS-based config | Tailwind v4 released 2024 | Config moves into CSS file; no `tailwind.config.js` needed |
| FastAPI `StreamingResponse` for SSE | `EventSourceResponse` from `fastapi.sse` | FastAPI 0.115+ | Native SSE with keep-alive, cache headers built in |
| SQLModel | SQLAlchemy 2.0 + raw Pydantic | Ongoing; SQLModel has coupling concerns | Locked decision; use SQLAlchemy 2.0 directly |

**Deprecated/outdated:**
- `fastapi-sessions` (PyPI): Unmaintained; use custom sessions table pattern instead
- `starsessions`: Adds dependency; custom `sessions` table is straightforward for this use case
- Tailwind standalone binary: Only supports Tailwind v3; must use npm + Tailwind CLI v4

---

## Open Questions

1. **Migration approach for schema evolution**
   - What we know: `pyproject.toml` has no Alembic; user left schema details to Claude's discretion
   - What's unclear: Whether to add Alembic for migration tracking or use `Base.metadata.create_all()` for v1
   - Recommendation: Use `Base.metadata.create_all()` for Phase 1 (single shop, no existing data to migrate). Add Alembic in a future phase only if schema changes are needed after deployment.

2. **Session token format**
   - What we know: Session stored as `id` in `sessions` table; cookie holds this ID
   - What's unclear: Whether to use a plain `uuid4` or a signed token via `itsdangerous`
   - Recommendation: Use `secrets.token_urlsafe(32)` as the session ID (sufficient entropy, no signing needed since the DB is the authority). `itsdangerous` is optional overhead for this threat model.

3. **Docker volume path for SQLite DB and `out/` directory**
   - What we know: `docker-compose.yml` must mount both `shop.db` and the pipeline `out/` directory
   - What's unclear: Exact path conventions for the `docker-compose.yml` volume mounts
   - Recommendation: Mount `./data/shop.db:/app/shop.db` and `./data/out:/app/out` in the compose file; document in README.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.x (already in `pyproject.toml`) |
| Config file | none — see Wave 0 (add `[tool.pytest.ini_options]` to `pyproject.toml`) |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AUTH-01 | Admin creates engineer account; account appears in DB | integration | `uv run pytest tests/test_auth.py::test_admin_creates_engineer -x` | ❌ Wave 0 |
| AUTH-02 | Login sets HttpOnly cookie; cookie persists to next request | integration | `uv run pytest tests/test_auth.py::test_session_persists -x` | ❌ Wave 0 |
| AUTH-03 | Logout deletes session and clears cookie | integration | `uv run pytest tests/test_auth.py::test_logout -x` | ❌ Wave 0 |
| AUTH-04 | Admin can view and deactivate engineer accounts | integration | `uv run pytest tests/test_admin.py::test_deactivate_engineer -x` | ❌ Wave 0 |
| AUTH-05 | Engineer GET /admin returns redirect; admin GET /admin returns 200 | integration | `uv run pytest tests/test_rbac.py::test_engineer_cannot_access_admin -x` | ❌ Wave 0 |
| AUTH-06 | reviewed_by / reviewed_at on action record | unit | `uv run pytest tests/test_models.py::test_reviewer_fields -x` | ❌ Wave 0 |
| SETUP-01 | Unauthenticated GET / redirects to /setup/step1 when setup incomplete | integration | `uv run pytest tests/test_setup.py::test_wizard_intercepts_all_routes -x` | ❌ Wave 0 |
| SETUP-02 | Excel upload returns mapping table partial with pre-filled dropdowns | integration | `uv run pytest tests/test_setup.py::test_form3_upload_autodetect -x` | ❌ Wave 0 |
| SETUP-03 | Empty file returns error before mapping UI | integration | `uv run pytest tests/test_setup.py::test_empty_file_error -x` | ❌ Wave 0 |
| SETUP-04 | Non-contiguous char numbers load without normalization | unit | `uv run pytest tests/test_setup.py::test_noncontiguous_char_no -x` | ❌ Wave 0 |
| DEPLOY-01 | docker compose up succeeds with no external services | manual | `docker compose up --build` + smoke test GET / | N/A — manual |
| DEPLOY-02 | All deps in image; `uv run python -c "import shop"` succeeds | manual | Run inside container post-build | N/A — manual |
| DEPLOY-03 | No outbound HTTP calls at runtime | manual | Network-isolated container test | N/A — manual |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/ -x -q`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/__init__.py` — makes tests/ a package
- [ ] `tests/conftest.py` — in-memory SQLite engine, TestClient fixture, admin user seed
- [ ] `tests/test_auth.py` — AUTH-01 through AUTH-05
- [ ] `tests/test_rbac.py` — AUTH-05 (RBAC enforcement)
- [ ] `tests/test_models.py` — AUTH-06 (reviewer fields)
- [ ] `tests/test_setup.py` — SETUP-01 through SETUP-04
- [ ] `tests/test_admin.py` — AUTH-04 (user management)
- [ ] Add `[tool.pytest.ini_options]` to `pyproject.toml` with `testpaths = ["tests"]`
- [ ] Framework install: `uv add --dev httpx` (required by FastAPI TestClient for async testing)

---

## Sources

### Primary (HIGH confidence)

- `https://fastapi.tiangolo.com/tutorial/middleware/` — Middleware pattern, redirect from middleware
- `https://fastapi.tiangolo.com/tutorial/server-sent-events/` — `EventSourceResponse`, `ServerSentEvent`, `fastapi.sse` import
- `https://fastapi.tiangolo.com/advanced/response-cookies/` — `set_cookie()` parameters; `httponly`, `samesite`, `secure`
- `https://fastapi.tiangolo.com/how-to/testing-database/` — TestClient + in-memory SQLite override pattern
- `https://pypi.org/project/pwdlib/` — Version 0.3.0 confirmed; bcrypt extra available
- `https://daisyui.com/docs/install/` — DaisyUI v5.5.19 confirmed; `@plugin "daisyui"` syntax; Tailwind v4 required
- `https://htmx.org/headers/hx-redirect/` — HX-Redirect header; why 302 doesn't work with HTMX
- `https://docs.astral.sh/uv/guides/integration/docker/` — uv multi-stage Docker pattern; `--no-editable`, `--locked`

### Secondary (MEDIUM confidence)

- `https://github.com/fastapi-users/fastapi-users/issues/1555` — bcrypt 5.0.0 + pwdlib incompatibility confirmed; pin to <5.0.0
- `https://github.com/frankie567/pwdlib/discussions/1` — pwdlib introduction; bcrypt cost factor config
- `https://daisyui.com/docs/upgrade/` — DaisyUI v4→v5 breaking changes (class renames, color system)
- `https://chaoticengineer.hashnode.dev/fastapi-sqlalchemy` — SQLAlchemy 2.0 patterns with FastAPI; sync vs async guidance

### Tertiary (LOW confidence — needs validation)

- Sliding session expiry pattern: inferred from standard web auth practice; no single authoritative FastAPI doc found — verify in implementation
- `itsdangerous` vs `secrets.token_urlsafe` for session token: recommendation is author's judgment; either approach is standard

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — all libraries verified via PyPI + official docs; versions confirmed
- Architecture: HIGH — patterns derived from official FastAPI docs; HTMX redirect behavior confirmed from HTMX official headers doc
- Pitfalls: HIGH for bcrypt pin (confirmed bug report), DaisyUI v5 breaking changes (confirmed upgrade guide), HTMX 302 (confirmed official docs); MEDIUM for wizard forward-skip (standard pattern, not FastAPI-specific)

**Research date:** 2026-03-02
**Valid until:** 2026-04-02 (30 days; DaisyUI and FastAPI move at moderate pace; bcrypt pin is fixed)