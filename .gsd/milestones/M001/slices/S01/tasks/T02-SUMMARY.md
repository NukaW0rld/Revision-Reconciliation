---
id: T02
parent: S01
milestone: M001
provides:
  - "shop/ Python package: importable foundation for all subsequent plans"
  - "SQLAlchemy 2.0 ORM models: User, UserSession, ShopConfig with correct table schemas"
  - "Auth service: hash_password/verify_password with bcrypt cost-12, create_session/delete_session"
  - "FastAPI app factory: create_app() returning configured FastAPI instance with middleware"
  - "Dependency injection chain: get_db(), get_current_user() (cookie+sliding window), require_admin()"
  - "SetupGuardMiddleware: redirects all requests to /setup/step1 until setup_complete=True"
requires: []
affects: []
key_files: []
key_decisions: []
patterns_established: []
observability_surfaces: []
drill_down_paths: []
duration: 7min
verification_result: passed
completed_at: 2026-03-03
blocker_discovered: false
---
# T02: 01-foundation 02

**# Phase 1 Plan 2: Core shop Package Summary**

## What Happened

# Phase 1 Plan 2: Core shop Package Summary

**SQLAlchemy 2.0 ORM foundation with bcrypt auth service, FastAPI app factory, and HTMX-aware setup guard middleware**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-03T01:13:20Z
- **Completed:** 2026-03-03T01:20:00Z
- **Tasks:** 3
- **Files modified:** 10 (9 created, 1 modified)

## Accomplishments
- Importable `shop/` package with SQLAlchemy 2.0 ORM models (User, UserSession, ShopConfig) using Mapped/mapped_column API
- Auth service with bcrypt cost-12 password hashing via pwdlib and session token management using secrets.token_urlsafe(32)
- FastAPI app factory (create_app()) with SetupGuardMiddleware, DB table creation on startup, and graceful static file mount
- Dependency injection chain: get_db() -> get_current_user() (cookie auth + sliding 8-hour window) -> require_admin() (silent redirect)
- SetupGuardMiddleware correctly distinguishes HTMX requests (HX-Redirect header, 200) from browser navigations (302)

## Task Commits

Each task was committed atomically:

1. **Task 1: Database module and ORM models** - `dc8a9d7` (feat)
2. **Task 2: Auth service — password hashing and session management** - `b0be2d0` (feat)
3. **Task 3: FastAPI app factory, dependencies, and middleware skeleton** - `f224c2e` (feat)

## Files Created/Modified
- `shop/__init__.py` - Package marker (empty)
- `shop/database.py` (14 lines) - SQLAlchemy engine, SessionLocal, DeclarativeBase
- `shop/models.py` (42 lines) - User, UserSession, ShopConfig ORM models with SQLAlchemy 2.0 API
- `shop/services/__init__.py` - Package marker (empty)
- `shop/services/auth.py` (34 lines) - hash_password, verify_password, create_session, delete_session
- `shop/dependencies.py` (44 lines) - get_db, get_current_user, require_admin FastAPI dependencies
- `shop/middleware/__init__.py` - Package marker (empty)
- `shop/middleware/setup_guard.py` (60 lines) - SetupGuardMiddleware with HTMX redirect support
- `shop/app.py` (28 lines) - create_app() factory; app module-level instance for uvicorn
- `pyproject.toml` (modified) - Added shop, shop.middleware, shop.services to packages list

## Decisions Made
- `secrets.token_urlsafe(32)` chosen over `itsdangerous` for session tokens: DB is the authority for session validity; signing overhead unnecessary for this threat model (per open question #2 in RESEARCH.md)
- `bcrypt` pinned `>=4.0.0,<5.0.0` to avoid `HasherNotAvailable` runtime error from bcrypt 5.0.0's removal of `__about__` attribute
- Routers NOT registered in app.py — intentional; Plans 03-05 will add auth, setup, admin routers
- `require_admin` raises HTTP 302 (not 307) for redirect; consistent with locked decision (silent redirect, no 403)
- SetupGuardMiddleware reads `shop_config` table directly (not via dependency injection) to stay lightweight

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added shop subpackages to pyproject.toml packages list**
- **Found during:** Task 3 (app factory and verification)
- **Issue:** `[tool.setuptools]` only listed `delta_preservation`; `shop`, `shop.middleware`, `shop.services` were not registered as packages, which would prevent proper installation
- **Fix:** Added `"shop", "shop.middleware", "shop.services"` to the packages list in `pyproject.toml`
- **Files modified:** `pyproject.toml`
- **Verification:** `uv run python -c "import shop"` succeeds after update
- **Committed in:** `f224c2e` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Required for the `shop` package to be properly installable. No scope creep.

## Issues Encountered
None - all three tasks executed cleanly against the plan specification.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All downstream plans (01-03 through 01-08) can now `from shop.models import ...`, `from shop.dependencies import ...`, and `from shop.services.auth import ...`
- conftest.py from plan 01-01 already imports from shop/ — those imports now resolve correctly
- Running `uv run pytest tests/ -x -q` shows 10 xfail tests (expected; routes not yet implemented)
- Plan 01-03 (auth routes, HTMX base template) is the next dependency

---
*Phase: 01-foundation*
*Completed: 2026-03-03*
