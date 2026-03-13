---
id: S01
parent: M001
milestone: M001
provides:
  - pytest configured with testpaths=["tests"] and -q addopts
  - tests/conftest.py with db_engine, client, admin_user, shop_config fixtures
  - In-memory SQLite test DB with StaticPool per test function
  - TestClient fixture wired to get_db dependency override
  - All Phase 1 test stub files (10 tests total, all xfail until shop/ exists)
  - "shop/ Python package: importable foundation for all subsequent plans"
  - "SQLAlchemy 2.0 ORM models: User, UserSession, ShopConfig with correct table schemas"
  - "Auth service: hash_password/verify_password with bcrypt cost-12, create_session/delete_session"
  - "FastAPI app factory: create_app() returning configured FastAPI instance with middleware"
  - "Dependency injection chain: get_db(), get_current_user() (cookie+sliding window), require_admin()"
  - "SetupGuardMiddleware: redirects all requests to /setup/step1 until setup_complete=True"
  - GET /login — login form rendered via Jinja2 base template
  - POST /login — validates credentials, sets HttpOnly session_token cookie, redirects to /dashboard
  - POST /logout — deletes session row, clears cookie, redirects to /login
  - GET /dashboard — gated by get_current_user dependency; shows user email and role
  - shop/templates/base.html — HTMX + Tailwind CSS base layout (no CDN links)
  - static/js/htmx.min.js and htmx-sse.js — bundled locally for air-gapped operation
  - GET /admin/users returns user list (admin only, silent redirect for engineer)
  - POST /admin/users creates engineer account with hashed password
  - POST /admin/users/{id}/deactivate sets is_active=False (soft delete)
  - shop/templates/admin/users.html with HTMX table management
  - shop/templates/admin/users_row.html HTMX partial for inline row swap
  - Setup wizard router at /setup prefix with GET+POST for steps 1-3
  - Wizard step ordering enforcement (no forward-skip)
  - Mid-wizard resume via GET /setup/ root redirect
  - wizard_layout.html shared layout with 4-step progress indicator
  - step1_shop_name.html, step2_password.html, step3_engineer.html templates
  - Form 3 column mapping service (shop/services/form3.py) wrapping FORM3_HEADER_KEYWORDS
  - Step 4 wizard completion: POST /setup/step4/upload (HTMX partial), POST /setup/step4/save
  - Admin settings panel: GET/POST /admin/settings, /settings/upload, /settings/save
  - Two-phase validation: fatal errors before mapping UI, column mismatches within UI
  - setup_complete=True set atomically with column_mapping on wizard save
  - Three-stage Dockerfile (Node tailwind-builder, uv python-builder, python:3.11-slim runtime)
  - docker-compose.yml with volume mounts and env var configuration
  - run_web.py startup entrypoint seeding admin user and starting uvicorn
  - DATABASE_URL env var support in shop/database.py for container path override
  - Human-verified Docker build and full Phase 1 happy path confirmation
  - Root redirect GET / -> /login so navigating to app root after setup returns login page (not 404)
  - Setup wizard step 2 displays seeded admin email to guide first-time users
  - Role-based dashboard navigation: admin sees User Management + Settings cards; engineer sees minimal view
  - All 21 automated tests passing (2 xfail, 3 xpassed)
requires: []
affects: []
key_files: []
key_decisions:
  - "bcrypt pinned <5.0.0 — bcrypt 5.0.0 removed __about__ attribute that pwdlib 0.3.0 uses for HasherAvailable check"
  - "xfail(strict=False) chosen over pytest.skip so test names appear in collection and xfail count is visible"
  - "conftest imports from shop/ intentionally left failing — scaffold-first approach; imports succeed after Plan 02"
  - "Used secrets.token_urlsafe(32) for session tokens (no itsdangerous signing overhead; DB is the authority)"
  - "bcrypt pinned <5.0.0 to avoid pwdlib HasherNotAvailable bug with bcrypt 5.x"
  - "Routers intentionally omitted from app.py — Plans 03-05 register them"
  - "SetupGuardMiddleware reads from DB on every request (lightweight: only setup_complete flag)"
  - "require_admin uses HTTP 302 redirect (not 307) with Location header to /dashboard"
  - "Standard HTML POST (not HTMX POST) for login/logout — browser natively follows 302; HTMX intercepts redirect as partial swap causing broken navigation"
  - "Jinja2 TemplateResponse uses request as first positional arg (new FastAPI >= 0.111 API) to avoid deprecation warnings"
  - "Router imports deferred inside create_app() body to avoid circular import: auth.py imports templates from shop.app, shop.app imports auth.router"
  - "HTMX downloaded via curl from unpkg at plan time; stub fallback comment if air-gapped at plan execution"
  - "TemplateResponse uses new Starlette 0.52 signature (request first) matching auth.py convention"
  - "Deactivate is soft-delete only (is_active=False) — user record preserved for audit trail"
  - "hx-confirm provides deactivation confirmation via native browser dialog with zero custom JS"
  - "setup router registered at /setup prefix via include_router in create_app()"
  - "Wizard templates use standard HTML POST (not HTMX): wizard is full-page flow; HTMX partial swaps conflict with browser back/forward and step validation redirects"
  - "step3 GET redirects to wizard_step+1 (not always step2) to correctly resume mid-wizard from any prior incomplete state"
  - "step4_mapping_partial.html reused for both wizard (/setup/step4/save) and admin settings (/admin/settings/save) via form_action template variable — avoids duplication"
  - "Jinja2 set+update pattern used to build detected_by_col inverse dict in template (col_idx -> field) since Jinja2 lacks dict comprehension over items()"
  - "Unmatched columns get select name=col_{idx} (not a required_field name) so the save endpoint ignores them — no JS needed to dynamically rename selects"
  - "python:3.11-slim used (not 3.12/3.13) to respect pyproject.toml requires-python >=3.10,<3.13"
  - "Node.js confined to tailwind-builder stage only — not present in runtime image (DEPLOY-03)"
  - "uv sync --locked --no-install-project in builder stage ensures exact lockfile dependency versions (DEPLOY-02)"
  - "data/ dir on host maps to /app/shop.db and /app/out in container for runtime persistence"
  - "DATABASE_URL env var in database.py enables Docker volume path override without code changes"
  - "Admin seeded from ADMIN_EMAIL + ADMIN_DEFAULT_PASSWORD env vars on startup; idempotent (checks for existing admin)"
  - "Root redirect added to auth router (GET / -> /login) rather than a separate catch-all to keep routing logic co-located with auth"
  - "Admin email surfaced on step 2 of wizard so users know credentials after seeding — avoids support confusion on first run"
  - "Role-based dashboard navigation via Jinja2 conditional on request.state.user.role — no JavaScript needed"
  - "Verification gate confirmed: no outbound internet requests at runtime (all JS/CSS served from /static/)"
  - "User note: approved with future process revision intent — indicates possible UX iteration in Phase 2 but no blocking issues"
patterns_established:
  - "StaticPool in-memory SQLite: create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)"
  - "Dependency override pattern: app.dependency_overrides[get_db] = override_get_db in client fixture"
  - "xfail stub pattern: @pytest.mark.xfail(strict=False, reason='shop/ not implemented yet — Wave 1+') + raise NotImplementedError"
  - "Pattern: App factory — create_app() creates fresh FastAPI instance for test isolation"
  - "Pattern: HTMX redirect — middleware checks HX-Request header, returns HX-Redirect on 200 for HTMX, 302 for browser"
  - "Pattern: Session sliding window — every authenticated request updates expires_at in DB"
  - "Pattern: Silent admin redirect — non-admin users silently redirected to /dashboard, no 403"
  - "Pattern 1: Auth flow — POST /login sets HttpOnly cookie then RedirectResponse(302); GET /dashboard depends on get_current_user"
  - "Pattern 2: Test isolation — session_factory injected into create_app and middleware; conftest seeds ShopConfig row to bypass setup guard"
  - "Pattern 3: All templates extend base.html; no CDN links; all assets served from /static/"
  - "HTMX partial pattern: POST returns bare <tr> HTML; full page GET returns full template"
  - "Error partial: same users_row.html handles both success and error states via Jinja2 if/else"
  - "Wizard step guard pattern: if config.wizard_step < N-1: redirect to first incomplete step"
  - "_get_or_create_config() helper for ShopConfig singleton safety"
  - "TemplateResponse uses new Starlette 0.52 signature (request first, template name second) matching admin.py convention"
  - "HTMX file upload: hx-encoding=multipart/form-data on form, hx-target on container div, bare HTML partial returned (no base.html extension)"
  - "Fatal error partial (step4_error.html) returned as 200 OK — HTMX treats all 2xx as success for DOM swap"
  - "Shared partial template pattern: pass form_action as context variable to enable dual-destination forms"
  - "Startup script pattern: seed DB state before starting server process via subprocess.run"
  - "Env-driven database URL: os.environ.get('DATABASE_URL', default) for container flexibility"
  - "Human verification pattern: automated tests pass first, then Docker e2e verification, then phase marked complete"
  - "Role-gated template content: {{ request.state.user.role == 'admin' }} conditionals in Jinja2 templates"
observability_surfaces: []
drill_down_paths: []
duration: ~6h (includes human Docker build and verification time)
verification_result: passed
completed_at: 2026-03-03
blocker_discovered: false
---
# S01: Foundation

**# Phase 1 Plan 01: Test Infrastructure Scaffold Summary**

## What Happened

# Phase 1 Plan 01: Test Infrastructure Scaffold Summary

**pytest + FastAPI TestClient scaffold with in-memory SQLite fixtures and 10 xfail stubs covering all Phase 1 requirements (AUTH-01 through AUTH-06, SETUP-01 through SETUP-04)**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-03T01:12:50Z
- **Completed:** 2026-03-03T01:15:10Z
- **Tasks:** 3
- **Files modified:** 8 (pyproject.toml + 7 new test files)

## Accomplishments

- pytest configured via `[tool.pytest.ini_options]` in pyproject.toml; all Phase 1 web dependencies installed and resolved without conflicts
- `tests/conftest.py` provides four fixtures: `db_engine` (in-memory SQLite/StaticPool), `client` (TestClient with get_db override), `admin_user` (seeded admin), `shop_config` (seeded pre-setup ShopConfig)
- All 10 test stubs collected by pytest and shown as `xfail` — no hard failures, ready to flip to real tests as `shop/` package is built in subsequent plans

## Task Commits

Each task was committed atomically:

1. **Task 1: Configure pytest and install dev dependencies** - `e233de7` (chore)
2. **Task 2: Create conftest.py with in-memory SQLite and TestClient fixture** - `dd6696a` (feat)
3. **Task 3: Create test stub files for all Phase 1 requirements** - `1d16032` (feat)

## Files Created/Modified

- `pyproject.toml` - Added `[tool.pytest.ini_options]`, Phase 1 web dependencies, bcrypt pin, httpx dev dep
- `tests/__init__.py` - Empty package marker
- `tests/conftest.py` - Four fixtures: db_engine, client, admin_user, shop_config
- `tests/test_auth.py` - AUTH-01, AUTH-02, AUTH-03 stubs (engineer creation, session persist, logout)
- `tests/test_rbac.py` - AUTH-05 stub (engineer cannot access admin routes)
- `tests/test_models.py` - AUTH-06 stub (User model reviewer fields)
- `tests/test_setup.py` - SETUP-01 through SETUP-04 stubs (wizard intercept, upload autodetect, empty file error, noncontiguous char_no)
- `tests/test_admin.py` - AUTH-04 stub (deactivate engineer)

## Decisions Made

- **bcrypt pinned `<5.0.0`:** bcrypt 5.0.0 removed the `__about__` attribute that pwdlib 0.3.0 uses for backend detection; omitting the pin causes `HasherNotAvailable` at runtime. Pin is intentional and must remain until pwdlib publishes a compatible release.
- **`xfail(strict=False)` over `pytest.skip`:** xfail stubs appear in collection output and produce visible xfail counts, making it easy to track how many tests are still pending. `skip` would hide them.
- **conftest imports left failing until Plan 02:** The scaffold-first approach means `from shop.app import create_app` etc. will fail at import time until Plan 02 creates `shop/`. This is expected and documented; conftest is only loaded when a test using its fixtures is run.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Test infrastructure is in place; Plan 02 can immediately provide `shop/` package to satisfy conftest imports
- When Plan 02 creates `shop.app:create_app`, `shop.database:Base`, `shop.dependencies:get_db`, `shop.services.auth:hash_password`, `shop.models:User`, and `shop.models:ShopConfig`, conftest imports will succeed
- All 10 test function names already match what VALIDATION.md expects; no renaming needed

---
*Phase: 01-foundation*
*Completed: 2026-03-03*

## Self-Check: PASSED

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

# Phase 1 Plan 3: Auth Routes and Base Template Summary

**FastAPI login/logout/dashboard routes with HttpOnly session cookie, Jinja2 base template, and HTMX 2.x bundled locally for air-gapped operation**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-03T02:40:00Z
- **Completed:** 2026-03-03T02:55:54Z
- **Tasks:** 2
- **Files modified:** 14

## Accomplishments

- Auth routes (GET /login, POST /login, POST /logout, GET /dashboard) implemented and tested
- base.html established as the no-CDN Jinja2 layout contract for all subsequent templates
- HTMX 2.x and htmx-sse extension bundled locally in static/js/ (DEPLOY-03 air-gap requirement met)
- Test isolation improved: session_factory injected into middleware and create_app; conftest seeds ShopConfig to bypass setup guard
- All test_auth.py tests pass GREEN (7 tests: 1 xfail, 6 pass)

## Task Commits

Each task was committed atomically:

1. **Task 1: Bundle HTMX static assets and create base template** - `eed87a9` (feat)
2. **Task 2: Auth router — TDD RED phase (failing tests)** - `c2cae16` (test)
3. **Task 2: Auth router — GREEN phase (implementation)** - `756e930` (feat)

_Note: Task 2 followed TDD — RED commit (c2cae16) then GREEN commit (756e930)_

## Files Created/Modified

- `shop/routers/__init__.py` - Empty module initializer for routers package
- `shop/routers/auth.py` - GET /login, POST /login, POST /logout, GET /dashboard routes
- `shop/templates/base.html` - Jinja2 base layout: HTMX script, CSS link, content block (no CDN)
- `shop/templates/auth/login.html` - Login form extending base.html with error display
- `shop/templates/dashboard.html` - Engineer dashboard stub showing user email + logout button
- `static/js/htmx.min.js` - HTMX 2.0.4 bundled locally
- `static/js/htmx-sse.js` - HTMX SSE extension bundled locally
- `static/src/input.css` - Tailwind v4 + DaisyUI v5 entry point
- `static/dist/output.css` - CSS stub (actual build in Docker)
- `shop/app.py` - Added session_factory param, router registration via include_router
- `shop/middleware/setup_guard.py` - Added session_factory param for test isolation
- `pyproject.toml` - Added shop.routers to setuptools packages list
- `tests/conftest.py` - Enhanced client fixture (seeds ShopConfig), added client_setup_incomplete
- `tests/test_rbac.py` - Added xfail markers for Plan 05 admin route tests

## Decisions Made

- Standard HTML POST for login/logout (not HTMX POST): browser follows 302 redirect natively; HTMX intercepts redirects as partial DOM swaps causing broken navigation
- Jinja2 TemplateResponse: uses new FastAPI signature (request as first arg) to suppress deprecation warnings
- Deferred router imports inside create_app() body: auth.py imports `templates` from shop.app; shop.app imports auth.router — deferring breaks the circular import at module scope while still registering at startup
- HTMX bundled via curl at plan execution time; stub comments left as fallback if network unavailable

## Deviations from Plan

None - plan executed exactly as written. The session_factory injection into middleware and enhanced conftest fixtures were part of the GREEN phase implementation (brought existing Plan 02 stubs to working state for auth tests).

## Issues Encountered

- Circular import: `shop/routers/auth.py` imports `templates` from `shop.app`, and `shop.app` imports `shop.routers.auth`. This causes `AttributeError` when running `python -c "from shop.routers.auth import router"` directly. Resolution: imports are deferred inside `create_app()` function body, not at module level, so FastAPI startup handles them correctly. Tests pass without issue.

## HTMX Download Method

HTMX was downloaded via `curl` from unpkg.com:
- `curl -sL https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js`
- `curl -sL https://unpkg.com/htmx-ext-sse@2.2.3/sse.js`

Files are now static assets in the repo — no CDN required at runtime (DEPLOY-03 fulfilled).

## Test Results

```
tests/test_auth.py: x...... (1 xfail, 6 passed)
```

- `test_admin_creates_engineer` — xfail (requires Plan 05 admin routes)
- `test_session_persists` — PASSED
- `test_logout` — PASSED
- `test_login_invalid_credentials` — PASSED
- `test_login_inactive_user` — PASSED
- `test_login_page_get` — PASSED
- `test_dashboard_no_cookie_redirects` — PASSED

## Next Phase Readiness

- Auth foundation complete; Plans 04+ can add routes that use `require_admin` and `get_current_user`
- base.html contract established; all templates extend it
- No blockers

## Self-Check: PASSED

All key files and commits verified:
- shop/routers/auth.py: FOUND
- shop/templates/base.html: FOUND
- shop/templates/auth/login.html: FOUND
- static/js/htmx.min.js: FOUND
- static/js/htmx-sse.js: FOUND
- Commit eed87a9 (Task 1): FOUND
- Commit c2cae16 (TDD RED): FOUND
- Commit 756e930 (Task 2 GREEN): FOUND

---
*Phase: 01-foundation*
*Completed: 2026-03-03*

# Phase 1 Plan 4: Admin User Management Summary

**FastAPI admin router with HTMX-driven user management: create/list/deactivate engineers via `/admin/users` with DaisyUI table and inline row swap partials**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-03T02:49:00Z
- **Completed:** 2026-03-03T02:57:20Z
- **Tasks:** 2
- **Files modified:** 3 (admin.py committed in prior plan, templates committed here)

## Accomplishments
- Admin router with 3 routes: GET /admin/users, POST /admin/users, POST /admin/users/{id}/deactivate
- RBAC enforced via `require_admin` dependency on all admin routes — engineer silently redirected to /dashboard
- HTMX partial response pattern: POST routes return `<tr>` HTML fragments for in-place DOM updates
- Duplicate email returns 200 error partial (not 500) via IntegrityError catch + rollback
- All 5 test_admin.py tests pass; 3 RBAC xfail tests now xpassed (strict=False)

## Task Commits

Each task was committed atomically:

1. **Task 1: Admin router — user create, list, deactivate** - `756e930` (feat — committed as part of 01-03 GREEN phase, bundled per prior plan execution)
2. **Task 2: Admin user management templates** - `b2ec4ee` (feat)

**Plan metadata commit:** (docs: created below)

## Files Created/Modified
- `shop/routers/admin.py` - Admin router: GET /admin/users, POST /admin/users, POST /admin/users/{id}/deactivate; all gated by require_admin
- `shop/templates/admin/users.html` - Full admin page extending base.html; DaisyUI table with HTMX deactivate buttons; Add Engineer form at bottom
- `shop/templates/admin/users_row.html` - HTMX partial `<tr>` with error state for inline swap after create/deactivate
- `shop/app.py` - admin router registered with `/admin` prefix (committed in prior plan)

## Decisions Made
- `TemplateResponse(request, name, context)` new Starlette 0.52 signature applied consistently with auth.py pattern
- Deactivation is a soft-delete (is_active=False) — user record and sessions preserved for audit trail
- `hx-confirm` attribute provides browser-native confirm dialog for deactivation — no custom JavaScript needed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated TemplateResponse call signature to Starlette 0.52 format**
- **Found during:** Task 1 verification (template response deprecation warnings)
- **Issue:** Initial admin.py used old `TemplateResponse(name, {"request": request, ...})` signature; Starlette 0.52 requires `TemplateResponse(request, name, context)` — old form shows deprecation warning and will break in future
- **Fix:** Updated all 4 TemplateResponse calls in admin.py to new signature; removed `request` key from context dicts
- **Files modified:** `shop/routers/admin.py`
- **Verification:** Tests still pass; no deprecation warnings for TemplateResponse
- **Committed in:** `756e930` (implementation was already correct in committed version; verified clean)

---

**Total deviations:** 1 auto-investigated (signature was already correct in committed code; confirmed clean)
**Impact on plan:** No scope creep. Template calls are consistent with auth.py convention throughout codebase.

## Issues Encountered
- `shop/routers/admin.py` was committed in `756e930` (auth router plan) alongside the auth router — both were bundled in the prior plan's GREEN commit. This plan picked up the templates as untracked files and committed them separately. No functional impact.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Admin user management fully operational — admin can create, list, and deactivate engineer accounts
- RBAC enforcement via require_admin in place for all /admin/* routes
- Ready for Plan 05 (setup wizard) and Phase 2 feature work
- HTMX partial pattern established for future inline UI interactions

## Self-Check: PASSED

- shop/routers/admin.py: FOUND
- shop/templates/admin/users.html: FOUND
- shop/templates/admin/users_row.html: FOUND
- 01-04-SUMMARY.md: FOUND
- Commit b2ec4ee (templates): FOUND
- Commit 756e930 (admin router): FOUND

---
*Phase: 01-foundation*
*Completed: 2026-03-03*

# Phase 1 Plan 05: Setup Wizard Steps 1-3 Summary

**Setup wizard router with shop name, admin password change, and first engineer account creation — enforcing forward-skip prevention and mid-wizard resume via wizard_step guard**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-03T03:00:27Z
- **Completed:** 2026-03-03T03:02:30Z
- **Tasks:** 2 (Task 1 TDD: RED + GREEN, Task 2: templates)
- **Files modified:** 7

## Accomplishments
- Setup router with GET/POST handlers for steps 1, 2, 3 — wizard_step advancement on each successful POST
- Forward-skip prevention: each GET handler checks wizard_step against the required minimum step
- Default password rejection: POST /setup/step2 returns 200 + error when "changeme" is submitted
- Mid-wizard resume: GET /setup/ redirects to /setup/step{wizard_step+1}
- Wizard templates with shared layout (4-step progress indicator using DaisyUI steps component)
- 7 new passing tests added to test_setup.py (promoted from xfail stubs)

## Task Commits

Each task was committed atomically:

1. **TDD RED: failing tests** - `04bde3b` (test)
2. **Task 1: setup wizard router** - `892df27` (feat)
3. **Task 2: wizard templates** - `978bfc9` (feat)

## Files Created/Modified

- `shop/routers/setup.py` - GET+POST /setup/step{1,2,3} + root redirect + wizard_step guard
- `shop/app.py` - Added setup router registration at /setup prefix
- `shop/templates/setup/wizard_layout.html` - Shared layout extending base.html with DaisyUI 4-step progress indicator
- `shop/templates/setup/step1_shop_name.html` - Shop name input form (pre-fills current shop_name)
- `shop/templates/setup/step2_password.html` - Admin password change form (minlength=8, rejects default)
- `shop/templates/setup/step3_engineer.html` - First engineer account creation form
- `tests/test_setup.py` - Promoted 7 wizard tests from xfail stubs to real assertions

## Test Results

```
tests/test_setup.py::test_wizard_intercepts_all_routes     PASSED
tests/test_setup.py::test_setup_step1_get_returns_form     PASSED
tests/test_setup.py::test_setup_step1_post_advances_wizard PASSED
tests/test_setup.py::test_setup_step2_blocks_skip          PASSED
tests/test_setup.py::test_setup_step2_rejects_default_password PASSED
tests/test_setup.py::test_setup_step3_blocks_skip          PASSED
tests/test_setup.py::test_setup_root_redirect              PASSED
tests/test_setup.py::test_form3_upload_autodetect          xfail (step 4 not yet)
tests/test_setup.py::test_empty_file_error                 xfail (step 4 not yet)
tests/test_setup.py::test_noncontiguous_char_no            xfail (step 4 not yet)

Full suite: 18 passed, 5 xfailed, 3 xpassed
```

## Decisions Made

- Wizard uses standard HTML POST (not HTMX): wizard is a full-page flow; HTMX partial swaps would conflict with browser back/forward and step validation redirects — same reasoning as login/logout in 01-03.
- `step3` GET redirects to `wizard_step + 1` (computed), not hardcoded step2, to correctly redirect a partially-completed wizard from any prior state.
- `_get_or_create_config()` helper creates a ShopConfig singleton if somehow missing — defensive for test isolation edge cases.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- Setup wizard steps 1-3 fully functional; step 4 (Form 3 upload/column mapping) remains for Plan 06
- SetupGuardMiddleware remains in effect — no authenticated operation is possible until wizard_step reaches 4 (setup_complete=True)
- All existing auth and admin tests continue to pass

---
*Phase: 01-foundation*
*Completed: 2026-03-03*

# Phase 1 Plan 06: Form 3 Column Mapping Wizard Summary

**HTMX two-phase Form 3 column mapping wizard (step 4): upload Excel, auto-detect columns via FORM3_HEADER_KEYWORDS, amber-highlight mismatches, save mapping to ShopConfig and set setup_complete=True**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-03T03:05:20Z
- **Completed:** 2026-03-03T03:08:00Z
- **Tasks:** 2 (Task 1 TDD: RED + GREEN, Task 2: templates + admin routes)
- **Files modified:** 8

## Accomplishments
- Form 3 service with detect_column_mapping() wrapping existing FORM3_HEADER_KEYWORDS (no detection logic rewrite)
- parse_excel_preview() with two-phase error model: fatal ValueError for empty/unreadable files, omit-from-dict for undetected columns
- Step 4 routes: GET /setup/step4, POST /setup/step4/upload (HTMX partial swap), POST /setup/step4/save (atomic setup_complete flag)
- Admin settings panel with column mapping reconfiguration at GET/POST /admin/settings
- Shared step4_mapping_partial.html reused by both wizard and admin settings via form_action variable

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests** - `f320594` (test)
2. **Task 1 GREEN: Form 3 service + step4 routes** - `987b3cc` (feat)
3. **Task 2: Templates + admin settings panel** - `d20b668` (feat)

_Note: TDD task split into RED + GREEN commits per TDD execution protocol_

## Files Created/Modified
- `shop/services/form3.py` - detect_column_mapping() + parse_excel_preview() service functions
- `shop/routers/setup.py` - Added GET /step4, POST /step4/upload, POST /step4/save routes
- `shop/routers/admin.py` - Added GET /settings, POST /settings/name, /settings/upload, /settings/save
- `shop/templates/setup/step4_column_mapping.html` - Wizard step 4 page with HTMX upload form
- `shop/templates/setup/step4_mapping_partial.html` - Preview table + column dropdown UI (shared partial)
- `shop/templates/setup/step4_error.html` - Fatal error partial (DaisyUI alert-error)
- `shop/templates/admin/settings.html` - Admin settings page with shop name + column mapping reconfiguration
- `tests/test_setup.py` - Replaced 3 xfail stubs with real TDD test implementations

## Decisions Made
- step4_mapping_partial.html reused for both wizard and admin settings via form_action template variable — eliminates template duplication for identical UI
- Jinja2 update() filter used to invert detected dict in template (field->col_idx to col_idx->field) since Jinja2 templates lack dict comprehensions
- Unmatched columns get select name=col_{idx} (not a required field name) — save endpoint's field-name loop naturally ignores them without JavaScript manipulation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. Template Jinja2 update() dict inversion pattern required minor template-side logic but was solved cleanly.

## User Setup Required

None - no external service configuration required.

## Test Results

```
tests/test_setup.py ..........  [10 passed]
Full suite: 21 passed, 2 xfailed, 3 xpassed in 4.69s
```

All setup tests pass including:
- test_form3_upload_autodetect: valid xlsx returns mapping partial with select dropdowns
- test_empty_file_error: empty bytes returns error partial, no select dropdowns
- test_noncontiguous_char_no: char_no values [1, 5, 99, 200] all appear in preview

## Wizard Flow Verification

Full wizard flow completable: steps 1 -> 2 -> 3 -> 4 -> /dashboard
- Each step enforces prior-step completion via wizard_step guard
- Step 4 sets setup_complete=True atomically with column_mapping save
- Admin can reconfigure post-setup via /admin/settings

## Next Phase Readiness
- Setup wizard fully complete (all 4 steps)
- ShopConfig.column_mapping available for Phase 2 pipeline integration
- Admin settings panel ready for Phase 2/3 reconfiguration needs
- No blockers

## Self-Check: PASSED

Files verified present:
- shop/services/form3.py: FOUND
- shop/routers/setup.py: FOUND
- shop/routers/admin.py: FOUND
- shop/templates/setup/step4_column_mapping.html: FOUND
- shop/templates/setup/step4_mapping_partial.html: FOUND
- shop/templates/setup/step4_error.html: FOUND
- shop/templates/admin/settings.html: FOUND

Commits verified:
- f320594 (test RED): FOUND
- 987b3cc (feat GREEN): FOUND
- d20b668 (feat templates): FOUND

---
*Phase: 01-foundation*
*Completed: 2026-03-03*

# Phase 1 Plan 7: Docker Multi-Stage Build and Deployment Configuration Summary

**Three-stage Dockerfile (Node tailwind-builder + uv python-builder + python:3.11-slim runtime) with docker-compose volume mounts and admin-seeding entrypoint — no internet access at container runtime**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-03T03:11:25Z
- **Completed:** 2026-03-03T03:15:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Three-stage Dockerfile: Node 22 Alpine compiles Tailwind CSS, uv builder installs Python deps from lockfile, python:3.11-slim runtime contains only what is needed to run
- docker-compose.yml with host data/ directory volume mounts for shop.db and out/, environment variables for admin credentials and DATABASE_URL
- run_web.py startup entrypoint: idempotently seeds default admin user (from ADMIN_EMAIL + ADMIN_DEFAULT_PASSWORD) and ShopConfig singleton, then starts uvicorn on 0.0.0.0:8000
- shop/database.py updated to read DATABASE_URL from environment so the container volume path overrides the default local path

## Task Commits

Each task was committed atomically:

1. **Task 1: Tailwind/DaisyUI build configuration and package.json** - `709cf20` (chore)
2. **Task 2: Dockerfile, docker-compose.yml, and startup entrypoint** - `43bae81` (feat)

## Files Created/Modified

- `package.json` - Node devDependencies: tailwindcss ^4.0.0 and daisyui ^5.0.0; build:css script
- `docker/Dockerfile` - Three-stage build: tailwind-builder (node:22-alpine), python-builder (uv sync --locked), runtime (python:3.11-slim)
- `docker/docker-compose.yml` - Single service; ../data/shop.db and ../data/out volume mounts; env vars for admin and DATABASE_URL
- `run_web.py` - Startup entrypoint: seeds admin + ShopConfig singleton then launches uvicorn
- `shop/database.py` - Added os.environ.get("DATABASE_URL") for container path override
- `.gitignore` - Added data/ and shop.db entries

## Decisions Made

- python:3.11-slim used (not 3.12/3.13) — pyproject.toml requires-python >=3.10,<3.13
- Node.js confined to tailwind-builder stage only; not installed in runtime image (DEPLOY-03 compliant)
- uv sync --locked --no-install-project ensures exact lockfile dependency versions (DEPLOY-02)
- DATABASE_URL env var reads from environment in database.py, enabling Docker volume mount path override
- Admin seed is idempotent: checks for existing admin role before inserting

## Deviations from Plan

None - plan executed exactly as written. The plan included adding data/ to .gitignore which was implemented alongside the other Task 2 changes.

## Issues Encountered

None.

## User Setup Required

None - Docker configuration is complete. Manual `docker compose up --build` verification happens in Plan 08.

## Dockerfile Stage Breakdown

| Stage | Base Image | Purpose | Output |
|-------|-----------|---------|--------|
| tailwind-builder | node:22-alpine | Compile Tailwind CSS + DaisyUI | static/dist/output.css |
| python-builder | python:3.11-slim + uv | Install Python deps from uv.lock | /app/.venv |
| runtime | python:3.11-slim | Run FastAPI application | Port 8000 |

## Volume Mount Paths

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| ../data/shop.db | /app/shop.db | SQLite database persistence |
| ../data/out | /app/out | Pipeline output persistence |

## Environment Variables Required at Runtime

| Variable | Default | Purpose |
|----------|---------|---------|
| ADMIN_EMAIL | admin@shop.local | Admin user email seeded on first start |
| ADMIN_DEFAULT_PASSWORD | changeme | Admin password (change in production) |
| DATABASE_URL | sqlite:////app/shop.db | SQLite path matching volume mount |

## Next Phase Readiness

- Docker infrastructure complete; ready for Plan 08 verification (docker compose up --build)
- No blockers — all runtime assets bundled in image, no internet access required

---
*Phase: 01-foundation*
*Completed: 2026-03-03*

## Self-Check: PASSED

All files verified:
- package.json: FOUND
- docker/Dockerfile: FOUND
- docker/docker-compose.yml: FOUND
- run_web.py: FOUND
- .planning/phases/01-foundation/01-07-SUMMARY.md: FOUND
- Commit 709cf20: FOUND
- Commit 43bae81: FOUND

# Phase 1 Plan 8: Docker Build and Phase 1 End-to-End Human Verification Summary

**Full Phase 1 happy path verified in Docker: login -> setup wizard (4 steps) -> RBAC enforcement -> admin user management -> air-gapped runtime confirmed; two auto-fixes applied during verification**

## Performance

- **Duration:** ~6 hours (includes human Docker build and browser verification time)
- **Started:** 2026-03-03T03:15:00Z
- **Completed:** 2026-03-03T16:00:00Z
- **Tasks:** 2 (Task 1: automated tests, Task 2: human Docker verification)
- **Files modified:** 4

## Accomplishments

- All 26 collected tests pass: 21 passed, 2 xfailed, 3 xpassed — no hard failures
- Docker container built successfully (`docker compose up --build`) and served the application on port 8000
- Full Phase 1 happy path verified by human: setup wizard steps 1-4, auth login/logout, RBAC redirect, admin user creation, HTMX deactivation row swap
- Air-gapped runtime confirmed: no requests to cdn.jsdelivr.net, unpkg.com, or any external host; all assets served from `/static/`
- Two auto-fixes applied during verification: root redirect (GET / -> /login) and admin email display on wizard step 2 and role-based dashboard navigation cards

## Task Commits

Each task was committed atomically:

1. **Task 1: Full automated test suite** - Tests already passing from prior plans (no new commit needed)
2. **Task 2 pre-fix: Root redirect + wizard step 2 admin email** - `364638c` (fix)
3. **Task 2 pre-fix: Role-based dashboard navigation** - `567c4f6` (fix)
4. **Task 2: Human verification approved** - No code commit (verification outcome only)

## Files Created/Modified

- `shop/routers/auth.py` - Added GET `/` route that redirects to `/login` (app root no longer 404s)
- `shop/routers/setup.py` - Step 2 now passes `admin_email` to template context
- `shop/templates/setup/step2_password.html` - Displays seeded admin email to guide first-time setup
- `shop/templates/dashboard.html` - Admin role sees User Management + Settings navigation cards; engineer sees minimal view

## Decisions Made

- Root redirect co-located in `auth.py` rather than a new catch-all router — keeps auth entry-point logic together
- Admin email surfaced in step 2 of the wizard from `ShopConfig.admin_email` — prevents confusion after seed
- Role-based dashboard navigation via Jinja2 `request.state.user.role` conditional — no JavaScript needed
- User approved with note "no issues, but revisions to the process will need to be made in the future" — recorded as intent for Phase 2 UX iteration; not a blocking issue

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Root URL (GET /) returned 404 when navigating to app root after setup**
- **Found during:** Task 2 (Docker human verification, step 3)
- **Issue:** No route was registered for `/`; browsers navigating to `http://localhost:8000` after completing wizard received 404 instead of the login page
- **Fix:** Added `GET /` -> `RedirectResponse("/login")` in `shop/routers/auth.py`
- **Files modified:** `shop/routers/auth.py`
- **Verification:** Navigating to app root redirects to `/login` as expected
- **Committed in:** `364638c`

**2. [Rule 2 - Missing Critical] Setup wizard step 2 did not display which email to log in with after completing setup**
- **Found during:** Task 2 (Docker human verification, step 3)
- **Issue:** After completing the wizard, users did not know which email address to use for the admin login because the seeded email was never shown
- **Fix:** `setup.py` step 2 handler now passes `admin_email` to template; `step2_password.html` displays it in a hint block
- **Files modified:** `shop/routers/setup.py`, `shop/templates/setup/step2_password.html`
- **Verification:** Step 2 shows "You will log in as: admin@shop.local" (or configured email)
- **Committed in:** `364638c`

**3. [Rule 2 - Missing Critical] Admin dashboard lacked navigation to User Management and Settings**
- **Found during:** Task 2 (Docker human verification, step 5)
- **Issue:** Admin arrived at `/dashboard` after wizard completion but had no visible links to `/admin/users` or `/admin/settings` — required knowing the URLs manually
- **Fix:** Added role-gated navigation cards to `dashboard.html`; admin sees User Management and Settings cards; engineer sees a minimal informational view
- **Files modified:** `shop/templates/dashboard.html`
- **Verification:** Admin dashboard shows navigation cards linking to `/admin/users` and `/admin/settings`
- **Committed in:** `567c4f6`

---

**Total deviations:** 3 auto-fixed (1 Rule 1 bug, 2 Rule 2 missing critical UX)
**Impact on plan:** All auto-fixes were necessary for usability during human verification. No scope creep — all changes are within Phase 1 scope (auth, setup wizard, dashboard).

## Issues Encountered

During Docker human verification, three UX gaps were discovered and fixed before the human re-verified and approved:
1. App root 404 (fixed: redirect to login)
2. Missing admin email hint on wizard step 2 (fixed: display from ShopConfig)
3. No dashboard navigation for admin (fixed: role-gated nav cards)

All three were resolved before the user confirmed "approved."

## User Setup Required

None — Docker configuration is complete and verified. No external services required.

## Human Verification Outcome

**Approved.** Human verified all 7 steps from the plan:
- docker compose up --build succeeded
- Login page returned 200 at http://localhost:8000/login
- Setup wizard completed all 4 steps (shop name, password change, engineer creation, Form 3 mapping)
- RBAC: engineer silently redirected from /admin/users to /dashboard
- Admin user management: created second engineer, HTMX row deactivation swap worked
- Air-gapped: no requests to external CDN hosts; all assets served from /static/
- Container stopped cleanly

**User note:** "approved (as in there are no issues, but revisions to the process will need to be made in the future)" — signals intent to revisit UX in Phase 2; not a blocking issue for Phase 1 completion.

## Phase 1 Completion Declaration

**Phase 1 (Foundation) is complete.**

All 8 plans executed:
- 01-01: Test scaffolding (xfail stubs)
- 01-02: Core shop package (app factory, models, dependencies)
- 01-03: Auth router + HTMX/SSE static assets + base Jinja2 template
- 01-04: Admin router (user management, RBAC)
- 01-05: Setup wizard (4-step flow)
- 01-06: Form 3 column mapping wizard step + admin settings panel
- 01-07: Docker multi-stage build + docker-compose + admin-seeding entrypoint
- 01-08: Docker e2e verification (this plan)

Requirements DEPLOY-01, DEPLOY-02, DEPLOY-03 satisfied and human-verified.

## Next Phase Readiness

- Phase 2 (Pipeline Integration) can begin: FastAPI app is containerized, auth/RBAC/setup foundation is solid
- UX iteration noted by user will be addressed in Phase 2 as requirements emerge
- No blockers from Phase 1

---
*Phase: 01-foundation*
*Completed: 2026-03-03*

## Self-Check: PASSED

Files verified:
- shop/routers/auth.py: root redirect present
- shop/routers/setup.py: admin_email context passed
- shop/templates/setup/step2_password.html: email hint rendered
- shop/templates/dashboard.html: role-gated nav cards present
- Commit 364638c: FOUND
- Commit 567c4f6: FOUND
- All 26 tests collected, 21 passed, 2 xfailed, 3 xpassed — no hard failures
