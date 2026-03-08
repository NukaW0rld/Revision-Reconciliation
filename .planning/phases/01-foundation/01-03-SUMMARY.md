---
phase: 01-foundation
plan: "03"
subsystem: auth
tags: [fastapi, jinja2, htmx, tailwind, daisyui, sqlite, session-cookie]

# Dependency graph
requires:
  - phase: 01-02
    provides: shop/services/auth.py (verify_password, create_session, delete_session), shop/dependencies.py (get_db, get_current_user), shop/models.py (User, UserSession)
provides:
  - GET /login — login form rendered via Jinja2 base template
  - POST /login — validates credentials, sets HttpOnly session_token cookie, redirects to /dashboard
  - POST /logout — deletes session row, clears cookie, redirects to /login
  - GET /dashboard — gated by get_current_user dependency; shows user email and role
  - shop/templates/base.html — HTMX + Tailwind CSS base layout (no CDN links)
  - static/js/htmx.min.js and htmx-sse.js — bundled locally for air-gapped operation
affects:
  - 01-04 (admin routes extend base.html, import auth patterns)
  - 01-05 (setup wizard extends base.html)
  - all subsequent UI plans (base.html is the layout contract)

# Tech tracking
tech-stack:
  added: [htmx 2.x (local bundle), htmx-sse extension, tailwindcss v4 (stub), daisyui v5 (stub)]
  patterns:
    - Standard HTML POST for login/logout (not HTMX) — browser handles 302 redirect natively; HTMX intercepts redirect as partial swap
    - Jinja2 TemplateResponse using new FastAPI signature (request as first positional arg)
    - session_factory parameter on create_app and SetupGuardMiddleware for test isolation
    - Router registration via include_router inside create_app to avoid circular import at module level

key-files:
  created:
    - shop/routers/__init__.py
    - shop/routers/auth.py
    - shop/templates/auth/login.html
    - shop/templates/base.html
    - shop/templates/dashboard.html
    - static/js/htmx.min.js
    - static/js/htmx-sse.js
    - static/src/input.css
    - static/dist/output.css
  modified:
    - shop/app.py (router registration, session_factory support)
    - shop/middleware/setup_guard.py (session_factory for test isolation)
    - pyproject.toml (added shop.routers to packages)
    - tests/conftest.py (client fixture seeds ShopConfig, adds client_setup_incomplete)
    - tests/test_rbac.py (xfail markers for Plan 05 admin routes)

key-decisions:
  - "Standard HTML POST (not HTMX POST) for login/logout — browser natively follows 302; HTMX intercepts redirect as partial swap causing broken navigation"
  - "Jinja2 TemplateResponse uses request as first positional arg (new FastAPI >= 0.111 API) to avoid deprecation warnings"
  - "Router imports deferred inside create_app() body to avoid circular import: auth.py imports templates from shop.app, shop.app imports auth.router"
  - "HTMX downloaded via curl from unpkg at plan time; stub fallback comment if air-gapped at plan execution"

patterns-established:
  - "Pattern 1: Auth flow — POST /login sets HttpOnly cookie then RedirectResponse(302); GET /dashboard depends on get_current_user"
  - "Pattern 2: Test isolation — session_factory injected into create_app and middleware; conftest seeds ShopConfig row to bypass setup guard"
  - "Pattern 3: All templates extend base.html; no CDN links; all assets served from /static/"

requirements-completed: [AUTH-02, AUTH-03, DEPLOY-03]

# Metrics
duration: 15min
completed: "2026-03-03"
---

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
