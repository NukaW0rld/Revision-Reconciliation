---
phase: 02-pipeline-bridge
plan: 04
subsystem: ui
tags: [fastapi, jinja2, daisyui, htmx, templates, nav, dashboard, runs]

# Dependency graph
requires:
  - phase: 02-pipeline-bridge
    plan: 01
    provides: "Run and RunAlert SQLAlchemy models, SqliteHuey task queue"
  - phase: 01-foundation
    provides: "User model, auth router, dashboard route, base.html"
provides:
  - "Persistent DaisyUI navbar in base.html (shop name, New Run, My Runs, alert bell, user email, logout)"
  - "Dashboard shows Submit New Run CTA + up to 10 recent run cards with status badges"
  - "GET /runs route returning filterable run list at /runs"
  - "shop/templates/runs/list.html with part_number filter form and zebra table"
  - "status_badge_class Jinja2 filter registered in app.py"
  - "Nav bar suppressed on login and setup wizard pages"
affects:
  - 02-05 (upload form at /runs/new — nav bar inherited)
  - 02-06 (run detail page — nav bar inherited)
  - 02-07 (alerts page — alert bell in nav shows unread count)
  - all future templates extending base.html

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Nav context pattern: _get_nav_context() returns unread_alert_count + shop_name for authenticated routes"
    - "{% block nav %}{% endblock %} override pattern to suppress nav on pre-auth pages (login, setup wizard)"
    - "status_badge_class Jinja2 filter at module level in app.py — available to all templates"
    - "APIRouter(redirect_slashes=False) for /runs prefix so /runs resolves without trailing slash redirect"

key-files:
  created:
    - shop/routers/runs.py
    - shop/templates/runs/list.html
  modified:
    - shop/templates/base.html
    - shop/templates/dashboard.html
    - shop/templates/auth/login.html
    - shop/templates/setup/wizard_layout.html
    - shop/routers/auth.py
    - shop/app.py

key-decisions:
  - "redirect_slashes=False on APIRouter for /runs prefix — FastAPI default redirects /runs -> /runs/ (307), breaking nav links; solved by empty string route path and redirect_slashes=False"
  - "Nav bar context (_get_nav_context) centralised helper in runs.py — avoids repeating unread count + shop_name query in every route handler"
  - "login.html and wizard_layout.html override {% block nav %} with empty block — no user context available on those pages, nav bar would render empty/broken"

patterns-established:
  - "All authenticated routes must pass user, shop_name, unread_alert_count to templates extending base.html"
  - "Pre-auth templates (login, setup) override {% block nav %}{% endblock %} to suppress nav bar"

requirements-completed: [UPLOAD-01]

# Metrics
duration: 8min
completed: 2026-03-04
---

# Phase 2 Plan 4: Nav Bar and Dashboard Summary

**Persistent DaisyUI sticky navbar in base.html with shop name/New Run/My Runs/alert bell/logout, updated dashboard with Submit New Run CTA and recent run cards, plus GET /runs list page filterable by part number**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-04T01:30:00Z
- **Completed:** 2026-03-04T01:38:30Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Sticky DaisyUI navbar on every authenticated page — shop name links to /dashboard, New Run links to /runs/new, My Runs links to /runs, alert bell with red badge for unread count, user email displayed, logout button
- Dashboard updated with prominent "Submit New Run" CTA button (btn-primary btn-lg), recent runs section showing up to 10 run cards with part number, revision arrows, status badge, date, reviewer; empty state message when no runs
- New GET /runs route at `shop/routers/runs.py` with optional `part_number` query param filter and `runs/list.html` template with DaisyUI zebra table, filter form, and empty states
- `status_badge_class` Jinja2 filter registered globally in `app.py` — maps run status strings to DaisyUI badge color classes
- Nav bar suppressed on login page and setup wizard via `{% block nav %}{% endblock %}` override

## Task Commits

Each task was committed atomically:

1. **Task 1: Add persistent nav bar to base.html** - `433183b` (feat)
2. **Task 2: Update dashboard route + template, create GET /runs + runs/list.html** - `679c119` (feat)

## Files Created/Modified

- `shop/templates/base.html` - Added sticky DaisyUI navbar with conditional user section and block nav override point
- `shop/templates/auth/login.html` - Added `{% block nav %}{% endblock %}` to suppress nav bar
- `shop/templates/setup/wizard_layout.html` - Added `{% block nav %}{% endblock %}` to suppress nav bar
- `shop/routers/auth.py` - Dashboard route now queries recent_runs, unread_alert_count, shop_name; passes all to template
- `shop/templates/dashboard.html` - Submit New Run CTA, recent run cards grid, admin links retained; removed standalone logout button (now in nav)
- `shop/routers/runs.py` - Created: GET /runs (empty string path with redirect_slashes=False), _get_nav_context() helper
- `shop/templates/runs/list.html` - Created: filter form, zebra table with part/rev/status/date/reviewer, empty states
- `shop/app.py` - Added status_badge_class filter, registered runs router at /runs prefix

## Decisions Made

- **redirect_slashes=False:** FastAPI by default redirects `/runs` to `/runs/` with 307, which breaks browser navigation from the nav bar. Using `APIRouter(redirect_slashes=False)` and registering the route as `""` (empty string) instead of `"/"` resolves `/runs` directly without redirect.
- **Nav context helper:** A `_get_nav_context()` function in runs.py centralizes the unread alert count and shop_name queries so they don't have to be repeated in every route handler.
- **Block nav override pattern:** Pre-auth pages (login, setup wizard) override `{% block nav %}` with an empty block rather than passing `user=None` — cleaner than conditionally checking for None in every template.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] FastAPI trailing slash redirect for /runs**
- **Found during:** Task 2 (GET /runs route)
- **Issue:** `@router.get("/")` registered under prefix `/runs` causes FastAPI to redirect `GET /runs` → `GET /runs/` with 307, breaking the nav bar "My Runs" link
- **Fix:** Changed route path to empty string `""` and added `redirect_slashes=False` to `APIRouter()`
- **Files modified:** shop/routers/runs.py
- **Verification:** Integration test confirms `GET /runs` returns 200 directly
- **Committed in:** 679c119 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug: FastAPI trailing slash redirect)
**Impact on plan:** Fix required for correct nav bar behavior. No scope creep.

## Issues Encountered

- FastAPI default `redirect_slashes=True` behavior caused 307 redirect when accessing `/runs` without trailing slash — fixed as described in Deviations above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Nav bar is live on all authenticated pages; future templates extending base.html inherit it automatically
- Dashboard shows run cards — ready for real run data once Plan 05 (upload) creates Run records
- GET /runs is filterable; ready for additional filters in later plans
- status_badge_class filter available to all run-related templates

---
*Phase: 02-pipeline-bridge*
*Completed: 2026-03-04*

## Self-Check: PASSED

- FOUND: shop/templates/base.html
- FOUND: shop/templates/dashboard.html
- FOUND: shop/routers/runs.py
- FOUND: shop/templates/runs/list.html
- FOUND: .planning/phases/02-pipeline-bridge/02-04-SUMMARY.md
- FOUND: commit 433183b (Task 1)
- FOUND: commit 679c119 (Task 2)
