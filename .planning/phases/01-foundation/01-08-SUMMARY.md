---
phase: 01-foundation
plan: 08
subsystem: infra
tags: [docker, verification, e2e, rbac, auth, setup-wizard, air-gapped]

# Dependency graph
requires:
  - phase: 01-foundation-01-07
    provides: Docker multi-stage build, docker-compose.yml, run_web.py admin-seeding entrypoint

provides:
  - Human-verified Docker build and full Phase 1 happy path confirmation
  - Root redirect GET / -> /login so navigating to app root after setup returns login page (not 404)
  - Setup wizard step 2 displays seeded admin email to guide first-time users
  - Role-based dashboard navigation: admin sees User Management + Settings cards; engineer sees minimal view
  - All 21 automated tests passing (2 xfail, 3 xpassed)

affects: [phase-02, deploy]

# Tech tracking
tech-stack:
  added: []
  patterns: [human-in-the-loop verification gate before phase completion, role-based dashboard navigation]

key-files:
  created: []
  modified:
    - shop/routers/auth.py
    - shop/routers/setup.py
    - shop/templates/setup/step2_password.html
    - shop/templates/dashboard.html

key-decisions:
  - "Root redirect added to auth router (GET / -> /login) rather than a separate catch-all to keep routing logic co-located with auth"
  - "Admin email surfaced on step 2 of wizard so users know credentials after seeding — avoids support confusion on first run"
  - "Role-based dashboard navigation via Jinja2 conditional on request.state.user.role — no JavaScript needed"
  - "Verification gate confirmed: no outbound internet requests at runtime (all JS/CSS served from /static/)"
  - "User note: approved with future process revision intent — indicates possible UX iteration in Phase 2 but no blocking issues"

patterns-established:
  - "Human verification pattern: automated tests pass first, then Docker e2e verification, then phase marked complete"
  - "Role-gated template content: {{ request.state.user.role == 'admin' }} conditionals in Jinja2 templates"

requirements-completed: [DEPLOY-01, DEPLOY-02, DEPLOY-03]

# Metrics
duration: ~6h (includes human Docker build and verification time)
completed: 2026-03-03
---

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
