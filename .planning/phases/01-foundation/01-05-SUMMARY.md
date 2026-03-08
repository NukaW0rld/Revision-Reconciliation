---
phase: 01-foundation
plan: 05
subsystem: ui
tags: [fastapi, jinja2, htmx, daisyui, wizard, setup, sqlalchemy]

# Dependency graph
requires:
  - phase: 01-02
    provides: ShopConfig model, User model, app factory, SetupGuardMiddleware, hash_password
  - phase: 01-03
    provides: base.html template, Jinja2 templates object, auth router pattern
  - phase: 01-04
    provides: admin router pattern (DaisyUI + TemplateResponse with new Starlette 0.52 signature)

provides:
  - Setup wizard router at /setup prefix with GET+POST for steps 1-3
  - Wizard step ordering enforcement (no forward-skip)
  - Mid-wizard resume via GET /setup/ root redirect
  - wizard_layout.html shared layout with 4-step progress indicator
  - step1_shop_name.html, step2_password.html, step3_engineer.html templates

affects: [01-06, Phase 2, SETUP-01, AUTH-01]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Wizard step guard: each GET handler checks wizard_step < N-1 before rendering
    - _get_or_create_config helper creates ShopConfig singleton if missing
    - Standard HTML POST for wizard (not HTMX) to preserve browser back/forward semantics

key-files:
  created:
    - shop/routers/setup.py
    - shop/templates/setup/wizard_layout.html
    - shop/templates/setup/step1_shop_name.html
    - shop/templates/setup/step2_password.html
    - shop/templates/setup/step3_engineer.html
  modified:
    - shop/app.py
    - tests/test_setup.py

key-decisions:
  - "setup router registered at /setup prefix via include_router in create_app()"
  - "Wizard templates use standard HTML POST (not HTMX): wizard is full-page flow; HTMX partial swaps conflict with browser back/forward and step validation redirects"
  - "step3 GET redirects to wizard_step+1 (not always step2) to correctly resume mid-wizard from any prior incomplete state"

patterns-established:
  - "Wizard step guard pattern: if config.wizard_step < N-1: redirect to first incomplete step"
  - "_get_or_create_config() helper for ShopConfig singleton safety"
  - "TemplateResponse uses new Starlette 0.52 signature (request first, template name second) matching admin.py convention"

requirements-completed: [SETUP-01, AUTH-01]

# Metrics
duration: 2min
completed: 2026-03-03
---

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
