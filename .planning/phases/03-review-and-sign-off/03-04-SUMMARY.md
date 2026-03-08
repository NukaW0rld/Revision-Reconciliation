---
phase: 03-review-and-sign-off
plan: 04
subsystem: ui
tags: [fastapi, jinja2, daisyui, sign-off, reviewer-reassign, gate-enforcement]

# Dependency graph
requires:
  - phase: 03-03
    provides: approve/override item endpoints and HTMX card partial
  - phase: 03-02
    provides: review queue, ReviewItem model, open_review_queue service
provides:
  - POST /review/{run_id}/sign-off/confirm route with pending-count gate
  - POST /review/{run_id}/reassign route (admin-only)
  - attempt_sign_off stub in services/review.py (Plan 05 replaces)
  - Sign-off sticky footer with disabled/enabled button in queue.html
  - DaisyUI confirmation modal before final POST in queue.html
  - Admin-only reassign form on run status page
affects: [03-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - DaisyUI dialog (showModal()) for confirmation modal before destructive POST
    - Admin-only form rendered via Jinja2 conditional on user.role
    - Sign-off gate enforced both server-side (pending count check) and UI (disabled button)

key-files:
  created: []
  modified:
    - shop/routers/review.py
    - shop/services/review.py
    - shop/routers/runs.py
    - shop/templates/review/queue.html
    - shop/templates/runs/status.html
    - tests/test_review.py

key-decisions:
  - "attempt_sign_off imported directly (not inline) from services.review — Plan 05 replaces stub with two-phase write"
  - "Sign-off modal uses standard HTML POST (not HTMX) per Phase 01 decision; DaisyUI showModal() for open trigger"
  - "Reassign form passes engineers list from runs.py run_status() — only active engineers with role=engineer are shown"

patterns-established:
  - "Server-side gate always checks pending count independently of UI state — UI disabled state is UX only, not a security boundary"
  - "Admin-only forms in templates use user.role == 'admin' conditional rendered server-side"

requirements-completed: [REVIEW-06, SIGNOFF-01]

# Metrics
duration: 3min
completed: 2026-03-07
---

# Phase 03 Plan 04: Sign-Off Gate and Reviewer Reassignment Summary

**Server-enforced sign-off gate with DaisyUI confirmation modal, pending-count disabled button, and admin-only reviewer reassign via form select**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-07T20:36:14Z
- **Completed:** 2026-03-07T20:39:35Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Sign-off gate POST route checks pending items and redirects with error if any remain; succeeds by calling attempt_sign_off stub
- Admin-only POST /reassign route validates role, run status, and reviewer validity before updating Run.reviewer_id
- queue.html upgraded from static card to sticky footer with disabled/enabled button and full DaisyUI confirmation modal
- Admin reassign select form rendered conditionally on status.html; engineers list passed from run_status() route

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests for sign-off gate and admin reassign** - `24c89d7` (test)
2. **Task 1 (GREEN): Sign-off gate route and admin reassign route** - `c151629` (feat)
3. **Task 2: Sign-off UI in queue.html and reassign form in status.html** - `26a911f` (feat)

**Plan metadata:** (docs commit below)

_Note: TDD task has two commits (test RED, then feat GREEN)_

## Files Created/Modified
- `shop/routers/review.py` - Added sign_off_confirm and reassign_run routes; imported attempt_sign_off
- `shop/services/review.py` - Added attempt_sign_off stub (sets signed_at, signed_by_id, status=signed_off)
- `shop/routers/runs.py` - run_status() now passes engineers list to template
- `shop/templates/review/queue.html` - Sticky sign-off footer with enabled/disabled button + DaisyUI modal
- `shop/templates/runs/status.html` - Admin-only reassign form below reviewer field
- `tests/test_review.py` - Implemented test_sign_off_gate and test_admin_can_reassign (replacing xfail stubs)

## Decisions Made
- `attempt_sign_off` imported at module level from services.review (not inline import), making it a clear seam for Plan 05 to replace
- Confirmation modal uses `showModal()` JS on button click then standard HTML form POST — matches Phase 01 decision to avoid HTMX for full-page actions
- Reassign form shows only active engineers (role=engineer, is_active=True) — admin cannot reassign to another admin via this form

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Plan 05 can replace the `attempt_sign_off` stub with the two-phase write atomicity implementation
- The `/review/{run_id}/generating` redirect target is now reachable from the confirmed sign-off flow

## Self-Check: PASSED
- `shop/routers/review.py` - EXISTS (sign_off_confirm, reassign_run routes present)
- `shop/services/review.py` - EXISTS (attempt_sign_off stub present)
- `shop/templates/review/queue.html` - EXISTS (145 lines, sticky footer + modal present)
- `shop/templates/runs/status.html` - EXISTS (admin reassign form present)
- Commits: 24c89d7, c151629, 26a911f all exist in git log

---
*Phase: 03-review-and-sign-off*
*Completed: 2026-03-07*
