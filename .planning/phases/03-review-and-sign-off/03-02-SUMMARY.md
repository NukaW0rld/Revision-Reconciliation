---
phase: 03-review-and-sign-off
plan: 02
subsystem: ui
tags: [fastapi, sqlalchemy, jinja2, htmx, daisyui, review-queue]

# Dependency graph
requires:
  - phase: 03-01
    provides: ReviewItem and Run models with review_items relationship

provides:
  - open_review_queue() service: idempotent ReviewItem population from delta_packet.json
  - GET /review/{run_id} route with filter params
  - Review queue page (queue.html) with filter dropdowns and item list
  - Progress bar partial (_progress_bar.html) as HTMX OOB swap target
  - Review router registered in app.py

affects:
  - 03-03-item-card (builds approve/override UI on top of queue page)
  - 03-04-sign-off (uses sign-off button area in queue.html)
  - 03-05-pdf-packet (reads ReviewItems populated by open_review_queue)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - open_review_queue idempotency pattern: count existing rows before insert
    - JS onchange form submit for filter dropdowns (no HTMX dependency for filter state)
    - _progress_bar.html as HTMX OOB swap target for Plan 03 approve/override responses

key-files:
  created:
    - shop/services/review.py
    - shop/routers/review.py
    - shop/templates/review/queue.html
    - shop/templates/review/_progress_bar.html
  modified:
    - shop/app.py
    - tests/test_review.py

key-decisions:
  - "JS onchange form submit chosen for filter dropdowns to avoid HTMX dependency for filter state — browser POST/GET is sufficient for full-page filter refresh"
  - "Counts (pending/approved/overridden) computed from all_items before filter is applied — sign-off gate needs unfiltered totals"
  - "open_review_queue uses existing_count check (not get_or_create per item) — simpler and avoids partial-insert race conditions"

patterns-established:
  - "Service functions take (db, run) not run_id — caller owns DB session, service uses it"
  - "Template filter dropdowns submit via JS onchange on <select> element inside <form method=GET>"

requirements-completed: [REVIEW-01, REVIEW-05, REVIEW-07]

# Metrics
duration: 5min
completed: 2026-03-07
---

# Phase 03 Plan 02: Review Queue GET Route Summary

**Review queue entry point: idempotent ReviewItem population from delta_packet.json via open_review_queue(), with GET /review/{run_id} route, filter dropdowns, and HTMX-ready progress bar partial**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-07T20:23:32Z
- **Completed:** 2026-03-07T20:28:20Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- `open_review_queue(db, run)` service creates ReviewItems from delta_packet.json on first call; subsequent calls return existing rows without duplicating
- Run.status transitions from completed/warning to "reviewing" on first queue open
- GET /review/{run_id} renders the queue page with all items, filter dropdowns (review status + pipeline classification), and pending/approved/overridden counts in header
- Sign-off button disabled when pending > 0; enabled when all items resolved
- Progress bar as HTMX OOB swap target ready for Plan 03 approve/override responses

## Task Commits

Each task was committed atomically:

1. **TDD RED - test_review_state_persisted** - `7373e5c` (test)
2. **Task 1: open_review_queue service** - `d1657ff` (feat)
3. **Task 2: review router + queue template** - `2e64438` (feat)

## Files Created/Modified

- `shop/services/review.py` - open_review_queue() idempotent service
- `shop/routers/review.py` - GET /review/{run_id} with filter params
- `shop/templates/review/queue.html` - Full review queue page extending base.html
- `shop/templates/review/_progress_bar.html` - Pending/approved/overridden count partial
- `shop/app.py` - Added review router registration at /review prefix
- `tests/test_review.py` - Implemented test_review_queue_loads, test_review_state_persisted, test_review_counts

## Decisions Made

- JS onchange form submit chosen for filter dropdowns (not HTMX) to avoid HTMX dependency for filter state — browser GET is sufficient for full-page filter refresh
- Counts computed from all_items before filter is applied so sign-off gate always sees accurate totals
- open_review_queue uses a single count check for idempotency rather than per-item get_or_create — simpler and avoids partial-insert edge cases

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - the circular import warning in the plan's `python -c "from shop.routers.review import router"` verification command is a pre-existing pattern in this codebase (same for runs.py, auth.py). All routers import `from shop.app import templates` which triggers the module-level `app = create_app()`, which in turn imports all routers. This only fails in direct isolated import; the app works correctly via `create_app()` and all tests pass.

## Self-Check

- [x] `shop/services/review.py` exists
- [x] `shop/routers/review.py` exists
- [x] `shop/templates/review/queue.html` exists
- [x] `shop/templates/review/_progress_bar.html` exists
- [x] Commits 7373e5c, d1657ff, 2e64438 exist
- [x] 63 passed, 9 xfailed in full suite

## Self-Check: PASSED

## Next Phase Readiness

- Review queue route and service fully functional
- Plan 03 can build approve/override HTMX endpoints — _progress_bar.html OOB swap target is ready
- Plan 04 sign-off section placeholder is in queue.html ready to be wired up

---
*Phase: 03-review-and-sign-off*
*Completed: 2026-03-07*
