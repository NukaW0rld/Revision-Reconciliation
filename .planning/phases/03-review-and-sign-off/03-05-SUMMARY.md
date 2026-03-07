---
phase: 03-review-and-sign-off
plan: "05"
subsystem: review
tags: [fastapi, sse, sqlalchemy, htmx, two-phase-write, sign-off, atomicity]

# Dependency graph
requires:
  - phase: 03-review-and-sign-off
    provides: "Run model with signing_off/signed_off states, ReviewItem, attempt_sign_off stub, sign_off_confirm route"
  - phase: 02-pipeline-bridge
    provides: "SSE pattern (EventSourceResponse, ServerSentEvent, db.expire+db.refresh polling loop)"
provides:
  - "attempt_sign_off() two-phase write with automatic rollback to reviewing on failure"
  - "SIGNOFF-03 immutability guard: signed_off runs cannot be re-signed"
  - "GET /review/{run_id}/generating — SSE-polling status page with terminal redirect"
  - "GET /review/{run_id}/sign-off/sse — SSE endpoint streaming sign-off status"
  - "?signed=1 success banner on runs/{run_id} status page"
affects: [04-audit-packet]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-phase write: Phase 1 commit sets transient state (signing_off), Phase 2 commit sets terminal state (signed_off); on Phase 2 exception, rollback + re-query + set reviewing"
    - "SSE immutability pattern: terminal states include both success and rollback states so SSE client always receives a redirect signal"

key-files:
  created:
    - shop/templates/review/generating.html
  modified:
    - shop/services/review.py
    - shop/routers/review.py
    - shop/routers/runs.py
    - shop/templates/runs/status.html
    - tests/test_review.py

key-decisions:
  - "Two-phase write rollback uses db.rollback() then re-queries run by ID because the session object may be in detached state post-rollback; not all SQLAlchemy backends refresh cleanly after rollback on the same object"
  - "SSE terminal set is {signed_off, reviewing} — reviewing is the rollback state; client redirects to /review/{id}?error=sign_off_failed on rollback"
  - "generating.html uses htmx:sseMessage DOM event (not hx-on:sse:status_update) to handle redirect logic in JavaScript — keeps redirect logic explicit and testable"
  - "signed=True passed from run_status() to status.html; banner displayed only when ?signed=1 query param is present"

patterns-established:
  - "SSE pattern (db.expire + db.refresh in polling loop) reused directly from Phase 02 run_sse"

requirements-completed: [SIGNOFF-02, SIGNOFF-03]

# Metrics
duration: 3min
completed: 2026-03-07
---

# Phase 3 Plan 05: Sign-Off Atomicity and Generating Status Page Summary

**Two-phase write attempt_sign_off() with automatic rollback to reviewing, plus SSE-polling generating status page that redirects engineers on sign-off completion or failure**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-07T20:42:01Z
- **Completed:** 2026-03-07T20:45:18Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Replaced stub `attempt_sign_off()` with two-phase write: Phase 1 commits `signing_off` (visible to SSE); Phase 2 commits `signed_off` + `signed_at` + `signed_by_id`; on Phase 2 exception, rolls back and sets `reviewing`
- Added SIGNOFF-03 immutability guard: returns `False` immediately without modification if `run.status == "signed_off"`
- Created `/review/{run_id}/generating` route and template with SSE listener that redirects to `/runs/{id}?signed=1` on success or `/review/{id}?error=sign_off_failed` on rollback
- Added `/review/{run_id}/sign-off/sse` SSE endpoint using `db.expire` + `db.refresh` polling loop (terminal states: `signed_off`, `reviewing`)
- Added success banner to `runs/status.html` shown when `?signed=1` query param is present

## Task Commits

Each task was committed atomically:

1. **TDD RED - SIGNOFF-02/03 tests** - `79cb7e8` (test)
2. **TDD GREEN - attempt_sign_off two-phase write** - `2e236e5` (feat)
3. **Task 2: Generating page + SSE route + signed banner** - `6072475` (feat)

## Files Created/Modified

- `shop/services/review.py` - attempt_sign_off() replaced with two-phase write and immutability guard
- `shop/routers/review.py` - added imports (asyncio, json, SSE), GET /generating route, GET /sign-off/sse route
- `shop/templates/review/generating.html` - new SSE-polling status page, extends base.html
- `shop/routers/runs.py` - run_status() now passes signed=True when ?signed=1 present
- `shop/templates/runs/status.html` - success alert banner shown when signed=True
- `tests/test_review.py` - replaced xfail stubs with real test implementations for SIGNOFF-02 and SIGNOFF-03

## Decisions Made

- Two-phase write rollback re-queries run by ID after `db.rollback()` because session identity-map state is unreliable post-rollback; same pattern established in Phase 02 for SSE polling
- SSE terminal set includes `reviewing` (rollback state) so the client never polls indefinitely; it always receives a redirect signal whether sign-off succeeded or failed
- `htmx:sseMessage` DOM event used in generating.html JavaScript (not inline hx-on attribute) to keep redirect logic explicit and debuggable

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- Sign-off atomicity complete; any Phase 2 exception during PDF/packet generation (Phase 4) will cleanly roll back to reviewing state
- WeasyPrint PDF generation (Phase 4 PACKET-01) can now write inside Phase 2 of `attempt_sign_off()` knowing the rollback mechanism is in place
- SSE pattern for sign-off generating page mirrors the pipeline status page pattern — no new concepts needed

## Self-Check: PASSED

- shop/services/review.py: FOUND
- shop/routers/review.py: FOUND
- shop/templates/review/generating.html: FOUND
- 03-05-SUMMARY.md: FOUND
- Commit 79cb7e8 (test RED): FOUND
- Commit 2e236e5 (feat GREEN): FOUND
- Commit 6072475 (feat Task 2): FOUND

---
*Phase: 03-review-and-sign-off*
*Completed: 2026-03-07*
