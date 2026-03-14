---
id: T05
parent: S02
milestone: M001
provides:
  - "GET /runs/{id} status page with server-rendered stage checklist"
  - "GET /runs/{id}/sse async generator SSE endpoint polling DB every 1s"
  - "POST /runs/{id}/abort transitions low_confidence warning run to failed state"
  - "status.html with PIPE-04 (failed), PIPE-05 (revB_balloon), PIPE-06 (low_confidence) warning states"
  - "_stage_checklist.html server-rendered step checklist partial"
  - "Inline JS EventSource listener: updates checklist from JSON SSE; reloads on close event"
requires: []
affects: []
key_files: []
key_decisions: []
patterns_established: []
observability_surfaces: []
drill_down_paths: []
duration: 10min
verification_result: passed
completed_at: 2026-03-04
blocker_discovered: false
---
# T05: 02-pipeline-bridge 05

**# Phase 2 Plan 5: Run Status Page + SSE Summary**

## What Happened

# Phase 2 Plan 5: Run Status Page + SSE Summary

**Real-time run status page with FastAPI native SSE generator, DaisyUI steps checklist, and three terminal warning states (failed, revB_balloon, low_confidence)**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-04T01:41:34Z
- **Completed:** 2026-03-04T01:51:34Z
- **Tasks:** 2 (+ TDD RED commit)
- **Files modified:** 4

## Accomplishments

- GET /runs/{id} returns status.html with server-rendered DaisyUI step checklist and run metadata; 404 on missing run
- GET /runs/{id}/sse async generator using FastAPI 0.135.x native SSE (response_class=EventSourceResponse + yield); sends stage_update events every 1s; closes stream on terminal state
- POST /runs/{id}/abort transitions low_confidence warning run to failed state, redirects to status page
- status.html implements all three warning UI states: hard fail (red alert), revB_balloon (yellow + Acknowledge button), low_confidence (yellow + confidence message + Proceed/Abort buttons)
- _stage_checklist.html initial server render with step-success/step-primary/step-error DaisyUI classes
- Inline JS EventSource listener updates checklist DOM from JSON SSE payload; reloads on close event

## Task Commits

Each task was committed atomically:

1. **TDD RED: Add failing tests for run status and SSE routes** - `a9a76e7` (test)
2. **Task 1: Add GET /runs/{id}, GET /runs/{id}/sse, and POST /runs/{id}/abort routes** - `43544e5` (feat)
3. **Task 2: Create run status templates** - `0d6d5fe` (feat)

## Files Created/Modified

- `shop/routers/runs.py` - Added STAGE_NAMES constant, run_status(), run_sse() async generator, abort_run() routes; added SSE/asyncio imports
- `shop/templates/runs/status.html` - Full status page: two-column layout, stage checklist, run metadata, three warning state sections, completed CTA, JS EventSource listener
- `shop/templates/runs/_stage_checklist.html` - Server-rendered initial stage checklist with DaisyUI steps-vertical
- `tests/test_runs.py` - Implemented test_stage_progress_updates and test_run_status_lifecycle (were xfail stubs)

## Decisions Made

- **FastAPI SSE generator pattern:** The route must be an async generator function (using `yield`) with `response_class=EventSourceResponse`. FastAPI 0.135.x routing layer detects `is_gen_callable` and automatically encodes `ServerSentEvent` objects. Returning `EventSourceResponse(generator)` fails with `AttributeError: 'ServerSentEvent' has no attribute 'encode'` because `StreamingResponse` receives `ServerSentEvent` objects instead of bytes.
- **SSE test uses terminal-state runs:** Testing SSE with a "running" state run causes `TestClient` to hang indefinitely (the generator loops with `asyncio.sleep(1.0)`). Fixed by testing SSE on a "completed" run — the generator yields one event + close event then breaks.
- **JS EventSource over HTMX SSE:** SSE sends JSON payload; client-side JS updates the `stage-checklist` DOM. On the `close` event, `window.location.reload()` triggers to show the terminal-state UI sections (rendered server-side via Jinja2 conditionals). This avoids the complexity of server-rendered HTMX partial responses inside SSE event data.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] SSE route required async generator pattern, not return EventSourceResponse(generator)**
- **Found during:** Task 1 (TDD GREEN phase)
- **Issue:** Plan showed `return EventSourceResponse(_generator())` which failed at runtime — `EventSourceResponse` extends `StreamingResponse` and expects bytes, not `ServerSentEvent` objects
- **Fix:** Converted route to async generator using `yield` directly in the route body with `response_class=EventSourceResponse` and `-> AsyncIterable[ServerSentEvent]` return type annotation
- **Files modified:** shop/routers/runs.py
- **Verification:** `uv run pytest tests/test_runs.py::test_stage_progress_updates` passes
- **Committed in:** 43544e5 (Task 1 feat commit)

**2. [Rule 1 - Bug] SSE test hung indefinitely with "running" state run**
- **Found during:** Task 1 (TDD GREEN phase)
- **Issue:** `TestClient.get("/runs/{id}/sse")` blocked forever because the SSE generator loops with `asyncio.sleep(1.0)` for non-terminal runs
- **Fix:** Changed SSE test to use a "completed" run — generator emits one `stage_update` event + one `close` event then breaks
- **Files modified:** tests/test_runs.py
- **Verification:** `uv run pytest tests/test_runs.py::test_stage_progress_updates` passes in <1s
- **Committed in:** 43544e5 (Task 1 feat commit)

---

**Total deviations:** 2 auto-fixed (Rule 1 — bugs in SSE implementation pattern and test design)
**Impact on plan:** Both fixes required for correct SSE operation and testability. No scope creep.

## Issues Encountered

- FastAPI 0.135.x SSE API differs from the code snippet in the plan: `EventSourceResponse` wraps `StreamingResponse` and cannot accept a generator of `ServerSentEvent` objects directly. The correct pattern is route-as-generator with `response_class=EventSourceResponse`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Run status page is live: GET /runs/{id} renders stage checklist and all three warning states
- SSE stream is functional: browser will receive real-time stage updates every 1s
- Abort endpoint is functional: POST /runs/{id}/abort transitions low_confidence warning runs
- /review/{id} links in all terminal state CTAs are Phase 3 placeholders (will 404 until Phase 3)
- Full test suite: 56 passed, 6 xfailed, 3 xpassed — no regressions

---
*Phase: 02-pipeline-bridge*
*Completed: 2026-03-04*

## Self-Check: PASSED

- FOUND: shop/routers/runs.py
- FOUND: shop/templates/runs/status.html
- FOUND: shop/templates/runs/_stage_checklist.html
- FOUND: .planning/phases/02-pipeline-bridge/02-05-SUMMARY.md
- FOUND: commit a9a76e7 (test RED)
- FOUND: commit 43544e5 (feat Task 1)
- FOUND: commit 0d6d5fe (feat Task 2)
