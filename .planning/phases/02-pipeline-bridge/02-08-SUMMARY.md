---
phase: 02-pipeline-bridge
plan: 08
subsystem: testing, ui, infra
tags: [pytest, huey, sse, docker, supervisord, tailwind, daisyui, htmx]

requires:
  - phase: 02-pipeline-bridge
    provides: upload form, run status page, SSE, alert system from plans 01-07

provides:
  - Fully implemented test_runs.py with all 12 tests passing (no xfail)
  - Docker end-to-end fixes: admin creation, pipeline task, alert bell, SSE banner
  - Phase 2 gate verification complete

affects:
  - 03-review-queue (alert dismiss, run status, completed runs list)

tech-stack:
  added: []
  patterns:
    - "Setup wizard creates admin on step 2 if supervisord skips seed_admin()"
    - "Huey task uses lazy _get_run_pipeline() fallback when module-level import is None"
    - "SSE close handler injects terminal banner immediately then reloads after 500ms"
    - "Huey task except block initializes run=None to avoid UnboundLocalError in failure handler"
    - "HTMX validate-pdf endpoint uses request.form() iteration to accept any file field name"

key-files:
  created:
    - .planning/phases/02-pipeline-bridge/02-08-SUMMARY.md
  modified:
    - shop/routers/setup.py
    - shop/templates/runs/new.html
    - shop/tasks.py
    - shop/templates/base.html
    - shop/templates/runs/status.html
    - shop/routers/runs.py

key-decisions:
  - "Admin is created during setup wizard step 2 (not in seed_admin) because supervisord starts uvicorn directly without running run_web.py"
  - "Bell badge links to /dashboard instead of /alerts (no alerts route exists; dashboard shows unread banners)"
  - "SSE close handler injects banner immediately from event data instead of waiting for full reload"
  - "Pipeline task except block must initialize run=None before try to prevent UnboundLocalError in failure handler"
  - "validate-pdf endpoint uses request.form() iteration instead of typed File() param to accept HTMX file field names"
  - "SSE raw_data vs data: ServerSentEvent(data=str) double-encodes; always use raw_data= for pre-serialised JSON payloads"
  - "SQLAlchemy SSE polling: db.expire(run)+db.refresh(run) required before each poll to bypass identity-map cache in async generators"
  - "validate-pdf single-page response: return hidden input (not empty string) to keep swap target populated and page selector reliably dismissible"
  - "Open Review Queue /review/{run_id} returning 404 is intentional — Phase 3 work, not a Phase 2 bug"

patterns-established:
  - "Lazy import fallback: module-level import failure returns None; task body uses _get_X() lazy getter as fallback"
  - "SSE terminal state: inject banner immediately via JS then reload after delay to avoid race condition"
  - "SSE payload encoding: use raw_data= on ServerSentEvent when payload is already a JSON string; data= causes double-encoding"
  - "SQLAlchemy SSE polling: call db.expire(obj); db.refresh(obj) before each poll to bypass identity-map cache in long-lived generator"
  - "HTMX file validation target: always return a form element (not empty string) so the swap target retains a valid input regardless of previous state"

requirements-completed:
  - UPLOAD-01
  - UPLOAD-02
  - UPLOAD-03
  - UPLOAD-04
  - UPLOAD-05
  - PIPE-01
  - PIPE-02
  - PIPE-03
  - PIPE-04
  - PIPE-05
  - PIPE-06
  - PIPE-11

duration: 25min
completed: 2026-03-04
---

# Phase 02 Plan 08: Phase 2 Gate — Tests, Docker Verification, and Bug Fixes Summary

**Full test suite green (60 tests), nine bugs fixed: admin creation, pipeline task callable, bell badge href, SSE terminal banner injection, rev label rename, task except UnboundLocalError, HTMX validate-pdf field name mismatch, SSE double-encoding and SQLAlchemy identity-map cache, page selector hidden input restoration**

## Performance

- **Duration:** 25 min
- **Started:** 2026-03-04T00:00:00Z
- **Completed:** 2026-03-04T00:25:00Z
- **Tasks:** 3 (Tasks 1 and 2 already done; Task 3 was 5 post-verification fixes)
- **Files modified:** 5

## Accomplishments

- All 12 test_runs.py stubs implemented and passing (no xfail)
- Full test suite green: 60 passed, 2 xfailed, 3 xpassed
- Fixed admin login broken after Docker setup wizard (admin never created by supervisord)
- Fixed pipeline task crash: `'NoneType' object is not callable` when delta_preservation import fails at module load
- Fixed bell badge 404 (linked to non-existent /alerts route)
- Fixed SSE terminal banner not auto-appearing (race condition between reload and DOM injection)
- Renamed Rev A/B Label fields to First/Second Revision Label

## Task Commits

Tasks 1 and 2 completed in prior session:

1. **Task 1: Implement all test_runs.py stubs** - `ce45e04` (feat)
2. **Task 2: Full test suite green + Tailwind rebuild** - `e6f5603` (feat)

Task 3 — five post-verification bug fixes:

3. **Issue 1: Admin creation in setup wizard step 2** - `2fbbd55` (fix)
4. **Issue 2: Rev label rename in /runs/new** - `0b5d473` (fix)
5. **Issue 3: Pipeline task lazy import fallback** - `ec3091d` (fix)
6. **Issue 4: Bell badge href fixed to /dashboard** - `34fc6c9` (fix)
7. **Issue 5: SSE terminal banner injection** - `e3bb773` (fix)

Post-gate re-verification bug fixes:

8. **Issue 6: Pipeline task except block UnboundLocalError** - `48b38e5` (fix)
9. **Issue 7: HTMX validate-pdf file field name mismatch** - `727db59` (fix)

Second round of post-gate fixes:

10. **Issue 8: Rev B page selector** - confirmed already implemented in dc18307 (no fix needed)
11. **Issue 9: Pipeline task diagnostic logging and db=None guard** - `441d1b8` (fix)

Third round of post-gate fixes (Docker verification bugs):

12. **Issue 10: Page selector doesn't reliably show/hide** - `186fa1d` (fix)
13. **Issue 11: SSE real-time updates not working** - `b3cc7ca` (fix)

## Files Created/Modified

- `shop/routers/setup.py` - Create admin user during step 2 if not already seeded
- `shop/templates/runs/new.html` - Rename "Rev A/B Label" to "First/Second Revision Label"
- `shop/tasks.py` - Lazy fallback to `_get_run_pipeline()` when module-level `run_pipeline` is None; guard except block with `run = None` and `db = None` initialization; add diagnostic print/logger calls
- `shop/templates/base.html` - Fix bell badge href from /alerts to /dashboard
- `shop/templates/runs/status.html` - Inject terminal banner immediately via SSE data; add 500ms delay before reload
- `shop/routers/runs.py` - Accept any file field name in validate-pdf endpoint for HTMX compatibility; add enqueue diagnostic logging; restore hidden page input for single-page PDFs; fix SSE double-encoding (raw_data vs data) and SQLAlchemy identity-map cache (db.expire+refresh)

## Decisions Made

- **Admin creation moved to wizard**: `supervisord` starts `uvicorn shop.app:app` directly, bypassing `run_web.py`'s `seed_admin()`. The setup wizard step 2 now creates the admin user if none exists, using the password chosen by the user. Default email `admin@shop.local` matches the environment variable default.
- **Bell links to /dashboard**: No `/alerts` route exists. The dashboard already shows unread alert banners at the top, making it the correct landing page for the bell button.
- **SSE banner injection pattern**: Rather than relying solely on `window.location.reload()` (which can race with the last DB commit), the close handler now injects a banner immediately from the last `stage_update` event data. The reload is deferred 500ms to let the user see the banner before the page refreshes to the full server-rendered version.
- **Lazy import fallback for run_pipeline**: The module-level `from delta_preservation.cli import run_pipeline` can fail in Docker (e.g., circular import during huey worker startup). The task now falls back to `_get_run_pipeline()` if the module-level symbol is None, ensuring the callable is always resolved at execution time.

## Deviations from Plan

All 5 issues were discovered during human Docker verification (Task 3 checkpoint). They were fixed as continuation of Task 3.

**1. [Rule 1 - Bug] Admin login broken — no admin user created by supervisord**
- **Found during:** Task 3 Docker verification
- **Issue:** `supervisord.conf` starts uvicorn directly (`uvicorn shop.app:app`), not via `run_web.py`. `run_web.py` contains `seed_admin()` which creates the default admin. Without it, no admin exists. Setup wizard step 2 only updated an existing admin's password (silent no-op when admin is None).
- **Fix:** Added `else` branch in `step2_post` to create the admin user with the wizard-chosen password when no admin exists.
- **Files modified:** `shop/routers/setup.py`
- **Committed in:** 2fbbd55

**2. [Rule 1 - Bug] Pipeline task crashes with NoneType callable**
- **Found during:** Task 3 Docker verification
- **Issue:** Module-level `from delta_preservation.cli import run_pipeline` can fail when huey worker imports `shop.tasks` at startup if delta_preservation is not yet importable. Fallback sets `run_pipeline = None`. Task body used `_run_pipeline = run_pipeline` (the None value) then called `_run_pipeline(...)`.
- **Fix:** Task body now uses `run_pipeline if run_pipeline is not None else _get_run_pipeline()`, deferring the import to execution time where the full Python path is available.
- **Files modified:** `shop/tasks.py`
- **Committed in:** ec3091d

**3. [Rule 1 - Bug] Bell badge linked to non-existent /alerts route**
- **Found during:** Task 3 Docker verification
- **Issue:** `base.html` had `<a href="/alerts">` but no `/alerts` GET route exists, returning 404 JSON.
- **Fix:** Changed href to `/dashboard` where unread alert banners are displayed.
- **Files modified:** `shop/templates/base.html`
- **Committed in:** 34fc6c9

**4. [Rule 1 - Bug] Failure alert banner didn't auto-show via SSE**
- **Found during:** Task 3 Docker verification
- **Issue:** SSE `close` handler called `window.location.reload()` immediately. Race condition: page may reload before the terminal banner is injected. Also, the banner was never rendered until the reloaded page fully loaded.
- **Fix:** Added `injectTerminalBanner(lastData)` call before reload using last `stage_update` event data. Added `terminal-banner-slot` div as JS injection target. Reload deferred 500ms.
- **Files modified:** `shop/templates/runs/status.html`
- **Committed in:** e3bb773

**5. [Rule 1 - Bug] Pipeline task except block references unbound `run` variable**
- **Found during:** Post-gate re-verification
- **Issue:** In `run_pipeline_task`, the `except` block called `db.refresh(run)` and `run.status = "failed"` but `run` is assigned inside the `try` block. If an exception occurs before the `db.query(Run)...first()` line (e.g., DB connection error), `run` is unbound, causing `UnboundLocalError` in the except handler. This secondary exception propagates silently, leaving the run permanently stuck at "queued".
- **Fix:** Initialize `run = None` before the try block. Guard the failure-persistence code with `if run is not None:` and wrap it in its own try/except to log secondary failures without swallowing them.
- **Files modified:** `shop/tasks.py`
- **Committed in:** 48b38e5

**6. [Rule 1 - Bug] HTMX sends file field as input name, but endpoint expected `file`**
- **Found during:** Post-gate re-verification
- **Issue:** The `validate_pdf` endpoint declared `file: UploadFile = File(...)`. HTMX sends file inputs using the `<input name="...">` attribute as the multipart field name. The Rev A input is named `revA_pdf` and Rev B is named `revB_pdf`. FastAPI couldn't find a `file` field in the multipart data and returned 422, so no page selector ever appeared.
- **Fix:** Changed endpoint signature to accept `Request` directly. Iterate `form.multi_items()` to find whichever field is an `UploadFile`, regardless of field name. Tests that use `files={"file": ...}` still work because the iteration picks up `file` as well.
- **Files modified:** `shop/routers/runs.py`
- **Committed in:** 727db59

**7. [Investigation] Rev B page selector already implemented**
- **Found during:** Second round post-gate review
- **Issue:** User reported Rev B page selector missing. On investigation, the Rev B `<input>` in `new.html` already has identical HTMX attributes to Rev A (`hx-post`, `hx-trigger="change"`, `hx-encoding`, `hx-vals='{"field": "revB"}'`, `hx-target="#revB-section"`, `hx-swap="innerHTML"`). The `_page_selector.html` partial uses `name="{{ field }}_page"` so it renders `name="revB_page"` for Rev B. The POST handler reads `revB_page: int = Form(0)`. Everything is correctly wired.
- **Fix:** No code change required — feature was already implemented in dc18307.

**8. [Rule 1 - Bug] Pipeline task: db=None guard and diagnostic logging missing**
- **Found during:** Second round post-gate review
- **Issue:** `db = _SessionLocal()` was called before the `try` block, so if `_SessionLocal()` threw an exception, `db` was never assigned and the `finally: db.close()` would raise `UnboundLocalError`. Additionally, the `except` block checked `run is not None` but not `db is not None`, meaning `db.commit()` could also fail if db creation threw. No diagnostic logging existed to confirm whether the Huey worker was entering the task body at all in Docker.
- **Fix:** Moved `db = _SessionLocal()` inside the `try` block. Initialized `db = None` before the try alongside `run = None`. Updated `except` guard to `if run is not None and db is not None:`. Updated `finally` to `if db is not None: db.close()`. Added `print("HUEY TASK STARTED run_id=", run_id)` and `logger.info(...)` at task entry. Added `print("HUEY TASK: enqueuing run_id=", run.id)` and `logger.info(...)` before the enqueue call in the router.
- **Files modified:** `shop/tasks.py`, `shop/routers/runs.py`
- **Committed in:** 441d1b8

**9. [Rule 1 - Bug] Page selector doesn't reliably show/hide for Rev B PDF**
- **Found during:** Third round Docker verification
- **Issue:** `validate_pdf` endpoint returned `HTMLResponse("")` for single-page valid PDFs. When user first uploaded a multi-page PDF (page selector appeared), then switched to a 1-page PDF, the empty response cleared the `#revB-section` div entirely — removing the hidden `revB_page` input. While this didn't cause a 422 (the route has `Form(0)` default), it meant the section was fully empty, and the page selector HTML (injected by the previous multi-page upload) would linger under HTMX timing edge cases.
- **Fix:** Return `HTMLResponse('<input type="hidden" name="{field}_page" value="0">')` for single-page PDFs to always keep the section populated with a valid form field, whether or not a page selector was shown before.
- **Files modified:** `shop/routers/runs.py`
- **Committed in:** 186fa1d

**10. [Rule 1 - Bug] SSE real-time updates not working — double-encoding and stale DB cache**
- **Found during:** Third round Docker verification
- **Issue:** Two compounding bugs prevented the stage checklist from updating in real time:
  (a) `ServerSentEvent(data=json_str)` JSON-encodes its `data` argument. When `data` is already a JSON string, the wire sends `data: "{\"status\": \"running\"}"` (a JSON-encoded string). `JSON.parse(e.data)` returned a string, not an object, so `updateChecklist(data)` received no usable fields.
  (b) SQLAlchemy identity-map caching: `db.query(Run).filter(...)` inside the polling loop returned the same Python instance from the session cache on every iteration. `run.status` never reflected database changes written by the Huey worker.
- **Fix:** (a) Changed `ServerSentEvent(data=payload)` to `ServerSentEvent(raw_data=payload)` so the pre-serialised JSON is sent verbatim without a second encoding pass. (b) Added `db.expire(run); db.refresh(run)` before each poll cycle to force a fresh SELECT from SQLite.
- **Non-issue noted:** "Open Review Queue" returning 404 is expected — `/review/{run_id}` is Phase 3 work and intentionally deferred.
- **Files modified:** `shop/routers/runs.py`
- **Committed in:** b3cc7ca

---

**Total deviations:** 11 auto-fixed/investigated (9 bugs, 1 UI rename, 1 no-op investigation)
**Impact on plan:** All fixes necessary for correct operation. No scope creep.

## Issues Encountered

None beyond the 9 bugs/investigations documented above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 2 gate complete: all 12 test_runs.py tests pass, Docker verified
- Phase 3 (Review Queue) can begin: completed runs exist in DB with delta_packet.json output
- Known: `/review/{run_id}` links in status page are placeholders (Phase 3 will implement)

---
*Phase: 02-pipeline-bridge*
*Completed: 2026-03-04*
