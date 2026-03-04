---
phase: 02-pipeline-bridge
plan: 08
subsystem: testing, ui, infra
tags: [pytest, huey, sse, docker, supervisord, tailwind, daisyui]

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

key-files:
  created:
    - .planning/phases/02-pipeline-bridge/02-08-SUMMARY.md
  modified:
    - shop/routers/setup.py
    - shop/templates/runs/new.html
    - shop/tasks.py
    - shop/templates/base.html
    - shop/templates/runs/status.html

key-decisions:
  - "Admin is created during setup wizard step 2 (not in seed_admin) because supervisord starts uvicorn directly without running run_web.py"
  - "Bell badge links to /dashboard instead of /alerts (no alerts route exists; dashboard shows unread banners)"
  - "SSE close handler injects banner immediately from event data instead of waiting for full reload"

patterns-established:
  - "Lazy import fallback: module-level import failure returns None; task body uses _get_X() lazy getter as fallback"
  - "SSE terminal state: inject banner immediately via JS then reload after delay to avoid race condition"

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

**Full test suite green (60 tests), five Docker verification bugs fixed: admin creation, pipeline task callable, bell badge href, SSE terminal banner injection, and rev label rename**

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

## Files Created/Modified

- `shop/routers/setup.py` - Create admin user during step 2 if not already seeded
- `shop/templates/runs/new.html` - Rename "Rev A/B Label" to "First/Second Revision Label"
- `shop/tasks.py` - Lazy fallback to `_get_run_pipeline()` when module-level `run_pipeline` is None
- `shop/templates/base.html` - Fix bell badge href from /alerts to /dashboard
- `shop/templates/runs/status.html` - Inject terminal banner immediately via SSE data; add 500ms delay before reload

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

---

**Total deviations:** 5 auto-fixed (4 bugs, 1 UI rename)
**Impact on plan:** All fixes necessary for correct Docker operation. No scope creep.

## Issues Encountered

None beyond the 5 Docker verification bugs documented above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 2 gate complete: all 12 test_runs.py tests pass, Docker verified
- Phase 3 (Review Queue) can begin: completed runs exist in DB with delta_packet.json output
- Known: `/review/{run_id}` links in status page are placeholders (Phase 3 will implement)

---
*Phase: 02-pipeline-bridge*
*Completed: 2026-03-04*
