---
id: S04
parent: M001
milestone: M001
provides:
  - WeasyPrint installed and importable in venv (weasyprint 68.1)
  - Dockerfile runtime stage includes Pango/HarfBuzz system deps for WeasyPrint
  - Run.parent_run_id nullable FK column (self-referential, for amendments)
  - Run.packet_versions JSON column (for versioned audit packets)
  - ShopConfig.retention_days Integer column (default 30, for cleanup scheduler)
  - run_schema_migrations() startup function for SQLite ALTER TABLE migrations
  - 14 xfail test stubs (PACKET-01..03, WORK-01..04, HISTORY-01..04, AMEND-01..03)
  - shop/services/exports.py with generate_audit_packet_csv, render_audit_packet_pdf, generate_and_store_audit_packet
  - shop/routers/exports.py with GET /exports/{run_id}/audit-packet.pdf and .csv
  - shop/templates/exports/audit_packet.html WeasyPrint template (cover + summary + per-item cards)
  - PDF generation integrated into attempt_sign_off before signed_off status commit
  - Download Audit Packet (PDF/CSV) buttons on status.html for signed_off runs
  - "create_amendment(db, parent_run, initiator_id) in shop/services/amendments.py — clones a signed-off run with pre-filled ReviewItems"
  - "POST /review/{run_id}/amend route — creates amendment and redirects to its review queue"
  - "Amend button + DaisyUI modal on signed-off run status page"
  - "Amendment banner on review queue page identifying parent run and locked files"
  - "Versioned packet list on status page linking to all packet versions"
  - "Version-aware PDF download via ?version=N query param on audit-packet.pdf route"
  - "3 passing amendment tests (AMEND-01, AMEND-02, AMEND-03)"
  - "Full test suite green (84 passed) with all Phase 4 tests (test_exports, test_history, test_amendments) passing"
  - "Docker image verified to build with WeasyPrint and Pango system dependencies"
  - "Human-approved end-to-end flow covering all five Phase 4 feature areas"
  - "Phase 4 milestone complete"
requires: []
affects: []
key_files: []
key_decisions:
  - "weasyprint added unpinned — let uv resolve latest compatible (resolved 68.1)"
  - "run_schema_migrations uses sa_inspect + ALTER TABLE per column — idempotent guard prevents double-add on repeated startup"
  - "parent_run_id is a nullable self-referential FK on runs — supports amendment lineage chains without a separate table"
  - "packet_versions is JSON (not a separate table) — stores ordered list of packet metadata snapshots, sufficient for Phase 4 audit trail needs"
  - "retention_days on ShopConfig (not a separate config table) — consistent with existing column_mapping/wizard_step pattern"
  - "Inline import of generate_and_store_audit_packet inside attempt_sign_off try block avoids circular import (exports.py imports shop.app.templates which imports review.py via router)"
  - "unittest.mock.patch targets shop.services.exports.generate_and_store_audit_packet — inline from-import resolves attribute at call time inside the patch context manager"
  - "base_url for WeasyPrint set to output_dir/snippets/ so basename img paths resolve; template uses | basename filter (already registered globally)"
  - "Re-download route checks packet_versions[0].path existence first; falls back to re-render for test environments without real output_dir"
  - "Amendment packet_versions initialized as copy of parent list — ensures generate_and_store_audit_packet computes version=2 (not version=1) on amendment sign-off"
  - "Standard HTML form POST (not HTMX) for amend modal — consistent with sign-off modal convention from Phase 01"
  - "Version-aware PDF download uses ?version=N query param; falls back to re-render if stored file absent"
  - "Amendment banner uses run.parent_run_id check — zero overhead, no extra DB query"
  - "No deviations from plan — one auto-fix applied (stale xfail markers on RBAC tests removed)"
patterns_established:
  - "Startup migration pattern: run_schema_migrations(engine) called in create_app() after Base.metadata.create_all() — ensures existing Docker deployments gain new columns without manual migration step"
  - "xfail(strict=False) stubs in named test files — test names visible in collection, xfail count verifiable, requirements traceable by test name prefix"
  - "WeasyPrint render: HTML(string=..., base_url=str(snippets_dir)).write_pdf()"
  - "CSV export: csv.DictWriter to io.StringIO, seek(0) before return, StreamingResponse with attachment header"
  - "Audit packet stored at output_dir/packets/v{N}.pdf with version metadata in packet_versions JSON list"
  - "Amendment creation: create_amendment() flush+clone+commit pattern for atomic Run+ReviewItems creation"
  - "Version-aware file serving: next((v for v in versions if v.get('version') == version), None) lookup pattern"
  - "Phase gate pattern: fix failures, run full suite, Docker build, human E2E as a single blocking verification task before closing a milestone"
observability_surfaces: []
drill_down_paths: []
duration: human-verify
verification_result: passed
completed_at: 2026-03-08
blocker_discovered: false
---
# S04: Exports History And Amendments

**# Phase 4 Plan 01: Foundation Summary**

## What Happened

# Phase 4 Plan 01: Foundation Summary

**WeasyPrint installed with Pango/HarfBuzz Docker deps; Run extended with parent_run_id + packet_versions FKs; ShopConfig gains retention_days; startup SQLite migration guard added; 14 Phase 4 xfail test stubs created**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-07T02:40:03Z
- **Completed:** 2026-03-07T02:42:35Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- WeasyPrint 68.1 installed via `uv sync` and verified importable (`from weasyprint import HTML` passes)
- Dockerfile runtime stage extended with three Pango/HarfBuzz apt packages needed for WeasyPrint to render fonts in Docker
- Three new DB columns added to existing SQLAlchemy models: `Run.parent_run_id` (self-ref FK), `Run.packet_versions` (JSON), `ShopConfig.retention_days` (Integer default 30)
- `run_schema_migrations(eng)` function added to `shop/database.py` — idempotent SQLite ALTER TABLE guard called at startup so existing shop.db deployments gain new columns without downtime
- 14 xfail test stubs created across three files, one per Phase 4 requirement group (exports, history, amendments)
- Full test suite: 70 passed, 16 xfailed (pre-existing), 14 new xfailed stubs, 0 failures

## Task Commits

1. **Task 1: Install WeasyPrint and add Dockerfile system deps** - `1d568bc` (chore)
2. **Task 2: Add DB columns, startup migration, xfail test stubs** - `685b178` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `pyproject.toml` - added `"weasyprint"` to dependencies list
- `uv.lock` - resolved weasyprint 68.1 + transitive deps (fonttools, pillow, pydyf, pyphen, tinycss2, tinyhtml5, webencodings, zopfli)
- `docker/Dockerfile` - runtime stage apt-get extended with libpango-1.0-0, libpangoft2-1.0-0, libharfbuzz-subset0
- `shop/models.py` - Run: parent_run_id (nullable ForeignKey("runs.id")), packet_versions (JSON nullable); ShopConfig: retention_days (Integer nullable default 30)
- `shop/database.py` - imports text/inspect; adds run_schema_migrations() function
- `shop/app.py` - calls run_schema_migrations(engine) after Base.metadata.create_all()
- `tests/test_exports.py` - 7 xfail stubs: PACKET-01..03, WORK-01..04
- `tests/test_history.py` - 4 xfail stubs: HISTORY-01..04
- `tests/test_amendments.py` - 3 xfail stubs: AMEND-01..03

## Decisions Made

- WeasyPrint added unpinned — uv resolves latest compatible (resolved 68.1); no version constraint needed for this internal tool
- `run_schema_migrations` uses `sa_inspect` to check existing column names before issuing ALTER TABLE — prevents errors on repeated startup with up-to-date schema
- `parent_run_id` is a self-referential nullable FK on the `runs` table — supports amendment lineage without a separate table, sufficient for Phase 4 scope
- `packet_versions` stored as JSON list on `Run` — ordered list of packet metadata snapshots; avoids a separate versioning table for Phase 4 audit trail requirements

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- WeasyPrint foundation ready for Plans 02 (audit packet PDF) and 03 (work order PDF)
- DB schema columns ready for Plans 04 (history/retention) and 05 (amendments)
- All 14 Phase 4 requirements have failing test stubs — plans 02-05 will implement and replace these stubs with real tests

---
*Phase: 04-exports-history-and-amendments*
*Completed: 2026-03-07*

# Phase 4 Plan 02: Audit Packet Exports Summary

**WeasyPrint PDF audit packet + CSV export with sign-off integration storing v1.pdf at output_dir/packets/ and re-download via /exports/{id}/audit-packet.{pdf,csv}**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-08T12:04:34Z
- **Completed:** 2026-03-08T12:10:38Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Audit packet export service generates CSV (per-ReviewItem rows) and PDF (WeasyPrint cover + summary + per-item cards with snippet images)
- Sign-off atomicity extended: `attempt_sign_off` generates and stores v1.pdf before committing `signed_off` status; failure rolls back to `reviewing`
- Download routes at `/exports/{id}/audit-packet.pdf` and `.csv` with Content-Disposition attachment headers; PDF re-download serves stored file via FileResponse
- Status page shows Download Audit Packet buttons for `signed_off` runs

## Task Commits

1. **Task 1: Build audit packet export service** - `5bb4ad8` (feat)
2. **Task 2: Wire sign-off integration, routes, status page** - `e768ff1` (feat)

**Plan metadata:** `7eb7e52` (docs: complete plan)

## Files Created/Modified

- `shop/services/exports.py` — generate_audit_packet_csv, render_audit_packet_pdf, generate_and_store_audit_packet
- `shop/routers/exports.py` — GET /exports/{id}/audit-packet.pdf and .csv with auth guard
- `shop/templates/exports/audit_packet.html` — WeasyPrint template: cover page, summary table, per-item detail cards
- `shop/services/review.py` — attempt_sign_off now calls generate_and_store_audit_packet before signed_off
- `shop/app.py` — exports router registered at /exports prefix
- `shop/templates/runs/status.html` — Download Audit Packet buttons for signed_off runs
- `tests/test_exports.py` — 3 passing tests (pdf import, csv rows, redownload) + 4 xfail stubs
- `tests/test_review.py` — sign-off tests updated to mock PDF generation

## Decisions Made

- Inline import of `generate_and_store_audit_packet` inside `attempt_sign_off` try block avoids circular import chain (exports.py -> shop.app.templates -> routers -> review.py)
- `unittest.mock.patch("shop.services.exports.generate_and_store_audit_packet")` works with inline from-imports because the `from ... import` resolves the attribute at execution time inside the patch context
- WeasyPrint `base_url` set to `output_dir/snippets/` directory so `| basename` filtered image paths resolve correctly without absolute paths in HTML

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_sign_off_rollback and test_signed_off_immutable broken by PDF integration**
- **Found during:** Task 2 (wire sign-off integration)
- **Issue:** Both tests used `db.commit` patching or direct `attempt_sign_off` calls that assumed no PDF generation. With `generate_and_store_audit_packet` now called inside the try block, the TypeError from `Path(None)` and commit-count mismatch broke both tests.
- **Fix:** Updated `test_sign_off_rollback` to use `unittest.mock.patch` on `generate_and_store_audit_packet` to simulate failure; updated `test_signed_off_immutable` to mock PDF generation for the successful first sign-off call.
- **Files modified:** tests/test_review.py
- **Verification:** Both tests pass; full suite 73 passed, 13 xfailed, 3 xpassed
- **Committed in:** e768ff1 (Task 2 commit)

**2. [Rule 2 - Missing Critical] Added session authentication to test_audit_packet_redownload**
- **Found during:** Task 2 (redownload test)
- **Issue:** CSV download route requires authentication; test was hitting the route without a session cookie, getting redirected to /login
- **Fix:** Added `_login_engineer` helper and updated `test_audit_packet_redownload` to authenticate before requesting
- **Files modified:** tests/test_exports.py
- **Verification:** Test returns 200 with attachment header
- **Committed in:** e768ff1 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical)
**Impact on plan:** Both fixes essential for test correctness after integrating PDF generation into sign-off flow. No scope creep.

## Issues Encountered

None beyond the auto-fixed test regressions above.

## Next Phase Readiness

- Audit packet PDF/CSV export complete; PACKET-01..03 satisfied
- `/exports/` router pattern established for Plan 03 work order routes
- `generate_and_store_audit_packet` version increment logic ready for Plan 04 amendment flow (v2, v3 packets)

---
*Phase: 04-exports-history-and-amendments*
*Completed: 2026-03-08*

# Phase 4 Plan 3: Work Order Exports Summary

Partial FAI work order PDF/CSV exports with RE-MEASURE/NEW priority labels, WeasyPrint template, download routes, and status page buttons for signed-off runs.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Work order service functions and WeasyPrint template | 82f63f0 | shop/services/exports.py, shop/templates/exports/work_order.html |
| 2 | Work order download routes, status page buttons, and test implementations | 94e7d3c | shop/routers/exports.py, shop/templates/runs/status.html, tests/test_exports.py |

## What Was Built

- `_effective_classification(item)` resolves override_classification if reviewer_decision == "overridden", otherwise returns pipeline_classification
- `_work_order_rows(db, run)` filters ReviewItems to changed/added only, maps changed→"RE-MEASURE" and added→"NEW", returns list of dicts
- `generate_work_order_csv(db, run)` returns StringIO seeked to 0 with fieldnames: char_no, priority, requirement_revB, drawing_reference
- `generate_work_order_pdf(db, run)` renders work_order.html via WeasyPrint and returns bytes
- `shop/templates/exports/work_order.html` — Letter-size WeasyPrint template with run header table, work items table (Char #, Priority, Requirement (Rev B), Drawing Reference), and empty-state message
- `GET /exports/{run_id}/work-order.pdf` — StreamingResponse with attachment header, 403 for non-signed_off runs
- `GET /exports/{run_id}/work-order.csv` — StreamingResponse with attachment header, 403 for non-signed_off runs
- Status page: added Generate Work Order PDF/CSV buttons alongside audit packet buttons in the signed_off block

## Test Results

```
tests/test_exports.py: 7 passed (3 audit packet + 4 work order)
Full suite: 81 passed, 5 xfailed, 3 xpassed
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added missing login call in test_work_order_button_visible**
- **Found during:** Task 2 test run
- **Issue:** Plan's test stub for test_work_order_button_visible did not call _login_engineer() before GET /runs/{id}. The route requires authentication and redirected to /login, causing "work-order.pdf" assertion to fail.
- **Fix:** Added `db_engine` parameter and `_login_engineer(client, db_engine, engineer_user)` call at start of test, matching the pattern used in test_audit_packet_redownload.
- **Files modified:** tests/test_exports.py
- **Commit:** 94e7d3c

## Self-Check: PASSED

All created files exist, all task commits present, 7/7 export tests pass.

# Phase 4 Plan 4: Run History Filters, Read-Only Review, and Retention Cleanup Summary

**One-liner:** Date-filtered run history, read-only review queue for signed-off runs, admin retention settings, and daily Huey cleanup task for old runs.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Date filter in history list + read-only review queue | 50b0298 | runs.py, review.py, list.html, queue.html, _item_card.html |
| 2 | Admin retention settings + Huey cleanup task + tests | 4d13fc7 | admin.py, settings.html, tasks.py, test_history.py |

## What Was Built

**Run History Date Filter (HISTORY-01):**
- `list_runs` accepts `date_from: str = None` query parameter
- Filters `Run.submitted_at >= datetime.strptime(date_from, "%Y-%m-%d")`; invalid dates silently ignored
- Combined filtering: `part_number` + `date_from` can be used together
- Template updated with a date input; Clear link condition covers both filters

**Read-Only Review Queue for Signed-Off Runs (HISTORY-02):**
- `review_queue` guard extended to allow `signed_off` status through (previously only `completed/warning/reviewing/signing_off`)
- `read_only = run.status == "signed_off"` passed to template context
- `queue.html` renders an info banner when `read_only` is true
- `_item_card.html` wraps approve/override action controls in `{% if not read_only %}` guard

**Admin Retention Settings (HISTORY-03):**
- `POST /admin/settings/retention` route saves `retention_days` (clamped 1–3650) to `ShopConfig`
- Admin `settings.html` has a "Run Retention" card section with a number input pre-filled from config

**Huey Cleanup Task (HISTORY-04):**
- `cleanup_old_runs` registered as `@huey.periodic_task(crontab(hour="0", minute="0"))` — runs daily at UTC midnight
- Deletes runs in `{queued, running, failed, completed}` statuses older than `retention_days`
- `signed_off`, `reviewing`, `signing_off` are never deleted
- Cleanup removes: `output_dir` (shutil.rmtree), `revA_path`, `revB_path`, `form3_path`
- Falls back to 30 days if `ShopConfig.retention_days` is None

## Test Results

```
tests/test_history.py ....   (4 passed)
Full suite: 77 passed, 9 xfailed, 3 xpassed
```

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- `shop/routers/runs.py` — FOUND, contains `date_from`
- `shop/routers/review.py` — FOUND, contains `read_only`
- `shop/tasks.py` — FOUND, contains `cleanup_old_runs`
- `shop/routers/admin.py` — FOUND, contains `retention`
- `shop/templates/admin/settings.html` — FOUND, contains `retention_days`
- Commit 50b0298 — FOUND
- Commit 4d13fc7 — FOUND

# Phase 4 Plan 05: Amendment Model Summary

**Amendment workflow: create_amendment service clones signed-off runs with pre-filled review decisions, preserving original packet and producing v2 on re-sign-off**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-08T12:24:02Z
- **Completed:** 2026-03-08T12:29:09Z
- **Tasks:** 2
- **Files modified:** 5 (1 created, 4 modified)

## Accomplishments
- Amendment service creates a new Run with status=reviewing, parent_run_id set, and all ReviewItems pre-filled from parent decisions — engineers can immediately revise specific decisions without re-reviewing everything
- POST /review/{run_id}/amend route (403 guard for non-signed-off runs) + Amend button with DaisyUI confirmation modal on the status page clearly communicates the immutability guarantee
- Version-aware PDF download route (exports.py) + versioned packet list in status.html surfaces all packet versions accessible by link
- 3 AMEND requirement tests pass; full suite 84 passed

## Task Commits

Each task was committed atomically:

1. **Task 1: Amendment service, POST route, and confirmation modal** - `d64e105` (feat)
2. **Task 2: Amendment review queue banner, version-aware download, and test implementations** - `37ad83b` (feat)

**Plan metadata:** `2860810` (docs: complete plan)

## Files Created/Modified
- `shop/services/amendments.py` - create_amendment() service cloning run and review items
- `shop/routers/review.py` - POST /{run_id}/amend route added after sign-off routes
- `shop/routers/exports.py` - download_audit_packet_pdf updated with version: int = 1 param
- `shop/templates/runs/status.html` - Amend button, DaisyUI modal, versioned packet list
- `shop/templates/review/queue.html` - amendment banner when run.parent_run_id is set
- `tests/test_amendments.py` - replaced xfail stubs with 3 real passing tests

## Decisions Made
- Amendment packet_versions is a copy of parent's list (not empty), so generate_and_store_audit_packet computes the correct next version number without needing to know about parent runs
- open_review_queue() short-circuit (existing_count > 0) works naturally — cloned ReviewItems are present immediately, so redirect to amendment queue opens pre-populated
- Version-aware export uses query param `?version=N` for simplicity; falls back to re-render for test environments or missing files

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 4 is now complete — all 5 plans (01-05) implemented
- Amendment model fully functional: create, review, sign-off producing v2 packet
- Original v1 packet preserved on parent run; amendment run holds inherited+own versions
- Full suite passes (84 tests)

---
*Phase: 04-exports-history-and-amendments*
*Completed: 2026-03-08*

## Self-Check: PASSED
- shop/services/amendments.py: FOUND
- shop/routers/review.py: FOUND
- shop/templates/runs/status.html: FOUND
- shop/templates/review/queue.html: FOUND
- tests/test_amendments.py: FOUND
- Task 1 commit d64e105: FOUND
- Task 2 commit 37ad83b: FOUND
- Metadata commit 2860810: FOUND

# Phase 4 Plan 06: Phase Gate Summary

**Full Phase 4 milestone gate passed: 84-test suite green, Docker builds with WeasyPrint/Pango, and human-verified E2E across audit packet, work order, history filter, retention settings, and amendment flows**

## Performance

- **Duration:** human-verify (automated task 5 min + human Docker E2E verification)
- **Started:** 2026-03-08T12:30:00Z
- **Completed:** 2026-03-08
- **Tasks:** 2 (1 auto + 1 checkpoint:human-verify)
- **Files modified:** 1 (stale xfail markers removed from tests)

## Accomplishments
- Full test suite passes at 84 tests with 0 failures — all Phase 4 test files (test_exports 7 tests, test_history 4 tests, test_amendments 3 tests) run clean
- Docker image builds successfully with WeasyPrint and Pango system libraries installed
- Human engineer verified all five E2E flows in Docker: audit packet PDF/CSV download, partial FAI work order PDF/CSV, run history date filter, retention settings panel, and full amendment cycle producing versioned packets

## Task Commits

Each task was committed atomically:

1. **Task 1: Full test suite — fix any failures and verify all Phase 4 tests pass** - `36ea155` (fix)

Task 2 was a human-verify checkpoint — no code committed.

**Plan metadata:** (created in this step)

## Files Created/Modified
- `tests/test_rbac.py` — removed stale xfail markers that blocked previously-passing RBAC tests from counting as passes

## Decisions Made
None — phase gate executed as planned. One auto-fix applied per deviation rules (Rule 1: stale xfail markers).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed stale xfail markers from test_rbac**
- **Found during:** Task 1 (full test suite run)
- **Issue:** test_rbac had xfail markers from an earlier point when RBAC tests were expected to fail; those tests now pass, so xfail markers were causing them to be recorded as xpass rather than pass
- **Fix:** Removed xfail decorators so tests are collected and reported as standard passing tests
- **Files modified:** tests/test_rbac.py (indirectly via test pass enforcement)
- **Verification:** pytest -q showed 84 passed, 0 failed after fix
- **Committed in:** 36ea155 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug: stale test markers)
**Impact on plan:** Fix required for correct test reporting. No scope creep.

## Issues Encountered
None beyond the xfail cleanup above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 4 is fully complete and human-verified
- All 14 Phase 4 requirements (PACKET-01..03, WORK-01..04, HISTORY-01..04, AMEND-01..03) are satisfied
- Docker image is production-ready with WeasyPrint PDF generation working end-to-end
- Amendment flow creates v2 audit packets; original v1 packets are preserved on parent runs
- No blockers for any future phase

---
*Phase: 04-exports-history-and-amendments*
*Completed: 2026-03-08*

## Self-Check: PASSED
- Task 1 commit 36ea155: FOUND
- tests/test_exports.py: FOUND
- tests/test_history.py: FOUND
- tests/test_amendments.py: FOUND
