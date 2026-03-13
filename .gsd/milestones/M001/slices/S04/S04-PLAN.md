# S04: Exports History And Amendments

**Goal:** Install WeasyPrint, add Dockerfile system dependencies, add new DB columns for amendment and packet versioning, add startup schema migration, and create xfail test stubs for all Phase 4 requirements.
**Demo:** Install WeasyPrint, add Dockerfile system dependencies, add new DB columns for amendment and packet versioning, add startup schema migration, and create xfail test stubs for all Phase 4 requirements.

## Must-Haves


## Tasks

- [x] **T01: 04-exports-history-and-amendments 01** `est:2min`
  - Install WeasyPrint, add Dockerfile system dependencies, add new DB columns for amendment and packet versioning, add startup schema migration, and create xfail test stubs for all Phase 4 requirements.

Purpose: Wave 1 foundation — every subsequent plan needs the schema and dependencies to be present before implementing exports, history, and amendment features.
Output: Updated pyproject.toml, Dockerfile, models.py, database.py, and three test files with xfail stubs.
- [x] **T02: 04-exports-history-and-amendments 02** `est:6min`
  - Build the audit packet export service (PDF + CSV), integrate PDF generation into the two-phase sign-off atomicity, add download routes, and surface download links on the run status page.

Purpose: PACKET-01..03 — engineers can download a formal PDF/CSV audit record after sign-off; re-download is always available from run history.
Output: shop/services/exports.py, shop/routers/exports.py, WeasyPrint template, updated review.py sign-off, updated status.html.
- [x] **T03: 04-exports-history-and-amendments 03**
  - Add the partial FAI work order export (PDF and CSV) and expose it via download routes and a status page button.

Purpose: WORK-01..04 — after sign-off, engineers generate a work order listing only characteristics that need re-measurement or new measurement, formatted for shop-floor use on monochrome printers.
Output: work_order.html WeasyPrint template, generate_work_order_pdf/csv service functions, new routes in exports router, updated status page.
- [x] **T04: 04-exports-history-and-amendments 04**
  - Extend the run history list with a date filter, add read-only mode to the review queue for signed-off runs, add admin retention settings, and implement the Huey periodic cleanup task.

Purpose: HISTORY-01..04 — engineers can browse all runs filtered by date, reopen finalized runs to view decisions, and admins can configure how long unfinished/failed runs are kept.
Output: Updated runs.py (date filter), review.py (read-only mode), admin.py + settings.html (retention UI), tasks.py (cleanup task), updated list.html and test_history.py.
- [x] **T05: 04-exports-history-and-amendments 05** `est:5min`
  - Implement the amendment model: create_amendment service, amend POST route, confirmation modal on the status page, and versioned packet list display. Amendment sign-off reuses the existing attempt_sign_off + generate_and_store_audit_packet path.

Purpose: AMEND-01..03 — engineers can reopen a finalized run to correct review decisions without destroying the original signed packet; each sign-off produces a new versioned packet accessible from the run record.
Output: shop/services/amendments.py, updates to review.py, status.html, queue.html, test_amendments.py.
- [x] **T06: 04-exports-history-and-amendments 06** `est:human-verify`
  - Phase 4 gate: fix any remaining test failures, run the full test suite to green, build Docker with WeasyPrint, and do a human-verified end-to-end flow covering all four feature areas.

Purpose: Confirms the entire phase is production-ready before the milestone closes.
Output: Green test suite, passing Docker build, human-verified e2e flow.

## Files Likely Touched

- `pyproject.toml`
- `docker/Dockerfile`
- `shop/models.py`
- `shop/database.py`
- `tests/test_exports.py`
- `tests/test_history.py`
- `tests/test_amendments.py`
- `shop/services/exports.py`
- `shop/services/review.py`
- `shop/routers/exports.py`
- `shop/app.py`
- `shop/templates/exports/audit_packet.html`
- `shop/templates/runs/status.html`
- `tests/test_exports.py`
- `shop/services/exports.py`
- `shop/routers/exports.py`
- `shop/templates/exports/work_order.html`
- `shop/templates/runs/status.html`
- `tests/test_exports.py`
- `shop/routers/runs.py`
- `shop/routers/admin.py`
- `shop/routers/review.py`
- `shop/tasks.py`
- `shop/templates/runs/list.html`
- `shop/templates/admin/settings.html`
- `tests/test_history.py`
- `shop/services/amendments.py`
- `shop/routers/review.py`
- `shop/templates/runs/status.html`
- `shop/templates/review/queue.html`
- `tests/test_amendments.py`
- `tests/test_exports.py`
- `tests/test_history.py`
- `tests/test_amendments.py`
