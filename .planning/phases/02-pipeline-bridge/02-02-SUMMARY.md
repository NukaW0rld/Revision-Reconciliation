---
phase: 02-pipeline-bridge
plan: 02
subsystem: upload
tags: [fastapi, htmx, pymupdf, fitz, openpyxl, file-upload, pdf-validation, run-creation]

# Dependency graph
requires:
  - phase: 02-01
    provides: "Run SQLAlchemy model, run_pipeline_task Huey stub, engineer_user + huey_immediate fixtures"
  - phase: 01-foundation
    provides: "User model, get_current_user dependency, templates, TestClient pattern"
provides:
  - "shop/services/runs.py with save_upload(), validate_pdf_bytes(), validate_excel_bytes(), create_run()"
  - "shop/routers/runs.py: GET /runs/new, POST /runs/validate-pdf, POST /runs/new"
  - "shop/templates/runs/new.html: upload form with HTMX inline PDF validation"
  - "shop/templates/runs/_page_selector.html: page selector partial for multi-page PDFs"
  - "shop/templates/runs/_pdf_error.html: error partial for raster PDF rejection"
  - "UPLOADS_DIR env-var pattern with local fallback for dev/test environments"
affects:
  - 02-03 (pipeline task — reads revA_path, revB_path, form3_path, part_number from Run)
  - 02-04 (SSE status — reads Run.status from runs created here)
  - 02-05 (upload validation — these are the upload routes being validated)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "UPLOADS_DIR env var with local fallback: same pattern as HUEY_DB from Plan 01"
    - "HTMX inline validation: hx-post on file input change event, hx-encoding=multipart/form-data, hx-vals for field name"
    - "raster PDF detection: text extraction primary check, image area >= 95% fallback"
    - "Deferred service imports inside router handlers to avoid circular imports"

key-files:
  created:
    - shop/services/runs.py
    - shop/routers/runs.py
    - shop/templates/runs/new.html
    - shop/templates/runs/_page_selector.html
    - shop/templates/runs/_pdf_error.html
    - tests/test_runs_service.py
  modified:
    - tests/test_runs.py

key-decisions:
  - "UPLOADS_DIR local fallback: same pattern as HUEY_DB from Plan 01 — falls back to project-root uploads/ when /app/data does not exist (dev/test)"
  - "Raster detection: text extraction is primary signal (if text found, it's vector); image area >= 95% is secondary heuristic for pages with no text"
  - "save_upload() takes file_bytes + suffix + run_uuid (not UploadFile) — enables testing without FastAPI UploadFile stubs"
  - "revA_page and revB_page default to 0 on POST /runs/new — single-page PDFs never trigger the page selector, so the default hidden input covers them"

patterns-established:
  - "Service function signature: save_upload(bytes, suffix, run_uuid) rather than save_upload(UploadFile, dest_dir) — easier to unit-test"
  - "Router reads file bytes before calling services — service functions are sync and pure (no async file IO)"

requirements-completed: [UPLOAD-01, UPLOAD-02, UPLOAD-03, UPLOAD-04, UPLOAD-05]

# Metrics
duration: 6min
completed: 2026-03-04
---

# Phase 2 Plan 2: Upload Form and Run Creation Summary

**Upload form with HTMX inline PDF validation, raster detection via PyMuPDF, run creation in DB, and Huey task enqueue on POST /runs/new**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-04T01:30:40Z
- **Completed:** 2026-03-04T01:36:52Z
- **Tasks:** 3 (+ TDD RED commits)
- **Files modified:** 7

## Accomplishments

- `shop/services/runs.py`: four service functions — save_upload(), validate_pdf_bytes() with raster heuristic, validate_excel_bytes(), create_run()
- `shop/routers/runs.py`: three upload routes — GET /runs/new form, POST /runs/validate-pdf HTMX partial, POST /runs/new full submit with validation/save/create/enqueue/redirect
- Upload form templates: new.html (HTMX wired, DaisyUI styled), _page_selector.html (multi-page dropdown), _pdf_error.html (raster error alert)
- 16 tests passing: 11 unit tests in test_runs_service.py, 5 UPLOAD requirement tests in test_runs.py (UPLOAD-01..05)

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for shop/services/runs.py** - `98eb1fc` (test)
2. **Task 1 GREEN: Implement shop/services/runs.py** - `6c85199` (feat)
3. **Task 2 RED: Implement UPLOAD-01..05 tests in test_runs.py** - `02f68dc` (test)
4. **Task 2 GREEN: Create runs router with upload routes** - `efe902c` (feat)
5. **Task 3: Create upload form templates** - `dc18307` (feat)

## Files Created/Modified

- `shop/services/runs.py` - Created: save_upload, validate_pdf_bytes, validate_excel_bytes, create_run
- `shop/routers/runs.py` - Created: GET /runs/new, POST /runs/validate-pdf, POST /runs/new (extends list_runs from prior partial)
- `shop/templates/runs/new.html` - Created: upload form with HTMX validation
- `shop/templates/runs/_page_selector.html` - Created: page selector partial for multi-page PDFs
- `shop/templates/runs/_pdf_error.html` - Created: error partial for raster PDF rejection
- `tests/test_runs_service.py` - Created: 11 unit tests for service functions
- `tests/test_runs.py` - Modified: implemented UPLOAD-01..05 as real tests (replaced xfail stubs)

## Decisions Made

- **UPLOADS_DIR local fallback:** When `/app/data` doesn't exist (dev/test), falls back to project-root `uploads/` — same pattern as HUEY_DB fallback in Plan 01.
- **Service bytes API:** `save_upload(file_bytes, suffix, run_uuid)` instead of `save_upload(UploadFile, dest_dir)` — easier to unit-test without FastAPI UploadFile stubs; router reads bytes before calling service.
- **Raster detection order:** Text extraction is the primary vector signal; image area coverage is only evaluated when no text is found. This avoids false positives on PDFs with embedded images plus text annotations.
- **Default page 0:** revA_page and revB_page default to 0 on POST /runs/new — single-page PDFs never trigger the page selector, so the hidden input (value=0) covers them without JavaScript.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] UPLOADS_DIR default causes PermissionError outside Docker**
- **Found during:** Task 2 (runs router) — first full test run hit the `/app/data/uploads` path
- **Issue:** `save_upload()` tries to `mkdir(parents=True)` at `/app/data/uploads/{run_uuid}` which raises `PermissionError: [Errno 13] Permission denied: '/app'` in dev/test environments
- **Fix:** Added same fallback pattern as `HUEY_DB` from Plan 01: check if `Path(_default_uploads_dir).parent.exists()`; fall back to project-root `uploads/` if not
- **Files modified:** `shop/services/runs.py`
- **Verification:** All UPLOAD tests pass with in-memory DB and tmp filesystem
- **Committed in:** `efe902c` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug: permission error outside Docker)
**Impact on plan:** Required for any non-Docker execution environment. No scope creep.

## Issues Encountered

- Pre-existing `test_pipeline_task.py` failures: 9 tests in the RED phase of Plan 03 (not yet implemented) were already failing before this plan ran. These are out of scope and not caused by Plan 02-02 changes. The Plan 02-02 UPLOAD tests all pass.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `POST /runs/new` creates Run and enqueues `run_pipeline_task` — Plan 03 replaces the task stub with real pipeline orchestration
- `GET /runs/{id}` status page not yet implemented — redirect from POST /runs/new will 404 until Plan 04 adds the detail page (acceptable for now)
- Upload form is fully functional and testable; engineers can submit runs once the status page exists

---
*Phase: 02-pipeline-bridge*
*Completed: 2026-03-04*
