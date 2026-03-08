---
phase: 02-pipeline-bridge
verified: 2026-03-04T11:30:00Z
status: human_needed
score: 16/16 must-haves verified
re_verification:
  previous_status: human_needed
  previous_score: 16/16
  gaps_closed: []
  gaps_remaining: []
  regressions: []
gaps: []
human_verification:
  - test: "Submit a run end-to-end in Docker: visit /runs/new, upload 3 files, submit"
    expected: "Redirect to /runs/{id} status page; stage checklist advances in real time via SSE"
    why_human: "SSE real-time behavior and Docker process startup (uvicorn + huey_consumer under supervisord) cannot be verified programmatically"
  - test: "Trigger a run failure and observe the dashboard as the assigned reviewer"
    expected: "Bell badge shows count; alert banner appears at dashboard top with dismiss button"
    why_human: "Alert dismissal and badge decrement require live browser interaction to verify HTMX swap behavior"
  - test: "Upload a multi-page PDF to the /runs/new form"
    expected: "Page selector partial appears inline without a page reload"
    why_human: "HTMX hx-trigger='change' + hx-post inline swap requires browser interaction to verify"
  - test: "Submit a run that triggers the low-confidence warning state"
    expected: "Status page shows 'Low Alignment Confidence' warning banner with 'Proceed to Review' and 'Abort Run' buttons"
    why_human: "Warning state render after SSE close + page reload requires live Docker environment"
---

# Phase 02: Pipeline Bridge Verification Report

**Phase Goal:** Engineers can submit a drawing comparison run and watch it progress stage by stage to completion or failure, with upload validation catching bad files before the pipeline starts
**Verified:** 2026-03-04T11:30:00Z
**Status:** human_needed — all automated checks pass; 4 items require live browser/Docker verification
**Re-verification:** Yes — second re-verification after previous human_needed (16/16). Full codebase scan performed from scratch.

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Engineer can POST to /runs/new with 3 files + metadata and be redirected to /runs/{id} | VERIFIED | `shop/routers/runs.py:submit_run()` validates files, calls `create_run()`, enqueues task, returns `RedirectResponse(f"/runs/{run.id}", status_code=302)`; `test_upload_creates_run` passes |
| 2 | Raster PDF uploaded to /runs/validate-pdf returns an error partial | VERIFIED | `validate_pdf_bytes()` in `shop/services/runs.py` detects raster via image-area heuristic; router returns `_pdf_error.html`; `test_raster_pdf_rejected` passes |
| 3 | Multi-page PDF uploaded to /runs/validate-pdf returns a page selector partial inline | VERIFIED | Router returns `_page_selector.html` when `page_count > 1`; template exists; `test_multipage_pdf_page_selector` passes |
| 4 | Unreadable or empty Excel returns a clear error before run is created | VERIFIED | `validate_excel_bytes()` raises `ValueError` routed as 422 with form + error; `test_invalid_excel_rejected` passes |
| 5 | Run record in DB has rev_a_label and rev_b_label matching form submission | VERIFIED | `create_run()` sets both columns; `test_rev_labels_stored` asserts `run.rev_a_label == "C"` and `run.rev_b_label == "D"` |
| 6 | run_pipeline() accepts optional stage_callback and calls it before each of the 8 stages | VERIFIED | `delta_preservation/cli.py` has `stage_callback: Optional[Callable[[int, str], None]] = None` in signature at line 56; 8 `if stage_callback: stage_callback(N, "name")` calls confirmed at lines 114-115, 122-123, 129-130, 136-137, 143-144, 185-186, 197-198, 499-500 |
| 7 | run_pipeline_task Huey task calls run_pipeline() with stage_callback that updates Run DB row | VERIFIED | `shop/tasks.py` defines `stage_callback()` closure, passes it to `_run_pipeline()`; `_update_stage()` commits `current_stage` and `current_stage_index` |
| 8 | RevA balloon failure sets Run.status=failed, Run.failure_stage, Run.failure_message | VERIFIED | `tasks.py` checks `len(items) == 0` after pipeline; sets `status="failed"`, `failure_stage="Balloon detection"`, descriptive `failure_message`; `test_revA_balloon_failure` passes |
| 9 | RevB balloon failure sets Run.status=warning via low-confidence unified path | VERIFIED | `tasks.py` computes `low_conf_ratio`; when >0.5, sets `status="warning"`, `warning_type="low_confidence"`; `test_revB_balloon_warning` passes |
| 10 | Alignment low-confidence sets Run.status=warning, warning_type=low_confidence with JSON confidence_summary | VERIFIED | Same code path; `confidence_summary` dict populated with `low_confidence_count`, `total`, `low_confidence_ratio`, `message`; `test_low_confidence_warning` passes |
| 11 | Successful completion sets Run.status=completed and Run.output_dir | VERIFIED | `tasks.py` success path: `run.status = "completed"`, `run.output_dir = str(out_dir)`, commit |
| 12 | On failure, RunAlert is created for the reviewer | VERIFIED | `_create_alert()` creates `RunAlert` with `run_id`, `user_id=run.reviewer_id`, `message`, `failure_stage`; called in both empty-items and exception paths; `test_failure_alert_created` passes |
| 13 | Run status page shows numbered stage checklist with correct visual states | VERIFIED | `_stage_checklist.html` renders 8-step `<ul class="steps steps-vertical">` with `step-success`/`step-primary`/`step-error` classes; `status.html` includes it via `{% include %}`; `test_stage_progress_updates` asserts `"stage-checklist" in resp.text` |
| 14 | SSE endpoint streams stage updates; closes on terminal state | VERIFIED | `GET /runs/{id}/sse` async generator yields `ServerSentEvent(raw_data=payload, event="stage_update")` every 1s; yields `ServerSentEvent(event="close", raw_data="done")` and breaks when `run.status in terminal`; `test_stage_progress_updates` asserts 200 on SSE endpoint |
| 15 | Bell icon shows red badge with unread alert count; alerts appear on dashboard; dismiss clears badge | VERIFIED | `base.html` has badge conditional on `unread_alert_count > 0`; dashboard route passes `unread_alerts` list; `_alert_banner.html` renders with `hx-post="/alerts/dismiss/{{alert.id}}"` HTMX dismiss; `auth.py` has `POST /alerts/dismiss/{id}` route marking `is_read=True`; `test_failure_alert_created` asserts 200 + empty body |
| 16 | Docker runs both uvicorn and huey_consumer under supervisord | VERIFIED | `docker/supervisord.conf` has `[program:uvicorn]` and `[program:huey_worker]` with `huey_consumer.py shop.tasks.huey --workers=1 --worker-type=thread`; `Dockerfile` installs `supervisor` via apt and uses `CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]`; `docker-compose.yml` has `HUEY_DB`, `OUT_DIR`, `UPLOADS_DIR` |

**Score:** 16/16 truths verified (automated)

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `shop/models.py` | Run and RunAlert SQLAlchemy models | VERIFIED | `class Run` with 20 columns (id, part_number, rev_a_label, rev_b_label, customer, job_number, status, current_stage, current_stage_index, failure_stage, failure_message, warning_type, confidence_summary, output_dir, revA_path, revB_path, form3_path, revA_page, revB_page, submitted_at, reviewer_id); `class RunAlert` with FK to both runs and users; 93 lines total |
| `shop/tasks.py` | SqliteHuey instance + full pipeline task | VERIFIED | 248 lines; `huey = SqliteHuey(filename=HUEY_DB)`; `run_pipeline_task` fully implemented with stage_callback, failure/warning/success paths, `_create_alert()`, `_update_stage()` |
| `shop/services/runs.py` | save_upload, create_run, validate_pdf_bytes, validate_excel_bytes | VERIFIED | All 4 functions present and substantive; dev/test path fallback for UPLOADS_DIR; `validate_pdf_bytes` uses text extraction + image area heuristic; 101 lines |
| `shop/routers/runs.py` | GET/POST /runs/new, POST /validate-pdf, GET /{id}, GET /{id}/sse, POST /{id}/abort, POST /{id}/acknowledge-warning | VERIFIED | 308 lines; all 6 routes present and implemented; router registered in `app.py` via `app.include_router(runs.router, prefix="/runs")` |
| `shop/templates/runs/new.html` | Upload form with HTMX inline validation | VERIFIED | Extends base.html; 3 file pickers; `hx-post="/runs/validate-pdf"` on revA and revB PDF inputs with `hx-trigger="change"`; error display via `{% if error %}` block |
| `shop/templates/runs/status.html` | Status page with stage checklist, warning states, SSE JS | VERIFIED | 198 lines; includes `_stage_checklist.html`; renders all 3 terminal states (failed/revB_balloon/low_confidence); native `EventSource` JS at line 118; `injectTerminalBanner()` + 500ms reload |
| `shop/templates/runs/_stage_checklist.html` | Stage checklist partial | VERIFIED | 21 lines; DaisyUI `steps steps-vertical` with 8 stages; correct conditional classes (`step-success`, `step-primary`, `step-error`) |
| `shop/templates/runs/_alert_banner.html` | Dismissible alert banner partial | VERIFIED | Renders part number, run ID, failure stage, View run link, Dismiss button with `hx-post="/alerts/dismiss/{{ alert.id }}"` targeting `#alert-banner-{{ alert.id }}` for outerHTML swap |
| `shop/templates/runs/list.html` | Run list with filter | VERIFIED | GET filter form with part_number input; table with status badges, part_number, revision, submitted_at, reviewer, view link; empty state for both filtered and unfiltered cases |
| `shop/templates/dashboard.html` | Dashboard with recent runs + Submit New Run CTA | VERIFIED | "Submit New Run" btn-primary btn-lg; `{% for alert in unread_alerts %}` banner block; `{% for run in recent_runs %}` cards with status badges |
| `shop/templates/base.html` | Persistent nav bar | VERIFIED | Sticky navbar; "New Run" (btn-primary) and "My Runs" (btn-ghost) links; alert bell badge conditional on `unread_alert_count > 0`; user email; logout button; all conditional on `{% if user %}` |
| `delta_preservation/cli.py` | run_pipeline() with stage_callback parameter | VERIFIED | `stage_callback: Optional[Callable[[int, str], None]] = None` in signature at line 56; called at all 8 stages (lines 114-115, 122-123, 129-130, 136-137, 143-144, 185-186, 197-198, 499-500) |
| `docker/supervisord.conf` | supervisord config with uvicorn + huey_worker | VERIFIED | Both programs defined; `[program:uvicorn]` with uvicorn startup; `[program:huey_worker]` with `huey_consumer.py shop.tasks.huey --workers=1 --worker-type=thread` |
| `docker/Dockerfile` | Dockerfile with supervisor install + supervisord CMD | VERIFIED | `apt-get install supervisor`; 3-stage build (tailwind-builder, python-builder, runtime); `COPY docker/supervisord.conf /etc/supervisor/conf.d/delta.conf`; `CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]` |
| `docker/docker-compose.yml` | compose with HUEY_DB, OUT_DIR, UPLOADS_DIR | VERIFIED | All 3 env vars present with correct paths; persistent data and out volume mounts |
| `tests/test_runs.py` | 12 implemented tests (no xfail) | VERIFIED | 12 substantive test functions; no `@pytest.mark.xfail` decorators; all 12 pass confirmed via `uv run pytest tests/test_runs.py` (12 passed) |
| `tests/conftest.py` | engineer_user + huey_immediate fixtures | VERIFIED | `engineer_user` fixture at line 113 seeds active engineer; `huey_immediate` fixture at line 133 sets `huey.immediate = True` and flushes storage on teardown |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `shop/tasks.py` | `shop/models.py` | `from shop.models import RunAlert` inside task body | VERIFIED | Deferred import inside `_create_alert()` and `run_pipeline_task()` body; `Run` imported at top of task body |
| `shop/tasks.py` | `delta_preservation/cli.py` | module-level `run_pipeline` + lazy `_get_run_pipeline()` | VERIFIED | Module-level import with `try/except ImportError` fallback; `_run_pipeline()` called at line 166 |
| `shop/routers/runs.py` | `shop/services/runs.py` | `from shop.services.runs import ...` inside `submit_run()` | VERIFIED | Deferred import of all 4 service functions at line 144; all used within `submit_run()` |
| `shop/routers/runs.py` | `shop/tasks.py` | `run_pipeline_task(run.id, ...)` | VERIFIED | Line 208: `run_pipeline_task(run.id, str(revA_path), str(revB_path), str(form3_path), part_number)` called after run creation |
| `shop/templates/runs/new.html` | `/runs/validate-pdf` | `hx-post="/runs/validate-pdf"` on file inputs | VERIFIED | Lines 59 and 85 in new.html on both revA_pdf and revB_pdf file inputs |
| `shop/templates/runs/status.html` | `/runs/{id}/sse` | `new EventSource("/runs/{{ run.id }}/sse")` | VERIFIED | Line 118 in status.html; `stage_update` listener at line 123; `close` listener with reload at line 129 |
| `shop/templates/dashboard.html` | `shop/routers/auth.py` | `recent_runs` and `unread_alerts` context variables | VERIFIED | `auth.py` dashboard route queries both at lines 94-97 and passes to template |
| `shop/templates/runs/_alert_banner.html` | `/alerts/dismiss/{id}` | `hx-post="/alerts/dismiss/{{ alert.id }}"` | VERIFIED | `auth.py` has `POST /alerts/dismiss/{alert_id}` route at line 73; marks `is_read=True` |
| `shop/templates/base.html` | `/runs/new` | nav bar "New Run" link | VERIFIED | `<a href="/runs/new" class="btn btn-primary btn-sm">New Run</a>` at line 19 |
| `docker/supervisord.conf` | `shop/tasks.py` | `huey_consumer.py shop.tasks.huey` | VERIFIED | Matches task registry `huey = SqliteHuey(...)` in `shop/tasks.py` |
| `docker/Dockerfile` | `docker/supervisord.conf` | `COPY docker/supervisord.conf /etc/supervisor/conf.d/delta.conf` + `CMD supervisord` | VERIFIED | Lines 37 and 45 in Dockerfile |
| `shop/app.py` | `shop/routers/runs.py` | `app.include_router(runs.router, prefix="/runs")` | VERIFIED | Line 49 in app.py; confirmed by full test suite (all runs routes respond) |

---

## Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|---------------|-------------|--------|----------|
| UPLOAD-01 | 02-02, 02-04, 02-08 | Engineer can upload Rev A PDF, Rev B PDF, Form 3 Excel with part metadata | SATISFIED | `POST /runs/new` creates Run with all 5 metadata fields; `test_upload_creates_run` passes; `test_rev_labels_stored` asserts label persistence |
| UPLOAD-02 | 02-02, 02-08 | Raster PDF detected and error shown before run submitted | SATISFIED | `validate_pdf_bytes()` image-area heuristic + `_pdf_error.html` partial; `test_raster_pdf_rejected` passes |
| UPLOAD-03 | 02-02, 02-08 | Multi-page PDF triggers page selector before run starts | SATISFIED | `_page_selector.html` returned by `/runs/validate-pdf` when page_count > 1; `test_multipage_pdf_page_selector` passes |
| UPLOAD-04 | 02-02, 02-08 | Excel validated as readable before upload accepted | SATISFIED | `validate_excel_bytes()` raises ValueError; 422 returned with error in form; `test_invalid_excel_rejected` passes |
| UPLOAD-05 | 02-02, 02-08 | Revision labels stored as part of run record | SATISFIED | `Run.rev_a_label` and `Run.rev_b_label` persisted via `create_run()`; `test_rev_labels_stored` asserts `run.rev_a_label == "C"` and `run.rev_b_label == "D"` |
| PIPE-01 | 02-01, 02-03, 02-07, 02-08 | Pipeline executes asynchronously in worker process | SATISFIED | Huey SqliteHuey task; supervisord runs `huey_consumer.py` as separate OS process in Docker; `test_pipeline_task_enqueued` passes |
| PIPE-02 | 02-03, 02-05, 02-08 | Stage-by-stage progress shown with current stage name | SATISFIED | `_update_stage()` commits `current_stage`/`current_stage_index`; SSE streams updates; stage checklist rendered; `test_stage_progress_updates` asserts stage-checklist and "Anchor building" in response |
| PIPE-03 | 02-01, 02-03, 02-08 | System distinguishes queued/running/completed/failed states | SATISFIED | All 4 statuses implemented in `run_pipeline_task`; `test_run_status_lifecycle` verifies completed, failed, and warning terminal states via HTTP |
| PIPE-04 | 02-03, 02-05, 02-08 | Rev A balloon detection failure = hard fail with error message | SATISFIED | Empty items check in `tasks.py` sets `status="failed"`, `failure_stage="Balloon detection"`, descriptive `failure_message`; status page shows red alert; `test_revA_balloon_failure` passes |
| PIPE-05 | 02-03, 02-05, 02-08 | Rev B balloon failure = partial-result warning; engineer decides before review | SATISFIED | Unified low_confidence path (near-zero alignment scores); status page shows yellow warning with Proceed/Abort buttons; `test_revB_balloon_warning` passes |
| PIPE-06 | 02-03, 02-05, 02-08 | Low alignment confidence = warning state with confidence distribution + proceed/abort | SATISFIED | `confidence_summary` JSON stored; status page shows message + "Proceed to Review"/"Abort Run" buttons; `test_low_confidence_warning` passes |
| PIPE-07 | 02-03 | Characteristics not located in Rev B: auto-classified as removed (conf >= 0.9) or unresolved (conf < 0.9) | SATISFIED | Handled by `delta_preservation/reconcile/classify.py` existing logic; noted in `tasks.py` lines 128-132 comment |
| PIPE-08 | 02-03 | Balloon found in Rev B but text extraction fails: low-confidence unchanged or unresolved | SATISFIED | Handled by existing `classify.py` logic; noted in `tasks.py` comment |
| PIPE-09 | 02-03 | Duplicate balloon numbers treated as one characteristic; count token changes tracked | SATISFIED | Handled by `normalize.py` `count_tokens` in MatchFingerprint; noted in `tasks.py` comment |
| PIPE-10 | 02-03 | GD&T feature control frames matched as opaque strings | SATISFIED | Handled by existing text matching in `classify.py`; noted in `tasks.py` comment |
| PIPE-11 | 02-01, 02-06, 02-08 | Reviewer receives in-app alert when their run fails | SATISFIED | `_create_alert()` creates RunAlert; dashboard shows banners; `POST /alerts/dismiss/{id}` marks read; `test_failure_alert_created` passes |

**All 16 Phase 2 requirements (UPLOAD-01..05, PIPE-01..11) are SATISFIED.**

Note: PIPE-07, PIPE-08, PIPE-09, PIPE-10 are handled by the existing pipeline classification logic (`delta_preservation/reconcile/classify.py`), not by Phase 2 web UI code. This is explicitly documented in `tasks.py` lines 128-132 and was the agreed design in 02-03-PLAN.md.

No orphaned requirements found. The traceability table in REQUIREMENTS.md confirms UPLOAD-01..05 and PIPE-01..11 are all mapped to Phase 2 exclusively (16 requirements total).

---

## Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `shop/routers/runs.py:306` | `acknowledge_warning()` redirects to `/review/{run_id}` (Phase 3 placeholder) | INFO | Expected — Phase 3 implements the review queue; placeholder is documented in the route docstring |
| `shop/templates/runs/status.html:72,90,102` | "Open Review Queue" / "Proceed to Review" / "Acknowledge" links go to `/review/{run.id}` (404) | INFO | Documented Phase 3 placeholder per 02-05-PLAN.md; links are intentional stubs until Phase 3 |
| `tests/test_runs.py:4,247` | Stale comments ("PIPE-* remain as xfail stubs") | INFO | Comments are outdated — all 12 tests are implemented and pass; comments are harmless documentation noise |

No blocker anti-patterns found. All `/review/` placeholders are scoped to Phase 3 and explicitly documented. The stale comments in test_runs.py do not affect correctness.

---

## Test Suite Status

Full suite run confirmed: **60 passed, 2 xfailed, 3 xpassed, 11 warnings**

- `tests/test_runs.py`: 12 passed (0 xfail, no stubs)
- Full test suite: 60 passed
- The 2 xfailed are in `test_auth.py` and `test_models.py` (pre-existing Phase 1 stubs, not Phase 2)
- The 3 xpassed are in `test_rbac.py` (pre-existing Phase 1 stubs that now pass, not Phase 2)

---

## Human Verification Required

### 1. End-to-End Run Submission in Docker

**Test:** `docker compose -f docker/docker-compose.yml build && docker compose -f docker/docker-compose.yml up`, complete setup wizard, visit `/runs/new`, upload 3 valid files + metadata, submit.
**Expected:** Redirect to `/runs/{id}`; within seconds stage checklist advances (spinner on current stage, checkmarks on completed stages); run reaches completed state after all 8 stages.
**Why human:** SSE real-time rendering requires a live browser. Docker supervisord process startup (uvicorn + huey_consumer) must be observed via Docker logs to confirm both processes start without import errors.

### 2. Failure Alert Flow

**Test:** Trigger a run failure (submit a run that will fail at balloon detection); observe dashboard as the assigned reviewer.
**Expected:** Bell badge shows count > 0; red alert banner appears at dashboard top with part number, run ID, failure stage, and "View run details" link; click "Dismiss" — banner disappears (HTMX outerHTML swap removes the `#alert-banner-{id}` div); badge count decrements on next page load.
**Why human:** HTMX `hx-swap="outerHTML"` dismiss and badge decrement require live browser DOM manipulation to verify end-to-end behavior.

### 3. Multi-Page PDF Inline Page Selector

**Test:** On `/runs/new`, select a multi-page PDF for the Rev A upload field.
**Expected:** Page selector dropdown appears inline inside the `#revA-section` div without a page reload; allows selecting a page number before submission.
**Why human:** HTMX `hx-trigger="change"` file upload behavior (form data encoding + target swap) requires a real browser to verify the inline swap targets `#revA-section` correctly.

### 4. Low-Confidence Warning State and Abort

**Test:** Submit a run that produces low alignment confidence (>50% of items with location score < 0.5 from a real PDF pair).
**Expected:** Status page shows "Low Alignment Confidence" warning banner with confidence ratio message and two buttons: "Proceed to Review" and "Abort Run"; clicking "Abort Run" submits the form, redirects to status page showing failed state with failure_stage="Alignment".
**Why human:** The low-confidence trigger requires a real pipeline run in Docker with actual PDFs that produce low alignment scores; cannot be verified with synthetic test data in a browser.

---

## Gaps Summary

No gaps. All automated checks pass. The phase goal is fully implemented and verified programmatically. Four items require human verification in a Docker environment for complete confidence on real-time SSE behavior, HTMX DOM swaps, and Docker process orchestration.

**Re-verification result:** All 16 truths from the previous verification remain verified. No regressions detected. No previously-passing items have broken. Full test suite remains at 60 passed, 2 xfailed, 3 xpassed.

---

_Verified: 2026-03-04T11:30:00Z_
_Verifier: Claude (gsd-verifier)_
_Re-verification: previous status was human_needed with 16/16 automated checks passed_
