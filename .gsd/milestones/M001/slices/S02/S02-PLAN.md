# S02: Pipeline Bridge

**Goal:** Lay the database and task queue foundation that every other Phase 2 plan depends on.
**Demo:** Lay the database and task queue foundation that every other Phase 2 plan depends on.

## Must-Haves


## Tasks

- [x] **T01: 02-pipeline-bridge 01** `est:3min`
  - Lay the database and task queue foundation that every other Phase 2 plan depends on.

Purpose: All subsequent plans require Run/RunAlert models, the Huey instance, and test scaffolding. This plan creates that shared foundation so Plans 02-07 can proceed independently in Wave 2.
Output: Run + RunAlert DB models, SqliteHuey instance in shop/tasks.py, test stubs in tests/test_runs.py, updated conftest.py fixtures, huey added to pyproject.toml.
- [x] **T02: 02-pipeline-bridge 02** `est:6min`
  - Build the upload form, server-side file validation, and run creation so engineers can submit a drawing comparison run.

Purpose: Engineers need to submit Rev A PDF, Rev B PDF, and Form 3 Excel with part metadata before any pipeline can run. This plan delivers the complete submission flow from form to redirect.
Output: /runs/new GET (form) + POST (submit), /runs/validate-pdf HTMX partial (inline PDF validation), Run record created and Huey task enqueued, redirect to /runs/{id} status page.
- [x] **T03: 02-pipeline-bridge 03** `est:7min`
  - Connect the Huey worker to the existing pipeline by adding stage progress callbacks and implementing full failure/warning classification in the task.

Purpose: The pipeline already works standalone. This plan adds the minimal wiring needed for the web app to observe progress and outcomes: a `stage_callback` parameter in `run_pipeline()` and a complete `run_pipeline_task()` implementation in `shop/tasks.py`.
Output: cli.py with stage_callback support, shop/tasks.py with full pipeline task (stage updates, failure handling, warning detection, alert creation).
- [x] **T04: 02-pipeline-bridge 04** `est:8min`
  - Add the persistent nav bar and update the dashboard with run visibility and submission CTA.

Purpose: Engineers need consistent navigation and run visibility from every page. The nav bar (locked decision) and dashboard update are foundational UI changes that all other Phase 2 templates extend.
Output: Updated base.html with nav bar, updated dashboard.html with run cards and Submit CTA, new runs/list.html, GET /runs route in runs router.
- [x] **T05: 02-pipeline-bridge 05** `est:10min`
  - Build the run status page with real-time stage progress via SSE and all three terminal/warning UI states.

Purpose: After submission, engineers need to see live stage progress and clear outcomes for failed, warning, and completed runs. The SSE endpoint bridges the Huey worker's DB updates to the browser in real time.
Output: GET /runs/{id} status page, GET /runs/{id}/sse streaming endpoint, POST /runs/{id}/abort, status.html with three distinct warning states, _stage_checklist.html HTMX partial.
- [x] **T06: 02-pipeline-bridge 06** `est:4min`
  - Wire up in-app failure alerts so the assigned reviewer sees unread failure notifications on the dashboard and in the nav bar badge.

Purpose: PIPE-11 requires the assigned reviewer to receive an in-app alert when their run fails. The RunAlert model and alert creation are in place (Plans 01+03). This plan surfaces those alerts in the UI.
Output: Dashboard alert banners for unread failures, bell badge count in nav bar, HTMX dismiss endpoint, POST /runs/{id}/acknowledge-warning for the revB_balloon warning gate.
- [x] **T07: 02-pipeline-bridge 07** `est:6min`
  - Update the Docker setup to run uvicorn and huey_consumer together under supervisord in the single container.

Purpose: The Huey worker must run as a separate OS process alongside uvicorn. Supervisord is the established pattern for managing multiple processes in a single Docker container (air-gapped deployment constraint).
Output: docker/supervisord.conf with uvicorn + huey_consumer programs, updated Dockerfile with supervisor install and supervisord CMD, updated docker-compose.yml with HUEY_DB env var.
- [x] **T08: 02-pipeline-bridge 08** `est:25min`
  - Implement remaining test stubs, run the full test suite, and perform human Docker verification of the Phase 2 pipeline bridge.

Purpose: This is the Phase 2 gate plan — automated suite must be green and a human must verify the end-to-end submission flow in Docker before Phase 3 begins.
Output: All test_runs.py stubs implemented and passing, full test suite green, Docker e2e verification approved by engineer.

## Files Likely Touched

- `shop/models.py`
- `shop/tasks.py`
- `tests/conftest.py`
- `tests/test_runs.py`
- `pyproject.toml`
- `shop/routers/runs.py`
- `shop/services/runs.py`
- `shop/app.py`
- `shop/templates/runs/new.html`
- `shop/templates/runs/_page_selector.html`
- `shop/templates/runs/_pdf_error.html`
- `delta_preservation/cli.py`
- `shop/tasks.py`
- `shop/templates/base.html`
- `shop/templates/dashboard.html`
- `shop/routers/runs.py`
- `shop/templates/runs/list.html`
- `shop/routers/runs.py`
- `shop/templates/runs/status.html`
- `shop/templates/runs/_stage_checklist.html`
- `shop/routers/runs.py`
- `shop/templates/dashboard.html`
- `shop/templates/runs/_alert_banner.html`
- `docker/Dockerfile`
- `docker/docker-compose.yml`
- `docker/supervisord.conf`
- `pyproject.toml`
- `shop/templates/setup`
