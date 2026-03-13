# S01: Two-step wizard & per-run column mapping

**Goal:** Reduce setup wizard from 4 steps to 2, add per-run xlsx column mapping confirmation to the new run form, remove column mapping from admin settings.
**Demo:** Fresh install → 2-step wizard → login → create new run → upload xlsx → see inline column mapping confirmation → submit run successfully. Admin settings page shows retention only, no column mapping.

## Must-Haves

- Wizard completes in 2 steps: shop name → admin password → redirect to `/login`
- `setup_complete=True` set at end of step 2 POST
- Removed wizard steps (`/setup/step3`, `/setup/step4/*`) redirect to `/login`
- `wizard_layout.html` progress bar shows 2 steps, not 4
- Step 2 button reads "Complete Setup" (not "Continue")
- New run form: xlsx file input triggers HTMX `validate-xlsx` endpoint on change
- `runs/_xlsx_mapping.html` partial: bare select elements inside a div (no `<form>` tag)
- Undetected columns shown with amber indicator and `(ignore)` default
- Admin settings page: no column mapping section, no upload/save routes
- All 87 existing tests pass (adjusted for removed steps); new tests cover 2-step wizard + per-run mapping

## Verification

- `uv run pytest --tb=short` — ≥87 tests pass, 0 failures
- `uv run pytest tests/test_setup.py -v` — wizard tests cover 2-step flow, removed steps redirect
- `uv run pytest tests/test_runs.py -v` — per-run xlsx mapping tests pass
- `docker/docker-compose.yml` builds and fresh container completes 2-step setup (manual check)

## Tasks

- [ ] **T01: Strip wizard to 2 steps and update guard logic** `est:45m`
  - Why: Core wizard simplification — all other tasks depend on setup completing at step 2
  - Files: `shop/routers/setup.py`, `shop/templates/setup/wizard_layout.html`, `shop/templates/setup/step2_password.html`
  - Do: (1) In `setup.py`, change `step2_post` to set `setup_complete=True` and redirect to `/login` instead of `/setup/step3`. (2) Remove `step3_get`, `step3_post`, `step4_get`, `step4_upload`, `step4_save` route handlers. (3) Add catch-all redirects for `/setup/step3` and `/setup/step4/{path:path}` that redirect to `/login` if setup is complete, or `/setup/` if not. (4) Update `setup_root` cap from `min(config.wizard_step + 1, 4)` to `min(config.wizard_step + 1, 2)`. (5) In `wizard_layout.html`, change progress steps to `[(1,'Shop'), (2,'Password')]` and title to "of 2". (6) In `step2_password.html`, change the submit button text to "Complete Setup". (7) Guard logic: treat any `wizard_step >= 2` as setup complete for existing deployments.
  - Verify: `uv run pytest tests/test_setup.py -v` — existing step 1/2 tests pass; step 3/4 tests will fail (expected, fixed in T04)
  - Done when: Fresh setup flow completes in 2 steps; `setup_complete=True` after step 2; removed routes redirect

- [ ] **T02: Add validate-xlsx endpoint and per-run column mapping partial** `est:45m`
  - Why: The new per-run column mapping confirmation — the main new feature of this milestone
  - Files: `shop/routers/runs.py`, `shop/templates/runs/_xlsx_mapping.html` (new), `shop/templates/runs/new.html`
  - Do: (1) In `runs.py`, add `POST /runs/validate-xlsx` endpoint following the `validate_pdf` pattern: auth-gated via `get_current_user`, reads xlsx from multipart form, calls `parse_excel_preview()` + `detect_column_mapping()`, returns `_xlsx_mapping.html` partial on success or `step4_error.html` on failure. (2) Create `runs/_xlsx_mapping.html` as a bare HTML fragment (no `<form>`, no submit button): a table/grid showing each detected column with a `<select>` for field assignment, amber highlighting for undetected columns with `(ignore)` default, info text explaining this is a confirmation of auto-detection. Include hidden inputs carrying the confirmed mapping if needed by future work. (3) In `new.html`, add HTMX attributes to the xlsx file input: `hx-post="/runs/validate-xlsx"`, `hx-trigger="change"`, `hx-encoding="multipart/form-data"`, `hx-target="#xlsx-mapping-section"`, `hx-swap="innerHTML"`. Add `<div id="xlsx-mapping-section"></div>` target below the file input. (4) The endpoint must be resilient to receiving extra form fields (PDF files in the same form).
  - Verify: Manual: start dev server, navigate to `/runs/new`, select an xlsx — mapping panel appears. Automated: new tests in T04.
  - Done when: xlsx file selection on new run form triggers HTMX request and renders column mapping confirmation inline

- [ ] **T03: Remove column mapping from admin settings** `est:15m`
  - Why: Column mapping is now per-run; the admin settings section is dead UI
  - Files: `shop/routers/admin.py`, `shop/templates/admin/settings.html`
  - Do: (1) Remove `settings_upload` and `settings_save` route handlers from `admin.py`. (2) Remove the column mapping card/section from `settings.html` (the `<div class="card">` block containing the xlsx upload form and mapping display). (3) Keep the retention settings section and the settings GET route intact. (4) Delete `shop/templates/setup/step3_engineer.html` and `shop/templates/setup/step4_column_mapping.html` (the step-level wrappers; `step4_mapping_partial.html` and `step4_error.html` are kept — they're reused).
  - Verify: `uv run pytest tests/test_admin.py -v` — admin tests pass; settings page renders without column mapping
  - Done when: Admin settings page shows retention only; removed routes return 405 or are absent

- [ ] **T04: Update test suite for new wizard and per-run mapping** `est:45m`
  - Why: Tests must cover the new behavior and stop testing removed behavior
  - Files: `tests/test_setup.py`, `tests/test_runs.py`
  - Do: (1) In `test_setup.py`: remove `test_setup_step3_blocks_skip` and any tests hitting `/setup/step3` or `/setup/step4/*`. Add test that step 2 POST sets `setup_complete=True` and redirects to `/login`. Add test that `/setup/step3` GET redirects (not 404). Add test that `/setup/step4/upload` GET redirects (not 404). (2) In `test_runs.py`: add `test_validate_xlsx_autodetect` — POST xlsx to `/runs/validate-xlsx`, assert response contains mapping select elements. Add `test_validate_xlsx_empty_file` — POST empty/invalid xlsx, assert error response. Add `test_validate_xlsx_requires_auth` — POST without session cookie, assert 302 to login. Reuse xlsx fixture data from the old `test_setup.py` tests (the openpyxl workbook creation pattern). (3) Use `_make_session_cookie()` pattern from existing tests for auth. (4) Run full suite: `uv run pytest --tb=short` — all ≥87 tests pass.
  - Verify: `uv run pytest --tb=short` — 0 failures, test count ≥87
  - Done when: Full test suite green with coverage of 2-step wizard, removed step redirects, and per-run xlsx mapping

## Files Likely Touched

- `shop/routers/setup.py`
- `shop/routers/runs.py`
- `shop/routers/admin.py`
- `shop/templates/setup/wizard_layout.html`
- `shop/templates/setup/step2_password.html`
- `shop/templates/setup/step3_engineer.html` (deleted)
- `shop/templates/setup/step4_column_mapping.html` (deleted)
- `shop/templates/runs/new.html`
- `shop/templates/runs/_xlsx_mapping.html` (new)
- `shop/templates/admin/settings.html`
- `tests/test_setup.py`
- `tests/test_runs.py`
