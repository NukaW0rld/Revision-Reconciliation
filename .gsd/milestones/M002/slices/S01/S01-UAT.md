# S01: Two-step wizard & per-run column mapping — UAT

**Milestone:** M002
**Written:** 2026-03-13

## UAT Type

- UAT mode: artifact-driven
- Why this mode is sufficient: All behavior is covered by contract tests (`uv run pytest --tb=short` — 93 passed, 0 failures). The per-run mapping confirmation is UI-only and doesn't feed the pipeline, so no operational integration test is required beyond the visual browser check completed during T02. Docker build verification is the remaining operational gate.

## Preconditions

- `uv sync` completed; Python env has all dependencies
- `assets/part1/FAIR.xlsx` present (used as the valid xlsx fixture)
- Dev server NOT running (tests use TestClient); OR dev server running at `http://localhost:8000` for browser checks
- Docker: `docker/docker-compose.yml` and `docker/Dockerfile` present for operational check (optional gate)

## Smoke Test

Run `uv run pytest --tb=short` — expect output ending in `93 passed, 2 xfailed, 0 failures`. Any failure means the slice is broken.

## Test Cases

### 1. Fresh setup completes in exactly 2 steps

1. Start dev server: `uv run python run_web.py`
2. Navigate to `http://localhost:8000/setup/step1`
3. Enter a shop name, submit
4. **Expected:** Redirected to `/setup/step2` (not step3); progress bar shows step 1 of 2
5. Enter an admin password, confirm it, submit
6. **Expected:** Redirected to `/login`; progress bar was showing step 2 of 2 with "Complete Setup" button; submit button text was "Complete Setup" (not "Continue")
7. Log in with `admin@shop.local` and the chosen password
8. **Expected:** Login succeeds; lands on dashboard

### 2. Removed wizard steps redirect gracefully (not 404)

1. Complete setup (or use a seeded test DB with `setup_complete=True`)
2. Navigate to `GET http://localhost:8000/setup/step3`
3. **Expected:** 302 redirect to `/login` (not 404, not a wizard page)
4. Navigate to `GET http://localhost:8000/setup/step4/upload`
5. **Expected:** 302 redirect to `/login`
6. `POST http://localhost:8000/setup/step3` (curl or dev tools)
7. **Expected:** 302 redirect to `/login`

Automated equivalent: `uv run pytest tests/test_setup.py::test_removed_step4_redirects_when_complete -v`

### 3. Removed wizard steps redirect to /setup/ when setup incomplete

1. Use `client_setup_incomplete` fixture behavior: DB has no `setup_complete=True` row
2. `GET /setup/step3`
3. **Expected:** 302 redirect to `/setup/` (not to `/login`, not 404)

Automated: `uv run pytest tests/test_setup.py::test_removed_step4_redirects_when_incomplete -v`

### 4. setup_complete=True set in DB after step 2

1. Run: `uv run pytest tests/test_setup.py::test_setup_step2_creates_admin -v`
2. **Expected:** PASSED — test verifies `setup_complete=True` in DB, `wizard_step=2`, and redirect to `/login`

### 5. New run form — xlsx triggers inline column mapping panel

1. Log in as an engineer or admin
2. Navigate to `/runs/new`
3. Upload `assets/part1/FAIR.xlsx` to the "Form 3 Excel" file input
4. **Expected:** Within ~1 second, `#xlsx-mapping-section` below the xlsx input populates with a table showing one `<select>` per column header. Detected columns (e.g., `char_no`, `nominal`, `upper_tol`, `lower_tol`) are pre-selected to their field name. Undetected columns show amber background with `(ignore)` as the selected option. Five preview data rows are visible below the header selects.

### 6. New run form — invalid xlsx shows error in place

1. Log in
2. Navigate to `/runs/new`
3. Create a plain-text file named `bad.xlsx` and upload it to the Form 3 input
4. **Expected:** `#xlsx-mapping-section` renders an error card ("Cannot read file: File is not a zip file" or similar) with instruction to correct and re-upload. No crash, no blank page.

Automated: `uv run pytest tests/test_runs.py::test_validate_xlsx_empty_file -v`

### 7. validate-xlsx requires authentication

1. Open a fresh browser session (no login) or use curl with no session cookie
2. `POST http://localhost:8000/runs/validate-xlsx` with any xlsx file
3. **Expected:** 302 redirect to `/login` (not 200, not 500)

Automated: `uv run pytest tests/test_runs.py::test_validate_xlsx_requires_auth -v`

### 8. Admin settings page has no column mapping section

1. Log in as admin
2. Navigate to `/admin/settings`
3. **Expected:** Page renders with Shop Name and Run Retention sections. No xlsx upload form, no column mapping table or card. Page does not 500.

Automated: `uv run pytest tests/test_admin.py -v` (all 5 pass)

### 9. Mapping partial is a bare fragment (no nested form)

1. Run: `grep "<form" shop/templates/runs/_xlsx_mapping.html`
2. **Expected:** No output — confirms the partial has no `<form>` tag
3. As a runtime check: submit the `/runs/new` form with a valid xlsx, both PDFs, and part metadata selected
4. **Expected:** Run is created and queued normally; browser redirects to run status page

### 10. Autodetect — noncontiguous characteristic numbers accepted

1. Run: `uv run pytest tests/test_runs.py::test_validate_xlsx_noncontiguous_char_no -v`
2. **Expected:** PASSED — xlsx with char numbers 1, 2, 5 (gap at 3–4) is parsed without error; mapping partial returned

## Edge Cases

### Step 2 with mismatched passwords

1. Navigate to `/setup/step2`
2. Enter password and a different confirmation
3. **Expected:** 200 response with error message; DB not updated; `setup_complete` remains False; no partial commit

### xlsx with all-undetected columns

1. Upload an xlsx where no column header matches any required field name
2. **Expected:** Mapping partial renders with all columns showing amber background and `(ignore)` selected; no error; user can manually correct via selects before submitting

### xlsx upload followed by different xlsx

1. Upload a valid xlsx → mapping panel appears
2. Upload a different xlsx to the same input
3. **Expected:** `#xlsx-mapping-section` updates to reflect the new file's mapping; old mapping is replaced (HTMX innerHTML swap)

### Existing deployments with wizard_step=4

- DB guard logic: `wizard_step >= 2` is treated as `setup_complete` for pre-M002 deployments that completed the 4-step wizard
- Verification: existing `ShopConfig` rows with `wizard_step=4` are served as complete without needing a migration

## Failure Signals

- `uv run pytest --tb=short` shows < 93 passed or any non-xfail failure → slice regression
- `/setup/step3` returns 404 → catch-all redirect routes missing or not registered
- `/setup/step3` returns 200 with wizard HTML → redirect routes not hit (routing order issue)
- `#xlsx-mapping-section` stays empty after xlsx upload → HTMX attributes missing on input, or `validate-xlsx` endpoint not registered, or auth redirect intercepted as partial swap
- `div.alert.alert-error` appears for a valid xlsx → `parse_excel_preview()` throwing on valid file; check `validate_xlsx: invalid file` in logs
- Admin settings page 500 → template still referencing deleted column mapping context variables
- `grep "<form" shop/templates/runs/_xlsx_mapping.html` returns a line → nested form created; outer form submission will break

## Requirements Proved By This UAT

- none (no REQUIREMENTS.md; requirements tracked in PROJECT.md)
- Covers all success criteria from M002-ROADMAP.md:
  - 2-step wizard ✓
  - /setup/step3 and /setup/step4 redirect (not 404) ✓
  - `/runs/new` xlsx triggers inline mapping confirmation ✓
  - Amber indicators for undetected columns ✓
  - Admin settings page has no column mapping ✓
  - 93 ≥ 87 tests passing ✓

## Not Proven By This UAT

- Docker fresh-install flow (2-step setup in container) — operational check requiring `docker compose up --build`; not covered by contract tests
- End-to-end run submission with column mapping selects actually influencing pipeline output — mapping UI is confirmation-only; pipeline still uses internal `detect_column_mapping()`; wiring is deferred
- Browser HTMX swap behavior under network failure or timeout conditions — not tested; error partial handles server-side errors only

## Notes for Tester

- The column mapping selects (`col_{idx}`) are submitted with `/runs/new` POST but currently ignored by the pipeline. This is by design for this milestone — the mapping is a UX confirmation of auto-detection, not a pipeline override. Don't expect pipeline behavior to change based on select values.
- `step4_mapping_partial.html` and `step4_error.html` remain in `shop/templates/setup/` despite being used by `/runs/validate-xlsx`. Their location is historical; they work correctly from there. Don't be confused by the path mismatch.
- The 2 xfailed tests are pre-existing (`xfail(strict=False)`) and are expected — they do not indicate problems.
