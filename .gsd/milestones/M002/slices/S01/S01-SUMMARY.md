---
id: S01
parent: M002
milestone: M002
provides:
  - 2-step setup wizard (shop name → admin password → /login)
  - setup_complete=True set at step 2 completion in single db.commit()
  - /setup/step3 and /setup/step4/* redirect gracefully (302) instead of 404
  - wizard_layout.html progress bar shows 2 steps, "of 2" title
  - step2_password.html submit button reads "Complete Setup"
  - POST /runs/validate-xlsx auth-gated HTMX endpoint for per-run xlsx column mapping
  - runs/_xlsx_mapping.html bare fragment with amber indicators for undetected columns
  - new.html xlsx input wired with HTMX change trigger → #xlsx-mapping-section target
  - admin settings page with retention section only (column mapping removed)
  - settings_upload and settings_save routes deleted from admin.py
  - step3_engineer.html and step4_column_mapping.html deleted
  - 93 passing tests (up from 87 baseline); 0 failures, 2 xfailed
requires: []
affects: []
key_files:
  - shop/routers/setup.py
  - shop/routers/runs.py
  - shop/routers/admin.py
  - shop/templates/setup/wizard_layout.html
  - shop/templates/setup/step2_password.html
  - shop/templates/runs/_xlsx_mapping.html
  - shop/templates/runs/new.html
  - shop/templates/admin/settings.html
  - tests/test_setup.py
  - tests/test_runs.py
key_decisions:
  - step3/step4 redirects use setup_complete flag (not wizard_step) to decide /login vs /setup/
  - step2_post commits wizard_step and setup_complete in a single db.commit() to avoid partial state
  - runs/_xlsx_mapping.html is a bare fragment (no <form> tag) — step4_mapping_partial.html NOT reused to avoid nested form breakage
  - xlsx file identification by filename suffix (.xlsx) first in multipart form; handles HTMX sending sibling PDF inputs
  - SETUP-02/03/04 tests migrated from /setup/step4/upload to /runs/validate-xlsx rather than deleted — xlsx parsing behavior unchanged
  - client + client_setup_incomplete fixtures share the same db_engine; never combine in one test function (client seeds setup_complete=True at id=1)
patterns_established:
  - catch-all redirect routes (GET + POST) for removed wizard steps avoid 404 on bookmarked URLs
  - bare HTML fragment partials (no <form> tag) for HTMX innerHTML swap inside existing forms
  - file-type disambiguation in multipart form by checking filename suffix before fallback
  - When removing a route, scan test files for direct URL references and update to replacement endpoint
observability_surfaces:
  - INFO "Setup wizard complete: admin password set, setup_complete=True" on step2_post success
  - INFO "validate_xlsx: preview ok, cols={n}, detected={fields}" on each successful xlsx parse
  - WARNING "validate_xlsx: invalid file — {message}" on ValueError from parse_excel_preview
  - DB: SELECT setup_complete, wizard_step FROM shop_config WHERE id=1 — post-setup: 1, 2
  - curl -sI /setup/step3 | grep Location → /login (complete) or /setup/ (incomplete)
  - Browser DevTools: POST /runs/validate-xlsx returns 200 HTML fragment; div.alert.alert-error = error path
drill_down_paths:
  - .gsd/milestones/M002/slices/S01/tasks/T01-SUMMARY.md
  - .gsd/milestones/M002/slices/S01/tasks/T02-SUMMARY.md
  - .gsd/milestones/M002/slices/S01/tasks/T03-SUMMARY.md
  - .gsd/milestones/M002/slices/S01/tasks/T04-SUMMARY.md
duration: ~105m (4 tasks)
verification_result: passed
completed_at: 2026-03-13
---

# S01: Two-step wizard & per-run column mapping

**Setup wizard stripped to 2 steps; per-run xlsx column mapping confirmation wired inline on the new run form; admin column mapping UI removed; 93 tests passing.**

## What Happened

**T01 — Wizard strip:** Removed `step3_get`, `step3_post`, `step4_get`, `step4_upload`, and `step4_save` route handlers from `setup.py`. Added four catch-all redirect routes (GET + POST for step3 and step4) that redirect to `/login` when `setup_complete=True` or `/setup/` when not. Changed `step2_post` to set `setup_complete=True` and redirect to `/login` instead of `/setup/step3`. Both `wizard_step=2` and `setup_complete=True` are written in a single `db.commit()`. Updated `wizard_layout.html` to a 2-step progress bar and changed the step 2 submit button to "Complete Setup".

**T02 — Per-run xlsx mapping endpoint:** Added `POST /runs/validate-xlsx` to `runs.py` following the `validate_pdf` pattern. The endpoint reads multipart form data, identifies the xlsx by `.xlsx` filename suffix (resilient to sibling PDF inputs in the same HTMX form), calls `parse_excel_preview()` + `detect_column_mapping()`, and returns either `runs/_xlsx_mapping.html` on success or `setup/step4_error.html` on `ValueError`. Created `runs/_xlsx_mapping.html` as a bare HTML fragment (no `<form>` tag): horizontally-scrollable table with per-column `<select>` elements, amber highlighting for undetected columns with `(ignore)` default, and 5 preview data rows. Wired the xlsx file input in `new.html` with HTMX `hx-post`, `hx-trigger="change"`, and `hx-target="#xlsx-mapping-section"`.

**T03 — Admin cleanup:** Removed `settings_upload` and `settings_save` route handlers from `admin.py` and stripped the associated `File`/`UploadFile` imports. Removed the column mapping card from `settings.html` — page now renders shop name and retention sections only. Deleted `step3_engineer.html` and `step4_column_mapping.html`. Found that SETUP-02/03/04 tests in `test_setup.py` were hitting the now-removed `/setup/step4/upload` endpoint; migrated them to `/runs/validate-xlsx` with field name `form3_xlsx`. Test count rose from 87 to 90 as a result.

**T04 — Test suite update:** Moved the three xlsx parsing tests (`test_form3_upload_autodetect`, `test_empty_file_error`, `test_noncontiguous_char_no`) from `test_setup.py` to `test_runs.py` under the `test_validate_xlsx_*` naming convention, adding the `_login_engineer()` auth pattern. Added `test_validate_xlsx_requires_auth` (auth gate). Added `test_removed_step4_redirects_when_complete` and `test_removed_step4_redirects_when_incomplete` to `test_setup.py` (split into two functions to avoid fixture DB collision). Final count: 93 passed, 0 failures, 2 xfailed.

## Verification

- `uv run pytest --tb=short` — **93 passed, 0 failures, 2 xfailed** ✓
- `uv run pytest tests/test_setup.py -v` — 12 passed (wizard 2-step flow, redirect tests) ✓
- `uv run pytest tests/test_runs.py -v` — 16 passed (including 4 new xlsx mapping tests) ✓
- `uv run pytest tests/test_admin.py -v` — 5 passed (settings page, no mapping section) ✓
- Browser: `/runs/new` → upload `assets/part1/FAIR.xlsx` → mapping table rendered inline with amber selects and preview rows ✓
- Browser: upload invalid file → error partial rendered in `#xlsx-mapping-section` ✓
- `grep "<form" shop/templates/runs/_xlsx_mapping.html` → no output (bare fragment confirmed) ✓
- `ls shop/templates/setup/` → step3_engineer.html and step4_column_mapping.html absent ✓

## Deviations

- T03 migrated SETUP-02/03/04 tests to `/runs/validate-xlsx` rather than deleting them — the plan didn't mention updating these tests, but fixing them was required to meet the ≥87 threshold. Tests remain valid (same xlsx parsing behavior, new endpoint).
- T04 split `test_removed_step4_redirects` into two functions due to fixture DB collision: `client` and `client_setup_incomplete` share the same `db_engine`; combining them in one test causes the `client`-seeded `setup_complete=True` row to be picked up by `_get_or_create_config`, breaking the incomplete-path assertion.

## Known Limitations

- Column mapping selects in `_xlsx_mapping.html` are named `col_{idx}` and submitted with the outer run form, but `POST /runs/new` currently ignores them — the pipeline calls `detect_column_mapping()` internally. The per-run mapping is a UX confirmation only, not a new pipeline input. Wiring confirmed selects into `load_form3()` is deferred to future work.
- `step4_mapping_partial.html` remains in `shop/templates/setup/` — it is still used by `/runs/validate-xlsx` error path. This file's name is historical; renaming would be a minor cleanup with no functional impact.

## Follow-ups

- Wire confirmed column mapping selects from `_xlsx_mapping.html` into the pipeline's `load_form3()` call so per-run overrides actually influence characteristic parsing.
- Rename `step4_mapping_partial.html` and `step4_error.html` to `runs/` scope to match their current usage context.

## Files Created/Modified

- `shop/routers/setup.py` — removed step3/step4 handlers; added redirect routes; step2_post completes setup; cap=2; unused imports removed
- `shop/routers/runs.py` — new `validate_xlsx` POST endpoint (auth-gated, multipart-aware, logs detection outcome)
- `shop/routers/admin.py` — removed settings_upload and settings_save routes; removed File/UploadFile imports
- `shop/templates/setup/wizard_layout.html` — 2-step progress bar, "of 2" title
- `shop/templates/setup/step2_password.html` — "Complete Setup" button text
- `shop/templates/setup/step3_engineer.html` — deleted
- `shop/templates/setup/step4_column_mapping.html` — deleted
- `shop/templates/runs/_xlsx_mapping.html` — new bare mapping fragment with amber indicators and preview rows
- `shop/templates/runs/new.html` — HTMX attributes on xlsx input, `#xlsx-mapping-section` target div
- `shop/templates/admin/settings.html` — column mapping card removed; retention section preserved
- `tests/test_setup.py` — added 4 new tests (step1→step2, step2 completion, step3 redirect ×2); removed 3 xlsx tests (moved); migrated SETUP-02/03/04
- `tests/test_runs.py` — added 4 new tests: test_validate_xlsx_autodetect, test_validate_xlsx_empty_file, test_validate_xlsx_noncontiguous_char_no, test_validate_xlsx_requires_auth; added _make_form3_xlsx_bytes() helper

## Forward Intelligence

### What the next slice should know
- `col_{idx}` selects are submitted with `/runs/new` POST but currently ignored by `run_pipeline_task`. Wiring them requires passing a `column_mapping` dict through `RunCreate` → `create_run()` → `run_pipeline_task()` → `load_form3()`.
- `detect_column_mapping()` returns a dict keyed by required field name (e.g. `"char_no"`) mapping to column index. The inverse (col_idx → field_name) is computed in the template via `detected_by_col`.
- The `client` + `client_setup_incomplete` fixture collision is a real trap: both share `db_engine`; `client` fixture always seeds `setup_complete=True` at `shop_config.id=1`. Any test mixing both fixtures will see setup as complete.

### What's fragile
- `_xlsx_mapping.html` preview rows rely on `preview_rows` being a list of lists from `parse_excel_preview()` — if that function's return shape changes, the template will silently render empty rows.
- The fallback from `.xlsx` suffix detection to "any file-like value" in `validate_xlsx` could theoretically pick up a PDF if the xlsx input has no filename. This is a HTMX form submission edge case; in practice the browser always sends the filename.

### Authoritative diagnostics
- `uv run pytest --tb=short` — ground truth for test health; 93 passed is the new baseline
- `grep "validate_xlsx" app.log` — primary signal for xlsx validation quality and failure rate in production
- `SELECT setup_complete, wizard_step FROM shop_config WHERE id=1` — definitive setup state check
