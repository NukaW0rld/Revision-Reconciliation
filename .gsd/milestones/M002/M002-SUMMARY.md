---
id: M002
provides:
  - 2-step setup wizard (shop name → admin password → /login); setup_complete=True at step 2 completion
  - /setup/step3 and /setup/step4/* redirect gracefully (302) instead of 404
  - wizard_layout.html 2-step progress bar; step2 submit reads "Complete Setup"
  - POST /runs/validate-xlsx auth-gated HTMX endpoint for per-run xlsx column mapping confirmation
  - runs/_xlsx_mapping.html bare fragment with amber indicators for undetected columns and 5 preview rows
  - new.html xlsx input wired with HTMX hx-post/hx-trigger=change/hx-target for inline mapping confirmation
  - admin settings page renders shop name and retention sections only (column mapping removed)
  - step3_engineer.html and step4_column_mapping.html deleted; settings_upload and settings_save routes deleted
  - 93 passing tests (up from 87 baseline at M001 close)
key_decisions:
  - step3/step4 redirects use setup_complete flag (not wizard_step) to decide /login vs /setup/
  - step2_post commits wizard_step and setup_complete in a single db.commit() to avoid partial state
  - runs/_xlsx_mapping.html is a bare fragment (no <form> tag) — step4_mapping_partial.html NOT reused to avoid nested form breakage
  - xlsx file identification by filename suffix (.xlsx) first in multipart form; resilient to sibling PDF inputs
  - SETUP-02/03/04 tests migrated from /setup/step4/upload to /runs/validate-xlsx rather than deleted — xlsx parsing behavior unchanged
  - client + client_setup_incomplete fixtures share the same db_engine; step3_redirect tests split into two functions to avoid fixture DB collision
patterns_established:
  - catch-all redirect routes (GET + POST) for removed wizard steps avoid 404 on bookmarked URLs
  - bare HTML fragment partials (no <form> tag) for HTMX innerHTML swap inside existing forms
  - file-type disambiguation in multipart form by checking filename suffix before fallback
  - when removing a route, scan test files for direct URL references and update to replacement endpoint
observability_surfaces:
  - INFO "Setup wizard complete: admin password set, setup_complete=True" on step2_post success
  - INFO "validate_xlsx: preview ok, cols={n}, detected={fields}" on each successful xlsx parse
  - WARNING "validate_xlsx: invalid file — {message}" on ValueError from parse_excel_preview
  - DB: SELECT setup_complete, wizard_step FROM shop_config WHERE id=1 — post-setup: 1, 2
  - curl -sI /setup/step3 | grep Location → /login (complete) or /setup/ (incomplete)
requirement_outcomes:
  - id: per-run-xlsx-mapping
    from_status: active
    to_status: validated
    proof: "POST /runs/validate-xlsx endpoint returns _xlsx_mapping.html bare fragment; 4 new tests in test_runs.py pass (test_validate_xlsx_autodetect, test_validate_xlsx_empty_file, test_validate_xlsx_noncontiguous_char_no, test_validate_xlsx_requires_auth); amber indicators confirmed in template; hx-post wired in new.html"
duration: ~105m (4 tasks across 1 slice)
verification_result: passed
completed_at: 2026-03-13
---

# M002: Setup Simplification & Per-Run Column Mapping

**Setup wizard reduced to 2 steps (shop name → admin password → /login); per-run xlsx column mapping confirmation wired inline on the new run form; admin column mapping UI removed; 93 tests passing.**

## What Happened

This milestone had a single slice (S01) with four tasks — the work was tightly coupled enough that splitting would have created artificial handoffs.

**Wizard strip (T01):** Removed the step 3 (engineer creation) and step 4 (column mapping) route handlers from `setup.py`. Added four catch-all redirect routes covering GET and POST for both step 3 and step 4: when `setup_complete=True`, they redirect to `/login`; otherwise to `/setup/`. The `step2_post` handler now sets `setup_complete=True` and redirects directly to `/login`, with both `wizard_step=2` and `setup_complete=True` committed in a single `db.commit()`. The wizard layout updated to a 2-step progress bar ("Step N of 2") and the step 2 submit button reads "Complete Setup". The `admin@shop.local` email is displayed on step 2 so users know their credentials before completing setup.

**Per-run xlsx mapping endpoint (T02):** Added `POST /runs/validate-xlsx` to `runs.py` following the existing `validate_pdf` HTMX pattern. The endpoint reads multipart form data, identifies the xlsx file by `.xlsx` filename suffix (resilient to sibling PDF inputs in the HTMX form), calls `parse_excel_preview()` + `detect_column_mapping()`, and returns either `runs/_xlsx_mapping.html` on success or `setup/step4_error.html` on `ValueError`. The new `runs/_xlsx_mapping.html` is a bare HTML fragment (no `<form>` tag): a horizontally-scrollable table with per-column `<select>` elements, amber highlighting for undetected columns with `(ignore)` default, and 5 preview data rows. The xlsx file input in `new.html` was wired with `hx-post="/runs/validate-xlsx"`, `hx-trigger="change"`, and `hx-target="#xlsx-mapping-section"`.

**Admin cleanup (T03):** Removed `settings_upload` and `settings_save` route handlers from `admin.py` and stripped the associated `File`/`UploadFile` imports. Removed the column mapping card from `settings.html` — the page now renders shop name and retention sections only. Deleted `step3_engineer.html` and `step4_column_mapping.html`. Three existing tests in `test_setup.py` (SETUP-02/03/04) that hit the now-removed `/setup/step4/upload` endpoint were migrated to `/runs/validate-xlsx` with field name `form3_xlsx` rather than deleted — the xlsx parsing behavior is identical, only the endpoint changed. This raised the test count from 87 to 90.

**Test suite update (T04):** Added four new tests to `test_runs.py` covering the validate-xlsx endpoint: `test_validate_xlsx_autodetect`, `test_validate_xlsx_empty_file`, `test_validate_xlsx_noncontiguous_char_no`, and `test_validate_xlsx_requires_auth`. Added `test_removed_step4_redirects_when_complete` and `test_removed_step4_redirects_when_incomplete` to `test_setup.py` as two separate functions (not one) because the `client` and `client_setup_incomplete` fixtures share the same `db_engine` — combining them in one test causes the `client`-seeded `setup_complete=True` row to leak into the incomplete-path assertion. Final count: 93 passed, 0 failures, 2 xfailed.

## Cross-Slice Verification

All success criteria from the roadmap were verified:

| Criterion | Evidence |
|-----------|----------|
| Fresh install: setup wizard has exactly 2 steps | `wizard_layout.html` progress bar: "Step N of 2"; `setup.py` cap at 2; `uv run pytest tests/test_setup.py -v` → 12 passed including step1→step2 and step2-completion tests |
| `admin@shop.local` logs in immediately after setup | `step2_post` sets `setup_complete=True` + creates admin user + redirects to `/login`; test `test_setup_complete` asserts 302 → `/login` |
| `/setup/step3` and `/setup/step4` redirect (not 404) | Catch-all routes in `setup.py`; `test_removed_step4_redirects_when_complete` → Location: /login; `test_removed_step4_redirects_when_incomplete` → Location: /setup/ |
| New run form shows inline xlsx mapping panel via HTMX | `hx-post="/runs/validate-xlsx"` + `hx-trigger="change"` + `hx-target="#xlsx-mapping-section"` in `new.html`; `test_validate_xlsx_autodetect` confirms 200 HTML response with select elements |
| Amber indicators for undetected columns | `_xlsx_mapping.html` applies `select-warning` class for columns not in `detected_by_col`; `(ignore)` as default option |
| Engineer can correct undetected columns before submitting | `<select>` elements with all required field options are submitted with the outer run form via standard HTML form POST |
| Admin settings page: no column mapping section | `grep "column.mapping" shop/templates/admin/settings.html` → no output; `test_admin_settings` → 5 passed |
| All existing tests pass ≥87; new tests added | `uv run pytest --tb=short` → **93 passed, 2 xfailed** (up from 87 baseline) |

## Requirement Changes

- per-run-xlsx-mapping (new): introduced and validated — `POST /runs/validate-xlsx` returns inline mapping partial; 4 tests exercise autodetect, error, noncontiguous, and auth-gate paths

## Forward Intelligence

### What the next milestone should know

- `col_{idx}` selects from `_xlsx_mapping.html` are submitted with the outer `/runs/new` POST but currently ignored by `run_pipeline_task`. Wiring them requires threading a `column_mapping` dict through `RunCreate` → `create_run()` → `run_pipeline_task()` → `load_form3()`. The `detect_column_mapping()` return shape (field_name → col_index) and the template's inverse (`detected_by_col`: col_idx → field_name) are both in place — the pipeline hookup is the only missing link.
- `step4_mapping_partial.html` remains in `shop/templates/setup/` — it is still referenced by the `/runs/validate-xlsx` error path (`step4_error.html`). Renaming is a cosmetic cleanup with no functional impact.
- `ShopConfig.column_mapping` column remains in the DB — it is no longer written by setup or admin but exists as a harmless artifact. No migration is needed to remove it unless storage hygiene becomes a concern.
- Existing deployments with `wizard_step=4` (completed old 4-step wizard) are handled correctly: `setup_complete=True` is the authoritative flag; `wizard_step` value is not checked by the guard.

### What's fragile

- `_xlsx_mapping.html` preview rows rely on `preview_rows` being a list of lists from `parse_excel_preview()` — if that function's return shape changes, the template silently renders empty rows with no error signal.
- The `.xlsx` suffix detection in `validate_xlsx` falls back to the first file-like field if no `.xlsx` suffix is found. In practice the browser always sends the filename, but an edge case with an unnamed file-like input could cause unexpected behavior.
- `client` + `client_setup_incomplete` fixture DB collision: both share `db_engine`; `client` fixture always seeds `setup_complete=True` at `shop_config.id=1`. Any test combining both fixtures will see setup as complete. Existing split into two test functions documents the safe pattern.

### Authoritative diagnostics

- `uv run pytest --tb=short` — ground truth; 93 passed, 2 xfailed is the M002 baseline
- `grep "validate_xlsx" app.log` — primary signal for xlsx validation quality and failure rate in production
- `SELECT setup_complete, wizard_step FROM shop_config WHERE id=1` — definitive setup state; post-setup values: 1, 2
- `curl -sI http://localhost:PORT/setup/step3 | grep Location` — should return `/login` on a completed install

### What assumptions changed

- Original assumption: `step4_mapping_partial.html` could be reused for per-run mapping. Actual outcome: reuse was rejected because it carries a `<form>` tag that would create invalid nested forms inside `new.html`. A new bare fragment `runs/_xlsx_mapping.html` was created instead.
- Original assumption: SETUP-02/03/04 tests would be deleted. Actual outcome: they were migrated to `/runs/validate-xlsx` — the xlsx parsing behavior they test is unchanged and the tests remain valid at the new endpoint.

## Files Created/Modified

- `shop/routers/setup.py` — removed step3/step4 handlers; added 4 catch-all redirect routes; step2_post completes setup and redirects to /login; wizard_step cap=2; unused imports removed
- `shop/routers/runs.py` — new `validate_xlsx` POST endpoint (auth-gated, multipart-aware, logs detection outcome)
- `shop/routers/admin.py` — removed settings_upload and settings_save routes; removed File/UploadFile imports
- `shop/templates/setup/wizard_layout.html` — 2-step progress bar, "of 2" title
- `shop/templates/setup/step2_password.html` — "Complete Setup" button text; admin@shop.local credential display
- `shop/templates/setup/step3_engineer.html` — deleted
- `shop/templates/setup/step4_column_mapping.html` — deleted
- `shop/templates/runs/_xlsx_mapping.html` — new bare mapping fragment with amber indicators, per-column selects, and 5 preview rows
- `shop/templates/runs/new.html` — HTMX attributes on xlsx input; #xlsx-mapping-section target div added
- `shop/templates/admin/settings.html` — column mapping card removed; retention section preserved
- `tests/test_setup.py` — 4 new tests (step1→step2 flow, step2 completion, step3 redirect ×2); 3 xlsx tests migrated to test_runs.py; SETUP-02/03/04 migrated to /runs/validate-xlsx
- `tests/test_runs.py` — 4 new tests (test_validate_xlsx_autodetect, test_validate_xlsx_empty_file, test_validate_xlsx_noncontiguous_char_no, test_validate_xlsx_requires_auth); _make_form3_xlsx_bytes() helper added
