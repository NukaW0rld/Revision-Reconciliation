---
id: T02
parent: S01
milestone: M002
provides:
  - POST /runs/validate-xlsx endpoint (auth-gated, HTMX-friendly)
  - runs/_xlsx_mapping.html bare fragment with auto-detected column mapping table
  - new.html xlsx input wired with HTMX change trigger + #xlsx-mapping-section target div
key_files:
  - shop/routers/runs.py
  - shop/templates/runs/_xlsx_mapping.html
  - shop/templates/runs/new.html
  - .gsd/milestones/M002/slices/S01/tasks/T02-PLAN.md
key_decisions:
  - xlsx file identification by filename suffix (.xlsx) first, then fallback to any file-like — handles HTMX sending sibling PDF inputs alongside the xlsx
  - reused setup/step4_error.html for error display rather than creating a new error partial
  - validate_xlsx returns HTMLResponse("") on no-file-found to cleanly clear the target div
patterns_established:
  - bare HTML fragment partials (no <form> tag) for HTMX innerHTML swap targets inside existing forms
  - file-type disambiguation in multipart form by checking filename suffix before falling back to any file-like value
observability_surfaces:
  - INFO "validate_xlsx: preview ok, cols={n}, detected={fields}" — on each successful parse; grep to audit detection quality
  - WARNING "validate_xlsx: invalid file — {message}" — on ValueError from parse_excel_preview; surfaces bad uploads in logs
  - Browser DevTools: POST /runs/validate-xlsx returns 200 with HTML fragment; div.alert.alert-error in body = error path
  - curl -X POST /runs/validate-xlsx -F "form3_xlsx=@bad.txt" -H "Cookie: ..." — triggers error partial directly
duration: 20m
verification_result: passed
completed_at: 2026-03-13
blocker_discovered: false
---

# T02: Add validate-xlsx endpoint and per-run column mapping partial

**`POST /runs/validate-xlsx` wired end-to-end: xlsx upload triggers HTMX inline column-mapping table with amber indicators on undetected columns; invalid files render error partial in place.**

## What Happened

Added `validate_xlsx` POST endpoint to `shop/routers/runs.py` following the `validate_pdf` pattern. The endpoint reads multipart form data, identifies the xlsx file by `.xlsx` filename suffix (with fallback to any file-like value), calls `parse_excel_preview()`, and returns either `runs/_xlsx_mapping.html` on success or `setup/step4_error.html` on `ValueError`.

Created `shop/templates/runs/_xlsx_mapping.html` as a bare HTML fragment (no `<form>` tag). It renders a horizontally-scrollable table: selects for each column in the header row (with `name="col_{idx}"`), detected columns pre-selected to their field name, undetected columns showing amber background with `(ignore)` as default. Includes a legend line explaining the amber indicator. Five preview data rows are shown below the header.

Updated `shop/templates/runs/new.html`: added `hx-post`, `hx-trigger="change"`, `hx-encoding="multipart/form-data"`, `hx-target="#xlsx-mapping-section"`, `hx-swap="innerHTML"` to the `form3_xlsx` file input, and added `<div id="xlsx-mapping-section"></div>` below the input. Updated helper label text to reflect inline validation.

Also added `## Observability Impact` section to `T02-PLAN.md` (pre-flight fix).

## Verification

- `uv run pytest tests/test_runs.py -v` → **12 passed** (no regressions)
- `uv run pytest --tb=short` → **87 passed, 3 failed, 2 xfailed** — meets ≥87 threshold; the 3 failures are the pre-existing step4 tests from T01, documented as slated for removal in T04
- Browser: `/runs/new` → upload `assets/part1/FAIR.xlsx` → mapping table rendered inline in `#xlsx-mapping-section`; selects visible, amber styling on undetected columns, preview rows showing AS9102 data, amber legend visible
- Browser: upload invalid file (`bad.xlsx` = plain text) → error partial "Cannot read file: File is not a zip file" with "Correct the file issue and try uploading again." rendered in same target div
- Confirmed `_xlsx_mapping.html` has no `<form>` tag: `grep "<form" _xlsx_mapping.html` → no output

## Diagnostics

- `grep "validate_xlsx" app.log` — lists all xlsx validation attempts with outcome
- `grep "validate_xlsx: invalid" app.log` — narrows to failures only
- Network tab: `POST /runs/validate-xlsx` should return 200 with HTML fragment; `div.alert.alert-error` in response body signals the error path
- Auth failure: `get_current_user` raises 401/redirect (same as all other auth-gated endpoints)

## Deviations

None. The plan was followed as written.

## Known Issues

Three pre-existing test failures (`test_form3_upload_autodetect`, `test_empty_file_error`, `test_noncontiguous_char_no`) POST to `/setup/step4/upload` which now redirects (T01's change). Documented in T01-SUMMARY; scheduled for removal in T04.

## Files Created/Modified

- `shop/routers/runs.py` — new `validate_xlsx` POST endpoint (auth-gated, multipart-aware, logs detection outcome)
- `shop/templates/runs/_xlsx_mapping.html` — new bare mapping fragment with amber indicators and preview rows
- `shop/templates/runs/new.html` — HTMX attributes on xlsx input, `#xlsx-mapping-section` target div, updated helper label
- `.gsd/milestones/M002/slices/S01/tasks/T02-PLAN.md` — added `## Observability Impact` section (pre-flight fix)
