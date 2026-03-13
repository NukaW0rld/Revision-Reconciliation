---
estimated_steps: 6
estimated_files: 4
---

# T02: Add validate-xlsx endpoint and per-run column mapping partial

**Slice:** S01 — Two-step wizard & per-run column mapping
**Milestone:** M002

## Description

Add a `/runs/validate-xlsx` POST endpoint that accepts an xlsx upload via HTMX, runs `parse_excel_preview()` + `detect_column_mapping()`, and returns an inline column mapping confirmation partial. The partial is a bare HTML fragment (no `<form>` wrapper) that sits inside the existing `runs/new.html` form. The xlsx file input gets HTMX attributes to trigger validation on file change.

## Steps

1. Read `shop/routers/runs.py` to understand the `validate_pdf` endpoint pattern and `get_current_user` dependency
2. Read `shop/services/form3.py` to confirm `parse_excel_preview()` and `detect_column_mapping()` signatures
3. Read `shop/templates/setup/step4_mapping_partial.html` to understand the existing mapping UI structure (for reference, not reuse — new partial must be form-free)
4. Add `validate_xlsx` POST endpoint to `runs.py`: auth-gated, reads xlsx bytes from multipart form (iterate `request.form()` to find xlsx file), calls `parse_excel_preview()` then `detect_column_mapping()`, returns `runs/_xlsx_mapping.html` on success or `setup/step4_error.html` on error
5. Create `shop/templates/runs/_xlsx_mapping.html`: bare fragment with a table showing column index, detected header text, and a `<select name="col_{idx}">` for each column with Form 3 field options; undetected columns get amber background and `(ignore)` as default; include a brief info label explaining auto-detection
6. Update `shop/templates/runs/new.html`: add HTMX attributes to xlsx file input (`hx-post="/runs/validate-xlsx"`, `hx-trigger="change"`, `hx-encoding="multipart/form-data"`, `hx-target="#xlsx-mapping-section"`, `hx-swap="innerHTML"`), add `<div id="xlsx-mapping-section"></div>` target div below the input

## Must-Haves

- [ ] `POST /runs/validate-xlsx` is auth-gated via `get_current_user`
- [ ] Endpoint returns mapping partial on valid xlsx, error partial on invalid
- [ ] `_xlsx_mapping.html` has no `<form>` tag — bare fragment only
- [ ] Undetected columns have amber visual indicator and `(ignore)` default
- [ ] xlsx file input in `new.html` has HTMX trigger on change
- [ ] Endpoint handles extra form fields (PDFs) without error

## Verification

- Dev server: navigate to `/runs/new`, select an xlsx file — mapping panel appears inline
- Select an invalid file — error message appears in the same target div
- Automated tests added in T04

## Inputs

- `shop/routers/runs.py` — validate_pdf pattern to follow
- `shop/services/form3.py` — `parse_excel_preview()`, `detect_column_mapping()` 
- `shop/templates/setup/step4_mapping_partial.html` — reference for mapping UI (not reused directly)
- `shop/templates/setup/step4_error.html` — reused for error display

## Expected Output

- `shop/routers/runs.py` — new `validate_xlsx` endpoint
- `shop/templates/runs/_xlsx_mapping.html` — new bare mapping confirmation partial
- `shop/templates/runs/new.html` — HTMX attributes on xlsx input + target div
