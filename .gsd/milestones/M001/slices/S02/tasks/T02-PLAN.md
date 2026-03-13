# T02: 02-pipeline-bridge 02

**Slice:** S02 — **Milestone:** M001

## Description

Build the upload form, server-side file validation, and run creation so engineers can submit a drawing comparison run.

Purpose: Engineers need to submit Rev A PDF, Rev B PDF, and Form 3 Excel with part metadata before any pipeline can run. This plan delivers the complete submission flow from form to redirect.
Output: /runs/new GET (form) + POST (submit), /runs/validate-pdf HTMX partial (inline PDF validation), Run record created and Huey task enqueued, redirect to /runs/{id} status page.

## Must-Haves

- [ ] "Engineer can POST to /runs/new with 3 files + metadata and be redirected to /runs/{id}"
- [ ] "Raster PDF uploaded to /runs/validate-pdf returns an error partial (no run created)"
- [ ] "Multi-page PDF uploaded to /runs/validate-pdf returns a page selector partial inline"
- [ ] "Unreadable or empty Excel returns a clear error before run is created"
- [ ] "Run record in DB has rev_a_label and rev_b_label matching form submission"

## Files

- `shop/routers/runs.py`
- `shop/services/runs.py`
- `shop/app.py`
- `shop/templates/runs/new.html`
- `shop/templates/runs/_page_selector.html`
- `shop/templates/runs/_pdf_error.html`
