# S02: Dashboard, Runs List & New Run Form — UAT

**Milestone:** M003
**Written:** 2026-03-14

## UAT Type

- UAT mode: live-runtime
- Why this mode is sufficient: All changes are CSS class replacements on existing templates. Visual inspection of the running app is the definitive check — grep/build/test verify structural correctness, but only a browser confirms no light-mode artifacts are visible against the dark theme.

## Preconditions

1. Dev server is running: `uv run python run_web.py` (default port 8000)
2. The app has completed first-run setup (shop name + admin password set)
3. An admin account exists and you can log in
4. `npm run build:css` has been run and `static/dist/output.css` is current
5. Browser devtools are available for network tab inspection if needed

## Smoke Test

Log in and navigate to `/dashboard`. The page should show dark `bg-base-200` bordered sections with no white card backgrounds or box-shadow artifacts. If the dashboard looks like an industrial dark interface, the slice is basically working.

## Test Cases

### 1. Dashboard — industrial dark aesthetic

1. Log in as admin
2. Navigate to `/dashboard`
3. Inspect the "Recent Runs" section and any admin stat cards
4. **Expected:** All content sections use dark bordered containers (`border-base-300` hairline border, `bg-base-200` fill). No white card backgrounds. No box shadows. Section headings use monospaced font. Status badges (queued/running/completed/failed) use amber/green/red palette, not DaisyUI defaults.

### 2. Dashboard — CTA button style

1. On `/dashboard`, locate the "New Run" (or equivalent) CTA button
2. **Expected:** Button uses `font-mono` or is styled consistently with the industrial theme — not a generic blue or green DaisyUI button with rounded corners.

### 3. Runs list — table header style

1. Navigate to `/runs` (the run list page)
2. Inspect the `<thead>` row of the runs table
3. **Expected:** Column headers are monospaced, uppercase, small (xs), with reduced-opacity text (`text-base-content/50`). No `table-zebra` alternating row colors. Rows show `hover:bg-base-300` highlight on mouse-over. Part number cell uses monospaced font.

### 4. Runs list — filter section style

1. On `/runs`, inspect the filter form area above the table
2. **Expected:** Filter labels are monospaced, uppercase, xs size. The filter section has a dark bordered container, no card shadow.

### 5. New run form — section wrappers

1. Navigate to `/runs/new`
2. Inspect each of the four file upload sections (Rev A PDF, Rev B PDF, Form 3 XLSX, and any options section)
3. **Expected:** Each section is a bordered `bg-base-200` div, not a `card bg-base-100 shadow` component. No white backgrounds. Section headings use monospaced font.

### 6. New run form — Rev A PDF upload and HTMX swap

1. On `/runs/new`, upload a valid PDF for the Rev A field
2. **Expected:** The `#revA-section` swap target receives the `_page_selector.html` partial. The injected partial renders as a plain dark warning div (`bg-warning/20 border border-warning`) with no card wrapper, no white background. The page selector heading uses monospaced font.

### 7. New run form — Rev B PDF upload and HTMX swap

1. Upload a valid PDF for the Rev B field
2. **Expected:** Same as above for `#revB-section`. Injected `_page_selector.html` partial is dark, no card artifact.

### 8. New run form — invalid PDF produces error partial

1. Upload a non-PDF file (e.g., rename a .txt to .pdf) or a corrupted file to the Rev A field
2. **Expected:** The `_pdf_error.html` partial is injected into the swap target. It renders as a `alert alert-error` DaisyUI component — red/dark themed, no white background. (This partial was unchanged — test confirms it remains theme-compatible.)

### 9. New run form — XLSX upload and column mapping partial

1. Upload a valid XLSX Form 3 file to the form
2. **Expected:** The `#xlsx-mapping-section` swap target receives `_xlsx_mapping.html`. Any undetected/ambiguous column dropdowns are highlighted with `bg-warning/15 border-warning` (amber-tinted, NOT `bg-amber-100`). The section background is dark. No hardcoded amber backgrounds.

### 10. Alert banner — left-accent style in HTMX context

1. Trigger a run alert (or inspect the `_alert_banner.html` template directly in a test run that has alerts)
2. Navigate to a page where alert banners are injected (dashboard or run list)
3. **Expected:** Alert banner has a left red accent border (`border-l-4 border-error`), dark fill (`bg-base-200`), monospaced run label. No white alert box. The `id="alert-banner-{{ alert.id }}"` attribute is present for HTMX outerHTML swap.

### 11. Page title block in top bar

1. Navigate to `/dashboard`, `/runs`, and `/runs/new` in sequence
2. **Expected:** The top bar (from `base.html`) shows the correct page title for each: "Dashboard", "All Runs", "New Run". The `{% block page_title %}` block is populated in all three templates.

## Edge Cases

### Empty runs list

1. Navigate to `/runs` when no runs exist in the database
2. **Expected:** The table renders with the styled mono headers but an appropriate empty state (no rows, or an empty state message). No crash, no white card artifact.

### Dashboard with no recent runs

1. Navigate to `/dashboard` with a fresh database (no runs submitted)
2. **Expected:** The "Recent Runs" section renders as a bordered dark container with an empty state message or empty table — no white card artifact, no JS error.

### Page selector after PDF re-upload

1. On `/runs/new`, upload a Rev A PDF, then upload a different Rev A PDF to the same field
2. **Expected:** `#revA-section` is re-swapped with the new `_page_selector.html` content. The injected partial is still dark-styled, no stale white card from the first upload.

## Failure Signals

- Any white or light-grey card backgrounds visible against the dark theme indicate `bg-base-100 shadow` was not fully replaced
- `bg-amber-100` visible in the xlsx mapping section means the hardcoded amber was not replaced with semantic tokens
- A white card wrapper around the page selector means `card`/`card-body` classes were not removed from `_page_selector.html`
- Missing page title in the top bar (blank breadcrumb) means `{% block page_title %}` is absent from the template
- HTMX swap target not updating after file upload means the swap ID was renamed — check network tab for 2xx response, then inspect the response fragment for the expected ID
- `npm run build:css` failure means a utility class was added that Tailwind can't find — run the build and check output for missing class warnings

## Requirements Proved By This UAT

- R005 — Dashboard renders with industrial dark aesthetic, no generic DaisyUI defaults visible
- R006 — New run form and column mapping partial use the new design; HTMX partial swaps produce visually integrated fragments
- R008 — Runs list table uses mono headers and hover rows matching the industrial palette

## Not Proven By This UAT

- R007 — Pipeline status page (S03 scope)
- R009, R010, R011 — Review queue keyboard shortcuts and sign-off footer (S04 scope)
- R012 — Admin screens (S05 scope)
- R013 — Full HTMX partial coverage (S05 terminal verification)
- End-to-end flow (login → run → review → sign-off) — requires S03 and S04 to complete

## Notes for Tester

- The quickest visual regression check is the "no white cards" test: with a dark browser background, any remaining `card bg-base-100 shadow` will appear as an obviously light box. Scan each page top-to-bottom.
- If you don't have a sample PDF or XLSX handy, `assets/part1/` contains test files from the existing pipeline fixtures.
- The `_pdf_error.html` template was intentionally not changed — its `alert alert-error` class is DaisyUI theme-aware and already correct. If it looks wrong, the issue is in the DaisyUI dark theme config from S01, not in this template.
- HTMX swap failures are silent on the client side (2xx, no JS error). If an upload produces no visual change, open the network tab and inspect the response body of the POST — confirm the fragment contains the expected `id=` attribute.
