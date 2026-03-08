# Phase 2: Pipeline Bridge - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

File upload, async pipeline execution, and run status tracking. Engineers can submit a drawing comparison run (Rev A PDF, Rev B PDF, Form 3 Excel + part metadata), watch it progress stage by stage through all 8 pipeline stages, and see clear outcomes for completed, failed, and warning-state runs. The review queue and sign-off workflow are Phase 3.

</domain>

<decisions>
## Implementation Decisions

### Upload form structure
- Single-page form: all 3 file pickers (Rev A PDF, Rev B PDF, Form 3 Excel) and all part metadata fields (part number, revision letters, customer, job number) on one page at `/runs/new`
- Multi-page PDF page selection: inline on the form — after the engineer picks a multi-page PDF, an HTMX partial replaces the section below that file picker with a page selector; the submit button is disabled until the engineer selects a page; everything stays on the same page
- Upload form accessed via a prominent "Submit New Run" CTA on the dashboard (not a nav bar link)
- After successful submission: auto-redirect to the run's status page; engineer sees stage progress immediately

### Run status & progress display
- Stage display: numbered checklist of all 8 stages — completed stages show a checkmark, current stage shows a spinner, pending stages are dim; current stage name is visually highlighted
- Stage names in order: Form 3 parsing → Balloon detection → Text extraction → Anchor building → Alignment → Candidate matching → Classification → Output
- Run state progression: queued → running (with current stage) → completed / failed / warning
- Dashboard shows 5–10 most recent runs with compact cards + "View all runs" link to `/runs`
- Run card content: part number (prominent), revision letters (A→B), status badge, submitted date, assigned reviewer name

### Navigation (added in Phase 2)
- Phase 2 adds a persistent top nav bar to `base.html`: shop name/logo, "New Run" link, "My Runs" / run list link, alert bell indicator, logged-in user email + logout
- All engineers see all runs in the run list (consistent with HISTORY-01); the run list is filterable by part number and date
- Alerts are personal: only the assigned reviewer sees failure notifications for their runs

### Warning states & blocking UX
- **PIPE-05 (Rev B balloon failure — partial result warning):** Inline yellow warning banner on the status page; the "Open Review Queue" button is replaced with "Acknowledge and Open Review"; engineer must click to proceed; no modal
- **PIPE-06 (alignment uniformly low confidence):** Inline warning section on the status page showing a plain-language confidence distribution summary (e.g., "87% of characteristics have confidence below 0.5") with two action buttons: "Proceed to Review" and "Abort Run"; abort cancels the run; proceed opens the review queue
- **PIPE-04 (Rev A balloon failure — hard fail):** Status page shows a "Failed" badge, stage checklist with the failing stage highlighted in red, and a plain-language error message identifying the failure stage and likely cause (e.g., "check that Rev A is a vector PDF, not a scanned image"); no recovery action in the UI; engineer must submit a new run

### In-app alerts
- Alert surface: bell icon with a red badge count in the nav bar; unread failure alerts also display as a dismissible banner at the top of the dashboard
- Alert content: part number, run ID, failure stage; includes a "View run" link; clicking the dashboard banner dismisses it and clears the badge
- Alert scope: personal to the assigned reviewer — other engineers do not see another's failure alerts

### Claude's Discretion
- Exact DaisyUI component choices for the stage checklist (steps component, timeline, or custom list)
- Confidence distribution display format (bar chart, percentage list, or text summary — choose whichever is clearest without adding a charting library)
- Exact run assignment UI placement (submitter defaults as reviewer; explicit reassignment is a Phase 3 concern per REVIEW-06)
- File storage location for uploaded PDFs and Excel files within the Docker container
- Huey task definition structure and queue configuration details

</decisions>

<specifics>
## Specific Ideas

- Multi-page PDF page selection should feel inline and instant — the HTMX partial should show page thumbnails or at minimum page numbers with dimensions so the engineer can identify the right page
- The alignment confidence summary (PIPE-06) must be plain language, not a raw number — something like "Most characteristics could not be reliably located in Rev B" is more useful than "mean confidence: 0.31"
- The nav bar bell badge should be visible from every page so the reviewer is never surprised by a failure they missed

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `delta_preservation/cli.py:run_pipeline()`: The callable the Huey worker invokes; accepts `revA_pdf`, `revB_pdf`, `form3_xlsx`, `out_dir`, `dpi`, `part_name`; returns Path to output directory
- `shop/services/form3.py:parse_excel_preview()`: Already parses Excel bytes for column mapping; can be reused or extended for upload-time Excel validation (UPLOAD-04)
- `shop/templates/base.html`: Currently has no nav bar — Phase 2 adds the persistent top nav bar here; SSE extension (`htmx-sse.js`) is already loaded
- `shop/dependencies.py:get_current_user()`: Existing auth dependency used in every new route
- `shop/models.py`: Has `User`, `UserSession`, `ShopConfig` — Phase 2 adds `Run` and `RunAlert` models

### Established Patterns
- Standard HTML POST (not HTMX POST) for form submissions that result in redirects — HTMX intercepts redirects as partial DOM swaps; file upload submit should be a standard form POST
- HTMX partials for inline UI updates that stay on the same page (e.g., multi-page PDF page selector after file pick)
- Router imports deferred inside `create_app()` body to break circular imports — Phase 2 `runs` router follows this pattern
- `dependencies.py:get_current_user()` for auth; `require_admin()` for admin-only routes; no equivalent `require_engineer()` needed — all authenticated users can submit runs
- DaisyUI `badge`, `card`, `btn`, `alert` components already in use; `steps` or `timeline` component available for stage checklist

### Integration Points
- `run_pipeline()` is called from a Huey worker task; the worker receives the file paths, runs the pipeline, and updates the `Run` DB record with the output path and final status
- `out/<run_id>/delta_packet.json` is the pipeline's output contract; Phase 2 must store the run output directory path in the `Run` record so Phase 3 can serve snippets and read the delta packet
- `delta_preservation/config.py:DEFAULT_SEARCH_RADIUS`, `MIN_ALIGNMENT_INLIERS`, `MIN_ALIGNMENT_RATIO` — alignment confidence thresholds used to determine the PIPE-06 warning condition
- SSE endpoint for stage progress: pipeline worker emits stage events; FastAPI SSE endpoint streams them to the status page via `hx-ext="sse"`

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-pipeline-bridge*
*Context gathered: 2026-03-03*
