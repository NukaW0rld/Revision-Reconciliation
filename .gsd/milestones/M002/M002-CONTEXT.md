# M002: Setup Simplification & Per-Run Column Mapping — Context

**Gathered:** 2026-03-13
**Status:** Queued — pending auto-mode execution

## Project Description

Delta Preservation is a web application for aerospace machine shop quality teams that automates change detection between drawing revisions. Engineers upload PDFs and an AS9102 Form 3 Excel, a pipeline classifies each characteristic, and a review queue drives sign-off.

## Why This Milestone

The current first-run setup wizard has four steps: shop name → admin password → first engineer account → Form 3 column mapping. Steps 3 and 4 create friction without corresponding value:

- **Step 3 (engineer creation)** is redundant with the existing admin user management page (`/admin/users`), which already supports adding engineers after login.
- **Step 4 (column mapping)** forces a one-time global mapping that is disconnected from actual run uploads. The `column_mapping` stored in `ShopConfig` is **never passed to `run_pipeline()`** — the pipeline (`delta_preservation/io/xlsx.py:load_form3`) does its own keyword-based column detection independently. The setup step gave false confidence that mapping was configured when the pipeline ignored it entirely.

Moving column mapping to the point of upload (per-run, inline on the new run form) makes the system honest: the engineer sees and confirms column detection against the actual file they are about to process.

## User-Visible Outcome

### When this milestone is complete, the user can:

- Complete first-run setup in 2 steps: enter shop name → set admin password (with `admin@shop.local` displayed as login email) → land on the login page
- Log in as `admin@shop.local` with the chosen password immediately after setup
- Add engineer accounts via the existing admin user management dashboard (no wizard step required)
- Submit a new run: after uploading the Form 3 xlsx, see an inline column mapping confirmation panel that auto-detects columns and allows correction of any undetected field before submitting
- Proceed through the existing pipeline → review → sign-off cycle unchanged

### Entry point / environment

- Entry point: `http://localhost:<port>/setup/step1` (first run), then `/runs/new` for column mapping
- Environment: local dev (uv run python run_web.py) and Docker (docker compose up)
- Live dependencies involved: SQLite database (ShopConfig, User), file uploads

## Completion Class

- Contract complete means: all tests pass; wizard has exactly 2 steps; per-run column mapping confirmation is exercised in tests; removed wizard steps return 404 or redirect; admin settings no longer shows column mapping section
- Integration complete means: fresh Docker container reaches `/dashboard` after 2 wizard steps; new run form shows column mapping confirmation panel after xlsx upload; pipeline runs to completion using per-run detected mapping
- Operational complete means: existing signed-off runs and audit packets are unaffected; no DB migration required (ShopConfig.column_mapping column stays but is no longer written by setup)

## Final Integrated Acceptance

To call this milestone complete, we must prove:

- Fresh Docker deployment: setup completes in 2 steps, admin logs in with `admin@shop.local`, runs new comparison from `/runs/new` with xlsx column confirmation, pipeline reaches completed state
- Existing test suite passes (87 baseline); new tests cover: 2-step wizard completion, per-run column mapping confirmation + correction, removed wizard steps redirect gracefully
- Admin settings page loads without the column mapping section

## Risks and Unknowns

- **Per-run mapping confirmation UX** — Need to decide where in the form this appears. The xlsx is uploaded via a file input, but submission is one POST. The inline HTMX pattern used for PDF validation (`hx-post="/runs/validate-pdf"`) is the established model to follow; column mapping confirmation should follow the same pattern (HTMX trigger on xlsx file change → inline partial swap showing mapping table → hidden inputs carrying confirmed mapping into the main POST).
- **`column_mapping` never wired to pipeline** — Confirmed: `load_form3()` does its own keyword detection. The per-run mapping UI is a UX confirmation, not a new pipeline input. The confirmed mapping does not need to be passed to `run_pipeline()` unless we later choose to override pipeline detection (out of scope here).
- **wizard_step guard logic** — Steps 3 and 4 routers must be removed or redirected; the `wizard_step` progression cap drops from 4 to 2; `setup_complete = True` is now set at end of step 2 POST. The `_step_redirect(step)` logic must be updated accordingly.
- **Test coverage for removed steps** — `tests/test_setup.py` has tests for step 3 (block-skip), step 4 (upload/autodetect/empty/noncontiguous). These must be removed or repurposed. New tests for per-run mapping confirmation are needed.
- **Admin settings template** — The column mapping section in `shop/templates/admin/settings.html` (lines 73-83 roughly) must be removed. The `/admin/settings/upload` and `/admin/settings/save` routes in `admin.py` must be removed. The shared `step4_mapping_partial.html` template has a `form_action` parameter — it can be reused for the per-run mapping partial without modification.

## Existing Codebase / Prior Art

- `shop/routers/setup.py` — 4-step wizard; steps 3 and 4 to be removed; `wizard_step` cap drops to 2; `setup_complete=True` moves to end of step 2 POST
- `shop/templates/setup/step3_engineer.html` — to be deleted
- `shop/templates/setup/step4_column_mapping.html` — to be deleted (step-level wrapper); inner partial reused
- `shop/templates/setup/step4_mapping_partial.html` — REUSE as per-run column mapping partial (already has `form_action` parameter); rename or keep in place
- `shop/templates/setup/step4_error.html` — REUSE for per-run xlsx error display
- `shop/routers/runs.py` — `submit_run` POST and `new_run_form` GET; xlsx validation happens here; column mapping confirmation fits as an HTMX partial triggered on xlsx file change (same pattern as validate-pdf)
- `shop/templates/runs/new.html` — Form 3 xlsx section currently has no HTMX trigger; add trigger + target div for mapping confirmation partial
- `shop/routers/admin.py` — `/settings/upload` and `/settings/save` routes to be removed; settings page GET and retention POST remain
- `shop/templates/admin/settings.html` — column mapping section to be removed
- `shop/services/form3.py` — `parse_excel_preview()` and `detect_column_mapping()` already implement auto-detection; no changes needed
- `shop/middleware/setup_guard.py` — no changes needed; still checks `setup_complete` flag
- `tests/test_setup.py` — step 3 block-skip test must be removed; step 4 upload/autodetect/empty/noncontiguous tests must be moved/repurposed as per-run mapping tests
- `delta_preservation/io/xlsx.py:load_form3()` — pipeline column detection; confirmed NOT consuming `ShopConfig.column_mapping`; no changes

> See `.gsd/DECISIONS.md` for all architectural and pattern decisions — it is an append-only register; read it during planning, append to it during execution.

## Relevant Requirements

- Active: `WORK-03` gap (requirement_revB blank) — not addressed by this milestone; remains active
- Active: `PIPE-05` dead code — not addressed by this milestone; remains active
- Active: `Run.signed_by` ORM gap — not addressed by this milestone; remains active
- This milestone introduces new in-scope capability: per-run column mapping confirmation (new requirement, not previously tracked)

## Scope

### In Scope

- Reduce setup wizard from 4 steps to 2 (shop name + admin password only)
- Display `admin@shop.local` as the fixed login email on step 2 (password setup screen)
- Set `setup_complete=True` after step 2 POST completes (was after step 4)
- Remove step 3 (engineer creation) and step 4 (column mapping) routes and templates from wizard
- Add HTMX-triggered per-run column mapping confirmation to the new run form (xlsx upload triggers auto-detect → inline mapping partial with hidden confirmed inputs → submitted with run POST)
- Add a validate-xlsx endpoint (analogous to validate-pdf) that returns the mapping confirmation partial
- Remove column mapping upload/save from admin settings page and router
- Update `wizard_step` cap and guard logic (max step = 2)
- Update tests: remove/repurpose wizard step 3/4 tests; add per-run column mapping confirmation tests
- Ensure fresh Docker setup + new run flow works end-to-end

### Out of Scope / Non-Goals

- Passing the confirmed column mapping to `run_pipeline()` / overriding `load_form3()` detection — pipeline detection is independent and sufficient; the UI confirmation is informational
- Any changes to the pipeline itself
- Changes to review, sign-off, export, amendment, or auth flows
- Adding the confirmed mapping as a Run model field (would require DB migration; not warranted)
- Removing `ShopConfig.column_mapping` column from the database (harmless to leave; no migration needed)
- Engineer password reset or account management beyond what already exists

## Technical Constraints

- Standard HTML POST (not HTMX) for wizard steps — established pattern from M001; HTMX intercepts 302 redirects as partial swaps
- HTMX partial swap pattern for per-run xlsx validation — matches existing validate-pdf pattern; consistent UX
- `wizard_step` in ShopConfig must remain valid for deployments that already completed the 4-step wizard (their `wizard_step=4`; guard logic must treat any value ≥ 2 as "step 2 complete"); `run_schema_migrations()` does not need a change since no new columns are added
- Python 3.10–3.13, uv, FastAPI, Jinja2, HTMX — no new dependencies

## Integration Points

- `ShopConfig` (DB) — `wizard_step` and `setup_complete` fields; `column_mapping` remains but is no longer written by setup
- `shop/services/form3.py` — `parse_excel_preview()` and `detect_column_mapping()` used by the new per-run validate-xlsx endpoint
- `shop/templates/setup/step4_mapping_partial.html` — reused for per-run confirmation partial (already parameterized with `form_action`)
- `shop/routers/runs.py` — new `/runs/validate-xlsx` endpoint; `submit_run` POST reads confirmed mapping from hidden form inputs (or ignores it — see scope note above)

## Open Questions

- None — scope is well-defined.
