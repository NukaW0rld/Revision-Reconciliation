# M002: Setup Simplification & Per-Run Column Mapping — Research

**Date:** 2026-03-13

## Summary

The codebase is well-understood and this milestone is surgical. The 4-step wizard lives entirely in `shop/routers/setup.py` with corresponding templates. Steps 3 and 4 are cleanly separated from steps 1 and 2 — there are no shared state transitions that would make removal risky. `setup_complete=True` currently fires at the end of `step4_save`; moving it to the end of `step2_post` (after the `if config.wizard_step < 2: config.wizard_step = 2` block) is the only wiring change needed. The wizard progress cap in `setup_root` (`min(config.wizard_step + 1, 4)`) drops to 2. Existing deployments with `wizard_step=4` are already `setup_complete=True` so the guard is irrelevant to them; the cap change only affects the setup root redirect.

The per-run column mapping confirmation requires careful design because `step4_mapping_partial.html` is a **standalone `<form>` with its own POST** — it cannot be dropped inline into `runs/new.html` as-is. The page-selector and pdf-error HTMX partials (`_page_selector.html`, `_pdf_error.html`) output raw HTML fragments that become part of the outer form; the column mapping confirmation must follow the same pattern. A new `runs/_xlsx_mapping.html` partial should output column-mapping `<select>` elements (no wrapping `<form>`, no submit button) that land inside the existing `runs/new.html` form via HTMX swap. The mapping confirm button is not needed — the selects are already in the outer form and submit with it. The existing `step4_mapping_partial.html` stays for wizard/admin use; the new partial is runs-specific.

The admin settings column mapping section (`shop/templates/admin/settings.html` lines 64-103, plus `settings_upload` and `settings_save` routes in `admin.py`) is straightforward removal. No tests currently cover the admin settings upload/save endpoints, so nothing breaks there. The test surgery is the most mechanical part: 4 tests in `test_setup.py` hit `/setup/step4/upload` (wizard-step-3 seeded, session cookie required). These must be moved to per-run mapping tests hitting the new `/runs/validate-xlsx` endpoint. The `test_setup_step3_blocks_skip` test must be deleted or repurposed as a 404 check.

## Recommendation

Implement in a single slice with four sequentially dependent tasks: (1) strip wizard to 2 steps + update all guard logic, (2) add `/runs/validate-xlsx` endpoint + `runs/_xlsx_mapping.html` partial + HTMX trigger on xlsx input, (3) remove admin settings column mapping routes and template section, (4) rewrite the test suite (remove wizard step 3/4 tests, add per-run mapping tests). The slice branch is `gsd/M002/S01`. No DB migration is required; no new dependencies are introduced.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| xlsx column auto-detection | `shop/services/form3.py:parse_excel_preview()` + `detect_column_mapping()` | Already implements keyword-based detection against `FORM3_HEADER_KEYWORDS`; tested via the 4 setup tests being repurposed |
| HTMX file-change validation partial | `validate_pdf` endpoint + `_page_selector.html` / `_pdf_error.html` pattern | Exact same pattern to follow: `hx-post`, `hx-trigger="change"`, `hx-encoding="multipart/form-data"`, `hx-target`, `hx-swap="innerHTML"` |
| Excel empty/unreadable error display | `shop/templates/setup/step4_error.html` | Reuse directly in `validate-xlsx` error path — already styled, no form wrapping |
| Auth in multipart HTMX endpoints | `get_current_user` dependency | `validate_pdf` already uses it; `validate-xlsx` must too — the partial is auth-gated |
| Test session cookie pattern | `_make_session_cookie()` in `test_admin.py` | Reuse this helper in per-run mapping tests; same `UserSession` seeding + cookie pattern |

## Existing Code and Patterns

- `shop/routers/setup.py` — `step2_post` currently redirects to step 3 then sets `wizard_step=2`; change: after setting `wizard_step=2`, also set `setup_complete=True` and redirect to `/login` instead of `/setup/step3`. Remove `step3_get`, `step3_post`, `step4_get`, `step4_upload`, `step4_save`. Update `setup_root` cap from 4 to 2.
- `shop/templates/setup/wizard_layout.html` — Progress bar hardcoded to `[(1,'Shop'), (2,'Password'), (3,'Engineer'), (4,'Form 3')]` with `of 4` in title. Both must change to 2 steps: `[(1,'Shop'), (2,'Password')]` and `of 2`.
- `shop/templates/setup/step2_password.html` — Already shows `admin_email` in an info alert. After milestone, this is the final wizard step so the continue button should read "Finish Setup" or "Complete Setup" rather than "Continue" to signal completion.
- `shop/templates/runs/new.html` — Form 3 xlsx file input currently has no HTMX attributes and only says "Excel validation occurs on submit." Add: `hx-post="/runs/validate-xlsx"`, `hx-trigger="change"`, `hx-encoding="multipart/form-data"`, `hx-target="#xlsx-mapping-section"`, `hx-swap="innerHTML"`. Add target div `<div id="xlsx-mapping-section"></div>` below the file input.
- `shop/templates/setup/step4_mapping_partial.html` — **Do not modify.** Used by wizard (form_action=/setup/step4/save) and admin settings (form_action=/admin/settings/save). Creates a standalone `<form>` with its own POST — correct for those use cases.
- `shop/templates/setup/step4_error.html` — Reuse verbatim in `validate-xlsx` error path. Already a bare HTML fragment (no `<form>` wrapper), suitable as a partial swap target.
- `shop/routers/runs.py:validate_pdf` — Template for the new `validate_xlsx` endpoint. Key detail: reads file via `form = await request.form()` iteration (not typed `UploadFile`) because HTMX includes all form fields in the multipart request. The new endpoint can use a typed `UploadFile` param since it only needs the xlsx field — but following the existing pattern (iterate form, find file) is safer and consistent.
- `shop/routers/admin.py:settings_upload` + `settings_save` — These routes reuse `parse_excel_preview` and `step4_mapping_partial.html` with `form_action="/admin/settings/save"`. Both routes and the admin settings template section (lines 64-103) are removed in task 3.
- `tests/test_setup.py` — `test_form3_upload_autodetect`, `test_empty_file_error`, `test_noncontiguous_char_no` all seed `wizard_step=3` and post to `/setup/step4/upload` with a session cookie. These 3 are the exact tests to repurpose as per-run mapping tests (same xlsx fixtures, new endpoint URL, no wizard_step seed). `test_setup_step3_blocks_skip` must be removed (step 3 no longer exists).

## Constraints

- `step4_mapping_partial.html` uses `<form method="post" action="{{ form_action }}">` — it is a complete form, not an inline fragment. A new partial is needed for the per-run context where selects must be inside the outer form.
- HTMX intercepts 302 redirects as partial swaps (see DECISIONS.md). The wizard steps 1 and 2 use standard HTML POST with `follow_redirects=False` in tests — that pattern must be preserved; no HTMX for wizard POSTs.
- `wizard_step=4` rows in existing deployed DBs: `setup_complete=True` is already set on those rows, so the setup guard never runs and the cap change has no effect on them. No migration needed.
- `validate-xlsx` endpoint must be auth-gated (`get_current_user` dependency) because it's inside the authenticated `/runs` area.
- The new `runs/_xlsx_mapping.html` partial outputs bare `<select name="col_N">` elements (no form, no submit button). The submit button on `runs/new.html` submits the entire form including these selects. The endpoint (`submit_run`) already ignores unrecognized form fields, so unmapped column selects will just be ignored — which is correct behavior.
- `submit_run` in `runs.py` currently has no logic to read confirmed column mapping from the POST. Per scope, it doesn't need to — the mapping confirmation is UX only. No changes needed to `submit_run` beyond this milestone.

## Common Pitfalls

- **New partial inside outer form** — If `step4_mapping_partial.html` is reused directly for the per-run case, the nested `<form>` will be invalid HTML and browsers will break outer form submission. Create `runs/_xlsx_mapping.html` as a fragment-only partial (no `<form>` tag, just the `<select>` elements with a surrounding `<div>`).
- **Wizard cap off-by-one** — `setup_root` does `min(config.wizard_step + 1, 4)`. After change it must be `min(config.wizard_step + 1, 2)`. If left at 4, a fresh install (wizard_step=0) correctly goes to step1 but a completed install (wizard_step=2) would still try to resolve step3 instead of redirecting to login.
- **step2_post redirect** — Currently redirects to `/setup/step3`. After change: redirect to `/login` (setup is complete). The `/setup/` root redirect and the guard middleware are not involved here; step2_post must produce the redirect directly.
- **Test cookie deprecation warning** — `cookies={"session_token": token}` on individual requests generates a `DeprecationWarning` in the current starlette version. The existing tests already have this warning (3 warnings in test_setup.py, 6 in test_admin.py). New tests should use the same pattern for consistency, not introduce a different approach — the warning is non-blocking.
- **`validate_excel_bytes` already called on submit** — `submit_run` already calls `validate_excel_bytes(form3_bytes)` which raises `ValueError` on bad xlsx. The `validate-xlsx` HTMX endpoint is in addition to this, not instead. The submit-time validation is a safety net and must be kept.
- **Removing admin settings routes without removing template section** — Both the router routes (`settings_upload`, `settings_save`) and the template section (the column mapping `<div class="card">` in `settings.html`) must be removed together. If only one is removed, the page will render a broken upload form or throw a 405.

## Open Risks

- **Amber column UX in runs context** — The per-run partial shows undetected columns in amber with a `(ignore)` default. If an engineer submits without fixing an amber column, the pipeline's own keyword detection will still find it (since `load_form3()` is independent). The UX confirmation is informational; silent ignore is acceptable. No risk here, just a UX note to document in the form label.
- **HTMX on xlsx input + full form multipart** — The existing `validate_pdf` HTMX endpoints receive the full form multipart (HTMX sends all fields when `hx-encoding="multipart/form-data"` is set on a form-child input). The new `validate-xlsx` endpoint only needs the xlsx bytes; it should be resilient to receiving extra form fields (pdf files). Using `form = await request.form()` iteration to find the first `.xlsx`-type file is the safe approach.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| FastAPI + HTMX + Jinja2 | — | No relevant skill found; existing codebase patterns are sufficient |

## Sources

- Codebase: `shop/routers/setup.py`, `shop/routers/runs.py`, `shop/routers/admin.py`, `shop/services/form3.py`, `shop/templates/setup/`, `shop/templates/runs/new.html`, `shop/templates/admin/settings.html`, `tests/test_setup.py`, `tests/conftest.py`
- `uv run pytest --tb=short` — 87 passed, 2 xfailed baseline confirmed
