# S05: Admin, Setup Wizard & Fragment Cleanup

**Goal:** Every screen and partial in the app uses the industrial dark-mode aesthetic — zero legacy DaisyUI card/shadow/alert patterns remain anywhere in the template tree.
**Demo:** Admin user management, admin settings, setup wizard steps, and all HTMX partials render with bordered sections, font-mono accents, and border-l-4 alert banners. The full app is visually consistent end-to-end.

## Must-Haves

- Admin user table uses industrial section wrapper, mono table headers, hover rows — matches `runs/list.html` pattern
- Admin "Add Engineer" form uses industrial input styling (`bg-base-200 font-mono focus:border-primary`) — matches `login.html` pattern
- Admin settings form sections use bordered wrappers with industrial inputs — no card/shadow
- `users_row.html` HTMX partial matches `users.html` row styling exactly (badge strategy consistent)
- Role badges keep `badge-primary`/`badge-neutral`; status indicators use `text-success`/`text-error` font-mono text (not badge classes)
- `{% block page_title %}` added to both admin templates
- Setup wizard sub-templates (`step1`, `step2`) use industrial input styling and font-mono labels
- `step2_password.html` `alert-info` → `border-l-4 border-info bg-base-300` pattern
- `wizard_layout.html` `alert-error` → `border-l-4 border-error bg-base-300` pattern
- `step4_error.html` → `border-l-4 border-error bg-base-200` pattern
- `step4_mapping_partial.html` → `bg-warning/15 border-warning` (matching `_xlsx_mapping.html`)
- Prior-slice cleanup: `login.html` alert-error → border-l-4, `runs/new.html` alert-error → border-l-4 + industrial inputs, `_pdf_error.html` → border-l-4
- All alert-error/alert-info/alert-success instances across entire template tree → 0
- All card bg-base-100 shadow instances across entire template tree → 0
- All HTMX swap targets preserved unchanged (`id="users-table"`, `hx-target="closest tr"`, `hx-target="#users-table tbody"`)
- `npm run build:css` → exit 0
- `pytest` → 93 passed, 0 failures

## Verification

- `grep -rn 'alert alert-\|card bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/` → 0 matches
- `grep -rn 'badge-success\|badge-error' shop/templates/` → 0 matches (status badges converted to text)
- `grep 'block page_title' shop/templates/admin/users.html shop/templates/admin/settings.html` → both present
- `grep 'id="users-table"' shop/templates/admin/users.html` → present
- `grep 'hx-target="#users-table tbody"' shop/templates/admin/users.html` → present
- `grep 'hx-target="closest tr"' shop/templates/admin/users.html` → present
- `npm run build:css` → exit 0
- `uv run pytest tests/` → 93 passed, 2 xfailed, 0 failures
- Visual spot-check: admin/users and admin/settings in running browser — dark bordered sections, mono headings, no light-mode artifacts

## Observability / Diagnostics

This slice is a CSS class substitution pass — no runtime logic changes. Failure modes are visual or structural:

- **Grep checks are the primary signal.** `grep -rn 'alert alert-\|card bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/` → 0 matches confirms no legacy patterns remain. Any non-zero output names the exact file and line to fix.
- **CSS build surface.** `npm run build:css` will emit Tailwind purge warnings if a newly-added class doesn't appear in the scanned source — if a class like `bg-warning/15` is unknown, it means the DaisyUI/Tailwind config doesn't support it (observable via build output).
- **Test suite.** Template rendering tests in `tests/test_admin.py` and `tests/test_runs.py` catch broken Jinja2 syntax or missing context variables — a Jinja2 `UndefinedError` surfaces as a 500 in tests.
- **Visual inspection.** Admin pages in browser confirm no light-mode artifacts (white card backgrounds, DaisyUI alert colors). The dark background of `bg-base-200` bordered sections should be consistent across all pages.
- **Redaction:** No secrets, credentials, or PII pass through templates modified in this slice.

## Tasks

- [x] **T01: Apply industrial aesthetic to all remaining templates and clean up prior-slice legacy patterns** `est:30m`
  - Why: Every pattern needed already exists in prior-slice templates; this is CSS class substitution across ~10 files with no structural changes
  - Files: `shop/templates/admin/users.html`, `shop/templates/admin/users_row.html`, `shop/templates/admin/settings.html`, `shop/templates/setup/step1_shop_name.html`, `shop/templates/setup/step2_password.html`, `shop/templates/setup/wizard_layout.html`, `shop/templates/setup/step4_error.html`, `shop/templates/setup/step4_mapping_partial.html`, `shop/templates/auth/login.html`, `shop/templates/runs/new.html`, `shop/templates/runs/_pdf_error.html`
  - Do: (1) Admin users.html — replace card/shadow wrappers with bordered sections, add mono table headers and hover rows, convert status badges to font-mono text-success/text-error, add industrial input styling to Add Engineer form, add page_title block, convert alert-error to border-l-4. (2) Admin users_row.html — match users.html row styling exactly: role badges keep badge-primary/badge-neutral, status → font-mono text, error row → border-l-4. (3) Admin settings.html — replace card/shadow with bordered sections, add industrial inputs, add page_title block, convert alert-error/alert-success to border-l-4. (4) Setup step1/step2 — add font-mono labels and industrial input styling; step2 alert-info → border-l-4 border-info bg-base-300. (5) wizard_layout.html — alert-error → border-l-4 border-error bg-base-300. (6) step4_error.html — alert-error → border-l-4 border-error bg-base-200. (7) step4_mapping_partial.html — bg-amber-100 border-amber-400 → bg-warning/15 border-warning. (8) login.html — alert-error → border-l-4 border-error. (9) runs/new.html — alert-error → border-l-4, inputs → industrial styling. (10) _pdf_error.html — alert-error → border-l-4 border-error. Constraint: all HTMX swap targets must be preserved verbatim.
  - Verify: `grep -rn 'alert alert-\|card bg-base-100 shadow\|bg-amber-100' shop/templates/` → 0; `npm run build:css` → exit 0; `uv run pytest tests/` → 93 passed; visual spot-check admin pages in browser
  - Done when: zero legacy DaisyUI alert/card/shadow classes remain in any template; CSS builds clean; tests pass; admin pages render with industrial aesthetic in browser

## Files Likely Touched

- `shop/templates/admin/users.html`
- `shop/templates/admin/users_row.html`
- `shop/templates/admin/settings.html`
- `shop/templates/setup/step1_shop_name.html`
- `shop/templates/setup/step2_password.html`
- `shop/templates/setup/wizard_layout.html`
- `shop/templates/setup/step4_error.html`
- `shop/templates/setup/step4_mapping_partial.html`
- `shop/templates/auth/login.html`
- `shop/templates/runs/new.html`
- `shop/templates/runs/_pdf_error.html`
