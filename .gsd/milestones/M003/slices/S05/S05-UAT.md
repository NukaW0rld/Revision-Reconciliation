# S05: Admin, Setup Wizard & Fragment Cleanup — UAT

**Milestone:** M003
**Written:** 2026-03-14

## UAT Type

- UAT mode: mixed (artifact-driven + live-runtime)
- Why this mode is sufficient: The primary deliverable is CSS class substitution — grep checks confirm zero legacy patterns (artifact-driven). Visual inspection in a running browser confirms admin screens render correctly with the industrial aesthetic (live-runtime). The test suite (93 passed) covers all admin and setup template rendering paths.

## Preconditions

1. App is running: `uv run python run_web.py` (default: http://localhost:8000)
2. First-run setup is complete (shop config exists) — or test with `client_setup_incomplete` fixture path
3. Admin user account available (email + password)
4. Engineer user account available for RBAC checks
5. `npm run build:css` has been run and `static/dist/output.css` is current

## Smoke Test

Navigate to `/admin/users` while logged in as an admin. Confirm the page renders with a dark background, no white card backgrounds, no DaisyUI default alert boxes visible anywhere, and user rows display "Active"/"Inactive" as green/red monospaced text (not colored badges).

## Test Cases

### 1. Admin user management — industrial layout

1. Log in as admin; navigate to `/admin/users`
2. Inspect the page header area — confirm `{% block page_title %}` renders "User Management" in the top bar
3. Inspect the users table section — confirm it has a dark bordered wrapper (`border border-base-300 bg-base-200`) with no white card or drop shadow visible
4. Inspect the table header row — confirm column labels use monospaced uppercase text (not standard sentence-case sans-serif)
5. Inspect the "Active" status column — confirm it renders as green monospaced text (`text-success font-mono`), not a green badge pill
6. Inspect the "Inactive" status column — confirm it renders as red monospaced text (`text-error font-mono`), not a red badge pill
7. Inspect the "Add Engineer" form section — confirm it has a dark bordered wrapper
8. **Expected:** Entire page has dark background, bordered sections, mono text for status — zero white cards, zero DaisyUI badge pills for status

### 2. Admin user management — Add Engineer form validation error

1. On `/admin/users`, submit the Add Engineer form with an email that already exists
2. **Expected:** Error message appears as a left-bordered banner (`border-l-4 border-error bg-base-300` or `bg-base-200`) — not as a DaisyUI `alert alert-error` box. Error text is legible against the dark background.

### 3. Admin user management — HTMX new user row insertion

1. On `/admin/users`, fill in valid name/email for a new engineer; submit the form
2. **Expected:** New row appears in the users table body via HTMX swap (no full page reload). The new row uses the same styling as existing rows: dark hover state, role badge (badge-primary/badge-neutral), status as font-mono text.

### 4. Admin user management — deactivate engineer HTMX update

1. On `/admin/users`, click "Deactivate" for an active engineer
2. **Expected:** The row updates in place via HTMX (no full page reload). The status cell changes from green "Active" text to red "Inactive" text. Row styling matches the template in `users_row.html`.

### 5. Admin settings — industrial layout

1. Navigate to `/admin/settings`
2. Confirm `{% block page_title %}` renders "Shop Settings" in the top bar
3. Inspect settings form sections — confirm bordered dark wrappers (no card/shadow)
4. Inspect form inputs — confirm dark background inputs with monospaced font and primary-color focus ring
5. **Expected:** Entire page dark, bordered sections, no white cards, industrial input styling throughout

### 6. Admin settings — success/error feedback banners

1. On `/admin/settings`, submit the form successfully (e.g., update shop name to current value)
2. **Expected:** Success banner appears as `border-l-4 border-success bg-base-200` (or similar) — not `alert alert-success`
3. Optionally, trigger a validation error
4. **Expected:** Error banner appears as `border-l-4 border-error` — not `alert alert-error`

### 7. Setup wizard step 1 — industrial input styling

1. Navigate to `/setup/` (or use a fresh DB state)
2. On step 1 (shop name), inspect the input field
3. **Expected:** Input has dark background (`bg-base-200`), monospaced font, primary-color focus ring — matches login.html and runs/new.html input styling

### 8. Setup wizard step 2 — info banner and industrial inputs

1. Proceed to step 2 (admin password setup)
2. Inspect the info banner (if present)
3. **Expected:** Info banner renders as `border-l-4 border-info bg-base-300` — not `alert alert-info`
4. Inspect password inputs
5. **Expected:** Industrial input styling consistent with step 1 and other forms

### 9. Setup wizard — error state

1. On the wizard layout, trigger an error condition (submit invalid data)
2. **Expected:** Error message appears as `border-l-4 border-error bg-base-300` inside the wizard panel — not `alert alert-error`

### 10. Setup wizard — step 4 error page

1. Navigate to (or simulate) a state where `step4_error.html` is rendered (e.g., form3 parsing failure)
2. **Expected:** Error message uses `border-l-4 border-error bg-base-200` — dark background, left accent, no white alert box

### 11. Setup wizard — step 4 mapping partial (column mapping)

1. In the setup wizard's column mapping step, observe any columns that were not auto-matched
2. **Expected:** Unmatched column rows have a subtle warning background (`bg-warning/15`) with `border-warning` accent — no amber/yellow Tailwind utility color (`bg-amber-100`, `border-amber-400`). Legend swatch matches the warning token color.

### 12. Login page — error banner

1. Navigate to `/login`; submit incorrect credentials
2. **Expected:** Error banner appears as `border-l-4 border-error bg-base-300` in the dark login panel — not `alert alert-error`. Dark background preserved, no white alert box.

### 13. New run form — validation error and input styling

1. Navigate to `/runs/new`; submit without selecting files
2. **Expected:** Any error banner uses `border-l-4 border-error bg-base-200` — not `alert alert-error`
3. Inspect the form inputs (part name, file selects)
4. **Expected:** Text inputs have industrial styling (`bg-base-200 font-mono focus:border-primary`) — no default browser or light-mode input appearance

### 14. PDF validation error partial

1. Submit the new run form with an invalid PDF (if testable in browser)
2. **Expected:** The `_pdf_error.html` partial renders with `border-l-4 border-error bg-base-200` — not `alert alert-error`. Fragment integrates visually with the surrounding dark form.

### 15. Full-page grep audit (automated)

Run from project root:
```bash
grep -rn 'alert alert-\|card bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/
grep -rn 'badge-success\|badge-error' shop/templates/
```
**Expected:** Both commands exit 1 (no matches). Any match output names a specific file and line requiring cleanup.

### 16. CSS build and test suite

```bash
npm run build:css
uv run pytest tests/
```
**Expected:** `npm run build:css` exits 0 with no Tailwind purge warnings for new classes. `pytest` exits 0 with 93 passed, 2 xfailed, 0 failures.

## Edge Cases

### Role badge preservation in users table

1. In the users table, find a user with Admin role and a user with Engineer role
2. **Expected:** Admin role shows `badge-primary` styling; Engineer role shows `badge-neutral` styling. These role badges are intentionally preserved (they indicate access level, not status). They should NOT have been converted to font-mono text.

### Nav notification dot (base.html)

1. While logged in as admin, create an alert condition that shows the red notification dot in the sidebar nav
2. **Expected:** Notification dot is a small red circle — visually identical to before. It is implemented as `bg-error text-error-content` inline div, not a DaisyUI `badge badge-error` — but the visual result should be indistinguishable.

### Empty admin table

1. Deactivate all engineers (leave only admin account)
2. **Expected:** Users table renders with an empty tbody — no layout breakage, no JavaScript errors, no template errors. The bordered section wrapper still renders correctly.

## Failure Signals

- Any white or light-gray background visible on admin pages → legacy `card bg-base-100` or `bg-white` not converted
- Colored pill/badge for "Active"/"Inactive" status → `badge-success`/`badge-error` not converted (grep check catches this)
- DaisyUI full-width colored alert box → `alert alert-*` not converted (grep check catches this)
- `npm run build:css` Tailwind purge warning → a class like `bg-warning/15` is in templates but not recognized by Tailwind config
- 500 response on any admin or setup page → Jinja2 UndefinedError or template syntax error (pytest catches this)
- `pytest` failure in `test_admin.py` → admin template breakage (check specific test for which route returned 500)

## Requirements Proved By This UAT

- R012 — Admin user management and settings screens redesigned with industrial aesthetic; verified in browser and test suite
- R013 — All HTMX fragment templates (`users_row.html`, `_pdf_error.html`) match redesigned aesthetic; swap targets preserved; confirmed by grep + browser
- R001 — Industrial dark-mode aesthetic now covers the entire template tree; confirmed by zero-match grep audit

## Not Proven By This UAT

- R014 (dark/light mode toggle) — deferred; not implemented in M003
- WeasyPrint PDF template aesthetics (`audit_packet.html`, `work_order.html`) — explicitly out of scope (D005); these are print templates with separate styling concerns
- Keyboard shortcut behavior in review queue — proved by S04 UAT, not re-proved here
- SSE stage checklist live updates — proved by S03 UAT, not re-proved here

## Notes for Tester

- The `badge-primary` and `badge-neutral` role badges on user rows are intentionally preserved — they indicate access level and are not status indicators. Do not confuse them with the converted `badge-success`/`badge-error` status badges.
- `bg-warning/15` uses Tailwind's opacity modifier syntax — if the color looks wrong (too opaque or missing entirely), run `npm run build:css` and check for purge warnings about this class.
- The setup wizard tests require a clean or reset database state. In the test suite, `client_setup_incomplete` fixture handles this isolation.
- Admin pages require admin-role login. Attempting to access `/admin/*` as an engineer user should redirect or return 403 — this is RBAC behavior tested in `test_rbac.py`, not a visual concern.
