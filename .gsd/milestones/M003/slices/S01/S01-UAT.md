# S01: Design Foundation & Layout Shell — UAT

**Milestone:** M003
**Written:** 2026-03-14

## UAT Type

- UAT mode: live-runtime
- Why this mode is sufficient: S01's primary deliverable is visual — theme colors, layout structure, and page rendering. Contract checks (CSS build, grep) verify structure; live-runtime checks verify the correct theme is served to a real browser at runtime. No human UAT required for this slice per the slice plan.

## Preconditions

1. `uv run python run_web.py` is running and listening on `http://localhost:8000`
2. A setup-complete database exists (or `DELTA_SKIP_SETUP=1` env var set) so the app routes correctly
3. `npm run build:css` has been run at least once since the last `input.css` change
4. For dashboard/sidebar verification: a valid user account exists to log in with

## Smoke Test

```
curl -s http://localhost:8000/login | grep 'data-theme="industrial"'
```
Must return a line containing `data-theme="industrial"`. If it does, the theme is active and the standalone shell is wired.

## Test Cases

### 1. CSS build produces industrial theme

1. Run `npm run build:css` from the project root
2. Check exit code: `echo $?`
3. Run `grep -c '\[data-theme' static/dist/output.css`
4. **Expected:** exit code 0; grep count = 1; build output shows daisyUI version line and "Done in Xms"

### 2. Login page — industrial dark aesthetic, no white card

1. Open `http://localhost:8000/login` in a browser (or `curl -s http://localhost:8000/login`)
2. Inspect the `<html>` element
3. **Expected:** `data-theme="industrial"` attribute present on `<html>`
4. Verify the page background is near-black (not white or light grey)
5. **Expected:** No `card bg-base-100 shadow-xl` pattern visible; no white card on white background
6. Verify the heading reads `DELTA.PRES` in a monospace font with amber dot separator
7. **Expected:** `font-mono` class on heading; `text-primary` on the dot (amber)
8. Verify the submit button reads "Authenticate" and is full-width amber
9. **Expected:** `btn-primary` class, full-width, amber color from industrial theme

### 3. Login page — left branding rail present

1. Open `http://localhost:8000/login` in a browser at desktop viewport (≥768px)
2. Verify the left panel is visible with identity labels
3. **Expected:** "AS9102 Compliance", "FAIR Delta System", "Quality Team Only" text visible in left rail
4. **Expected:** Left rail uses `font-mono text-xs` with `tracking-widest uppercase` styling

### 4. Login error state — alert renders in dark theme

1. POST to `http://localhost:8000/login` with invalid credentials (or submit the login form with wrong email/password)
2. **Expected:** Error alert appears with `alert-error` styling — dark red background, white text, no white card
3. **Expected:** Alert uses `font-mono` class consistent with page typography
4. **Expected:** Form inputs are not cleared and the page does not break layout

### 5. Setup wizard step 1 — dark theme, custom step indicator

1. Trigger first-run state (clear DB or use a fresh test DB) so `/setup/step1` is accessible, or navigate directly if app is mid-setup
2. Open `http://localhost:8000/setup/step1`
3. Inspect `<html>` element
4. **Expected:** `data-theme="industrial"` present; near-black background (no sidebar, no top bar)
5. Verify the step progress indicator at the top of the form panel
6. **Expected:** Two step nodes visible; step 1 (active) shows `01` with `border-primary` styling in amber; step 2 shows `02` with muted/inactive styling
7. Verify the form panel has hard edges (no border-radius)
8. **Expected:** Panel uses `rounded-none` or radius = 0 from theme tokens (no soft card corners)

### 6. Setup wizard step 2 — completed step indicator

1. Navigate to `http://localhost:8000/setup/step2`
2. **Expected:** `data-theme="industrial"` on `<html>`; no sidebar
3. Verify the step progress indicator
4. **Expected:** Step 1 node is filled (`bg-primary`) with a checkmark SVG; step 2 node is active (`border-primary text-primary`)
5. Verify the password form renders with dark-themed inputs and amber submit button

### 7. Dashboard — sidebar + top bar layout present

1. Log in with valid credentials and navigate to `http://localhost:8000/` (dashboard)
2. **Expected:** Dark near-black background; no white/light background
3. Verify sidebar is visible on the left
4. **Expected:** `<aside>` element with `w-56` class; navigation links for Dashboard, Runs, New Run, Admin visible
5. Verify top bar is visible at the top of the content area
6. **Expected:** Sticky header with shop name and user info; `bg-base-200 border-b border-base-300` styling

### 8. Sidebar active-link highlighting

1. Log in and navigate to `http://localhost:8000/` (dashboard)
2. Inspect the sidebar nav links
3. **Expected:** "Dashboard" link has `border-l-2 border-primary` class (amber left border active indicator); other links do not
4. Navigate to `/runs/`
5. **Expected:** "Runs" link is now active with `border-l-2 border-primary`; "Dashboard" link is inactive

### 9. All semantic color classes resolve from industrial theme

1. From a running browser with DevTools open, inspect any `btn-primary` element on the dashboard
2. **Expected:** Computed color is amber (`oklch(75% 0.19 75)` or equivalent computed RGB ~`#c49a12` / amber)
3. Inspect any `badge-success` or green status element
4. **Expected:** Computed color is green from the industrial theme (not DaisyUI default blue/green)
5. **Expected:** No generic blue `#570df8` DaisyUI primary color visible anywhere

### 10. No sidebar on standalone pages

1. Open `http://localhost:8000/login`
2. **Expected:** No `<aside>` element in the DOM; no sidebar navigation visible
3. Open `http://localhost:8000/setup/step1`
4. **Expected:** No `<aside>` element in the DOM; no sidebar navigation visible
5. Confirm both pages are visually full-width without a left navigation column

## Edge Cases

### Login with empty fields

1. Submit the login form with both fields empty
2. **Expected:** Browser-native validation fires (required fields); no server error; no layout break

### Very long shop name in top bar

1. Log in when shop name is set to a long string (e.g., "Aerospace Manufacturing Quality Assurance Division 1")
2. View the top bar on the dashboard
3. **Expected:** Shop name truncates or wraps gracefully; top bar does not overflow or break layout

### No nav context vars on a page (missing shop_name / unread_alert_count)

1. Access a page that doesn't inject nav context (this is an internal check — review any page template that extends base.html but doesn't call `_get_nav_context()`)
2. **Expected:** Page renders without a Jinja2 UndefinedError; `| default()` guards prevent crashes; shop name shows blank or default rather than throwing 500

## Failure Signals

- `npm run build:css` exits non-zero → theme vars syntax error in `input.css`; check for unclosed braces or invalid OKLCH values
- `grep -c '\[data-theme' static/dist/output.css` returns 0 → theme block was dropped; rebuild and inspect `input.css`
- `http://localhost:8000/login` shows a white card or DaisyUI default styling → `_base_standalone.html` is not being extended; check the `{% extends %}` directive in `login.html`
- `http://localhost:8000/login` shows a sidebar → `login.html` still extends `base.html` instead of `_base_standalone.html`
- Jinja2 `UndefinedError` on `request` → `{% set path = request.url.path if request is defined else "" %}` guard missing from `base.html`
- `btn-primary` renders in blue instead of amber → DaisyUI default theme overriding industrial; check `data-theme` attribute on `<html>` and verify the custom theme block compiled
- Test suite below 93 passed → template or routing regression; run `uv run pytest tests/ -v` to identify the failing test

## Requirements Proved By This UAT

- R001 — Industrial dark-mode aesthetic system: OKLCH dark base, amber primary, and all semantic color vars compiled and active in browser
- R002 — Sidebar + top-bar navigation layout: sidebar renders on authenticated pages with active-link highlighting; top bar visible with context slot
- R003 — Login screen redesign: dark two-panel layout with monospace aerospace typography; no generic white card pattern
- R004 — Setup wizard redesign: dark panel with custom step indicator; both steps render correctly

## Not Proven By This UAT

- Sidebar behavior on non-dashboard authenticated pages (runs list, run detail, review queue) — these templates are not yet on the new base; proved in S02–S04
- HTMX partial swaps integrating visually with the new layout — proved in S02–S04
- Review queue keyboard shortcuts — proved in S04
- Sign-off flow end-to-end — proved in S04
- Admin screens — proved in S05

## Notes for Tester

- The app may need a seeded/setup database to reach the dashboard. Run the setup wizard at `/setup/step1` on first launch to create the admin account.
- If testing with a fresh DB, the setup wizard middleware (`setup_guard.py`) will redirect all routes to `/setup/` until setup is complete — this is expected behavior.
- The left branding rail on the login page is hidden on mobile viewports (`hidden lg:flex`). Use a desktop viewport (≥1024px) to verify it.
- DaisyUI v5 class names differ slightly from v4 in some utilities — if a class doesn't render as expected, verify against the compiled `output.css` rather than assuming the class name is wrong.
