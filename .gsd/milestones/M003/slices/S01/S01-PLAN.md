# S01: Design Foundation & Layout Shell

**Goal:** Establish the industrial dark-mode aesthetic, sidebar + top-bar layout, and standalone page shell — the visual foundation every downstream slice builds on.
**Demo:** App loads with dark industrial theme, sidebar navigation, and a redesigned login screen. Setup wizard renders in the new aesthetic without sidebar chrome.

## Must-Haves

- Custom `[data-theme="industrial"]` compiles via `npm run build:css` with no errors
- `base.html` renders a fixed sidebar + sticky top bar + scrollable content area
- Sidebar shows nav items (Dashboard, Runs, New Run, Admin) with active-link highlighting via `request.url.path`
- Top bar shows shop name, user email, alert badge, and logout
- `_base_standalone.html` provides the dark theme without sidebar for login/wizard
- Login page uses the industrial aesthetic — dark bg, no generic card-on-white
- Setup wizard (both steps) uses the industrial aesthetic with step progress indicator
- All DaisyUI semantic color vars defined (`--color-primary`, `--color-success`, `--color-error`, `--color-warning`, `--color-info`) so downstream badge/alert classes work
- Block names `title`, `nav`, `content` preserved in `base.html` for template compatibility
- Template context vars `shop_name`, `unread_alert_count`, `user` referenced with `| default()` guards so pages that don't pass all vars still render

## Proof Level

- This slice proves: contract (base layout + theme that all downstream templates will extend)
- Real runtime required: yes (must load in browser to verify theme renders correctly)
- Human/UAT required: no

## Verification

- `npm run build:css` exits 0 with no errors
- `uv run python run_web.py` starts without errors; login page loads at `http://localhost:8000/login` with dark theme visible
- After login, dashboard page shows sidebar + top bar layout
- Setup wizard at `/setup/` (when setup incomplete) renders dark-themed wizard without sidebar

## Observability / Diagnostics

- Runtime signals: Jinja2 `TemplateNotFound` errors surface in FastAPI error responses if template names change
- Inspection surfaces: browser DevTools — verify `data-theme="industrial"` on `<html>`, check computed CSS vars
- Failure visibility: CSS build errors in `npm run build:css` stdout; broken template extends show as 500 errors in browser

## Integration Closure

- Upstream surfaces consumed: `_get_nav_context()` returns `shop_name` + `unread_alert_count`; `user` object from auth middleware
- New wiring introduced in this slice: `_base_standalone.html` (new template); `data-theme="industrial"` attribute on `<html>`
- What remains before the milestone is truly usable end-to-end: S02-S05 must redesign all remaining templates to use the new base layout and theme

## Tasks

- [x] **T01: Define industrial dark theme and build sidebar + top-bar layout shell** `est:1h`
  - Why: The custom DaisyUI theme and base layout are the load-bearing foundation — nothing else can be built until `output.css` compiles with the right colors and `base.html` provides the sidebar/top-bar shell
  - Files: `static/src/input.css`, `shop/templates/base.html`, `shop/templates/_base_standalone.html`
  - Do: Define `[data-theme="industrial"]` in `input.css` with OKLCH color vars (dark bases, amber primary, green success, all semantic colors, zero-radius tokens). Rewrite `base.html` as a flex sidebar + top-bar + scrollable content layout. Preserve block names `title`, `nav`, `content`. Create `_base_standalone.html` as a minimal dark-themed shell without sidebar for standalone pages. Use `| default()` for optional context vars. Run `npm run build:css` to verify. Load the app in browser and verify the theme visually.
  - Verify: `npm run build:css` exits 0; app loads at localhost with dark theme and sidebar visible on dashboard
  - Done when: `output.css` contains `[data-theme=industrial]` rules; `base.html` renders sidebar + top bar; `_base_standalone.html` exists and renders dark-themed pages

- [x] **T02: Redesign login screen and setup wizard with industrial aesthetic** `est:45m`
  - Why: Login (R003) and setup wizard (R004) are the first-impression screens; they must use the industrial theme without sidebar chrome
  - Files: `shop/templates/auth/login.html`, `shop/templates/setup/wizard_layout.html`
  - Do: Rewrite `login.html` to extend `_base_standalone.html` with a distinctive dark login panel — strong typography, monospaced accents, amber primary button. Rewrite `wizard_layout.html` to extend `_base_standalone.html` with industrial step indicators and form styling. Verify both pages render correctly in running browser.
  - Verify: Login page at `/login` renders dark-themed with no generic DaisyUI card appearance; setup wizard steps render dark-themed with step progress visible
  - Done when: Both login and setup wizard pages load in browser with the industrial aesthetic; no `card-on-white` pattern visible; form validation and error states still function

## Files Likely Touched

- `static/src/input.css`
- `static/dist/output.css` (generated)
- `shop/templates/base.html`
- `shop/templates/_base_standalone.html` (new)
- `shop/templates/auth/login.html`
- `shop/templates/setup/wizard_layout.html`
