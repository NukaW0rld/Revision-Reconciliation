---
id: S01
parent: M003
milestone: M003
provides:
  - Industrial DaisyUI v5 dark theme compiled in static/dist/output.css
  - sidebar + top-bar layout shell in shop/templates/base.html
  - standalone dark-theme base shell in shop/templates/_base_standalone.html
  - industrial dark login page at shop/templates/auth/login.html
  - industrial dark setup wizard layout at shop/templates/setup/wizard_layout.html
requires: []
affects:
  - S02
  - S03
  - S04
  - S05
key_files:
  - static/src/input.css
  - static/dist/output.css
  - shop/templates/base.html
  - shop/templates/_base_standalone.html
  - shop/templates/auth/login.html
  - shop/templates/setup/wizard_layout.html
key_decisions:
  - D007: _base_standalone.html is the standalone shell; login and wizard must extend it, not base.html
  - D008: DaisyUI v5 custom themes defined as raw [data-theme="name"]{...} CSS blocks outside @plugin directive
  - D009: Wizard step indicator uses custom flex divs instead of DaisyUI ul.steps — DaisyUI steps pseudo-element backgrounds don't render cleanly against oklch(13%) dark base
patterns_established:
  - DaisyUI v5 custom themes are raw [data-theme="industrial"]{...} CSS blocks in input.css, placed after the @plugin directive, not inside it
  - Sidebar active-link highlighting via Jinja2 request.url.path + path.startswith() comparisons
  - All optional template context vars guarded with | default() to prevent UndefinedError
  - <main class="flex-1 overflow-y-auto p-6"> is the scroll container (not body), enabling sticky footers inside the content pane
  - Login two-panel layout — left branding rail (hidden on mobile), right form panel; monospace aerospace identity markers
  - Wizard step indicator — flex + connector line + bordered nodes; active=border-primary, completed=bg-primary + checkmark SVG
observability_surfaces:
  - CSS build: `npm run build:css` exit code 0 = success; output shows daisyUI version
  - Theme presence: `grep -c '\[data-theme' static/dist/output.css` → 1
  - Runtime theme: `data-theme="industrial"` on <html> element in browser DevTools
  - Sidebar active state: border-l-2 border-primary class on active nav link
  - Template errors: 500 responses with Jinja2 TemplateNotFound/UndefinedError in FastAPI error body
drill_down_paths:
  - .gsd/milestones/M003/slices/S01/tasks/T01-SUMMARY.md
  - .gsd/milestones/M003/slices/S01/tasks/T02-SUMMARY.md
duration: ~1h
verification_result: passed
completed_at: 2026-03-14
---

# S01: Design Foundation & Layout Shell

**Established the industrial dark-mode aesthetic with OKLCH color tokens, sidebar + top-bar layout shell, and standalone page bases for login and setup wizard — all verified in a running browser with 93 tests passing.**

## What Happened

T01 defined the visual foundation. The `[data-theme="industrial"]` block was added to `input.css` using OKLCH color vars: near-black neutrals (`oklch(13% 0.005 260)` base-100), aerospace amber primary (`oklch(75% 0.19 75)`), green success, amber-orange warning, red-orange error, electric teal accent, and all required content/neutral variants. Radius tokens set to 0 for hard industrial edges. `base.html` was rewritten as a flex layout: a fixed-width `<aside w-56>` sidebar (sticky, full-height) with active-link highlighting via `request.url.path` comparisons, plus a right-column flex container holding a sticky header and `overflow-y-auto` content area. Block names `title`, `nav`, `content`, and `page_title` preserved. `_base_standalone.html` created as a 12-line minimal shell — same head + theme, no sidebar — for pages that must not render navigation chrome.

T02 wired login and wizard to the new standalone base and redesigned both with the industrial aesthetic. Login uses a full-viewport two-panel layout: left branding rail with AS9102/FAIR identity markers and "Clearance: Quality Team Only" labels, right panel with `DELTA.PRES` heading in `font-mono`, amber dot, monospace form labels with `tracking-widest uppercase`, and a full-width amber `Authenticate` button. The generic `card bg-base-100 shadow-xl` pattern was eliminated. The wizard layout uses a centered panel with a custom flex step indicator — numbered bordered nodes, `bg-primary` fill with checkmark SVG for completed steps, `border-primary` for the active step. DaisyUI's `ul.steps` component was not used (see Deviations). Both sub-templates (`step1_shop_name.html`, `step2_password.html`) required zero changes.

## Verification

- `npm run build:css` → exit 0, 212ms, no warnings; daisyUI 5.5.19 reported
- `grep -c '\[data-theme' static/dist/output.css` → 1 (theme block compiled)
- `curl http://localhost:8000/login` → HTTP 200, `data-theme="industrial"` on `<html>`, `font-mono` and `DELTA.PRES` heading present, no `card bg-base-100 shadow-xl`
- `curl http://localhost:8000/setup/step1` → HTTP 200, `data-theme="industrial"`, `border-primary` step indicator nodes present
- `curl http://localhost:8000/setup/step2` → HTTP 200, `data-theme="industrial"`, `bg-primary` filled completed step + `border-primary` active step
- `base.html` verified: `w-56`, `sticky top-0`, `overflow-y-auto`, `{% block page_title %}`, `{% block nav %}`, `{% block content %}` all present
- `uv run pytest tests/` → **93 passed, 2 xfailed, 0 failures** (identical to pre-slice baseline)

## Requirements Advanced

- R001 — Industrial dark-mode aesthetic system: OKLCH color tokens defined and compiled; dark base, amber primary, green success, amber-orange warning visible in output.css
- R002 — Sidebar + top-bar navigation layout: `base.html` delivers fixed sidebar with active-link highlighting and sticky top bar with `page_title` slot
- R003 — Login screen redesign: two-panel industrial layout, no white card, monospace aerospace typography
- R004 — Setup wizard redesign: dark-themed wizard with custom step progress indicator, no generic DaisyUI defaults visible

## Requirements Validated

- R003 — Login page verified in running browser: `data-theme="industrial"`, no `card bg-base-100 shadow-xl`, `font-mono` headings, amber `btn-primary` submit
- R004 — Wizard steps verified in running browser: custom step indicator with `border-primary`/`bg-primary`, dark panel, both sub-templates unmodified

## New Requirements Surfaced

- none

## Requirements Invalidated or Re-scoped

- none

## Deviations

**DaisyUI `ul.steps` not used for wizard progress indicator.** The task plan said "style the DaisyUI steps component." In practice, DaisyUI steps uses CSS pseudo-element counter backgrounds that don't respond cleanly to `--color-base-*` vars at `oklch(13%)` darkness — the result was visually ambiguous. Custom flex divs with explicit border/background classes give identical visual semantics with full theme control. Captured as D009.

**Google Fonts `@import` not added.** Plan mentioned optional font import. Existing system sans stack is sufficient for the industrial feel; the monospaced accents use `font-mono` (system mono). Deferred indefinitely — not worth a rebuild for this slice.

## Known Limitations

- Sidebar `Admin` link always visible regardless of user role (not gated). Role-gating deferred to downstream slices.
- Fixed-width sidebar causes layout issues at narrow viewports. Responsive layout is R014 (deferred).
- All templates except login and wizard still extend `base.html` and show sidebar — correct for authenticated pages. Unauthenticated pages that may still extend `base.html` will be addressed in their owning slice (S02–S05).

## Follow-ups

- S02–S05 must update remaining ~15 templates to extend the new `base.html` and render in the industrial aesthetic
- Role-gating on the Admin sidebar link can be added in S05 as part of the admin screen work
- Font import (if desired) can be added to `input.css` in any subsequent slice without rebuild risk

## Files Created/Modified

- `static/src/input.css` — expanded to ~45 lines with `[data-theme="industrial"]` OKLCH color vars
- `static/dist/output.css` — rebuilt with industrial theme compiled (84KB minified)
- `shop/templates/base.html` — rewritten: 44-line top-nav → ~115-line sidebar + top-bar layout; all blocks preserved
- `shop/templates/_base_standalone.html` — new: 12-line minimal dark-theme shell for login/wizard
- `shop/templates/auth/login.html` — rewritten: extends `_base_standalone.html`, two-panel industrial layout
- `shop/templates/setup/wizard_layout.html` — rewritten: extends `_base_standalone.html`, custom step indicator

## Forward Intelligence

### What the next slice should know
- All authenticated page templates (dashboard, runs list, new run, status, review) still extend the old `base.html` and will render correctly in the new layout — the block names and structure are preserved. S02/S03/S04 can update content blocks without touching the layout again.
- The `{% block page_title %}` slot in the top bar is available for per-page breadcrumbs or context — use it to show the current run name or section title.
- `_base_standalone.html` is the correct base for any page that must not show sidebar chrome (login, wizard, potentially a public-facing status page).
- `| default()` guards are on all context vars (`shop_name`, `unread_alert_count`, `user`) in `base.html` — pages that don't inject nav context won't crash.

### What's fragile
- DaisyUI semantic color classes (`btn-primary`, `badge-success`, `alert-warning`, etc.) depend on the theme vars defined in `input.css`. If those vars change names or values, all downstream badges/buttons shift. The current var set is complete and stable.
- The `overflow-y-auto` scroll container is on `<main>`, not `<body>`. Any sticky/fixed elements (footers, banners) in content pages must be positioned relative to the main container, not the viewport — use `sticky bottom-0` inside the content block.

### Authoritative diagnostics
- `grep -c '\[data-theme' static/dist/output.css` → must be 1; if 0, the theme block was dropped from input.css or the build target changed
- `curl -s http://localhost:8000/login | grep data-theme` → fastest runtime check that the standalone shell is active
- Test suite (`uv run pytest tests/`) → 93 passed, 2 xfailed is the baseline; any regression here is a template/routing break

### What assumptions changed
- Assumed DaisyUI v5 theme vars would need `@plugin "daisyui" { themes: [...] }` syntax — actually the cleanest approach is a raw `[data-theme="industrial"]{...}` CSS block after the @plugin directive. The plugin handles semantic class generation; the raw block handles the custom color values.
