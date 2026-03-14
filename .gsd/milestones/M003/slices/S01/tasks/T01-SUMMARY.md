---
id: T01
parent: S01
milestone: M003
provides:
  - industrial DaisyUI v5 dark theme compiled in output.css
  - sidebar + top-bar layout shell in base.html
  - standalone dark-theme base (_base_standalone.html)
key_files:
  - static/src/input.css
  - static/dist/output.css
  - shop/templates/base.html
  - shop/templates/_base_standalone.html
key_decisions:
  - D007 already captured: login/wizard must extend _base_standalone.html, not base.html (T02 work)
patterns_established:
  - DaisyUI v5 custom themes defined as raw `[data-theme="name"]{...}` CSS blocks in input.css, outside @plugin directive
  - Sidebar active-link highlighting via Jinja2 `request.url.path` + `path.startswith()` comparisons
  - All optional template context vars guarded with `| default()` to prevent UndefinedError on pages without nav context
  - `<main class="flex-1 overflow-y-auto p-6">` as scroll container (not body) so sticky footers work inside content pane
observability_surfaces:
  - CSS build: `npm run build:css` stderr/stdout; exit code 0 = success
  - Theme in output: `grep -c '\[data-theme' static/dist/output.css` → 1
  - Runtime: `data-theme="industrial"` on `<html>` element inspectable in DevTools
  - Template errors: 500 responses with Jinja2 TemplateNotFound/UndefinedError in FastAPI error body
duration: 30m
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T01: Define industrial dark theme and build sidebar + top-bar layout shell

**Defined OKLCH-based `industrial` DaisyUI v5 theme and rebuilt `base.html` as a sidebar + top-bar layout shell; `npm run build:css` exits 0, app starts clean, all 93 tests pass.**

## What Happened

Added `[data-theme="industrial"]{...}` block to `static/src/input.css` with full set of OKLCH color vars: near-black neutrals (base-100/200/300), aerospace amber primary (`oklch(75% 0.19 75)`), green success, amber-orange warning, red-orange error, electric teal accent, muted slate secondary, and all content/neutral variants. Set radius tokens to 0 for hard industrial edges. No font import added (sharp edges + existing font stack is sufficient; revisit in later slice if needed).

Rewrote `base.html` as a `flex body` layout: fixed-width `<aside w-56>` sidebar (sticky, full height) with amber-accented active-link highlighting via `request.url.path` comparisons, plus a right-column flex container holding a sticky header and `overflow-y-auto` main content area. Preserved all block names (`title`, `nav`, `content`). Added `{% block page_title %}` in top bar for per-page context. All optional context vars guarded with `| default()`.

Created `_base_standalone.html` as a minimal 12-line shell — same head/theme, no sidebar chrome. This is what T02 will wire login and wizard pages to extend.

## Verification

- `npm run build:css` → exit 0, 248ms, no warnings
- `grep -c '\[data-theme' static/dist/output.css` → 1 (theme block present in minified output)
- `curl http://localhost:8000/login` → HTTP 200, response contains `data-theme="industrial"` and `<aside class="w-56 h-screen sticky top-0..."`
- `curl http://localhost:8000/setup/step2` → HTTP 200, `data-theme="industrial"` confirmed in response
- `uv run pytest tests/` → 93 passed, 2 xfailed, 0 failures

## Diagnostics

- Theme presence: `grep -o 'industrial[^;]*' static/dist/output.css` shows first color var compiled
- Full var check: `grep -o '\-\-color-[a-z-]*:[^;]*' static/dist/output.css` lists all theme vars in minified output
- Active-nav: browser DevTools on any authenticated page — sidebar link with `border-l-2 border-primary` class = active
- Failure shape: if `request` is absent from template context, `{% set path = request.url.path if request is defined else "" %}` degrades to empty string silently (no active highlight, no crash)

## Deviations

- Plan mentioned optional Google Fonts `@import` — not added. Existing sans stack is fine; font variety can be added in a later pass without blocking the layout work.
- `{% block page_title %}` added inside top bar header (not in original plan) — provides a per-page breadcrumb slot without requiring downstream templates to inject a separate context var.

## Known Issues

- Login and setup wizard pages still extend `base.html` and therefore render the sidebar. This is intentional — T02 decouples them to `_base_standalone.html`.
- Sidebar `Admin` link always visible (not role-gated). Plan defers role-gating to downstream slices.
- Fixed-width sidebar causes layout breakage at narrow viewports (R014 deferred).

## Files Created/Modified

- `static/src/input.css` — expanded from 2 lines to ~45 lines with `[data-theme="industrial"]` OKLCH vars
- `static/dist/output.css` — rebuilt with industrial theme compiled in (84KB minified)
- `shop/templates/base.html` — rewritten from 44-line top-nav to ~115-line sidebar + top-bar layout
- `shop/templates/_base_standalone.html` — new 12-line minimal standalone dark-theme shell
- `.gsd/milestones/M003/slices/S01/tasks/T01-PLAN.md` — added `## Observability Impact` section (pre-flight fix)
