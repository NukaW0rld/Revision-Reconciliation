---
estimated_steps: 5
estimated_files: 3
---

# T01: Define industrial dark theme and build sidebar + top-bar layout shell

**Slice:** S01 — Design Foundation & Layout Shell
**Milestone:** M003

## Description

Define the custom DaisyUI v5 dark theme (`industrial`) and rebuild `base.html` as a sidebar + top-bar layout. This is the single highest-risk task in S01 — if the theme doesn't compile or the layout breaks template inheritance, every downstream slice is blocked. Also create `_base_standalone.html` for login/wizard pages that need the dark theme without sidebar chrome.

## Steps

1. Add `[data-theme="industrial"]{...}` block to `static/src/input.css` after the existing `@plugin "daisyui"` line. Define all required OKLCH color variables: `--color-base-100/200/300` (very dark neutrals), `--color-primary` (amber ~oklch(75% 0.19 75)), `--color-secondary`, `--color-accent`, `--color-success` (green), `--color-warning` (amber-orange), `--color-error` (red), `--color-info`, `--color-base-content` (light text). Set `--radius-selector`, `--radius-field`, `--radius-box` to 0 for sharp industrial edges. Set `--border` to a subtle value. Optionally add a Google Fonts `@import` for a distinctive monospace or industrial sans font.

2. Rewrite `shop/templates/base.html`: set `data-theme="industrial"` on `<html>`. Structure as `<body class="min-h-screen flex">` → `<aside>` (fixed-width sidebar, `w-56 h-screen sticky top-0`, dark bg, nav links with active-state highlighting via `{% if request.url.path == '/...' %}`) → `<div class="flex flex-col flex-1 min-h-screen">` → sticky top bar (shop name, user info, alert badge, logout) → `<main class="flex-1 overflow-y-auto p-6">{% block content %}{% endblock %}</main>`. Preserve `{% block title %}`, `{% block nav %}`, `{% block content %}` block names. Use `| default()` for `shop_name`, `unread_alert_count`, and `user` to handle routes that don't inject nav context.

3. Create `shop/templates/_base_standalone.html`: minimal HTML document with `data-theme="industrial"`, the same `<head>` (CSS + HTMX), but no sidebar or top bar. Provide `{% block title %}`, `{% block content %}` blocks. This is what login and wizard pages will extend.

4. Run `npm run build:css` and verify it exits cleanly. Check that `static/dist/output.css` contains the `[data-theme=industrial]` rules.

5. Start the app with `uv run python run_web.py`, navigate to the login page and dashboard in the browser. Verify: dark background visible, sidebar renders on authenticated pages, amber primary color applied to buttons.

## Must-Haves

- [ ] `[data-theme="industrial"]` defined in `input.css` with all DaisyUI semantic color vars
- [ ] `npm run build:css` exits 0
- [ ] `base.html` uses flex sidebar + top-bar layout with `data-theme="industrial"` on `<html>`
- [ ] Block names `title`, `nav`, `content` preserved in `base.html`
- [ ] Sidebar nav items: Dashboard, Runs, New Run, Admin (role-gated display deferred to downstream)
- [ ] Active-link highlighting via `request.url.path` comparison
- [ ] Top bar shows `shop_name`, user email, alert count badge, logout button
- [ ] `_base_standalone.html` exists with dark theme, no sidebar
- [ ] `| default()` guards on optional template vars

## Verification

- `npm run build:css` returns exit code 0
- `grep -c 'data-theme' static/dist/output.css` returns at least 1
- App loads at `http://localhost:8000/login` showing dark background
- After login, sidebar visible on left side of dashboard page

## Inputs

- `static/src/input.css` — current 2-line file (Tailwind import + DaisyUI plugin)
- `shop/templates/base.html` — current 44-line top-nav layout to be replaced
- S01-RESEARCH.md findings on DaisyUI v5 custom theme format (`[data-theme="X"]{...}` raw CSS)
- S01-RESEARCH.md findings on sidebar layout (`flex h-screen` pattern)

## Observability Impact

- **New signals:** CSS build errors surface immediately in `npm run build:css` stdout; a non-zero exit code means the theme definition is malformed. After build, `grep -c 'data-theme' static/dist/output.css` confirms the theme block compiled through.
- **Runtime inspection:** `data-theme="industrial"` attribute on `<html>` is visible in browser DevTools Element panel. Computed CSS custom properties (`--color-primary`, `--color-base-100`, etc.) are inspectable via DevTools `Computed` tab filtered to `--color-`. Jinja2 template errors surface as HTTP 500 responses with `TemplateNotFound` or `TemplateRenderingError` in FastAPI error responses.
- **Failure state:** If `base.html` blocks are renamed or removed, all child templates that `{% extends "base.html" %}` will throw `BlockNotFound` at render time — visible immediately on any page load. If `| default()` guards are absent, pages that omit `shop_name`/`user`/`unread_alert_count` from context will throw `UndefinedError` in Jinja2.
- **Diagnostic command:** `grep -n 'data-theme\|--color-primary\|--color-base-100' static/dist/output.css` shows whether the theme block compiled and which vars landed in the output.

## Expected Output

- `static/src/input.css` — expanded with industrial theme definition (~30-50 lines of CSS vars)
- `static/dist/output.css` — rebuilt with industrial theme compiled in
- `shop/templates/base.html` — rewritten as sidebar + top-bar layout (~60-80 lines)
- `shop/templates/_base_standalone.html` — new minimal standalone layout (~20-25 lines)
