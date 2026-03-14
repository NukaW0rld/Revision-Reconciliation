# S01: Design Foundation & Layout Shell — Research

**Date:** 2026-03-14

## Summary

S01 owns R001 (industrial dark-mode aesthetic), R002 (sidebar + top-bar layout), R003 (login redesign), and R004 (setup wizard redesign). Everything downstream depends on `base.html` and `output.css` being right first.

The codebase is clean and well-structured. The existing `base.html` is a minimal DaisyUI v4-style top-nav layout (44 lines). Templates extend it uniformly via `{% extends "base.html" %}` and `{% block content %}`. The CSS build is Tailwind v4 + DaisyUI v5, using just two lines in `input.css`. Build is clean and fast (~200ms).

The DaisyUI v5 custom theme mechanism is different from v4: custom themes are defined as raw CSS `[data-theme="industrial"]{--color-base-100:...}` blocks in `input.css`, not registered via plugin options or config. This is verified — it compiles and the theme name appears in output. The `@plugin "daisyui"` with options only handles built-in themes from its internal `themesObject`. The approach: define `[data-theme="industrial"]` directly in `input.css`, then set `data-theme="industrial"` on `<html>` in `base.html`. No theme-controller toggling needed (we're committing to dark only per D001).

The sidebar layout requires a structural change to `base.html`. Current structure is `navbar → container mx-auto`. New structure will be `flex h-screen` with a fixed-width sidebar, a right-side column containing a top bar and scrollable content area. Login (`auth/login.html`) and setup wizard (`setup/wizard_layout.html`) already override `{% block nav %}` to suppress it — they'll stay standalone, no sidebar needed.

Context variables available for the sidebar and top bar from `_get_nav_context()`: `shop_name`, `unread_alert_count`. The `user` object is available everywhere (email, role). Sidebar nav items: Dashboard (`/dashboard`), Runs (`/runs`), New Run (`/runs/new`), Admin (`/admin/users`, `/admin/settings` — role-gated). Active link highlighting can be done via Jinja2 `request.url.path` comparison.

The HTMX swap ID `#item-list` is in `queue.html` (not a partial), so the sidebar layout change doesn't affect it. The sticky `#signoff-footer` in `_signoff_footer.html` will need its `sticky bottom-0` rethought in a sidebar layout where the content area has its own scroll — `sticky bottom-0` within a scrollable flex column works fine as-is.

## Recommendation

**Define a single custom dark theme named `industrial` via raw CSS in `input.css`, set `data-theme="industrial"` on `<html>` in `base.html`, and build a sidebar layout using a CSS flexbox shell with a fixed sidebar.**

Theme approach: Add `[data-theme="industrial"]{...}` to `input.css` with carefully chosen OKLCH values:
- `--color-base-100/200/300`: very dark near-black backgrounds (no blues, neutral-cool)
- `--color-primary`: amber (`oklch(~75% 0.19 75)`) — matches aerospace tooling amber
- `--color-success`: green (`oklch(~68% 0.18 145)`)
- `--color-warning`: amber-orange (same hue family as primary)
- `--color-error`: red-orange
- `--radius-selector/field/box`: 0 or `2px` — hard-edged, industrial, not rounded

Layout approach: `<html data-theme="industrial">` → `<body class="min-h-screen flex">` → left: `<aside class="w-56 h-screen sticky top-0 flex-shrink-0 ...">` with nav links → right: `<div class="flex flex-col flex-1 min-h-screen">` → top bar (fixed, `sticky top-0`) + scrollable `<main>`.

Typography: Use `font-mono` for characteristic numbers and technical values (already used in templates). The base font can use a Tailwind sans stack — or import a narrow industrial sans (IBM Plex Mono or similar). Keep it simple: a `@import url(...)` in `input.css` pointing to Google Fonts, then override `--font-sans` or use a custom CSS variable.

Login and setup wizard: standalone dark-mode pages, no sidebar, full-screen centered panel design. These extend `base.html` with `{% block nav %}{% endblock %}` suppression — they can either continue extending `base.html` (which will now include the sidebar shell) or the sidebar HTML can be conditionally omitted via a `{% block sidebar %}`. **Cleaner approach**: `base.html` provides the full authenticated shell (sidebar + topbar); login and wizard get their own layout templates (`_standalone_layout.html` concept — but since they already suppress nav, the easiest path is to keep `base.html` for auth pages and have login/wizard render their own `<html>…</html>` directly without `extends`, or use a separate `base_standalone.html`). Decision: login and wizard should NOT extend the sidebar base — they'll be self-contained HTML files with the same dark theme but no sidebar chrome.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Dark theme variables | DaisyUI v5 CSS custom properties (`[data-theme="industrial"]{...}`) | Integrates with all DaisyUI component classes; no class name changes needed in templates |
| Sidebar active-link highlighting | Jinja2 `request.url.path` | No JS needed; FastAPI `Request` already injected via middleware/template context |
| Font loading | Google Fonts CSS `@import` in `input.css` | Zero JS, zero build steps; Tailwind passes through `@import` |
| Sticky signoff footer in scrollable content | CSS `sticky bottom-0` inside a flex column that has `overflow-y-auto` | Works in all modern browsers; no JS position management |

## Existing Code and Patterns

- `shop/templates/base.html` — current 44-line top-nav layout; completely replaced in S01; the `{% block nav %}`, `{% block content %}`, and `{% block title %}` block names are used by every template — preserve these block names
- `shop/templates/auth/login.html` — overrides `{% block nav %}{% endblock %}` to suppress navbar; after S01 it becomes self-contained (no `extends base.html`)
- `shop/templates/setup/wizard_layout.html` — same suppression pattern; also becomes self-contained after S01
- `static/src/input.css` — currently two lines; S01 adds the theme definition and any utility overrides here
- `shop/routers/runs.py:_get_nav_context()` — returns `{unread_alert_count, shop_name}` — these must remain available in new top bar
- `shop/app.py:_status_badge_class()` — Jinja2 filter; maps run status → DaisyUI badge variant name (success/error/warning/info/ghost) — these variant names must survive the theme rename
- `shop/templates/review/_signoff_footer.html` — root element is `<div id="signoff-footer" class="sticky bottom-0 ...">` — the id and sticky positioning must be preserved; the `sticky bottom-0` behavior works fine inside a flex-column scrollable content pane

## Constraints

- DaisyUI badge variant names used in templates (`badge-success`, `badge-error`, `badge-warning`, `badge-info`, `badge-ghost`) must have corresponding color vars in the custom theme — define `--color-success/error/warning/info` in `[data-theme="industrial"]`
- Block names in `base.html` (`title`, `nav`, `content`) are used by all ~20 templates — changing them is a full-codebase edit; preserve them
- `{% block nav %}` is overridden to suppress by login and wizard — after switching them to self-contained templates, this is no longer relevant
- `unread_alert_count` and `shop_name` must appear in the top bar context — these are injected by `_get_nav_context()` in runs/review routers and by the auth router; no backend changes needed, just reference them in the template
- `@plugin "daisyui"` in `input.css` does **not** accept custom theme names — custom theme vars must be raw CSS outside the plugin directive
- The CSS build command is `npm run build:css` from project root; `input.css` is at `static/src/input.css`, output at `static/dist/output.css` — paths must not change
- Tailwind v4 uses `@import "tailwindcss"` not `@tailwind base/components/utilities` — do not use the v3 directives

## Common Pitfalls

- **Registering custom theme via `@plugin "daisyui"` options** — DaisyUI v5's plugin only applies built-in themes from its internal `themesObject`. Custom themes must be raw CSS `[data-theme="industrial"]{...}` blocks in `input.css`. Verified by test build.
- **Missing color vars for DaisyUI component variants** — if `--color-success`, `--color-error`, `--color-warning`, `--color-info` are not defined in the custom theme, badge and alert component classes will fall back to defaults or render wrong. All status-semantic vars must be explicitly set.
- **Sidebar + sticky footer interaction** — `sticky bottom-0` only works within a positioned ancestor with overflow. In the new layout, the `<main>` content area must have `overflow-y-auto` (not the outer `<body>`) for the signoff footer to stick to the bottom of the scroll container. This is intentional and correct.
- **Login/wizard still extending base.html** — if login and wizard continue to use `{% extends "base.html" %}` after base.html gains the sidebar shell, they'll render the sidebar chrome even though they suppress `{% block nav %}`. The sidebar is added outside `{% block nav %}`, so suppressing nav won't hide it. **Must decouple**: login and wizard become self-contained HTML or extend a minimal `_base_standalone.html`.
- **`request` not available in template context** — FastAPI's Jinja2 doesn't auto-inject `request` unless using `TemplateResponse(request=request, ...)` which the routers do. Check each router injects `request` for active-nav highlighting to work. All existing router calls use `templates.TemplateResponse` with explicit context dicts — `request` is passed as a key in those dicts, so `request.url.path` is available in templates.
- **DaisyUI radius tokens** — `--radius-selector`, `--radius-field`, `--radius-box` are DaisyUI v5 tokens (not standard Tailwind). Setting them to `0` gives full industrial sharp edges. Don't set `border-radius` via Tailwind `rounded-*` classes on components — override via theme tokens instead.

## Open Risks

- **Font import latency** — Google Fonts `@import` in CSS adds a blocking request on first load; use `display=swap` and consider self-hosting in a later pass if performance matters
- **`request` context availability** — need to verify that every router rendering an authenticated template passes `request` in the context dict; active-nav link highlighting depends on this. If any router omits it, active highlighting silently breaks for that page.
- **DaisyUI `steps-vertical` in SSE checklist** — the pipeline stage checklist uses `ul.steps.steps-vertical`; this DaisyUI component may need style tuning under the industrial theme (step colors, connector lines) — minor, but visually important on the status page (S03 concern, not S01)
- **Sidebar width + content overflow on small screens** — no responsive requirement exists (R014 deferred), but a fixed-width sidebar at narrow viewports will cause layout breakage; acceptable for now per scope, but worth a note

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| Tailwind CSS v4 | — | none found (well-documented, inline knowledge sufficient) |
| DaisyUI v5 | — | none found |
| HTMX | — | none found |

## Sources

- DaisyUI v5 theme CSS variables: read directly from `node_modules/daisyui/themes.css` — all built-in themes use `[data-theme=X]{--color-*: oklch(...); --radius-*: ...; --border: ...}` format
- DaisyUI v5 plugin options handler: read from `node_modules/daisyui/functions/pluginOptionsHandler.js` — custom themes not in `themesObject` are silently skipped; raw CSS approach confirmed via test build
- Template context variables: read from `shop/routers/runs.py`, `shop/routers/review.py`, `shop/routers/auth.py`
- HTMX swap IDs: verified from `review/_item_card.html`, `review/_signoff_footer.html`, `review/_progress_bar.html`, `runs/status.html`, `runs/new.html`
