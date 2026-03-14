# S02: Dashboard, Runs List & New Run Form — Research

**Date:** 2026-03-14

## Summary

S02 owns six templates and three HTMX partials. The scope is well-bounded: `dashboard.html`, `runs/list.html`, `runs/new.html` are full pages extending `base.html`; `runs/_xlsx_mapping.html`, `runs/_page_selector.html`, `runs/_pdf_error.html` are HTMX-swapped fragments. `runs/_alert_banner.html` sits on the boundary — it's included server-side in `dashboard.html` but also returned standalone via HTMX (dismiss POST). All six pages already extend `base.html` and inherit the S01 layout shell — no structural layout work needed. The work is content redesign: replace generic DaisyUI card/form patterns with industrial aesthetic, use themed color tokens throughout, and fix one known dark-mode incompatibility.

The one real problem is `bg-amber-100 border-amber-400` in `_xlsx_mapping.html` — hardcoded Tailwind color-scale classes that render a white background on dark themes. These must be replaced with theme-aware equivalents. The correct replacement is `bg-warning/15 border-warning` to use the industrial amber warning token at reduced opacity. The same issue appears in `setup/step4_mapping_partial.html` but that's S05 scope — note it but don't touch it here.

The new run form uses `card bg-base-100` wrappers which are technically valid but visually generic — they should become `border border-base-300 bg-base-200` bordered sections consistent with the sidebar/topbar aesthetic. All HTMX swap IDs (`#revA-section`, `#revB-section`, `#xlsx-mapping-section`) are in `new.html` and must be preserved exactly. The dashboard alert banner (`_alert_banner.html`) uses `hx-target="#alert-banner-{{ alert.id }}"` with `hx-swap="outerHTML"` — its root element ID is the swap target; preserve it.

## Recommendation

Work template by template in this order: `dashboard.html` (simplest, establishes patterns), `runs/list.html` (table redesign), `runs/new.html` (form + file upload), then the three partials. Establish a consistent section card pattern early (`border border-base-300 bg-base-200` or `bg-base-200/50 border border-base-300`) and reuse it across all three full pages. Run `npm run build:css` once at the end of the task and `uv run pytest tests/` to confirm no regressions.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Status badge colors (queued/running/completed/failed/warning/reviewing/signing_off/signed_off) | `badge badge-{{ run.status \| status_badge_class }}` — already wired via Jinja2 filter in `app.py`; maps to `badge-success/error/warning/info/ghost` | Filter is registered and tested; theme vars make badge colors correct automatically |
| Table layout for runs list | DaisyUI `table` class (no `table-zebra` in dark mode — use `table` only) | `table-zebra` applies `bg-base-200` striping which is fine with the industrial theme; keep or replace with explicit hover rows |
| File input styling | `file-input file-input-bordered` | DaisyUI component; works with themed vars |
| Select dropdown styling | `select select-bordered select-sm` | DaisyUI component; theme-aware |

## Existing Code and Patterns

- `shop/templates/dashboard.html` — extends `base.html`; renders recent runs as `card bg-base-100` grid cards with `badge-{{ status }}`, admin cards, alert banner includes, CTA button. Replace cards with bordered `div` sections using `bg-base-200`; keep `{% block page_title %}Dashboard{% endblock %}` for top bar.
- `shop/templates/runs/list.html` — extends `base.html`; has filter form (`input input-bordered input-sm`) and `table table-zebra` for runs. Redesign filter area with inline field labels; replace table header with `font-mono uppercase text-xs` header row; keep `badge-{{ status }}` cells.
- `shop/templates/runs/new.html` — extends `base.html`; has four `card bg-base-100 shadow` sections wrapping file inputs and text fields; HTMX file validation on change. Replace cards with `border border-base-300` sections; keep all HTMX attributes verbatim; keep `id="revA-section"`, `id="revB-section"`, `id="xlsx-mapping-section"`.
- `shop/templates/runs/_xlsx_mapping.html` — HTMX partial swapped into `#xlsx-mapping-section`; contains column-mapping table with `select select-bordered`; uses **`bg-amber-100 border-amber-400`** hardcoded classes for undetected columns → must replace with `bg-warning/15 border-warning`.
- `shop/templates/runs/_page_selector.html` — HTMX partial swapped into `#revA-section` or `#revB-section`; uses `card bg-warning bg-opacity-20 border border-warning` — already theme-aware; minimal touch needed (remove `card`, use `div`).
- `shop/templates/runs/_pdf_error.html` — HTMX partial swapped into same targets; uses `alert alert-error` — theme-aware, just works. Minor: remove hardcoded SVG stroke path duplication if desired.
- `shop/templates/runs/_alert_banner.html` — included server-side in dashboard AND returned standalone by dismiss POST; root element is `<div id="alert-banner-{{ alert.id }}">` — this ID is the HTMX outerHTML swap target; **must not change**. Replace `alert alert-error` with themed `border-l-4 border-error bg-base-200` style consistent with industrial aesthetic, or keep DaisyUI alert (it's theme-aware).
- `shop/routers/auth.py` — `dashboard()` injects: `user`, `recent_runs`, `unread_alerts`, `unread_alert_count`, `shop_name`. The `unread_alerts` list (up to 5) is rendered inline via `{% include "runs/_alert_banner.html" %}`.
- `shop/routers/runs.py` — `list_runs()` injects: `user`, `runs`, `part_number`, `date_from`, `**nav` (unread_alert_count, shop_name). `new_run_form()` only injects `user`. Error re-render of `new.html` adds `error` key.
- `shop/app.py:_status_badge_class` — Jinja2 filter mapping run status strings to DaisyUI badge modifier classes. Stable; don't change.
- `static/src/input.css` — Theme vars: amber primary (`oklch(75% 0.19 75)`), green success (`oklch(68% 0.18 145)`), warning amber (`oklch(72% 0.17 65)`), error red-orange (`oklch(62% 0.22 25)`), info teal (`oklch(65% 0.15 230)`). All DaisyUI semantic classes resolve against these.
- `shop/templates/base.html` — S01 pattern: `{% block page_title %}` in topbar, `{% block content %}` in `<main class="flex-1 overflow-y-auto p-6">`. Sticky footers must use `sticky bottom-0` inside the content block.

## Constraints

- **HTMX swap IDs are inviolable**: `#revA-section`, `#revB-section`, `#xlsx-mapping-section` in `new.html`; `#alert-banner-{{ alert.id }}` in `_alert_banner.html`. Changing these IDs breaks HTMX without router changes.
- **Template context variables are fixed**: routers pass specific keys; new keys can be added via default() guards but existing keys cannot be removed or renamed.
- **`npm run build:css` must stay clean** after all changes — no unknown utility classes.
- **No Python changes**: pure template + CSS changes only (R015/R016 out of scope).
- **`bg-amber-100 border-amber-400` must be replaced** — these are hardcoded Tailwind color-scale classes that render a white/light background incompatible with the dark theme. Use `bg-warning/15 border-warning` instead.
- `table-zebra` alternates rows with `bg-base-200` — acceptable on dark theme; the industrial look may prefer no zebra striping and instead `hover:bg-base-300` row hover. Either works.
- The `version` variable referenced in `base.html` (`v{{ version | default("—") }}`) is never passed by any router — the `| default("—")` guard keeps it safe. Don't add `version` to router context.

## Common Pitfalls

- **Hardcoded light colors in partials** — `bg-amber-100 border-amber-400` in `_xlsx_mapping.html` will render light on dark background. Use `bg-warning/15 border-warning` (Tailwind opacity modifier syntax with CSS custom properties works in TW v4).
- **Removing HTMX target IDs** — the three `id=` divs in `new.html` and the root `id=` in `_alert_banner.html` are HTMX swap targets. Accidentally moving them inside a redesigned wrapper div or changing their IDs silently breaks validation HTMX without any error.
- **card vs bordered section** — replacing `card bg-base-100 shadow` with `card bg-base-200` causes DaisyUI card to render a slightly different background tone; using raw `div` with `border border-base-300 bg-base-200 p-4` is more predictable and matches the industrial aesthetic (no card shadow, hard edges via `--radius-box: 0`).
- **Form submit button inside card** — `new.html` has the submit button outside all cards at the bottom of the form; ensure it stays outside in the redesign to avoid it being visually buried.
- **Alert banner `alert alert-error`** — on dark theme this is `bg-error text-error-content`. The error oklch is `62% 0.22 25` (red-orange) which is legible. However a subtle `border-l-4 border-error` pattern on `bg-base-200` may read better for an alert that appears in the content pane. Either is valid — pick one and be consistent.
- **`bg-opacity-20` deprecated** — `_page_selector.html` uses `bg-warning bg-opacity-20`; in Tailwind v4 prefer `bg-warning/20` instead.

## Open Risks

- **DaisyUI `table` in dark mode**: `table-zebra` on dark base should work but hasn't been tested in a running browser with the current industrial theme. If the striping color is too close to base-100, switch to `table` without zebra and add `hover:bg-base-300` to `<tr>` elements.
- **File input appearance on dark theme**: `file-input file-input-bordered` styling varies across browsers; the DaisyUI component applies theme vars but the OS file dialog button may still appear light. This is acceptable (browser limitation) but worth noting in verification.
- **HTMX partial fragments rendered without page context**: `_xlsx_mapping.html`, `_page_selector.html`, `_pdf_error.html` are returned as raw HTML fragments by FastAPI — they don't get `<html>/<body>` wrappers or the `output.css` link. The CSS is already loaded on the parent page, so all DaisyUI classes just work. No risk here, just be aware.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| Tailwind CSS v4 + DaisyUI v5 | (installed via S01 research) | none found / already resolved in S01 |
| HTMX | (no dedicated skill) | none found |

## Sources

- Context from `.gsd/milestones/M003/slices/S01/S01-SUMMARY.md` — S01 patterns, theme vars, what's already working
- Template source files read directly: `dashboard.html`, `runs/list.html`, `runs/new.html`, `_xlsx_mapping.html`, `_page_selector.html`, `_pdf_error.html`, `_alert_banner.html`
- Router source: `shop/routers/runs.py`, `shop/routers/auth.py` — context variables injected into each template
- `shop/app.py` — `_status_badge_class` filter definition
- `shop/models.py` — Run status enum values and field names
- `static/src/input.css` — active OKLCH theme vars
