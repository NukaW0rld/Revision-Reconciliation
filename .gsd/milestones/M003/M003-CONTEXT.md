# M003: UI/UX Redesign — Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

## Project Description

Complete redesign of all ~20 Jinja2 templates in the Delta Preservation web app. No backend changes. The aesthetic direction is "industrial precision" — dark-mode, monospaced accents, amber/green status colors — software that feels like it belongs in an aerospace manufacturing quality lab, not a generic SaaS dashboard.

## Why This Milestone

M001 and M002 delivered a fully functional tool. The UI is generic DaisyUI defaults with no design identity. The people using this are quality engineers in aerospace shops — the tool should communicate the seriousness of AS9102 FAIR compliance work, not look like a bootstrapped startup's MVP.

## User-Visible Outcome

### When this milestone is complete, the user can:
- Open the app and immediately see a dark, authoritative interface with sidebar navigation
- Submit a run, watch pipeline progress, and review characteristics — all on the new design
- Process the review queue with keyboard shortcuts (A=approve, O=override, J/K=navigate)
- Sign off and download the audit packet — everything consistent from login through export

### Entry point / environment
- Entry point: `http://localhost:8000` (uvicorn dev server) or Docker
- Environment: browser, local dev or Docker
- Live dependencies involved: SQLite database, Huey task queue, pipeline subprocess

## Completion Class

- Contract complete means: all templates render without error; HTMX swaps produce matching-aesthetic fragments
- Integration complete means: full flow from login → new run → pipeline progress → review → sign-off renders correctly in a running browser
- Operational complete means: `npm run build:css` produces clean output; no template rendering errors

## Final Integrated Acceptance

To call this milestone complete, we must prove:
- A reviewer can log in, submit a run, watch it process, open the review queue, approve items with keyboard shortcuts, and sign off — all on the new design
- Every partial template (HTMX-swapped fragments) visually matches the page it's injected into

## Risks and Unknowns

- Tailwind v4 + DaisyUI v5 CSS build: DaisyUI v5 has a different plugin API than v4 — must verify `@plugin "daisyui"` produces dark theme variables correctly
- HTMX OOB swaps: `_signoff_footer_oob.html` and `_item_card.html` OOB updates must produce HTML that visually integrates with the new page layout without a full reload
- SSE + JS checklist updates in `status.html`: the inline JS that updates the stage checklist from SSE events must still target the correct DOM IDs after the redesign
- Full-screen review queue keyboard nav: HTMX approve/override POSTs must not interfere with keyboard event listeners; need to ensure focus is managed correctly after swap

## Existing Codebase / Prior Art

- `shop/templates/base.html` — current base layout; will be completely replaced
- `shop/templates/review/_item_card.html` — most complex partial; HTMX OOB, approve/override forms
- `shop/templates/review/_signoff_footer_oob.html` — OOB swap target; must preserve `id="signoff-footer"` and `id="progress-bar"` for HTMX
- `shop/templates/runs/status.html` — inline JS with SSE listener; must preserve stage checklist DOM IDs
- `shop/routers/review.py` — injects `run_id`, `items`, `pending`, `approved`, `overridden`, `total`, `read_only`, `is_amendment` into review templates
- `shop/routers/runs.py::_get_nav_context()` — returns `unread_alert_count`, `shop_name` for nav
- `static/src/input.css` — single `@import "tailwindcss"` + `@plugin "daisyui"` — custom theme vars go here
- `static/dist/output.css` — compiled output; built via `npm run build:css`
- `static/js/htmx.min.js`, `static/js/htmx-sse.js` — JS assets, untouched

## Relevant Requirements

- R001 — Industrial dark-mode aesthetic: dark bg, amber/green status, monospaced accents
- R002 — Sidebar + top-bar layout: replaces single top nav on all authenticated pages
- R003/R004 — Login and setup wizard: no sidebar (pre-auth), but same dark aesthetic
- R009/R010 — Full-screen review queue with keyboard shortcuts: most design-intensive screen
- R013 — All HTMX partials updated: fragments must match the page aesthetic

## Scope

### In Scope
- All files in `shop/templates/` — base layout, auth, setup, runs, review, admin, exports (web views only)
- `static/src/input.css` — custom CSS variables, dark theme definition, any utility classes
- Inline JavaScript in templates (keyboard nav, SSE handlers) — behavior may change if better UX exists

### Out of Scope / Non-Goals
- `shop/templates/exports/audit_packet.html` and `exports/work_order.html` — these are PDF/print templates rendered by WeasyPrint; print aesthetics are separate from screen UI
- Any Python router, service, model, or task file
- `static/js/htmx.min.js`, `static/js/htmx-sse.js` — JS libraries not modified
- New routes, endpoints, or data

## Technical Constraints

- Stack: Tailwind CSS v4 + DaisyUI v5 + HTMX (no React, no build-time JS bundling)
- CSS build: `npm run build:css` — must stay clean after all changes
- HTMX swap IDs that must be preserved: `#signoff-footer`, `#progress-bar`, `#item-list`, `#stage-checklist`, `#review-item-{char_no}`, `#users-table tbody`, `#revA-section`, `#revB-section`, `#xlsx-mapping-section`
- Template context variables passed by routers must not change — only HTML/CSS changes
- DaisyUI v5 theme: define custom dark theme in `input.css` using `@plugin "daisyui"` theme API

## Integration Points

- HTMX — partial swaps and OOB updates must produce HTML compatible with the new layout
- SSE (`/runs/{id}/sse`) — `stage_update` and `close` events drive JS checklist updates; DOM IDs must be stable
- WeasyPrint — exports templates are NOT redesigned (print-specific)

## Open Questions

- None — design direction confirmed: industrial dark-mode, sidebar + top bar, full-screen review queue with keyboard nav
