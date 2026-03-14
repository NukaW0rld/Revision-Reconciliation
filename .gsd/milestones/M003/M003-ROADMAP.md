# M003: UI/UX Redesign

**Vision:** Replace every screen of the Delta Preservation web app with a cohesive industrial dark-mode design — dark background, amber/green status colors, monospaced accents, sidebar navigation, and a full-screen focused review queue with keyboard shortcuts. The result should feel like aerospace tooling software, not a generic SaaS dashboard.

## Success Criteria

- Opening the app in a browser shows a dark, authoritative interface with no generic DaisyUI default aesthetics visible
- Every authenticated page has a persistent sidebar for navigation and a top bar for context
- The review queue is full-screen, shows one item at a time, and responds to A/O/J/K keyboard shortcuts
- `npm run build:css` completes without errors
- All HTMX partial swaps produce fragments that visually integrate with the new layout
- The full flow (login → new run → pipeline progress → review → sign-off) is navigable without visual inconsistency

## Key Risks / Unknowns

- DaisyUI v5 dark theme API — `@plugin "daisyui"` theme definition differs from v4; must verify custom dark theme variables compile correctly before building other screens on top
- HTMX OOB swaps for `_signoff_footer_oob.html` and `_item_card.html` — fragments must produce visually integrated HTML without a full reload; DOM IDs must be preserved
- SSE + JS stage checklist — inline JS in `status.html` references specific DOM IDs; must remain functional after redesign
- Keyboard nav + HTMX interaction — HTMX approve POST swaps the item card; keyboard event listeners must reattach or be designed to survive DOM mutation

## Proof Strategy

- DaisyUI v5 dark theme API → retire in S01 by building and loading the base layout with custom dark theme — verified by running the app and seeing the correct colors
- HTMX OOB swap compatibility → retire in S04 by exercising the review queue approve/override flow end-to-end in a running browser
- Keyboard nav + HTMX interaction → retire in S04 by processing multiple items with keyboard shortcuts only

## Verification Classes

- Contract verification: `npm run build:css` clean; all templates render without Jinja2 errors; HTMX swap IDs present in partials
- Integration verification: full login → run → review → sign-off flow in a running browser
- Operational verification: none (no service lifecycle changes)
- UAT / human verification: reviewer processes items via keyboard shortcuts in a running browser

## Milestone Definition of Done

This milestone is complete only when all are true:

- All ~20 templates redesigned with industrial dark-mode aesthetic (no generic DaisyUI defaults visible)
- Sidebar + top-bar layout operational on every authenticated page
- Review queue keyboard shortcuts functional (A=approve, O=override, J/K=navigate)
- `npm run build:css` produces clean output
- All HTMX OOB swap IDs preserved; swaps produce visually integrated fragments
- Full flow (login → new run → pipeline → review → sign-off) navigable in a running browser

## Requirement Coverage

- Covers: R001, R002, R003, R004, R005, R006, R007, R008, R009, R010, R011, R012, R013
- Partially covers: none
- Leaves for later: R014 (dark/light mode toggle — deferred)
- Orphan risks: none

## Slices

- [x] **S01: Design Foundation & Layout Shell** `risk:high` `depends:[]`
  > After this: app loads with dark industrial theme, sidebar nav, new login and setup wizard screens — full visual identity established and verified in a running browser

- [ ] **S02: Dashboard, Runs List & New Run Form** `risk:medium` `depends:[S01]`
  > After this: the complete run submission flow is navigable in the new design — dashboard, run list, new run form, and inline column mapping partial all render correctly

- [ ] **S03: Pipeline Status & Run Detail** `risk:medium` `depends:[S01]`
  > After this: the run status page renders in the new design with all status states (running, failed, warning, completed, signed-off); SSE stage checklist updates live in the browser

- [ ] **S04: Review Queue — Full-Screen Focused Mode** `risk:high` `depends:[S01]`
  > After this: a reviewer can open the review queue, navigate items with J/K, approve with A, open override panel with O, and sign off — all keyboard-driven, all in the new design

- [ ] **S05: Admin, Setup Wizard & Fragment Cleanup** `risk:low` `depends:[S01,S02,S03,S04]`
  > After this: every screen and partial in the app is on the new design — admin screens, setup wizard, all HTMX fragments; the app is visually consistent end-to-end

## Boundary Map

### S01 → S02, S03, S04, S05

Produces:
- `shop/templates/base.html` — redesigned base layout with sidebar + top bar; all authenticated pages extend this
- `static/src/input.css` — custom dark theme variables (CSS custom properties), DaisyUI v5 theme definition
- `static/dist/output.css` — compiled Tailwind output; all downstream slices depend on this being built
- `shop/templates/auth/login.html` — redesigned login screen (standalone, no sidebar)
- `shop/templates/setup/wizard_layout.html` — redesigned setup wizard shell (standalone, no sidebar)

Consumes:
- nothing (first slice)

### S02 → S05

Produces:
- `shop/templates/dashboard.html` — redesigned dashboard extending new base
- `shop/templates/runs/list.html` — redesigned run list
- `shop/templates/runs/new.html` — redesigned new run form
- `shop/templates/runs/_xlsx_mapping.html` — HTMX partial for column mapping (preserves `#xlsx-mapping-section` swap target)
- `shop/templates/runs/_page_selector.html` — HTMX partial for PDF page selection
- `shop/templates/runs/_pdf_error.html` — HTMX partial for PDF validation error

Consumes from S01:
- `base.html` — sidebar + top bar layout
- `input.css` / `output.css` — design tokens and utility classes

### S03 → S05

Produces:
- `shop/templates/runs/status.html` — redesigned run status page with SSE-driven stage checklist (preserves `#stage-checklist` DOM ID and SSE JS)
- `shop/templates/runs/_stage_checklist.html` — initial server-rendered checklist partial
- `shop/templates/runs/_alert_banner.html` — HTMX partial for run alert banners

Consumes from S01:
- `base.html`, `output.css`

### S04 → S05

Produces:
- `shop/templates/review/queue.html` — redesigned full-screen focused review queue
- `shop/templates/review/_item_card.html` — HTMX partial for single item (preserves `#review-item-{char_no}` swap target and OOB update structure)
- `shop/templates/review/_progress_bar.html` — progress counter partial (preserves `#progress-bar` swap target)
- `shop/templates/review/_signoff_footer.html` — sticky sign-off footer (preserves `#signoff-footer` swap target)
- `shop/templates/review/_signoff_footer_oob.html` — OOB version of sign-off footer
- `shop/templates/review/generating.html` — intermediate state page
- Keyboard shortcut JS — inline in `queue.html`; A/O/J/K handlers wired to HTMX or native form submit

Consumes from S01:
- `base.html`, `output.css`

### S05 (terminal)

Produces:
- `shop/templates/admin/users.html` — redesigned user management
- `shop/templates/admin/users_row.html` — HTMX partial for new user row
- `shop/templates/admin/settings.html` — redesigned settings form
- `shop/templates/setup/step1_shop_name.html` — redesigned wizard step
- `shop/templates/setup/step2_password.html` — redesigned wizard step
- `shop/templates/setup/step4_error.html` — redesigned error state
- `shop/templates/setup/step4_mapping_partial.html` — mapping partial in setup context
- Visual consistency audit — all templates verified to match aesthetic

Consumes from S01–S04:
- All redesigned templates as reference for aesthetic consistency
- `output.css` final build
