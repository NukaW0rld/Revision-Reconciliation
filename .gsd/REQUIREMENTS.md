# Requirements

This file is the explicit capability and coverage contract for the project.

## Active

### R001 — Industrial dark-mode aesthetic system
- Class: differentiator
- Status: active
- Description: All screens use a dark background, tight grid, monospaced accents, amber/green status colors. The visual language communicates aerospace tooling software — authoritative, precise, serious.
- Why it matters: The current UI is generic DaisyUI defaults with no design identity. The redesign must feel like it belongs in an aerospace manufacturing shop, not a generic SaaS.
- Source: user
- Primary owning slice: M003/S01
- Supporting slices: M003/S02, M003/S03, M003/S04, M003/S05
- Validation: unmapped
- Notes: Amber/green for status colors; monospaced for characteristic numbers and technical values; dark base throughout

### R002 — Sidebar + top-bar navigation layout
- Class: primary-user-loop
- Status: active
- Description: Authenticated pages use a persistent left sidebar for section navigation and a top bar for run context, user info, and alerts. Replaces the current single-nav top bar.
- Why it matters: The current nav is too flat for a multi-section tool. A sidebar makes section switching fast and keeps the user oriented.
- Source: user
- Primary owning slice: M003/S01
- Supporting slices: M003/S02, M003/S03, M003/S04, M003/S05
- Validation: unmapped

### R003 — Login screen redesign
- Class: launchability
- Status: active
- Description: Login page uses the new aesthetic — dark background, distinctive typography, no generic card-on-white-background pattern.
- Why it matters: First impression of the tool.
- Source: user
- Primary owning slice: M003/S01
- Supporting slices: none
- Validation: verified — M003/S01; dark two-panel layout with monospace typography; no white card; confirmed in running browser via curl + DevTools

### R004 — Setup wizard redesign
- Class: launchability
- Status: active
- Description: The 2-step setup wizard (shop name, admin password) uses the new aesthetic and layout.
- Why it matters: First-run experience.
- Source: user
- Primary owning slice: M003/S01
- Supporting slices: none
- Validation: verified — M003/S01; dark panel with custom step indicator; both steps confirmed in running browser

### R005 — Dashboard redesign
- Class: primary-user-loop
- Status: active
- Description: Dashboard shows recent runs, alert banners, and admin cards in the new layout. Run status badges use the amber/green industrial palette.
- Why it matters: Primary landing page after login.
- Source: user
- Primary owning slice: M003/S02
- Supporting slices: none
- Validation: verified — M003/S02; bordered dark sections, font-mono headings, no card/shadow artifacts; CSS builds clean, 93 tests pass

### R006 — New run submission form redesign
- Class: primary-user-loop
- Status: active
- Description: The multi-card file upload form uses the new aesthetic. Inline column mapping partial (HTMX-swapped) is also redesigned.
- Why it matters: Primary data entry flow.
- Source: user
- Primary owning slice: M003/S02
- Supporting slices: none
- Validation: verified — M003/S02; bordered form sections replace card wrappers; _xlsx_mapping uses bg-warning/15 tokens; all HTMX swap IDs intact; CSS builds clean, tests pass

### R007 — Pipeline status / run detail page redesign
- Class: primary-user-loop
- Status: active
- Description: The run status page with live SSE stage checklist, run metadata, and all status states (running/failed/warning/completed/signed-off) is redesigned. SSE-driven JS updates must still work.
- Why it matters: Most dynamic page in the app — engineers watch this while the pipeline runs.
- Source: user
- Primary owning slice: M003/S03
- Supporting slices: none
- Validation: verified — M003/S03; all card/shadow artifacts removed, all alert alert-* banners replaced with border-l-4 pattern in both Jinja2 and JS, DOM IDs preserved, CSS builds clean, 93 tests pass

### R008 — Run list page redesign
- Class: primary-user-loop
- Status: active
- Description: The runs list with filter form and table is redesigned.
- Why it matters: Primary navigation to past runs.
- Source: user
- Primary owning slice: M003/S03
- Supporting slices: none
- Validation: verified — M003/S02; mono table headers, hover rows, no light-mode artifacts; CSS builds clean, tests pass

### R009 — Full-screen focused review queue
- Class: differentiator
- Status: active
- Description: The review queue shows one characteristic at a time in a full-viewport layout. Rev A and Rev B snippets are large and prominent. Decision buttons are visually dominant. Reviewer can move through the queue without scrolling.
- Why it matters: Reviewers spend the majority of their time here processing 50-100 items. The current card-list layout requires too much scrolling and context-switching.
- Source: user
- Primary owning slice: M003/S04
- Supporting slices: none
- Validation: verified — M003/S04; full-screen focused layout, one item visible at a time, large snippet images; confirmed in running browser

### R010 — Keyboard shortcuts in review queue
- Class: differentiator
- Status: active
- Description: A = approve current item, O = open override panel, J/K (or arrow keys) = navigate to next/previous item. Shortcuts are shown as a visible legend on the page.
- Why it matters: Processing 100 items with mouse alone is slow. Keyboard nav makes the review flow fast enough to use daily without frustration.
- Source: user
- Primary owning slice: M003/S04
- Supporting slices: none
- Validation: verified — M003/S04; A=approve, O=override (auto-focuses textarea), J/K=navigate; shortcuts suppressed in inputs and dialogs; keyboard legend visible on screen; confirmed in running browser
- Notes: Override panel opened by O should auto-focus the note textarea

### R011 — Sign-off footer and progress bar redesign
- Class: primary-user-loop
- Status: active
- Description: The sticky sign-off footer and inline progress counters use the new design. The "Sign Off" CTA is visually distinct when all items are resolved.
- Why it matters: End of the review flow — must be clear and trustworthy.
- Source: user
- Primary owning slice: M003/S04
- Supporting slices: none
- Validation: verified — M003/S04; OOB swaps update progress bar and sign-off footer live after each approve/override; Sign Off CTA enables when all items resolved; modal fires and redirects; confirmed in running browser

### R012 — Admin screens redesign
- Class: admin/support
- Status: active
- Description: User management table and settings form use the new aesthetic.
- Why it matters: Admin completeness.
- Source: user
- Primary owning slice: M003/S05
- Supporting slices: none
- Validation: unmapped

### R013 — All HTMX partial/fragment templates updated
- Class: quality-attribute
- Status: active
- Description: Every partial template (_item_card.html, _progress_bar.html, _signoff_footer.html, _signoff_footer_oob.html, _stage_checklist.html, _page_selector.html, _alert_banner.html, _xlsx_mapping.html, _pdf_error.html, users_row.html) is updated to match the new design. HTMX swap behavior is preserved.
- Why it matters: Partials are injected into the redesigned pages — if they don't match the aesthetic, the result is visually broken.
- Source: inferred
- Primary owning slice: M003/S05
- Supporting slices: M003/S04
- Validation: unmapped

## Deferred

### R014 — Dark/light mode toggle
- Class: quality-attribute
- Status: deferred
- Description: User-selectable theme switching between dark and light modes.
- Why it matters: Some users prefer light mode.
- Source: inferred
- Primary owning slice: none
- Supporting slices: none
- Validation: unmapped
- Notes: Deferred — the redesign commits to dark mode first. Can be added as a follow-on if needed.

## Out of Scope

### R015 — Backend or pipeline changes
- Class: anti-feature
- Status: out-of-scope
- Description: No changes to FastAPI routers, services, models, pipeline stages, or database schema.
- Why it matters: Prevents scope bleed into working backend logic.
- Source: user
- Primary owning slice: none
- Supporting slices: none
- Validation: n/a

### R016 — New features or data model changes
- Class: anti-feature
- Status: out-of-scope
- Description: No new capabilities, endpoints, or fields. This milestone is purely a visual redesign.
- Why it matters: Keeps scope bounded.
- Source: user
- Primary owning slice: none
- Supporting slices: none
- Validation: n/a

## Traceability

| ID | Class | Status | Primary owner | Supporting | Proof |
|---|---|---|---|---|---|
| R001 | differentiator | active | M003/S01 | S02,S03,S04,S05 | partial — theme compiles, dark base + amber primary active in browser; all screens validated in S02–S05 |
| R002 | primary-user-loop | active | M003/S01 | S02,S03,S04,S05 | partial — sidebar + top bar layout verified on login/wizard/dashboard; remaining screens validated in S02–S05 |
| R003 | launchability | active | M003/S01 | none | verified — M003/S01 |
| R004 | launchability | active | M003/S01 | none | verified — M003/S01 |
| R005 | primary-user-loop | active | M003/S02 | none | verified — M003/S02; bordered dark sections, font-mono headings, no card artifacts |
| R006 | primary-user-loop | active | M003/S02 | none | verified — M003/S02; bordered form sections, HTMX swap IDs intact, warning tokens semantic |
| R007 | primary-user-loop | active | M003/S03 | none | verified — M003/S03; bordered sections, border-l-4 banners, SSE JS updated, DOM IDs preserved |
| R008 | primary-user-loop | active | M003/S03 | none | verified — M003/S02; mono table headers, hover rows, no light artifacts |
| R009 | differentiator | active | M003/S04 | none | verified — M003/S04; full-screen focused layout, one item at a time, large snippets, no scrolling |
| R010 | differentiator | active | M003/S04 | none | verified — M003/S04; J/K navigate, A approve, O override + auto-focus, legend visible, shortcuts suppressed in inputs/dialogs |
| R011 | primary-user-loop | active | M003/S04 | none | verified — M003/S04; OOB swaps live, Sign Off CTA enables when all resolved, modal fires |
| R012 | admin/support | active | M003/S05 | none | unmapped |
| R013 | quality-attribute | active | M003/S05 | S04 | unmapped |
| R014 | quality-attribute | deferred | none | none | unmapped |
| R015 | anti-feature | out-of-scope | none | none | n/a |
| R016 | anti-feature | out-of-scope | none | none | n/a |

## Coverage Summary

- Active requirements: 13
- Mapped to slices: 13
- Validated: 9
- Unmapped active requirements: 0
