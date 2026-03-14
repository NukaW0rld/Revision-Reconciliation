---
id: S02
parent: M003
milestone: M003
provides:
  - Industrial dark-mode aesthetic applied to all S02-owned templates (dashboard, runs list, new run form, four HTMX partials)
requires:
  - slice: S01
    provides: base.html sidebar+top-bar layout, compiled output.css with dark theme tokens
affects:
  - S05
key_files:
  - shop/templates/dashboard.html
  - shop/templates/runs/list.html
  - shop/templates/runs/new.html
  - shop/templates/runs/_xlsx_mapping.html
  - shop/templates/runs/_page_selector.html
  - shop/templates/runs/_pdf_error.html
  - shop/templates/runs/_alert_banner.html
key_decisions:
  - Section wrapper pattern settled as `border border-base-300 bg-base-200 p-4/p-5` — no card/shadow; used consistently across all three page templates
  - `_pdf_error.html` required no changes — `alert alert-error` is DaisyUI theme-aware and already correct in dark mode
  - Alert banner uses left-accent pattern (`border-l-4 border-error bg-base-200`) not full alert box — matches industrial aesthetic while preserving outerHTML HTMX swap ID
  - Warning highlights use `bg-warning/15 border-warning` (DaisyUI semantic tokens) instead of hardcoded `bg-amber-100 border-amber-400` — survives theme changes
patterns_established:
  - Section wrappers: `border border-base-300 bg-base-200 p-4/p-5` replaces `card bg-base-100 shadow`
  - Section headings: `text-base font-semibold font-mono` replaces `card-title`
  - Table headers: `font-mono uppercase text-xs tracking-wider text-base-content/50`
  - Table rows: `hover:bg-base-300 transition-colors` (plain table, no table-zebra)
  - Warning highlights: `bg-warning/15 border-warning` (semantic tokens)
  - Warning panels: `bg-warning/20 border border-warning p-3` plain div, no card wrapper
  - Alert banners: `border-l-4 border-error bg-base-200 p-3 flex items-center gap-3`
observability_surfaces:
  - grep -rn 'card bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/ → 0 matches = clean
  - grep 'id="revA-section"\|id="revB-section"\|id="xlsx-mapping-section"\|id="alert-banner-' shop/templates/runs/ → all four present
drill_down_paths:
  - .gsd/milestones/M003/slices/S02/tasks/T01-SUMMARY.md
duration: ~20m (single task)
verification_result: passed
completed_at: 2026-03-14
---

# S02: Dashboard, Runs List & New Run Form

**CSS-only redesign of seven templates: all generic DaisyUI card/shadow patterns replaced with bordered industrial sections, hardcoded amber colors swapped for semantic tokens, and deprecated opacity syntax corrected — CSS builds clean, 93 tests pass, all HTMX swap targets intact.**

## What Happened

Single task (T01) covered all seven S02-owned templates with CSS class changes only — no structural HTML changes, no backend modifications.

**dashboard.html** — Admin cards and recent run cards converted to bordered `bg-base-200` sections. `font-mono` added to h1, h2, section headings, and CTA button. Added `{% block page_title %}Dashboard{% endblock %}`.

**runs/list.html** — Filter labels styled with `text-xs font-mono uppercase tracking-wider`. `table-zebra` removed; `<thead>` gets mono uppercase headers; `<tr>` elements get `hover:bg-base-300 transition-colors`; part number cells get `font-mono`. Added `{% block page_title %}All Runs{% endblock %}`.

**runs/new.html** — All four `card bg-base-100 shadow` wrappers replaced with `border border-base-300 bg-base-200 p-5`. `card-body` removed, `card-title` replaced with `text-base font-semibold font-mono mb-3`. All HTMX attributes and swap targets (`#revA-section`, `#revB-section`, `#xlsx-mapping-section`) preserved verbatim. Added `{% block page_title %}New Run{% endblock %}`.

**_xlsx_mapping.html** — `bg-amber-100 border-amber-400` → `bg-warning/15 border-warning` on undetected column selects. Legend swatch updated to match.

**_page_selector.html** — `card` wrapper and `card-body` removed. `bg-warning bg-opacity-20` → `bg-warning/20`. Plain div with `bg-warning/20 border border-warning p-3 mt-2`.

**_pdf_error.html** — No changes. `alert alert-error` is DaisyUI theme-aware; renders correctly in dark mode without modification.

**_alert_banner.html** — `alert alert-error shadow-sm` → `border-l-4 border-error bg-base-200 p-3 flex items-center gap-3`. Icon gets explicit `text-error` class. Run label gets `font-mono text-sm`. HTMX swap ID `id="alert-banner-{{ alert.id }}"` preserved exactly.

## Verification

```
npm run build:css → exit 0, 223ms, no warnings ✓
uv run pytest tests/ → 93 passed, 2 xfailed, 0 failures ✓
grep -c 'bg-base-100 shadow' dashboard.html new.html → 0 ✓
grep -c 'bg-amber-100' _xlsx_mapping.html → 0 ✓
grep -c 'bg-opacity-20' _page_selector.html → 0 ✓
grep id="revA-section" new.html → present ✓
grep id="revB-section" new.html → present ✓
grep id="xlsx-mapping-section" new.html → present ✓
grep id="alert-banner-" _alert_banner.html → present ✓
```

## Requirements Advanced

- R005 — Dashboard now renders with bordered industrial sections, font-mono headings, and themed status badges
- R006 — New run form uses bordered section wrappers; xlsx mapping partial uses semantic warning tokens; page selector uses plain warning div
- R008 — Runs list table uses mono headers, hover rows, and no light-mode artifacts

## Requirements Validated

- R005 — Dashboard fully redesigned; template source confirmed clean of `bg-base-100 shadow`; CSS builds clean; tests pass
- R006 — New run form and all three partials redesigned; all HTMX swap targets preserved; CSS builds clean; tests pass
- R008 — Runs list redesigned; CSS builds clean; tests pass

## New Requirements Surfaced

- none

## Requirements Invalidated or Re-scoped

- none

## Deviations

- `_pdf_error.html` received no changes — plan stated "minimal or no changes allowed" and the template was confirmed theme-aware. No action taken.
- Alert banner icon: added `text-error` class for explicit color in non-alert container context — not specified in plan but required for correct dark-mode rendering.

## Known Limitations

- Visual validation (running browser) was not performed in this slice — T01 relied on grep/build/test verification. A human reviewer should visually confirm the three pages render without light-mode artifacts.

## Follow-ups

- none — all deferred items are covered by S03–S05

## Files Created/Modified

- `shop/templates/dashboard.html` — industrial bordered sections, font-mono accents, page_title block
- `shop/templates/runs/list.html` — mono table headers, hover rows, page_title block
- `shop/templates/runs/new.html` — bordered form sections, all HTMX preserved, page_title block
- `shop/templates/runs/_xlsx_mapping.html` — bg-warning/15 border-warning replaces hardcoded amber
- `shop/templates/runs/_page_selector.html` — plain div, bg-warning/20, deprecated opacity removed
- `shop/templates/runs/_pdf_error.html` — no changes (already theme-aware)
- `shop/templates/runs/_alert_banner.html` — border-l-4 accent pattern, swap ID preserved

## Forward Intelligence

### What the next slice should know
- The `border border-base-300 bg-base-200 p-4/p-5` section wrapper pattern is the canonical choice for all authenticated page content blocks — use it in S03 (status page) and S04 (review queue item card)
- `{% block page_title %}` is required in all page templates that extend base.html; the top bar renders it as the breadcrumb/context label
- `_pdf_error.html` uses `alert alert-error` which is already theme-aware — no redesign needed even in S05

### What's fragile
- HTMX swap IDs in `new.html` (`#revA-section`, `#revB-section`, `#xlsx-mapping-section`) — these are referenced by hx-target attributes in the same template; any rename breaks the swap silently (2xx response, empty target)
- `_alert_banner.html` ID is `id="alert-banner-{{ alert.id }}"` — the HTMX outerHTML swap depends on this exact format; if the alert model ever changes its `id` field name, this breaks silently

### Authoritative diagnostics
- `grep -rn 'card bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/` → 0 matches = clean slate for downstream slices
- `npm run build:css` is the fastest build signal — 223ms; run it after any CSS class change to catch missing utilities immediately

### What assumptions changed
- Original plan assumed `_pdf_error.html` might need minor updates — actual execution confirmed it required zero changes; the `alert alert-error` component is fully DaisyUI theme-aware
