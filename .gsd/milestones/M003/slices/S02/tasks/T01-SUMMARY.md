---
id: T01
parent: S02
milestone: M003
provides:
  - Industrial dark-mode aesthetic applied to all S02-owned templates
key_files:
  - shop/templates/dashboard.html
  - shop/templates/runs/list.html
  - shop/templates/runs/new.html
  - shop/templates/runs/_xlsx_mapping.html
  - shop/templates/runs/_page_selector.html
  - shop/templates/runs/_pdf_error.html
  - shop/templates/runs/_alert_banner.html
key_decisions:
  - _pdf_error.html required no changes — `alert alert-error` is DaisyUI theme-aware and renders correctly on dark mode
  - Alert banner uses `border-l-4 border-error bg-base-200` left-accent pattern (vs full alert box) to match industrial aesthetic while preserving HTMX outerHTML swap ID
patterns_established:
  - Section wrappers: `border border-base-300 bg-base-200 p-4/p-5` instead of `card bg-base-100 shadow`
  - Section headings: `text-base font-semibold font-mono` instead of `card-title`
  - Table headers: `font-mono uppercase text-xs tracking-wider text-base-content/50`
  - Table rows: `hover:bg-base-300 transition-colors`
  - Warning highlights: `bg-warning/15 border-warning` instead of hardcoded `bg-amber-100 border-amber-400`
  - Warning panels: `bg-warning/20 border border-warning p-3` (plain div, no card wrapper)
  - Alert banners: `border-l-4 border-error bg-base-200 p-3 flex items-center gap-3`
observability_surfaces:
  - grep -rn 'card bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/ → 0 matches = clean
  - grep 'id="revA-section"\|id="revB-section"\|id="xlsx-mapping-section"\|id="alert-banner-' shop/templates/runs/ → confirms HTMX targets intact
duration: ~20m
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T01: Redesign dashboard, runs list, new run form, and HTMX partials with industrial aesthetic

**Replaced all `card bg-base-100 shadow` patterns with `border border-base-300 bg-base-200` bordered sections across seven templates, fixed hardcoded amber colors, and corrected deprecated opacity syntax.**

## What Happened

All seven S02-owned templates updated with CSS class changes only — no structural HTML changes, no backend modifications.

- **dashboard.html**: Admin cards and recent run cards converted to bordered sections. Added `font-mono` to h1, h2, section headings, and CTA button. Added `{% block page_title %}Dashboard{% endblock %}`.
- **runs/list.html**: Filter labels styled with `text-xs font-mono uppercase tracking-wider`. `table-zebra` replaced with plain `table`. `<thead>` gets `font-mono uppercase text-xs tracking-wider text-base-content/50`. `<tr>` elements get `hover:bg-base-300 transition-colors`. Part number cells get `font-mono`. Added `{% block page_title %}All Runs{% endblock %}`.
- **runs/new.html**: All four `card bg-base-100 shadow mb-*` wrappers replaced with `border border-base-300 bg-base-200 p-5 mb-*`. `card-body` removed, padding is on the section div. `card-title` replaced with `text-base font-semibold font-mono mb-3`. All HTMX attributes and swap targets preserved verbatim. Added `{% block page_title %}New Run{% endblock %}`.
- **_xlsx_mapping.html**: `bg-amber-100 border-amber-400` → `bg-warning/15 border-warning` on undetected column selects. Legend swatch updated to match. Legend text updated to "Highlighted columns" (no longer amber-specific).
- **_page_selector.html**: `card` wrapper and `card-body` removed. `bg-warning bg-opacity-20` → `bg-warning/20`. Plain div with `bg-warning/20 border border-warning p-3 mt-2`. Section heading gets `font-mono text-sm`.
- **_pdf_error.html**: No changes needed — `alert alert-error` is DaisyUI theme-aware.
- **_alert_banner.html**: `alert alert-error shadow-sm` → `border-l-4 border-error bg-base-200 p-3 flex items-center gap-3`. Icon gets `text-error` class for explicit color. Run label gets `font-mono text-sm`. HTMX swap ID `id="alert-banner-{{ alert.id }}"` preserved exactly.

## Verification

```
grep -c 'bg-base-100 shadow' shop/templates/dashboard.html shop/templates/runs/new.html
# dashboard.html:0  new.html:0 ✓

grep -c 'bg-amber-100' shop/templates/runs/_xlsx_mapping.html
# 0 ✓

grep -c 'bg-opacity-20' shop/templates/runs/_page_selector.html
# 0 ✓

grep HTMX IDs in new.html → id="revA-section", id="revB-section", id="xlsx-mapping-section" ✓
grep id="alert-banner-" in _alert_banner.html ✓

{% block page_title %} in dashboard.html, list.html, new.html ✓

npm run build:css → exit 0 (223ms, no warnings) ✓
uv run pytest tests/ → 93 passed, 2 xfailed, 0 failures ✓
```

## Diagnostics

- Visual: load `/dashboard`, `/runs`, `/runs/new` — dark `bg-base-200` sections with `border-base-300` borders, no white card artifacts
- HTMX: upload a PDF on new run form — `#revA-section` / `#revB-section` swap targets fire correctly; upload xlsx — `#xlsx-mapping-section` swaps in the mapping table
- Pattern grep: `grep -rn 'card bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/` → 0 matches

## Deviations

- `_pdf_error.html` received no changes (plan allowed "minimal or no changes"). Confirmed theme-aware, no action needed.
- Alert banner icon: added `text-error` class for explicit color in non-alert container context (plan didn't specify, but necessary for correct rendering).

## Known Issues

None.

## Files Created/Modified

- `shop/templates/dashboard.html` — industrial bordered sections, font-mono accents, page_title block
- `shop/templates/runs/list.html` — mono table headers, hover rows, page_title block
- `shop/templates/runs/new.html` — bordered form sections replacing cards, all HTMX preserved, page_title block
- `shop/templates/runs/_xlsx_mapping.html` — bg-warning/15 border-warning replaces hardcoded amber
- `shop/templates/runs/_page_selector.html` — plain div, bg-warning/20, deprecated opacity removed
- `shop/templates/runs/_pdf_error.html` — no changes (already theme-aware)
- `shop/templates/runs/_alert_banner.html` — border-l-4 accent pattern, swap ID preserved
- `.gsd/milestones/M003/slices/S02/S02-PLAN.md` — added Observability/Diagnostics section
- `.gsd/milestones/M003/slices/S02/tasks/T01-PLAN.md` — added Observability Impact section
