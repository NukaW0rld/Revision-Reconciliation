---
estimated_steps: 5
estimated_files: 7
---

# T01: Redesign dashboard, runs list, new run form, and HTMX partials with industrial aesthetic

**Slice:** S02 — Dashboard, Runs List & New Run Form
**Milestone:** M003

## Description

Replace all generic DaisyUI card/shadow patterns in S02-owned templates with the industrial dark-mode aesthetic established in S01. This covers three full pages (dashboard, runs list, new run form) and four HTMX partials (xlsx mapping, page selector, pdf error, alert banner). The work is purely CSS class changes — no structural HTML changes, no new elements, no backend modifications. The one real bug to fix is hardcoded `bg-amber-100 border-amber-400` in the xlsx mapping partial, which renders a white background on the dark theme.

## Steps

1. **Dashboard (`dashboard.html`)**: Set `{% block page_title %}Dashboard{% endblock %}`. Replace `card bg-base-100 shadow hover:shadow-md` on admin cards and recent run cards with `border border-base-300 bg-base-200 hover:bg-base-300 transition-colors`. Replace `card-body` / `card-title` with raw `p-4`/`p-5` and `text-base font-semibold font-mono`. Add monospace accent to CTA button text. Keep the alert banner include and grid layout unchanged.

2. **Runs list (`runs/list.html`)**: Set `{% block page_title %}All Runs{% endblock %}`. Restyle filter form labels with `text-xs font-mono uppercase tracking-wider text-base-content/60`. Replace `table-zebra` with plain `table` and add `hover:bg-base-300 transition-colors` to `<tr>` elements. Style `<thead>` with `font-mono uppercase text-xs tracking-wider text-base-content/50`. Add `font-mono` to part number cells.

3. **New run form (`new.html`)**: Set `{% block page_title %}New Run{% endblock %}`. Replace all four `card bg-base-100 shadow mb-*` wrappers with `border border-base-300 bg-base-200 p-5 mb-*`. Replace `card-body` padding with the section div's own padding. Replace `card-title` with `text-base font-semibold font-mono mb-3`. Keep all HTMX attributes and swap target IDs (`#revA-section`, `#revB-section`, `#xlsx-mapping-section`) exactly as they are. Keep submit button outside all sections.

4. **HTMX partials**: 
   - `_xlsx_mapping.html`: Replace `bg-amber-100 border-amber-400` with `bg-warning/15 border-warning` on undetected column selects. Replace the amber legend swatch classes from `bg-amber-100 border border-amber-400` to `bg-warning/15 border border-warning`.
   - `_page_selector.html`: Remove `card` class and `card-body` wrapper — use plain `div` with `bg-warning/20 border border-warning p-3 mt-2`. Replace deprecated `bg-warning bg-opacity-20` with `bg-warning/20`.
   - `_pdf_error.html`: Keep `alert alert-error` — it's theme-aware and reads correctly on dark mode. Minimal or no changes needed.
   - `_alert_banner.html`: Restyle from `alert alert-error shadow-sm` to `border-l-4 border-error bg-base-200 p-3 flex items-center gap-3` for consistency with industrial aesthetic. Preserve `id="alert-banner-{{ alert.id }}"` exactly — it's the HTMX outerHTML swap target.

5. **Build and verify**: Run `npm run build:css`, run `uv run pytest tests/`, grep for eliminated patterns and preserved IDs.

## Must-Haves

- [ ] No `card bg-base-100 shadow` patterns remain in dashboard.html or new.html
- [ ] No `bg-amber-100` or `border-amber-400` in _xlsx_mapping.html
- [ ] No deprecated `bg-opacity-20` in _page_selector.html
- [ ] HTMX swap IDs preserved: `#revA-section`, `#revB-section`, `#xlsx-mapping-section`, `#alert-banner-{{ alert.id }}`
- [ ] `{% block page_title %}` set on all three full pages
- [ ] `npm run build:css` exits 0
- [ ] `uv run pytest tests/` passes with 93+ tests, 0 failures

## Verification

- `npm run build:css` → exit 0
- `uv run pytest tests/` → all pass
- `grep -c 'bg-base-100 shadow' shop/templates/dashboard.html shop/templates/runs/new.html` → 0 for each
- `grep -c 'bg-amber-100' shop/templates/runs/_xlsx_mapping.html` → 0
- `grep -c 'bg-opacity-20' shop/templates/runs/_page_selector.html` → 0
- `grep 'id="revA-section"' shop/templates/runs/new.html` → found
- `grep 'id="revB-section"' shop/templates/runs/new.html` → found
- `grep 'id="xlsx-mapping-section"' shop/templates/runs/new.html` → found
- `grep 'id="alert-banner-' shop/templates/runs/_alert_banner.html` → found

## Observability Impact

This task makes only CSS class changes — no runtime logic, no new endpoints, no data model changes.

**Signals that change:**
- Visual: dark-themed bordered sections replace card shadows; light backgrounds disappear from dark theme
- HTMX: swap behavior unchanged — `#revA-section`, `#revB-section`, `#xlsx-mapping-section`, `#alert-banner-{{ alert.id }}` remain valid targets
- CSS build output: `static/dist/output.css` includes new utility classes (`bg-warning/15`, `border-warning`, `bg-warning/20`)

**How a future agent inspects this task:**
- `grep -rn 'card bg-base-100 shadow' shop/templates/` → 0 matches = done
- `grep -rn 'bg-amber-100\|border-amber-400' shop/templates/` → 0 matches = amber hardcode fixed
- `grep -rn 'bg-opacity-20' shop/templates/` → 0 matches = deprecated syntax removed
- Browser: load `/dashboard`, `/runs`, `/runs/new` — all sections show dark `bg-base-200` backgrounds with `border-base-300` borders

**Failure state visibility:**
- Light-mode artifacts (white cards on dark page) = class replacement missed
- HTMX swap silently stops working = ID renamed or removed
- CSS build warns about unknown utilities = class typo in template

## Inputs

- `shop/templates/base.html` — S01 layout shell with sidebar, top bar, `{% block page_title %}`, `{% block content %}`
- `static/src/input.css` — Industrial theme OKLCH color tokens
- S01 summary — established patterns: `border border-base-300 bg-base-200` sections, `font-mono` headings, no card shadows
- S02 research — specific changes per template, HTMX ID inventory, dark-mode incompatibilities identified

## Expected Output

- `shop/templates/dashboard.html` — industrial dark aesthetic, bordered sections, monospaced accents
- `shop/templates/runs/list.html` — industrial table with mono headers, hover rows, themed badges
- `shop/templates/runs/new.html` — bordered form sections replacing cards, all HTMX preserved
- `shop/templates/runs/_xlsx_mapping.html` — `bg-warning/15 border-warning` replacing hardcoded amber
- `shop/templates/runs/_page_selector.html` — fixed deprecated opacity syntax, card class removed
- `shop/templates/runs/_pdf_error.html` — minimal changes (already theme-aware)
- `shop/templates/runs/_alert_banner.html` — industrial border-l accent pattern, swap ID preserved
