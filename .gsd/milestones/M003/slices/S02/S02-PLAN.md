# S02: Dashboard, Runs List & New Run Form

**Goal:** All run-related pages and partials render in the industrial dark-mode aesthetic with no generic DaisyUI card/shadow patterns remaining.
**Demo:** Navigate dashboard → runs list → new run form in a running browser; all three pages show bordered sections on dark backgrounds with monospaced headers, themed status badges, and no light-mode artifacts.

## Must-Haves

- Dashboard uses bordered `bg-base-200` sections instead of `card bg-base-100 shadow` for admin cards and recent run cards
- Runs list table uses `font-mono uppercase text-xs` header row with `hover:bg-base-300` row hover
- New run form replaces all `card bg-base-100 shadow` wrappers with `border border-base-300 bg-base-200` sections
- `_xlsx_mapping.html` replaces `bg-amber-100 border-amber-400` with `bg-warning/15 border-warning`
- `_page_selector.html` replaces deprecated `bg-opacity-20` with `bg-warning/20` and drops `card` class
- All HTMX swap target IDs preserved: `#revA-section`, `#revB-section`, `#xlsx-mapping-section`, `#alert-banner-{{ alert.id }}`
- `npm run build:css` exits clean
- `uv run pytest tests/` passes with no regressions

## Verification

- `npm run build:css` → exit 0, no warnings
- `uv run pytest tests/` → 93+ passed, 0 failures
- `grep -c 'bg-base-100 shadow' shop/templates/dashboard.html shop/templates/runs/new.html` → 0 matches
- `grep -c 'bg-amber-100' shop/templates/runs/_xlsx_mapping.html` → 0 matches
- `grep 'id="revA-section"\|id="revB-section"\|id="xlsx-mapping-section"' shop/templates/runs/new.html` → all three present
- `grep 'id="alert-banner-' shop/templates/runs/_alert_banner.html` → present

## Tasks

- [x] **T01: Redesign dashboard, runs list, new run form, and HTMX partials with industrial aesthetic** `est:45m`
  - Why: All S02-owned templates still use generic DaisyUI card/shadow patterns from the pre-redesign codebase; this is the entire scope of S02
  - Files: `shop/templates/dashboard.html`, `shop/templates/runs/list.html`, `shop/templates/runs/new.html`, `shop/templates/runs/_xlsx_mapping.html`, `shop/templates/runs/_page_selector.html`, `shop/templates/runs/_pdf_error.html`, `shop/templates/runs/_alert_banner.html`
  - Do: Replace `card bg-base-100 shadow` with `border border-base-300 bg-base-200 p-4/p-5` throughout; add `font-mono` to section headings and `{% block page_title %}` content; replace hardcoded `bg-amber-100 border-amber-400` in xlsx mapping with `bg-warning/15 border-warning`; fix deprecated `bg-opacity-20` in page selector; restyle runs list table header with `font-mono uppercase text-xs tracking-wider`; add `hover:bg-base-300` to table rows; restyle alert banner with `border-l-4 border-error bg-base-200` pattern; preserve all HTMX swap target IDs verbatim; rebuild CSS
  - Verify: `npm run build:css` clean; `uv run pytest tests/` passes; grep confirms no `bg-base-100 shadow` or `bg-amber-100` remnants; all HTMX swap IDs present
  - Done when: all seven templates updated, CSS builds clean, tests pass, no light-mode artifacts in template source

## Observability / Diagnostics

This slice is CSS-only — no runtime behavior changes. Diagnostic signals:

- **Visual regression**: Run the dev server (`uv run python run_web.py`) and navigate dashboard → runs list → new run form. Any `bg-base-100` sections or light backgrounds on dark theme are visible light-mode artifacts.
- **Pattern grep**: `grep -rn 'bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/` → must return 0 matches after completion.
- **HTMX swap ID integrity**: `grep 'id="revA-section"\|id="revB-section"\|id="xlsx-mapping-section"\|id="alert-banner-' shop/templates/runs/` — missing IDs cause silent HTMX swap failures with no client-side error; check network tab for 2xx responses with empty swap targets.
- **CSS build**: `npm run build:css` failures surface as missing utility classes at runtime (invisible until page load).
- **Test failures**: `uv run pytest tests/` — any non-zero exit indicates regression; 93+ tests expected.
- **Redaction**: No secrets in templates; nothing to redact.

## Files Likely Touched

- `shop/templates/dashboard.html`
- `shop/templates/runs/list.html`
- `shop/templates/runs/new.html`
- `shop/templates/runs/_xlsx_mapping.html`
- `shop/templates/runs/_page_selector.html`
- `shop/templates/runs/_pdf_error.html`
- `shop/templates/runs/_alert_banner.html`
