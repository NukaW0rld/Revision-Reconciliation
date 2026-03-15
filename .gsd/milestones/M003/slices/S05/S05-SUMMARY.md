---
id: S05
parent: M003
milestone: M003
provides:
  - Zero legacy DaisyUI alert/card/shadow patterns across entire template tree
  - Admin screens (users.html, users_row.html, settings.html) redesigned with industrial aesthetic
  - Setup wizard sub-templates (step1, step2, wizard_layout, step4_error, step4_mapping_partial) updated
  - Prior-slice cleanup: login.html, runs/new.html, runs/_pdf_error.html legacy patterns removed
  - base.html nav badge converted; full template tree passes zero-legacy-pattern grep
requires:
  - slice: S01
    provides: base.html sidebar+topbar layout, output.css design tokens
  - slice: S02
    provides: runs/list.html and dashboard.html as aesthetic reference; bordered section and border-l-4 patterns established
  - slice: S03
    provides: status.html and _alert_banner.html as reference; confirmed border-l-4 pattern for all status banners
  - slice: S04
    provides: review queue templates as reference; confirmed fragment/OOB swap compatibility
affects: []
key_files:
  - shop/templates/admin/users.html
  - shop/templates/admin/users_row.html
  - shop/templates/admin/settings.html
  - shop/templates/setup/step1_shop_name.html
  - shop/templates/setup/step2_password.html
  - shop/templates/setup/wizard_layout.html
  - shop/templates/setup/step4_error.html
  - shop/templates/setup/step4_mapping_partial.html
  - shop/templates/auth/login.html
  - shop/templates/runs/new.html
  - shop/templates/runs/_pdf_error.html
  - shop/templates/base.html
key_decisions:
  - base.html nav badge-error notification dot converted to inline bg-error/text-error-content classes (not DaisyUI badge) to satisfy zero-badge-error grep requirement while preserving visual appearance
  - step4_mapping_partial.html legend copy updated from "Amber columns" to "Undetected columns" to remove color-name reference that no longer matched warning token styling
patterns_established:
  - border-l-4 border-{severity} bg-base-200 p-3 flex items-center gap-3 font-mono text-sm — alert banner (bg-base-300 inside standalone panels)
  - border border-base-300 bg-base-200 p-5 — section wrapper replacing card bg-base-100 shadow
  - font-mono text-xs tracking-widest uppercase text-base-content/60 — table/form label pattern
  - input input-bordered bg-base-200 font-mono text-sm focus:border-primary focus:outline-none — text input pattern
  - font-mono text-xs text-success / text-error — status indicator replacing badge-success/badge-error
  - bg-warning/15 border-warning — undetected column highlight in xlsx mapping partial
observability_surfaces:
  - grep -rn 'alert alert-\|card bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/ → 0 matches (zero legacy patterns)
  - grep -rn 'badge-success\|badge-error' shop/templates/ → 0 matches (zero badge status classes)
  - npm run build:css → exit 0 (CSS class resolution and compile check)
  - uv run pytest tests/ → 93 passed 2 xfailed (Jinja2 render errors surface as 500s in admin/setup test coverage)
drill_down_paths:
  - .gsd/milestones/M003/slices/S05/tasks/T01-SUMMARY.md
duration: ~25m
verification_result: passed
completed_at: 2026-03-14
---

# S05: Admin, Setup Wizard & Fragment Cleanup

**Every screen and partial in the app is now on the industrial dark-mode design — zero legacy DaisyUI alert/card/shadow classes remain anywhere in the template tree.**

## What Happened

Single task (T01) — CSS class substitution pass across 12 templates grouped in three logical batches:

**Admin screens (3 files):** `users.html` replaced both `card bg-base-100 shadow` wrappers with `border border-base-300 bg-base-200 p-5` bordered sections. Table headers converted to `font-mono uppercase text-xs tracking-wider text-base-content/50`. Row hover states added with `hover:bg-base-300 transition-colors`. Status badges (`badge-success`/`badge-error`) converted to `font-mono text-xs text-success/text-error` inline text. Add Engineer form inputs and labels styled with industrial pattern. `{% block page_title %}User Management{% endblock %}` added. HTMX targets (`id="users-table"`, `hx-target="#users-table tbody"`, `hx-target="closest tr"`) preserved verbatim. `users_row.html` matched exactly — same status text treatment, same role badge strategy (badge-primary/badge-neutral kept), error row uses `border-l-4 border-error`. `settings.html` received same section wrapper treatment; both `alert-error` and `alert-success` converted to `border-l-4` banners with appropriate severity colors; industrial inputs; `{% block page_title %}Shop Settings{% endblock %}` added.

**Setup wizard (5 files):** `step1_shop_name.html` and `step2_password.html` got font-mono labels and industrial input styling. Step2's `alert-info` converted to `border-l-4 border-info bg-base-300`. `wizard_layout.html` `alert-error` → `border-l-4 border-error bg-base-300` (bg-base-300 inside bg-base-200 panel for layering). `step4_error.html` `alert-error` → `border-l-4 border-error bg-base-200`. `step4_mapping_partial.html` `bg-amber-100 border-amber-400` → `bg-warning/15 border-warning` on select elements and legend swatch; table headers mono/uppercase; legend copy updated to remove "Amber" color reference.

**Prior-slice cleanup (3 files + base.html):** `login.html` `alert-error` → `border-l-4 border-error bg-base-300` (standalone panel, bg-base-300). `runs/new.html` `alert-error` → `border-l-4 border-error bg-base-200`; all five text inputs converted to industrial pattern; container wrapper simplified. `_pdf_error.html` `alert-error` → `border-l-4 border-error bg-base-200`. `base.html` nav notification dot's `badge badge-error badge-xs` converted to inline `bg-error text-error-content` with manual sizing — not in the original 11-file scope, added to satisfy the zero `badge-error` grep requirement.

## Verification

All slice-level checks passed:

```
grep -rn 'alert alert-\|card bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/
# → exit 1 (0 matches) ✓

grep -rn 'badge-success\|badge-error' shop/templates/
# → exit 1 (0 matches) ✓

grep 'block page_title' shop/templates/admin/users.html shop/templates/admin/settings.html
# → both present ✓

grep 'id="users-table"' shop/templates/admin/users.html
grep 'hx-target="#users-table tbody"' shop/templates/admin/users.html
grep 'hx-target="closest tr"' shop/templates/admin/users.html
# → all present ✓

npm run build:css → exit 0 ✓ (DaisyUI 5.5.19, Tailwind v4.2.1, 218ms)
uv run pytest tests/ → 93 passed, 2 xfailed, 0 failures ✓
```

Visual: admin/users and admin/settings confirmed in browser — dark bordered sections, mono headings, `text-success` "Active" text, no light-mode artifacts.

## Requirements Advanced

- R001 — Industrial dark-mode aesthetic now covers every template in the tree; no prior-slice omissions remain
- R012 — Admin user management and settings forms redesigned with full industrial aesthetic
- R013 — All HTMX partial/fragment templates now match the redesigned aesthetic; swap targets preserved

## Requirements Validated

- R012 — admin/users.html and admin/settings.html confirmed in browser: dark bordered sections, mono headings, status as font-mono text, no card/shadow/badge artifacts
- R013 — grep confirms zero legacy patterns; HTMX targets verified by grep; CSS builds clean; test suite (93 passed) covers all admin template rendering paths

## New Requirements Surfaced

- none

## Requirements Invalidated or Re-scoped

- none

## Deviations

- **base.html added to touched files** — not in the original 11-file list. The nav `badge badge-error badge-xs` notification dot needed conversion to meet the `badge-error` grep requirement.
- **Legend copy updated in step4_mapping_partial.html** — changed "Amber columns were not auto-detected" to "Undetected columns were not auto-matched" to remove the now-incorrect color-name reference. Minor copy improvement, not in plan.

## Known Limitations

None — this is the terminal slice of M003. All screens are on the industrial design. No visual inconsistencies remain in the template tree.

## Follow-ups

- R014 (dark/light mode toggle) remains deferred — no action needed here.
- WeasyPrint PDF templates (`audit_packet.html`, `work_order.html`) were explicitly out of scope (D005) — if a print redesign milestone is ever opened, these are the remaining files with non-industrial styling.

## Files Created/Modified

- `shop/templates/admin/users.html` — card/shadow → bordered sections; status badge → font-mono text; industrial inputs/labels; page_title block; HTMX targets preserved
- `shop/templates/admin/users_row.html` — row styling matches users.html exactly; status → font-mono text; error row → border-l-4
- `shop/templates/admin/settings.html` — card/shadow → bordered sections; alert-error/success → border-l-4; industrial inputs; page_title block
- `shop/templates/setup/step1_shop_name.html` — font-mono label; industrial input
- `shop/templates/setup/step2_password.html` — font-mono labels; industrial inputs; alert-info → border-l-4 border-info bg-base-300
- `shop/templates/setup/wizard_layout.html` — alert-error → border-l-4 border-error bg-base-300
- `shop/templates/setup/step4_error.html` — alert-error → border-l-4 border-error bg-base-200
- `shop/templates/setup/step4_mapping_partial.html` — bg-amber-100 → bg-warning/15; legend swatch and copy updated; table headers styled
- `shop/templates/auth/login.html` — alert-error → border-l-4 border-error bg-base-300
- `shop/templates/runs/new.html` — alert-error → border-l-4; five text inputs → industrial styling; labels styled
- `shop/templates/runs/_pdf_error.html` — alert-error → border-l-4 border-error bg-base-200
- `shop/templates/base.html` — nav badge-error notification dot → inline bg-error/text-error-content classes

## Forward Intelligence

### What the next slice should know

- The entire template tree is now on the industrial design. `grep -rn 'alert alert-\|card bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/` returning 0 matches is the canonical clean-state check — run it first after any template change.
- WeasyPrint templates (`shop/templates/audit_packet.html`, `shop/templates/work_order.html`) were intentionally excluded from M003 scope (D005). They use separate print-specific styling and were not touched.

### What's fragile

- `bg-warning/15` opacity modifier — Tailwind v4 supports this natively but it depends on `bg-warning` being defined as a CSS custom property in the DaisyUI theme. If the theme is ever regenerated or DaisyUI is upgraded, verify this token still resolves in the build output.
- base.html badge conversion — The `bg-error text-error-content` inline sizing is manual (7px × 7px dot). If DaisyUI ever adds a non-badge utility for notification dots, this could be simplified.

### Authoritative diagnostics

- `grep -rn 'alert alert-\|card bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/` — zero matches is the single most important invariant for visual consistency; trust this over visual inspection alone
- `uv run pytest tests/test_admin.py -v` — fastest targeted check for admin template Jinja2 breakage (500s surface as test failures)
- `npm run build:css` output — Tailwind purge warnings indicate a class was added to templates but isn't in the scanned source

### What assumptions changed

- base.html was assumed to be clean after S01 — the `badge-error` nav dot was missed during S01 cleanup and required an unplanned edit in S05.
