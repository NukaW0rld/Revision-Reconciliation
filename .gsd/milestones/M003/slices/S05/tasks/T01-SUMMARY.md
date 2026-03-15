---
id: T01
parent: S05
milestone: M003
provides:
  - Industrial aesthetic applied to all remaining templates; zero legacy DaisyUI alert/card/shadow patterns in template tree
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
  - base.html nav notification badge (badge-error) converted from DaisyUI badge to inline bg-error/text-error classes to satisfy zero-badge-error grep check; functionally equivalent
patterns_established:
  - border-l-4 border-{severity} bg-base-200 p-3 flex items-center gap-3 font-mono text-sm — alert banner pattern (bg-base-300 inside standalone panels)
  - border border-base-300 bg-base-200 p-5 — section wrapper replacing card bg-base-100 shadow
  - font-mono text-xs tracking-widest uppercase text-base-content/60 — label pattern
  - input input-bordered bg-base-200 font-mono text-sm focus:border-primary focus:outline-none — text input pattern
  - font-mono text-xs text-success / text-error — status indicator replacing badge-success/badge-error
  - bg-warning/15 border-warning — undetected column highlight in xlsx mapping partial
observability_surfaces:
  - grep -rn 'alert alert-\|card bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/ → 0 matches (primary inspection signal)
  - grep -rn 'badge-success\|badge-error' shop/templates/ → 0 matches
  - npm run build:css → exit 0 (CSS class resolution check)
  - uv run pytest tests/ → 93 passed 2 xfailed (Jinja2 render errors surface as 500s in test suite)
duration: ~25m
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T01: Apply industrial aesthetic to all remaining templates and clean up prior-slice legacy patterns

**CSS class substitution pass complete — zero legacy DaisyUI alert/card/shadow patterns remain across all 11 templates.**

## What Happened

Worked through all three template groups in order:

**Admin (3 files):** `users.html` — replaced both `card bg-base-100 shadow` wrappers with `border border-base-300 bg-base-200 p-5` sections; converted table headers to `font-mono uppercase text-xs tracking-wider text-base-content/50`; converted tbody rows to `hover:bg-base-300 transition-colors`; converted status badges to `font-mono text-xs text-success/text-error`; styled Add Engineer inputs and labels with industrial pattern; added `{% block page_title %}User Management{% endblock %}`; removed `container mx-auto p-8` wrapper. `users_row.html` — matched exactly: same status text treatment, same role badge strategy, error row → `border-l-4 border-error`. `settings.html` — same section wrapper pattern; both alert-error and alert-success → `border-l-4` banners with appropriate colors; industrial inputs; added `{% block page_title %}Shop Settings{% endblock %}`.

**Setup wizard (5 files):** `step1_shop_name.html` and `step2_password.html` — font-mono labels + industrial inputs; step2 `alert-info` → `border-l-4 border-info bg-base-300`. `wizard_layout.html` — `alert-error` → `border-l-4 border-error bg-base-300` (bg-base-300 because it's inside a bg-base-200 panel). `step4_error.html` — `alert-error` → `border-l-4 border-error bg-base-200`. `step4_mapping_partial.html` — `bg-amber-100 border-amber-400` → `bg-warning/15 border-warning` on both the select elements and the legend swatch; table headers styled with mono/uppercase pattern; updated legend text to remove "Amber" reference.

**Prior-slice cleanup (3 files):** `login.html` — `alert-error` → `border-l-4 border-error bg-base-300` (bg-base-300 because it's inside the dark standalone panel). `runs/new.html` — `alert-error` → `border-l-4 border-error bg-base-200`; all five text inputs styled with industrial pattern; labels styled; removed outer `container mx-auto px-4 py-8 max-w-3xl` wrapper (now uses `max-w-3xl` div inside content). `_pdf_error.html` — `alert-error` → `border-l-4 border-error bg-base-200`.

**base.html (1 extra file):** The nav alert-count notification dot used `badge badge-error badge-xs` — converted to `bg-error text-error-content` with manual sizing to satisfy the zero `badge-error` grep check without changing appearance.

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

npm run build:css → exit 0 ✓
uv run pytest tests/ → 93 passed, 2 xfailed, 0 failures ✓
```

Visual: admin/users and admin/settings confirmed in browser — dark bordered sections, mono headings, green "Active" text (not badge), no light-mode artifacts.

## Diagnostics

Primary inspection: `grep -rn 'alert alert-\|card bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/` — zero matches confirms clean state. Jinja2 errors surface as 500s in test suite; `uv run pytest tests/test_admin.py -v` is the fastest targeted check for admin template breakage.

## Deviations

- **base.html added to touched files** — not in the original 11-file list. The nav `badge-error` on the alert count dot needed conversion to meet the zero `badge-error` grep requirement.
- **Legend text updated in step4_mapping_partial.html** — changed "Amber columns were not auto-detected" to "Undetected columns were not auto-matched" to remove the color-name reference that no longer matches the new warning styling. Minor copy improvement, not in the plan.

## Known Issues

None.

## Files Created/Modified

- `shop/templates/admin/users.html` — card/shadow → bordered sections; status badge → text; industrial inputs/labels; page_title block; HTMX targets preserved
- `shop/templates/admin/users_row.html` — row styling matches users.html exactly; status → font-mono text; error row → border-l-4
- `shop/templates/admin/settings.html` — card/shadow → bordered sections; alert-error/success → border-l-4; industrial inputs; page_title block
- `shop/templates/setup/step1_shop_name.html` — font-mono label; industrial input
- `shop/templates/setup/step2_password.html` — font-mono labels; industrial inputs; alert-info → border-l-4 border-info bg-base-300
- `shop/templates/setup/wizard_layout.html` — alert-error → border-l-4 border-error bg-base-300
- `shop/templates/setup/step4_error.html` — alert-error → border-l-4 border-error bg-base-200
- `shop/templates/setup/step4_mapping_partial.html` — bg-amber-100 → bg-warning/15; legend swatch updated; table headers styled
- `shop/templates/auth/login.html` — alert-error → border-l-4 border-error bg-base-300
- `shop/templates/runs/new.html` — alert-error → border-l-4; five text inputs → industrial styling; labels styled
- `shop/templates/runs/_pdf_error.html` — alert-error → border-l-4 border-error bg-base-200
- `shop/templates/base.html` — nav badge-error notification dot → inline bg-error classes
