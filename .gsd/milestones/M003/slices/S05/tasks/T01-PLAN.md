---
estimated_steps: 10
estimated_files: 11
---

# T01: Apply industrial aesthetic to all remaining templates and clean up prior-slice legacy patterns

**Slice:** S05 — Admin, Setup Wizard & Fragment Cleanup
**Milestone:** M003

## Description

CSS-class substitution pass across all remaining templates that still carry legacy DaisyUI patterns (card/shadow wrappers, alert-error/info/success boxes, plain inputs, hardcoded amber colors). Every pattern to apply already exists in prior-slice templates — this task ports them consistently. No structural HTML changes, no backend modifications.

Three template groups: admin screens (users.html, users_row.html, settings.html), setup wizard fragments (step1, step2, wizard_layout alert, step4_error, step4_mapping_partial), and prior-slice cleanup (login.html alert, runs/new.html alert + inputs, _pdf_error.html alert).

## Steps

1. **Admin users.html** — Replace `card bg-base-100 shadow` wrappers with `border border-base-300 bg-base-200 p-5` sections. Replace `card-body`/`card-title` with `text-base font-semibold font-mono mb-3` headings. Style table: `font-mono uppercase text-xs tracking-wider text-base-content/50` on `<thead>`, `hover:bg-base-300 transition-colors` on `<tbody tr>`. Convert status `badge-success`/`badge-error` to `font-mono text-xs` with `text-success`/`text-error`. Keep role `badge-primary`/`badge-neutral`. Convert `alert alert-error` to `border-l-4 border-error bg-base-200 p-3 flex items-center gap-3 font-mono text-sm`. Style inputs with `bg-base-200 font-mono text-sm focus:border-primary focus:outline-none`. Style labels with `font-mono text-xs tracking-widest uppercase text-base-content/60`. Add `{% block page_title %}User Management{% endblock %}`. Remove `container mx-auto p-8` (base.html provides padding). Preserve all HTMX attributes verbatim: `id="users-table"`, `hx-target="#users-table tbody"`, `hx-swap="beforeend"`, `hx-target="closest tr"`, `hx-swap="outerHTML"`.

2. **Admin users_row.html** — Match users.html row styling exactly. Convert status badge to `font-mono text-xs text-success`/`text-error`. Keep role badge `badge-primary`/`badge-neutral`. Convert error row `alert alert-error` to `border-l-4 border-error bg-base-200 p-3 font-mono text-sm`. Preserve all HTMX attributes and `id="user-{{ user.id }}"`.

3. **Admin settings.html** — Replace both `card bg-base-100 shadow` sections with `border border-base-300 bg-base-200 p-5`. Replace `card-body`/`card-title` with `text-base font-semibold font-mono mb-3`. Style inputs with `bg-base-200 font-mono text-sm focus:border-primary focus:outline-none`. Style labels with `font-mono text-xs tracking-widest uppercase text-base-content/60`. Convert `alert alert-error` to `border-l-4 border-error bg-base-200 p-3 flex items-center gap-3 font-mono text-sm`. Convert `alert alert-success` to `border-l-4 border-success bg-base-200 p-3 flex items-center gap-3 font-mono text-sm`. Add `{% block page_title %}Shop Settings{% endblock %}`. Remove `container mx-auto p-8 max-w-4xl` (base.html provides padding; add `max-w-4xl` inside content block if needed).

4. **Setup step1_shop_name.html** — Style heading with `font-mono`. Style label span with `font-mono text-xs tracking-widest uppercase text-base-content/60`. Style input with `bg-base-200 font-mono text-sm focus:border-primary focus:outline-none`.

5. **Setup step2_password.html** — Same label/input styling as step1. Convert `alert alert-info` to `border-l-4 border-info bg-base-300 p-3 flex items-center gap-3 font-mono text-sm`.

6. **Setup wizard_layout.html** — Convert `alert alert-error` to `border-l-4 border-error bg-base-300 p-3 flex items-center gap-3 font-mono text-sm` (bg-base-300 because it's inside a bg-base-200 panel).

7. **Setup step4_error.html** — Convert `alert alert-error` to `border-l-4 border-error bg-base-200 p-3 flex items-center gap-3 font-mono text-sm`. Keep SVG icon with `text-error`. Keep follow-up text.

8. **Setup step4_mapping_partial.html** — Replace `bg-amber-100 border-amber-400` with `bg-warning/15 border-warning` on undetected selects. Update legend swatch to `bg-warning/15 border border-warning`. Apply `font-mono uppercase text-xs tracking-wider text-base-content/50` to table headers.

9. **Prior-slice cleanup — login.html** — Convert `alert alert-error` (line ~38) to `border-l-4 border-error bg-base-300 p-3 flex items-center gap-3 font-mono text-sm` (bg-base-300 because it's inside the dark standalone panel).

10. **Prior-slice cleanup — runs/new.html + _pdf_error.html** — Convert `alert alert-error` in new.html to `border-l-4 border-error bg-base-200 p-3 flex items-center gap-3 font-mono text-sm`. Add industrial input styling (`bg-base-200 font-mono text-sm focus:border-primary focus:outline-none`) to all five text inputs. Style labels with `font-mono text-xs tracking-widest uppercase text-base-content/60`. Convert `_pdf_error.html` from `alert alert-error` to `border-l-4 border-error bg-base-200 p-3 flex items-center gap-3 font-mono text-sm`.

## Must-Haves

- [ ] Zero `alert alert-` classes in any template under `shop/templates/`
- [ ] Zero `card bg-base-100 shadow` patterns in any template under `shop/templates/`
- [ ] Zero `bg-amber-100` or `bg-opacity-20` patterns in any template
- [ ] Zero `badge-success` or `badge-error` in any template (status → text color)
- [ ] `{% block page_title %}` present in both admin templates
- [ ] All HTMX swap targets in users.html/users_row.html preserved verbatim
- [ ] `users_row.html` row styling matches `users.html` tbody rows exactly (badge strategy identical)
- [ ] `npm run build:css` → exit 0
- [ ] `uv run pytest tests/` → 93 passed, 0 failures

## Verification

- `grep -rn 'alert alert-\|card bg-base-100 shadow\|bg-amber-100\|bg-opacity-20' shop/templates/` → 0 matches
- `grep -rn 'badge-success\|badge-error' shop/templates/` → 0 matches
- `grep 'block page_title' shop/templates/admin/users.html shop/templates/admin/settings.html` → both present
- `grep 'id="users-table"' shop/templates/admin/users.html` → present
- `npm run build:css` → exit 0
- `uv run pytest tests/` → 93 passed, 2 xfailed, 0 failures
- Visual: admin/users and admin/settings render in browser with dark bordered sections, mono headers, no light-mode artifacts

## Inputs

- `shop/templates/auth/login.html` — canonical reference for industrial input styling and font-mono label pattern
- `shop/templates/runs/list.html` — canonical reference for table header/row styling
- `shop/templates/runs/_xlsx_mapping.html` — canonical reference for warning highlight pattern (`bg-warning/15 border-warning`)
- `shop/templates/runs/status.html` — canonical reference for `border-l-4` banner pattern
- S01–S04 summaries — established patterns and decisions (preloaded in context)

## Observability Impact

This task modifies only HTML/CSS class strings — no runtime logic, no new signals introduced. The relevant observability surface is:

- **Verification grep commands** are the primary inspection tool. After this task, `grep -rn 'alert alert-\|card bg-base-100 shadow\|bg-amber-100' shop/templates/` must return exit 1 (no matches). Non-zero output directly names the file needing attention.
- **CSS build output.** If a class like `bg-warning/15` is not recognized by Tailwind's config, `npm run build:css` will either silently drop it or warn. Zero-exit build confirms class resolution.
- **Jinja2 render errors** surface as 500 responses in the test suite — any `UndefinedError` from a missing context variable indicates a broken template block. Tests remain the fastest way to catch this.
- **No new failure state.** No new async paths, background processes, or API boundaries introduced. No logging changes needed.

## Expected Output

- 11 template files modified with industrial CSS classes — no structural HTML changes
- Zero legacy DaisyUI card/shadow/alert patterns remaining in the template tree
- CSS builds clean, all tests pass
- Admin pages visually confirmed in browser
