# S05: Admin, Setup Wizard & Fragment Cleanup — Research

**Date:** 2026-03-14

## Summary

S05 is the terminal slice: it finishes every remaining template that S01–S04 didn't touch, plus picks up a handful of missed patterns from prior slices. The work is two-thirds CSS class substitution (low-risk, no structural changes) and one-third cleanup of inconsistencies that leaked through prior slice boundaries.

**Admin templates** (`admin/users.html`, `admin/settings.html`, `admin/users_row.html`) are the heaviest items — both page templates still carry generic card/shadow wrappers, DaisyUI `alert alert-*` boxes, and unstyled form inputs. The HTMX swap targets in users.html (`#users-table tbody`, `closest tr`) are preserved unchanged; the redesign only touches CSS classes.

**Setup wizard sub-templates** (`step2_password.html`, `step4_error.html`, `step4_mapping_partial.html`) are minor. The wizard layout shell (`wizard_layout.html`) was redesigned in S01 but still has one missed `alert alert-error` inside its panel; `step2_password.html` has one `alert-info` similarly missed. `step4_mapping_partial.html` is structurally identical to `_xlsx_mapping.html` (already fixed in S02) — apply the same `bg-warning/15 border-warning` swap. `step4_mapping_partial.html` appears to have no Python caller and is likely dead code, but it must still be updated for visual consistency.

**Prior-slice cleanup** — three templates outside S05's formal scope have confirmed legacy patterns that must be addressed:
- `auth/login.html` — one `alert alert-error` missed by S01 (inside the industrial standalone panel)
- `runs/new.html` — one `alert alert-error` missed by S02, plus form inputs using plain `input input-bordered` (no `bg-base-200 font-mono focus:border-primary` industrial styling)
- `runs/_pdf_error.html` — `alert alert-error` was intentionally left in S02 ("DaisyUI theme-aware, renders correctly"), but it produces a solid full-colored error box inside the bordered `bg-base-200` section on the new-run form — visually inconsistent. S05 should convert it to the `border-l-4 border-error` pattern.

The baseline is strong: `npm run build:css` → 215ms exit 0, 93 tests pass, all HTMX swap IDs intact from S04.

## Recommendation

**Single-task execution.** All remaining changes are CSS-class substitutions of a known, tested type. There is no structural HTML to invent — every pattern to apply already exists in prior-slice templates. Execute as one task (T01) covering all templates: admin screens, setup fragments, and three prior-slice cleanup items. Run `npm run build:css` and `pytest` after completion; do a visual spot-check of admin/users and admin/settings in the running browser.

**Badge classes in admin.** DaisyUI v5 badge-primary/neutral/success/error are fully theme-aware (confirmed via compiled CSS — they use `--color-primary`, `--color-neutral`, etc. from the industrial theme). The S04 `_progress_bar.html` replaced badge classes with font-mono text because badge-ghost was semantically wrong for counters; role/status badges in the admin user table are a legitimate semantic use and can remain as badges. Alternatively, follow S04 precedent and convert to font-mono spans. Either is fine — pick one and apply consistently to both `users.html` and `users_row.html`.

**Recommended badge treatment for admin:** Keep `badge-primary` (admin role) and `badge-neutral` (engineer role) — they're semantic and theme-aware. Replace `badge-success` / `badge-error` for active/inactive status with font-mono text using `text-success` / `text-error` to match the `_progress_bar.html` precedent. This is consistent: status = text color; role = badge.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Section wrappers | `border border-base-300 bg-base-200 p-4/p-5` (D010, S02) | Already in dashboard, runs list, new run, item card — use verbatim |
| Alert banners | `border-l-4 border-{color} bg-base-200 p-4 flex items-start gap-3` (D011/D012, S02/S03) | Used in status.html, alert_banner.html, item_card.html — same structure |
| Section headings | `text-base font-semibold font-mono mb-3` (S02 pattern) | All authenticated pages use this pattern |
| Industrial inputs | `input input-bordered bg-base-200 font-mono text-sm focus:border-primary focus:outline-none` (login.html S01) | Login established this; admin and new-run form inputs should match |
| Warning highlight in mapping table | `bg-warning/15 border-warning` (S02, `_xlsx_mapping.html`) | Identical columns needing identical treatment in `step4_mapping_partial.html` |
| Font-mono labels | `font-mono text-xs tracking-widest uppercase text-base-content/60` (login.html S01) | Use for form field labels in admin + step1/step2 sub-templates |
| Table headers | `font-mono uppercase text-xs tracking-wider text-base-content/50` (S02, runs/list.html) | Admin user table thead needs same treatment |

## Existing Code and Patterns

- `shop/templates/runs/_xlsx_mapping.html` — canonical reference for the column-mapping table pattern; `step4_mapping_partial.html` is structurally identical; just port the CSS classes
- `shop/templates/auth/login.html` — canonical reference for industrial input styling (`bg-base-200 font-mono focus:border-primary`) and label styling (`font-mono text-xs tracking-widest uppercase`)
- `shop/templates/runs/list.html` — canonical reference for plain table design: `font-mono uppercase text-xs tracking-wider text-base-content/50` thead, `hover:bg-base-300 transition-colors` tbody rows, no `table-zebra`
- `shop/templates/review/_progress_bar.html` — font-mono text-color pattern for status indicators (alternative to badge classes)
- `shop/templates/runs/status.html` — `border-l-4` banner pattern used in both Jinja2 and JS (template for admin/settings alert-error / alert-success banners)
- `shop/routers/admin.py` — confirms context vars: `users` list, `current_user`, `config`, `error`, `success`. **Does not call `_get_nav_context()`** — `base.html` `| default()` guards handle missing `shop_name`/`unread_alert_count` gracefully (verified in S01)
- `shop/routers/setup.py` — `step4` routes are all redirect-only; `step4_mapping_partial.html` has no Python renderer (confirmed dead code but update anyway)
- `shop/routers/runs.py:125` — renders `setup/step4_error.html` with `{"error": str(exc)}` context on invalid PDF upload

## Constraints

- **HTMX swap targets must not change** (D004): `id="users-table"`, `hx-target="#users-table tbody"` (beforeend swap for new user row), `hx-target="closest tr"` (outerHTML swap for deactivate). All three HTMX interactions in `users.html` / `users_row.html` must be preserved verbatim.
- **`form-control` and `label-text` produce no CSS in DaisyUI v5** — confirmed via compiled output.css (grep returns 0). These classes are layout-neutral divs/spans but label text is visually unstyled without explicit font classes. S01's login.html already solved this correctly: just add the font-mono uppercase classes to the `<span>`.
- **`step4_error.html` is rendered by `runs` router, not `setup` router** — it appears in `shop/routers/runs.py:125` as an HTMX partial response to invalid file upload. Its container is the `#revA-section` or `#revB-section` bordered div in `new.html`. Convert to `border-l-4 border-error` to match the section background.
- **`wizard_layout.html` alert block is inside the main panel div** (`bg-base-200 border border-base-300 p-8`) — the error alert renders inside a dark bordered panel; `alert alert-error` here produces a solid red-orange box inside the dark panel, which is visually jarring. Convert to `border-l-4 border-error bg-base-300 p-3 flex items-center gap-3 font-mono text-sm` to match the wizard's aesthetic.
- **`step2_password.html` has `alert-info`** (the admin email info box) — `alert-info` uses `--color-info` as full background; convert to `border-l-4 border-info bg-base-300` pattern.
- **`{% block page_title %}` needed in admin templates** — `base.html` renders it in the top bar. Both `admin/users.html` and `admin/settings.html` only have `{% block title %}` (browser tab title). Add `{% block page_title %}` blocks for top-bar context.

## Common Pitfalls

- **Skipping `users_row.html`** — it's an HTMX partial (rendered standalone by POST `/admin/users` and POST `/admin/users/{id}/deactivate`); it must be updated in tandem with `users.html` since both render the same `<tr>` structure. If `users.html` drops badge classes but `users_row.html` keeps them, newly-added rows will look different from initial rows.
- **`_pdf_error.html` alert-error "it works fine" trap** — S02 marked it intentionally left because "DaisyUI theme-aware renders correctly." This is technically true (dark error color), but the result is a solid colored box injected inside a dark bordered section — visually inconsistent with the border-l-4 pattern everywhere else. S05 should finish the job.
- **Admin router doesn't inject nav context** — `shop_name` and `unread_alert_count` are missing from admin router responses. The `| default()` guards in `base.html` handle this (verified S01). Do not add nav context injection to the admin router (out of scope per R015/R016).
- **`table-zebra` in admin users table** — currently not present (the original `table` class is plain). Don't add `table-zebra`; follow the S02 pattern of hover rows only.
- **`step4_mapping_partial.html` dead code** — it has no Python renderer. Still update it; a future route might add it back.
- **Badge consistency between `users.html` and `users_row.html`** — whichever badge strategy is chosen for the initial user list rows must be applied identically to the partial rows; inconsistent rows will be visible to users when new engineers are added or deactivated.

## Open Risks

- **Visual regression in runs/new.html form inputs** — S02 redesigned the section wrappers but left `form-control` / `input-bordered` inputs plain (no `bg-base-200 font-mono`). This is a real but minor visual inconsistency — the inputs will look slightly different from the login inputs. S05 should upgrade them, but it's a low-risk class addition.
- **Alert-info in step2_password.html** — there is no explicit `border-l-4 border-info` precedent established in prior slices (S02–S04 only handled error/warning/success). `--color-info` is in the industrial theme (teal accent used as `--color-accent`). The pattern still applies; just use `border-info` in the accent class.
- **step4_mapping_partial.html context variable `form_action`** — this partial uses a `{{ form_action }}` Jinja2 var that no current router injects. It's dead code. Don't change the template structure; just update CSS classes.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| Tailwind CSS v4 / DaisyUI v5 | none | no skill needed — patterns fully established in S01–S04 |

## Sources

- S01 Summary / Decisions D007–D009 — standalone base pattern, DaisyUI v5 theme definition, wizard step indicator (source: `.gsd/milestones/M003/slices/S01/S01-SUMMARY.md`)
- S02 Summary / Decisions D010–D011 — section wrapper pattern, alert banner pattern, badge treatment (source: `.gsd/milestones/M003/slices/S02/S02-SUMMARY.md`)
- S03 Summary / Decision D012 — border-l-4 unified in Jinja2 + JS (source: `.gsd/milestones/M003/slices/S03/S03-SUMMARY.md`)
- S04 Summary / Decisions D013–D015 — progress bar font-mono vs badge, override panel (source: `.gsd/milestones/M003/slices/S04/S04-SUMMARY.md`)
- DaisyUI v5 compiled CSS (`static/dist/output.css`) — confirmed `alert-error` uses `--color-error` as full background; `badge-primary/neutral/success/error` use semantic theme vars; `form-control`/`label-text` generate no CSS in v5
