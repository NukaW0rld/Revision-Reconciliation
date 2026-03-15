---
estimated_steps: 7
estimated_files: 2
---

# T01: Redesign status.html and _stage_checklist.html with industrial aesthetic

**Slice:** S03 — Pipeline Status & Run Detail
**Milestone:** M003

## Description

Apply the CSS-class-only redesign to the run status page and stage checklist partial. Replace all card/shadow wrappers with the canonical S02 `border border-base-300 bg-base-200` section pattern, swap `alert alert-{type}` banners for `border-l-4` left-accent pattern (in both Jinja2 and SSE JS strings), and add the `page_title` block. Preserve all load-bearing DOM IDs, DaisyUI `steps` component, and modal markup.

## Steps

1. Replace the three `card bg-base-100 shadow-sm` + `card-body` + `card-title` wrappers (Pipeline Stages section, Run Details section, Packet Versions card) with `border border-base-300 bg-base-200 p-4` section divs + `text-base font-semibold font-mono mb-3` headings. Ensure `mb-3` or `mb-2` replaces the implicit spacing that `card-body` provided.

2. Add `{% block page_title %}Run #{{ run.id }}{% endblock %}` after the `{% block title %}` line (or alongside it — must be a separate block).

3. Replace all Jinja2 status banner `alert alert-{type}` blocks with `border-l-4 border-{type} bg-base-200 p-4 flex items-start gap-3` pattern. This covers: (a) signed-off success banner at top, (b) failed state alert, (c) two warning state alerts, (d) the back-link + heading area gets `font-mono` treatment.

4. Update `injectTerminalBanner()` JS strings to use the same `border-l-4` classes instead of `alert alert-error/warning/success`. All three branches (failed, warning, completed) must match their Jinja2 counterparts exactly in class structure.

5. Leave untouched: `steps steps-vertical` in `_stage_checklist.html`, `updateChecklist()` JS, modal `dialog`/`modal-box`/`modal-backdrop`, `status_badge_class` filter, and the `{% include "runs/_stage_checklist.html" %}` relationship.

6. Style the button group area (back link, reassign form, download buttons) with `font-mono` accents on the heading and `btn-ghost` on the back link — consistent with S02 patterns.

7. Run `npm run build:css`, `uv run pytest tests/`, and grep verification checks.

## Must-Haves

- [ ] Zero `card bg-base-100 shadow` or `card-body` or `card-title` in `status.html`
- [ ] Zero `alert alert-` in `status.html` (both Jinja2 and JS)
- [ ] `id="stage-checklist"` present in `_stage_checklist.html`
- [ ] `id="terminal-banner-slot"` present in `status.html`
- [ ] `{% block page_title %}` present in `status.html`
- [ ] `steps steps-vertical` retained in `_stage_checklist.html`
- [ ] `modal-box` and `modal-backdrop` untouched in modal section
- [ ] `npm run build:css` exit 0
- [ ] `uv run pytest tests/` → 93 passed, 2 xfailed

## Verification

- `npm run build:css` → exit 0, no warnings
- `uv run pytest tests/` → 93 passed, 2 xfailed, 0 failures
- `grep -c 'card bg-base-100 shadow\|card-body\|card-title' shop/templates/runs/status.html` → 0
- `grep -c 'alert alert-' shop/templates/runs/status.html` → 0
- `grep 'id="stage-checklist"' shop/templates/runs/_stage_checklist.html` → match
- `grep 'id="terminal-banner-slot"' shop/templates/runs/status.html` → match
- `grep 'block page_title' shop/templates/runs/status.html` → match

## Inputs

- `shop/templates/runs/status.html` — current 287-line template with card wrappers, alert banners, SSE JS
- `shop/templates/runs/_stage_checklist.html` — 22-line partial with DaisyUI steps
- S01 summary: `base.html` provides sidebar + top-bar with `page_title` slot; theme vars compiled
- S02 summary: canonical section wrapper pattern is `border border-base-300 bg-base-200 p-4/p-5`; heading pattern is `text-base font-semibold font-mono mb-3`; alert banner pattern is `border-l-4 border-error bg-base-200 p-3 flex items-center gap-3`

## Observability Impact

**What signals change:**
- `#terminal-banner-slot` receives `border-l-4` HTML instead of `alert alert-*` HTML via `injectTerminalBanner()`. Inspect in devtools Elements after SSE close event fires to verify class structure.
- All three Jinja2 banner branches (failed/warning/completed-top) now use `border-l-4 border-{color}` wrapper — rendered output observable by loading each run state in browser.

**How a future agent inspects this task:**
- `grep -c 'alert alert-' shop/templates/runs/status.html` → must be 0; any non-zero count means reversion or incomplete migration.
- `grep -c 'card bg-base-100 shadow\|card-body\|card-title' shop/templates/runs/status.html` → must be 0.
- `grep 'block page_title' shop/templates/runs/status.html` → must match.
- `grep 'border-l-4' shop/templates/runs/status.html` → should show multiple matches (one per status-state banner).

**Failure state visibility:**
- Missing `border-l-4` utility from Tailwind output → banner div has no left border → visually unstyled but functional; caught by visual browser check and `npm run build:css`.
- If JS banner strings still use `alert alert-error`, the injected terminal banner won't match Jinja2 rendered states — asymmetry visible by comparing SSE-triggered state vs. hard-reload state in devtools.
- Modal regression (if modal markup touched accidentally) → Amend button click does nothing; covered by pytest `test_review.py::test_amend_*` tests.

## Expected Output

- `shop/templates/runs/status.html` — all card/shadow patterns replaced with bordered sections; all `alert alert-*` replaced with `border-l-4` in both Jinja2 and JS; `page_title` block added; all DOM IDs preserved
- `shop/templates/runs/_stage_checklist.html` — no changes expected (already correct with `steps-vertical`)
