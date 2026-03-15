# S03: Pipeline Status & Run Detail

**Goal:** The run status page renders in the industrial dark-mode aesthetic with all status states working — SSE stage checklist updates live, terminal status banners match the S02 `border-l-4` pattern in both Jinja2 and JS-injected HTML.
**Demo:** Navigate to a run's status page → dark bordered sections replace card/shadow wrappers, `font-mono` headings, all status-state banners use left-accent style, DaisyUI `steps-vertical` renders correctly against `bg-base-200`, SSE JS continues to function.

## Must-Haves

- All `card bg-base-100 shadow-sm` + `card-body` + `card-title` patterns replaced with `border border-base-300 bg-base-200 p-4/p-5` wrappers + `text-base font-semibold font-mono mb-3` headings
- `{% block page_title %}Run #{{ run.id }}{% endblock %}` present
- Status banners (failed/warning/completed/signed-off) use `border-l-4 border-{color} bg-base-200` — not `alert alert-{type}`
- SSE JS `injectTerminalBanner()` strings updated to match the same `border-l-4` pattern
- `id="stage-checklist"` and `id="terminal-banner-slot"` preserved exactly
- DaisyUI `steps steps-vertical` + `step-success/primary/error` retained in both `_stage_checklist.html` and the JS `updateChecklist()`
- DaisyUI modal (`modal-box`, `modal-backdrop`) untouched
- `npm run build:css` clean
- `uv run pytest tests/` passes (93 passed baseline)

## Verification

- `npm run build:css` → exit 0
- `uv run pytest tests/` → 93 passed, 2 xfailed, 0 failures
- `grep -c 'card bg-base-100 shadow\|card-body\|card-title' shop/templates/runs/status.html` → 0
- `grep -c 'alert alert-' shop/templates/runs/status.html` → 0 (all `alert alert-*` replaced with `border-l-4`)
- `grep 'id="stage-checklist"' shop/templates/runs/_stage_checklist.html` → present
- `grep 'id="terminal-banner-slot"' shop/templates/runs/status.html` → present
- `grep 'block page_title' shop/templates/runs/status.html` → present

## Tasks

- [x] **T01: Redesign status.html and _stage_checklist.html with industrial aesthetic** `est:30m`
  - Why: Only task — applies the full CSS-class redesign to the two S03-owned templates, including the coordinated Jinja2 + JS banner update
  - Files: `shop/templates/runs/status.html`, `shop/templates/runs/_stage_checklist.html`
  - Do: (1) Replace all three card/shadow wrappers with `border border-base-300 bg-base-200 p-4/p-5` + `font-mono` headings. (2) Add `{% block page_title %}`. (3) Replace `alert alert-error/warning/success` in Jinja2 status blocks with `border-l-4 border-error/warning/success bg-base-200 p-4 flex items-start gap-3`. (4) Update `injectTerminalBanner()` JS strings to match the same `border-l-4` pattern. (5) Leave `steps steps-vertical`, modal, and `status_badge_class` filter untouched. (6) Leave `_stage_checklist.html` structurally unchanged — only wrap the include in the new section wrapper in `status.html`. (7) Convert the Packet Versions card to bordered section.
  - Verify: `npm run build:css` clean; `uv run pytest tests/` passes; grep confirms no card/shadow or alert-alert remnants; DOM IDs preserved
  - Done when: zero `card bg-base-100 shadow` or `alert alert-` in status.html; all verification commands pass

## Observability / Diagnostics

**Runtime signals:**
- All status-state banners (failed/warning/completed/signed-off) render visually distinct left-accent strips — easy to spot regressions in browser.
- SSE `injectTerminalBanner()` injects the same markup into `#terminal-banner-slot`; devtools Elements panel shows the injected HTML in real time.
- `updateChecklist()` live-patches `#stage-checklist` innerHTML; check the element in devtools during an active run to confirm updates arrive.

**Inspection surfaces:**
- `grep -c 'card bg-base-100 shadow\|card-body\|card-title' shop/templates/runs/status.html` → 0 confirms clean removal.
- `grep -c 'alert alert-' shop/templates/runs/status.html` → 0 confirms no legacy DaisyUI alert classes survive in Jinja2 or JS strings.
- Browser hard-refresh on `/runs/<id>` for each terminal status (failed/warning/completed/signed_off) validates each Jinja2 branch independently.

**Failure visibility:**
- If `border-l-4` classes are missing from the Tailwind JIT scan, the banner renders unstyled — visually obvious and caught by `npm run build:css` + browser check.
- Modal (`modal-box`, `modal-backdrop`) is structurally untouched; if it regresses the Amend button on signed-off runs silently fails — covered by existing pytest suite.

**Redaction:**
- No secrets or PII flow through template rendering paths. Run ID, part number, and email are display data only.

## Files Likely Touched

- `shop/templates/runs/status.html`
- `shop/templates/runs/_stage_checklist.html`
