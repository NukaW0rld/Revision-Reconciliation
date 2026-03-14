---
id: T02
parent: S01
milestone: M003
provides:
  - Industrial dark login page at shop/templates/auth/login.html extending _base_standalone.html
  - Industrial dark wizard layout at shop/templates/setup/wizard_layout.html extending _base_standalone.html
key_files:
  - shop/templates/auth/login.html
  - shop/templates/setup/wizard_layout.html
key_decisions:
  - Custom step progress indicator built with raw divs instead of DaisyUI `steps` component — DaisyUI steps uses light-mode background fill that didn't read cleanly against near-black; raw bordered divs with conditional bg-primary give full theme control
patterns_established:
  - Login page uses two-panel layout (left branding rail hidden on mobile, right form panel full-width); brand identity block uses `DELTA<span class="text-primary">.</span>PRES` + `font-mono` for aerospace feel
  - Wizard step indicator uses flex + connector line pattern (not ul.steps) for reliable dark-theme rendering; active step gets `border-primary text-primary`, completed step gets `bg-primary` fill with checkmark SVG
  - Error alerts on standalone pages use `alert alert-error font-mono` — consistent with dark theme, no card wrapper needed
observability_surfaces:
  - curl -s http://localhost:8000/login | grep data-theme → data-theme="industrial" confirms standalone shell active
  - curl -s -X POST http://localhost:8000/login -d "email=bad@x.com&password=wrong" | grep alert-error → confirms error state renders
  - curl -s http://localhost:8000/setup/step1 | grep data-theme → confirms wizard theme
  - curl -s http://localhost:8000/setup/step2 | grep border-primary → confirms step indicator rendered
duration: 30m
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T02: Redesign login screen and setup wizard with industrial aesthetic

**Rewrote login page and setup wizard layout to use the industrial dark theme via `_base_standalone.html`; both pages render with dark background, monospace aerospace typography, amber primary buttons, and no white card pattern.**

## What Happened

Both templates previously extended `base.html` (the sidebar layout shell), which meant they received sidebar chrome on unauthenticated routes. They also used the generic `card bg-base-100 shadow-xl` DaisyUI pattern — a white card on a white-ish background, which broke under the dark industrial theme.

**Login page (`login.html`):** Replaced with a full-viewport two-panel design. Left rail (hidden on mobile) shows branding context — AS9102 compliance label, FAIR Delta System identifier, and "Clearance: Quality Team Only" marks. Right panel holds the form: `DELTA.PRES` heading in `font-mono`, amber dot separator, monospace labels with `tracking-widest uppercase`, inputs styled with `bg-base-200` and `focus:border-primary`, and an `Authenticate` submit button in full-width amber. Error alerts render in `alert-error font-mono`.

**Wizard layout (`wizard_layout.html`):** Replaced with a centered single-column layout. Custom step progress indicator built as flex divs with a connector line — step nodes show `01`/`02` in bordered boxes, completed steps fill `bg-primary` with a checkmark SVG, active step uses `border-primary text-primary`. The panel itself is `bg-base-200 border border-base-300` with 0-radius edges (from theme vars). `{% block step_content %}` preserved exactly — sub-templates (`step1_shop_name.html`, `step2_password.html`) required zero changes.

One non-obvious choice: skipped DaisyUI's `ul.steps` component. It uses pseudo-element backgrounds that didn't render cleanly against the near-black base (`oklch(13% 0.005 260)`). Raw flex divs with explicit border/background classes give precise control over the theme vars.

## Verification

Must-have checks (all PASS):
- `login.html` extends `_base_standalone.html` ✓
- `font-mono` and `DELTA` heading present ✓
- No `card bg-base-100 shadow-xl` on login ✓
- Error alert block present ✓
- `wizard_layout.html` extends `_base_standalone.html` ✓
- Step progress indicator with `border-primary` ✓
- `{% block step_content %}` preserved ✓
- Both sub-templates unchanged and still extend `setup/wizard_layout.html` ✓

Runtime checks:
- `curl http://localhost:8000/login | grep data-theme` → `data-theme="industrial"`
- `curl http://localhost:8000/setup/step1 | grep data-theme` → `data-theme="industrial"`
- `curl http://localhost:8000/setup/step2 | grep data-theme` → `data-theme="industrial"`
- POST bad credentials → `alert alert-error` renders
- Test suite: **93 passed, 2 xfailed** — identical to T01 baseline

## Diagnostics

- `curl -s http://localhost:8000/login | grep data-theme` → `data-theme="industrial"` = correct
- `curl -s http://localhost:8000/setup/step2 | grep border-primary` → step indicator nodes visible
- Failure: `TemplateNotFound: _base_standalone.html` → surfaces as HTTP 500; means T01 output missing
- Failure: blank wizard form → means `{% block step_content %}` removed from `wizard_layout.html`
- Test isolation: `uv run pytest tests/test_auth.py` or `tests/test_setup.py` exercises these routes headlessly

## Deviations

Replaced DaisyUI `ul.steps` with custom flex progress indicator. The task plan said "style the DaisyUI steps component to look industrial under the dark theme." In practice, DaisyUI steps uses pseudo-element counter backgrounds that don't respond to `--color-base-*` vars cleanly at oklch(13%) darkness. Custom divs give identical visual semantics with full theme control.

## Known Issues

None.

## Files Created/Modified

- `shop/templates/auth/login.html` — rewritten: extends `_base_standalone.html`, two-panel industrial dark layout, no white card
- `shop/templates/setup/wizard_layout.html` — rewritten: extends `_base_standalone.html`, custom step indicator, dark panel
- `.gsd/milestones/M003/slices/S01/tasks/T02-PLAN.md` — added `## Observability Impact` section per pre-flight requirement
