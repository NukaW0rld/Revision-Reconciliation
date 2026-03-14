---
estimated_steps: 4
estimated_files: 2
---

# T02: Redesign login screen and setup wizard with industrial aesthetic

**Slice:** S01 — Design Foundation & Layout Shell
**Milestone:** M003

## Description

Rewrite the login page and setup wizard layout to use the industrial dark theme via `_base_standalone.html`. These are first-impression screens — login is the literal front door, and the setup wizard is the first-run experience. Both must feel like aerospace tooling software, not a generic DaisyUI card-on-white pattern. The step sub-templates (`step1_shop_name.html`, `step2_password.html`) extend `wizard_layout.html` via `{% block step_content %}` — they'll inherit the new look automatically once the layout is updated.

## Steps

1. Rewrite `shop/templates/auth/login.html` to extend `_base_standalone.html`. Design a full-viewport dark login panel: centered vertically, strong typography (app name as a prominent heading with monospaced or industrial feel), amber primary submit button, properly styled form inputs matching the dark theme. Include error alert styling. Remove the generic DaisyUI `card bg-base-100 shadow-xl` pattern.

2. Rewrite `shop/templates/setup/wizard_layout.html` to extend `_base_standalone.html`. Design a centered wizard panel with industrial step progress indicators, dark-themed form inputs, and amber primary buttons. The `{% block step_content %}` block must be preserved for step sub-templates. Style the DaisyUI `steps` component to look industrial under the dark theme (this should come free from the theme vars).

3. Start the app and verify login page renders at `/login` with the industrial aesthetic — dark background, no white card, proper form styling. Test error state by submitting bad credentials.

4. Verify setup wizard renders at `/setup/` (requires incomplete setup state — check if `step1_shop_name.html` and `step2_password.html` render correctly by reviewing their template structure, since they extend `wizard_layout.html` and use only `{% block step_content %}`).

## Must-Haves

- [ ] `login.html` extends `_base_standalone.html` (not `base.html`)
- [ ] Login page has dark background, industrial typography, amber primary button
- [ ] No generic `card bg-base-100 shadow-xl` pattern on login
- [ ] Error alerts render correctly on login page
- [ ] `wizard_layout.html` extends `_base_standalone.html` (not `base.html`)
- [ ] Wizard step progress indicator visible and styled for dark theme
- [ ] `{% block step_content %}` preserved in wizard layout
- [ ] Step sub-templates (`step1_shop_name.html`, `step2_password.html`) continue to render without changes

## Verification

- Login page at `http://localhost:8000/login` renders with dark industrial aesthetic
- Submitting bad credentials shows error alert styled in the dark theme
- Setup wizard pages render with dark theme and step progress indicators
- `step1_shop_name.html` and `step2_password.html` do not need changes (they inherit from wizard layout)

## Inputs

- `shop/templates/_base_standalone.html` — produced by T01
- `shop/templates/auth/login.html` — current generic DaisyUI login page
- `shop/templates/setup/wizard_layout.html` — current generic DaisyUI wizard layout
- `shop/templates/setup/step1_shop_name.html`, `step2_password.html` — step sub-templates (read-only, verify they still work)

## Expected Output

- `shop/templates/auth/login.html` — rewritten with industrial dark login design
- `shop/templates/setup/wizard_layout.html` — rewritten with industrial dark wizard design

## Observability Impact

**What changes:** Two templates that were previously rendered via `base.html` (which has no `data-theme` attribute of its own) now render via `_base_standalone.html`, which sets `data-theme="industrial"` on `<html>`. Any unauthenticated user hitting `/login` or `/setup/*` will now see the dark theme.

**How to inspect:**
- `curl -s http://localhost:8000/login | grep data-theme` → should return `data-theme="industrial"`
- `curl -s http://localhost:8000/setup/step1 | grep data-theme` → same
- `curl -s http://localhost:8000/setup/step2 | grep data-theme` → same
- Error state: `curl -s -X POST http://localhost:8000/login -d "email=bad@x.com&password=wrong" | grep alert-error` → confirms error block renders

**Failure shapes:**
- `TemplateNotFound: _base_standalone.html` → T01 output missing or template path wrong; surfaces as HTTP 500 on `/login`
- If `{% block step_content %}` is removed from `wizard_layout.html` → step sub-templates render blank panels (no form content); no crash, but page is empty below the progress indicator
- If `{% block content %}` is missing from `_base_standalone.html` → same failure as above but for both pages
