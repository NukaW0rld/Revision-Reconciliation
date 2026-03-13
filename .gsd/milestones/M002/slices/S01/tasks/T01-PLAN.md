---
estimated_steps: 7
estimated_files: 4
---

# T01: Strip wizard to 2 steps and update guard logic

**Slice:** S01 — Two-step wizard & per-run column mapping
**Milestone:** M002

## Description

Remove wizard steps 3 (engineer creation) and 4 (column mapping) from the setup flow. Setup completes after step 2 (admin password). The wizard progress bar drops from 4 steps to 2. Removed routes redirect gracefully instead of 404. Existing deployments with `wizard_step=4` are unaffected since they already have `setup_complete=True`.

## Steps

1. Read `shop/routers/setup.py` to confirm current step 2 POST logic, step 3/4 handlers, and `setup_root` cap
2. Modify `step2_post`: after `config.wizard_step = 2`, add `config.setup_complete = True` and change redirect from `/setup/step3` to `/login`
3. Remove route handlers: `step3_get`, `step3_post`, `step4_get`, `step4_upload`, `step4_save`
4. Add catch-all redirect routes for `/setup/step3` and `/setup/step4/{path:path}` — redirect to `/login` if `setup_complete`, else to `/setup/`
5. Update `setup_root`: change cap from `min(config.wizard_step + 1, 4)` to `min(config.wizard_step + 1, 2)`
6. Edit `shop/templates/setup/wizard_layout.html`: change steps list to `[(1,'Shop'), (2,'Password')]` and step count to "of 2"
7. Edit `shop/templates/setup/step2_password.html`: change submit button text to "Complete Setup"

## Must-Haves

- [ ] `step2_post` sets `setup_complete=True` and redirects to `/login`
- [ ] Step 3 and step 4 route handlers removed from `setup.py`
- [ ] `/setup/step3` and `/setup/step4/*` redirect (not 404)
- [ ] `setup_root` cap is 2 (not 4)
- [ ] Wizard progress bar shows 2 steps
- [ ] Step 2 button reads "Complete Setup"

## Verification

- `uv run pytest tests/test_setup.py::test_setup_step1_to_step2 -v` — passes (step 1→2 flow intact)
- `uv run pytest tests/test_setup.py::test_setup_step2_creates_admin -v` — passes (admin creation still works)
- Step 3/4 tests expected to fail at this point (repaired in T04)

## Inputs

- `shop/routers/setup.py` — current 4-step wizard with step2_post redirecting to step3
- `shop/templates/setup/wizard_layout.html` — progress bar with 4 steps
- `shop/templates/setup/step2_password.html` — current "Continue" button

## Expected Output

- `shop/routers/setup.py` — 2-step wizard; step 3/4 handlers removed; redirect routes added
- `shop/templates/setup/wizard_layout.html` — 2-step progress bar
- `shop/templates/setup/step2_password.html` — "Complete Setup" button
