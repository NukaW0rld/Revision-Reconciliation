# T05: 01-foundation 05

**Slice:** S01 — **Milestone:** M001

## Description

Build the setup wizard middleware guard and steps 1-3: shop name, admin password change, and first engineer account creation. After this plan, the undismissable setup guard is in effect and the first three wizard steps are completable.

Purpose: SETUP-01 (undismissable wizard) is one of the most critical Phase 1 requirements — the system must not allow any authenticated operation until configuration is complete.
Output: Working wizard middleware + steps 1-3 with step-ordering enforcement and mid-wizard resume.

## Must-Haves

- [ ] "Any request to / or /dashboard when setup_complete=False redirects to /setup/step1"
- [ ] "Admin cannot skip from step 1 to step 3 by direct URL navigation"
- [ ] "Step 1 saves shop name to shop_config; step 2 changes admin password and blocks reuse of default; step 3 creates first engineer account"
- [ ] "Mid-wizard resume: closing browser and revisiting /setup redirects to the first incomplete step"
- [ ] "Step 4 is not accessible until steps 1-3 are complete (wizard_step < 3 redirects back)"
- [ ] "Setup middleware exempts /setup/*, /static/*, /login, /logout from the redirect"

## Files

- `shop/routers/setup.py`
- `shop/templates/setup/step1_shop_name.html`
- `shop/templates/setup/step2_password.html`
- `shop/templates/setup/step3_engineer.html`
- `shop/templates/setup/wizard_layout.html`
