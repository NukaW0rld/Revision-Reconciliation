# T04: 03-review-and-sign-off 04

**Slice:** S03 — **Milestone:** M001

## Description

Sign-off gate enforcement and reviewer reassignment. The gate makes the sign-off button visible but disabled (with remaining count) until all items are resolved, then enables it behind a confirmation modal. Admin reassignment allows changing the assigned reviewer on any non-signed-off run.

Purpose: SIGNOFF-01 requires the gate to be enforced both in the UI and on the server. REVIEW-06 requires admin to be able to reassign. These are pure gating/UI concerns that don't touch the sign-off atomicity logic (Plan 05).
Output: Updated queue.html (sign-off section with modal), sign_off_confirm server-side gate check, reassign route on review router, reassign form on run status page (admin-only).

## Must-Haves

- [ ] "Sign-off button is disabled and shows remaining pending count when any item is pending"
- [ ] "Sign-off button is enabled when all items are resolved (pending == 0)"
- [ ] "Clicking sign-off shows a confirmation modal with run summary before posting"
- [ ] "Admin can reassign the reviewer on any run; engineer cannot access the reassign form"

## Files

- `shop/routers/review.py`
- `shop/templates/review/queue.html`
- `shop/routers/runs.py`
- `shop/templates/runs/status.html`
- `shop/services/review.py`
