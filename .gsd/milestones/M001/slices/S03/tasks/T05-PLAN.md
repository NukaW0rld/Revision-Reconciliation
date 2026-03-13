# T05: 03-review-and-sign-off 05

**Slice:** S03 — **Milestone:** M001

## Description

Two-phase sign-off atomicity and the generating status page with SSE polling.

Purpose: SIGNOFF-02 requires that if sign-off fails partway through, the run rolls back to reviewable state — no signed-but-no-packet state can exist. SIGNOFF-03 requires immutability: once signed_off, the run cannot be re-signed. The generating page reuses the SSE pattern from runs/status.html to show the engineer the sign-off is in progress and redirect when complete.
Output: attempt_sign_off() with two-phase write and rollback, generating.html template, SSE route for sign-off status polling, immutability guard in sign_off_confirm route.

## Must-Haves

- [ ] "If attempt_sign_off raises an exception after signing_off transition, Run.status rolls back to 'reviewing'"
- [ ] "No run can be in signed_off state without signed_at timestamp"
- [ ] "A second sign-off attempt on a signed_off run is blocked (redirected to run summary, not re-signed)"
- [ ] "GET /review/{run_id}/generating serves an SSE-polling status page that redirects on terminal state"
- [ ] "SSE endpoint emits status updates; when status is signed_off, client redirects to run summary with success; when reviewing (rollback), client redirects to queue with error"

## Files

- `shop/services/review.py`
- `shop/routers/review.py`
- `shop/templates/review/generating.html`
