# S03: Review And Sign Off

**Goal:** Wave 1 prerequisite scaffold: test stubs, ReviewItem DB model, and pipeline patch for removed-item Rev B bboxes.
**Demo:** Wave 1 prerequisite scaffold: test stubs, ReviewItem DB model, and pipeline patch for removed-item Rev B bboxes.

## Must-Haves


## Tasks

- [x] **T01: 03-review-and-sign-off 01** `est:7min`
  - Wave 1 prerequisite scaffold: test stubs, ReviewItem DB model, and pipeline patch for removed-item Rev B bboxes.

Purpose: Nothing in plans 02-05 can be built without the DB model (no table = no query) or the removed-item bbox patch (no bbox = all "removed" cards show placeholder regardless of implementation). Test stubs guarantee pytest collection is green throughout the phase even before individual tests are implemented.
Output: tests/test_review.py (10 xfail stubs), expanded shop/models.py (ReviewItem + Run extensions), patched delta_preservation/cli.py (predicted Rev B bbox for removed items).
- [x] **T02: 03-review-and-sign-off 02** `est:5min`
  - Review queue GET route: opens the review queue for a completed run, populates ReviewItem rows from delta_packet.json (idempotent), and renders the scrollable list with filter dropdowns and item count header.

Purpose: This is the entry point engineers use after a pipeline run completes. It transitions Run.status from completed/warning to "reviewing" on first open and serves all subsequent views from the ReviewItem table.
Output: shop/routers/review.py, shop/services/review.py, shop/templates/review/queue.html, shop/templates/review/_progress_bar.html, review router registered in shop/app.py.
- [x] **T03: 03-review-and-sign-off 03**
  - Per-item approve/override HTMX actions, the review item card template, and the snippet file serving route.

Purpose: The review queue (Plan 02) renders a list of items. This plan makes each item interactive — approve and override are saved immediately via HTMX POST and the card swaps to its resolved state. The progress counter updates via OOB swap. Snippet images are served from the run output directory via a dedicated route. This plan also wires queue.html to replace its Plan 02 placeholder item loop with the real _item_card.html include.
Output: _item_card.html template (unresolved + resolved states), approve and override POST routes added to review.py, snippet GET route, modal overlay for full-size snippet viewing, queue.html updated to use real card partials.
- [x] **T04: 03-review-and-sign-off 04** `est:3min`
  - Sign-off gate enforcement and reviewer reassignment. The gate makes the sign-off button visible but disabled (with remaining count) until all items are resolved, then enables it behind a confirmation modal. Admin reassignment allows changing the assigned reviewer on any non-signed-off run.

Purpose: SIGNOFF-01 requires the gate to be enforced both in the UI and on the server. REVIEW-06 requires admin to be able to reassign. These are pure gating/UI concerns that don't touch the sign-off atomicity logic (Plan 05).
Output: Updated queue.html (sign-off section with modal), sign_off_confirm server-side gate check, reassign route on review router, reassign form on run status page (admin-only).
- [x] **T05: 03-review-and-sign-off 05** `est:3min`
  - Two-phase sign-off atomicity and the generating status page with SSE polling.

Purpose: SIGNOFF-02 requires that if sign-off fails partway through, the run rolls back to reviewable state — no signed-but-no-packet state can exist. SIGNOFF-03 requires immutability: once signed_off, the run cannot be re-signed. The generating page reuses the SSE pattern from runs/status.html to show the engineer the sign-off is in progress and redirect when complete.
Output: attempt_sign_off() with two-phase write and rollback, generating.html template, SSE route for sign-off status polling, immutability guard in sign_off_confirm route.
- [x] **T06: 03-review-and-sign-off 06** `est:human-verify + bug-fixes`
  - Phase 3 gate: complete all xfail test stubs to passing, run the full suite to confirm nothing regressed, and perform human verification of the review workflow in Docker.

Purpose: Plans 02-05 may have left some test stubs as xfail. This plan finishes implementing any remaining stubs, then the engineer verifies the complete workflow manually in the running Docker container.
Output: All 10 tests in test_review.py passing; Docker human verification approved.

## Files Likely Touched

- `tests/test_review.py`
- `shop/models.py`
- `delta_preservation/cli.py`
- `shop/routers/review.py`
- `shop/services/review.py`
- `shop/templates/review/queue.html`
- `shop/templates/review/_progress_bar.html`
- `shop/app.py`
- `shop/routers/review.py`
- `shop/templates/review/_item_card.html`
- `shop/templates/review/queue.html`
- `shop/routers/review.py`
- `shop/templates/review/queue.html`
- `shop/routers/runs.py`
- `shop/templates/runs/status.html`
- `shop/services/review.py`
- `shop/services/review.py`
- `shop/routers/review.py`
- `shop/templates/review/generating.html`
- `tests/test_review.py`
