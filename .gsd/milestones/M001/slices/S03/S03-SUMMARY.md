---
id: S03
parent: M001
milestone: M001
provides:
  - ReviewItem SQLAlchemy model with full review/override fields
  - Run.review_items relationship, signed_at, signed_by_id columns
  - status_badge_class handles reviewing/signing_off/signed_off
  - Pipeline emits predicted Rev B bbox for removed DeltaItems via homography
  - 10 xfail stubs for REVIEW-01..07 and SIGNOFF-01..03
  - open_review_queue() service: idempotent ReviewItem population from delta_packet.json
  - GET /review/{run_id} route with filter params
  - Review queue page (queue.html) with filter dropdowns and item list
  - Progress bar partial (_progress_bar.html) as HTMX OOB swap target
  - Review router registered in app.py
  - POST /review/{run_id}/sign-off/confirm route with pending-count gate
  - POST /review/{run_id}/reassign route (admin-only)
  - attempt_sign_off stub in services/review.py (Plan 05 replaces)
  - Sign-off sticky footer with disabled/enabled button in queue.html
  - DaisyUI confirmation modal before final POST in queue.html
  - Admin-only reassign form on run status page
  - "attempt_sign_off() two-phase write with automatic rollback to reviewing on failure"
  - "SIGNOFF-03 immutability guard: signed_off runs cannot be re-signed"
  - "GET /review/{run_id}/generating — SSE-polling status page with terminal redirect"
  - "GET /review/{run_id}/sign-off/sse — SSE endpoint streaming sign-off status"
  - "?signed=1 success banner on runs/{run_id} status page"
  - All 10 REVIEW-* and SIGNOFF-* integration tests passing (not xfail)
  - Full pytest suite green (70 passed, 2 xfailed, 3 xpassed)
  - Human-verified end-to-end review workflow in Docker
  - Bug fixes: run_id in template context, confidence badge label, stage checklist all-stages-done for post-pipeline statuses, HTMX OOB sign-off footer
requires: []
affects: []
key_files: []
key_decisions:
  - "foreign_keys=[reviewer_id] added to Run.reviewer and User.runs relationships to resolve AmbiguousForeignKeysError when Run gained second FK to users (signed_by_id)"
  - "apply_transform_bbox import kept inside try block in cli.py to avoid module-level failure if alignment module changes"
  - "Removed-item revB bbox uses homography transform of revA_bbox_pdf; falls back to None if exception (review card shows placeholder)"
  - "JS onchange form submit chosen for filter dropdowns to avoid HTMX dependency for filter state — browser POST/GET is sufficient for full-page filter refresh"
  - "Counts (pending/approved/overridden) computed from all_items before filter is applied — sign-off gate needs unfiltered totals"
  - "open_review_queue uses existing_count check (not get_or_create per item) — simpler and avoids partial-insert race conditions"
  - "attempt_sign_off imported directly (not inline) from services.review — Plan 05 replaces stub with two-phase write"
  - "Sign-off modal uses standard HTML POST (not HTMX) per Phase 01 decision; DaisyUI showModal() for open trigger"
  - "Reassign form passes engineers list from runs.py run_status() — only active engineers with role=engineer are shown"
  - "Two-phase write rollback uses db.rollback() then re-queries run by ID because the session object may be in detached state post-rollback; not all SQLAlchemy backends refresh cleanly after rollback on the same object"
  - "SSE terminal set is {signed_off, reviewing} — reviewing is the rollback state; client redirects to /review/{id}?error=sign_off_failed on rollback"
  - "generating.html uses htmx:sseMessage DOM event (not hx-on:sse:status_update) to handle redirect logic in JavaScript — keeps redirect logic explicit and testable"
  - "signed=True passed from run_status() to status.html; banner displayed only when ?signed=1 query param is present"
  - "run_id passed explicitly to review_queue() TemplateResponse — Jinja2 silently interpolates missing variables as empty string, producing broken HTMX action URLs"
  - "Confidence badge prefixed with 'Confidence: ' label directly in _item_card.html — no JavaScript needed"
  - "Stage checklist is_done extended to all post-pipeline statuses (reviewing/signing_off/signed_off/warning) — any status after pipeline completion should show all 8 stages green"
  - "Dual-partial OOB pattern for signoff footer avoids double-id nesting: _signoff_footer.html (no hx-swap-oob) for initial queue.html render, _signoff_footer_oob.html (hx-swap-oob=outerHTML) for HTMX response"
  - "run object passed explicitly to approve/override TemplateResponse so signoff footer OOB can evaluate run.status for button enable/disable"
patterns_established:
  - "Scaffold stubs: xfail(strict=False) + raise NotImplementedError keeps collection green across plans 02-05"
  - "Service functions take (db, run) not run_id — caller owns DB session, service uses it"
  - "Template filter dropdowns submit via JS onchange on <select> element inside <form method=GET>"
  - "Server-side gate always checks pending count independently of UI state — UI disabled state is UX only, not a security boundary"
  - "Admin-only forms in templates use user.role == 'admin' conditional rendered server-side"
  - "SSE pattern (db.expire + db.refresh in polling loop) reused directly from Phase 02 run_sse"
  - "Template context audit: verify every {{ variable }} in modified templates has a corresponding key in the TemplateResponse context dict"
  - "OOB swap with outerHTML: the OOB element IS the replacement content including its root tag; use dual partials to avoid double-id nesting"
observability_surfaces: []
drill_down_paths: []
duration: human-verify + bug-fixes
verification_result: passed
completed_at: 2026-03-07
blocker_discovered: false
---
# S03: Review And Sign Off

**# Phase 3 Plan 01: Wave 1 Scaffold Summary**

## What Happened

# Phase 3 Plan 01: Wave 1 Scaffold Summary

**SQLAlchemy ReviewItem model with run/sign-off extensions, pipeline removed-item bbox prediction via homography, and 10 xfail pytest stubs covering all review and sign-off requirements**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-03-07T20:14:10Z
- **Completed:** 2026-03-07T20:20:57Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- ReviewItem table with pipeline_classification, confidence, bbox (JSON), reviewer_decision, override fields — ready for Phase 3 plans 02-05 to query and write
- Run extended with review_items (cascade delete-orphan), signed_at, signed_by_id; status comment updated with reviewing/signing_off/signed_off values
- Pipeline now emits a predicted Rev B bbox for removed DeltaItems using the estimated homography transform so review cards can show the expected location in Rev B
- 10 xfail stubs in tests/test_review.py keep pytest collection green throughout the phase; full suite: 60 passed, 12 xfailed, 3 xpassed, 0 failures

## Task Commits

Each task was committed atomically:

1. **Task 1: ReviewItem DB model + Run extensions** - `a75b524` (feat)
2. **Task 2: Pipeline patch — predicted Rev B bbox for removed items** - `fb33d6d` (feat)
3. **Task 3: Test stubs for all 10 review requirements** - `fe6b7f9` (test)

## Files Created/Modified

- `shop/models.py` — Added Float import, ReviewItem class, review_items/signed_at/signed_by_id on Run, foreign_keys fix on reviewer relationship
- `shop/app.py` — Extended _status_badge_class with reviewing/signing_off/signed_off entries
- `delta_preservation/cli.py` — Added elif branch for removed items: revB_bbox_pdf = apply_transform_bbox(revA_bbox_pdf, transform.H)
- `tests/test_review.py` — Created with 10 xfail stubs (REVIEW-01..07, SIGNOFF-01..03)

## Decisions Made

- `foreign_keys=[reviewer_id]` added to Run.reviewer and User.runs relationships — SQLAlchemy raises AmbiguousForeignKeysError when a model has two FKs to the same table; specifying foreign_keys disambiguates the join
- apply_transform_bbox import kept inside the try block in cli.py — avoids module-level import failure if the alignment module's public API changes; failure falls back to revB_bbox_pdf = None (acceptable: review card shows "Image unavailable" placeholder)
- Removed-item bbox uses forward homography (revA → revB) not the full snippet pipeline — no snippet is cropped at that location since the characteristic was not found; image_path will be None and the review card handles that gracefully

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added foreign_keys to User.runs and Run.reviewer relationships**
- **Found during:** Task 1 (ReviewItem DB model + Run extensions)
- **Issue:** Adding signed_by_id (second FK to users.id on Run) caused SQLAlchemy AmbiguousForeignKeysError when instantiating any model because User.runs relationship could not determine which FK to join on
- **Fix:** Added `foreign_keys="Run.reviewer_id"` to User.runs and `foreign_keys=[reviewer_id]` to Run.reviewer
- **Files modified:** shop/models.py
- **Verification:** `uv run python -c "from shop.models import ReviewItem, Run; r = Run(); print(hasattr(r, 'review_items'), hasattr(r, 'signed_at'))"` prints `True True`
- **Committed in:** a75b524 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Required fix — without it no model was importable. No scope creep.

## Issues Encountered

None beyond the FK ambiguity fixed above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- ReviewItem table and Run extensions ready; conftest.py `Base.metadata.create_all` will create the new table automatically in all tests
- Pipeline now provides revB bbox for removed items; Plan 02 review card can render predicted location
- 10 xfail stubs in place; Plan 02 will implement REVIEW-01, REVIEW-02, REVIEW-07 first

---
*Phase: 03-review-and-sign-off*
*Completed: 2026-03-07*

# Phase 03 Plan 02: Review Queue GET Route Summary

**Review queue entry point: idempotent ReviewItem population from delta_packet.json via open_review_queue(), with GET /review/{run_id} route, filter dropdowns, and HTMX-ready progress bar partial**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-07T20:23:32Z
- **Completed:** 2026-03-07T20:28:20Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- `open_review_queue(db, run)` service creates ReviewItems from delta_packet.json on first call; subsequent calls return existing rows without duplicating
- Run.status transitions from completed/warning to "reviewing" on first queue open
- GET /review/{run_id} renders the queue page with all items, filter dropdowns (review status + pipeline classification), and pending/approved/overridden counts in header
- Sign-off button disabled when pending > 0; enabled when all items resolved
- Progress bar as HTMX OOB swap target ready for Plan 03 approve/override responses

## Task Commits

Each task was committed atomically:

1. **TDD RED - test_review_state_persisted** - `7373e5c` (test)
2. **Task 1: open_review_queue service** - `d1657ff` (feat)
3. **Task 2: review router + queue template** - `2e64438` (feat)

## Files Created/Modified

- `shop/services/review.py` - open_review_queue() idempotent service
- `shop/routers/review.py` - GET /review/{run_id} with filter params
- `shop/templates/review/queue.html` - Full review queue page extending base.html
- `shop/templates/review/_progress_bar.html` - Pending/approved/overridden count partial
- `shop/app.py` - Added review router registration at /review prefix
- `tests/test_review.py` - Implemented test_review_queue_loads, test_review_state_persisted, test_review_counts

## Decisions Made

- JS onchange form submit chosen for filter dropdowns (not HTMX) to avoid HTMX dependency for filter state — browser GET is sufficient for full-page filter refresh
- Counts computed from all_items before filter is applied so sign-off gate always sees accurate totals
- open_review_queue uses a single count check for idempotency rather than per-item get_or_create — simpler and avoids partial-insert edge cases

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - the circular import warning in the plan's `python -c "from shop.routers.review import router"` verification command is a pre-existing pattern in this codebase (same for runs.py, auth.py). All routers import `from shop.app import templates` which triggers the module-level `app = create_app()`, which in turn imports all routers. This only fails in direct isolated import; the app works correctly via `create_app()` and all tests pass.

## Self-Check

- [x] `shop/services/review.py` exists
- [x] `shop/routers/review.py` exists
- [x] `shop/templates/review/queue.html` exists
- [x] `shop/templates/review/_progress_bar.html` exists
- [x] Commits 7373e5c, d1657ff, 2e64438 exist
- [x] 63 passed, 9 xfailed in full suite

## Self-Check: PASSED

## Next Phase Readiness

- Review queue route and service fully functional
- Plan 03 can build approve/override HTMX endpoints — _progress_bar.html OOB swap target is ready
- Plan 04 sign-off section placeholder is in queue.html ready to be wired up

---
*Phase: 03-review-and-sign-off*
*Completed: 2026-03-07*

# Phase 3 Plan 03: Approve/Override HTMX Actions and Item Card Template Summary

**One-liner:** Per-item approve and override POST routes with HTMX OOB progress-bar swap, `_item_card.html` partial (unresolved + resolved states), snippet file-serving route, and queue.html wired to real card includes.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Approve and override POST routes | b8013df | shop/routers/review.py, shop/templates/review/_item_card.html, tests/test_review.py |
| 2 | Review item card template and queue wiring | 851c59f | shop/app.py, shop/templates/review/queue.html, tests/test_review.py |

## What Was Built

### Task 1: Approve and Override POST Routes

Added three routes to `shop/routers/review.py`:

- `POST /{run_id}/items/{char_no}/approve` — Sets `reviewer_decision='approved'`, stamps `reviewed_by_id` and `reviewed_at`, returns `_item_card.html` with `oob_update=True` to update progress bar.
- `POST /{run_id}/items/{char_no}/override` — Validates `override_note` is non-empty (422 with error message in card HTML if empty), sets `reviewer_decision='overridden'`, saves `override_classification` and `override_note`.
- `GET /{run_id}/snippets/{filename}` — Serves PNG snippets from `run.output_dir/snippets/`; validates filename against path traversal (rejects `..`, `/`, non-`.png`); returns `FileResponse`.

Both POST routes return the updated `_item_card.html` partial plus an OOB `#progress-bar` div with recalculated pending/approved/overridden counts.

### Task 2: Item Card Template and Queue Wiring

- Created `shop/templates/review/_item_card.html` — Two-state card (unresolved shows Approve form + Override collapsible, resolved shows status badge). Snippet images are clickable and open in native `<dialog>` modals. Missing snippets show "Image unavailable" placeholder; removed items with no revB bbox show "Not found in Rev B".
- Added `basename` Jinja2 filter to `shop/app.py` templates env to extract filename from snippet path.
- Updated `shop/templates/review/queue.html` to replace the placeholder div loop with `{% include "review/_item_card.html" %}`, passing `run_id` already in the template context.

## Verification

All specified tests pass:
- `test_approve_item` — POST approve sets `reviewer_decision='approved'` in DB; returns 200
- `test_override_requires_note` — Empty note → 422; valid note → 200 with `reviewer_decision='overridden'`
- `test_review_item_card_html` — Resolved card contains "Approved" badge, `border-success`, `hx-swap-oob`
- `test_review_counts` — Queue page shows correct pending/approved/overridden counts

Full suite: 66 passed, 6 xfailed, 3 xpassed.

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written, with one minor addition:

**[Rule 1 - Refactor] Extracted _item_counts() helper**
- **Found during:** Task 1 implementation
- **Issue:** approve_item and override_item share identical count-query logic (3 lines each)
- **Fix:** Extracted `_item_counts(db, run_id)` helper returning `(all_items, pending, approved, overridden)` to eliminate duplication
- **Files modified:** shop/routers/review.py

## Self-Check: PASSED

- FOUND: shop/templates/review/_item_card.html
- FOUND: shop/routers/review.py (with approve, override, snippet routes)
- FOUND: shop/templates/review/queue.html (uses {% include "review/_item_card.html" %})
- FOUND commit b8013df (Task 1)
- FOUND commit 851c59f (Task 2)

# Phase 03 Plan 04: Sign-Off Gate and Reviewer Reassignment Summary

**Server-enforced sign-off gate with DaisyUI confirmation modal, pending-count disabled button, and admin-only reviewer reassign via form select**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-07T20:36:14Z
- **Completed:** 2026-03-07T20:39:35Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Sign-off gate POST route checks pending items and redirects with error if any remain; succeeds by calling attempt_sign_off stub
- Admin-only POST /reassign route validates role, run status, and reviewer validity before updating Run.reviewer_id
- queue.html upgraded from static card to sticky footer with disabled/enabled button and full DaisyUI confirmation modal
- Admin reassign select form rendered conditionally on status.html; engineers list passed from run_status() route

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests for sign-off gate and admin reassign** - `24c89d7` (test)
2. **Task 1 (GREEN): Sign-off gate route and admin reassign route** - `c151629` (feat)
3. **Task 2: Sign-off UI in queue.html and reassign form in status.html** - `26a911f` (feat)

**Plan metadata:** (docs commit below)

_Note: TDD task has two commits (test RED, then feat GREEN)_

## Files Created/Modified
- `shop/routers/review.py` - Added sign_off_confirm and reassign_run routes; imported attempt_sign_off
- `shop/services/review.py` - Added attempt_sign_off stub (sets signed_at, signed_by_id, status=signed_off)
- `shop/routers/runs.py` - run_status() now passes engineers list to template
- `shop/templates/review/queue.html` - Sticky sign-off footer with enabled/disabled button + DaisyUI modal
- `shop/templates/runs/status.html` - Admin-only reassign form below reviewer field
- `tests/test_review.py` - Implemented test_sign_off_gate and test_admin_can_reassign (replacing xfail stubs)

## Decisions Made
- `attempt_sign_off` imported at module level from services.review (not inline import), making it a clear seam for Plan 05 to replace
- Confirmation modal uses `showModal()` JS on button click then standard HTML form POST — matches Phase 01 decision to avoid HTMX for full-page actions
- Reassign form shows only active engineers (role=engineer, is_active=True) — admin cannot reassign to another admin via this form

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Plan 05 can replace the `attempt_sign_off` stub with the two-phase write atomicity implementation
- The `/review/{run_id}/generating` redirect target is now reachable from the confirmed sign-off flow

## Self-Check: PASSED
- `shop/routers/review.py` - EXISTS (sign_off_confirm, reassign_run routes present)
- `shop/services/review.py` - EXISTS (attempt_sign_off stub present)
- `shop/templates/review/queue.html` - EXISTS (145 lines, sticky footer + modal present)
- `shop/templates/runs/status.html` - EXISTS (admin reassign form present)
- Commits: 24c89d7, c151629, 26a911f all exist in git log

---
*Phase: 03-review-and-sign-off*
*Completed: 2026-03-07*

# Phase 3 Plan 05: Sign-Off Atomicity and Generating Status Page Summary

**Two-phase write attempt_sign_off() with automatic rollback to reviewing, plus SSE-polling generating status page that redirects engineers on sign-off completion or failure**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-07T20:42:01Z
- **Completed:** 2026-03-07T20:45:18Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Replaced stub `attempt_sign_off()` with two-phase write: Phase 1 commits `signing_off` (visible to SSE); Phase 2 commits `signed_off` + `signed_at` + `signed_by_id`; on Phase 2 exception, rolls back and sets `reviewing`
- Added SIGNOFF-03 immutability guard: returns `False` immediately without modification if `run.status == "signed_off"`
- Created `/review/{run_id}/generating` route and template with SSE listener that redirects to `/runs/{id}?signed=1` on success or `/review/{id}?error=sign_off_failed` on rollback
- Added `/review/{run_id}/sign-off/sse` SSE endpoint using `db.expire` + `db.refresh` polling loop (terminal states: `signed_off`, `reviewing`)
- Added success banner to `runs/status.html` shown when `?signed=1` query param is present

## Task Commits

Each task was committed atomically:

1. **TDD RED - SIGNOFF-02/03 tests** - `79cb7e8` (test)
2. **TDD GREEN - attempt_sign_off two-phase write** - `2e236e5` (feat)
3. **Task 2: Generating page + SSE route + signed banner** - `6072475` (feat)

## Files Created/Modified

- `shop/services/review.py` - attempt_sign_off() replaced with two-phase write and immutability guard
- `shop/routers/review.py` - added imports (asyncio, json, SSE), GET /generating route, GET /sign-off/sse route
- `shop/templates/review/generating.html` - new SSE-polling status page, extends base.html
- `shop/routers/runs.py` - run_status() now passes signed=True when ?signed=1 present
- `shop/templates/runs/status.html` - success alert banner shown when signed=True
- `tests/test_review.py` - replaced xfail stubs with real test implementations for SIGNOFF-02 and SIGNOFF-03

## Decisions Made

- Two-phase write rollback re-queries run by ID after `db.rollback()` because session identity-map state is unreliable post-rollback; same pattern established in Phase 02 for SSE polling
- SSE terminal set includes `reviewing` (rollback state) so the client never polls indefinitely; it always receives a redirect signal whether sign-off succeeded or failed
- `htmx:sseMessage` DOM event used in generating.html JavaScript (not inline hx-on attribute) to keep redirect logic explicit and debuggable

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- Sign-off atomicity complete; any Phase 2 exception during PDF/packet generation (Phase 4) will cleanly roll back to reviewing state
- WeasyPrint PDF generation (Phase 4 PACKET-01) can now write inside Phase 2 of `attempt_sign_off()` knowing the rollback mechanism is in place
- SSE pattern for sign-off generating page mirrors the pipeline status page pattern — no new concepts needed

## Self-Check: PASSED

- shop/services/review.py: FOUND
- shop/routers/review.py: FOUND
- shop/templates/review/generating.html: FOUND
- 03-05-SUMMARY.md: FOUND
- Commit 79cb7e8 (test RED): FOUND
- Commit 2e236e5 (feat GREEN): FOUND
- Commit 6072475 (feat Task 2): FOUND

---
*Phase: 03-review-and-sign-off*
*Completed: 2026-03-07*

# Phase 3 Plan 06: Phase Gate — Review and Sign-Off Summary

**All 10 REVIEW-*/SIGNOFF-* integration tests passing; Docker e2e verification approved after four bug fixes including dynamic HTMX sign-off footer and all-stages-green stage checklist**

## Performance

- **Duration:** human-verify gate + ~20 min bug fixes
- **Started:** 2026-03-07
- **Completed:** 2026-03-07
- **Tasks:** 2 (Task 1 already done from 03-02 to 03-05; Task 2 human verification produced 4 total bug fixes across two rounds)
- **Files modified:** 6

## Accomplishments

- All 10 tests in `tests/test_review.py` passing (no xfail, no skip)
- Full test suite green: 70 passed, 2 xfailed, 3 xpassed
- Human Docker verification (round 1) revealed 2 bugs fixed in commit 01e2b88
- Human Docker verification (round 2) revealed 2 more bugs; fixed in b5b0141 and 2e0a9bc
- Phase 3 gate fully passed — review and sign-off workflow is complete and verified end-to-end

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement remaining xfail stubs to passing** — already complete from prior plans (Plans 03-02 through 03-05)
2. **Task 2 round 1: Human verification bug fixes** — `01e2b88` (fix: pass run_id to queue template context; label confidence badge)
3. **Bug 2 fix: stage checklist post-pipeline statuses** — `b5b0141` (fix)
4. **Bug 1 fix: HTMX OOB sign-off footer** — `2e0a9bc` (fix)

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `shop/routers/review.py` — Added `run_id=run.id` to `review_queue()` TemplateResponse; added `run` to approve/override TemplateResponse contexts
- `shop/templates/review/_item_card.html` — Added "Confidence: " prefix to badge; added _signoff_footer_oob.html OOB include
- `shop/templates/review/queue.html` — Replaced inline sticky footer with _signoff_footer.html include
- `shop/templates/runs/_stage_checklist.html` — Extended is_done to all post-pipeline statuses
- `shop/templates/review/_signoff_footer.html` — New: initial-render partial (id="signoff-footer", no hx-swap-oob)
- `shop/templates/review/_signoff_footer_oob.html` — New: OOB variant with hx-swap-oob="outerHTML"

## Decisions Made

- `run_id` must be passed explicitly in the `review_queue()` TemplateResponse context. Jinja2 resolves undefined variables to empty string (no KeyError), which caused `{{ run_id }}` in HTMX `hx-post` attributes and `<img src>` to produce broken URLs.
- Confidence badge label added directly in the template with a static prefix — no JavaScript or dynamic rendering needed.
- Stage checklist `is_done` condition extended to all post-pipeline statuses so all 8 stages appear green/ticked after sign-off, not just after "completed".
- Dual-partial OOB pattern for sign-off footer: a single partial with `id="signoff-footer"` wrapped in an OOB div (also with `id="signoff-footer"`) creates double-id nesting; two separate files (_signoff_footer.html and _signoff_footer_oob.html) solve this cleanly.

## Deviations from Plan

### Auto-fixed Issues (via human verification gate — two rounds)

**1. [Rule 1 - Bug] Missing run_id in review_queue() TemplateResponse context**
- **Found during:** Task 2 round 1 (Human verification in Docker)
- **Issue:** `{{ run_id }}` resolved to empty string in template; snippet `<img src>` produced `/review//snippets/foo.png` (404); HTMX form actions produced `/review//items/{char_no}/approve` (404)
- **Fix:** Added `run_id=run.id` to the `TemplateResponse` kwargs in `shop/routers/review.py`
- **Files modified:** `shop/routers/review.py`
- **Verification:** Snippet images load; approve/override POSTs succeed
- **Committed in:** 01e2b88

**2. [Rule 1 - Bug] Confidence badge had no label**
- **Found during:** Task 2 round 1 (Human verification in Docker)
- **Issue:** Badge showed a raw number (e.g. "0.87") with no context label — unclear to reviewer what the number represented
- **Fix:** Added "Confidence: " prefix directly in `_item_card.html` badge text
- **Files modified:** `shop/templates/review/_item_card.html`
- **Verification:** Badge now displays "Confidence: 87%" style label
- **Committed in:** 01e2b88

**3. [Rule 1 - Bug] Stage 8 not ticked after sign-off**
- **Found during:** Task 2 round 2 (Human verification in Docker)
- **Issue:** `_stage_checklist.html` only treated `run.status == "completed"` as all-stages-done; after sign-off `run.status` is `"signed_off"` so stage 8 was not marked done
- **Fix:** Changed is_done condition to `run.status in ["completed", "reviewing", "signing_off", "signed_off", "warning"]`
- **Files modified:** `shop/templates/runs/_stage_checklist.html`
- **Verification:** 70 tests pass; no regressions
- **Committed in:** b5b0141

**4. [Rule 1 - Bug] Sign-off footer doesn't update after HTMX approve/override**
- **Found during:** Task 2 round 2 (Human verification in Docker)
- **Issue:** Sticky footer rendered `pending` count and Sign Off button state at page-load only; HTMX approve/override responses updated `#progress-bar` via OOB but never updated the footer
- **Fix:** Extracted footer into two partials (_signoff_footer.html for initial render, _signoff_footer_oob.html with hx-swap-oob="outerHTML"); added OOB include to `_item_card.html` oob_update block; passed `run` to approve/override TemplateResponse
- **Files modified:** `shop/templates/review/_signoff_footer.html` (new), `shop/templates/review/_signoff_footer_oob.html` (new), `shop/templates/review/_item_card.html`, `shop/templates/review/queue.html`, `shop/routers/review.py`
- **Verification:** 70 tests pass; no regressions
- **Committed in:** 2e0a9bc

---

**Total deviations:** 4 auto-fixed (all Rule 1 - Bug)
**Impact on plan:** All fixes required for correct reviewer UX and accurate visual feedback. No scope creep.

## Issues Encountered

None beyond the four bugs identified and fixed during human verification (two rounds).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 3 is complete. All REVIEW-* and SIGNOFF-* requirements satisfied.
- Phase 4 (Audit Packet) can begin: the signed-off `Run` record, `ReviewItem` decisions, and snippet evidence are all committed to the DB and filesystem as expected.
- Existing blocker documented: WeasyPrint behavior with real 300 DPI engineering drawing PNG crops is unverified — run a proof-of-concept during Phase 4 before finalizing the audit packet PDF template.

## Self-Check: PASSED

- `shop/templates/review/_signoff_footer.html` — exists (created)
- `shop/templates/review/_signoff_footer_oob.html` — exists (created)
- `shop/templates/runs/_stage_checklist.html` — modified (b5b0141)
- `shop/routers/review.py` — modified (2e0a9bc)
- Commits b5b0141 and 2e0a9bc present in git log

---
*Phase: 03-review-and-sign-off*
*Completed: 2026-03-07*
