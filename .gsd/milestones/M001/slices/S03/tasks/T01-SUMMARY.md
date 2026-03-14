---
id: T01
parent: S03
milestone: M001
provides:
  - ReviewItem SQLAlchemy model with full review/override fields
  - Run.review_items relationship, signed_at, signed_by_id columns
  - status_badge_class handles reviewing/signing_off/signed_off
  - Pipeline emits predicted Rev B bbox for removed DeltaItems via homography
  - 10 xfail stubs for REVIEW-01..07 and SIGNOFF-01..03
requires: []
affects: []
key_files: []
key_decisions: []
patterns_established: []
observability_surfaces: []
drill_down_paths: []
duration: 7min
verification_result: passed
completed_at: 2026-03-07
blocker_discovered: false
---
# T01: 03-review-and-sign-off 01

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
