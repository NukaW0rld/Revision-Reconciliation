---
phase: 03-exceptions-history-layer
plan: 01
subsystem: database
tags: [alembic, sqlalchemy, debug-review, history]
requires: []
provides:
  - Durable accepted-alternate history persistence separate from ground truth fixtures
  - Router-level sync from admin debug verdict saves into reusable history rows
  - Regression coverage for history creation, deactivation, and truth immutability
affects: [review, conformance, debug-report]
tech-stack:
  added: [alembic migration]
  patterns: [db-backed alternate history snapshots, reversible verdict sync]
key-files:
  created:
    - alembic/versions/0002_accepted_alternate_history.py
    - shop/services/alternate_history.py
    - tests/test_debug_history.py
  modified:
    - shop/models.py
    - shop/routers/review.py
    - tests/test_alembic_baseline.py
key-decisions:
  - "Persist acceptable alternates in a dedicated SQLAlchemy table instead of mutating ground_truth fixtures."
  - "Map review items back to packet rows using review-item ordering so history snapshots stay aligned with the exact debug row."
  - "Deactivate active history rows when verdicts move away from acceptable_alternate to prevent stale reuse."
patterns-established:
  - "Alternate history lives in shop/ and is synchronized from the admin debug save path."
  - "History records snapshot reviewed classification, normalized requirement text, mismatch codes, and rationale for later conservative reuse."
requirements-completed: [HIST-01, HIST-02]
duration: 4m
completed: 2026-04-11
---

# Phase 03: Exceptions History Layer Summary

**Accepted alternate outcomes now persist as DB-backed history snapshots with reversible verdict sync and immutable-truth regression coverage**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-11T17:22:26-05:00
- **Completed:** 2026-04-11T17:26:38-05:00
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Added Phase 3 regression coverage proving acceptable alternates create durable history rows, deactivate cleanly, and never mutate `ground_truth.json`.
- Added the `AcceptedAlternateHistory` ORM model and `0002_accepted_alternate_history` migration for a reusable history layer scoped away from canonical truth.
- Hooked admin debug verdict saves into a dedicated sync service that upserts active acceptable-alternate records and deactivates stale ones when verdicts change.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add persistence-contract regression coverage for accepted alternates** - `81a059e` (`test`)
2. **Task 2: Implement the accepted alternate history model, migration, and verdict-sync service** - `719464a` (`feat`)

**Plan metadata:** summary committed in the next docs commit.

## Files Created/Modified

- `tests/test_debug_history.py` - Integration coverage for history creation, deactivation, and truth immutability.
- `shop/models.py` - `AcceptedAlternateHistory` ORM model with audit and reuse fields.
- `alembic/versions/0002_accepted_alternate_history.py` - Schema migration for the accepted alternate history table and lookup index.
- `shop/services/alternate_history.py` - Packet-aware sync service for accepted alternate persistence.
- `shop/routers/review.py` - Admin debug verdict route now synchronizes accepted alternate history after successful saves.
- `tests/test_alembic_baseline.py` - Baseline migration test now asserts the repository’s current Alembic head dynamically.

## Decisions Made

- Kept history synchronization in a standalone `shop.services.alternate_history` module so Phase 3 persistence does not create a service import cycle or couple to truth fixtures.
- Stored `matched_truth_char_no` as a string snapshot because packet truth tokens can be either integers or synthetic string ids.
- Preserved mismatch ordering from packet evaluation when persisting `mismatch_codes` so later reuse can compare the exact reviewed fingerprint.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated the Alembic baseline test for the new head revision**
- **Found during:** Task 2 (history model and migration implementation)
- **Issue:** `tests/test_alembic_baseline.py` still hard-coded revision `0001`, so adding `0002` broke the migration smoke test.
- **Fix:** Changed the test to resolve the current Alembic head dynamically from the configured script directory.
- **Files modified:** `tests/test_alembic_baseline.py`
- **Verification:** `uv run pytest -q tests/test_debug_history.py tests/test_alembic_baseline.py -x`
- **Committed in:** `719464a` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Required to keep the migration contract accurate after adding the new history schema. No scope creep.

## Issues Encountered

- Initial delegated execution stalled after the Task 1 commit, so the remaining implementation and verification were completed inline without changing scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Wave 2 can now load durable accepted-alternate history and apply conservative same-part reuse on top of canonical truth evaluation.
- History records now carry the exact reviewed fingerprint fields (`part_number`, `matched_truth_char_no`, reviewed outcome, mismatch codes, rationale) needed for later-run conformance.

---
*Phase: 03-exceptions-history-layer*
*Completed: 2026-04-11*
