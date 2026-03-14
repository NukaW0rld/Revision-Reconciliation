---
id: T05
parent: S04
milestone: M001
provides:
  - "create_amendment(db, parent_run, initiator_id) in shop/services/amendments.py — clones a signed-off run with pre-filled ReviewItems"
  - "POST /review/{run_id}/amend route — creates amendment and redirects to its review queue"
  - "Amend button + DaisyUI modal on signed-off run status page"
  - "Amendment banner on review queue page identifying parent run and locked files"
  - "Versioned packet list on status page linking to all packet versions"
  - "Version-aware PDF download via ?version=N query param on audit-packet.pdf route"
  - "3 passing amendment tests (AMEND-01, AMEND-02, AMEND-03)"
requires: []
affects: []
key_files: []
key_decisions: []
patterns_established: []
observability_surfaces: []
drill_down_paths: []
duration: 5min
verification_result: passed
completed_at: 2026-03-08
blocker_discovered: false
---
# T05: 04-exports-history-and-amendments 05

**# Phase 4 Plan 05: Amendment Model Summary**

## What Happened

# Phase 4 Plan 05: Amendment Model Summary

**Amendment workflow: create_amendment service clones signed-off runs with pre-filled review decisions, preserving original packet and producing v2 on re-sign-off**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-08T12:24:02Z
- **Completed:** 2026-03-08T12:29:09Z
- **Tasks:** 2
- **Files modified:** 5 (1 created, 4 modified)

## Accomplishments
- Amendment service creates a new Run with status=reviewing, parent_run_id set, and all ReviewItems pre-filled from parent decisions — engineers can immediately revise specific decisions without re-reviewing everything
- POST /review/{run_id}/amend route (403 guard for non-signed-off runs) + Amend button with DaisyUI confirmation modal on the status page clearly communicates the immutability guarantee
- Version-aware PDF download route (exports.py) + versioned packet list in status.html surfaces all packet versions accessible by link
- 3 AMEND requirement tests pass; full suite 84 passed

## Task Commits

Each task was committed atomically:

1. **Task 1: Amendment service, POST route, and confirmation modal** - `d64e105` (feat)
2. **Task 2: Amendment review queue banner, version-aware download, and test implementations** - `37ad83b` (feat)

**Plan metadata:** `2860810` (docs: complete plan)

## Files Created/Modified
- `shop/services/amendments.py` - create_amendment() service cloning run and review items
- `shop/routers/review.py` - POST /{run_id}/amend route added after sign-off routes
- `shop/routers/exports.py` - download_audit_packet_pdf updated with version: int = 1 param
- `shop/templates/runs/status.html` - Amend button, DaisyUI modal, versioned packet list
- `shop/templates/review/queue.html` - amendment banner when run.parent_run_id is set
- `tests/test_amendments.py` - replaced xfail stubs with 3 real passing tests

## Decisions Made
- Amendment packet_versions is a copy of parent's list (not empty), so generate_and_store_audit_packet computes the correct next version number without needing to know about parent runs
- open_review_queue() short-circuit (existing_count > 0) works naturally — cloned ReviewItems are present immediately, so redirect to amendment queue opens pre-populated
- Version-aware export uses query param `?version=N` for simplicity; falls back to re-render for test environments or missing files

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 4 is now complete — all 5 plans (01-05) implemented
- Amendment model fully functional: create, review, sign-off producing v2 packet
- Original v1 packet preserved on parent run; amendment run holds inherited+own versions
- Full suite passes (84 tests)

---
*Phase: 04-exports-history-and-amendments*
*Completed: 2026-03-08*

## Self-Check: PASSED
- shop/services/amendments.py: FOUND
- shop/routers/review.py: FOUND
- shop/templates/runs/status.html: FOUND
- shop/templates/review/queue.html: FOUND
- tests/test_amendments.py: FOUND
- Task 1 commit d64e105: FOUND
- Task 2 commit 37ad83b: FOUND
- Metadata commit 2860810: FOUND
