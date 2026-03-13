# T06: 03-review-and-sign-off 06

**Slice:** S03 — **Milestone:** M001

## Description

Phase 3 gate: complete all xfail test stubs to passing, run the full suite to confirm nothing regressed, and perform human verification of the review workflow in Docker.

Purpose: Plans 02-05 may have left some test stubs as xfail. This plan finishes implementing any remaining stubs, then the engineer verifies the complete workflow manually in the running Docker container.
Output: All 10 tests in test_review.py passing; Docker human verification approved.

## Must-Haves

- [ ] "All 10 requirement tests in test_review.py pass (none xfail, none skip)"
- [ ] "Full pytest suite is green"
- [ ] "Review queue accessible from completed run status page"
- [ ] "Approve and override HTMX saves update card state without page reload"
- [ ] "Sign-off button enables only when all items are resolved"
- [ ] "Sign-off and rollback behave atomically per SIGNOFF-02"

## Files

- `tests/test_review.py`
