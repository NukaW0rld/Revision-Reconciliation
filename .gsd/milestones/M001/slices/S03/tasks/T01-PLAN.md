# T01: 03-review-and-sign-off 01

**Slice:** S03 — **Milestone:** M001

## Description

Wave 1 prerequisite scaffold: test stubs, ReviewItem DB model, and pipeline patch for removed-item Rev B bboxes.

Purpose: Nothing in plans 02-05 can be built without the DB model (no table = no query) or the removed-item bbox patch (no bbox = all "removed" cards show placeholder regardless of implementation). Test stubs guarantee pytest collection is green throughout the phase even before individual tests are implemented.
Output: tests/test_review.py (10 xfail stubs), expanded shop/models.py (ReviewItem + Run extensions), patched delta_preservation/cli.py (predicted Rev B bbox for removed items).

## Must-Haves

- [ ] "ReviewItem table exists and is created by Base.metadata.create_all"
- [ ] "Run model has review_items relationship, signed_at, and signed_by_id columns"
- [ ] "Pipeline emits a predicted Rev B bbox for removed characteristics"
- [ ] "All 10 test stubs are collected by pytest (xfail or skip)"

## Files

- `tests/test_review.py`
- `shop/models.py`
- `delta_preservation/cli.py`
