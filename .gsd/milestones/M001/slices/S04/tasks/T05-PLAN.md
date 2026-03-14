# T05: 04-exports-history-and-amendments 05

**Slice:** S04 — **Milestone:** M001

## Description

Implement the amendment model: create_amendment service, amend POST route, confirmation modal on the status page, and versioned packet list display. Amendment sign-off reuses the existing attempt_sign_off + generate_and_store_audit_packet path.

Purpose: AMEND-01..03 — engineers can reopen a finalized run to correct review decisions without destroying the original signed packet; each sign-off produces a new versioned packet accessible from the run record.
Output: shop/services/amendments.py, updates to review.py, status.html, queue.html, test_amendments.py.

## Must-Haves

- [ ] "Signed-off run status page has an Amend button that shows a confirmation modal"
- [ ] "Submitting the modal creates a new amendment Run with pre-filled ReviewItems from the original"
- [ ] "Amendment input files (revA_path, revB_path, form3_path) are the same as the parent — cannot be changed"
- [ ] "Amendment review queue opens with all items pre-filled from parent reviewer decisions"
- [ ] "Amendment sign-off produces a new versioned packet (v2.pdf); both v1 and v2 are listed on the run page"
- [ ] "Original signed packet (v1.pdf on parent run) is never overwritten"

## Files

- `shop/services/amendments.py`
- `shop/routers/review.py`
- `shop/templates/runs/status.html`
- `shop/templates/review/queue.html`
- `tests/test_amendments.py`
