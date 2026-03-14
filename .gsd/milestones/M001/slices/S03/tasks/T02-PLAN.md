# T02: 03-review-and-sign-off 02

**Slice:** S03 — **Milestone:** M001

## Description

Review queue GET route: opens the review queue for a completed run, populates ReviewItem rows from delta_packet.json (idempotent), and renders the scrollable list with filter dropdowns and item count header.

Purpose: This is the entry point engineers use after a pipeline run completes. It transitions Run.status from completed/warning to "reviewing" on first open and serves all subsequent views from the ReviewItem table.
Output: shop/routers/review.py, shop/services/review.py, shop/templates/review/queue.html, shop/templates/review/_progress_bar.html, review router registered in shop/app.py.

## Must-Haves

- [ ] "GET /review/{run_id} returns 200 for completed runs; redirects 302 for non-completed runs"
- [ ] "Opening the review queue is idempotent — opening twice does not create duplicate ReviewItem rows"
- [ ] "Queue page lists all items ordered by char_no with pipeline classification and confidence"
- [ ] "Pending/approved/overridden counts appear in the page header regardless of active filter"
- [ ] "Filter dropdowns (review status + pipeline classification) reduce visible items without hiding counts"

## Files

- `shop/routers/review.py`
- `shop/services/review.py`
- `shop/templates/review/queue.html`
- `shop/templates/review/_progress_bar.html`
- `shop/app.py`
