# T03: 03-review-and-sign-off 03

**Slice:** S03 — **Milestone:** M001

## Description

Per-item approve/override HTMX actions, the review item card template, and the snippet file serving route.

Purpose: The review queue (Plan 02) renders a list of items. This plan makes each item interactive — approve and override are saved immediately via HTMX POST and the card swaps to its resolved state. The progress counter updates via OOB swap. Snippet images are served from the run output directory via a dedicated route. This plan also wires queue.html to replace its Plan 02 placeholder item loop with the real _item_card.html include.
Output: _item_card.html template (unresolved + resolved states), approve and override POST routes added to review.py, snippet GET route, modal overlay for full-size snippet viewing, queue.html updated to use real card partials.

## Must-Haves

- [ ] "Each review card shows char_no, pipeline classification, confidence, and Rev A/Rev B snippets"
- [ ] "Approve POST saves reviewer_decision='approved' and returns updated card via HTMX swap"
- [ ] "Override POST with empty note returns 422 and re-renders card with error message"
- [ ] "Override POST with valid note saves reviewer_decision='overridden' and override_note"
- [ ] "After approve/override the progress counter updates via HTMX OOB swap"
- [ ] "Snippet images are served via a dedicated route; missing images show placeholder"
- [ ] "Clicking a snippet opens it full-size in a modal overlay"
- [ ] "Review queue renders real _item_card.html partials (not placeholder divs)"

## Files

- `shop/routers/review.py`
- `shop/templates/review/_item_card.html`
- `shop/templates/review/queue.html`
