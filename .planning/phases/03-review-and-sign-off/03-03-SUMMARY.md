---
phase: 03-review-and-sign-off
plan: "03"
subsystem: review-ui
tags: [htmx, fastapi, jinja2, review, snippets]
dependency_graph:
  requires: [03-01, 03-02]
  provides: [approve-route, override-route, snippet-route, item-card-template]
  affects: [03-04, 03-05]
tech_stack:
  added: []
  patterns: [htmx-oob-swap, fastapi-form, fileresponse, jinja2-filter]
key_files:
  created:
    - shop/templates/review/_item_card.html
  modified:
    - shop/routers/review.py
    - shop/templates/review/queue.html
    - shop/app.py
    - tests/test_review.py
decisions:
  - "_item_card.html OOB update: progress-bar div with hx-swap-oob='true' appended inside card div only when oob_update=True — avoids rendering OOB markup on initial queue page load"
  - "basename filter added to app.py templates env (not router) so it is available across all templates globally"
  - "_item_counts() helper extracted in review.py to avoid code duplication between approve_item and override_item"
metrics:
  duration: "3 min"
  completed_date: "2026-03-07"
  tasks: 2
  files_modified: 5
---

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
