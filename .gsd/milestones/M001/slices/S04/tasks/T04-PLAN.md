# T04: 04-exports-history-and-amendments 04

**Slice:** S04 — **Milestone:** M001

## Description

Extend the run history list with a date filter, add read-only mode to the review queue for signed-off runs, add admin retention settings, and implement the Huey periodic cleanup task.

Purpose: HISTORY-01..04 — engineers can browse all runs filtered by date, reopen finalized runs to view decisions, and admins can configure how long unfinished/failed runs are kept.
Output: Updated runs.py (date filter), review.py (read-only mode), admin.py + settings.html (retention UI), tasks.py (cleanup task), updated list.html and test_history.py.

## Must-Haves

- [ ] "Run history list accepts a date_from query param and filters runs submitted on or after that date"
- [ ] "Signed-off run opens its review queue in read-only mode (no approve/override controls)"
- [ ] "Signed-off runs are never deleted by the retention cleanup task"
- [ ] "Admin settings page has a Run Retention section with a number-of-days input"
- [ ] "Saving retention settings persists retention_days to ShopConfig"
- [ ] "Huey periodic_task runs daily and deletes runs older than retention_days in eligible statuses"

## Files

- `shop/routers/runs.py`
- `shop/routers/admin.py`
- `shop/routers/review.py`
- `shop/tasks.py`
- `shop/templates/runs/list.html`
- `shop/templates/admin/settings.html`
- `tests/test_history.py`
