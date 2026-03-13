---
id: T04
parent: S04
milestone: M001
provides: []
requires: []
affects: []
key_files: []
key_decisions: []
patterns_established: []
observability_surfaces: []
drill_down_paths: []
duration: 
verification_result: passed
completed_at: 
blocker_discovered: false
---
# T04: 04-exports-history-and-amendments 04

**# Phase 4 Plan 4: Run History Filters, Read-Only Review, and Retention Cleanup Summary**

## What Happened

# Phase 4 Plan 4: Run History Filters, Read-Only Review, and Retention Cleanup Summary

**One-liner:** Date-filtered run history, read-only review queue for signed-off runs, admin retention settings, and daily Huey cleanup task for old runs.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Date filter in history list + read-only review queue | 50b0298 | runs.py, review.py, list.html, queue.html, _item_card.html |
| 2 | Admin retention settings + Huey cleanup task + tests | 4d13fc7 | admin.py, settings.html, tasks.py, test_history.py |

## What Was Built

**Run History Date Filter (HISTORY-01):**
- `list_runs` accepts `date_from: str = None` query parameter
- Filters `Run.submitted_at >= datetime.strptime(date_from, "%Y-%m-%d")`; invalid dates silently ignored
- Combined filtering: `part_number` + `date_from` can be used together
- Template updated with a date input; Clear link condition covers both filters

**Read-Only Review Queue for Signed-Off Runs (HISTORY-02):**
- `review_queue` guard extended to allow `signed_off` status through (previously only `completed/warning/reviewing/signing_off`)
- `read_only = run.status == "signed_off"` passed to template context
- `queue.html` renders an info banner when `read_only` is true
- `_item_card.html` wraps approve/override action controls in `{% if not read_only %}` guard

**Admin Retention Settings (HISTORY-03):**
- `POST /admin/settings/retention` route saves `retention_days` (clamped 1–3650) to `ShopConfig`
- Admin `settings.html` has a "Run Retention" card section with a number input pre-filled from config

**Huey Cleanup Task (HISTORY-04):**
- `cleanup_old_runs` registered as `@huey.periodic_task(crontab(hour="0", minute="0"))` — runs daily at UTC midnight
- Deletes runs in `{queued, running, failed, completed}` statuses older than `retention_days`
- `signed_off`, `reviewing`, `signing_off` are never deleted
- Cleanup removes: `output_dir` (shutil.rmtree), `revA_path`, `revB_path`, `form3_path`
- Falls back to 30 days if `ShopConfig.retention_days` is None

## Test Results

```
tests/test_history.py ....   (4 passed)
Full suite: 77 passed, 9 xfailed, 3 xpassed
```

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- `shop/routers/runs.py` — FOUND, contains `date_from`
- `shop/routers/review.py` — FOUND, contains `read_only`
- `shop/tasks.py` — FOUND, contains `cleanup_old_runs`
- `shop/routers/admin.py` — FOUND, contains `retention`
- `shop/templates/admin/settings.html` — FOUND, contains `retention_days`
- Commit 50b0298 — FOUND
- Commit 4d13fc7 — FOUND
