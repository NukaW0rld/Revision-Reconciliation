---
id: T03
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
# T03: 04-exports-history-and-amendments 03

**# Phase 4 Plan 3: Work Order Exports Summary**

## What Happened

# Phase 4 Plan 3: Work Order Exports Summary

Partial FAI work order PDF/CSV exports with RE-MEASURE/NEW priority labels, WeasyPrint template, download routes, and status page buttons for signed-off runs.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Work order service functions and WeasyPrint template | 82f63f0 | shop/services/exports.py, shop/templates/exports/work_order.html |
| 2 | Work order download routes, status page buttons, and test implementations | 94e7d3c | shop/routers/exports.py, shop/templates/runs/status.html, tests/test_exports.py |

## What Was Built

- `_effective_classification(item)` resolves override_classification if reviewer_decision == "overridden", otherwise returns pipeline_classification
- `_work_order_rows(db, run)` filters ReviewItems to changed/added only, maps changed→"RE-MEASURE" and added→"NEW", returns list of dicts
- `generate_work_order_csv(db, run)` returns StringIO seeked to 0 with fieldnames: char_no, priority, requirement_revB, drawing_reference
- `generate_work_order_pdf(db, run)` renders work_order.html via WeasyPrint and returns bytes
- `shop/templates/exports/work_order.html` — Letter-size WeasyPrint template with run header table, work items table (Char #, Priority, Requirement (Rev B), Drawing Reference), and empty-state message
- `GET /exports/{run_id}/work-order.pdf` — StreamingResponse with attachment header, 403 for non-signed_off runs
- `GET /exports/{run_id}/work-order.csv` — StreamingResponse with attachment header, 403 for non-signed_off runs
- Status page: added Generate Work Order PDF/CSV buttons alongside audit packet buttons in the signed_off block

## Test Results

```
tests/test_exports.py: 7 passed (3 audit packet + 4 work order)
Full suite: 81 passed, 5 xfailed, 3 xpassed
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added missing login call in test_work_order_button_visible**
- **Found during:** Task 2 test run
- **Issue:** Plan's test stub for test_work_order_button_visible did not call _login_engineer() before GET /runs/{id}. The route requires authentication and redirected to /login, causing "work-order.pdf" assertion to fail.
- **Fix:** Added `db_engine` parameter and `_login_engineer(client, db_engine, engineer_user)` call at start of test, matching the pattern used in test_audit_packet_redownload.
- **Files modified:** tests/test_exports.py
- **Commit:** 94e7d3c

## Self-Check: PASSED

All created files exist, all task commits present, 7/7 export tests pass.
