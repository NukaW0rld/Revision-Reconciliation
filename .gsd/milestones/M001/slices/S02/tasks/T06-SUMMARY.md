---
id: T06
parent: S02
milestone: M001
provides:
  - POST /alerts/dismiss/{id}: marks RunAlert.is_read=True, returns empty HTMX response
  - POST /runs/{run_id}/acknowledge-warning: redirects to /review/{run_id} placeholder
  - Dashboard unread alert banners: reviewer-scoped, dismissible via HTMX
  - Alert banner partial: shop/templates/runs/_alert_banner.html
requires: []
affects: []
key_files: []
key_decisions: []
patterns_established: []
observability_surfaces: []
drill_down_paths: []
duration: 4min
verification_result: passed
completed_at: 2026-03-04
blocker_discovered: false
---
# T06: 02-pipeline-bridge 06

**# Phase 02 Plan 06: Failure Alert UI Summary**

## What Happened

# Phase 02 Plan 06: Failure Alert UI Summary

**HTMX-dismissible dashboard alert banners for run failures with POST /alerts/dismiss/{id} endpoint, scoped to the assigned reviewer**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-04T01:41:28Z
- **Completed:** 2026-03-04T01:45:35Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Implemented PIPE-11: reviewer sees unread failure alert banners on dashboard after a pipeline failure
- POST /alerts/dismiss/{id} marks RunAlert.is_read=True and returns empty 200 so HTMX swaps the banner element out of the DOM
- Alert banner partial shows part number, run ID, failure stage, and View run link with DaisyUI alert-error styling
- Dashboard GET /dashboard now passes unread_alerts list to template (derived from single query, supports both banners and badge count)
- POST /runs/{run_id}/acknowledge-warning added as Phase 3 placeholder (redirects to /review/{id})

## Task Commits

Each task was committed atomically:

1. **Task 1 TDD RED: Add failing test for alert dismiss endpoint** - `2ad8deb` (test)
2. **Task 1 TDD GREEN: Add alert dismiss and acknowledge-warning routes** - `adda11a` (feat)
3. **Task 2: Alert banner partial + dashboard update** - `1ef771b` (feat)

_Note: TDD task has two commits (test RED then feat GREEN)_

## Files Created/Modified

- `shop/templates/runs/_alert_banner.html` - Dismissible DaisyUI error alert with SVG icon, part number, run ID, failure stage, View run link, and HTMX dismiss button
- `shop/routers/auth.py` - Added POST /alerts/dismiss/{id} route (dismiss_alert); updated GET /dashboard to query RunAlert list and pass unread_alerts to template
- `shop/templates/dashboard.html` - Added unread_alerts banners section above Submit New Run CTA using include of _alert_banner.html partial
- `tests/test_runs.py` - Replaced xfail PIPE-11 stub with real integration test: seeds Run+RunAlert, POSTs dismiss endpoint, asserts 200 empty body + is_read=True in DB

## Decisions Made

- Dismiss route in auth.py (no prefix) so URL becomes `/alerts/dismiss/{id}` not `/runs/alerts/dismiss/{id}` — consistent with the template hx-post path and avoids routing conflicts
- Dashboard queries `RunAlert` rows and passes the list to template; `unread_alert_count = len(unread_alerts)` avoids a second COUNT query
- `acknowledge-warning` is a Phase 3 placeholder redirect — full review queue implementation deferred per plan specification

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- PIPE-11 complete: reviewer receives in-app failure alerts on dashboard with HTMX dismiss
- Phase 3 can implement the review queue and wire it to the `/review/{run_id}` placeholder that acknowledge-warning now targets
- Plans 05 (status page) needs `status.html` and `_stage_checklist.html` templates — those untracked files exist in the workspace from a prior execution but were not committed (out of scope for this plan)

## Self-Check: PASSED

All created files exist on disk. All task commits verified in git history.

---
*Phase: 02-pipeline-bridge*
*Completed: 2026-03-04*
