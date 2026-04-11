---
phase: 02-focused-debug-workflow
plan: 03
subsystem: ui
tags: [debug-ui, run-status, review-queue, templates]
requires:
  - phase: 02-01
    provides: exception-only debug queue state keyed by ReviewItem.id
  - phase: 02-02
    provides: phase-2 verdict vocabulary and explicit debug report row states
provides:
  - Admin status-page debug summary for all-conforming and mixed runs
  - Mismatch-first exception cards with Phase 2 verdict vocabulary
  - Exception-only debug export readiness UX across queue and status surfaces
affects: [review-routing, debug-export, admin-triage]
tech-stack:
  added: []
  patterns: [item-id keyed debug UI context, read-only queue seeding without forced status transition]
key-files:
  created: [tests/test_run_status_debug_summary.py]
  modified: [shop/services/review.py, shop/routers/review.py, shop/routers/runs.py, shop/templates/review/_item_card_debug.html, shop/templates/review/_signoff_footer.html, shop/templates/review/queue.html, shop/templates/runs/status.html, tests/test_focused_debug_queue.py, tests/test_debug_verdicts.py, tests/test_debug_internals.py]
key-decisions:
  - "Status-page debug summaries seed ReviewItems without forcing a completed run into reviewing state."
  - "Exception readiness and debug export gating depend only on unresolved exception rows, not canonical matches."
patterns-established:
  - "Use item-id keyed semantic/debug maps for the admin queue so duplicate or null char numbers never collide."
  - "Lead debug cards with ordered mismatches, then put heavier diagnostics behind closed details blocks."
requirements-completed: [DREV-01, DREV-02, DREV-03, DREV-04]
duration: 50min
completed: 2026-04-10
---

# Phase 02-03 Summary

**Admins now get an auto-pass-aware status-page summary and a mismatch-first exception queue that uses the new Phase 2 verdict vocabulary without letting canonical matches block debug export readiness.**

## Performance

- **Duration:** 50 min
- **Started:** 2026-04-10T22:10:00-05:00
- **Completed:** 2026-04-10T23:00:57-05:00
- **Tasks:** 3
- **Files modified:** 12

## Accomplishments

- Added an admin-only `Debug evaluation summary` card on the run status page, including all-conforming messaging, exception counts, canonical match inspection, and debug-report download links when ready.
- Reworked the admin debug queue so mismatch summaries appear first, only `algorithm_error` / `acceptable_alternate` are offered, and heavier diagnostics stay collapsed by default.
- Made debug export readiness consistent across status and queue surfaces by basing it solely on unresolved exception rows and preserving stable item identity through ReviewItem ids.

## Task Commits

Each task was committed atomically where the code boundaries allowed:

1. **Task 1: Add integration coverage for run-details debug summary and focused exception rendering** - `74c84c8` (`test`)
2. **Task 2 + Task 3: Implement run-details summary plus mismatch-first exception workflow** - `b824997` (`feat`)

## Files Created/Modified

- `tests/test_run_status_debug_summary.py` - Covers all-conforming status-page flow, mixed-run counts, and ready-to-download debug reports.
- `tests/test_focused_debug_queue.py` - Verifies exception-only queue rendering, Phase 2 verdict options, mismatch-first ordering, and collapsed diagnostics.
- `tests/test_debug_verdicts.py` - Aligns the older admin debug workflow tests with the new verdict vocabulary and footer/status copy.
- `tests/test_debug_internals.py` - Keeps the debug-router integration test valid under the exception-only queue behavior.
- `shop/services/review.py` - Adds item-id keyed debug helpers, status-page summaries, read-only queue seeding, and exception-only export readiness logic.
- `shop/routers/review.py` - Wires item-id keyed debug context into the queue and blocks debug exports until all exception rows are resolved.
- `shop/routers/runs.py` - Loads the admin debug summary onto the run status page.
- `shop/templates/runs/status.html` - Renders the new admin summary card, canonical match disclosure, and debug-report CTA.
- `shop/templates/review/queue.html` - Reframes debug mode as an exception queue and uses item-id keyed debug context.
- `shop/templates/review/_item_card_debug.html` - Moves mismatch details ahead of the form, exposes only Phase 2 verdicts, and collapses diagnostics.
- `shop/templates/review/_signoff_footer.html` - Shows exception-only readiness copy in the main footer.
- `shop/templates/review/_signoff_footer_oob.html` - Mirrors the new readiness copy for HTMX updates.

## Decisions Made

- Preserved `open_review_queue()` as the transition point into `reviewing`, but introduced read-only queue seeding for status-page summaries and export assembly so admin inspection does not force a state change.
- Kept the queue export button label as `Export debug_report.json` while changing the readiness copy around it to `Debug report ready` versus `Resolve all exception rows to export debug_report.json`.

## Deviations from Plan

### Auto-fixed Issues

**1. Shared helper boundary across Tasks 2 and 3**
- **Found during:** Wave 3 implementation
- **Issue:** The run-status summary, export gating, and mismatch-first queue UI all depended on the same new `shop/services/review.py` helpers for item-id keyed context and read-only queue seeding.
- **Fix:** Landed Tasks 2 and 3 together in a single feature commit so the service, routers, and templates stayed coherent.
- **Files modified:** `shop/services/review.py`, `shop/routers/review.py`, `shop/routers/runs.py`, `shop/templates/review/*`, `shop/templates/runs/status.html`
- **Verification:** `uv run pytest -q tests/test_focused_debug_queue.py tests/test_run_status_debug_summary.py tests/test_debug_verdicts.py tests/test_debug_internals.py -x`
- **Committed in:** `b824997`

---

**Total deviations:** 1 auto-fixed (shared helper boundary)
**Impact on plan:** No scope creep. The combined feature commit kept the queue/status/export contract internally consistent.

## Issues Encountered

- A legacy integration test in `tests/test_debug_internals.py` seeded only conforming rows and started failing once the debug queue correctly redirected all-conforming runs back to the status page. The fixture packet was updated to seed a real `review_needed` row so the router wiring test still exercised the debug card path.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 is now coherent end-to-end: status page, exception queue, verdict persistence, and export readiness all follow the same exception-only contract.
- Manual browser verification was not run in a live server session; automated FastAPI integration tests covered the rendered HTML paths instead.
