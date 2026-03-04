---
phase: 02-pipeline-bridge
plan: 03
subsystem: pipeline
tags: [huey, sqlalchemy, delta-preservation, stage-callback, run-status, alert]

# Dependency graph
requires:
  - phase: 02-01
    provides: Run and RunAlert models, Huey task stub, SessionLocal factory
  - phase: 01-02
    provides: SQLAlchemy DB session and User model
provides:
  - run_pipeline() with optional stage_callback parameter (8-stage observable execution)
  - run_pipeline_task() full Huey implementation with stage progress, failure/warning/success paths
  - RunAlert creation on Rev A balloon failure or exception
affects:
  - 02-04 (run status page reads current_stage and status from DB)
  - 02-05 (SSE stage-progress stream emits events from stage_callback)
  - 02-06 (human review reads output_dir from Run.output_dir)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Huey task calls underlying function via .call_local() in tests to bypass queue"
    - "Module-level SessionLocal/run_pipeline for mock.patch compatibility in task tests"
    - "stage_callback guard: if stage_callback: callback(...) — avoids AttributeError on None"
    - "Post-run delta packet JSON read to detect balloon failure (0 items) and low confidence"

key-files:
  created:
    - tests/test_cli_stage_callback.py
    - tests/test_pipeline_task.py
  modified:
    - delta_preservation/cli.py
    - shop/tasks.py

key-decisions:
  - "stage_callback called BEFORE each stage so UI shows 'running: Stage N' during execution"
  - "Rev A balloon failure detected post-run via empty items in delta_packet.json, not mid-pipeline"
  - "RevB balloon failure v1: surfaces as low_confidence warning (alignment inlier_ratio near zero)"
  - "Low-confidence threshold: >50% of items with location score < 0.5 triggers warning state"
  - "PIPE-07/08/09/10 (unchanged/changed/removed/uncertain) handled by existing classify.py"
  - "run_pipeline_task.call_local() used in tests to execute task synchronously without Huey queue"

patterns-established:
  - "Module-level sentinel pattern: expose SessionLocal/run_pipeline at top level so tests can patch them"
  - "Post-run analysis pattern: read output artifacts after pipeline returns to determine final status"

requirements-completed: [PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05, PIPE-06, PIPE-07, PIPE-08, PIPE-09, PIPE-10, PIPE-11]

# Metrics
duration: 7min
completed: 2026-03-03
---

# Phase 2 Plan 3: Pipeline Bridge — Stage Callback and Task Implementation Summary

**stage_callback wiring in run_pipeline() and full Huey run_pipeline_task() with stage updates, Rev A balloon failure detection, low-confidence alignment warning, and RunAlert creation on failure**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-03-04T01:30:22Z
- **Completed:** 2026-03-04T01:37:30Z
- **Tasks:** 2
- **Files modified:** 4 (2 source, 2 tests)

## Accomplishments
- Added `stage_callback: Optional[Callable[[int, str], None]] = None` parameter to `run_pipeline()` with calls at all 8 stages (0-indexed)
- Implemented full `run_pipeline_task()` Huey task: sets status=running, updates DB at each stage via callback, detects Rev A balloon failure post-run, detects low-confidence alignment, sets completed/failed/warning states, creates RunAlert
- 16 new TDD tests covering all task paths; full suite now 53 passed (was 21)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add stage_callback to run_pipeline()** - `83b796d` (feat + test)
2. **Task 2: Implement full run_pipeline_task()** - `018edf0` (feat + test)

**Plan metadata:** (see final commit below)

## Files Created/Modified
- `delta_preservation/cli.py` - Added `Callable` import and `stage_callback` parameter; 8 guard-and-call blocks before each stage
- `shop/tasks.py` - Full run_pipeline_task() replacing stub; module-level SessionLocal/run_pipeline exports; _update_stage and _create_alert helpers
- `tests/test_cli_stage_callback.py` - TDD tests: signature, 8-call count, call order, None default (5 tests)
- `tests/test_pipeline_task.py` - TDD tests: success path, stage updates, running at start, balloon failure, exception failure, no-reviewer edge case, low-confidence warning, minority-scores no-warning, nonexistent run (11 tests)

## Decisions Made
- Stage callback called BEFORE each stage (not after) so the UI can show "currently running: Stage N" during execution
- Rev A balloon failure is detected post-run by checking `delta_packet.json` items length == 0 rather than mid-pipeline interception, because stage_callback fires BEFORE each stage and cannot observe stage outputs
- RevB balloon failure surfaces through low-confidence warning path in v1 (alignment produces near-zero inlier ratios when RevB has no balloons)
- Low-confidence threshold set at >50% of items with location score < 0.5 (majority of characteristics have poor Rev B location)
- `run_pipeline_task.call_local()` used in tests to execute synchronously without Huey's SQLite queue

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Huey task `@huey.task()` decorator wraps the function in a `TaskWrapper`; direct calls like `run_pipeline_task(...)` enqueue rather than execute. Solved by using `.call_local()` in tests.
- `mock.patch("shop.tasks.SessionLocal", ...)` requires the symbol to exist at module level. Solved by eagerly importing `SessionLocal` and `run_pipeline` at module scope with try/except for ImportError safety.

## Next Phase Readiness
- Stage callback infrastructure ready for Plan 05 (SSE stage-progress stream)
- Run status transitions (queued/running/completed/failed/warning) fully implemented
- RunAlert creation tested and working for reviewer notification in Plan 06+

---
*Phase: 02-pipeline-bridge*
*Completed: 2026-03-03*

## Self-Check: PASSED

All files present, all commits exist, 53 tests passing with no regressions.
