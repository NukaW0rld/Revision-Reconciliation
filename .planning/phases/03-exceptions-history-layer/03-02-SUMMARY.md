---
phase: 03-exceptions-history-layer
plan: 02
subsystem: api
tags: [history-reuse, conformance, pydantic, huey]
requires:
  - phase: 03-01
    provides: durable accepted-alternate history snapshots keyed by run, part, and truth identity
provides:
  - Conservative accepted-alternate reuse layered after canonical truth evaluation
  - Packet/report metadata that distinguishes canonical conformance from history-backed alternates
  - Task-side loading of same-part active alternates into the standalone-capable pipeline
affects: [review, output-formatting, pipeline-task]
tech-stack:
  added: []
  patterns: [post-truth alternate reuse, evaluator-safe history records]
key-files:
  created:
    - tests/test_history_conformance.py
  modified:
    - delta_preservation/types.py
    - delta_preservation/evaluation/conformance.py
    - delta_preservation/cli.py
    - shop/tasks.py
    - shop/services/alternate_history.py
    - shop/services/review.py
    - tests/test_debug_internals.py
key-decisions:
  - "Keep canonical truth mismatches intact even when accepted alternate history upgrades the row to conforming."
  - "Load active alternates in shop/tasks.py and pass plain data into run_pipeline so delta_preservation stays decoupled from SQLAlchemy."
  - "Require exact reviewed-classification, requirement, and mismatch fingerprint matches before a history row can auto-conform a later packet row."
patterns-established:
  - "History-backed conformance is additive: truth evaluation runs first, alternate reuse only upgrades review_needed rows."
  - "Debug report rows use evaluation.conformance_source plus history_reference to distinguish acceptable alternates from canonical matches."
requirements-completed: [HIST-03]
duration: 2m
completed: 2026-04-11
---

# Phase 03: Exceptions History Layer Summary

**Later runs can now auto-conform through same-part accepted alternate history while preserving canonical-truth mismatches and explicit report provenance**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-11T17:33:25-05:00
- **Completed:** 2026-04-11T17:35:56-05:00
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Added reuse-focused regression coverage proving same-part accepted alternates can auto-conform later rows only when the reviewed fingerprint matches exactly.
- Extended packet evaluation metadata with `conformance_source` and `history_reference`, then applied accepted alternate history as an additive pass after canonical truth evaluation.
- Wired `shop/tasks.py` and `assemble_debug_report_payload()` so reruns load same-part active alternates and exports show `acceptable_alternate` instead of `canonical_match` when history drove conformance.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add regression coverage for same-part history reuse and report references** - `44f089e` (`test`)
2. **Task 2: Implement conservative history-backed conformance and report plumbing** - `e2e4d02` (`feat`)

**Plan metadata:** summary committed in the next docs commit.

## Files Created/Modified

- `tests/test_history_conformance.py` - Reuse tests for same-part acceptance, mismatch fingerprint protection, and cross-part isolation.
- `tests/test_debug_internals.py` - Report-state assertion for history-backed `acceptable_alternate` rows with populated `history_reference`.
- `delta_preservation/types.py` - Additive evaluation metadata and evaluator-safe accepted alternate record model.
- `delta_preservation/evaluation/conformance.py` - Pure-data accepted alternate reuse helper layered after canonical truth evaluation.
- `delta_preservation/cli.py` - Optional accepted alternate input and additive reuse pass inside `run_pipeline(...)`.
- `shop/services/alternate_history.py` - Loader for active same-part alternate records plus normalized requirement storage.
- `shop/tasks.py` - Task-side alternate-history loading and pipeline injection.
- `shop/services/review.py` - Debug report rows now surface `acceptable_alternate` state and durable history provenance.

## Decisions Made

- Preserved the existing mismatch list when a row becomes conforming through accepted alternate history so the export still shows exactly what differed from canonical truth.
- Scoped alternate reuse to the task-side part filter plus exact truth identity matching rather than teaching the evaluator any broader cross-part lookup behavior.
- Used `history_reference` as a minimal provenance payload (`history_id`, `source_run_id`) so the report stays auditable without leaking extra database shape into the packet contract.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Missing Critical] Enforced part scoping at the pipeline boundary**
- **Found during:** Post-wave code review
- **Issue:** The task loader filtered alternate history by part, but `run_pipeline(...)` would still accept a mixed-part alternate list from any future non-task caller.
- **Fix:** Added an exact `alternate.part_number == part_name` filter before applying accepted alternate history inside `delta_preservation/cli.py`.
- **Files modified:** `delta_preservation/cli.py`
- **Verification:** `uv run pytest -q tests/test_history_conformance.py tests/test_debug_internals.py tests/test_output_formatting.py tests/test_debug_history.py tests/test_pipeline_task.py -x`
- **Committed in:** `b51a346` (post-review fix)

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Tightened the same-part reuse contract without expanding scope.

## Issues Encountered

- Advisory code review surfaced a part-scoping gap for alternate-history reuse at the pipeline boundary; it was fixed before phase verification.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase-level verification can now check the full exceptions-history layer end to end: persistence, same-part reuse, and report provenance are all implemented.
- Future contradiction-analysis or broader reuse work can build on the new evaluator-safe history record contract without changing the canonical truth path.

---
*Phase: 03-exceptions-history-layer*
*Completed: 2026-04-11*
