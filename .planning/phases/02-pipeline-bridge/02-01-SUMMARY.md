---
phase: 02-pipeline-bridge
plan: 01
subsystem: database
tags: [sqlalchemy, huey, sqlite, task-queue, models, testing]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: "User model, Base, SessionLocal, hash_password, conftest.py fixture pattern"
provides:
  - "Run SQLAlchemy model with all required columns (status, stages, failure/warning fields, reviewer FK)"
  - "RunAlert SQLAlchemy model linked to Run and User"
  - "SqliteHuey instance in shop/tasks.py with HUEY_DB env var"
  - "run_pipeline_task stub in shop/tasks.py"
  - "tests/test_runs.py with 12 xfail stubs for UPLOAD and PIPE requirements"
  - "engineer_user and huey_immediate fixtures in tests/conftest.py"
  - "huey 2.6.0 installed in pyproject.toml"
affects:
  - 02-02 (runs router — imports Run model and run_pipeline_task)
  - 02-03 (pipeline task implementation — replaces stub in tasks.py)
  - 02-04 (SSE progress — reads Run.current_stage from DB)
  - 02-05 (upload validation — creates Run records)
  - 02-06 (warning states — reads Run.warning_type)
  - 02-07 (alerts — creates RunAlert records)

# Tech tracking
tech-stack:
  added:
    - "huey 2.6.0 (SqliteHuey, stdlib sqlite3, zero external dependencies)"
  patterns:
    - "HUEY_DB env var pattern with local fallback for dev/test environments"
    - "Deferred shop.* imports inside Huey task body to avoid circular imports"
    - "huey.immediate = True for synchronous test execution (no consumer process needed)"
    - "xfail(strict=False) stubs so test names appear in collection output from day one"

key-files:
  created:
    - shop/tasks.py
    - tests/test_runs.py
  modified:
    - shop/models.py
    - tests/conftest.py
    - pyproject.toml

key-decisions:
  - "HUEY_DB local fallback: when /app/data does not exist (dev/test), tasks.py falls back to project-root huey.db — keeps import-time behavior consistent across Docker and local environments"
  - "All shop.* imports deferred inside run_pipeline_task body — avoids circular imports since tasks.py is imported by both web app and worker"
  - "Run.reviewer_id is nullable FK to users.id — submitter defaults as reviewer, explicit reassignment is Phase 3"
  - "RunAlert linked to both run_id and user_id — alerts are personal to the assigned reviewer per RESEARCH.md constraint"

patterns-established:
  - "Huey task stub pattern: set status=running, placeholder body, set status=completed — Plan 03 replaces the body"
  - "Test fixture isolation: huey_immediate sets huey.immediate=True and flushes storage after each test"

requirements-completed: [PIPE-01, PIPE-03, PIPE-11]

# Metrics
duration: 3min
completed: 2026-03-04
---

# Phase 2 Plan 1: Foundation Summary

**Run + RunAlert SQLAlchemy models, SqliteHuey task queue in shop/tasks.py, and 12 xfail test stubs establishing the shared DB and task foundation for all Phase 2 plans**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-04T01:23:45Z
- **Completed:** 2026-03-04T01:27:28Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- Run model with 20+ columns covering all run lifecycle fields (status, current_stage, failure_stage, failure_message, warning_type, confidence_summary, reviewer FK)
- RunAlert model linked to both Run and User for personal reviewer alerts
- SqliteHuey instance in shop/tasks.py with env-var path and dev fallback; run_pipeline_task stub ready for Plan 03 implementation
- 12 xfail test stubs collected by pytest (UPLOAD-01..05, PIPE-01..06, PIPE-11) plus engineer_user and huey_immediate fixtures

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Run and RunAlert models to shop/models.py and install huey** - `8ae3f1e` (feat)
2. **Task 2: Create shop/tasks.py with SqliteHuey instance and pipeline task stub** - `ceed34c` (feat)
3. **Task 3: Add test scaffolding (tests/test_runs.py stubs + conftest fixtures)** - `b7e19d0` (feat)

## Files Created/Modified

- `shop/models.py` - Added Run and RunAlert models; added runs relationship to User
- `shop/tasks.py` - Created: SqliteHuey instance + run_pipeline_task stub with deferred imports
- `tests/test_runs.py` - Created: 12 xfail stubs for all UPLOAD and PIPE requirements
- `tests/conftest.py` - Added engineer_user and huey_immediate fixtures; import Run/RunAlert
- `pyproject.toml` - Added huey to dependencies list

## Decisions Made

- **HUEY_DB local fallback:** SqliteHuey at import time requires a writable path. When `/app/data` does not exist (dev/test environment outside Docker), tasks.py falls back to a local `huey.db` at the project root. This keeps import behavior consistent — no environment-specific conditionals needed in tests.
- **Deferred shop.* imports in task body:** `from shop.database import SessionLocal` and `from shop.models import Run` are inside the task function, not at module level. This avoids circular imports since shop/tasks.py is imported by both the web app and the Huey worker consumer.
- **Nullable reviewer_id:** Run.reviewer_id defaults to nullable FK — the submitter will be set as the default reviewer in the runs router (Plan 02), with explicit reassignment deferred to Phase 3.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] SqliteHuey path fallback for dev/test environments**
- **Found during:** Task 2 (Create shop/tasks.py)
- **Issue:** `SqliteHuey(filename="/app/data/huey.db")` raises `sqlite3.OperationalError: unable to open database file` at import time when `/app/data/` does not exist (local dev, CI, test environments outside Docker)
- **Fix:** Added a `Path(_default_huey_db).parent.exists()` check at module load; falls back to `<project_root>/huey.db` when the Docker volume path is unavailable
- **Files modified:** shop/tasks.py
- **Verification:** `uv run python -c "from shop.tasks import huey, run_pipeline_task; print('OK')"` passes
- **Committed in:** ceed34c (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug: import-time path failure)
**Impact on plan:** Required fix for any non-Docker execution environment. No scope creep.

## Issues Encountered

- SqliteHuey requires the parent directory to exist at import time — fixed with path existence check (see Deviations above).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Run + RunAlert models are importable and create tables correctly (verified with `Base.metadata.create_all`)
- shop/tasks.py exports `huey` and `run_pipeline_task` — Plans 02-07 can import freely
- test stubs are collected by pytest and ready to be implemented in Plans 02-07
- Full test suite: 21 passed, 14 xfailed, 3 xpassed — no regressions

---
*Phase: 02-pipeline-bridge*
*Completed: 2026-03-04*
