---
phase: 02-pipeline-bridge
plan: 07
subsystem: infra
tags: [docker, supervisor, huey, uvicorn, sqlite]

# Dependency graph
requires:
  - phase: 02-01
    provides: shop/tasks.py with shop.tasks.huey and HUEY_DB env var pattern
  - phase: 02-03
    provides: huey pipeline task confirmed working with thread workers
provides:
  - docker/supervisord.conf running uvicorn on port 8000 and huey_consumer as separate OS processes
  - Dockerfile runtime stage with supervisor installed and supervisord as CMD
  - docker-compose.yml with HUEY_DB, OUT_DIR, UPLOADS_DIR env vars mapped to Docker volumes
affects:
  - 02-08
  - phase 3 deployment verification

# Tech tracking
tech-stack:
  added: [supervisor (apt-get in runtime stage)]
  patterns: [supervisord multi-process container pattern for air-gapped single-container deployment]

key-files:
  created: [docker/supervisord.conf]
  modified: [docker/Dockerfile, docker/docker-compose.yml]

key-decisions:
  - "supervisord pidfile=/tmp/supervisord.pid to avoid /run permission issues in slim container"
  - "priority=10 for uvicorn, priority=20 for huey_worker — uvicorn starts first"
  - "COPY --from=python-builder /bin/uv /bin/uv copies uv binary to runtime for huey_consumer.py path resolution"
  - "COPY run.py run_web.py pyproject.toml uv.lock ./ — pyproject.toml needed in runtime for uv path discovery"

patterns-established:
  - "Multi-process Docker container: supervisord manages uvicorn + background worker, both logging to stdout/stderr for Docker log aggregation"
  - "Env var pattern: HUEY_DB, OUT_DIR, UPLOADS_DIR all set in docker-compose.yml pointing to /app/data and /app/out mounted volumes"

requirements-completed: [PIPE-01]

# Metrics
duration: 6min
completed: 2026-03-04
---

# Phase 2 Plan 7: Supervisord Multi-Process Docker Container Summary

**supervisord config managing uvicorn (port 8000) + huey_consumer thread worker in single Python 3.11-slim container, with HUEY_DB/OUT_DIR/UPLOADS_DIR env vars wired to Docker volume mounts**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-04T01:37:00Z
- **Completed:** 2026-03-04T01:43:18Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created docker/supervisord.conf with [program:uvicorn] and [program:huey_worker] both logging to stdout for Docker log aggregation
- Updated Dockerfile runtime stage to install supervisor via apt-get, copy uv binary, use supervisord as CMD, and create /app/data/uploads
- Updated docker-compose.yml with HUEY_DB=/app/data/huey.db, OUT_DIR=/app/out, UPLOADS_DIR=/app/data/uploads matching volume mounts

## Task Commits

Each task was committed atomically:

1. **Task 1: Create supervisord.conf and update Dockerfile** - `66686e9` (feat)
2. **Task 2: Update docker-compose.yml with HUEY_DB and OUT_DIR env vars** - `1fb8d8c` (feat)

**Plan metadata:** (see final commit below)

## Files Created/Modified
- `docker/supervisord.conf` - supervisord config with [program:uvicorn] (priority=10) and [program:huey_worker] (priority=20, --workers=1 --worker-type=thread)
- `docker/Dockerfile` - Runtime stage now installs supervisor, copies uv binary, copies supervisord.conf, changes CMD to supervisord, adds /app/data/uploads to mkdir
- `docker/docker-compose.yml` - Adds HUEY_DB, OUT_DIR, UPLOADS_DIR environment variables

## Decisions Made
- `pidfile=/tmp/supervisord.pid` used instead of default `/run/supervisord.pid` to avoid permission issues in the slim container image
- `priority=10` for uvicorn (starts first), `priority=20` for huey_worker (starts after)
- uv binary copied from python-builder stage (`COPY --from=python-builder /bin/uv /bin/uv`) so huey_consumer.py in .venv/bin can be discovered and executed correctly
- pyproject.toml and uv.lock copied to runtime stage alongside run.py/run_web.py for completeness

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- 3 pre-existing test failures in tests/test_runs.py (test_stage_progress_updates, test_run_status_lifecycle, test_failure_alert_created) are TDD RED tests from plans 02-05 and 02-06, confirmed pre-existing before this plan. Not caused by Docker changes.

## User Setup Required

None - no external service configuration required. Docker compose env vars are self-contained in docker-compose.yml.

## Next Phase Readiness
- Docker container now runs uvicorn + huey_consumer together under supervisord — ready for end-to-end integration test in plan 02-08
- All env vars (HUEY_DB, OUT_DIR, UPLOADS_DIR, DATABASE_URL) wired to volume mounts

## Self-Check: PASSED

All files verified present on disk. All task commits verified in git history.

---
*Phase: 02-pipeline-bridge*
*Completed: 2026-03-04*
