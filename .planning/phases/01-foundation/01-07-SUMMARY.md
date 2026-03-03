---
phase: 01-foundation
plan: 07
subsystem: infra
tags: [docker, tailwindcss, daisyui, uvicorn, sqlite, multi-stage-build]

# Dependency graph
requires:
  - phase: 01-foundation-01-06
    provides: shop app factory, models, templates, static assets

provides:
  - Three-stage Dockerfile (Node tailwind-builder, uv python-builder, python:3.11-slim runtime)
  - docker-compose.yml with volume mounts and env var configuration
  - run_web.py startup entrypoint seeding admin user and starting uvicorn
  - DATABASE_URL env var support in shop/database.py for container path override

affects: [deploy, docker-compose-build]

# Tech tracking
tech-stack:
  added: [node:22-alpine (build-time only), tailwindcss ^4.0.0 (build-time only), daisyui ^5.0.0 (build-time only)]
  patterns: [multi-stage Docker build isolates Node from runtime, admin seed on startup from env vars]

key-files:
  created:
    - docker/Dockerfile
    - docker/docker-compose.yml
    - run_web.py
    - package.json
  modified:
    - shop/database.py
    - .gitignore

key-decisions:
  - "python:3.11-slim used (not 3.12/3.13) to respect pyproject.toml requires-python >=3.10,<3.13"
  - "Node.js confined to tailwind-builder stage only — not present in runtime image (DEPLOY-03)"
  - "uv sync --locked --no-install-project in builder stage ensures exact lockfile dependency versions (DEPLOY-02)"
  - "data/ dir on host maps to /app/shop.db and /app/out in container for runtime persistence"
  - "DATABASE_URL env var in database.py enables Docker volume path override without code changes"
  - "Admin seeded from ADMIN_EMAIL + ADMIN_DEFAULT_PASSWORD env vars on startup; idempotent (checks for existing admin)"

patterns-established:
  - "Startup script pattern: seed DB state before starting server process via subprocess.run"
  - "Env-driven database URL: os.environ.get('DATABASE_URL', default) for container flexibility"

requirements-completed: [DEPLOY-01, DEPLOY-02, DEPLOY-03]

# Metrics
duration: 4min
completed: 2026-03-03
---

# Phase 1 Plan 7: Docker Multi-Stage Build and Deployment Configuration Summary

**Three-stage Dockerfile (Node tailwind-builder + uv python-builder + python:3.11-slim runtime) with docker-compose volume mounts and admin-seeding entrypoint — no internet access at container runtime**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-03T03:11:25Z
- **Completed:** 2026-03-03T03:15:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Three-stage Dockerfile: Node 22 Alpine compiles Tailwind CSS, uv builder installs Python deps from lockfile, python:3.11-slim runtime contains only what is needed to run
- docker-compose.yml with host data/ directory volume mounts for shop.db and out/, environment variables for admin credentials and DATABASE_URL
- run_web.py startup entrypoint: idempotently seeds default admin user (from ADMIN_EMAIL + ADMIN_DEFAULT_PASSWORD) and ShopConfig singleton, then starts uvicorn on 0.0.0.0:8000
- shop/database.py updated to read DATABASE_URL from environment so the container volume path overrides the default local path

## Task Commits

Each task was committed atomically:

1. **Task 1: Tailwind/DaisyUI build configuration and package.json** - `709cf20` (chore)
2. **Task 2: Dockerfile, docker-compose.yml, and startup entrypoint** - `43bae81` (feat)

## Files Created/Modified

- `package.json` - Node devDependencies: tailwindcss ^4.0.0 and daisyui ^5.0.0; build:css script
- `docker/Dockerfile` - Three-stage build: tailwind-builder (node:22-alpine), python-builder (uv sync --locked), runtime (python:3.11-slim)
- `docker/docker-compose.yml` - Single service; ../data/shop.db and ../data/out volume mounts; env vars for admin and DATABASE_URL
- `run_web.py` - Startup entrypoint: seeds admin + ShopConfig singleton then launches uvicorn
- `shop/database.py` - Added os.environ.get("DATABASE_URL") for container path override
- `.gitignore` - Added data/ and shop.db entries

## Decisions Made

- python:3.11-slim used (not 3.12/3.13) — pyproject.toml requires-python >=3.10,<3.13
- Node.js confined to tailwind-builder stage only; not installed in runtime image (DEPLOY-03 compliant)
- uv sync --locked --no-install-project ensures exact lockfile dependency versions (DEPLOY-02)
- DATABASE_URL env var reads from environment in database.py, enabling Docker volume mount path override
- Admin seed is idempotent: checks for existing admin role before inserting

## Deviations from Plan

None - plan executed exactly as written. The plan included adding data/ to .gitignore which was implemented alongside the other Task 2 changes.

## Issues Encountered

None.

## User Setup Required

None - Docker configuration is complete. Manual `docker compose up --build` verification happens in Plan 08.

## Dockerfile Stage Breakdown

| Stage | Base Image | Purpose | Output |
|-------|-----------|---------|--------|
| tailwind-builder | node:22-alpine | Compile Tailwind CSS + DaisyUI | static/dist/output.css |
| python-builder | python:3.11-slim + uv | Install Python deps from uv.lock | /app/.venv |
| runtime | python:3.11-slim | Run FastAPI application | Port 8000 |

## Volume Mount Paths

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| ../data/shop.db | /app/shop.db | SQLite database persistence |
| ../data/out | /app/out | Pipeline output persistence |

## Environment Variables Required at Runtime

| Variable | Default | Purpose |
|----------|---------|---------|
| ADMIN_EMAIL | admin@shop.local | Admin user email seeded on first start |
| ADMIN_DEFAULT_PASSWORD | changeme | Admin password (change in production) |
| DATABASE_URL | sqlite:////app/shop.db | SQLite path matching volume mount |

## Next Phase Readiness

- Docker infrastructure complete; ready for Plan 08 verification (docker compose up --build)
- No blockers — all runtime assets bundled in image, no internet access required

---
*Phase: 01-foundation*
*Completed: 2026-03-03*

## Self-Check: PASSED

All files verified:
- package.json: FOUND
- docker/Dockerfile: FOUND
- docker/docker-compose.yml: FOUND
- run_web.py: FOUND
- .planning/phases/01-foundation/01-07-SUMMARY.md: FOUND
- Commit 709cf20: FOUND
- Commit 43bae81: FOUND
