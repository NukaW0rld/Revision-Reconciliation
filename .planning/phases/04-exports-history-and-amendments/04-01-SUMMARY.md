---
phase: 04-exports-history-and-amendments
plan: "01"
subsystem: database, infra, testing
tags: [weasyprint, sqlalchemy, sqlite, migration, pytest, xfail]

requires:
  - phase: 03-review-and-sign-off
    provides: Run model with reviewer/sign-off fields that new columns extend

provides:
  - WeasyPrint installed and importable in venv (weasyprint 68.1)
  - Dockerfile runtime stage includes Pango/HarfBuzz system deps for WeasyPrint
  - Run.parent_run_id nullable FK column (self-referential, for amendments)
  - Run.packet_versions JSON column (for versioned audit packets)
  - ShopConfig.retention_days Integer column (default 30, for cleanup scheduler)
  - run_schema_migrations() startup function for SQLite ALTER TABLE migrations
  - 14 xfail test stubs (PACKET-01..03, WORK-01..04, HISTORY-01..04, AMEND-01..03)

affects:
  - 04-02 (audit packet PDF export — needs weasyprint + Run.packet_versions)
  - 04-03 (work order export — needs weasyprint)
  - 04-04 (run history/retention — needs ShopConfig.retention_days)
  - 04-05 (amendments — needs Run.parent_run_id + Run.packet_versions)

tech-stack:
  added:
    - weasyprint==68.1 (PDF generation from HTML/CSS)
  patterns:
    - run_schema_migrations(): SQLite ALTER TABLE guard pattern — inspects existing columns, adds only missing ones; called after create_all() in create_app()

key-files:
  created:
    - tests/test_exports.py
    - tests/test_history.py
    - tests/test_amendments.py
  modified:
    - pyproject.toml (added weasyprint dependency)
    - uv.lock (resolved weasyprint 68.1 + transitive deps)
    - docker/Dockerfile (added libpango-1.0-0, libpangoft2-1.0-0, libharfbuzz-subset0 to runtime apt-get)
    - shop/models.py (parent_run_id FK + packet_versions JSON on Run; retention_days Integer on ShopConfig)
    - shop/database.py (run_schema_migrations function + text/inspect imports)
    - shop/app.py (run_schema_migrations(engine) call after create_all)

key-decisions:
  - "weasyprint added unpinned — let uv resolve latest compatible (resolved 68.1)"
  - "run_schema_migrations uses sa_inspect + ALTER TABLE per column — idempotent guard prevents double-add on repeated startup"
  - "parent_run_id is a nullable self-referential FK on runs — supports amendment lineage chains without a separate table"
  - "packet_versions is JSON (not a separate table) — stores ordered list of packet metadata snapshots, sufficient for Phase 4 audit trail needs"
  - "retention_days on ShopConfig (not a separate config table) — consistent with existing column_mapping/wizard_step pattern"

patterns-established:
  - "Startup migration pattern: run_schema_migrations(engine) called in create_app() after Base.metadata.create_all() — ensures existing Docker deployments gain new columns without manual migration step"
  - "xfail(strict=False) stubs in named test files — test names visible in collection, xfail count verifiable, requirements traceable by test name prefix"

requirements-completed:
  - PACKET-01
  - PACKET-02
  - PACKET-03
  - WORK-01
  - WORK-02
  - WORK-03
  - WORK-04
  - HISTORY-01
  - HISTORY-02
  - HISTORY-03
  - HISTORY-04
  - AMEND-01
  - AMEND-02
  - AMEND-03

duration: 2min
completed: 2026-03-07
---

# Phase 4 Plan 01: Foundation Summary

**WeasyPrint installed with Pango/HarfBuzz Docker deps; Run extended with parent_run_id + packet_versions FKs; ShopConfig gains retention_days; startup SQLite migration guard added; 14 Phase 4 xfail test stubs created**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-07T02:40:03Z
- **Completed:** 2026-03-07T02:42:35Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- WeasyPrint 68.1 installed via `uv sync` and verified importable (`from weasyprint import HTML` passes)
- Dockerfile runtime stage extended with three Pango/HarfBuzz apt packages needed for WeasyPrint to render fonts in Docker
- Three new DB columns added to existing SQLAlchemy models: `Run.parent_run_id` (self-ref FK), `Run.packet_versions` (JSON), `ShopConfig.retention_days` (Integer default 30)
- `run_schema_migrations(eng)` function added to `shop/database.py` — idempotent SQLite ALTER TABLE guard called at startup so existing shop.db deployments gain new columns without downtime
- 14 xfail test stubs created across three files, one per Phase 4 requirement group (exports, history, amendments)
- Full test suite: 70 passed, 16 xfailed (pre-existing), 14 new xfailed stubs, 0 failures

## Task Commits

1. **Task 1: Install WeasyPrint and add Dockerfile system deps** - `1d568bc` (chore)
2. **Task 2: Add DB columns, startup migration, xfail test stubs** - `685b178` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `pyproject.toml` - added `"weasyprint"` to dependencies list
- `uv.lock` - resolved weasyprint 68.1 + transitive deps (fonttools, pillow, pydyf, pyphen, tinycss2, tinyhtml5, webencodings, zopfli)
- `docker/Dockerfile` - runtime stage apt-get extended with libpango-1.0-0, libpangoft2-1.0-0, libharfbuzz-subset0
- `shop/models.py` - Run: parent_run_id (nullable ForeignKey("runs.id")), packet_versions (JSON nullable); ShopConfig: retention_days (Integer nullable default 30)
- `shop/database.py` - imports text/inspect; adds run_schema_migrations() function
- `shop/app.py` - calls run_schema_migrations(engine) after Base.metadata.create_all()
- `tests/test_exports.py` - 7 xfail stubs: PACKET-01..03, WORK-01..04
- `tests/test_history.py` - 4 xfail stubs: HISTORY-01..04
- `tests/test_amendments.py` - 3 xfail stubs: AMEND-01..03

## Decisions Made

- WeasyPrint added unpinned — uv resolves latest compatible (resolved 68.1); no version constraint needed for this internal tool
- `run_schema_migrations` uses `sa_inspect` to check existing column names before issuing ALTER TABLE — prevents errors on repeated startup with up-to-date schema
- `parent_run_id` is a self-referential nullable FK on the `runs` table — supports amendment lineage without a separate table, sufficient for Phase 4 scope
- `packet_versions` stored as JSON list on `Run` — ordered list of packet metadata snapshots; avoids a separate versioning table for Phase 4 audit trail requirements

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- WeasyPrint foundation ready for Plans 02 (audit packet PDF) and 03 (work order PDF)
- DB schema columns ready for Plans 04 (history/retention) and 05 (amendments)
- All 14 Phase 4 requirements have failing test stubs — plans 02-05 will implement and replace these stubs with real tests

---
*Phase: 04-exports-history-and-amendments*
*Completed: 2026-03-07*
