---
id: T01
parent: S04
milestone: M001
provides:
  - WeasyPrint installed and importable in venv (weasyprint 68.1)
  - Dockerfile runtime stage includes Pango/HarfBuzz system deps for WeasyPrint
  - Run.parent_run_id nullable FK column (self-referential, for amendments)
  - Run.packet_versions JSON column (for versioned audit packets)
  - ShopConfig.retention_days Integer column (default 30, for cleanup scheduler)
  - run_schema_migrations() startup function for SQLite ALTER TABLE migrations
  - 14 xfail test stubs (PACKET-01..03, WORK-01..04, HISTORY-01..04, AMEND-01..03)
requires: []
affects: []
key_files: []
key_decisions: []
patterns_established: []
observability_surfaces: []
drill_down_paths: []
duration: 2min
verification_result: passed
completed_at: 2026-03-07
blocker_discovered: false
---
# T01: 04-exports-history-and-amendments 01

**# Phase 4 Plan 01: Foundation Summary**

## What Happened

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
