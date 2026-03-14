# T01: 04-exports-history-and-amendments 01

**Slice:** S04 — **Milestone:** M001

## Description

Install WeasyPrint, add Dockerfile system dependencies, add new DB columns for amendment and packet versioning, add startup schema migration, and create xfail test stubs for all Phase 4 requirements.

Purpose: Wave 1 foundation — every subsequent plan needs the schema and dependencies to be present before implementing exports, history, and amendment features.
Output: Updated pyproject.toml, Dockerfile, models.py, database.py, and three test files with xfail stubs.

## Must-Haves

- [ ] "WeasyPrint is importable in the venv (uv sync installs it)"
- [ ] "Dockerfile runtime stage has Pango/HarfBuzz system deps for WeasyPrint"
- [ ] "Run model has parent_run_id (nullable FK) and packet_versions (JSON) columns"
- [ ] "ShopConfig model has retention_days (Integer, default 30) column"
- [ ] "DB migration function adds new columns to existing SQLite DBs on startup"
- [ ] "All three test files exist with xfail stubs for every requirement"

## Files

- `pyproject.toml`
- `docker/Dockerfile`
- `shop/models.py`
- `shop/database.py`
- `tests/test_exports.py`
- `tests/test_history.py`
- `tests/test_amendments.py`
