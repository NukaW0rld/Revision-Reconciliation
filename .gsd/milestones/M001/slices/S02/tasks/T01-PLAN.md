# T01: 02-pipeline-bridge 01

**Slice:** S02 — **Milestone:** M001

## Description

Lay the database and task queue foundation that every other Phase 2 plan depends on.

Purpose: All subsequent plans require Run/RunAlert models, the Huey instance, and test scaffolding. This plan creates that shared foundation so Plans 02-07 can proceed independently in Wave 2.
Output: Run + RunAlert DB models, SqliteHuey instance in shop/tasks.py, test stubs in tests/test_runs.py, updated conftest.py fixtures, huey added to pyproject.toml.

## Must-Haves

- [ ] "Run model exists in DB with all required columns (status, current_stage, failure fields, warning fields, output_dir, reviewer)"
- [ ] "RunAlert model exists in DB linked to Run and User"
- [ ] "Huey SqliteHuey instance is importable from shop.tasks and uses absolute HUEY_DB path"
- [ ] "test_runs.py test stubs are collected by pytest (xfail, not skipped)"
- [ ] "conftest.py huey_immediate fixture exists for test isolation"
- [ ] "uv add huey succeeds and huey appears in pyproject.toml"

## Files

- `shop/models.py`
- `shop/tasks.py`
- `tests/conftest.py`
- `tests/test_runs.py`
- `pyproject.toml`
