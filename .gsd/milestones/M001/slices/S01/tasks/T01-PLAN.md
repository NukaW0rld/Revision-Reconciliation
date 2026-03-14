# T01: 01-foundation 01

**Slice:** S01 — **Milestone:** M001

## Description

Create the test infrastructure scaffold for Phase 1 before any production code is written. This establishes the pytest fixtures, in-memory SQLite override, and test stubs that all subsequent plans' verify steps depend on.

Purpose: Without this Wave 0 plan, no automated verify command can pass — every subsequent plan's `<automated>` command references test files that don't yet exist.
Output: Working pytest setup with TestClient fixture; all test files importable with stubs that fail gracefully (pytest.skip or xfail) until production code arrives.

## Must-Haves

- [ ] "pytest discovers and collects all test files without import errors"
- [ ] "In-memory SQLite test DB is created and torn down per test function"
- [ ] "TestClient fixture provides a working FastAPI app with DB override"
- [ ] "All test files have stubs for their requirement-mapped behaviors"

## Files

- `pyproject.toml`
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_auth.py`
- `tests/test_rbac.py`
- `tests/test_models.py`
- `tests/test_setup.py`
- `tests/test_admin.py`
