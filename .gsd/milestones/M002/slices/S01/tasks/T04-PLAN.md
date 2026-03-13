---
estimated_steps: 6
estimated_files: 2
---

# T04: Update test suite for new wizard and per-run mapping

**Slice:** S01 — Two-step wizard & per-run column mapping
**Milestone:** M002

## Description

Remove tests for deleted wizard steps, add tests for 2-step wizard completion and graceful redirects, and add per-run xlsx column mapping confirmation tests. The full suite must pass with ≥87 tests.

## Steps

1. Read `tests/test_setup.py` to identify all step 3/4 tests to remove
2. Remove step 3/4 tests: `test_setup_step3_blocks_skip`, `test_form3_upload_autodetect`, `test_empty_file_error`, `test_noncontiguous_char_no` (and any others hitting `/setup/step3` or `/setup/step4/*`)
3. Add wizard completion tests in `test_setup.py`: (a) `test_step2_completes_setup` — POST step 2 with valid password, assert `setup_complete=True` in DB and redirect to `/login`; (b) `test_removed_step3_redirects` — GET `/setup/step3` after setup complete, assert 302/307 redirect; (c) `test_removed_step4_redirects` — GET `/setup/step4/upload` after setup complete, assert redirect
4. Read `tests/test_runs.py` to understand existing test patterns and session cookie usage
5. Add per-run mapping tests in `test_runs.py`: (a) `test_validate_xlsx_autodetect` — POST valid xlsx to `/runs/validate-xlsx` with auth cookie, assert 200 and response contains select elements with detected field assignments; (b) `test_validate_xlsx_empty_file` — POST invalid/empty xlsx, assert error response (reuses `step4_error.html`); (c) `test_validate_xlsx_requires_auth` — POST without auth cookie, assert redirect to login
6. Run full suite: `uv run pytest --tb=short` — verify ≥87 pass, 0 failures

## Must-Haves

- [ ] All step 3/4 setup tests removed
- [ ] Test for step 2 setting `setup_complete=True` 
- [ ] Tests for removed step routes redirecting (not 404)
- [ ] Test for successful xlsx auto-detection via `/runs/validate-xlsx`
- [ ] Test for xlsx validation error handling
- [ ] Test for auth requirement on validate-xlsx endpoint
- [ ] Full suite: `uv run pytest --tb=short` passes with ≥87 tests

## Verification

- `uv run pytest --tb=short` — 0 failures, ≥87 tests passed
- `uv run pytest tests/test_setup.py tests/test_runs.py -v` — all new tests visible and passing

## Inputs

- `tests/test_setup.py` — existing step 3/4 tests to remove; step 1/2 test patterns to follow
- `tests/test_runs.py` — existing run test patterns; session cookie helper
- `tests/conftest.py` — shared fixtures

## Expected Output

- `tests/test_setup.py` — step 3/4 tests removed; new wizard completion + redirect tests added
- `tests/test_runs.py` — new per-run xlsx mapping tests added

## Observability Impact

Test changes are not runtime artifacts, but their outcomes are observable via:

- **Suite pass/fail:** `uv run pytest --tb=short` — authoritative signal; ≥87 pass, 0 failures confirms the slice's test contract is intact.
- **New test names:** `grep "^def test_" tests/test_setup.py tests/test_runs.py` lists all test functions; new tests `test_removed_step4_redirects_when_complete`, `test_removed_step4_redirects_when_incomplete`, `test_validate_xlsx_autodetect`, `test_validate_xlsx_empty_file`, `test_validate_xlsx_noncontiguous_char_no`, `test_validate_xlsx_requires_auth` should all appear.
- **Removed tests confirmed gone:** `grep "test_form3_upload_autodetect\|test_empty_file_error\|test_noncontiguous_char_no" tests/test_setup.py` → empty (those tests now live in test_runs.py).
- **Auth gate check:** `test_validate_xlsx_requires_auth` directly verifies that unauthenticated requests to `/runs/validate-xlsx` are rejected; if the auth middleware is misconfigured, this test fails.
- **Failure shape:** A failing `test_removed_step4_redirects_when_incomplete` likely means `ShopConfig` default `setup_complete` is not `False`, or multiple config rows exist (fixture isolation issue). Check with `SELECT id, setup_complete FROM shop_config;` in the test DB.
