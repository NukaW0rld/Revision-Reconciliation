---
id: T04
parent: S01
milestone: M002
provides:
  - test_setup.py with step3/step4 redirect tests and no xlsx endpoint tests
  - test_runs.py with 4 new per-run xlsx mapping tests (autodetect, empty file, noncontiguous char_no, auth gate)
key_files:
  - tests/test_setup.py
  - tests/test_runs.py
key_decisions:
  - Moved test_form3_upload_autodetect/test_empty_file_error/test_noncontiguous_char_no from test_setup.py to test_runs.py — they test /runs/validate-xlsx (a runs endpoint), not setup wizard behavior
  - Split test_removed_step4_redirects into two tests (_when_complete and _when_incomplete) to avoid fixture DB collision — client and client_setup_incomplete share the same db_engine within a test function, so combining them causes the client-seeded setup_complete=True row to be picked up by _get_or_create_config in the second assertion
  - test_validate_xlsx_requires_auth uses the `client` fixture (which has setup_complete=True) without login — no session cookie means no auth → redirect; this is the correct isolation for an auth gate test
patterns_established:
  - When combining client + client_setup_incomplete fixtures in one test, they share the same db_engine; the client fixture seeds setup_complete=True at id=1, which takes precedence over client_setup_incomplete's empty DB. Avoid mixing these two fixtures in a single test function.
  - Use _login_engineer() helper in test_runs.py for all auth-requiring run endpoint tests rather than manual session seeding with UserSession ORM
observability_surfaces:
  - grep "^def test_" tests/test_setup.py tests/test_runs.py — lists all test functions; new tests appear
  - grep "test_form3_upload_autodetect\|test_empty_file_error\|test_noncontiguous_char_no" tests/test_setup.py → empty (moved to test_runs.py)
  - uv run pytest --tb=short — 93 passed, 0 failures, 2 xfailed
duration: ~20min
verification_result: passed
completed_at: 2026-03-13
blocker_discovered: false
---

# T04: Update test suite for new wizard and per-run mapping

**Test suite updated: xlsx mapping tests moved to test_runs.py, step4 redirect tests added, auth gate test added — 93 passed.**

## What Happened

Read existing test_setup.py and test_runs.py. Found that T03 had already migrated the xlsx parsing tests (test_form3_upload_autodetect, test_empty_file_error, test_noncontiguous_char_no) to hit `/runs/validate-xlsx` but left them in test_setup.py. T04 moved them to test_runs.py using the `_login_engineer()` helper pattern, and added `test_validate_xlsx_noncontiguous_char_no` and `test_validate_xlsx_requires_auth` as new tests.

Added step4 redirect tests to test_setup.py (`test_removed_step4_redirects_when_complete`, `test_removed_step4_redirects_when_incomplete`). Discovered a fixture DB collision: combining `client` and `client_setup_incomplete` in one test function causes the `client` fixture's seeded `setup_complete=True` row to be found by `_get_or_create_config`, making the incomplete-path assertion fail. Fixed by splitting into two separate test functions.

The step2 completion coverage was already present via `test_setup_step2_creates_admin` (checks `setup_complete=True` in DB + redirect to `/login`). No new test needed — existing test covers the must-have.

## Verification

```
uv run pytest --tb=short
# 93 passed, 2 xfailed, 0 failures

uv run pytest tests/test_setup.py tests/test_runs.py
# 28 passed (12 setup, 16 runs)
```

All must-haves confirmed:
- step3/step4 xlsx tests removed from test_setup.py ✓
- test_setup_step2_creates_admin covers setup_complete=True ✓  
- test_removed_step4_redirects_when_complete + _when_incomplete in test_setup.py ✓
- test_validate_xlsx_autodetect in test_runs.py ✓
- test_validate_xlsx_empty_file in test_runs.py ✓
- test_validate_xlsx_requires_auth in test_runs.py ✓
- 93 ≥ 87 ✓

## Diagnostics

- `grep "^def test_" tests/test_setup.py tests/test_runs.py` — lists all test names
- `grep "test_form3_upload_autodetect\|test_empty_file_error\|test_noncontiguous_char_no" tests/test_setup.py` → empty (moved)
- Fixture DB collision pattern: if `test_removed_step4_redirects_when_incomplete` fails with `/login` instead of `/setup/`, the `client` fixture has seeded `setup_complete=True` into the shared db_engine. Never combine `client` and `client_setup_incomplete` in one test function.
- Auth gate: if `test_validate_xlsx_requires_auth` fails with 200 instead of 302, the `get_current_user` dependency is not enforcing auth on `/runs/validate-xlsx`.

## Deviations

- Existing tests `test_setup_step3_redirects_to_login_when_complete` and `test_setup_step3_redirects_to_setup_when_incomplete` were already present from a prior task — these satisfy the "test_removed_step3_redirects" must-have without renaming.
- The plan said to add `test_validate_xlsx_noncontiguous_char_no` implicitly via moving test_noncontiguous_char_no — done but renamed to follow the new `test_validate_xlsx_*` naming convention.
- test_removed_step4_redirects was split into two tests to avoid fixture DB collision (see key_decisions).

## Known Issues

None.

## Files Created/Modified

- `tests/test_setup.py` — removed 3 xlsx tests (moved to test_runs.py); added test_removed_step4_redirects_when_complete and test_removed_step4_redirects_when_incomplete
- `tests/test_runs.py` — added 4 new tests: test_validate_xlsx_autodetect, test_validate_xlsx_empty_file, test_validate_xlsx_noncontiguous_char_no, test_validate_xlsx_requires_auth; added _make_form3_xlsx_bytes() helper
- `.gsd/milestones/M002/slices/S01/tasks/T04-PLAN.md` — added Observability Impact section (pre-flight fix)
