---
id: T03
parent: S01
milestone: M002
provides:
  - admin settings page with retention section only (column mapping removed)
  - settings_upload and settings_save routes deleted from admin.py
  - step3_engineer.html and step4_column_mapping.html deleted
  - test_setup.py SETUP-02/03/04 migrated from removed /setup/step4/upload to /runs/validate-xlsx
key_files:
  - shop/routers/admin.py
  - shop/templates/admin/settings.html
  - tests/test_setup.py
key_decisions:
  - SETUP-02/03/04 tests were pointing at the old /setup/step4/upload endpoint removed in T01; migrated them to /runs/validate-xlsx (T02's endpoint) rather than deleting them — the xlsx parsing behavior is the same, the tests remain valid
patterns_established:
  - When removing a route, scan test files for direct URL references to that route and update them to the replacement endpoint
observability_surfaces:
  - "grep 'settings_upload|settings_save' shop/routers/admin.py — should return empty"
  - "ls shop/templates/setup/ — must NOT contain step3_engineer.html or step4_column_mapping.html"
  - "POST /admin/settings/upload returns 404 in production (route removed)"
  - "uv run pytest tests/test_admin.py tests/test_setup.py -v — all pass"
duration: ~20 minutes
verification_result: passed
completed_at: 2026-03-13
blocker_discovered: false
---

# T03: Remove column mapping from admin settings and delete unused templates

**Admin settings page now shows retention section only; xlsx parsing tests migrated to /runs/validate-xlsx.**

## What Happened

Removed `settings_upload` and `settings_save` route handlers from `admin.py` and stripped the associated `File`/`UploadFile` imports (no longer needed). Removed the column mapping card from `settings.html` — the page now renders shop name and run retention sections only. Deleted `step3_engineer.html` and `step4_column_mapping.html`; `step4_mapping_partial.html` and `step4_error.html` were retained (still used by `/runs/validate-xlsx`).

Found that 3 tests in `test_setup.py` (SETUP-02, SETUP-03, SETUP-04) were hitting `/setup/step4/upload` — the wizard step removed in T01. They were returning 302 (redirect) instead of 200. Migrated them to `/runs/validate-xlsx` with field name `form3_xlsx` and removed the `wizard_step=3` seeding that was only needed for the old wizard guard. All 3 now pass cleanly.

## Verification

- `uv run pytest tests/test_admin.py -v` — 5/5 passed
- `uv run pytest tests/test_setup.py -v` — 13/13 passed (including migrated SETUP-02/03/04)
- `uv run pytest --tb=short` — 90 passed, 2 xfailed, 0 failures (up from 87 due to SETUP-02/03/04 fix)
- `ls shop/templates/setup/` confirms step3_engineer.html and step4_column_mapping.html absent; step4_error.html and step4_mapping_partial.html present

## Diagnostics

- `grep "settings_upload\|settings_save" shop/routers/admin.py` → empty (routes gone)
- `ls shop/templates/setup/` → step1_shop_name.html, step2_password.html, step4_error.html, step4_mapping_partial.html, wizard_layout.html
- `POST /admin/settings/upload` returns 404 in production (unauthenticated: may 302 first)
- xlsx parsing tests now exercise `/runs/validate-xlsx` with `form3_xlsx` field name

## Deviations

Three test_setup.py tests (SETUP-02, SETUP-03, SETUP-04) were failing because they targeted the removed `/setup/step4/upload` endpoint. The plan didn't mention updating tests, but fixing them was necessary to meet the ≥87 tests passing slice requirement. Tests were migrated to `/runs/validate-xlsx` rather than deleted — the xlsx parsing behavior they verify is unchanged.

## Known Issues

none

## Files Created/Modified

- `shop/routers/admin.py` — removed settings_upload and settings_save routes; removed File/UploadFile imports
- `shop/templates/admin/settings.html` — removed column mapping card (lines 64–103)
- `shop/templates/setup/step3_engineer.html` — deleted
- `shop/templates/setup/step4_column_mapping.html` — deleted
- `tests/test_setup.py` — migrated SETUP-02/03/04 from /setup/step4/upload to /runs/validate-xlsx; removed wizard_step seeding
- `.gsd/milestones/M002/slices/S01/tasks/T03-PLAN.md` — added Observability Impact section
