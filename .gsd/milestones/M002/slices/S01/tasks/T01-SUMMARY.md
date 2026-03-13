---
id: T01
parent: S01
milestone: M002
provides:
  - 2-step setup wizard (shop name → admin password → /login)
  - setup_complete=True set at step 2 completion
  - /setup/step3 and /setup/step4/* redirect gracefully instead of 404
  - wizard_layout.html progress bar shows 2 steps
  - step2_password.html submit button reads "Complete Setup"
  - test_setup_step1_to_step2 and test_setup_step2_creates_admin tests added
key_files:
  - shop/routers/setup.py
  - shop/templates/setup/wizard_layout.html
  - shop/templates/setup/step2_password.html
  - tests/test_setup.py
key_decisions:
  - step3/step4 redirects use setup_complete flag (not wizard_step) to decide /login vs /setup/
  - step2_post commits wizard_step and setup_complete in a single db.commit() to avoid partial state
  - unused imports (BytesIO, File, UploadFile, IntegrityError, parse_excel_preview, REQUIRED_FIELDS) removed from setup.py
patterns_established:
  - catch-all redirect routes (GET + POST) for removed wizard steps to avoid 404
observability_surfaces:
  - INFO log "Setup wizard complete: admin password set, setup_complete=True" on step2_post success
  - DB: SELECT setup_complete, wizard_step FROM shop_config WHERE id=1 — post-setup: 1, 2
  - curl -sI /setup/step3 | grep Location → /login (complete) or /setup/ (incomplete)
duration: 25m
verification_result: passed
completed_at: 2026-03-13
blocker_discovered: false
---

# T01: Strip wizard to 2 steps and update guard logic

**Wizard stripped to 2 steps; step 3/4 removed and replaced with graceful redirect routes; setup completes at step 2 with `setup_complete=True`.**

## What Happened

Removed `step3_get`, `step3_post`, `step4_get`, `step4_upload`, and `step4_save` route handlers from `setup.py`. Replaced with four catch-all redirect routes (GET + POST for step3 and step4) that redirect to `/login` when setup is complete or `/setup/` when not. Modified `step2_post` to set `setup_complete=True` and redirect to `/login` instead of `/setup/step3`. Both DB fields (`wizard_step=2` and `setup_complete=True`) are written in a single `db.commit()` call to avoid partial state. Updated `setup_root` cap from 4 to 2. Updated wizard layout to 2-step progress bar and "of 2" title. Changed step 2 submit button text to "Complete Setup". Cleaned up now-unused imports from setup.py. Added `test_setup_step1_to_step2` and `test_setup_step2_creates_admin` tests, plus two new redirect-behavior tests.

## Verification

- `uv run pytest tests/test_setup.py::test_setup_step1_to_step2 -v` — PASSED
- `uv run pytest tests/test_setup.py::test_setup_step2_creates_admin -v` — PASSED
- `uv run pytest tests/test_setup.py -v` — 10 passed, 3 failed (expected: `test_form3_upload_autodetect`, `test_empty_file_error`, `test_noncontiguous_char_no` test old step4 upload behavior — slated for removal in T04)
- `uv run pytest --tb=short` — 87 passed, 3 failed (same 3 step4 tests), 2 xfailed — meets ≥87 threshold

## Diagnostics

- `grep "Setup wizard complete" <logfile>` — confirms step2_post completed successfully
- `SELECT setup_complete, wizard_step FROM shop_config WHERE id=1;` — expect `1, 2` after setup
- `curl -sI http://host/setup/step3 | grep Location` — expect `/login` (complete) or `/setup/` (incomplete)
- Failure shape: password mismatch/default-password → 200 with error in template; DB error → 500 (exception propagates)

## Deviations

None. All steps followed as planned. The unused-import cleanup was incidental hygiene not in the plan but caused no risk.

## Known Issues

Three existing tests (`test_form3_upload_autodetect`, `test_empty_file_error`, `test_noncontiguous_char_no`) now fail because they POST to `/setup/step4/upload` which returns 302 instead of 200. These tests cover removed functionality and are scheduled for removal/replacement in T04.

## Files Created/Modified

- `shop/routers/setup.py` — removed step3/step4 handlers, added redirect routes, step2_post now completes setup, cap updated to 2, unused imports removed
- `shop/templates/setup/wizard_layout.html` — 2-step progress bar, "of 2" title
- `shop/templates/setup/step2_password.html` — "Complete Setup" button text
- `tests/test_setup.py` — added test_setup_step1_to_step2, test_setup_step2_creates_admin, test_setup_step3_redirects_to_login_when_complete, test_setup_step3_redirects_to_setup_when_incomplete
- `.gsd/milestones/M002/slices/S01/S01-PLAN.md` — added Observability / Diagnostics section
- `.gsd/milestones/M002/slices/S01/tasks/T01-PLAN.md` — added Observability Impact section
