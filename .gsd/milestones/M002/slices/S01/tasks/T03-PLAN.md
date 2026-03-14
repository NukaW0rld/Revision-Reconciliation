---
estimated_steps: 4
estimated_files: 4
---

# T03: Remove column mapping from admin settings and delete unused templates

**Slice:** S01 — Two-step wizard & per-run column mapping
**Milestone:** M002

## Description

Remove the column mapping upload/save routes from admin settings and the corresponding template section. Delete the wizard step 3 and step 4 wrapper templates (keeping `step4_mapping_partial.html` and `step4_error.html` which are reused).

## Steps

1. Read `shop/routers/admin.py` to locate `settings_upload` and `settings_save` route handlers
2. Remove `settings_upload` and `settings_save` from `admin.py` (keep settings GET and retention POST)
3. Edit `shop/templates/admin/settings.html`: remove the column mapping card section (the `<div class="card">` with xlsx upload form and mapping display)
4. Delete `shop/templates/setup/step3_engineer.html` and `shop/templates/setup/step4_column_mapping.html`

## Must-Haves

- [ ] `settings_upload` and `settings_save` routes removed from `admin.py`
- [ ] Column mapping card removed from `settings.html`
- [ ] `step3_engineer.html` and `step4_column_mapping.html` deleted
- [ ] `step4_mapping_partial.html` and `step4_error.html` retained (used by validate-xlsx)
- [ ] Admin settings page renders correctly with retention section only

## Verification

- `uv run pytest tests/test_admin.py -v` — all admin tests pass (no tests cover removed routes)
- Visual: settings page loads without column mapping section

## Inputs

- `shop/routers/admin.py` — current settings routes
- `shop/templates/admin/settings.html` — current settings page with column mapping section

## Expected Output

- `shop/routers/admin.py` — settings_upload and settings_save removed
- `shop/templates/admin/settings.html` — column mapping section removed
- `shop/templates/setup/step3_engineer.html` — deleted
- `shop/templates/setup/step4_column_mapping.html` — deleted

## Observability Impact

**What changes:** `POST /admin/settings/upload` and `POST /admin/settings/save` are removed. Any request to those URLs will return 404. The admin settings page no longer renders a column mapping card.

**How a future agent inspects this task:**
- `grep "settings_upload\|settings_save" shop/routers/admin.py` — should return empty (routes gone)
- `ls shop/templates/setup/` — should NOT contain `step3_engineer.html` or `step4_column_mapping.html`
- `curl -s -o /dev/null -w "%{http_code}" -X POST http://host/admin/settings/upload` — expect 404 (or 302 if not authenticated)
- `uv run pytest tests/test_admin.py tests/test_setup.py -v` — all pass

**Failure surface:** If these routes were called by any remaining template (e.g. settings.html still had the form), requests to the removed endpoints would return 404 in production. The tests guard against template drift by verifying the settings page renders and the test_setup.py xlsx parsing tests continue to pass against the new `/runs/validate-xlsx` endpoint.
