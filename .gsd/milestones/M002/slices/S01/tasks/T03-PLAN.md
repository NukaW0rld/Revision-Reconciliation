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
