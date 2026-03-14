# T06: 01-foundation 06

**Slice:** S01 — **Milestone:** M001

## Description

Build wizard step 4: Form 3 column mapping. The admin uploads a sample Excel file, sees a preview table with auto-detected column assignments, corrects any mismatches, and saves. Also adds the admin settings panel for post-setup reconfiguration.

Purpose: SETUP-02 and SETUP-03 are the most complex UI in Phase 1 — two-phase validation (fatal errors before UI, column issues in UI) with HTMX partial swap for the mapping table.
Output: Step 4 completable, setup_complete flag set, admin can reconfigure later.

## Must-Haves

- [ ] "POST /setup/step4/upload with valid xlsx returns HTMX partial with mapping dropdowns pre-filled by auto-detect"
- [ ] "POST /setup/step4/upload with empty file returns error HTML before mapping UI is shown"
- [ ] "POST /setup/step4/upload with unreadable file returns error HTML before mapping UI is shown"
- [ ] "Unmatched columns are highlighted in amber with 'Unmatched' in their dropdown"
- [ ] "First 5 rows of uploaded sheet are shown in the preview table"
- [ ] "POST /setup/step4/save stores column mapping JSON to shop_config; sets setup_complete=True; redirects to /dashboard"
- [ ] "Non-contiguous char_no values (e.g., 1, 5, 99, 200) are preserved as-is in the mapping"
- [ ] "Admin can reconfigure column mapping post-setup via GET /admin/settings"

## Files

- `shop/routers/setup.py`
- `shop/services/form3.py`
- `shop/templates/setup/step4_column_mapping.html`
- `shop/templates/setup/step4_error.html`
- `shop/templates/setup/step4_mapping_partial.html`
- `shop/templates/admin/settings.html`
