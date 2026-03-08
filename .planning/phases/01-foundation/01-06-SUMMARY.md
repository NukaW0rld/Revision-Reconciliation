---
phase: 01-foundation
plan: 06
subsystem: ui
tags: [htmx, jinja2, openpyxl, form3, setup-wizard, admin]

# Dependency graph
requires:
  - phase: 01-foundation/01-05
    provides: Setup wizard router (steps 1-3), wizard_layout template, ShopConfig wizard_step tracking
  - phase: 01-foundation/01-02
    provides: ShopConfig model with column_mapping JSON field and setup_complete flag

provides:
  - Form 3 column mapping service (shop/services/form3.py) wrapping FORM3_HEADER_KEYWORDS
  - Step 4 wizard completion: POST /setup/step4/upload (HTMX partial), POST /setup/step4/save
  - Admin settings panel: GET/POST /admin/settings, /settings/upload, /settings/save
  - Two-phase validation: fatal errors before mapping UI, column mismatches within UI
  - setup_complete=True set atomically with column_mapping on wizard save

affects:
  - Phase 2 pipeline (reads ShopConfig.column_mapping to parse uploaded Form 3 files)
  - Phase 3 audit (setup_complete gate must be passed before runs are permitted)

# Tech tracking
tech-stack:
  added: [openpyxl (already in delta_preservation dependency chain)]
  patterns:
    - HTMX multipart file upload with hx-encoding=multipart/form-data and innerHTML swap
    - Two-phase validation: ValueError = fatal (error partial), missing detection = non-fatal (amber highlight)
    - Shared partial template (step4_mapping_partial.html) parameterized by form_action for reuse in wizard and admin

key-files:
  created:
    - shop/services/form3.py
    - shop/templates/setup/step4_column_mapping.html
    - shop/templates/setup/step4_mapping_partial.html
    - shop/templates/setup/step4_error.html
    - shop/templates/admin/settings.html
  modified:
    - shop/routers/setup.py
    - shop/routers/admin.py
    - tests/test_setup.py

key-decisions:
  - "step4_mapping_partial.html reused for both wizard (/setup/step4/save) and admin settings (/admin/settings/save) via form_action template variable — avoids duplication"
  - "Jinja2 set+update pattern used to build detected_by_col inverse dict in template (col_idx -> field) since Jinja2 lacks dict comprehension over items()"
  - "Unmatched columns get select name=col_{idx} (not a required_field name) so the save endpoint ignores them — no JS needed to dynamically rename selects"

patterns-established:
  - "HTMX file upload: hx-encoding=multipart/form-data on form, hx-target on container div, bare HTML partial returned (no base.html extension)"
  - "Fatal error partial (step4_error.html) returned as 200 OK — HTMX treats all 2xx as success for DOM swap"
  - "Shared partial template pattern: pass form_action as context variable to enable dual-destination forms"

requirements-completed: [SETUP-02, SETUP-03, SETUP-04, AUTH-05]

# Metrics
duration: 3min
completed: 2026-03-03
---

# Phase 1 Plan 06: Form 3 Column Mapping Wizard Summary

**HTMX two-phase Form 3 column mapping wizard (step 4): upload Excel, auto-detect columns via FORM3_HEADER_KEYWORDS, amber-highlight mismatches, save mapping to ShopConfig and set setup_complete=True**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-03T03:05:20Z
- **Completed:** 2026-03-03T03:08:00Z
- **Tasks:** 2 (Task 1 TDD: RED + GREEN, Task 2: templates + admin routes)
- **Files modified:** 8

## Accomplishments
- Form 3 service with detect_column_mapping() wrapping existing FORM3_HEADER_KEYWORDS (no detection logic rewrite)
- parse_excel_preview() with two-phase error model: fatal ValueError for empty/unreadable files, omit-from-dict for undetected columns
- Step 4 routes: GET /setup/step4, POST /setup/step4/upload (HTMX partial swap), POST /setup/step4/save (atomic setup_complete flag)
- Admin settings panel with column mapping reconfiguration at GET/POST /admin/settings
- Shared step4_mapping_partial.html reused by both wizard and admin settings via form_action variable

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests** - `f320594` (test)
2. **Task 1 GREEN: Form 3 service + step4 routes** - `987b3cc` (feat)
3. **Task 2: Templates + admin settings panel** - `d20b668` (feat)

_Note: TDD task split into RED + GREEN commits per TDD execution protocol_

## Files Created/Modified
- `shop/services/form3.py` - detect_column_mapping() + parse_excel_preview() service functions
- `shop/routers/setup.py` - Added GET /step4, POST /step4/upload, POST /step4/save routes
- `shop/routers/admin.py` - Added GET /settings, POST /settings/name, /settings/upload, /settings/save
- `shop/templates/setup/step4_column_mapping.html` - Wizard step 4 page with HTMX upload form
- `shop/templates/setup/step4_mapping_partial.html` - Preview table + column dropdown UI (shared partial)
- `shop/templates/setup/step4_error.html` - Fatal error partial (DaisyUI alert-error)
- `shop/templates/admin/settings.html` - Admin settings page with shop name + column mapping reconfiguration
- `tests/test_setup.py` - Replaced 3 xfail stubs with real TDD test implementations

## Decisions Made
- step4_mapping_partial.html reused for both wizard and admin settings via form_action template variable — eliminates template duplication for identical UI
- Jinja2 update() filter used to invert detected dict in template (field->col_idx to col_idx->field) since Jinja2 templates lack dict comprehensions
- Unmatched columns get select name=col_{idx} (not a required field name) — save endpoint's field-name loop naturally ignores them without JavaScript manipulation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. Template Jinja2 update() dict inversion pattern required minor template-side logic but was solved cleanly.

## User Setup Required

None - no external service configuration required.

## Test Results

```
tests/test_setup.py ..........  [10 passed]
Full suite: 21 passed, 2 xfailed, 3 xpassed in 4.69s
```

All setup tests pass including:
- test_form3_upload_autodetect: valid xlsx returns mapping partial with select dropdowns
- test_empty_file_error: empty bytes returns error partial, no select dropdowns
- test_noncontiguous_char_no: char_no values [1, 5, 99, 200] all appear in preview

## Wizard Flow Verification

Full wizard flow completable: steps 1 -> 2 -> 3 -> 4 -> /dashboard
- Each step enforces prior-step completion via wizard_step guard
- Step 4 sets setup_complete=True atomically with column_mapping save
- Admin can reconfigure post-setup via /admin/settings

## Next Phase Readiness
- Setup wizard fully complete (all 4 steps)
- ShopConfig.column_mapping available for Phase 2 pipeline integration
- Admin settings panel ready for Phase 2/3 reconfiguration needs
- No blockers

## Self-Check: PASSED

Files verified present:
- shop/services/form3.py: FOUND
- shop/routers/setup.py: FOUND
- shop/routers/admin.py: FOUND
- shop/templates/setup/step4_column_mapping.html: FOUND
- shop/templates/setup/step4_mapping_partial.html: FOUND
- shop/templates/setup/step4_error.html: FOUND
- shop/templates/admin/settings.html: FOUND

Commits verified:
- f320594 (test RED): FOUND
- 987b3cc (feat GREEN): FOUND
- d20b668 (feat templates): FOUND

---
*Phase: 01-foundation*
*Completed: 2026-03-03*
