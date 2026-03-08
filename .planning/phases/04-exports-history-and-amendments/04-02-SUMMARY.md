---
phase: 04-exports-history-and-amendments
plan: "02"
subsystem: exports
tags: [weasyprint, pdf, csv, fastapi, jinja2, audit-packet, sign-off]

# Dependency graph
requires:
  - phase: 04-exports-history-and-amendments-01
    provides: Run.packet_versions JSON column and parent_run_id for amendment lineage
  - phase: 03-review-and-sign-off
    provides: attempt_sign_off two-phase write atomicity pattern and ReviewItem model

provides:
  - shop/services/exports.py with generate_audit_packet_csv, render_audit_packet_pdf, generate_and_store_audit_packet
  - shop/routers/exports.py with GET /exports/{run_id}/audit-packet.pdf and .csv
  - shop/templates/exports/audit_packet.html WeasyPrint template (cover + summary + per-item cards)
  - PDF generation integrated into attempt_sign_off before signed_off status commit
  - Download Audit Packet (PDF/CSV) buttons on status.html for signed_off runs

affects:
  - 04-03-work-order (uses exports router pattern)
  - 04-04-amendment (uses generate_and_store_audit_packet for v2+ packets)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Inline import inside try block for circular-import-safe Phase 4 integration"
    - "unittest.mock.patch for mocking inline-imported functions in tests"
    - "Re-download from stored path (FileResponse) with fallback re-render (StreamingResponse)"

key-files:
  created:
    - shop/services/exports.py
    - shop/routers/exports.py
    - shop/templates/exports/audit_packet.html
  modified:
    - shop/services/review.py
    - shop/app.py
    - shop/templates/runs/status.html
    - tests/test_exports.py
    - tests/test_review.py

key-decisions:
  - "Inline import of generate_and_store_audit_packet inside attempt_sign_off try block avoids circular import (exports.py imports shop.app.templates which imports review.py via router)"
  - "unittest.mock.patch targets shop.services.exports.generate_and_store_audit_packet — inline from-import resolves attribute at call time inside the patch context manager"
  - "base_url for WeasyPrint set to output_dir/snippets/ so basename img paths resolve; template uses | basename filter (already registered globally)"
  - "Re-download route checks packet_versions[0].path existence first; falls back to re-render for test environments without real output_dir"

patterns-established:
  - "WeasyPrint render: HTML(string=..., base_url=str(snippets_dir)).write_pdf()"
  - "CSV export: csv.DictWriter to io.StringIO, seek(0) before return, StreamingResponse with attachment header"
  - "Audit packet stored at output_dir/packets/v{N}.pdf with version metadata in packet_versions JSON list"

requirements-completed: [PACKET-01, PACKET-02, PACKET-03]

# Metrics
duration: 6min
completed: 2026-03-08
---

# Phase 4 Plan 02: Audit Packet Exports Summary

**WeasyPrint PDF audit packet + CSV export with sign-off integration storing v1.pdf at output_dir/packets/ and re-download via /exports/{id}/audit-packet.{pdf,csv}**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-08T12:04:34Z
- **Completed:** 2026-03-08T12:10:38Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Audit packet export service generates CSV (per-ReviewItem rows) and PDF (WeasyPrint cover + summary + per-item cards with snippet images)
- Sign-off atomicity extended: `attempt_sign_off` generates and stores v1.pdf before committing `signed_off` status; failure rolls back to `reviewing`
- Download routes at `/exports/{id}/audit-packet.pdf` and `.csv` with Content-Disposition attachment headers; PDF re-download serves stored file via FileResponse
- Status page shows Download Audit Packet buttons for `signed_off` runs

## Task Commits

1. **Task 1: Build audit packet export service** - `5bb4ad8` (feat)
2. **Task 2: Wire sign-off integration, routes, status page** - `e768ff1` (feat)

**Plan metadata:** TBD (docs: complete plan)

## Files Created/Modified

- `shop/services/exports.py` — generate_audit_packet_csv, render_audit_packet_pdf, generate_and_store_audit_packet
- `shop/routers/exports.py` — GET /exports/{id}/audit-packet.pdf and .csv with auth guard
- `shop/templates/exports/audit_packet.html` — WeasyPrint template: cover page, summary table, per-item detail cards
- `shop/services/review.py` — attempt_sign_off now calls generate_and_store_audit_packet before signed_off
- `shop/app.py` — exports router registered at /exports prefix
- `shop/templates/runs/status.html` — Download Audit Packet buttons for signed_off runs
- `tests/test_exports.py` — 3 passing tests (pdf import, csv rows, redownload) + 4 xfail stubs
- `tests/test_review.py` — sign-off tests updated to mock PDF generation

## Decisions Made

- Inline import of `generate_and_store_audit_packet` inside `attempt_sign_off` try block avoids circular import chain (exports.py -> shop.app.templates -> routers -> review.py)
- `unittest.mock.patch("shop.services.exports.generate_and_store_audit_packet")` works with inline from-imports because the `from ... import` resolves the attribute at execution time inside the patch context
- WeasyPrint `base_url` set to `output_dir/snippets/` directory so `| basename` filtered image paths resolve correctly without absolute paths in HTML

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_sign_off_rollback and test_signed_off_immutable broken by PDF integration**
- **Found during:** Task 2 (wire sign-off integration)
- **Issue:** Both tests used `db.commit` patching or direct `attempt_sign_off` calls that assumed no PDF generation. With `generate_and_store_audit_packet` now called inside the try block, the TypeError from `Path(None)` and commit-count mismatch broke both tests.
- **Fix:** Updated `test_sign_off_rollback` to use `unittest.mock.patch` on `generate_and_store_audit_packet` to simulate failure; updated `test_signed_off_immutable` to mock PDF generation for the successful first sign-off call.
- **Files modified:** tests/test_review.py
- **Verification:** Both tests pass; full suite 73 passed, 13 xfailed, 3 xpassed
- **Committed in:** e768ff1 (Task 2 commit)

**2. [Rule 2 - Missing Critical] Added session authentication to test_audit_packet_redownload**
- **Found during:** Task 2 (redownload test)
- **Issue:** CSV download route requires authentication; test was hitting the route without a session cookie, getting redirected to /login
- **Fix:** Added `_login_engineer` helper and updated `test_audit_packet_redownload` to authenticate before requesting
- **Files modified:** tests/test_exports.py
- **Verification:** Test returns 200 with attachment header
- **Committed in:** e768ff1 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical)
**Impact on plan:** Both fixes essential for test correctness after integrating PDF generation into sign-off flow. No scope creep.

## Issues Encountered

None beyond the auto-fixed test regressions above.

## Next Phase Readiness

- Audit packet PDF/CSV export complete; PACKET-01..03 satisfied
- `/exports/` router pattern established for Plan 03 work order routes
- `generate_and_store_audit_packet` version increment logic ready for Plan 04 amendment flow (v2, v3 packets)

---
*Phase: 04-exports-history-and-amendments*
*Completed: 2026-03-08*
