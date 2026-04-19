---
phase: 10-debug-exception-gating-and-advisory-surfacing
plan: "03"
subsystem: export-advisory-surfacing
tags: [signed-snapshot, exports, advisory, confidence_flags, audit-packet, work-order, debug-state]
dependency_graph:
  requires:
    - shop/services/review.py::write_signed_debug_snapshot
    - shop/services/review.py::assemble_debug_report_payload
    - shop/services/exports.py::generate_and_store_audit_packet
    - Run.packet_versions[].debug_snapshot_path
  provides:
    - shop/services/exports.py::_load_signed_debug_snapshot
    - shop/services/exports.py::_snapshot_advisory_by_item_id
    - shop/services/exports.py::_load_signed_debug_summary
    - shop/routers/exports.py::_get_signed_run snapshot gate (409)
    - audit_packet.html::Confidence Flags + Signed Debug Summary
    - work_order.html::Signed Debug Summary + Confidence Flags column
  affects:
    - shop/services/exports.py
    - shop/routers/exports.py
    - shop/templates/exports/audit_packet.html
    - shop/templates/exports/work_order.html
    - tests/test_exports.py
tech_stack:
  added: []
  patterns:
    - signed-snapshot-over-live-rederivation for export traceability
    - review_item_id keyed advisory merge (not char_no) for correctness with duplicate/None char numbers
    - additive export surfacing — advisory state annotates rows without replacing classification contract
    - synthetic missing-added truth rows excluded from work-order action list
key_files:
  created: []
  modified:
    - shop/services/exports.py
    - shop/routers/exports.py
    - shop/templates/exports/audit_packet.html
    - shop/templates/exports/work_order.html
    - tests/test_exports.py
decisions:
  - "_load_signed_debug_snapshot raises ValueError (not returns None) so callers can distinguish missing vs present snapshot"
  - "_get_signed_run requires_snapshot=True by default; 409 Conflict when snapshot unavailable, not silent fallback to mutable state"
  - "_snapshot_advisory_by_item_id keys by review_item_id; rows with review_item_id=None (synthetic missing-added) are excluded from export merge"
  - "confidence_flags rendered as semicolon-joined string in CSV; list in template context for badge rendering"
  - "Snapshot loading is best-effort in CSV/PDF service functions (ValueError caught silently) so legacy runs without snapshots still export"
  - "Work-order action list never includes synthetic missing-added truth rows — they appear only in audit/debug summary layer"
metrics:
  duration: "7 min"
  completed: "2026-04-18T00:13:00Z"
  tasks_completed: 2
  files_modified: 5
---

# Phase 10 Plan 03: Signed Export Advisory Surfacing Summary

Signed audit/work-order exports now read confidence_flags and debug state from the sign-off-time snapshot rather than mutable current files, and synthetic missing-added truth rows remain outside the work-order action list.

## What Was Built

**Task 1 — Signed snapshot loader and export route gate:**

- Added `_load_signed_debug_snapshot(run, version=None)` to `shop/services/exports.py`. Resolves the matching `packet_versions` entry, reads the stored `debug_snapshot_path` JSON, and returns the captured payload plus version metadata. Raises `ValueError` when missing or unreadable so callers can enforce the contract.
- Tightened `_get_signed_run()` in `shop/routers/exports.py` with `require_snapshot=True` default. Now raises 409 Conflict when a run is `signed_off` but the snapshot contract is missing or file is unreadable — rather than silently falling back to mutable current state.
- Updated `_make_signed_run`, `_make_run_with_mixed_items`, and `_make_run_with_override` test helpers to write real `debug_snapshot_path` files so existing route tests pass the new contract check.
- Added `test_export_routes_require_signed_debug_snapshot_metadata`: asserts 409 for runs without a snapshot, 200 for runs with a valid snapshot.

**Task 2 — Advisory/debug state surfaced in audit/work-order exports:**

- Added `_snapshot_advisory_by_item_id(snapshot_payload)`: extracts `confidence_flags`, `row_state`, `debug_verdict` from snapshot rows keyed by `review_item_id`. Synthetic missing-added rows (`review_item_id=None`) are excluded from the merge map.
- Added `_load_signed_debug_summary(run)`: convenience wrapper returning `{debug_total, unresolved_exception_count, version}` dict from the signed snapshot (empty dict on failure for legacy runs).
- `generate_audit_packet_csv`: now includes `confidence_flags`, `debug_row_state`, and `debug_verdict` columns merged from the signed snapshot by `ReviewItem.id`.
- `render_audit_packet_pdf`: passes `advisory_by_item_id` and `signed_debug_summary` to `audit_packet.html`.
- `_work_order_rows`: accepts optional `advisory_by_item_id`; merges `confidence_flags` onto actionable changed/added rows. Synthetic rows stay excluded (they come only from the debug snapshot's `review_item_id=None` entries, which are never in the `ReviewItem` DB query).
- `generate_work_order_csv/pdf`: load signed snapshot, merge advisory state, pass `signed_debug_summary` to template.
- `audit_packet.html`: added a "Signed Debug Summary" panel before the characteristic table, and "Confidence Flags" + "Debug State" columns in the summary table per-item.
- `work_order.html`: added "Signed Debug Summary" header note and "Confidence Flags" column in both Re-Measure and New Characteristics tables.
- Added `test_audit_packet_csv_includes_confidence_flags_and_debug_state_from_signed_snapshot`.
- Added `test_work_order_csv_uses_signed_snapshot_advisories_without_materializing_missing_truth_rows`.

## Tests Added

| Test | File | Asserts |
|------|------|---------|
| `test_export_routes_require_signed_debug_snapshot_metadata` | `test_exports.py` | 409 when snapshot absent; 200 when valid snapshot present |
| `test_audit_packet_csv_includes_confidence_flags_and_debug_state_from_signed_snapshot` | `test_exports.py` | Advisory flag in confidence_flags column; debug_row_state column; synthetic row excluded |
| `test_work_order_csv_uses_signed_snapshot_advisories_without_materializing_missing_truth_rows` | `test_exports.py` | Only chars 1+2 in work order; advisory flag on changed row; "Signed Debug Summary" and "Confidence Flags" in HTML |

## Commits

| Task | Hash | Message |
|------|------|---------|
| 1 | b7718a1 | feat(10-03): add _load_signed_debug_snapshot and gate export routes on snapshot contract |
| 2 | e13edd3 | feat(10-03): surface signed advisory/debug state in audit and work-order exports |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Existing route-testing helpers created signed runs without a debug_snapshot_path**

- **Found during:** Task 1 — running existing tests after tightening `_get_signed_run()` caused 3 route tests to fail with 409.
- **Issue:** `_make_signed_run`, `_make_run_with_mixed_items`, and `_make_run_with_override` created `packet_versions` entries without `debug_snapshot_path`, so `_load_signed_debug_snapshot` raised `ValueError` and the route returned 409.
- **Fix:** Updated all three helpers to write a minimal JSON snapshot file to a real temp directory and record `debug_snapshot_path` in `packet_versions`.
- **Files modified:** `tests/test_exports.py`
- **Commit:** b7718a1

## Known Stubs

None.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: T-10-05 mitigated | shop/services/exports.py | Exports now read from packet_versions[].debug_snapshot_path and fail clearly (409) if the snapshot contract is missing |
| threat_flag: T-10-06 mitigated | shop/services/exports.py | Synthetic missing-added truth rows excluded from work-order action list via review_item_id=None filter in _snapshot_advisory_by_item_id |

## Self-Check: PASSED

- `shop/services/exports.py` contains `def _load_signed_debug_snapshot(` — confirmed
- `shop/routers/exports.py` contains `debug_snapshot_path` gate check via `_load_signed_debug_snapshot` call — confirmed
- `shop/templates/exports/audit_packet.html` contains `Confidence Flags` — confirmed (line 69)
- `shop/templates/exports/work_order.html` contains `Signed Debug Summary` — confirmed (line 42)
- `rg -n "confidence_flags|debug_row_state|debug_verdict" shop/services/exports.py` returns matches — confirmed
- All 3 new test functions pass — confirmed (14 total tests in test_exports.py pass)
- Commits b7718a1 and e13edd3 exist — confirmed
