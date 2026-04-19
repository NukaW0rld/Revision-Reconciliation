---
status: complete
phase: 10-debug-exception-gating-and-advisory-surfacing
source: 10-01-SUMMARY.md, 10-02-SUMMARY.md, 10-03-SUMMARY.md
started: 2026-04-19T00:49:00Z
updated: 2026-04-19T00:52:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Advisory Flags on Normal Review Queue
expected: On a normal review queue for a run with packet-native confidence_flags, each item card shows a "Packet Advisories" block listing the advisory flag text from the packet.
result: pass
verified_by: test_review_queue_surfaces_packet_confidence_flags_on_standard_item_cards + template grep confirmed "Packet Advisories" in _item_card.html

### 2. Advisory Flags on Debug Review Queue
expected: On the debug review queue, item cards show a "Packet Advisories" block above the mismatch/debug verdict area, displaying packet-native confidence_flags without re-deriving them.
result: pass
verified_by: test_debug_queue_surfaces_packet_confidence_flags_without_rederiving_them + template grep confirmed "Packet Advisories" in _item_card_debug.html

### 3. Advisory Flags on Run Status Page
expected: On the run status page, exception rows with advisory flags show confidence_flags inline. A collapsible "Exception rows with advisories" section appears for exception rows carrying advisory flags.
result: pass
verified_by: test_status_page_surfaces_confidence_flags_for_exception_rows + template grep confirmed collapsible "Exception rows with advisories" in status.html

### 4. Sign-Off Blocked by Debug Exceptions
expected: When all normal review items are approved but unresolved debug exceptions remain (review_needed rows), attempting sign-off redirects to an error state (debug_exceptions_pending) rather than completing.
result: pass
verified_by: test_sign_off_blocks_when_debug_exceptions_remain_even_with_zero_pending_items

### 5. Sign-Off Footer Shows Exception Gate
expected: The sign-off footer renders "Resolve debug exceptions before sign-off (N unresolved — open debug queue)" and the sign-off CTA button is disabled when unresolved exceptions exist.
result: pass
verified_by: test_review_queue_signoff_footer_shows_debug_exception_gate + template grep confirmed exact text in _signoff_footer.html and _signoff_footer_oob.html

### 6. Signed Debug Snapshot on Sign-Off
expected: After a successful sign-off, a versioned debug snapshot file (v{N}-debug-report.json) is written to the output packets directory, and the packet_versions entry records debug_snapshot_path, debug_total, and unresolved_exception_count=0.
result: pass
verified_by: test_attempt_sign_off_persists_signed_debug_snapshot_metadata

### 7. Export Routes Require Signed Snapshot
expected: Requesting an export (audit packet or work order) for a signed-off run that lacks a debug_snapshot_path returns HTTP 409 Conflict instead of silently exporting with mutable state.
result: pass
verified_by: test_export_routes_require_signed_debug_snapshot_metadata

### 8. Audit Packet Includes Advisory State
expected: The audit packet CSV includes confidence_flags, debug_row_state, and debug_verdict columns merged from the signed snapshot. The audit packet PDF shows a "Signed Debug Summary" panel and per-item "Confidence Flags" / "Debug State" columns.
result: pass
verified_by: test_audit_packet_csv_includes_confidence_flags_and_debug_state_from_signed_snapshot + template grep confirmed "Signed Debug Summary" and "Confidence Flags" in audit_packet.html

### 9. Work Order Uses Signed Advisories
expected: The work order CSV/PDF includes confidence_flags from the signed snapshot on actionable rows. Synthetic missing-added truth rows (review_item_id=None) are excluded from the work-order action list. Work order HTML shows "Signed Debug Summary" header and "Confidence Flags" column.
result: pass
verified_by: test_work_order_csv_uses_signed_snapshot_advisories_without_materializing_missing_truth_rows + template grep confirmed "Signed Debug Summary" and "Confidence Flags" in work_order.html

## Summary

total: 9
passed: 9
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none]
