---
phase: 10-debug-exception-gating-and-advisory-surfacing
verified: 2026-04-18T00:30:00Z
status: passed
score: 9/9 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 10: Debug Exception Gating and Advisory Surfacing Verification Report

**Phase Goal:** Make unresolved debug exceptions and classifier advisory flags visible and enforceable at sign-off/export time so maintainer decisions happen on the same evidence the packet records.
**Verified:** 2026-04-18
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

All truths are drawn from ROADMAP.md success criteria (3) and merged with per-plan must-haves (9 items across 3 plans). No must-haves reduce roadmap scope.

#### ROADMAP Success Criteria

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC-1 | Normal maintainer debug/review surfaces render classifier `confidence_flags` alongside affected items. | VERIFIED | `advisory_flags_by_item_id` in `review.py:762`; "Packet Advisories" block in `_item_card.html:91` and `_item_card_debug.html:90`; `confidence_flags` rendered in `status.html:112,118,127,158,160`. |
| SC-2 | Sign-off and audit/work-order export routes block completion while unresolved debug exceptions remain, unless an explicit acknowledgement path is provided and recorded. | VERIFIED | `build_signoff_gate_state` (`review.py:711`) enforces the gate at service level; `attempt_sign_off` (`review.py:913`) returns False without status mutation when gate fails; router redirects to `?error=debug_exceptions_pending` (`routers/review.py:614`); `_get_signed_run` returns 409 when snapshot contract missing (`routers/exports.py:28-34`). All 3 blocking tests pass. |
| SC-3 | Exported artifacts preserve the same advisory and gating state the maintainer saw during review. | VERIFIED | `_load_signed_debug_snapshot` (`exports.py:12`) reads the captured snapshot, not mutable current state; `_snapshot_advisory_by_item_id` merges advisory state by `review_item_id`; "Confidence Flags" column in `audit_packet.html:69`; "Signed Debug Summary" in `work_order.html:42`. Tests for CSV advisory column and no synthetic row materialisation pass. |

#### Plan 01 Must-Haves

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| P01-T1 | Normal review, debug review, and run-status surfaces render packet-native `confidence_flags` next to affected rows without re-deriving from `reasons` or mismatch text. | VERIFIED | `advisory_flags_by_item_id` keys by `ReviewItem.id` using packet join; templates render `confidence_flags` directly from packet context. Test `test_review_queue_surfaces_packet_confidence_flags_on_standard_item_cards` passes. |
| P01-T2 | Packet advisory joins are keyed by `ReviewItem.id` so duplicate or `None` `char_no` rows keep the correct advisory state. | VERIFIED | `advisory_flags_by_item_id` explicitly keys by `item.id` (`review.py:775`), not `char_no`. |
| P01-T3 | `build_run_debug_summary()` exposes row-level advisory data for both exception and conforming rows so the status page can show the same flag text the debug packet carries. | VERIFIED | `review.py:804` adds `confidence_flags` to packet-backed rows; `review.py:829` adds `[]` for synthetic rows. |

#### Plan 02 Must-Haves

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| P02-T1 | No Phase 10 acknowledgement bypass is introduced; sign-off remains a hard block until unresolved debug exceptions are cleared. | VERIFIED | `attempt_sign_off` calls `build_signoff_gate_state` before any mutation and returns False when `can_sign_off=False`. `write_signed_debug_snapshot` is non-fatal but does not circumvent the gate — gate check happens before snapshot writing. |
| P02-T2 | The normal review queue footer/modal shows the same unresolved-debug gate that the sign-off service enforces. | VERIFIED | `_signoff_footer.html:15` contains "Resolve debug exceptions before sign-off"; router passes `signoff_gate_state` to template context. Test `test_review_queue_signoff_footer_shows_debug_exception_gate` passes. |
| P02-T3 | Successful sign-off captures a signed debug snapshot and stores its metadata alongside the versioned packet entry before `run.status` flips to `signed_off`. | VERIFIED | `write_signed_debug_snapshot` (`review.py:844`) writes `v{version}-debug-report.json`, records `debug_snapshot_path`, `debug_total`, `unresolved_exception_count=0` on `packet_versions`. Status flips at `review.py:932` after snapshot is written. Test `test_attempt_sign_off_persists_signed_debug_snapshot_metadata` passes. |

#### Plan 03 Must-Haves

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| P03-T1 | Signed audit/work-order exports render from the sign-off-time debug snapshot rather than current mutable debug verdict files. | VERIFIED | `_load_signed_debug_snapshot` reads `debug_snapshot_path` from `packet_versions`; `_get_signed_run` requires readable snapshot (409 otherwise). Test `test_export_routes_require_signed_debug_snapshot_metadata` passes. |
| P03-T2 | Audit/work-order exports surface packet-native advisory flags and signed debug state without replacing the core classification/reviewer-decision contract. | VERIFIED | `_snapshot_advisory_by_item_id` merges advisory state additively by `review_item_id`; classification columns are unchanged. Test `test_audit_packet_csv_includes_confidence_flags_and_debug_state_from_signed_snapshot` passes. |
| P03-T3 | Synthetic missing-added truth rows may appear in signed audit/debug summary context, but they do not become work-order action rows. | VERIFIED | `_snapshot_advisory_by_item_id` excludes `review_item_id=None` rows from merge map (`exports.py:71-72`). Test `test_work_order_csv_uses_signed_snapshot_advisories_without_materializing_missing_truth_rows` passes asserting only chars 1+2 appear as work-order action rows. |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `shop/services/review.py` | `advisory_flags_by_item_id` and `build_signoff_gate_state` and `write_signed_debug_snapshot` | VERIFIED | All three functions present at lines 762, 711, 844. `confidence_flags` in `build_run_debug_summary` row payloads at line 804. |
| `shop/templates/review/_item_card.html` | "Packet Advisories" advisory block | VERIFIED | Line 91: `Packet Advisories` |
| `shop/templates/review/_item_card_debug.html` | "Packet Advisories" advisory block | VERIFIED | Line 90: `Packet Advisories` |
| `shop/templates/runs/status.html` | `confidence_flags` rendering | VERIFIED | Lines 112, 118, 127, 158, 160 |
| `shop/templates/review/_signoff_footer.html` | "Resolve debug exceptions before sign-off" gate message | VERIFIED | Line 15 |
| `shop/services/exports.py` | `_load_signed_debug_snapshot`, `_snapshot_advisory_by_item_id`, `_load_signed_debug_summary` | VERIFIED | All three present at lines 12, 67, 333 |
| `shop/templates/exports/audit_packet.html` | "Confidence Flags" column | VERIFIED | Line 69 |
| `shop/templates/exports/work_order.html` | "Signed Debug Summary" | VERIFIED | Line 42 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `review.py::build_debug_queue_state` | `review/_item_card.html` | `ReviewItem.id -> DeltaItem.confidence_flags` | WIRED | `advisory_flags_by_item_id` performs the join; router passes `advisory_flags` dict to template context; template renders via `advisory_flags.get(item.id)`. |
| `review.py::build_run_debug_summary` | `runs/status.html` | summary row advisory state | WIRED | `confidence_flags` added to every row payload; status.html iterates `debug_summary.exception_rows` and renders `row.confidence_flags`. |
| `review.py::build_run_debug_summary` | `review.py::attempt_sign_off` | `debug_report_ready` / `unresolved_exception_count` | WIRED | `build_signoff_gate_state` at line 711 computes `can_sign_off`; `attempt_sign_off` checks at line 913. |
| `review.py::write_signed_debug_snapshot` | `Run.packet_versions` | `debug_snapshot_path` metadata | WIRED | Lines 882-892: `debug_snapshot_path`, `debug_total`, `unresolved_exception_count=0` written to matching `packet_versions` entry. |
| `Run.packet_versions[].debug_snapshot_path` | `shop/services/exports.py` | signed debug snapshot lookup | WIRED | `_load_signed_debug_snapshot` resolves `packet_versions` entry and reads `debug_snapshot_path` file. `_get_signed_run` raises 409 if missing. |
| signed debug snapshot rows | audit/work-order export rows | `review_item_id` merge | WIRED | `_snapshot_advisory_by_item_id` keys by `review_item_id`; advisory state merged onto CSV rows and PDF template context. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `_item_card.html` advisory block | `advisory_flags.get(item.id)` | `advisory_flags_by_item_id` reads `DeltaItem.confidence_flags` from DB via `build_debug_queue_state` | Yes — persisted packet data, not re-derived | FLOWING |
| `status.html` confidence_flags | `row.confidence_flags` | `build_run_debug_summary` reads `DeltaItem.confidence_flags` via packet join | Yes — packet-native, not reconstructed | FLOWING |
| export CSV `confidence_flags` column | advisory merge | `_snapshot_advisory_by_item_id` parses the stored JSON snapshot written at sign-off time | Yes — snapshot captured at sign-off, not mutable | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 9 named Phase 10 tests pass | `uv run pytest -v <9 test IDs>` | 9 passed in 2.08s | PASS |
| `advisory_flags_by_item_id` exists in review.py | grep | Line 762 | PASS |
| `build_signoff_gate_state` exists in review.py | grep | Line 711 | PASS |
| `write_signed_debug_snapshot` exists in review.py | grep | Line 844 | PASS |
| `_load_signed_debug_snapshot` exists in exports.py | grep | Line 12 | PASS |
| `_get_signed_run` gates on snapshot (409) | grep | Lines 28-34 | PASS |
| All 6 implementation commits present | git log | d48210c, 2a9c41f, d96bcd7, c9042c0, b7718a1, e13edd3 | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CLS-01 | 10-01, 10-03 | Adjacent balloon bleed suppression with confidence flag | SATISFIED (pre-existing) | Already [x] in REQUIREMENTS.md. Phase 10 surfaces the persisted `confidence_flags` it produces — confirmed flowing from packet through review to export. |
| ADD-01 | 10-01, 10-02, 10-03 | All ground-truth-added characteristics present | SATISFIED (pre-existing) | Already [x] in REQUIREMENTS.md. Phase 10 ensures synthetic missing-added rows visible in audit context (not work-order). |
| ADD-02 | 10-01, 10-02, 10-03 | Spurious added rows suppressed | SATISFIED (pre-existing) | Already [x] in REQUIREMENTS.md. |
| SNP-01 | 10-01, 10-02, 10-03 | Title block exclusion | SATISFIED (pre-existing) | Already [x] in REQUIREMENTS.md. |
| VER-01 | 10-01, 10-02, 10-03 | Post-fix ground-truth re-run | SATISFIED (pre-existing) | Already [x] in REQUIREMENTS.md. |

**Note on requirement IDs in Phase 10 plans:** The ROADMAP explicitly states Phase 10 has "— (audit integration closure only)" for requirements. The IDs listed in plan frontmatter are "affected requirements" — Phase 10 wires their outputs into review/export surfaces. All five were already satisfied by Phases 5-9. No new algorithm requirements were introduced and none were orphaned.

**Orphaned requirements check:** TST-01 and TST-02 remain open ([ ]) in REQUIREMENTS.md but are assigned to Phase 7 per the Traceability table — not Phase 10. They are not orphaned with respect to Phase 10.

### Anti-Patterns Found

No anti-patterns detected. Grep scan of `shop/services/review.py`, `shop/services/exports.py`, `shop/routers/review.py`, `shop/routers/exports.py`, and the four template files found zero TODO/FIXME/PLACEHOLDER/stub return patterns.

### Human Verification Required

None. All observable truths are verifiable via code inspection and automated tests. The phase does not introduce new visual layouts requiring user-acceptance testing and does not involve external service integrations.

### Gaps Summary

No gaps. All 9 must-haves verified, all 6 key links wired, data flows from packet through service through template to export, all 9 named tests pass.

---

_Verified: 2026-04-18_
_Verifier: Claude (gsd-verifier)_
