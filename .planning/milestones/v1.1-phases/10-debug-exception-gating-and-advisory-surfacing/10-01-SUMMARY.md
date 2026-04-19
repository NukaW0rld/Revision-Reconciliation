---
phase: 10-debug-exception-gating-and-advisory-surfacing
plan: "01"
subsystem: review-advisory-surfacing
tags: [advisory, confidence_flags, review-queue, debug-queue, run-status, packet-native]
dependency_graph:
  requires: [delta_preservation/types.py::DeltaItem.confidence_flags, shop/services/review.py::build_debug_queue_state]
  provides: [advisory_flags_by_item_id, build_run_debug_summary.confidence_flags, _item_card.html::Packet Advisories, _item_card_debug.html::Packet Advisories, status.html::confidence_flags]
  affects: [shop/services/review.py, shop/routers/review.py, shop/templates/review/_item_card.html, shop/templates/review/_item_card_debug.html, shop/templates/runs/status.html]
tech_stack:
  added: []
  patterns: [service-first advisory extraction keyed by ReviewItem.id, packet-native flag rendering without re-derivation]
key_files:
  created: []
  modified:
    - shop/services/review.py
    - shop/routers/review.py
    - shop/templates/review/_item_card.html
    - shop/templates/review/_item_card_debug.html
    - shop/templates/runs/status.html
    - tests/test_review.py
    - tests/test_debug_verdicts.py
    - tests/test_run_status_debug_summary.py
decisions:
  - "advisory_flags_by_item_id keys by ReviewItem.id (not char_no) to tolerate duplicate and None characteristic numbers"
  - "confidence_flags rendered directly from packet data; no re-derivation from reasons or mismatch text"
  - "item_advisory_flags passed as named context to _render_debug_item_card for HTMX partial renders; advisory_flags dict used via include path for queue renders"
  - "Legacy packets omitting confidence_flags yield [] via getattr fallback"
metrics:
  duration: "5 min"
  completed: "2026-04-18T23:42:49Z"
  tasks_completed: 2
  files_modified: 8
---

# Phase 10 Plan 01: Advisory Flag Surfacing Summary

Surface packet-native `DeltaItem.confidence_flags` on normal review, debug review, and run-status maintainer surfaces using `ReviewItem.id`-keyed packet joins.

## What Was Built

**Task 1 — Service and router threading:**

- Added `advisory_flags_by_item_id(db, run) -> dict[int, list[str]]` to `shop/services/review.py`. Uses the existing `build_debug_queue_state` packet/review join, keys by `ReviewItem.id`, and defaults legacy packets to `[]`.
- Extended `build_run_debug_summary()` row payloads with `confidence_flags` (packet-native for packet-backed rows, `[]` for synthetic missing-added rows).
- Updated `shop/routers/review.py` to import and call `advisory_flags_by_item_id`, pass `advisory_flags` dict to the queue template context, and pass `item_advisory_flags` to `_render_debug_item_card` for HTMX partial renders.

**Task 2 — Template rendering:**

- `shop/templates/review/_item_card.html`: renders a "Packet Advisories" block when `advisory_flags.get(item.id)` is non-empty.
- `shop/templates/review/_item_card_debug.html`: renders a "Packet Advisories" block above the mismatch/debug verdict area using `item_advisory_flags` (HTMX path) or `advisory_flags.get(item.id)` (include path).
- `shop/templates/runs/status.html`: renders `confidence_flags` inline on conforming rows, and as a collapsible "Exception rows with advisories" section for exception rows that carry advisory flags.

## Tests Added

| Test | File | Asserts |
|------|------|---------|
| `test_review_queue_surfaces_packet_confidence_flags_on_standard_item_cards` | `test_review.py` | Normal review queue renders packet-native advisory text and "Packet Advisories" header |
| `test_debug_queue_surfaces_packet_confidence_flags_without_rederiving_them` | `test_debug_verdicts.py` | Debug queue renders packet-native advisory text and "Packet Advisories" header |
| `test_status_page_surfaces_confidence_flags_for_exception_rows` | `test_run_status_debug_summary.py` | Status page shows advisory text on exception rows; retains "Open Exception Queue" CTA and unresolved-count display |

## Commits

| Task | Hash | Message |
|------|------|---------|
| 1 | d48210c | feat(10-01): thread packet-native advisory flags through review service keyed by ReviewItem.id |
| 2 | 2a9c41f | feat(10-01): render packet-native advisory flags on normal review, debug review, and status surfaces |

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, or schema changes introduced. The advisory surfacing is read-only presentation of existing persisted packet data.

## Self-Check: PASSED

- `shop/services/review.py` contains `def advisory_flags_by_item_id(` — confirmed
- `shop/services/review.py` contains `confidence_flags` in `build_run_debug_summary` row payloads — confirmed
- `rg -n "confidence_flags" shop/services/review.py shop/routers/review.py` matches in both — confirmed
- `shop/templates/review/_item_card.html` contains `Packet Advisories` — confirmed
- `shop/templates/review/_item_card_debug.html` contains `Packet Advisories` — confirmed
- `shop/templates/runs/status.html` contains `confidence_flags` — confirmed
- All 3 test functions pass — confirmed (53 total tests in the 3 files pass)
