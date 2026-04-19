---
phase: 11-web-run-to-review-e2e-automation
plan: "02"
subsystem: test-harness
tags: [integration-test, live-run, e2e, sign-off, advisory, phase11]
dependency_graph:
  requires:
    - tests/test_web_run_review_e2e.py (Phase 11-01 artifact — extended in place)
    - shop/services/review.py (build_debug_queue_state, write_debug_verdicts, open_review_queue, attempt_sign_off)
    - shop/routers/exports.py (_get_signed_run, audit-packet.csv, work-order.csv routes)
    - shop/services/exports.py (_load_signed_debug_snapshot, _snapshot_advisory_by_item_id, generate_audit_packet_csv)
  provides:
    - tests/test_web_run_review_e2e.py (Phase 11 clearance-to-export and companion advisory proof)
  affects: []
tech_stack:
  added: []
  patterns:
    - write_debug_verdicts for service-level verdict injection in live-run E2E tests
    - Seeded signed debug snapshot with packet_item.confidence_flags for advisory companion case
    - _snapshot_advisory_by_item_id keyed by review_item_id for CSV advisory column sourcing
key_files:
  created: []
  modified:
    - tests/test_web_run_review_e2e.py
decisions:
  - "Use write_debug_verdicts (service-level) to clear debug exceptions in E2E test — route path already covered by test_live_run_blocks_signoff_until_debug_queue_is_cleared in Plan 01"
  - "Companion seeded snapshot patches review_item_id after flush so the snapshot item lookup by item.id resolves correctly in generate_audit_packet_csv"
  - "Companion test seeds packet_item.confidence_flags inside the snapshot payload (not the live delta_packet.json) to prove _snapshot_advisory_by_item_id is the source, not live packet read"
  - "Single commit for both tasks: both extend the same file and both tests must coexist to satisfy the Phase 11 milestone artifact contract"
metrics:
  duration_minutes: 6
  completed_date: "2026-04-19T04:14:45Z"
  tasks_completed: 2
  files_changed: 1
---

# Phase 11 Plan 02: Clearance-to-Export and Companion Advisory Proof Summary

Extends the dedicated Phase 11 integration module with proof that the live part6 run can traverse the full review-to-signed-export path after debug resolution, plus a companion seeded case that closes advisory-coverage inside the same artifact.

## What Was Built

**`tests/test_web_run_review_e2e.py`** — Two new tests added to the Phase 11 artifact:

### `test_live_run_can_be_cleared_signed_and_exported_after_debug_resolution`

Proves the live corpus run can reach the signed export endpoints:

1. Submits real part6 assets through `/runs/new` (reusing the Plan 01 `_submit_live_run` helper).
2. Opens the review queue by visiting `GET /review/{run_id}`.
3. Resolves all `review_needed` exception items and synthetic missing-added truth items with `acceptable_alternate` verdicts using `write_debug_verdicts` (service-level — route path already exercised in Plan 01).
4. Approves all normal review items via direct DB update.
5. Asserts `POST /review/{run_id}/sign-off/confirm` redirects to the generating path (gate is clear).
6. Asserts `run.status == "signed_off"`.
7. Asserts at least one `packet_versions` entry carries a readable `debug_snapshot_path` file (threat T-11-03).
8. Asserts `GET /exports/{run_id}/audit-packet.csv` returns 200.
9. Asserts `GET /exports/{run_id}/work-order.csv` returns 200.

### `test_phase11_companion_seeded_snapshot_surfaces_advisories`

Closes the advisory-coverage clause inside the Phase 11 artifact (threat T-11-04):

1. Seeds the smallest possible signed-off Run with one ReviewItem (char_no=1) and a debug snapshot whose `packet_item.confidence_flags` contains `"low_confidence_classification"`.
2. Patches `review_item_id` in the snapshot after `db.flush()` so the advisory lookup by `item.id` in `_snapshot_advisory_by_item_id` resolves correctly.
3. Asserts `GET /exports/{run_id}/audit-packet.csv` returns 200.
4. Parses the CSV and asserts the `confidence_flags` column for `char_no=1` contains the exact seeded flag string — proving the value flows from the signed snapshot contract, not from reconstructed `reasons` text.

The companion case is kept inside `tests/test_web_run_review_e2e.py` and is explicitly labeled as the approved fallback for non-live advisory coverage, per plan frontmatter `must_haves.truths`.

## Deviations from Plan

None — plan executed exactly as written. Both tests use the mechanisms specified in the plan action sections.

## Known Stubs

None — both tests exercise real service code paths with real or structurally complete seeded data.

## Threat Flags

None — test file only; no new network endpoints, auth paths, or schema changes introduced.

## Self-Check

- [x] `tests/test_web_run_review_e2e.py` contains `test_live_run_can_be_cleared_signed_and_exported_after_debug_resolution`
- [x] `tests/test_web_run_review_e2e.py` contains `test_phase11_companion_seeded_snapshot_surfaces_advisories`
- [x] Task 1 asserts `run.status == "signed_off"` and `debug_snapshot_path` in `packet_versions`
- [x] Task 2 advisory assertion is driven by seeded `confidence_flags` in `packet_item`, not `reasons`
- [x] Commit d1d666d exists
- [x] `uv run pytest -q tests/test_web_run_review_e2e.py -k "can_be_cleared_signed_and_exported" -x` — 1 passed
- [x] `uv run pytest -q tests/test_web_run_review_e2e.py -k "companion_seeded_snapshot_surfaces_advisories" -x` — 1 passed
- [x] `uv run pytest -q tests/test_web_run_review_e2e.py tests/test_runs.py tests/test_review.py tests/test_exports.py -x` — 55 passed

## Self-Check: PASSED
