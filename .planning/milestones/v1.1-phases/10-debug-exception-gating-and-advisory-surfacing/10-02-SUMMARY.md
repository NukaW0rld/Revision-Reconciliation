---
phase: 10-debug-exception-gating-and-advisory-surfacing
plan: "02"
subsystem: review-signoff-gate
tags: [sign-off, debug-exceptions, gating, signed-snapshot, packet-versions]
dependency_graph:
  requires:
    - shop/services/review.py::build_run_debug_summary
    - shop/services/review.py::build_debug_queue_state
    - shop/services/review.py::assemble_debug_report_payload
    - shop/services/exports.py::generate_and_store_audit_packet
  provides:
    - shop/services/review.py::build_signoff_gate_state
    - shop/services/review.py::write_signed_debug_snapshot
    - shop/templates/review/_signoff_footer.html::debug exception gate message
    - shop/routers/review.py::debug_exceptions_pending error code
  affects:
    - shop/services/review.py
    - shop/routers/review.py
    - shop/templates/review/_signoff_footer.html
    - shop/templates/review/_signoff_footer_oob.html
    - tests/test_review.py
    - tests/test_debug_verdicts.py
tech_stack:
  added: []
  patterns:
    - service-level preflight gate combining pending normal items and review_needed debug exceptions
    - versioned signed debug snapshot captured atomically before run.status flips to signed_off
    - strict D-01 exception contract (review_needed + missing-added-truth only, not all non-conforming rows)
key_files:
  created: []
  modified:
    - shop/services/review.py
    - shop/routers/review.py
    - shop/templates/review/_signoff_footer.html
    - shop/templates/review/_signoff_footer_oob.html
    - tests/test_review.py
    - tests/test_debug_verdicts.py
decisions:
  - "build_signoff_gate_state uses strict review_needed exception contract (build_debug_queue_state exception_items) rather than build_run_debug_summary to avoid false positives from items without evaluation"
  - "write_signed_debug_snapshot is non-fatal: snapshot failure is silently tolerated so sign-off itself is not blocked by a JSON write error"
  - "version number for snapshot derived from len(packet_versions) after generate_and_store_audit_packet appends its entry"
  - "signoff_gate_state passed as None to template context in debug mode (debug queue has its own footer logic)"
metrics:
  duration: "9 min"
  completed: "2026-04-18T23:54:34Z"
  tasks_completed: 2
  files_modified: 6
---

# Phase 10 Plan 02: Sign-Off Gate and Signed Debug Snapshot Summary

Centralize sign-off preflight in `build_signoff_gate_state()`, enforce it on both the router and service boundaries, and capture a versioned signed debug snapshot before `run.status` flips to `signed_off`.

## What Was Built

**Task 1 — Shared sign-off gate and normal queue surfacing:**

- Added `build_signoff_gate_state(db, run) -> dict` to `shop/services/review.py`. Combines pending normal review items and unresolved debug exceptions (strict D-01 contract: `review_needed` rows + unresolved missing-added-truth rows) into a single `can_sign_off` predicate with `debug_queue_url` when exceptions remain.
- Updated `shop/routers/review.py` to import `build_signoff_gate_state`, pass `signoff_gate_state` to the queue template context (set to `None` in debug mode), and check the gate in `sign_off_confirm()` before calling `attempt_sign_off()` — redirecting to `?error=debug_exceptions_pending` when blocked.
- Updated `_signoff_footer.html` and `_signoff_footer_oob.html` to render "Resolve debug exceptions before sign-off (N unresolved — open debug queue)" and disable the sign-off CTA when `unresolved_exceptions > 0`.

**Task 2 — Service-level gate enforcement and signed debug snapshot:**

- Updated `attempt_sign_off()` to call `build_signoff_gate_state()` as a preflight before any status mutation. Returns `False` without touching `run.status`, `signed_at`, or `signed_by_id` when the gate is not clear.
- Added `write_signed_debug_snapshot(db, run, version) -> str | None` to capture a versioned debug snapshot at `output_dir/packets/v{version}-debug-report.json` using `assemble_debug_report_payload()`. Records `debug_snapshot_path`, `debug_total`, and `unresolved_exception_count=0` on the matching `packet_versions` entry. Non-fatal: silently returns `None` if snapshot cannot be written.

**Deviation — Rule 1 (Bug fix):**

`build_signoff_gate_state` initially used `build_run_debug_summary` to count unresolved exceptions. `build_run_debug_summary` treats any non-conforming row (including rows with `None` evaluation) as an exception row. This caused `test_sign_off_gate` to fail because the existing `_make_run_with_packet` helper creates items without evaluation. Fixed by switching to `build_debug_queue_state` + `verdicts_by_item_id` to enforce the D-01 strict contract (only `review_needed` rows + unresolved missing-added-truth count as blockers).

## Tests Added

| Test | File | Asserts |
|------|------|---------|
| `test_sign_off_blocks_when_debug_exceptions_remain_even_with_zero_pending_items` | `test_review.py` | Router redirects to `?error=debug_exceptions_pending` when all items approved but one review_needed exception remains |
| `test_review_queue_signoff_footer_shows_debug_exception_gate` | `test_review.py` | Footer renders "Resolve debug exceptions before sign-off" and disables CTA when exceptions remain |
| `test_attempt_sign_off_persists_signed_debug_snapshot_metadata` | `test_debug_verdicts.py` | Successful sign-off writes v1-debug-report.json to disk and records debug_snapshot_path/debug_total/unresolved_exception_count=0 on packet_versions entry |

## Commits

| Task | Hash | Message |
|------|------|---------|
| 1 | d96bcd7 | feat(10-02): introduce build_signoff_gate_state and expose debug exception gate on normal review queue |
| 2 | c9042c0 | feat(10-02): enforce debug exception gate in attempt_sign_off and capture signed debug snapshot |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `build_signoff_gate_state` used overly broad exception definition**

- **Found during:** Task 2 test run — `test_sign_off_gate` failed because items without `evaluation` were counted as unresolved exceptions.
- **Issue:** `build_run_debug_summary` promotes all non-conforming rows to exceptions, including items where `evaluation is None` (no debug exception, just no evaluation data). This is wider than the D-01 blocker contract.
- **Fix:** Changed `build_signoff_gate_state` to use `build_debug_queue_state` directly, counting only `exception_items` (rows with `evaluation.status == "review_needed"`) and `missing_added_truth_items` without a saved verdict.
- **Files modified:** `shop/services/review.py`
- **Commit:** c9042c0

## Known Stubs

None.

## Threat Flags

None — no new network endpoints or auth paths introduced. The sign-off gate is an enforcement tightening of an existing route. The signed debug snapshot is a new file write in an already-trusted output directory.

## Self-Check: PASSED

- `shop/services/review.py` contains `def build_signoff_gate_state(` — confirmed (line 711)
- `shop/services/review.py` contains `def write_signed_debug_snapshot(` — confirmed (line 844)
- `shop/services/review.py` records `debug_snapshot_path` on packet_versions entry — confirmed (line 888)
- `rg -n "debug_snapshot_path|unresolved_exception_count" shop/services/review.py` returns matches — confirmed
- `shop/templates/review/_signoff_footer.html` contains `Resolve debug exceptions before sign-off` — confirmed
- `shop/routers/review.py` contains `debug_exceptions_pending` — confirmed (line 614)
- All 3 new test functions pass — confirmed (51 total tests in both files pass)
- Commits d96bcd7 and c9042c0 exist — confirmed
