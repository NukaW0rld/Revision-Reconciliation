---
phase: 10
slug: debug-exception-gating-and-advisory-surfacing
status: draft
nyquist_compliant: false
wave_0_complete: true
created: 2026-04-18
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Plan / Wave Graph

| Plan | Wave | Depends On | Validation Focus |
|------|------|------------|------------------|
| `10-01` | 1 | — | Surface packet-native `confidence_flags` on normal review, debug review, and status surfaces without losing row identity. |
| `10-02` | 2 | `10-01` | Enforce one unresolved-debug sign-off gate and capture a versioned signed debug snapshot at sign-off time. |
| `10-03` | 3 | `10-01`, `10-02` | Make signed audit/work-order exports read from the captured debug snapshot and surface advisory/debug state consistently. |

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest >=8` via `uv run pytest` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest -q tests/test_review.py tests/test_run_status_debug_summary.py tests/test_debug_verdicts.py tests/test_exports.py -x` |
| **Full suite command** | `uv run pytest -q tests/test_review.py tests/test_run_status_debug_summary.py tests/test_debug_verdicts.py tests/test_exports.py tests/test_history.py tests/test_amendments.py -x` |
| **Estimated quick runtime** | ~20-45 seconds |

---

## Sampling Rate

- After every task commit: run the narrowest task-local command from the map below.
- After every wave:
  - Wave 1: `uv run pytest -q tests/test_review.py tests/test_run_status_debug_summary.py tests/test_debug_verdicts.py -k "confidence_flags or advisory" -x`
  - Wave 2: `uv run pytest -q tests/test_review.py tests/test_debug_verdicts.py -k "sign_off or debug_exceptions or signed_debug_snapshot" -x`
  - Wave 3: `uv run pytest -q tests/test_exports.py tests/test_review.py -k "audit_packet or work_order or signed_snapshot" -x`
- Before `$gsd-verify-work`: the full suite command above must be green and a manual PDF spot-check must confirm the signed advisory/debug summary layout is readable.
- Max feedback latency: keep task-local checks under 30 seconds; allow the signed-export wave to run slightly longer because it spans multiple route/export surfaces.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 10-01 | 1 | CLS-01, VER-01 | T-10-01, T-10-02 | Packet-native advisory flags are joined by `ReviewItem.id` and preserved for duplicate/`None` `char_no` rows. | unit + route | `uv run pytest -q tests/test_review.py tests/test_debug_verdicts.py -k "confidence_flags_on_standard_item_cards or debug_queue_surfaces_packet_confidence_flags_without_rederiving_them" -x` | ✅ existing files extended | ⬜ pending |
| 10-01-02 | 10-01 | 1 | CLS-01, VER-01 | T-10-02 | Normal review/debug/status templates render the exact packet advisory text, not a template-only paraphrase. | route + template | `uv run pytest -q tests/test_review.py tests/test_run_status_debug_summary.py tests/test_debug_verdicts.py -k "confidence_flags or advisory" -x` | ✅ existing files extended | ⬜ pending |
| 10-02-01 | 10-02 | 2 | ADD-01, ADD-02, SNP-01, VER-01 | T-10-03 | Sign-off remains blocked while unresolved debug exceptions exist, even when normal review decisions are complete. | route + service | `uv run pytest -q tests/test_review.py -k "debug_exceptions_remain_even_with_zero_pending_items or signoff_footer_shows_debug_exception_gate" -x` | ✅ existing files extended | ⬜ pending |
| 10-02-02 | 10-02 | 2 | VER-01 | T-10-04 | Successful sign-off persists a versioned signed debug snapshot aligned with the packet version metadata. | service + route | `uv run pytest -q tests/test_review.py tests/test_debug_verdicts.py -k "sign_off or signed_debug_snapshot_metadata" -x` | ✅ existing files extended | ⬜ pending |
| 10-03-01 | 10-03 | 3 | VER-01 | T-10-05 | Export routes require signed debug snapshot metadata and do not silently fall back to current mutable debug state. | route + service | `uv run pytest -q tests/test_exports.py -k "signed_debug_snapshot_metadata" -x` | ✅ existing files extended | ⬜ pending |
| 10-03-02 | 10-03 | 3 | CLS-01, VER-01 | T-10-05, T-10-06 | Audit/work-order exports include signed advisory/debug state while keeping synthetic missing-truth rows out of the work-order action list. | export CSV + template | `uv run pytest -q tests/test_exports.py -k "confidence_flags_and_debug_state_from_signed_snapshot or work_order_csv_uses_signed_snapshot_advisories_without_materializing_missing_truth_rows" -x` | ✅ existing files extended | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] Existing pytest infrastructure already covers review queue, status page, sign-off, debug verdicts, and export routes.
- [x] No new test framework install is required.
- [x] Existing signed-run and export fixture helpers in `tests/test_review.py`, `tests/test_debug_verdicts.py`, and `tests/test_exports.py` are sufficient to extend for Phase 10.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Signed audit packet PDF and work-order PDF advisory/debug layout | CLS-01, VER-01 | The automated suite primarily asserts CSV/route behavior; final PDF readability still requires a human check. | Sign off a run with at least one advisory-flagged row, download the audit packet PDF and work-order PDF, and confirm the advisory/debug summary text appears on the intended row/section without overlap or truncation. |

---

## Threat References

| Threat ID | Category | Concern |
|-----------|----------|---------|
| T-10-01 | Integrity | Advisory joins keyed by `char_no` can attach flags to the wrong row when duplicate or `None` characteristic numbers exist. |
| T-10-02 | Integrity | Templates can drift from packet truth if they reconstruct warnings from `reasons` or mismatches instead of `confidence_flags`. |
| T-10-03 | Integrity | A UI-only sign-off gate can still be bypassed by route/service calls. |
| T-10-04 | Traceability | Without a signed debug snapshot, signed exports can drift away from the evidence cleared at sign-off. |
| T-10-05 | Traceability | Export routes that trust `run.status` alone can serve artifacts without the signed debug state contract. |
| T-10-06 | Integrity | Synthetic missing-added debug rows can leak into work-order actions if export merges do not distinguish audit/debug summary rows from actionable review items. |

---

## Validation Sign-Off

- [x] Existing infrastructure covers all planned work areas
- [x] Sampling continuity is preserved across the three planned waves
- [x] Wave 0 gaps are already closed by current fixtures and test helpers
- [x] No watch-mode flags are used
- [ ] `nyquist_compliant: true` will be set after execution evidence and manual PDF spot-checks are captured

**Approval:** pending
