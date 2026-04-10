---
phase: 02
slug: focused-debug-workflow
status: revised
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-10
revised: 2026-04-10
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Plan / Wave Graph

| Plan | Wave | Depends On | Validation Focus |
|------|------|------------|------------------|
| `02-01` | 1 | — | Exception-only queue membership, zero-exception redirect, stable row identity |
| `02-02` | 2 | `02-01` | Exception outcome vocabulary, stale legacy verdict handling, report row states |
| `02-03` | 3 | `02-01`, `02-02` | Run-details debug summary, mismatch-first debug UI, exception-only export UX |

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest 9.0.2` via `uv run pytest` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest -q tests/test_debug_verdicts.py::test_algorithm_error_accepts_reviewer_accepted_classification_when_label_matches_pipeline tests/test_debug_verdicts.py::test_load_debug_verdicts_for_render_ignores_legacy_verdict_entries tests/test_debug_verdicts.py::test_legacy_debug_verdicts_require_phase2_reentry_on_strict_paths tests/test_debug_internals.py::test_debug_report_rows_keep_ordered_mismatches_and_history_placeholder -x` |
| **Full suite command** | `uv run pytest -q` |
| **Estimated quick runtime** | ~15-25 seconds |

---

## Sampling Rate

- **After every task commit:** run the narrowest task-local smoke command from the verification map below.
- **After every plan wave:**
  - Wave 1: `uv run pytest -q tests/test_focused_debug_queue.py::test_admin_debug_queue_only_shows_review_needed_rows tests/test_focused_debug_queue.py::test_admin_debug_queue_redirects_all_conforming_run_to_status_page tests/test_debug_row_identity.py::test_duplicate_and_none_char_rows_keep_distinct_review_item_ids -x`
  - Wave 2: `uv run pytest -q tests/test_debug_verdicts.py::test_algorithm_error_accepts_reviewer_accepted_classification_when_label_matches_pipeline tests/test_debug_verdicts.py::test_load_debug_verdicts_for_render_ignores_legacy_verdict_entries tests/test_debug_verdicts.py::test_legacy_debug_verdicts_require_phase2_reentry_on_strict_paths tests/test_debug_internals.py::test_debug_report_rows_keep_ordered_mismatches_and_history_placeholder -x`
  - Wave 3: `uv run pytest -q tests/test_run_status_debug_summary.py::test_all_conforming_admin_run_stays_on_status_page tests/test_focused_debug_queue.py::test_debug_queue_renders_mismatch_summary_before_collapsed_details tests/test_focused_debug_queue.py::test_debug_form_exposes_phase2_exception_outcomes_only -x`
- **Before `/gsd-verify-work`:** full suite must be green.
- **Max feedback latency:** target under 30 seconds for task-level smoke runs.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 02-01 | 1 | DREV-01 | T-02-01, T-02-03 | Queue-state regression coverage proves only `review_needed` rows enter the admin debug queue and duplicate/null characteristic rows keep stable identity. | integration | `uv run pytest -q tests/test_focused_debug_queue.py::test_admin_debug_queue_only_shows_review_needed_rows tests/test_debug_row_identity.py::test_duplicate_and_none_char_rows_keep_distinct_review_item_ids -x` | ❌ created by task | ⬜ pending |
| 02-01-02 | 02-01 | 1 | DREV-01, DREV-02 | T-02-01, T-02-02, T-02-03 | Implementation preserves exception-only membership, `ReviewItem.id` ordering, and redirects zero-exception debug opens back to run status. | integration | `uv run pytest -q tests/test_focused_debug_queue.py::test_admin_debug_queue_only_shows_review_needed_rows tests/test_focused_debug_queue.py::test_admin_debug_queue_redirects_all_conforming_run_to_status_page tests/test_debug_row_identity.py::test_duplicate_and_none_char_rows_keep_distinct_review_item_ids -x` | ✅ after 02-01-01 | ⬜ pending |
| 02-02-01 | 02-02 | 2 | DREV-03, DREV-04, RPT-01, RPT-02, RPT-03 | T-02-04, T-02-05, T-02-06 | Coverage proves only the Phase 2 vocabulary is valid, stale legacy verdict payloads are treated as stale, and report rows preserve ordered mismatches plus `history_reference: null`. | unit + integration | `uv run pytest -q tests/test_debug_verdicts.py::test_algorithm_error_accepts_reviewer_accepted_classification_when_label_matches_pipeline tests/test_debug_verdicts.py::test_load_debug_verdicts_for_render_ignores_legacy_verdict_entries tests/test_debug_verdicts.py::test_legacy_debug_verdicts_require_phase2_reentry_on_strict_paths tests/test_debug_internals.py::test_debug_report_rows_keep_ordered_mismatches_and_history_placeholder -x` | ✅ | ⬜ pending |
| 02-02-02 | 02-02 | 2 | DREV-03, DREV-04, RPT-01, RPT-02, RPT-03 | T-02-04, T-02-05, T-02-06 | Strict paths reject legacy verdict vocab with a clear Phase 2 re-entry message, `algorithm_error` accepts reviewer-accepted classifications that may match the pipeline label, and export readiness ignores conforming rows. | unit + integration | `uv run pytest -q tests/test_debug_verdicts.py::test_algorithm_error_accepts_reviewer_accepted_classification_when_label_matches_pipeline tests/test_debug_verdicts.py::test_load_debug_verdicts_for_render_ignores_legacy_verdict_entries tests/test_debug_verdicts.py::test_legacy_debug_verdicts_require_phase2_reentry_on_strict_paths tests/test_debug_internals.py::test_debug_report_rows_keep_ordered_mismatches_and_history_placeholder -x` | ✅ | ⬜ pending |
| 02-03-01 | 02-03 | 3 | DREV-01, DREV-02, DREV-03, DREV-04 | T-02-07, T-02-08 | Integration coverage proves the status-page summary, mismatch-first cards, and collapsed diagnostics render the focused exception workflow. | integration | `uv run pytest -q tests/test_run_status_debug_summary.py::test_all_conforming_admin_run_stays_on_status_page tests/test_focused_debug_queue.py::test_debug_queue_renders_mismatch_summary_before_collapsed_details tests/test_focused_debug_queue.py::test_debug_form_exposes_phase2_exception_outcomes_only -x` | ❌ `tests/test_run_status_debug_summary.py` created by task | ⬜ pending |
| 02-03-02 | 02-03 | 3 | DREV-02 | T-02-08 | Run details expose auto-pass counts, keep zero-exception runs on status, and surface debug-report access without entering the queue. | integration | `uv run pytest -q tests/test_run_status_debug_summary.py::test_all_conforming_admin_run_stays_on_status_page tests/test_run_status_debug_summary.py::test_mixed_run_shows_open_exception_queue_cta tests/test_run_status_debug_summary.py::test_resolved_exception_run_shows_debug_report_download -x` | ✅ after 02-03-01 | ⬜ pending |
| 02-03-03 | 02-03 | 3 | DREV-01, DREV-03, DREV-04 | T-02-07, T-02-08, T-02-09 | Debug UI renders mismatch summary before heavy details, shows only Phase 2 exception outcomes, and keys helper data by stable `ReviewItem.id`-based identity. | integration | `uv run pytest -q tests/test_focused_debug_queue.py::test_debug_queue_renders_mismatch_summary_before_collapsed_details tests/test_focused_debug_queue.py::test_debug_form_exposes_phase2_exception_outcomes_only tests/test_focused_debug_queue.py::test_debug_footer_uses_exception_only_readiness_counts -x` | ✅ after 02-01-01 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] Existing `uv run pytest` infrastructure already covers the Nyquist baseline; no separate Wave 0 scaffolding task is required.
- [x] Missing phase-specific test modules are intentionally created in Wave 1 (`02-01`) instead of being treated as pre-plan blockers.
- [x] Task-level smoke commands now use specific test cases or narrow slices rather than rerunning whole files.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Exception cards are mismatch-first with heavy diagnostics collapsed by default | DREV-01, DREV-03, DREV-04 | Requires visual confirmation of the rendered admin debug UX and disclosure defaults | Run the app, open a completed run in debug mode with at least one exception row, verify the mismatch summary and review form are visible without opening secondary panels, and confirm scores/semantic internals/bbox centers remain behind collapsed disclosures. |
| Zero-exception runs stay on run details and show a clean all-conforming summary instead of an empty queue | DREV-01, DREV-02 | Requires end-to-end confirmation of navigation and CTA behavior across templates/routes | Load a completed run whose packet evaluation has only `conforming` rows, verify `/runs/{id}` shows the debug summary and export CTA, and confirm the reviewer is not routed into an empty exception queue. |

---

## Validation Sign-Off

- [x] All planned tasks have `<automated>` verification commands
- [x] Sampling continuity is preserved across waves
- [x] Wave 0 is satisfied by existing infrastructure plus Wave 1 test creation
- [x] No watch-mode flags are used
- [x] Task-level feedback latency targets under 30 seconds
- [x] `nyquist_compliant: true` is set in frontmatter

**Approval:** pending
