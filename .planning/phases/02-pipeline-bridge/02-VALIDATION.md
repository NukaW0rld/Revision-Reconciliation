---
phase: 2
slug: pipeline-bridge
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-03
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pytest.ini` / `pyproject.toml [tool.pytest]` |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 1 | UPLOAD-01 | integration | `pytest tests/test_upload.py -k "test_submission_form"` | ❌ W0 | ⬜ pending |
| 2-01-02 | 01 | 1 | UPLOAD-02 | integration | `pytest tests/test_upload.py -k "test_raster_rejection"` | ❌ W0 | ⬜ pending |
| 2-01-03 | 01 | 1 | UPLOAD-03 | integration | `pytest tests/test_upload.py -k "test_multipage_prompt"` | ❌ W0 | ⬜ pending |
| 2-01-04 | 01 | 1 | UPLOAD-04 | integration | `pytest tests/test_upload.py -k "test_excel_rejection"` | ❌ W0 | ⬜ pending |
| 2-01-05 | 01 | 1 | UPLOAD-05 | integration | `pytest tests/test_upload.py -k "test_metadata_stored"` | ❌ W0 | ⬜ pending |
| 2-02-01 | 02 | 1 | PIPE-01 | unit | `pytest tests/test_pipeline_task.py -k "test_huey_task"` | ❌ W0 | ⬜ pending |
| 2-02-02 | 02 | 1 | PIPE-02 | unit | `pytest tests/test_pipeline_task.py -k "test_stage_callback"` | ❌ W0 | ⬜ pending |
| 2-02-03 | 02 | 1 | PIPE-03 | integration | `pytest tests/test_status_page.py -k "test_sse_progress"` | ❌ W0 | ⬜ pending |
| 2-02-04 | 02 | 2 | PIPE-04 | integration | `pytest tests/test_pipeline_task.py -k "test_reva_balloon_fail"` | ❌ W0 | ⬜ pending |
| 2-02-05 | 02 | 2 | PIPE-05 | integration | `pytest tests/test_pipeline_task.py -k "test_revb_balloon_warning"` | ❌ W0 | ⬜ pending |
| 2-02-06 | 02 | 2 | PIPE-06 | integration | `pytest tests/test_pipeline_task.py -k "test_alignment_confidence"` | ❌ W0 | ⬜ pending |
| 2-02-07 | 02 | 2 | PIPE-07 | unit | `pytest tests/test_classify.py -k "test_hard_fail_wiring"` | ❌ W0 | ⬜ pending |
| 2-02-08 | 02 | 2 | PIPE-08 | unit | `pytest tests/test_classify.py -k "test_partial_warning_wiring"` | ❌ W0 | ⬜ pending |
| 2-03-01 | 03 | 2 | PIPE-09 | integration | `pytest tests/test_alerts.py -k "test_failure_alert"` | ❌ W0 | ⬜ pending |
| 2-03-02 | 03 | 2 | PIPE-10 | integration | `pytest tests/test_alerts.py -k "test_alert_display"` | ❌ W0 | ⬜ pending |
| 2-03-03 | 03 | 2 | PIPE-11 | integration | `pytest tests/test_alerts.py -k "test_alert_dismiss"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_upload.py` — stubs for UPLOAD-01 through UPLOAD-05
- [ ] `tests/test_pipeline_task.py` — stubs for PIPE-01 through PIPE-08
- [ ] `tests/test_status_page.py` — stubs for PIPE-03 (SSE progress)
- [ ] `tests/test_alerts.py` — stubs for PIPE-09 through PIPE-11
- [ ] `tests/conftest.py` — shared fixtures (test client, DB setup, mock Huey)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Stage checklist UI renders correctly with spinner/checkmark | PIPE-03 | Visual rendering; Playwright not configured | Load `/runs/{id}` status page, submit a run, observe spinner advances through stages |
| Low-confidence inline warning section displays correctly | PIPE-06 | Visual rendering of inline warning buttons | Trigger low-confidence alignment, verify "Proceed to Review" / "Abort Run" buttons appear |
| Alert bell badge shows correct count | PIPE-10 | Nav bar badge rendering | Trigger a run failure, check bell icon shows red badge |
| Multi-page PDF page selector appears inline | UPLOAD-03 | HTMX partial rendering | Upload multi-page PDF, verify page selector appears without page reload |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
