---
phase: 4
slug: exports-history-and-amendments
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/test_exports.py tests/test_history.py tests/test_amendments.py -q` |
| **Full suite command** | `pytest -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_exports.py tests/test_history.py tests/test_amendments.py -q`
- **After every plan wave:** Run `pytest -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 4-??-01 | 01 | 0 | PACKET-01 | unit | `pytest tests/test_exports.py::test_audit_packet_pdf_bytes -x` | ❌ W0 | ⬜ pending |
| 4-??-02 | 01 | 0 | PACKET-02 | unit | `pytest tests/test_exports.py::test_audit_packet_csv_rows -x` | ❌ W0 | ⬜ pending |
| 4-??-03 | 01 | 0 | PACKET-03 | integration | `pytest tests/test_exports.py::test_audit_packet_redownload -x` | ❌ W0 | ⬜ pending |
| 4-??-04 | 01 | 0 | WORK-01 | integration | `pytest tests/test_exports.py::test_work_order_button_visible -x` | ❌ W0 | ⬜ pending |
| 4-??-05 | 01 | 0 | WORK-02 | unit | `pytest tests/test_exports.py::test_work_order_filters_status -x` | ❌ W0 | ⬜ pending |
| 4-??-06 | 01 | 0 | WORK-03 | unit | `pytest tests/test_exports.py::test_work_order_priority_labels -x` | ❌ W0 | ⬜ pending |
| 4-??-07 | 01 | 0 | WORK-04 | integration | `pytest tests/test_exports.py::test_work_order_pdf_csv -x` | ❌ W0 | ⬜ pending |
| 4-??-08 | 02 | 0 | HISTORY-01 | integration | `pytest tests/test_history.py::test_history_filters -x` | ❌ W0 | ⬜ pending |
| 4-??-09 | 02 | 0 | HISTORY-02 | integration | `pytest tests/test_history.py::test_signed_off_readonly_view -x` | ❌ W0 | ⬜ pending |
| 4-??-10 | 02 | 0 | HISTORY-03 | unit | `pytest tests/test_history.py::test_cleanup_exempt_signed_off -x` | ❌ W0 | ⬜ pending |
| 4-??-11 | 02 | 0 | HISTORY-04 | unit | `pytest tests/test_history.py::test_cleanup_deletes_old_runs -x` | ❌ W0 | ⬜ pending |
| 4-??-12 | 03 | 0 | AMEND-01 | unit | `pytest tests/test_amendments.py::test_create_amendment -x` | ❌ W0 | ⬜ pending |
| 4-??-13 | 03 | 0 | AMEND-02 | unit | `pytest tests/test_amendments.py::test_amendment_files_locked -x` | ❌ W0 | ⬜ pending |
| 4-??-14 | 03 | 0 | AMEND-03 | integration | `pytest tests/test_amendments.py::test_amendment_versioned_packet -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_exports.py` — stubs for PACKET-01..03, WORK-01..04
- [ ] `tests/test_history.py` — stubs for HISTORY-01..04
- [ ] `tests/test_amendments.py` — stubs for AMEND-01..03
- [ ] WeasyPrint POC: install + render test packet to verify 300 DPI PNG quality
- [ ] Add `"weasyprint"` to `pyproject.toml` dependencies
- [ ] Add `libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0` to Dockerfile runtime apt-get
- [ ] Add schema migration function for new columns: `parent_run_id`, `packet_versions`, `retention_days`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| PDF visual quality — 300 DPI PNG crops render without distortion | PACKET-01 | WeasyPrint rendering is visual; automated test only checks bytes | Open generated PDF, inspect characteristic detail cards for image quality and layout integrity |
| Work order readable on monochrome shop printer | WORK-03 | Print output requires physical device | Print work order PDF, verify RE-MEASURE/NEW labels are legible in black-and-white |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
