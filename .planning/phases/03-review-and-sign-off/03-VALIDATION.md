---
phase: 3
slug: review-and-sign-off
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` — `[tool.pytest.ini_options]` testpaths=["tests"] addopts="-q" |
| **Quick run command** | `pytest tests/test_review.py -x -q` |
| **Full suite command** | `pytest -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_review.py -x -q`
- **After every plan wave:** Run `pytest -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-??-01 | TBD | 0 | REVIEW-01 | integration | `pytest tests/test_review.py::test_review_queue_loads -x` | ❌ W0 | ⬜ pending |
| 3-??-02 | TBD | 0 | REVIEW-02 | integration | `pytest tests/test_review.py::test_review_item_card_html -x` | ❌ W0 | ⬜ pending |
| 3-??-03 | TBD | 0 | REVIEW-03 | integration | `pytest tests/test_review.py::test_approve_item -x` | ❌ W0 | ⬜ pending |
| 3-??-04 | TBD | 0 | REVIEW-04 | integration | `pytest tests/test_review.py::test_override_requires_note -x` | ❌ W0 | ⬜ pending |
| 3-??-05 | TBD | 0 | REVIEW-05 | integration | `pytest tests/test_review.py::test_review_state_persisted -x` | ❌ W0 | ⬜ pending |
| 3-??-06 | TBD | 0 | REVIEW-06 | integration | `pytest tests/test_review.py::test_admin_can_reassign -x` | ❌ W0 | ⬜ pending |
| 3-??-07 | TBD | 0 | REVIEW-07 | integration | `pytest tests/test_review.py::test_review_counts -x` | ❌ W0 | ⬜ pending |
| 3-??-08 | TBD | 0 | SIGNOFF-01 | integration | `pytest tests/test_review.py::test_sign_off_gate -x` | ❌ W0 | ⬜ pending |
| 3-??-09 | TBD | 0 | SIGNOFF-02 | integration | `pytest tests/test_review.py::test_sign_off_rollback -x` | ❌ W0 | ⬜ pending |
| 3-??-10 | TBD | 0 | SIGNOFF-03 | integration | `pytest tests/test_review.py::test_signed_off_immutable -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_review.py` — stubs for REVIEW-01 through REVIEW-07, SIGNOFF-01 through SIGNOFF-03 (10 tests)
- [ ] `shop/models.py` — `ReviewItem` model + `Run.review_items` relationship + `Run.signed_at`/`Run.signed_by_id` columns
- [ ] Pipeline patch in `delta_preservation/cli.py` — predicted Rev B bbox for removed items

*Note: `tests/conftest.py` already has all needed fixtures: `client`, `db_engine`, `engineer_user`, `admin_user`, `huey_immediate`*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Snippet modal opens full-size on click | REVIEW-02 | Browser interaction with modal overlay | Open review queue, click any snippet image, verify modal appears with full-size image; press Escape to close |
| Sign-off confirmation modal displays correctly | SIGNOFF-01 | Browser modal rendering | With all items resolved, click Sign Off; verify confirmation modal appears before POST |
| SSE generating status page redirects correctly | SIGNOFF-02 | SSE + redirect timing in browser | After sign-off confirm, verify redirect to /review/{id}/generating, then to run summary on completion |
| Resolved card shows green border + badge after HTMX swap | REVIEW-03 | Visual rendering of DaisyUI classes | Approve an item; verify card border turns green and Approve button is replaced by status badge without page reload |
| Filter dropdowns update visible items without reload | REVIEW-01 | Browser form GET behavior | Set status filter to "pending"; verify only pending items shown; count matches header counter |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
