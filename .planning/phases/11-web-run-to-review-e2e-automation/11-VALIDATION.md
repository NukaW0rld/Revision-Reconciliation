---
phase: 11
slug: web-run-to-review-e2e-automation
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-18
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest -q tests/test_web_run_review_e2e.py -x` |
| **Full suite command** | `uv run pytest -q tests/test_web_run_review_e2e.py tests/test_runs.py tests/test_review.py tests/test_exports.py -x` |
| **Estimated runtime** | ~240 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -q tests/test_web_run_review_e2e.py -x`
- **After every plan wave:** Run `uv run pytest -q tests/test_web_run_review_e2e.py tests/test_runs.py tests/test_review.py tests/test_exports.py -x`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 240 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | TST-02 | T-11-01 | Live `/runs/new` writes uploads and packet output only inside isolated temp roots while the inline task uses the test DB session. | integration | `uv run pytest -q tests/test_web_run_review_e2e.py -k "submission_persists_packet" -x` | ✅ file created | ✅ green |
| 11-01-02 | 01 | 1 | VER-01 | T-11-02 | A representative live run reaches the status page and review/debug surfaces, then blocks sign-off while unresolved debug exceptions remain. | integration | `uv run pytest -q tests/test_web_run_review_e2e.py -k "blocks_signoff_until_debug_queue_is_cleared" -x` | ✅ existing file extended | ✅ green |
| 11-02-01 | 02 | 2 | VER-01 | T-11-03 | After review/debug clearance, the same live run can sign off and reach signed export routes with snapshot metadata recorded. | integration | `uv run pytest -q tests/test_web_run_review_e2e.py -k "can_be_cleared_signed_and_exported" -x` | ✅ existing file extended | ✅ green |
| 11-02-02 | 02 | 2 | TST-02 | T-11-04 | A companion seeded case keeps advisory/export coverage in the Phase 11 module when the live corpus fixture has no stable packet flags. | integration | `uv run pytest -q tests/test_web_run_review_e2e.py -k "companion_seeded_snapshot_surfaces_advisories" -x` | ✅ existing file extended | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/test_web_run_review_e2e.py` — dedicated Phase 11 integration artifact
- [x] `tests/conftest.py` shared helper was not required; Phase 11 local setup remained isolated inside the dedicated module

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 240s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** 2026-04-19 — all Phase 11 plan surfaces are green and the dedicated live-run artifact plus cross-surface regression suite pass.

---

## Validation Audit 2026-04-19

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

**Execution evidence**

- `uv run pytest -q tests/test_web_run_review_e2e.py -x` -> pass
- `uv run pytest -q tests/test_web_run_review_e2e.py tests/test_runs.py tests/test_review.py tests/test_exports.py -x` -> pass

**Audit note:** No missing Nyquist coverage remained. This audit only reconciled the stale pending map in `11-VALIDATION.md` with the already-landed Phase 11 execution state and current green test evidence.
