---
phase: 03
slug: exceptions-history-layer
status: ready
nyquist_compliant: true
wave_0_complete: false
created: 2026-04-11
---

# Phase 03 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Plan / Wave Graph

| Plan | Wave | Depends On | Validation Focus |
|------|------|------------|------------------|
| `03-01` | 1 | - | Durable history model, migration, sync from debug verdicts, structured reviewed outcome fields |
| `03-02` | 2 | `03-01` | Same-part history reuse, alternate-backed conforming rows, report/state history references |

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest` via `uv run pytest` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest -q tests/test_debug_history.py tests/test_history_conformance.py -x` |
| **Full suite command** | `uv run pytest -q` |
| **Estimated quick runtime** | ~20-35 seconds |

---

## Sampling Rate

- After every task commit: run the task-local smoke command from the verification map below.
- After every plan wave:
  - Wave 1: `uv run pytest -q tests/test_debug_history.py -x`
  - Wave 2: `uv run pytest -q tests/test_history_conformance.py tests/test_debug_internals.py -x`
- Before `/gsd-verify-work`: full suite must be green.
- Max feedback latency: target under 35 seconds for task-level smoke runs.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 03-01 | 1 | HIST-01, HIST-02 | T-03-01, T-03-02 | Coverage proves acceptable alternates persist in a separate DB layer with run/part/characteristic/outcome/rationale fields and do not touch truth fixtures. | integration | `uv run pytest -q tests/test_debug_history.py -x` | ❌ created by task | ⬜ pending |
| 03-01-02 | 03-01 | 1 | HIST-01, HIST-02 | T-03-02, T-03-04 | Saving or editing a debug verdict keeps history sync idempotent and deactivates/supersedes stale accepted-alternate records. | integration | `uv run pytest -q tests/test_debug_history.py -x` | ✅ after 03-01-01 | ⬜ pending |
| 03-02-01 | 03-02 | 2 | HIST-03 | T-03-03 | A later run for the same part and truth identity auto-conforms only when the stored reviewed outcome fingerprint matches. | integration | `uv run pytest -q tests/test_history_conformance.py -x` | ❌ created by task | ⬜ pending |
| 03-02-02 | 03-02 | 2 | HIST-03 | T-03-03 | `debug_report.json` and run summaries distinguish canonical truth conformance from history-backed acceptable alternates and populate `history_reference`. | integration | `uv run pytest -q tests/test_history_conformance.py tests/test_debug_internals.py -x` | ✅ existing `tests/test_debug_internals.py` | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_debug_history.py` - new regression coverage for the history model, migration-facing service contract, and verdict-sync behavior.
- [ ] `tests/test_history_conformance.py` - new regression coverage for same-part reuse and alternate-backed report state.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Saving an `acceptable_alternate` on one run makes the next run for that same part show history-backed conformance in the admin surfaces | HIST-03 | Requires a realistic run -> review -> rerun walkthrough with persisted DB state and rendered status/report output | Complete one run with an exception row, save `acceptable_alternate`, rerun the same part, then verify the status page/debug export reflects an accepted alternate rather than an unresolved exception. |
| Editing a previously accepted alternate away from `acceptable_alternate` removes or deactivates reuse on the next rerun | HIST-01, HIST-03 | End-to-end state transition across two runs plus DB-backed sync is hard to validate from unit assertions alone | Save an accepted alternate, confirm reuse on a second run, change the original verdict to `algorithm_error`, rerun again, and verify the row is no longer auto-conforming through history. |

---

## Validation Sign-Off

- [x] All planned tasks have automated verification commands
- [x] Sampling continuity is preserved across waves
- [x] Wave 0 gaps are explicit and bounded to Phase 3-specific test files
- [x] No watch-mode flags are used
- [x] Task-level feedback latency targets under 35 seconds
- [x] `nyquist_compliant: true` is set in frontmatter

**Approval:** pending
