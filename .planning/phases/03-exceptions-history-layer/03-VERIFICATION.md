---
phase: "03"
name: "exceptions-history-layer"
created: 2026-04-11
status: passed
---

# Phase 03: exceptions-history-layer — Verification

## Goal-Backward Verification

**Phase Goal:** Persist accepted-alternate debug verdicts as durable same-part history and reuse them conservatively in later runs without changing canonical ground truth.

## Checks

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 1 | `HIST-01` accepted alternates persist in a separate history layer and are deactivated when the source verdict changes | ✅ | `uv run pytest -q tests/test_debug_history.py -x` |
| 2 | `HIST-02` history persistence keeps reviewed classification, requirement, mismatch fingerprint, and rationale without mutating `ground_truth.json` | ✅ | `uv run pytest -q tests/test_debug_history.py tests/test_alembic_baseline.py -x` |
| 3 | `HIST-03` later runs only auto-conform from history when the same part, truth identity, reviewed outcome, and mismatch set all match exactly | ✅ | `uv run pytest -q tests/test_history_conformance.py tests/test_debug_internals.py tests/test_pipeline_task.py -x` |
| 4 | Admin debug/report surfaces distinguish canonical conformance from history-backed acceptable alternates and keep `history_reference` serialized | ✅ | `uv run pytest -q tests/test_debug_internals.py tests/test_debug_verdicts.py tests/test_run_status_debug_summary.py -x` |
| 5 | Phase integration does not regress the broader pipeline, review queue, or run-status surfaces | ✅ | `uv run pytest -q` |

## Result

Phase 03 passed verification on 2026-04-11.

- Accepted-alternate verdicts now persist as active/superseded history records.
- Same-part reruns can auto-conform from approved alternates without weakening canonical truth checks.
- Debug and status surfaces correctly distinguish canonical matches from history-backed acceptable alternates.
- Full-suite regression cleanup restored green verification for strict ground-truth pipeline tests and best-effort admin debug/status rendering.

## Notes

- Manual walkthroughs listed in `03-VALIDATION.md` remain useful follow-up checks, but no blocking gaps remained after automated verification.
