# Phase 11: web-run-to-review-e2e-automation - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `11-CONTEXT.md`; this log preserves the analysis.

**Date:** 2026-04-18
**Phase:** 11-web-run-to-review-e2e-automation
**Mode:** assumptions
**Areas analyzed:** Test Harness, Live Fixture Source, Coverage Shape, Advisory Coverage

## Assumptions Presented

### Test Harness

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 11 should use the existing `pytest` `TestClient` harness with `huey_immediate` so `/runs/new` executes the background task inline, rather than adding browser-level automation or a real Huey worker. | Confident | `tests/conftest.py`, `shop/routers/runs.py`, `tests/test_runs.py`, `tests/test_pipeline_task.py` |

### Live Fixture Source

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| The core Phase 11 proof should submit real corpus files from `assets/partN/` through `/runs/new`, and patch `shop.services.runs.UPLOADS_DIR` plus `shop.tasks.OUT_DIR` to per-test temp paths so the run writes a real `delta_packet.json` in isolation. | Confident | `.planning/ROADMAP.md`, `.planning/phases/07-regression-tests-and-verification/07-CONTEXT.md`, `tests/test_phase6_asset_regression.py`, `shop/services/runs.py`, `shop/tasks.py` |

### Coverage Shape

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Phase 11 likely needs a dedicated web-flow integration module that stitches existing slices together, rather than only extending `tests/test_runs.py`, `tests/test_review.py`, or `tests/test_exports.py` in place. | Likely | `tests/test_runs.py`, `tests/test_review.py`, `tests/test_exports.py`, `.planning/phases/10-debug-exception-gating-and-advisory-surfacing/10-CONTEXT.md`, `.planning/ROADMAP.md`, `.planning/v1.1-MILESTONE-AUDIT.md` |

### Advisory Coverage

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| The live fixture-backed run is likely sufficient for debug blockers and sign-off/export gating, but advisory-flag surfacing probably still needs either a verified flagged corpus part or a small companion seeded packet/snapshot case because checked-in corpus snapshots do not currently show non-empty `confidence_flags`. | Likely | `tests/fixtures/phase7_algorithm_only/part1-debug-report.json`, `tests/fixtures/phase7_algorithm_only/part6-debug-report.json`, `assets/debug_report_part1.json`, `tests/test_review.py`, `tests/test_exports.py` |

## Corrections Made

No corrections — all assumptions confirmed.

