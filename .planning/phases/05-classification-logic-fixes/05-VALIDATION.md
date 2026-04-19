---
phase: 05
slug: classification-logic-fixes
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-16
audited: 2026-04-16
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest >=8` via `uv run pytest` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/test_classify_bugfixes.py::TestConfidenceFlagsCompatibility tests/test_pipeline_semantic_packet.py tests/test_output_formatting.py tests/test_classify_bugfixes.py::TestAdjacencyBleed tests/test_classify_bugfixes.py::TestCountAdded tests/test_classify_bugfixes.py::TestAsymmetricTolerance tests/test_classify_bugfixes.py::TestToleranceOverlapThreshold tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation tests/test_classify_phase5_regression.py -x` |
| **Full suite command** | `uv run pytest -x` |
| **Observed runtime** | ~1 second for phase-local smokes / 30.27 seconds for full suite |

---

## Sampling Rate

- After every task commit: run the narrowest task-local smoke command from the verification map below.
- After every wave:
  - Wave 0: `uv run pytest tests/test_classify_bugfixes.py::TestConfidenceFlagsCompatibility tests/test_pipeline_semantic_packet.py tests/test_output_formatting.py -x`
  - Wave 1: `uv run pytest tests/test_classify_bugfixes.py::TestAdjacencyBleed tests/test_classify_bugfixes.py::TestCountAdded tests/test_classify_bugfixes.py::TestAsymmetricTolerance tests/test_classify_bugfixes.py::TestToleranceOverlapThreshold -x`
  - Wave 2: `uv run pytest tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation -x`
  - Wave 3: `uv run pytest tests/test_classify_phase5_regression.py -x`
- Before `/gsd-verify-work`: full suite must be green; the full 9-part pipeline rerun remains the Phase 7 / `VER-01` gate.
- Max feedback latency: keep task-local smokes under 5 seconds and the full suite under 60 seconds.

---

## Per-Task Verification Map

*One row per plan. Each plan bundles 1–3 tasks that share the same verification surface.*

| Plan | Wave | Requirement(s) | Threat Refs | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|------|------|----------------|-------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01 (scaffold) | 0 | CLS-01, CLS-02, CLS-03 | T-05-01, T-05-02 | Backward-compatible schema addition plus CLI fallback for legacy test doubles | unit | `uv run pytest tests/test_classify_bugfixes.py::TestConfidenceFlagsCompatibility tests/test_pipeline_semantic_packet.py tests/test_output_formatting.py -x` | ✅ existing files extended | ✅ green |
| 05-02 (CLS-01 bleed) | 1 | CLS-01 | T-05-04, T-05-05 | Suppresses bleed only when anchor content is proven in one chunk; preserves real slash-separated changes | unit | `uv run pytest tests/test_classify_bugfixes.py::TestAdjacencyBleed tests/test_classify_bugfixes.py::TestCountAdded -x` | ✅ existing file extended | ✅ green |
| 05-03 (CLS-03 asym) | 1 | CLS-03 | T-05-07, T-05-08 | Kind-transition check runs before `tolerances_match` and covers leading-decimal asymmetric forms | unit | `uv run pytest tests/test_classify_bugfixes.py::TestAsymmetricTolerance tests/test_classify_bugfixes.py::TestToleranceOverlapThreshold -x` | ✅ existing file extended | ✅ green |
| 05-04 (CLS-02 pair) | 2 | CLS-02 | T-05-10, T-05-11, T-05-12 | Same-page, nearest-compatible merge only; grouped added items use explicit text+bbox metadata | unit | `uv run pytest tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation -x` | ✅ existing files extended | ✅ green |
| 05-05 (regression) | 3 | CLS-01, CLS-02, CLS-03 | T-05-14 | Explicit snapshot exemplar guard plus synthetic packet-level CLS-02 regression | integration | `uv run pytest tests/test_classify_phase5_regression.py -x` | ✅ file created | ✅ green |

*Status: ✅ green · ❌ red · ⚠️ flaky*

**Full-suite gate (run before `/gsd-verify-work`):** `uv run pytest -x`

---

## Wave 0 Requirements

- [x] `tests/test_classify_bugfixes.py` includes `TestConfidenceFlagsCompatibility` with inline legacy-payload coverage and no empty placeholders.
- [x] `tests/test_pipeline_semantic_packet.py` and `tests/test_output_formatting.py` define `confidence_flags` on their fake internal DeltaItem helpers.
- [x] `delta_preservation.cli` forwards `confidence_flags` via `getattr(delta_internal, "confidence_flags", [])`.

---

## Snapshot Regression Gate (Phase-Specific)

| Check | Command | Pass Criterion |
|-------|---------|----------------|
| Snapshot exemplar harness | `uv run pytest tests/test_classify_phase5_regression.py -x` | The checked-in Phase 5 exemplar strings are present in the snapshots and the helper assertions pass |
| CLS-01 exemplars | `4 x Ø8 THRU ALL / ⌴ Ø13.5 ↧ 8.5`, `2X Ø.201 ↧ 0.50 / 1/4-20 UNC - 2B` | Bleed helper returns true only for the explicit adjacency-bleed exemplars |
| CLS-02 synthetic regression | grouped added item near removed anchor | Reconciled to a single `changed` row; cross-page candidate stays separate |
| CLS-03 exemplar | `2X 22.0° +0.3° / −0.1°` | Asymmetric shape detector matches; benign `70 / 30` does not trigger bleed |

*`ground_truth.json` files are never modified. Full 9-part pipeline reruns remain a Phase 7 / `VER-01` gate, not a Phase 5 validation requirement.*

---

## Manual-Only Verifications

None. The exact CLS-01 confidence-flag wording is already asserted in `tests/test_classify_bugfixes.py::TestAdjacencyBleed`, so Phase 5 has no remaining manual-only validation surface.

---

## Validation Sign-Off

- [x] All planned work areas have automated verification commands.
- [x] Sampling continuity is preserved across waves.
- [x] Wave 0 covers schema compatibility, CLI fallback, and fake internal test doubles.
- [x] No watch-mode flags are used.
- [x] Feedback latency stays well under 60 seconds for the full suite.
- [x] Snapshot regression harness is identified and wired to the phase gate.
- [x] `nyquist_compliant: true` is set in frontmatter.

**Approval:** 2026-04-16 — all Phase 5 plan surfaces are green, Wave 0 is complete, and the current full suite passes (`351 passed, 2 xfailed, 2 warnings`).

---

## Validation Audit 2026-04-16

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

**Execution evidence**

- `uv run pytest tests/test_classify_bugfixes.py::TestConfidenceFlagsCompatibility tests/test_pipeline_semantic_packet.py tests/test_output_formatting.py -x` → `20 passed`
- `uv run pytest tests/test_classify_bugfixes.py::TestAdjacencyBleed tests/test_classify_bugfixes.py::TestCountAdded tests/test_classify_bugfixes.py::TestAsymmetricTolerance tests/test_classify_bugfixes.py::TestToleranceOverlapThreshold tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation tests/test_classify_phase5_regression.py -x` → `29 passed`
- `uv run pytest -x` → `351 passed, 2 xfailed, 2 warnings in 30.27s`

**Warnings note:** the two warnings are pre-existing Alembic deprecation warnings from `tests/test_alembic_baseline.py`; they do not block Phase 5 Nyquist compliance.
