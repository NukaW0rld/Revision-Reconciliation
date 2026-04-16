---
phase: 5
slug: classification-logic-fixes
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-16
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/test_classify_bugfixes.py -x` |
| **Full suite command** | `pytest` |
| **Estimated runtime** | ~15 seconds (quick) / ~60 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_classify_bugfixes.py -x`
- **After every plan wave:** Run `pytest`
- **Before `/gsd-verify-work`:** Full suite must be green + 9-part corpus re-run shows zero regressions
- **Max feedback latency:** 15 seconds (quick) / 60 seconds (full)

---

## Per-Task Verification Map

*One row per plan (each plan bundles 1–3 tasks that share the same pytest binding). Populated from emitted PLAN.md frontmatter after planner completion.*

| Plan | Wave | Requirement(s) | Threat Refs | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|------|------|----------------|-------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01 (scaffold) | 0 | CLS-01, CLS-02, CLS-03 | T-05-01, T-05-02 | Backward-compat field addition | unit | `pytest tests/test_classify_bugfixes.py::TestConfidenceFlagsBackwardCompat -x` | ❌ W0 | ⬜ pending |
| 05-02 (CLS-01 bleed) | 1 | CLS-01 | T-05-04, T-05-05 | Preserves verdict on bleed; protects CLS-03 signal | unit | `pytest tests/test_classify_bugfixes.py::TestAdjacencyBleed tests/test_classify_bugfixes.py::TestCountAdded -x` | ❌ W0 | ⬜ pending |
| 05-03 (CLS-03 asym) | 1 | CLS-03 | T-05-07, T-05-08 | Promotes unchanged→changed on kind flip | unit | `pytest tests/test_classify_bugfixes.py::TestAsymmetricTolerance -x` | ❌ W0 | ⬜ pending |
| 05-04 (CLS-02 pair) | 2 | CLS-02 | T-05-10, T-05-11, T-05-12 | Type-compatible merge only; far-apart not merged | unit | `pytest tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation -x` | ❌ W0 | ⬜ pending |
| 05-05 (regression) | 3 | CLS-01, CLS-02, CLS-03 | T-05-14 | Zero helper over-firing against 9-part snapshots | integration | `pytest tests/test_classify_phase5_regression.py -x` | ❌ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky. "File Exists" column: ❌ W0 = file created/extended during Wave 0 scaffold (Plan 05-01).*

**Full-suite gate (run before `/gsd-verify-work`):** `pytest -x`

---

## Wave 0 Requirements

- [ ] `tests/test_classify_bugfixes.py` — extend with `TestAdjacencyBleed`, `TestRemovedAddedReconciliation`, `TestAsymmetricTolerance` classes (reuse existing `_span`/`_anchor`/`_classify` helpers)
- [ ] `tests/fixtures/classify/` — JSON fixtures representing the adjacency-bleed, close-proximity, and asymmetric-tolerance exemplars from `assets/debug_report_part*.json`
- [ ] Confirm `DeltaItem.confidence_flags` serializes/deserializes against existing `debug_report_part*.json` fixtures (backward-compat check)

---

## Corpus Regression Gate (Phase-Specific)

| Check | Command | Pass Criterion |
|-------|---------|----------------|
| 9-part corpus re-run | `pytest tests/test_corpus_regression.py` (or existing corpus harness) | Every `debug_report_part*.json` shows ≥ pre-fix conforming count; **zero** previously-passing chars flip to a wrong classification |
| CLS-01 exemplars | part 1 chars 11/12, part 4 char 7 | Confidence flag attached, not marked `changed` on bleed alone |
| CLS-02 exemplars | parts 2, 3, 4, 8 removed+added pairs | Emitted as single `changed` rows |
| CLS-03 exemplar | part 7 char 4 (`±1°` → `+0.3°/−0.1°`) | Classified `changed`, not `unchanged` |

*`ground_truth.json` files are **never modified**.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Human-readable confidence flag copy | CLS-01 | User-facing wording needs reviewer sign-off | Spot-check emitted DeltaItem JSON for part 1 char 11; confirm phrase matches ROADMAP success criterion |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers `tests/test_classify_bugfixes.py` extensions and fixture directory
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s for full suite
- [ ] 9-part corpus regression harness identified and wired to Phase gate
- [ ] `nyquist_compliant: true` set in frontmatter after Wave 0 passes

**Approval:** pending
