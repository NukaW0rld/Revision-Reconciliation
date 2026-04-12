---
phase: 01
slug: evaluation-foundation
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-10
---

# Phase 01 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest` via `uv run pytest` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/test_ground_truth_loader.py tests/test_ground_truth_evaluation.py tests/test_snippet_evaluation.py -q` |
| **Full suite command** | `uv run pytest -q` |
| **Estimated runtime** | ~30 seconds locally |

---

## Sampling Rate

- After every task commit: run the task-specific `uv run pytest ... -q` command from the plan
- After every plan wave: run `uv run pytest -q`
- Before `/gsd-verify-work`: full suite must be green
- Max feedback latency: 30 seconds for targeted checks

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-T1 | 01 | 1 | GTRU-01, GTRU-02 | - | Truth fixtures are loaded from exact asset keys only | unit | `uv run pytest tests/test_ground_truth_loader.py -q` | `tests/test_ground_truth_loader.py` | pending |
| 01-01-T2 | 01 | 1 | GTRU-02, GTRU-03 | - | Missing or malformed truth fails the run without mutating fixtures | unit | `uv run pytest tests/test_pipeline_task.py -q` | `tests/test_pipeline_task.py` | pending |
| 01-02-T1 | 02 | 2 | EVAL-01, EVAL-02 | - | Classification and requirement comparisons are deterministic and additive | unit | `uv run pytest tests/test_ground_truth_evaluation.py -q` | `tests/test_ground_truth_evaluation.py` | pending |
| 01-02-T2 | 02 | 2 | EVAL-01, EVAL-02 | - | Evaluated rows are serialized into `delta_packet.json` | unit | `uv run pytest tests/test_ground_truth_evaluation.py tests/test_output_formatting.py -q` | `tests/test_output_formatting.py` | pending |
| 01-03-T1 | 03 | 3 | EVAL-03, EVAL-04 | - | Snippet tolerance accepts visually valid context by deterministic bbox rules | unit | `uv run pytest tests/test_snippet_evaluation.py -q` | `tests/test_snippet_evaluation.py` | pending |
| 01-03-T2 | 03 | 3 | EVAL-05 | - | Ordered mismatch entries remain available to review helpers without manual verdicts | integration | `uv run pytest tests/test_snippet_evaluation.py tests/test_debug_internals.py -q` | `tests/test_debug_internals.py` | pending |

Status values: `pending`, `green`, `red`, `flaky`

---

## Wave 0 Requirements

Existing infrastructure covers all Phase 1 requirements.

---

## Manual-Only Verifications

All Phase 1 behaviors have automated verification targets. Visual review remains a downstream consumer concern for later phases, not a blocker for this validation strategy.

---

## Validation Sign-Off

- [ ] All tasks have automated verify commands
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all missing references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s for targeted checks
- [ ] `nyquist_compliant: true` set in frontmatter after plan verification

**Approval:** approved 2026-04-10
