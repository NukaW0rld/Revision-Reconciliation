---
phase: 09
slug: full-corpus-added-characteristic-closure
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-18
reconciled: 2026-04-19
---

# Phase 09 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Plan / Wave Graph

| Plan | Wave | Depends On | Validation Focus |
|------|------|------------|------------------|
| `09-01` | 1 | — | Repair packet/evaluation truth-token accounting and deterministic added-text normalization so Part 1/2 stop failing for non-detector reasons. |
| `09-02` | 2 | `09-01` | Close the remaining detector/suppressor misses with shared-path fixes while preserving the Phase 6 ADD-02 guardrails. |
| `09-03` | 3 | `09-01`, `09-02` | Refresh standalone evidence, algorithm-only fixtures, and Phase 06/07 verification artifacts from the post-fix corpus state. |

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest >=8` via `uv run pytest` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest -q tests/test_added_detection_phase6.py tests/test_added_truth_selection.py tests/test_debug_row_identity.py tests/test_phase7_benchmark.py -x` |
| **Full suite command** | `uv run pytest -q tests/test_added_detection_phase6.py tests/test_added_truth_selection.py tests/test_debug_row_identity.py tests/test_phase7_benchmark.py tests/test_phase6_asset_regression.py -x` |
| **Estimated quick runtime** | ~20-40 seconds |

---

## Sampling Rate

- After every task commit: run the narrowest task-local command from the map below.
- After every wave:
  - Wave 1: `uv run pytest -q tests/test_added_truth_selection.py tests/test_debug_row_identity.py tests/test_phase7_benchmark.py -k "part1 or part2 or added" -x`
  - Wave 2: `uv run pytest -q tests/test_added_detection_phase6.py tests/test_added_truth_selection.py tests/test_debug_row_identity.py -x`
  - Wave 3: `uv run pytest -q tests/test_phase7_benchmark.py tests/test_phase6_asset_regression.py -x`
- Before `$gsd-verify-work`: the full suite command above must be green and the refreshed standalone rerun evidence must be captured.
- Max feedback latency: keep task-local checks under 30 seconds; allow the final evidence-refresh wave to spend longer on standalone reruns.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 09-01 | 1 | ADD-01 | T-09-01, T-09-02 | Canonical added rows already matched by the evaluator are counted consistently in maintainer/debug missing-added accounting. | unit | `uv run pytest -q tests/test_debug_row_identity.py tests/test_added_truth_selection.py -k "added or truth" -x` | ✅ existing files extended | ⬜ pending |
| 09-01-02 | 09-01 | 1 | ADD-01 | T-09-02 | Deterministic normalization treats harmless leading-zero and spacing variants as the same added requirement without fuzzy matching. | unit | `uv run pytest -q tests/test_added_truth_selection.py tests/test_phase7_benchmark.py -k "part2 or normalization or added" -x` | ✅ existing files extended | ⬜ pending |
| 09-02-01 | 09-02 | 2 | ADD-01, ADD-02 | T-09-03, T-09-04 | Explained-by-match suppression no longer absorbs unrelated same-row annotations, but still suppresses truly explained fragments. | unit | `uv run pytest -q tests/test_added_detection_phase6.py -k "suppression or part9 or added" -x` | ✅ existing file extended | ⬜ pending |
| 09-02-02 | 09-02 | 2 | ADD-01, SNP-01 | T-09-04 | Remaining Parts 3-5 misses are closed through shared detector/grouping behavior without breaking title-block/snippet ownership rules. | integration | `uv run pytest -q tests/test_added_detection_phase6.py tests/test_phase6_asset_regression.py -x` | ✅ existing files extended | ⬜ pending |
| 09-03-01 | 09-03 | 3 | ADD-01 | T-09-05 | Fresh standalone reruns across parts 1-9 yield empty `missing_added_truth_indexes` and the refreshed algorithm-only fixtures preserve that state. | corpus rerun + document | `uv run pytest -q tests/test_phase7_benchmark.py tests/test_phase6_asset_regression.py -x` | ✅ existing fixtures + docs updated | ⬜ pending |
| 09-03-02 | 09-03 | 3 | ADD-01, ADD-02, SNP-01 | T-09-05 | Phase 06/07 verification artifacts explicitly distinguish refreshed current evidence from frozen historical Phase 6 assets. | document | `rg -n "missing_added_truth_indexes|historical Phase 6|algorithm-only fixture set|Phase 9" .planning/phases/06-added-characteristic-detection-and-snippet-accuracy/06-VERIFICATION.md .planning/phases/07-regression-tests-and-verification/07-04-ALGORITHM-ONLY-BASELINE.md .planning/phases/07-regression-tests-and-verification/07-VERIFICATION.md` | ✅ existing docs updated | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] Existing pytest infrastructure already covers detector, evaluator, and benchmark surfaces.
- [x] No new framework install is required.
- [x] Existing standalone rerun entrypoint (`run.py`) and algorithm-only fixture location (`tests/fixtures/phase7_algorithm_only/`) are available.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Fresh full-corpus standalone rerun | ADD-01 | The final closure claim depends on regenerated packet/evidence artifacts, not only unit tests. | Run `uv run python run.py partN` for `part1` through `part9`, capture each resulting packet/debug artifact, and confirm the refreshed benchmark/docs record empty `missing_added_truth_indexes`. |

---

## Threat References

| Threat ID | Category | Concern |
|-----------|----------|---------|
| T-09-01 | Integrity | Review/debug missing-added accounting can disagree with packet truth claims if token semantics are inconsistent. |
| T-09-02 | Integrity | Deterministic normalization could over-broaden and accidentally collapse distinct added rows if it becomes fuzzy instead of rule-based. |
| T-09-03 | Integrity | Narrowing Part 9 suppression incorrectly could reintroduce the fragment false positives Phase 6 already fixed. |
| T-09-04 | Evidence | Parts 3-5 closure work could drift into part-specific hacks or title-block/snippet regressions if not guarded by existing Phase 6 tests. |
| T-09-05 | Traceability | Final evidence refresh could overwrite historical Phase 6 artifacts or refresh benchmark docs from an unverified intermediate state. |

---

## Validation Sign-Off

- [x] Existing infrastructure covers all planned work areas
- [x] Sampling continuity is preserved across the anticipated three waves
- [x] Wave 0 gaps are already closed by existing tooling and fixtures
- [x] No watch-mode flags are used
- [x] `nyquist_compliant: true` set after Phase 12 reconciliation

**Approval:** pending

---

## Verification Closure Reconciliation (Phase 12 update)

The `nyquist_compliant` flag was flipped to `true` during Phase 12 process-artifact closure because `09-VERIFICATION.md` records the phase as `verified` with 3/3 success criteria met: all 9 corpus parts verified at zero `missing_added_truth_indexes`, no new false positives introduced, and Phase 06/07 artifacts refreshed to reflect the final full-corpus result. See `09-VERIFICATION.md` and `v1.1-MILESTONE-AUDIT.md (tech_debt: phase 09)` for the complete audit record of this transition.
