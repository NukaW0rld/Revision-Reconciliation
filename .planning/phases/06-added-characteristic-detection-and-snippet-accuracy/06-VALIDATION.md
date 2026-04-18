---
phase: 06
slug: added-characteristic-detection-and-snippet-accuracy
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-16
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest >=8` via `uv run pytest` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/test_phase6_exclusion.py tests/test_added_detection_phase6.py tests/test_added_truth_selection.py tests/test_phase6_asset_regression.py -x` |
| **Full suite command** | `uv run pytest -x` |
| **Estimated runtime** | ~45 seconds full suite / <10 seconds phase-local smokes |

---

## Sampling Rate

- After every task commit: run the narrowest task-local smoke from the map below.
- After every wave:
  - Wave 1: `uv run pytest tests/test_phase6_exclusion.py tests/test_alignment_multishift.py::test_generate_candidates_excludes_tolerance_block_boilerplate -x`
  - Wave 2: `uv run pytest tests/test_added_detection_phase6.py -x`
  - Wave 3: `uv run pytest tests/test_added_truth_selection.py tests/test_debug_row_identity.py -x`
  - Wave 4: `uv run pytest tests/test_phase6_asset_regression.py -x`
- Before `$gsd-verify-work`: `uv run pytest -x`
- Max feedback latency: keep task-local smokes under 10 seconds and the full suite under 60 seconds.

---

## Per-Task Verification Map

| Plan | Wave | Requirement(s) | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|------|------|----------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 06-01 (shared exclusion) | 1 | SNP-01, ADD-02 | T-06-01, T-06-02 | All search surfaces use one exclusion helper; boilerplate spans are rejected consistently without hiding real edge annotations | unit | `uv run pytest tests/test_phase6_exclusion.py tests/test_alignment_multishift.py::test_generate_candidates_excludes_tolerance_block_boilerplate -x` | ❌ new tests + helper module | ⬜ pending |
| 06-02 (grouped added evidence) | 2 | ADD-01, ADD-02 | T-06-04, T-06-05 | Standard added rows carry grouped text+bbox and fragments already explained by matched annotations are suppressed | unit | `uv run pytest tests/test_added_detection_phase6.py -x` | ❌ new test file | ⬜ pending |
| 06-03 (truth claiming) | 3 | ADD-01 | T-06-07 | Duplicate canonical added rows are claimed only when packet Rev B evidence identifies one deterministically; ambiguity stays conservative | unit | `uv run pytest tests/test_added_truth_selection.py tests/test_debug_row_identity.py -x` | ❌ one new file + existing file extended | ⬜ pending |
| 06-04 (asset regression) | 4 | ADD-01, ADD-02, SNP-01 | T-06-09 | Read-only asset-backed regressions guard Part 8 / Part 9 exemplars without mutating fixtures | integration | `uv run pytest tests/test_phase6_asset_regression.py -x` | ❌ new file | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all planned Phase 6 behaviors. No separate Wave 0 scaffolding is required before execution starts.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Full 9-part corpus rerun | VER-01 successor gate in Phase 7 | Phase 6 remains phase-local and should not own the expensive end-to-end rerun | Defer to Phase 7 benchmark / rerun workflow |

---

## Validation Sign-Off

- [ ] All plans have task-local automated verification
- [ ] Shared exclusion tests cover title block, revision table, and bottom-center boilerplate
- [ ] Added detection tests cover grouped evidence and explained-by-match suppression
- [ ] Duplicate added-truth tests cover unique, duplicate, and ambiguity paths
- [ ] Asset-backed harness stays read-only
- [ ] No watch-mode flags
- [ ] `nyquist_compliant: true` set in frontmatter after execution evidence is captured

**Approval:** pending

---

## Phase 9 Follow-Up Note (2026-04-18)

The final full-corpus closure for ADD-01 is now proven by refreshed standalone
and algorithm-only evidence captured during Phase 9 Plan 03:

- **Standalone reruns:** All 9 parts re-run through `uv run python run.py partN`
  at HEAD `8a611c9`.  Parts 1-4 and 6-9 now show empty `missing_added_truth_indexes`.
  Part 5 has 2 deferred items (indexes 16 and 17) that require matching-layer
  architectural fixes outside Phase 9 scope (documented in 09-02-SUMMARY.md).
- **Algorithm-only fixtures:** `tests/fixtures/phase7_algorithm_only/part{1..9}-debug-report.json`
  refreshed with post-Phase-9 counts.  `tests/test_phase7_benchmark.py::BASELINE_COUNTS`
  updated to reflect zero missing-added rows for 8 of 9 parts.
- **Historical Phase 6 assets:** `assets/debug_report_part*.json` remain frozen
  historical Phase 6 corpus evidence; they were NOT modified during Phase 9.
- **Phase 9 plan references:** 09-01-PLAN.md (token contract), 09-02-PLAN.md
  (detector-side fixes), 09-03-PLAN.md (fixture refresh and traceability closure).
