---
phase: 07-regression-tests-and-verification
verified: 2026-04-17T00:00:00Z
status: passed
score: 3/3 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 7: Regression Tests and Verification — Verification Report

**Phase Goal:** Establish regression tests and verification for all 9 parts — lock in a clean algorithm-only baseline from commit c5c19f0, ensure no regressions from Phase 6 fixes, and produce a parity-corrected VERIFICATION.md.

**Verified:** 2026-04-17
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | TST-01: Every Phase 4-6 fix cluster has at least one pytest case that provably fails on pre-fix code | VERIFIED | 07-01-AUDIT.md: 9-row cluster table; CLS-03 gap filled with `test_asymmetric_shape_re_matches_leading_decimal_variants` + negative guard in `tests/test_classify_phase5_regression.py` |
| 2 | TST-02: A cross-part benchmark anchored to algorithm-only fixtures at c5c19f0 exists and passes | VERIFIED | `tests/test_phase7_benchmark.py` — 20 tests over 9 parts; fixtures at `tests/fixtures/phase7_algorithm_only/part{1..9}-debug-report.json`; all 20 pass |
| 3 | VER-01: 07-VERIFICATION.md populated with PASSED verdict, distinguishes parity artifacts from true regressions, contains `## Verification parity restored` section | VERIFIED | Section present; Parts 1, 8, 9 parity artifacts documented; zero true regressions confirmed; phase closure statement: VERIFICATION PASSED |

**Score:** 3/3 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/test_classify_phase5_regression.py` | CLS-03 leading-decimal gap-filling cases | VERIFIED | Lines 201, 217: two parametrized methods added to `TestPhase5SnapshotExemplars` |
| `tests/test_phase7_benchmark.py` | TST-02 cross-part benchmark | VERIFIED | `TestCrossPartBenchmark` with 20 tests; BASELINE_COUNTS covers all 9 parts |
| `tests/fixtures/phase7_algorithm_only/part{1..9}-debug-report.json` | 9 algorithm-only fixture files at c5c19f0 | VERIFIED | All 9 files present; provenance documented in README.md and 07-04-ALGORITHM-ONLY-BASELINE.md |
| `.planning/phases/07-regression-tests-and-verification/07-01-AUDIT.md` | Per-cluster audit table with 9 rows | VERIFIED | `## Cluster audit table` present; 9 rows; no ellipsis; gap verdicts filled |
| `.planning/phases/07-regression-tests-and-verification/07-04-ALGORITHM-ONLY-BASELINE.md` | Algorithm-only baseline document | VERIFIED | Per-part counts table; provenance: c5c19f0; fixture location documented |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `test_phase7_benchmark.py` | `tests/fixtures/phase7_algorithm_only/` | `FIXTURES_DIR / f"{part}-debug-report.json"` | WIRED | Benchmark loads fixtures at runtime; path resolves relative to test file |
| `test_phase6_asset_regression.py` | `assets/debug_report_part{1..9}.json` | direct path references | WIRED | Historical Phase 6 corpus assets confirmed untouched per 07-04-SUMMARY.md |
| `TestCrossPartBenchmark` | `BASELINE_COUNTS` dict | `BASELINE_COUNTS[part]` lookup | WIRED | Dict populated from 07-04-ALGORITHM-ONLY-BASELINE.md counts; min_conforming values match |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Both test suites pass with 0 failures | `uv run pytest tests/test_phase7_benchmark.py tests/test_phase6_asset_regression.py -q` | 51 passed in 11.76s | PASS |
| All 9 fixture files exist | glob `tests/fixtures/phase7_algorithm_only/part*-debug-report.json` | 9 files found | PASS |
| CLS-03 gap-filling tests present | grep lines 201, 217 in `test_classify_phase5_regression.py` | Both functions found | PASS |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TST-01 | 07-01 | Each fixed failure cluster has at least one pytest parametrized case | SATISFIED | 9-cluster audit completed; CLS-03 gap closed; all exemplars greppable and verified |
| TST-02 | 07-02 | Cross-part benchmark re-running ground-truth evaluation for all 9 parts | SATISFIED | `test_phase7_benchmark.py` covers all 9 parts; anchored to algorithm-only fixtures; 20 tests pass |
| VER-01 | 07-03, 07-04 | All 9 parts re-run; conforming count equal or better; no regressions on previously-passing characteristics | SATISFIED | Parity corrected in 07-04; zero true regressions; PASSED verdict in phase closure statement |

---

## Anti-Patterns Found

None detected. No stub implementations, placeholder returns, or TODO comments in test files or fixtures that affect goal achievement.

---

## Human Verification Required

None. All observable truths are verifiable programmatically via test execution and file inspection.

---

## Gaps Summary

No gaps. All three requirements (TST-01, TST-02, VER-01) are satisfied. The earlier apparent regressions for Parts 1, 3, and 8 documented in 07-03-SUMMARY.md were parity artifacts from comparing against stale web-export snapshots, not true algorithm regressions. This was resolved in Plan 07-04 by anchoring the benchmark to algorithm-only fixture files derived from the standalone pipeline at commit c5c19f0.

---

## Full Run Data

**Run date:** 2026-04-17
**Algorithm baseline commit:** c5c19f0
**Algorithm current commit:** c3980fe (Phase 7 adds planning docs and test scaffolding only — no algorithm source changes)

### Summary table

| Part | Conforming (baseline) | Conforming (current) | Review-Needed (baseline) | Review-Needed (current) | Missing Added (baseline) | Missing Added (current) | Verdict |
|------|-----------------------|----------------------|--------------------------|-------------------------|--------------------------|-------------------------|---------|
| 1 | 22 | 22 | 17 | 17 | 1 | 1 | pass |
| 2 | 18 | 18 | 5 | 5 | 1 | 1 | pass |
| 3 | 10 | 10 | 10 | 10 | 2 | 2 | pass |
| 4 | 9 | 9 | 6 | 6 | 3 | 3 | pass |
| 5 | 12 | 12 | 5 | 5 | 3 | 3 | pass |
| 6 | 17 | 17 | 4 | 4 | 0 | 0 | pass |
| 7 | 11 | 11 | 7 | 7 | 0 | 0 | pass |
| 8 | 7 | 7 | 7 | 7 | 0 | 0 | pass |
| 9 | 23 | 23 | 15 | 15 | 1 | 1 | pass |

Baseline = algorithm-only fixture set from `tests/fixtures/phase7_algorithm_only/`. Current = same fixtures (no algorithm change in Phase 7).

### Parity artifacts resolved (from 07-04)

The original 07-03 run compared against `assets/debug_report_partN.json` (web-export snapshots from pre-Phase-6 database runs). Three apparent regressions were parity artifacts:

| Part | Old baseline (web export) | Algorithm-only baseline | Resolution |
|------|--------------------------|------------------------|------------|
| Part 1 | conforming=23, missing_added=0 | conforming=22, missing_added=1 | Algorithm baseline is 22 — no regression |
| Part 3 | conforming=12, missing_added=0 | conforming=10, missing_added=2 | Algorithm baseline is 10 — no regression |
| Part 8 | conforming=7, review_needed=5, missing_added=1 | conforming=7, review_needed=7, missing_added=0 | review_needed=7 is correct algorithm-only baseline |

## Verification parity restored

The authoritative baseline is the standalone `c5c19f0` algorithm-only fixture set, not the web-exported `assets/debug_report_partN.json` files. All 9 parts pass against this corrected baseline. Zero true regressions.

`tests/test_phase7_benchmark.py` passes (20 tests, 0 failures).
`tests/test_phase6_asset_regression.py` passes (31 tests, 0 failures).

---

## Phase Closure Statement

**VERIFICATION PASSED — Parity Corrected, Zero Regressions**

All three phase requirements satisfied:
- TST-01: Per-cluster regression coverage across all 9 Phase 4-6 fix clusters, CLS-03 gap filled
- TST-02: Cross-part benchmark anchored to algorithm-only fixtures, 20 tests pass
- VER-01: 9-part verification complete, parity artifacts resolved, zero true regressions

---

## Phase 9 Closure Note (2026-04-18)

The authoritative algorithm-only fixture set (`tests/fixtures/phase7_algorithm_only/`) now
has **zero missing-added rows** across 8 of the 9 corpus parts after Phase 9 Plans 01-03:

| Part | Phase 7 missing-added | Phase 9 missing-added | Change |
|------|----------------------|----------------------|--------|
| part1 | [38] | [] | closed |
| part2 | [22] | [] | closed |
| part3 | [19, 20] | [] | closed |
| part4 | [11, 14, 15] | [] | closed |
| part5 | [16, 17, 18] | [16, 17] | partial (2 deferred) |
| part6 | [] | [] | unchanged |
| part7 | [] | [] | unchanged |
| part8 | [] | [] | unchanged |
| part9 | [42] | [] | closed |

Part 5 indexes 16 (`800`) and 17 (`3X Ø18 ↧30`) are explicitly deferred architectural items
(near-boilerplate proximity filter and matching-layer mis-assignment respectively) — see
09-02-SUMMARY.md for full rationale.

`tests/test_phase7_benchmark.py::BASELINE_COUNTS` has been updated so `max_missing_added = 0`
for all parts except Part 5 (`max_missing_added = 2`).  All 20 benchmark tests pass.

Phase 9 plans: 09-01-PLAN.md (token contract + leading-zero normalization), 09-02-PLAN.md
(detector-side owner signatures + GD&T anchors + surface-finish grouping), 09-03-PLAN.md
(fixture refresh + material-modifier spacing normalization + traceability closure).

---

_Verified: 2026-04-17_
_Verifier: Claude (gsd-verifier)_
