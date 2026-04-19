# Phase 7: Regression Tests and Verification - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-17
**Phase:** 07-regression-tests-and-verification
**Areas discussed:** TST-01 gap strategy, TST-02 benchmark design, VER-01 verification format, Baseline manifest source, Regression root-cause triage

---

## TST-01 Gap Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Audit + fill gaps only | Read each existing test file, verify at least one parametrized case per cluster provably fails on pre-fix code. Add new tests only where that check fails — no redundant duplication. | |
| New consolidating test file | Write a single test_phase7_regression.py covering all 6 clusters with new parametrized cases regardless of what Phases 4-6 already shipped. | |
| Audit + consolidating summary | Audit existing files first, add gap-filling cases to the existing phase-specific test files, then write a short test_phase7_regression.py with one exemplar per cluster as a milestone checkpoint. | ✓ |

**User's choice:** Audit + consolidating summary
**Notes:** Gap-filling goes into existing phase-specific test files. `test_phase7_regression.py` serves as a readable milestone artifact — one exemplar per cluster, not exhaustive re-coverage.

---

## TST-02 Benchmark Design

| Option | Description | Selected |
|--------|-------------|----------|
| Snapshot-based (fast) | Load existing debug_report_partN.json files, compute conformance counts, assert against locked baseline dict. No pipeline execution. | ✓ |
| Evaluation re-run (medium) | Load ground_truth.json + delta.json from each part's out/ directory, re-run only the evaluation layer. Requires out/ artifacts to exist. | |
| Full pipeline integration (slow) | Run run_pipeline() for all 9 parts inside a slow-marked pytest integration test. Takes minutes, requires all 9 asset PDFs. | |

**User's choice:** Snapshot-based (fast)
**Notes:** All 9 debug_report_partN.json files are already committed in assets/, so all 9 parts are covered without any new pipeline runs.

---

## TST-02 Part Coverage

| Option | Description | Selected |
|--------|-------------|----------|
| Only parts with committed debug_report JSON | Use only the files already in assets/. No new pipeline runs. | |
| All 9 parts — generate missing reports first | Run pipeline for missing parts, commit JSON files, lock all 9. | |

**User's choice:** Only parts with committed debug_report JSON
**Notes:** Discovery during discussion: all 9 debug_report files already exist in assets/ (parts 1-9), so this decision effectively covers all 9 parts anyway.

---

## VER-01 Verification Format

| Option | Description | Selected |
|--------|-------------|----------|
| Manual run + VERIFICATION.md artifact | Developer runs python run.py partN for each part, captures counts, commits VERIFICATION.md. Phase not complete until doc exists. | ✓ |
| Script-assisted + VERIFICATION.md | Ship a helper script that iterates all 9 parts, runs pipeline, generates VERIFICATION.md automatically. Still requires manual trigger. | |
| Pytest integration test (slow marker) | pytest test file marked @pytest.mark.slow running all 9 pipeline integrations. | |

**User's choice:** Manual run + VERIFICATION.md artifact
**Notes:** VERIFICATION.md is a hard gate — phase cannot be closed without it.

---

## Baseline Manifest Source

| Option | Description | Selected |
|--------|-------------|----------|
| Hardcoded dict in test file | BASELINE_COUNTS dict defined directly in test_phase7_benchmark.py. Easy to read and diff. | ✓ |
| Committed JSON fixture in assets/ | assets/baseline_counts.json loaded at runtime. Decouples baseline from test logic. | |
| Derived dynamically | Test computes expected counts from current debug_report files on first run and stores as a lockfile. | |

**User's choice:** Hardcoded dict in test file
**Notes:** Counts should be derived from actual conformance evaluation of current debug_report files at implementation time — not estimated. Implementation plan should include a derivation step before writing the dict.

---

## Regression Root-Cause Triage

| Option | Description | Selected |
|--------|-------------|----------|
| Treat Parts 1, 3, and 8 as equal algorithm regressions | Assume every verification failure is a real algorithm issue and triage them in parallel. | |
| Separate verification-contract issues from real regressions | Fix baseline parity first, then treat only the remaining failures as algorithm regressions. | ✓ |
| Keep existing Phase 7 verification as-is | Accept the mixed-mode snapshot baseline and use the current blocked verification report without changing the parity contract. | |

**User's choice:** Separate verification-contract issues from real regressions
**Notes:** Working stance approved by user: Part 1 is not a clean regression because the baseline snapshot used accepted-alternate-backed conformance while standalone `run.py` reruns do not; Part 8 is not a clean regression because the committed snapshot baseline still contains a stale `(no_eval)` missing-added row; Part 3 is the first likely real algorithm regression and should be investigated first after the verification contract is made apples-to-apples.

---

## Claude's Discretion

- Whether `test_phase7_regression.py` re-imports from phase-specific test modules or writes minimal inline exemplars per cluster
- Exact assertion style for `review_needed` count in TST-02 (>= or ==)

## Deferred Ideas

None — discussion stayed within phase scope.
