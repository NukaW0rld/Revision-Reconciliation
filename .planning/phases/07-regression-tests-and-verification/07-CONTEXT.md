# Phase 7: Regression Tests and Verification - Context

**Gathered:** 2026-04-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Write parametrized pytest regression coverage for all 6 fix clusters from Phases 4-6, add a
cross-part snapshot benchmark that guards aggregate accuracy, and document a full 9-part
ground-truth re-run as a committed VERIFICATION.md artifact. No web tier changes, no algorithm
fixes, and no edits to `ground_truth.json`. Phase is not closeable until VERIFICATION.md is
committed.

</domain>

<decisions>
## Implementation Decisions

### TST-01: Per-Cluster Regression Coverage
- **D-01:** Audit each existing Phase 4-6 test file against its fix cluster to verify at least one
  parametrized case provably fails on the pre-fix code path. Gap-filling cases are added to the
  existing phase-specific test files (not a separate new file per cluster). The audit covers all 6
  clusters: GD&T compact token splitting, GD&T word-name normalization, composite FCF capture,
  adjacency bleed suppression (CLS-01), removed+added resolution (CLS-02), asymmetric tolerance
  detection (CLS-03), missing added rows (ADD-01/ADD-02), and title block exclusion (SNP-01).
- **D-02:** After gap-filling, write a single `tests/test_phase7_regression.py` that imports or
  re-runs one exemplar per cluster as an explicit milestone checkpoint. This file serves as a
  human-readable audit trail showing all 6 (or 8) fix areas are covered and passing, without
  duplicating every parametrized case from the phase-specific files.

### TST-02: Cross-Part Benchmark
- **D-03:** The cross-part benchmark uses a snapshot-based approach: load the existing
  `assets/debug_report_partN.json` files (all 9 parts now committed), compute conformance counts
  (conforming count, review_needed count, `missing_added_truth_indexes` length) per part using
  the evaluation layer, and assert those counts meet or exceed a locked baseline. No pipeline
  execution at test time — benchmark runs in milliseconds.
- **D-04:** The benchmark covers all 9 parts (parts 1-9), since all 9 `debug_report_partN.json`
  files are present in `assets/`. No new pipeline runs are required.
- **D-05:** The locked baseline lives as a hardcoded `BASELINE_COUNTS` dict inside
  `tests/test_phase7_benchmark.py`. Keys are part names (e.g. `"part1"`); values hold expected
  minimum conforming count and maximum missing_added count. Updated by editing the test when the
  baseline genuinely improves. The assertion is directional: conforming count must be >= baseline,
  `missing_added_truth_indexes` length must be <= baseline ceiling.

### VER-01: Full 9-Part Verification
- **D-06:** VER-01 is a manual verification step, not an automated pytest test. The developer runs
  `python run.py <part_name>` for each of the 9 parts against the current algorithm, captures
  per-part ground-truth evaluation results (conforming count, exception count,
  `missing_added_truth_indexes`), and commits a `VERIFICATION.md` to the phase directory. Phase 7
  is not complete and must not be marked closed until `VERIFICATION.md` exists and is committed.
- **D-07:** `VERIFICATION.md` must record: part name, run date, conforming count, review_needed
  count, `missing_added_truth_indexes`, and a pass/fail verdict against the pre-Phase-4 baseline
  for each part. Any regression (conforming count lower than pre-fix baseline for a previously
  passing part) must be flagged explicitly.

### Claude's Discretion
- Whether `test_phase7_regression.py` re-imports parametrized cases directly from the phase-specific
  test modules or writes minimal inline fixture exemplars per cluster — choose the approach that
  keeps the file readable and avoids circular imports.
- Exact assertion thresholds for `review_needed` count in TST-02 — Claude may use >= (lower is
  better) or an equality check if counts are known to be stable.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` — TST-01, TST-02, VER-01 acceptance criteria (lines 61-74)
- `.planning/ROADMAP.md` — Phase 7 success criteria

### Existing Phase 4-6 Test Files (audit targets for TST-01)
- `tests/test_semantic_extraction.py` — GD&T compact token, word-name normalization, composite FCF
  (Phase 4 clusters; verify at least one parametrized fail-on-unfixed case per cluster)
- `tests/test_classify_bugfixes.py` — CLS-01 adjacency bleed suppression, CLS-03 asymmetric
  tolerance detection (Phase 5)
- `tests/test_classify_phase5_regression.py` — CLS-02 removed+added reconciliation (Phase 5)
- `tests/test_phase6_exclusion.py` — SNP-01 title block exclusion contract (Phase 6)
- `tests/test_phase6_asset_regression.py` — ADD-01/ADD-02 added detection and false-positive
  suppression for Parts 8 and 9 (Phase 6)
- `tests/test_added_detection_phase6.py` — Phase 6 added detection path (Phase 6)

### Snapshot Fixtures for TST-02 Benchmark
- `assets/debug_report_part1.json` through `assets/debug_report_part9.json` — post-fix pipeline
  snapshots; used as benchmark input, not re-run at test time
- `assets/part1/ground_truth.json` through `assets/part9/ground_truth.json` — canonical truth for
  each part; read-only

### Evaluation Layer
- `delta_preservation/evaluation/conformance.py` — conformance evaluation logic used by TST-02
  benchmark to compute counts from snapshot data

### Prior Phase Context (background only)
- `.planning/phases/04-gd-t-parser-fixes/04-CONTEXT.md` — Phase 4 decisions
- `.planning/phases/06-added-characteristic-detection-and-snippet-accuracy/06-CONTEXT.md` — Phase 6
  decisions

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `assets/debug_report_partN.json` (all 9 parts): post-fix snapshot files already committed;
  TST-02 benchmark reads these directly without running the pipeline
- `tests/conftest.py`: shared fixtures (session-scoped DB, tmp directories) available to new test
  files
- `delta_preservation/evaluation/conformance.py`: `evaluate_packet()` or equivalent function
  computes conformance counts from a `DeltaPacket` + ground truth; TST-02 should call this layer
  rather than reimplementing count logic inline

### Established Patterns
- Regression tests use `pytest.mark.parametrize` with explicit input strings and expected outcomes
  (established in Phase 4 D-08 and followed in all subsequent phases)
- Test files are named `test_<scope>.py` (snake_case) and placed flat in `tests/`
- Snapshot-based tests load JSON with `json.load(open(...))` and assert on specific fields by
  path (see `test_classify_phase5_regression.py` for reference)
- Phase milestone test files use a descriptive module docstring explaining the three test classes
  and what each validates (see `test_phase6_asset_regression.py` header for style)

### Integration Points
- `test_phase7_regression.py` must not break the existing 350+ passing tests — only adds new
  parametrized cases and imports
- TST-02 benchmark reads `debug_report_partN.json` which are standard `DeltaPacket`-shaped JSON;
  `evaluation/conformance.py` is the correct layer to call for count derivation

</code_context>

<specifics>
## Specific Ideas

- For `test_phase7_regression.py`, one checkpoint exemplar per cluster is sufficient — the goal is
  a readable milestone artifact, not exhaustive re-coverage. Importing the most specific
  parametrized case from each phase-specific file (if technically clean) is preferred over
  copy-pasting fixture data.
- TST-02 `BASELINE_COUNTS` dict should be populated from the actual conformance counts computed
  from the current `debug_report_partN.json` files at the time of implementation — not estimated.
  The implementation plan should include a step to derive these counts before writing the test.
- `VERIFICATION.md` must include a comparison table with pre-Phase-4 baseline counts (which can
  be sourced from the v1.0 milestone audit or the oldest committed debug reports) to make the
  improvement visible.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 07-regression-tests-and-verification*
*Context gathered: 2026-04-17*
