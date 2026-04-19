---
status: complete
phase: 05-classification-logic-fixes
source: [05-01-SUMMARY.md, 05-02-SUMMARY.md, 05-03-SUMMARY.md, 05-04-SUMMARY.md, 05-05-SUMMARY.md]
started: 2026-04-17T01:15:20Z
updated: 2026-04-17T01:20:06Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

[testing complete]

## Tests

### 1. Confidence Flags Compatibility
expected: Run a Phase 5 classification path that serializes pipeline output, or run the targeted compatibility tests. The pipeline/CLI should not crash when an internal DeltaItem lacks `confidence_flags`, and missing `confidence_flags` should deserialize/serialize as `[]` instead of raising an error.
result: pass
verified_by: `uv run pytest tests/test_classify_bugfixes.py::TestConfidenceFlagsCompatibility tests/test_pipeline_semantic_packet.py tests/test_output_formatting.py -x` (20 passed)

### 2. Adjacency Bleed Is Suppressed and Flagged
expected: For a slash-merged Rev B span such as `4 x Ø8 THRU ALL / ⌴ Ø13.5 ↧ 8.5` or `2X Ø.201 ↧ 0.50 / 1/4-20 UNC - 2B`, the item is not classified as changed from `count_added` alone. The output carries the advisory confidence flag `Rev B text may contain adjacent balloon content`.
result: pass
verified_by: `uv run pytest tests/test_classify_bugfixes.py::TestAdjacencyBleed tests/test_classify_bugfixes.py::TestCountAdded tests/test_classify_bugfixes.py::TestAsymmetricTolerance tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation tests/test_classify_phase5_regression.py -x` (28 passed)

### 3. Legitimate Slash-Separated Changes Stay Changed
expected: Benign slash text such as `70 / 30`, asymmetric tolerance text, or ordinary no-slash count changes are not over-suppressed by the bleed logic. Real changed cases still classify as `changed` and do not receive the bleed flag unless adjacent-balloon bleed is actually present.
result: pass
verified_by: `uv run pytest tests/test_classify_bugfixes.py::TestAdjacencyBleed tests/test_classify_bugfixes.py::TestCountAdded tests/test_classify_bugfixes.py::TestAsymmetricTolerance tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation tests/test_classify_phase5_regression.py -x` (28 passed)

### 4. Symmetric to Asymmetric Tolerance Changes Classify as Changed
expected: A tolerance transition like `22.0° ±1°` to `22.0° +0.3° / −0.1°` is classified as `changed`, even if overlap logic might otherwise look compatible. The output includes a reason indicating the tolerance kind transition or asymmetric change.
result: pass
verified_by: `uv run pytest tests/test_classify_bugfixes.py::TestAdjacencyBleed tests/test_classify_bugfixes.py::TestCountAdded tests/test_classify_bugfixes.py::TestAsymmetricTolerance tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation tests/test_classify_phase5_regression.py -x` (28 passed)

### 5. Same-Page Removed+Added Pairs Reconcile into One Changed Row
expected: When a removed Rev A characteristic has a close unmatched added characteristic on the same page, the pipeline emits a single `changed` item instead of separate `removed` and `added` rows.
result: pass
verified_by: `uv run pytest tests/test_classify_bugfixes.py::TestAdjacencyBleed tests/test_classify_bugfixes.py::TestCountAdded tests/test_classify_bugfixes.py::TestAsymmetricTolerance tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation tests/test_classify_phase5_regression.py -x` (28 passed)

### 6. Cross-Page or Incompatible Added Candidates Stay Separate
expected: Cross-page, too-far, or type-incompatible added candidates do not reconcile with removed items. The output remains separate `removed` and `added` rows in those cases.
result: pass
verified_by: `uv run pytest tests/test_classify_bugfixes.py::TestAdjacencyBleed tests/test_classify_bugfixes.py::TestCountAdded tests/test_classify_bugfixes.py::TestAsymmetricTolerance tests/test_classify_bugfixes.py::TestRemovedAddedReconciliation tests/test_classify_phase5_regression.py -x` (28 passed)

### 7. Regression Harness and Full Suite Stay Green
expected: `pytest tests/test_classify_phase5_regression.py -x` passes against the checked-in debug-report fixtures, and the broader regression suite for Phase 5/full tests stays green without editing any `ground_truth.json` files.
result: pass
verified_by: `uv run pytest tests/test_classify_phase5_regression.py -x` (9 passed) and `uv run pytest -x` (351 passed, 2 xfailed, 2 warnings)

## Summary

total: 7
passed: 7
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none yet]
