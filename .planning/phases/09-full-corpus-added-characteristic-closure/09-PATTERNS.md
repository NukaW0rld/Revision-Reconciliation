# Phase 9: Full-Corpus Added Characteristic Closure — Pattern Map

**Mapped:** 2026-04-18
**Files analyzed:** 10
**Analogs found:** 10 / 10 — every anticipated touchpoint already has a clear in-repo analog.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `delta_preservation/evaluation/conformance.py` | evaluator truth selection | packet row -> canonical truth row | existing `select_truth_row_for_item()` exact-text + bbox tie-break flow | exact |
| `shop/services/review.py` | maintainer/debug accounting | packet evaluation token -> synthetic missing-added rows | existing `_load_missing_added_truth_items()` collector | exact |
| `delta_preservation/reconcile/normalize.py` | shared normalization helper | raw requirement text -> canonical comparison form | existing `classify_requirement_type()` / parser normalization patterns | strong |
| `delta_preservation/reconcile/classify.py` | added detection + suppressor | unmatched Rev B spans -> added `DeltaItem` rows | existing `detect_added_characteristics()` grouping, suppressor, and CLS-02 post-pass | exact |
| `tests/test_added_truth_selection.py` | evaluator behavior test | added packet requirement -> selected truth row | existing duplicate/ambiguity tests | exact |
| `tests/test_debug_row_identity.py` | debug-accounting regression | packet evaluation -> debug queue state | existing missing-added and identity tests | exact |
| `tests/test_added_detection_phase6.py` | detector/suppressor regression | unmatched spans -> added packet rows | existing grouped-evidence and suppressor tests | exact |
| `tests/test_phase7_benchmark.py` | locked benchmark contract | refreshed per-part fixtures -> acceptance ceiling | existing algorithm-only baseline assertions | exact |
| `tests/fixtures/phase7_algorithm_only/part*-debug-report.json` | refreshed baseline fixtures | standalone rerun output -> committed benchmark input | existing Phase 7 algorithm-only fixture set | exact |
| `.planning/phases/06-*/06-VERIFICATION.md`, `.planning/phases/07-*/07-04-ALGORITHM-ONLY-BASELINE.md`, `.planning/phases/07-*/07-VERIFICATION.md` | evidence refresh docs | refreshed corpus result -> milestone proof | existing Phase 7 parity-correction docs | strong |

## Pattern Assignments

### Packet/evaluator truth contract

**Analog:** `delta_preservation/evaluation/conformance.py::select_truth_row_for_item()`

**Pattern to preserve**

- deterministic normalization first
- exact requirement match before geometry tie-break
- conservative ambiguity when evidence is insufficient
- serialize the chosen truth row through one shared token path

**Planning implication**

Part 1 and Part 2 fixes should stay in evaluator/token semantics, not be patched in the review surface or benchmark.

### Debug-accounting consumer contract

**Analog:** `shop/services/review.py::_load_missing_added_truth_items()`

**Pattern to preserve**

- consume evaluator output directly
- avoid re-deriving added truth ownership from raw packet text
- keep the consumer read-only relative to canonical truth selection

**Planning implication**

If token semantics change, update the consumer to accept the shared contract rather than inventing Phase-9-only branches.

### Explained-by-match suppressor

**Analog:** `delta_preservation/reconcile/classify.py` block building `matched_annotation_signatures`

**Pattern to preserve**

- content ownership and bbox ownership are both required
- proximity alone is not enough to suppress
- positive and negative suppressor cases already live together in `tests/test_added_detection_phase6.py`

**Planning implication**

Phase 9 should narrow the owner-signature sweep and add direct tests around the Part 9 flatness case, not remove the suppressor or weaken the content gate.

### Baseline refresh

**Analog:** Phase 7 Plan 04 and its artifacts

**Pattern to preserve**

- refresh algorithm-only fixtures in `tests/fixtures/phase7_algorithm_only/`
- keep `assets/debug_report_part*.json` frozen as historical Phase 6 evidence
- update the benchmark provenance doc and verification report together

**Planning implication**

The final plan should touch fixture refresh and docs in one wave so the benchmark, baseline doc, and verification narrative stay in sync.

## Recommended File Set For Plans

### Plan 01

- `delta_preservation/evaluation/conformance.py`
- `shop/services/review.py`
- `tests/test_added_truth_selection.py`
- `tests/test_debug_row_identity.py`
- `tests/test_phase7_benchmark.py`

### Plan 02

- `delta_preservation/reconcile/classify.py`
- `delta_preservation/reconcile/normalize.py` (only if Phase 9 still needs shared alias/canonicalization support)
- `tests/test_added_detection_phase6.py`
- `tests/test_phase6_asset_regression.py`

### Plan 03

- `tests/fixtures/phase7_algorithm_only/part1-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part2-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part3-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part4-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part5-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part6-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part7-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part8-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part9-debug-report.json`
- `.planning/phases/07-regression-tests-and-verification/07-04-ALGORITHM-ONLY-BASELINE.md`
- `.planning/phases/07-regression-tests-and-verification/07-VERIFICATION.md`
- `.planning/phases/06-added-characteristic-detection-and-snippet-accuracy/06-VERIFICATION.md`

## Anti-Patterns To Avoid

- fixing Part 1 entirely in `shop/services/review.py` while leaving evaluator token semantics inconsistent
- broadening Part 2 normalization with fuzzy matching or per-part aliases
- suppressing Part 9 misses by deleting the explained-by-match pass
- refreshing current benchmark evidence by overwriting historical Phase 6 asset snapshots
