# Phase 9: full-corpus-added-characteristic-closure - Context

**Gathered:** 2026-04-18 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Finish the added-characteristic closure work by eliminating the current algorithm-only `missing_added_truth_indexes` across the full 9-part corpus and refreshing the supporting evidence so milestone claims reflect the final behavior. Scope includes shared pipeline behavior, truth-selection logic, and any supporting evidence-accounting needed to make missing-added reporting truthful. No edits to `ground_truth.json`, no UI-only masking, and no Phase 10 gating/export work.

</domain>

<decisions>
## Implementation Decisions

### Authoritative Baseline and Evidence Refresh
- **D-01:** Phase 9 uses fresh standalone `run.py` reruns plus the Phase 7 algorithm-only fixture set in `tests/fixtures/phase7_algorithm_only/` as the authoritative starting baseline. The older `assets/debug_report_part*.json` files remain historical Phase 6 corpus evidence, not the source of truth for current miss counts.
- **D-02:** Refresh the algorithm-only verification evidence after fixes land. Preserve the frozen Phase 6 corpus assets used by `tests/test_phase6_asset_regression.py`; update the algorithm-only fixtures and verification/docs provenance instead of silently overwriting the historical snapshots.

### Closure Scope Across the Whole Corpus
- **D-03:** Phase 9 must close the full current mixed miss bucket, not only the confirmed Part 9 flatness loss. The planning baseline is: `part1 [38]`, `part2 [22]`, `part3 [19, 20]`, `part4 [11, 14, 15]`, `part5 [16, 17, 18]`, `part6 []`, `part7 []`, `part8 []`, `part9 [42]`.
- **D-04:** The remaining misses are treated as multiple failure families, not one generic detector bug. Planning should separate: Part 1 accounting, Part 2 normalization/claiming, older heterogeneous Parts 3-5 detector/grouping misses, and the confirmed Part 9 suppressor loss.

### Shared Fix Ownership
- **D-05:** Correctness fixes belong in the shared packet/evaluation contract: `delta_preservation/reconcile/classify.py`, `delta_preservation/evaluation/conformance.py`, `delta_preservation/cli.py`, and supporting missing-added accounting that derives maintainer-facing evidence. Do not solve Phase 9 with truth-file edits or review-UI masking.
- **D-06:** The Part 1 false miss is treated as an evidence-accounting defect, not a detector recall problem. The fix must make canonical added-row claims visible to missing-added accounting without weakening the added-truth contract.

### Added-Detection Safety Rails
- **D-07:** Keep the Phase 6 grouped-evidence, duplicate-claim, and content-plus-geometry suppression safety rails. Phase 9 may narrow owner-signature construction or ownership sweep rules, but should not globally loosen proximity/content guards in ways that reintroduce fragment false positives.
- **D-08:** The currently confirmed real detector loss is Part 9 truth index `42` (`⏥ .01`). Planning should treat the explained-by-match suppressor in `detect_added_characteristics()` as the primary root-cause path for that miss unless fresh reruns disprove it.

### Corpus Normalization and Alias Scope
- **D-09:** Requirement-format-only mismatches that block canonical claiming, such as `.635 / .615` versus `0.635 / 0.615`, should be solved through deterministic shared normalization in the evaluation path rather than fuzzy matching or part-specific exceptions.
- **D-10:** `↗` / `⌰` alias support is only pulled into Phase 9 if a remaining miss or the final evidence refresh still depends on those corpus-extracted glyph forms. If needed, implement aliases in shared normalization/comparison logic with direct tests; otherwise keep the documented deferral closed without widening scope.

### the agent's Discretion
- Exact plan decomposition and wave order across the failure families above.
- Whether refreshed post-fix evidence lives as updated Phase 7 algorithm-only fixtures, a new Phase 9 fixture snapshot set, or both, as long as benchmark provenance stays explicit and the frozen Phase 6 corpus assets remain distinguishable from current baselines.
- Exact regression-test placement across existing Phase 6/7 files versus a new Phase 9 test module.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope and milestone contract
- `.planning/ROADMAP.md` — Phase 9 goal, success criteria, and downstream dependency on Phase 10
- `.planning/REQUIREMENTS.md` — `ADD-01` active requirement and milestone-level constraints
- `.planning/PROJECT.md` — no-part-specific-hacks constraint, immutable ground-truth rule, maintainer-only scope

### Accepted baseline and prior evidence
- `.planning/phases/07-regression-tests-and-verification/07-04-ALGORITHM-ONLY-BASELINE.md` — authoritative algorithm-only miss set for parts 1-9
- `.planning/phases/07-regression-tests-and-verification/07-VERIFICATION.md` — benchmark provenance and parity contract
- `.planning/phases/07-regression-tests-and-verification/07-CONTEXT.md` — locked Phase 7 decisions about algorithm-only verification mode
- `.planning/phases/06-added-characteristic-detection-and-snippet-accuracy/06-CONTEXT.md` — Phase 6 decisions for grouped evidence, truth claiming, and exclusion behavior
- `.planning/phases/06-added-characteristic-detection-and-snippet-accuracy/06-05-SUMMARY.md` — targeted Part 8/9 close-out and the remaining unsolved miss set
- `.planning/phases/06-added-characteristic-detection-and-snippet-accuracy/06-VERIFICATION.md` — what Phase 6 verified versus what was deferred
- `.planning/debug/phase06-aggregate-added-gaps.md` — mixed-bucket diagnosis for the remaining corpus misses
- `.planning/debug/phase06-part9-flatness-miss.md` — confirmed Part 9 suppressor root cause for truth index 42
- `.planning/phases/04-gd-t-parser-fixes/04-VALIDATION.md` — explicit deferral of `↗` / `⌰` alias work to Phase 9 only if closure still needs it

### Core code paths
- `run.py` — authoritative standalone rerun entrypoint used by the accepted baseline
- `delta_preservation/cli.py` — packet assembly and evaluation wiring for added rows
- `delta_preservation/reconcile/classify.py` — added-detection passes, grouping, suppressor, and post-pass reconciliation
- `delta_preservation/evaluation/conformance.py` — requirement normalization, added truth selection, and ambiguity handling
- `shop/services/review.py` — current derivation of maintainer-facing `missing_added_truth_indexes`
- `delta_preservation/reconcile/normalize.py` — shared normalization entry point and any needed alias work
- `delta_preservation/reconcile/match.py` — matched-annotation ownership constraints and family hints used by added detection

### Existing regression coverage
- `tests/test_phase7_benchmark.py` — locked algorithm-only per-part baseline ceilings
- `tests/test_added_detection_phase6.py` — Phase 6 grouped-evidence and suppressor guardrails
- `tests/test_added_truth_selection.py` — duplicate added-truth claiming behavior
- `tests/test_phase6_asset_regression.py` — frozen Phase 6 corpus exemplars and historical asset contract
- `tests/test_debug_row_identity.py` — queue/debug identity and added-truth claiming coverage

### Baseline fixtures and canonical truth
- `tests/fixtures/phase7_algorithm_only/part1-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part2-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part3-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part4-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part5-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part6-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part7-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part8-debug-report.json`
- `tests/fixtures/phase7_algorithm_only/part9-debug-report.json`
- `assets/part1/ground_truth.json`
- `assets/part2/ground_truth.json`
- `assets/part3/ground_truth.json`
- `assets/part4/ground_truth.json`
- `assets/part5/ground_truth.json`
- `assets/part6/ground_truth.json`
- `assets/part7/ground_truth.json`
- `assets/part8/ground_truth.json`
- `assets/part9/ground_truth.json`

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `run.py` already defines the accepted standalone rerun path that Phase 7 used to lock the algorithm-only baseline.
- `delta_preservation/reconcile/classify.py::detect_added_characteristics()` already has the three-pass added detector, grouped added-evidence contract (`added_requirement_text`, `added_bbox`, `added_page`), and the explained-by-match suppressor that now needs narrower ownership construction.
- `delta_preservation/evaluation/conformance.py::select_truth_row_for_item()` already owns deterministic added-truth selection and spacing-tolerant normalization; it is the right place for additional requirement-format normalization that still stays deterministic.
- `shop/services/review.py::_load_missing_added_truth_items()` already computes maintainer-facing `missing_added_truth_indexes`; this is the current accounting path implicated by the Part 1 false miss.
- `tests/test_added_detection_phase6.py`, `tests/test_added_truth_selection.py`, and `tests/test_debug_row_identity.py` already encode the Phase 6 safety rails and can be extended for Phase 9 without inventing a new harness.

### Established Patterns
- Recent algorithm fixes stay in shared reconcile/evaluation paths and avoid per-part conditionals or `ground_truth.json` edits.
- Frozen Phase 6 asset snapshots are treated as historical regression evidence, while current benchmarking uses algorithm-only fixtures with explicit provenance.
- Added-truth claiming is intentionally conservative: exact normalized requirement first, then bbox containment / nearest-center tie-break, then ambiguity instead of guessing.
- False-positive suppression currently requires both semantic/text ownership and bbox ownership; Phase 9 should preserve that contract while narrowing how matched owner signatures are assembled.

### Integration Points
- `delta_preservation/cli.py` is the single place where rerun output becomes packet rows and evaluations; refreshed evidence must flow through this path.
- `tests/test_phase7_benchmark.py` consumes top-level `missing_added_truth_indexes`, so any Part 1 accounting fix must be reflected in the same derived signal the benchmark and maintainer surfaces use.
- `shop/services/review.py` consumes `evaluation.matched_truth_char_no` tokens from packet items; mismatches between evaluator tokens and missing-added accounting can surface false blockers even when packet evaluation is correct.
- If alias support is still needed, it must line up across normalization/comparison and any added-detection family hints so the same requirement form is recognized consistently through detection, claiming, and verification.

</code_context>

<specifics>
## Specific Ideas

- The accepted current algorithm-only miss set is: Part 1 `[38]`, Part 2 `[22]`, Part 3 `[19, 20]`, Part 4 `[11, 14, 15]`, Part 5 `[16, 17, 18]`, Part 9 `[42]`.
- Part 1's false miss is already diagnosed as a claim-accounting issue: the packet matches the canonical added row, but maintainer-facing missing-added accounting ignores the integer truth token because it only accepts `added:<index>` strings.
- Part 2's remaining miss is already diagnosed as a deterministic normalization/claiming issue: the packet emits `0.635 / 0.615` while truth expects `.635 / .615`.
- The confirmed Part 9 real miss is the added flatness row `⏥ .01` at truth index 42. The current diagnosis says `detect_added_characteristics()` suppresses it because the matched-owner signature for an existing characteristic sweeps in unrelated same-row unmatched spans within ±200 pt.
- Historical Phase 6 assets remain useful as frozen exemplars for regression coverage, but final success for this phase is measured against fresh algorithm-only reruns and refreshed evidence.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within the Phase 9 boundary. Phase 10 maintainer gating/advisory work and any UI/export blocking behavior remain separate.

</deferred>

---

*Phase: 09-full-corpus-added-characteristic-closure*
*Context gathered: 2026-04-18 (assumptions mode)*
