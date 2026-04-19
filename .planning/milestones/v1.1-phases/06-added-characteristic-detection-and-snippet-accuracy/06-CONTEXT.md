# Phase 6: Added Characteristic Detection and Snippet Accuracy - Context

**Gathered:** 2026-04-16 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix the shared pipeline path so all canonical added characteristics are emitted and claimable in packet output, fragment-level false-positive added rows are suppressed, and title-block or revision-table content is excluded consistently from search windows and snippet targeting. This phase is limited to reconcile, packet-generation, and evaluation-layer behavior that affects canonical truth claiming. No web UI changes, no review-queue workaround, and no edits to `ground_truth.json`.

</domain>

<decisions>
## Implementation Decisions

### Pipeline Ownership
- **D-01:** Phase 6 is implemented in the shared pipeline path, centered on `delta_preservation/reconcile/classify.py`, `delta_preservation/cli.py`, and the evaluation path that claims added truth rows. `shop/services/review.py` remains a consumer of packet/evaluation results, not the place where Phase 6 correctness is fixed.

### Added Detection and False-Positive Suppression
- **D-02:** Added-characteristic detection must reason about grouped annotations, not only raw unmatched seed spans. When a new added row is emitted, the grouped annotation text and grouped bbox become the canonical evidence payload for that row.
- **D-03:** Suppression of false-positive added rows must use an "already explained by an existing matched characteristic" rule, not a blunt proximity-only rule. Nearby unmatched spans are suppressed only when a matched/grouped characteristic already fully explains their content.
- **D-04:** Fragment-like outputs such as Part 8's `.005` / `.045 A` and Part 9's truncated `⌖∅` are treated as Phase 6 regressions. A valid fix must preserve full annotation ownership from detection through packet generation.

### Title-Block and Boilerplate Exclusion
- **D-05:** Title-block, revision-table, and boilerplate exclusion uses one shared contract across anchor building, candidate generation, rescue scans, and added detection. Phase 6 must remove the current situation where each stage approximates these regions differently.
- **D-06:** The shared exclusion contract must combine geometric corner zones with boilerplate-text rejection, and all callers must estimate page dimensions the same way before applying the exclusion rules.

### Added Truth Claiming
- **D-07:** Phase 6 must make duplicate canonical added rows claimable without per-part hacks. Normalized requirement text alone is not sufficient for Part 9 because multiple truth rows share the same requirement text.
- **D-08:** Deterministic claiming of duplicate added truth rows should use richer grouped evidence already produced by the pipeline, such as grouped requirement text, grouped bbox, snippet center, or comparable location evidence. It must not rely on manual mappings or truth-file edits.

### the agent's Discretion
- Whether the shared exclusion contract lives in `delta_preservation/reconcile/match.py` as an expanded helper or is moved to a new shared reconcile utility module.
- Whether duplicate added-truth disambiguation is solved by preserving more packet evidence, by extending evaluator selection logic, or by a combination of both, as long as it stays deterministic and general across the corpus.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope and requirements
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/PROJECT.md`

### Failing debug-corpus evidence
- `assets/debug_report_part8.json`
- `assets/debug_report_part9.json`
- `assets/part8/ground_truth.json`
- `assets/part9/ground_truth.json`

### Core pipeline code
- `delta_preservation/reconcile/classify.py`
- `delta_preservation/reconcile/match.py`
- `delta_preservation/reconcile/anchors.py`
- `delta_preservation/cli.py`

### Added-truth evaluation path
- `delta_preservation/evaluation/conformance.py`
- `delta_preservation/evaluation/snippet_rules.py`

### Existing regression coverage
- `tests/test_classify_bugfixes.py`
- `tests/test_debug_row_identity.py`

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `delta_preservation/reconcile/classify.py::detect_added_characteristics` already owns the three-pass added detector: GD&T grouping, stacked tolerance-pair detection, and standard span-by-span detection.
- `delta_preservation/reconcile/classify.py::DeltaItem` already carries internal-only Phase 5 style metadata fields (`added_requirement_text`, `added_bbox`, `added_page`) that Phase 6 can reuse instead of inventing a new side channel.
- `delta_preservation/reconcile/match.py::_span_is_excluded_for_matching` already combines corner-region exclusion with boilerplate-text rejection and is the strongest existing exclusion helper in the repo.
- `delta_preservation/reconcile/match.py::_group_candidate_spans` already groups companion spans into a fuller callout for matching; Phase 6 can reuse that pattern for added detection instead of treating every unmatched span independently.
- `delta_preservation/cli.py` already expands matched annotations into grouped `requirement_revB` text and snippet evidence. The added-row path can be brought up to the same contract.

### Established Patterns
- Recent algorithm fixes stay inside shared reconcile paths and use post-pass reconciliation or richer internal metadata instead of UI-layer workarounds.
- Packet rows are the contract the debug workflow consumes. If a pipeline row cannot claim canonical truth, the review layer correctly reports `missing_added_truth_indexes` rather than silently repairing it.
- The codebase already treats grouped callouts as first-class for matched characteristics, but added characteristics still fall back too easily to a single seed span.
- Multiple files currently duplicate title-block and revision-table heuristics with different thresholds, so a localized fix in one file will not satisfy SNP-01 by itself.

### Integration Points
- `delta_preservation/cli.py` calls `detect_added_characteristics(...)`, then `reconcile_removed_added_pairs(...)`, then converts internal `DeltaItem` objects into packet `DeltaItem` models and runs truth evaluation.
- `delta_preservation/evaluation/conformance.py` decides whether an added packet row can claim a canonical added truth row; duplicate normalized requirement text in Part 9 makes this a real constraint for Phase 6.
- `shop/services/review.py` surfaces whatever `missing_added_truth_indexes` the evaluator leaves unresolved; it should not need new workaround logic if Phase 6 is fixed correctly upstream.

</code_context>

<specifics>
## Specific Ideas

- Part 8 currently misses canonical added truth row 10 (`⌰ .002 A`) while also emitting fragment-like added rows such as `.005` and `.045 A`. Phase 6 must fix both sides of that failure, not just the missing row count.
- Part 9 currently has eight canonical added rows in ground truth but only three pipeline added rows in the saved packet, and one of those rows collapses a full GD&T callout to `⌖∅`. That is evidence-contract loss, not just detector under-recall.
- Duplicate added requirements in Part 9 such as repeated `Ø.250 ±.008` mean the evaluator cannot rely on unique normalized text matches alone when claiming truth rows.

</specifics>

<deferred>
## Deferred Ideas

None — analysis stayed within phase scope.

</deferred>

---

*Phase: 06-added-characteristic-detection-and-snippet-accuracy*
*Context gathered: 2026-04-16 (assumptions mode)*
