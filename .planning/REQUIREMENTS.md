# Requirements: v1.1 Cross-part Characteristic Matching Refinement

## Milestone Goal

Fix the concrete algorithm errors documented across 9 debug-corpus parts to improve aggregate
classification accuracy. Every fix must generalize — no per-part hacks. Ground truth files are
canonical and must never be edited by the pipeline.

## Active Requirements

### GD&T Parsing

- [x] **GDT-01**: Compact GD&T tokens (e.g. `⌖∅0.35ABC`, `⌓0.5A`, `⏥0.2`) written without
  spaces are correctly split into control symbol + diameter prefix + tolerance value + datum refs
  before semantic comparison, eliminating `gdt_malformed_frame` fallback for these inputs.

- [x] **GDT-02**: Spelled-out GD&T control names in Form 3 text ("circularity", "runout",
  "total runout", "position", "flatness", "perpendicularity") are normalized to their Unicode
  symbol equivalents before semantic extraction, so word-form and symbol-form of the same
  characteristic compare equal.

- [x] **GDT-03**: Composite / multi-compartment GD&T feature control frames (multiple FCFs
  listed for one characteristic) are captured in full rather than only the first compartment,
  so requirements like `⌓ .05 D B C / ⌓ .01 D` are not partially lost.

### Classification Logic

- [x] **CLS-01**: When a matched Rev B span contains a `/`-separated multi-balloon bleed
  (adjacent balloon content merged into one PDF text span), the `count_added` false-positive
  signal is suppressed and the item is tagged with a "Rev B text may contain adjacent balloon
  content" confidence flag instead of being mis-classified as changed.

- [x] **CLS-02**: When no Rev B match is found for a Rev A characteristic (candidate "removed")
  and an unmatched added characteristic exists at a spatially close location on the same page,
  the pair is resolved as a single "changed" characteristic rather than a separate
  removed + added pair.

- [x] **CLS-03**: A symmetric tolerance that changes to an asymmetric form (e.g. `±1°` →
  `+0.3° / −0.1°`) is correctly detected as a tolerance change and classified as "changed",
  not "unchanged" on the basis of a matching primary dimension.

### Added Characteristic Detection

- [x] **ADD-01
**: All ground-truth-added characteristics are present in the pipeline output for
  every part in the debug corpus. Fresh milestone verification still shows
  `missing_added_truth_indexes` on parts 1-5 and 9; after this fix those lists must be empty
  across the full corpus.

- [x] **ADD-02**: Spurious added-characteristic rows (false positives where a raw PDF span
  fragment is incorrectly classified as a new characteristic) are suppressed when the span's
  content is already fully explained by an existing matched characteristic.

### Snippet / Location

- [x] **SNP-01**: Title block regions (bottom-right corner, revision table) are reliably
  excluded from the search window for all characteristic types, eliminating the pattern of
  `snippet_outside_revA` mismatches where the algorithm latches onto title-block text instead
  of the correct drawing annotation.

## Regression Tests

- [ ] **TST-01**: Each fixed failure cluster has at least one pytest parametrized case that
  encodes the exact input (requirement strings, span text) and asserts the correct
  classification outcome, so regressions are caught without re-running the full pipeline.

- [ ] **TST-02**: A cross-part benchmark test re-runs the ground-truth evaluation for all 9
  parts (or validates the scoring logic against known counts) and fails if aggregate accuracy
  drops below the post-fix baseline.

## Verification

- [ ] **VER-01**: After all fixes, all 9 parts are re-run through the pipeline and their
  ground-truth evaluation results (conforming count, exception count, missing_added_truth_indexes)
  are compared against pre-fix baselines. Each part must show equal or better conforming
  count with no regressions on previously-passing characteristics.

## Future Requirements (deferred)

- Cross-part contradiction detection between accepted alternates
- Benchmark trend summaries across runs
- Overfitting warning before reviewer accepts a new alternate
- Composite GD&T with positional-zone modifiers (Part 9 multi-datum advanced FCFs)

## Out of Scope

- Automatic edits to `ground_truth.json` — canonical truth is manually curated
- Improving the PDF scanner / OCR engine (bleed is a scanner artifact; the classifier detects it but does not fix the scan)
- Multi-user or end-user-facing changes — this milestone is maintainer tooling only
- Changes to the review UI or admin workflow

## Traceability

| REQ-ID | Phase | Plan |
|--------|-------|------|
| GDT-01 | Phase 8 | 08-01-PLAN.md, 08-02-PLAN.md |
| GDT-02 | Phase 8 | 08-01-PLAN.md, 08-02-PLAN.md |
| GDT-03 | Phase 8 | 08-01-PLAN.md, 08-02-PLAN.md |
| CLS-01 | Phase 5 | — |
| CLS-02 | Phase 5 | — |
| CLS-03 | Phase 5 | — |
| ADD-01 | Phase 9 | — |
| ADD-02 | Phase 6 | — |
| SNP-01 | Phase 6 | — |
| TST-01 | Phase 7 | — |
| TST-02 | Phase 7 | — |
| VER-01 | Phase 7 | — |

Audit gap-closure phases 10 and 11 address cross-phase integration and live web-flow gaps
without reassigning already-validated algorithm requirements.
