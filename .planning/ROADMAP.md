# Roadmap: Delta Preservation

## Milestones

- ✅ **v1.0 Ground Truth Debug Workflow** — Phases 1-3, shipped 2026-04-12 ([roadmap archive](milestones/v1.0-ROADMAP.md), [requirements archive](milestones/v1.0-REQUIREMENTS.md), [audit](milestones/v1.0-MILESTONE-AUDIT.md))
- 🚧 **v1.1 Cross-part Characteristic Matching Refinement** — Phases 4-7 (in progress)

## Phases

<details>
<summary>✅ v1.0 Ground Truth Debug Workflow (Phases 1-3) — SHIPPED 2026-04-12</summary>

See [milestones/v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md) for full phase details.

</details>

### 🚧 v1.1 Cross-part Characteristic Matching Refinement (In Progress)

**Milestone Goal:** Fix the concrete algorithm errors discovered across all 9 debug parts to improve aggregate classification accuracy. Every fix must generalize across all parts — no per-part hacks.

- [ ] **Phase 4: GD&T Parser Fixes** - Correct compact token splitting, word-name normalization, and composite frame capture
- [ ] **Phase 5: Classification Logic Fixes** - Suppress adjacency bleed false positives, resolve removed+added pairs as changed, detect asymmetric tolerance changes
- [ ] **Phase 6: Added Characteristic Detection and Snippet Accuracy** - Fix missing added rows for parts 8/9, suppress false-positive added rows, exclude title block regions from search windows
- [ ] **Phase 7: Regression Tests and Verification** - Add parametrized regression tests per fix cluster, cross-part benchmark, and full 9-part verification run

## Phase Details

### Phase 4: GD&T Parser Fixes
**Goal**: The GD&T parser correctly handles all token forms found in the 9-part debug corpus — compact concatenated frames, spelled-out word names, and composite multi-compartment FCFs — so malformed-frame fallback no longer fires on valid inputs.
**Depends on**: Phase 3 (v1.0)
**Requirements**: GDT-01, GDT-02, GDT-03
**Success Criteria** (what must be TRUE):
  1. A compact token like `⌖∅0.35ABC` is split into symbol + diameter prefix + tolerance + datum refs with no `gdt_malformed_frame` flag emitted.
  2. A Form 3 text entry using "circularity", "runout", or "total runout" (any supported word name) is recognized as equivalent to its Unicode symbol form and compares equal to a symbol-form drawing annotation.
  3. A composite FCF string like `⌓ .05 D B C / ⌓ .01 D` produces two captured compartments, not one, and neither compartment is silently dropped.
  4. Previously failing GD&T parse cases in the debug corpus now produce structured parse results, reducing exception-queue entries attributable to parser fallback.
**Plans**: TBD

### Phase 5: Classification Logic Fixes
**Goal**: The classifier correctly handles three known error patterns — adjacency bleed false positives, close-proximity removed/added pairs that should be a single changed item, and symmetric-to-asymmetric tolerance changes — without mis-classifying any of them.
**Depends on**: Phase 4
**Requirements**: CLS-01, CLS-02, CLS-03
**Success Criteria** (what must be TRUE):
  1. A Rev B text span containing a `/`-merged multi-balloon bleed does not produce a `count_added` false-positive; instead the item carries a "Rev B text may contain adjacent balloon content" confidence flag and is not classified as changed on that basis alone.
  2. A removed Rev A characteristic with an unmatched added characteristic at spatially close proximity on the same page is emitted as a single "changed" row, not a separate removed + added pair.
  3. A tolerance change from `±1°` to `+0.3° / −0.1°` (or any symmetric → asymmetric form) is classified as "changed", not "unchanged".
  4. No previously-passing characteristics across the 9-part corpus regress to a wrong classification after these fixes are applied.
**Plans**: 5 plans
  - [x] 05-01-PLAN.md — Wave 0 scaffold: confidence_flags on both DeltaItem models, CLI getattr fallback for legacy test doubles, and inline compatibility tests
  - [x] 05-02-PLAN.md — CLS-01 adjacency-bleed suppressor with anchor-aware chunk verification and explicit slash-separated negative cases
  - [x] 05-03-PLAN.md — CLS-03 kind-transition pre-check plus broadened asymmetric fallback detection for leading-decimal forms
  - [ ] 05-04-PLAN.md — CLS-02 removed+added reconciliation post-pass with explicit added metadata contract (text, bbox, page) and nearest-compatible merge rules
  - [ ] 05-05-PLAN.md — Phase-5 regression harness using explicit snapshot exemplars plus synthetic CLS-02 packet-level regression coverage

### Phase 6: Added Characteristic Detection and Snippet Accuracy
**Goal**: All ground-truth-added characteristics are present in pipeline output for every part, false-positive added rows are suppressed, and title block regions are reliably excluded from search windows so snippet matches land on actual drawing annotations.
**Depends on**: Phase 5
**Requirements**: ADD-01, ADD-02, SNP-01
**Success Criteria** (what must be TRUE):
  1. After a clean pipeline run, `missing_added_truth_indexes` is empty for parts 8 and 9 (the two currently failing parts).
  2. A raw PDF span whose content is fully explained by an existing matched characteristic does not produce a spurious added-characteristic row.
  3. No characteristic search window captures text from the title block or revision table; any previously-failing `snippet_outside_revA` cases caused by title-block capture are resolved.
  4. The aggregate added-characteristic count across all 9 parts matches or exceeds the ground-truth added count with no false-positive increase.
**Plans**: TBD

### Phase 7: Regression Tests and Verification
**Goal**: Every fix cluster from Phases 4-6 is covered by at least one fast parametrized pytest case, a cross-part benchmark guards aggregate accuracy, and a full 9-part ground-truth re-run confirms no regressions against the pre-fix baseline.
**Depends on**: Phase 6
**Requirements**: TST-01, TST-02, VER-01
**Success Criteria** (what must be TRUE):
  1. Each fix cluster (GD&T parsing, adjacency bleed, removed+added resolution, asymmetric tolerance, missing added rows, title block exclusion) has at least one parametrized pytest case that fails on the unfixed code and passes on the fixed code.
  2. A cross-part benchmark test validates scoring logic against known post-fix counts and fails the test suite if aggregate accuracy drops below the established baseline.
  3. All 9 parts are re-run through the pipeline and their ground-truth evaluation results show equal or better conforming count versus the pre-fix baseline, with zero regressions on previously-passing characteristics.
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Ground Truth Fixtures | v1.0 | 3/3 | Complete | 2026-04-12 |
| 2. Debug Queue & Export | v1.0 | 3/3 | Complete | 2026-04-12 |
| 3. Accepted Alternates | v1.0 | 2/2 | Complete | 2026-04-12 |
| 4. GD&T Parser Fixes | v1.1 | 0/TBD | Not started | - |
| 5. Classification Logic Fixes | v1.1 | 1/5 | In Progress|  |
| 6. Added Characteristic Detection and Snippet Accuracy | v1.1 | 0/TBD | Not started | - |
| 7. Regression Tests and Verification | v1.1 | 0/TBD | Not started | - |
