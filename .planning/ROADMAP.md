# Roadmap: Delta Preservation

## Milestones

- ✅ **v1.0 Ground Truth Debug Workflow** — Phases 1-3, shipped 2026-04-12 ([roadmap archive](milestones/v1.0-ROADMAP.md), [requirements archive](milestones/v1.0-REQUIREMENTS.md), [audit](milestones/v1.0-MILESTONE-AUDIT.md))
- 🚧 **v1.1 Cross-part Characteristic Matching Refinement** — Phases 4-11 (in progress)

## Phases

<details>
<summary>✅ v1.0 Ground Truth Debug Workflow (Phases 1-3) — SHIPPED 2026-04-12</summary>

See [milestones/v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md) for full phase details.

</details>

### 🚧 v1.1 Cross-part Characteristic Matching Refinement (In Progress)

**Milestone Goal:** Fix the concrete algorithm errors discovered across all 9 debug parts to improve aggregate classification accuracy. Every fix must generalize across all parts — no per-part hacks.

- [x] **Phase 4: GD&T Parser Fixes** - Correct compact token splitting, word-name normalization, and composite frame capture (completed 2026-04-17)
- [x] **Phase 5: Classification Logic Fixes** - Suppress adjacency bleed false positives, resolve removed+added pairs as changed, detect asymmetric tolerance changes (completed 2026-04-17)
- [x] **Phase 6: Added Characteristic Detection and Snippet Accuracy** - Fix missing added rows for parts 8/9, suppress false-positive added rows, exclude title block regions from search windows (completed 2026-04-17)
- [x] **Phase 7: Regression Tests and Verification** - Add parametrized regression tests per fix cluster, cross-part benchmark, and full 9-part verification run (completed 2026-04-17)
- [x] **Phase 8: GD&T Verification Recovery** - Rebuild the missing Phase 04 verification chain and restore GD&T requirement traceability (completed 2026-04-17)
- [x] **Phase 9: Full-Corpus Added Characteristic Closure** - Eliminate the remaining missing added truth rows across the 9-part corpus and refresh the supporting evidence (completed 2026-04-18)
- [x] **Phase 10: Debug Exception Gating and Advisory Surfacing** - Block sign-off/export on unresolved debug exceptions and surface classifier advisory flags in maintainer-facing surfaces (completed 2026-04-19)
- [x] **Phase 11: Web Run-to-Review E2E Automation** - Cover the live `/runs/new` through review/debug/export path with automated integration tests (completed 2026-04-19)
- [x] **Phase 12: Process Artifact Closure and Metadata Reconciliation** - Create missing VERIFICATION.md files, fix stale checkboxes/statuses, reconcile Nyquist VALIDATION.md flags to pass milestone audit (completed 2026-04-19)

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
**Plans**: 2 plans
  - [x] 04-01-PLAN.md — Word-form normalization and compact-token parsing with regression coverage
  - [x] 04-02-PLAN.md — Composite GD&T frame capture with compartment-aware comparison

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
  - [x] 05-04-PLAN.md — CLS-02 removed+added reconciliation post-pass with explicit added metadata contract (text, bbox, page) and nearest-compatible merge rules
  - [x] 05-05-PLAN.md — Phase-5 regression harness using explicit snapshot exemplars plus synthetic CLS-02 packet-level regression coverage

### Phase 6: Added Characteristic Detection and Snippet Accuracy
**Goal**: All ground-truth-added characteristics are present in pipeline output for every part, false-positive added rows are suppressed, and title block regions are reliably excluded from search windows so snippet matches land on actual drawing annotations.
**Depends on**: Phase 5
**Requirements**: ADD-01, ADD-02, SNP-01
**Success Criteria** (what must be TRUE):
  1. After a clean pipeline run, `missing_added_truth_indexes` is empty for parts 8 and 9 (the two currently failing parts).
  2. A raw PDF span whose content is fully explained by an existing matched characteristic does not produce a spurious added-characteristic row.
  3. No characteristic search window captures text from the title block or revision table; any previously-failing `snippet_outside_revA` cases caused by title-block capture are resolved.
  4. The aggregate added-characteristic count across all 9 parts matches or exceeds the ground-truth added count with no false-positive increase.
**Plans**: 4 plans
  - [x] 06-01-PLAN.md — Wave 1 shared exclusion contract across anchors, matching, rescue scans, and added detection
  - [x] 06-02-PLAN.md — Wave 2 grouped added-evidence contract plus explained-by-match false-positive suppression
  - [x] 06-03-PLAN.md — Wave 3 deterministic duplicate added-truth claiming from packet Rev B evidence
  - [x] 06-04-PLAN.md — Wave 4 read-only Part 8 / Part 9 asset-backed regression harness

### Phase 7: Regression Tests and Verification
**Goal**: Every fix cluster from Phases 4-6 is covered by at least one fast parametrized pytest case, a cross-part benchmark guards aggregate accuracy, and a full 9-part ground-truth re-run confirms no regressions against the pre-fix baseline.
**Depends on**: Phase 6
**Requirements**: TST-01, TST-02, VER-01
**Success Criteria** (what must be TRUE):
  1. Each fix cluster (GD&T parsing, adjacency bleed, removed+added resolution, asymmetric tolerance, missing added rows, title block exclusion) has at least one parametrized pytest case that fails on the unfixed code and passes on the fixed code.
  2. A cross-part benchmark test validates scoring logic against known post-fix counts and fails the test suite if aggregate accuracy drops below the established baseline.
  3. All 9 parts are re-run through the pipeline and their ground-truth evaluation results show equal or better conforming count versus the pre-fix baseline, with zero regressions on previously-passing characteristics.
**Plans**: 4 plans
  - [x] 07-01-PLAN.md — Per-cluster audit and gap identification
  - [x] 07-02-PLAN.md — TST-02 benchmark baseline derivation
  - [x] 07-03-PLAN.md — VER-01 full 9-part ground-truth re-run
  - [x] 07-04-PLAN.md — Algorithm-only baseline fixtures and parity correction

### Phase 8: GD&T Verification Recovery
**Goal**: Close the Phase 04 audit gap by restoring the missing verification chain, reconciling any stale manual checks, and re-proving the implemented GD&T fixes against current evidence so the orphaned GD&T requirements become satisfied again.
**Depends on**: Phase 7
**Requirements**: GDT-01, GDT-02, GDT-03
**Gap Closure**: FLOW-03
**Success Criteria** (what must be TRUE):
  1. Phase 04 has a substantive `04-VERIFICATION.md` that verifies the shipped GD&T parser behavior against the current codebase and evidence, not just plan summaries.
  2. `GDT-01`, `GDT-02`, and `GDT-03` are no longer orphaned in milestone verification and point to current verification evidence.
  3. Any stale manual-only validation checks or residual parser debt called out in the audit are either closed with evidence or explicitly re-scoped.

### Phase 9: Full-Corpus Added Characteristic Closure
**Goal**: Finish the added-characteristic work by eliminating the remaining missing truth rows across the corpus and refreshing the Phase 06/07 evidence so the final algorithm state matches milestone claims.
**Depends on**: Phase 7
**Requirements**: ADD-01
**Affected Requirements**: ADD-02, SNP-01
**Gap Closure**: FLOW-01
**Success Criteria** (what must be TRUE):
  1. Fresh evaluation across all 9 debug parts yields empty `missing_added_truth_indexes`, including the current misses on parts 1-5 and 9.
  2. Added-characteristic output does not introduce new false positives relative to the current Phase 06/07 baseline while closing the missing-row gap.
  3. Phase 06 verification/validation artifacts and any downstream milestone evidence are refreshed to reflect the final full-corpus result.

### Phase 10: Debug Exception Gating and Advisory Surfacing
**Goal**: Make unresolved debug exceptions and classifier advisory flags visible and enforceable at sign-off/export time so maintainer decisions happen on the same evidence the packet records.
**Depends on**: Phase 9
**Requirements**: — (audit integration closure only)
**Affected Requirements**: CLS-01, ADD-01, ADD-02, SNP-01, VER-01
**Gap Closure**: GAP-01, GAP-02, FLOW-02
**Success Criteria** (what must be TRUE):
  1. Normal maintainer debug/review surfaces render classifier `confidence_flags` alongside the affected items.
  2. Sign-off and audit/work-order export routes block completion while unresolved debug exceptions remain, unless an explicit acknowledgement path is provided and recorded.
  3. Exported artifacts preserve the same advisory and gating state the maintainer saw during review.

### Phase 11: Web Run-to-Review E2E Automation
**Goal**: Prove the live web workflow from run submission through review/debug/export with automated coverage, not just algorithm-only fixtures.
**Depends on**: Phase 10
**Requirements**: — (audit flow closure only)
**Affected Requirements**: TST-02, VER-01
**Gap Closure**: GAP-03
**Success Criteria** (what must be TRUE):
  1. An automated web integration test covers `/runs/new` submission, background task completion, `delta_packet.json` persistence, and review surface load for a representative fixture-backed run.
  2. The same test coverage asserts the debug/export/sign-off behavior introduced by earlier phases, including blockers and surfaced advisory state where applicable.
  3. Milestone verification no longer relies solely on algorithm-only artifacts to prove the live maintainer workflow.
**Plans**: 2 plans
  - [x] 11-01-PLAN.md — Dedicated live `/runs/new` -> packet -> review/debug -> blocked sign-off proof using an isolated real corpus run
  - [x] 11-02-PLAN.md — Cleared live sign-off/export reachability plus a companion seeded advisory case kept inside the Phase 11 artifact

### Phase 12: Process Artifact Closure and Metadata Reconciliation
**Goal**: Close all process-artifact gaps flagged by the v1.1 milestone audit — create missing VERIFICATION.md files for Phases 8 and 11, fix stale metadata in REQUIREMENTS.md and phase VERIFICATION artifacts, update ROADMAP progress tracking, and reconcile Nyquist VALIDATION.md flags so the milestone re-audit passes cleanly.
**Depends on**: Phase 11
**Requirements**: — (audit process closure only)
**Affected Requirements**: TST-01 (stale checkbox), GDT-01/02/03 (Phase 8 verification), ADD-01/ADD-02/SNP-01 (Phase 6 verification status)
**Gap Closure**: Closes all gaps from v1.1-MILESTONE-AUDIT.md
**Success Criteria** (what must be TRUE):
  1. `08-VERIFICATION.md` exists and documents the already-complete Phase 8 work with evidence references (summaries, commits, 04-VERIFICATION.md creation).
  2. `11-VERIFICATION.md` exists and documents the already-complete Phase 11 work with evidence references (summaries, commits, test file).
  3. ROADMAP Phase 11 progress table shows `2/2 | Complete` and plan checkboxes are `[x]`.
  4. REQUIREMENTS.md TST-01 checkbox is `[x]`.
  5. Phase 6 `06-VERIFICATION.md` status is updated from `human_needed` to `passed`.
  6. Nyquist VALIDATION.md files for Phases 6, 9, and 10 show `nyquist_compliant: true`; Phase 7 has a VALIDATION.md.
  7. A re-run of `/gsd-audit-milestone` returns `passed` status with 0 gaps.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Ground Truth Fixtures | v1.0 | 3/3 | Complete | 2026-04-12 |
| 2. Debug Queue & Export | v1.0 | 3/3 | Complete | 2026-04-12 |
| 3. Accepted Alternates | v1.0 | 2/2 | Complete | 2026-04-12 |
| 4. GD&T Parser Fixes | v1.1 | 2/2 | Complete | 2026-04-17 |
| 5. Classification Logic Fixes | v1.1 | 5/5 | Complete   | 2026-04-17 |
| 6. Added Characteristic Detection and Snippet Accuracy | v1.1 | 4/4 | Complete   | 2026-04-17 |
| 7. Regression Tests and Verification | v1.1 | 4/4 | Complete | 2026-04-17 |
| 8. GD&T Verification Recovery | v1.1 | 2/2 | Complete | 2026-04-17 |
| 9. Full-Corpus Added Characteristic Closure | v1.1 | 6/6 | Complete | 2026-04-18 |
| 10. Debug Exception Gating and Advisory Surfacing | v1.1 | 3/3 | Complete    | 2026-04-19 |
| 11. Web Run-to-Review E2E Automation | v1.1 | 2/2 | Complete | 2026-04-19 |
| 12. Process Artifact Closure and Metadata Reconciliation | v1.1 | 4/4 | Complete | 2026-04-19 |
