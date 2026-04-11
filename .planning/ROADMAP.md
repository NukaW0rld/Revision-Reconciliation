# Roadmap: Delta Preservation

## Overview

This milestone turns the existing manual debug loop into a deterministic evaluation workflow for benchmark runs. The path is: build reliable ground-truth comparison first, then narrow the debug UI to only the rows worth inspecting, then persist accepted alternates in a separate history layer so future runs can reuse those decisions without mutating canonical truth.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Evaluation Foundation** - Validate ground truth and auto-score completed runs against canonical truth
- [x] **Phase 2: Focused Debug Workflow** - Show only mismatches/ambiguity for manual review and enrich `debug_report.json` (completed 2026-04-11)
- [ ] **Phase 3: Exceptions History Layer** - Persist accepted alternate outcomes separately from `ground_truth.json`

## Phase Details

### Phase 1: Evaluation Foundation
**Goal**: Completed runs can be deterministically compared against immutable part-level ground truth, including tolerant snippet checks that reflect visually acceptable evidence.
**Depends on**: Nothing (first phase)
**Requirements**: [GTRU-01, GTRU-02, GTRU-03, EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05]
**Success Criteria** (what must be TRUE):
  1. A completed run loads the correct `ground_truth.json` and fails clearly when truth data is missing or malformed.
  2. Each review item is automatically classified as conforming or review-needed based on canonical classification, Rev B requirement, and snippet-tolerance evaluation.
  3. Auto-pass decisions do not mutate `ground_truth.json` or inject part-specific logic into the classifier.
  4. Nonconforming rows include explicit mismatch reasons that downstream review/export can consume.
**Plans**: 3 plans

Plans:
- [x] 01-01: Define and validate ground-truth and evaluation contracts
- [x] 01-02: Implement automatic conformance evaluation for classification and requirement matching
- [x] 01-03: Implement tolerant snippet acceptance rules and structured mismatch reasons

### Phase 2: Focused Debug Workflow
**Goal**: The debug surface and exported report prioritize only nonconforming or ambiguous rows while still preserving visibility into auto-passed results.
**Depends on**: Phase 1
**Requirements**: [DREV-01, DREV-02, DREV-03, DREV-04, RPT-01, RPT-02, RPT-03]
**Success Criteria** (what must be TRUE):
  1. Admin debug review defaults to an exception-only queue for a run instead of requiring verdict entry for every row.
  2. Auto-passed rows remain visible in run details or exported report without cluttering the main manual-review queue.
  3. `debug_report.json` is generated from evaluation output and distinguishes canonical matches, acceptable alternate matches, and unresolved review-needed rows.
  4. Review-needed rows preserve actionable mismatch reasons and reviewer rationale.
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — Rework debug queue state around exception-only review
- [x] 02-02-PLAN.md — Extend review/export payloads to represent auto-pass and review-needed outcomes
- [x] 02-03-PLAN.md — Update admin debug UX for focused review and rationale capture

### Phase 3: Exceptions History Layer
**Goal**: Accepted alternate outcomes are stored in a separate history layer and can be reused for later runs of the same part and characteristic without changing canonical truth.
**Depends on**: Phase 2
**Requirements**: [HIST-01, HIST-02, HIST-03]
**Success Criteria** (what must be TRUE):
  1. A reviewer can record an acceptable alternate outcome for a nonconforming characteristic without editing `ground_truth.json`.
  2. Each stored alternate outcome includes run, part, characteristic, reviewed result, and rationale fields.
  3. A later run for the same part and characteristic can treat that approved alternate as conforming and reflect it in evaluation/report output.
  4. The history layer remains separate from future contradiction-analysis work, which is still deferred.
**Plans**: 2 plans

Plans:
- [ ] 03-01: Design and persist the exceptions/history model
- [ ] 03-02: Integrate history-backed acceptable alternates into evaluation and reports

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Evaluation Foundation | 3/3 | Complete | 2026-04-10 |
| 2. Focused Debug Workflow | 3/3 | Complete    | 2026-04-11 |
| 3. Exceptions History Layer | 0/2 | Not started | - |
