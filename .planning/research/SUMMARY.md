# Project Research Summary

**Project:** Delta Preservation
**Domain:** Ground-truth-assisted debug infrastructure for reconciliation accuracy
**Researched:** 2026-04-09
**Confidence:** HIGH

## Executive Summary

This project is not adding a new external product surface. It is adding an internal debug-and-evaluation layer to an existing drawing-reconciliation system so repeated benchmark runs can be judged quickly, consistently, and without rewriting the canonical truth for each part. The recommended approach is to keep the current Python/FastAPI/SQLAlchemy stack, add a deterministic post-run evaluator against `ground_truth.json`, auto-pass conforming rows, and preserve only disagreements in a focused review flow plus a separate exception/history layer.

The main architectural risk is not implementation complexity by itself, but overfitting: if benchmark-specific fixes or accepted alternates leak into the classifier or truth files, the algorithm may look excellent on one part while getting weaker across the broader corpus. The research therefore strongly favors immutable ground truth, append-only alternate-outcome history, and contradiction detection across runs and parts.

## Key Findings

### Recommended Stack

The existing stack is already the right fit for this milestone. Python remains the best place for evaluation logic because the pipeline, artifact generation, and domain heuristics already live there. FastAPI/Jinja2 remain the right debug surface because the current review flow already supports debug verdicts and report export. SQLAlchemy is the right persistence boundary for structured exception/history records once the milestone moves beyond a file-only spike.

**Core technologies:**
- Python: evaluation and report-generation logic — stays close to the reconciliation domain
- FastAPI + Jinja2: focused exception review surface — extends the existing debug UI instead of replacing it
- SQLAlchemy: exception/history persistence and contradiction analysis inputs — avoids burying important judgments in ad hoc notes

### Expected Features

The must-have feature set is tightly focused on reducing manual review while preserving judgment where ambiguity exists.

**Must have (table stakes):**
- Ground truth loading and validation — the benchmark is only trustworthy if fixture contracts are strict
- Automatic conformance evaluation — conforming rows should be auto-marked correct
- Exception-only manual review — human attention should go only to nonconforming or ambiguous rows
- Deterministic `debug_report.json` export — every run needs a portable artifact of passes, mismatches, and rationale
- Separate exceptions/history persistence — alternate acceptable outcomes must not rewrite truth

**Should have (competitive):**
- Cross-part contradiction detection — protects against part-specific rules and inconsistent review decisions
- Tolerant snippet acceptance — reduces false mismatches when the right annotation/context is still visible

**Defer (v2+):**
- Automatic learning or self-tuning from review history — explicitly out of scope for this milestone

### Architecture Approach

The recommended architecture adds a post-run debug evaluation layer between existing pipeline output and the current debug UI/export flow. That layer loads immutable truth, evaluates each row, records auto-passes and review-needed rows, and writes structured exception/history records for accepted alternates. Contradiction analysis should consume that history later, not mutate the core reconciliation algorithm directly.

**Major components:**
1. Ground truth catalog — validates immutable part-level truth files
2. Conformance evaluator — compares packet rows to truth using classification, requirement, and snippet-tolerance rules
3. Exception/history store — records accepted alternates and rationale separately from truth
4. Focused review/export surface — shows only mismatches/ambiguity and emits final debug reports

### Critical Pitfalls

1. **Benchmark overfitting** — keep evaluator/history logic separate from the classifier and scan for contradictions across parts
2. **Ground truth drift** — never auto-edit `ground_truth.json`; preserve canonical truth in version control
3. **Brittle snippet matching** — use tolerant visibility rules instead of exact coordinate equality
4. **Manual review still too broad** — exceptions-only review is essential or the milestone fails its speed goal
5. **Unstructured contradiction history** — alternate acceptable outcomes must be queryable, not just narrative notes

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Evaluation Foundation
**Rationale:** Everything else depends on reliable truth loading and deterministic auto-evaluation.
**Delivers:** Ground truth schema validation, conformance evaluator, snippet tolerance rules
**Addresses:** Automatic conformance evaluation, tolerant snippet acceptance
**Avoids:** Ground truth drift, brittle snippet matching, benchmark overfitting

### Phase 2: Focused Debug Workflow
**Rationale:** Once rows can be auto-scored, the debug surface should stop making the developer review every item.
**Delivers:** Exception-only queue behavior, enriched mismatch reasons, updated `debug_report.json`
**Uses:** Existing FastAPI/Jinja debug surfaces
**Implements:** Focused review/export component

### Phase 3: Exceptions and Alternate Outcomes
**Rationale:** Acceptable alternates need structured persistence before contradiction analysis is useful.
**Delivers:** Separate exception/history layer with normalized accepted-alternate records
**Uses:** SQLAlchemy or a transitional append-only artifact format
**Implements:** Exception/history component

### Phase 4: Cross-Part Consistency Checks
**Rationale:** The core value is consistency across many parts, not just fewer mismatches on one fixture.
**Delivers:** Contradiction detection and cross-run consistency signals
**Uses:** Structured history records from Phase 3
**Implements:** Contradiction-analysis component

### Phase Ordering Rationale

- Truth validation and auto-evaluation come first because every later workflow depends on trustworthy pass/fail signals.
- Exception-only review comes before contradiction analysis because the review surface must produce structured signals worth analyzing.
- Contradiction detection comes last because it depends on enough alternate-outcome history to reveal conflicts across parts.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** snippet tolerance heuristics may need deeper exploration against real fixture artifacts
- **Phase 4:** contradiction detection rules need careful normalization so they catch genuine conflicts instead of harmless variation

Phases with standard patterns (skip research-phase):
- **Phase 2:** focused UI/report integration is a straightforward extension of the existing debug workflow
- **Phase 3:** append-only history persistence follows standard schema-and-audit patterns once field design is clear

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Brownfield recommendation is strongly anchored in the current working system |
| Features | HIGH | Derived directly from the developer's stated workflow pain and milestone goals |
| Architecture | HIGH | Extends known existing boundaries instead of introducing speculative new services |
| Pitfalls | HIGH | The major risks were explicitly discussed during initialization and are reinforced by the current debug flow |

**Overall confidence:** HIGH

### Gaps to Address

- Snippet tolerance needs a concrete acceptance rule during planning/execution, not just the principle that visible context matters
- The exceptions/history layer needs a concrete storage choice during planning: transitional file artifacts or first-class database tables
- Contradiction detection needs a normalized vocabulary for "acceptable alternate outcome" categories

## Sources

### Primary (HIGH confidence)
- `/home/khoa2/delta-preservation/.planning/PROJECT.md` — milestone scope and constraints
- `/home/khoa2/delta-preservation/.planning/codebase/ARCHITECTURE.md` — current system structure
- `/home/khoa2/delta-preservation/.planning/codebase/STACK.md` — current technology choices
- `/home/khoa2/delta-preservation/shop/services/review.py` — current debug verdict/export implementation
- `/home/khoa2/delta-preservation/tests/test_debug_verdicts.py` — current debug workflow behavior and export guarantees

### Secondary (MEDIUM confidence)
- `/home/khoa2/delta-preservation/README.md` — current system capabilities and debug/report context
- User-provided description of current and desired debugging workflow during project initialization

### Tertiary (LOW confidence)
- None

---
*Research completed: 2026-04-09*
*Ready for roadmap: yes*
