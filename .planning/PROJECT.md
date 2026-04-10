# Delta Preservation

## What This Is

Delta Preservation is a brownfield aerospace drawing-reconciliation system with two validated surfaces: a standalone pipeline that compares Rev A and Rev B drawings against AS9102 Form 3 data, and a web workflow for review, sign-off, and debug inspection. The current project focus is not a new end-user feature, but a faster and more consistent internal debug loop that compares each run against stable part-level ground truth and helps one developer improve the algorithm across many drawing pairs without overfitting to any single part.

## Core Value

Improve reconciliation accuracy across many parts through a consistent, evidence-rich debug workflow that strengthens the algorithm without teaching it part-specific hacks.

## Requirements

### Validated

- ✓ Automated reconciliation pipeline compares Rev A and Rev B drawings against FAIR/Form 3 characteristics and emits classifications with evidence artifacts — existing
- ✓ Web application supports run submission, background processing, review queues, and sign-off workflow for completed runs — existing
- ✓ Admin debug mode captures per-item verdicts and exports a debug report once a run has been fully reviewed — existing

### Active

- [ ] Each part run can automatically compare pipeline output against that part's `ground_truth.json`
- [ ] Characteristics that conform to ground truth are auto-marked correct without manual verdict entry
- [ ] Nonconforming or ambiguous characteristics are isolated into a focused manual-review workflow and per-run `debug_report.json`
- [ ] A separate exceptions/history layer can record alternate acceptable outcomes without mutating `ground_truth.json`
- [ ] The debug workflow can surface cross-part contradiction risk so review decisions improve general algorithm behavior rather than overfitting to one part

### Out of Scope

- Automatic updates to `ground_truth.json` — the canonical ground truth must remain stable and manually curated
- Automatic algorithm self-tuning from review history — this milestone stops at better debugging infrastructure and evidence, not a learning loop
- End-user-facing debug tooling — this workflow is for the solo developer, not future production users

## Context

The existing system already processes revision drawing pairs plus FAIR spreadsheets and produces evidence-rich delta outputs through a CLI pipeline and a FastAPI review application. Each part in `assets/` now also has a `ground_truth.json` file containing the expected Rev B requirement plus Rev A/Rev B snippet coordinates for each characteristic, which creates a path to replace repetitive manual debug review. Previously, debugging required launching the Docker UI, running each part in debug mode, and manually marking every characteristic as correct, incorrect, or partially correct while also supplying corrected answers and explanations. The next milestone is to use the new ground-truth assets to speed that loop up, auto-pass conforming items, and reserve manual effort for exceptions and ambiguity. Snippet correctness should be tolerant rather than pixel-exact: if the relevant characteristic annotation and surrounding context are visible in the snippet image, that result can be considered acceptable.

## Constraints

- **Generalization**: Debug decisions must improve behavior across many parts — the system cannot be shaped around one drawing pair at the expense of others
- **Ground truth stability**: `ground_truth.json` files are canonical references and must not be auto-edited by the workflow
- **Human review boundary**: Nonconforming results still require manual inspection because more than one outcome may be acceptable in some cases
- **Developer-only scope**: This milestone is optimized for one maintainer's iterative run -> review -> rerun loop, not a multi-user debug product
- **Snippet tolerance**: Snippet evaluation should favor useful visible context over exact center-coordinate matching

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as a brownfield debugging-infrastructure milestone instead of a new product feature | The current need is faster and more consistent algorithm refinement on real drawing pairs | — Pending |
| Keep `ground_truth.json` immutable during runs | Stable ground truth prevents accidental drift and preserves a trusted baseline across iterations | — Pending |
| Store alternate acceptable outcomes in a separate exceptions/history layer | Some mismatches may still be acceptable, and that nuance should not overwrite the canonical truth | — Pending |
| Auto-pass conforming characteristics and focus human review on nonconforming items | Manual review should be reserved for disagreement, ambiguity, and contradiction detection | — Pending |
| Stop short of automatic learning/self-correction | The immediate goal is better debugging evidence and consistency, not training the algorithm from review history | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone**:
1. Full review of all sections
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-09 after initialization*
