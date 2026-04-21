---
phase: 260421-phx
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - README.md
  - .planning/quick/260421-phx-i-want-you-to-update-the-readme-md-of-th/260421-phx-SUMMARY.md
autonomous: true
requirements:
  - QUICK-260421-phx
must_haves:
  truths:
    - "README.md explains Delta Preservation as a brownfield aerospace drawing-reconciliation system with both standalone pipeline and web workflow surfaces."
    - "README.md reflects the current stage: milestone v1.1 shipped on 2026-04-19, v1.0 and v1.1 capabilities are validated, and the next focus is cross-part consistency tooling."
    - "README.md describes the maintainer-facing ground-truth debug loop, including immutable ground_truth.json files, exception-only review, accepted alternates, debug reports, and sign-off gating."
    - "README.md explains the core pipeline, web workflow, data model, repository structure, install/run/test commands, configuration, constraints, known limitations, and active next items without requiring firsthand use."
    - "README.md does not update or imply automatic edits to canonical ground truth and does not present deferred algorithm work as already solved."
  artifacts:
    - path: "README.md"
      provides: "Extremely comprehensive, current project description for new readers"
      contains: "Current project stage"
    - path: ".planning/quick/260421-phx-i-want-you-to-update-the-readme-md-of-th/260421-phx-SUMMARY.md"
      provides: "Quick workflow completion summary"
      contains: "README.md updated"
  key_links:
    - from: "README.md current-stage narrative"
      to: ".planning/PROJECT.md"
      via: "current value, constraints, validated/current/out-of-scope requirement status"
      pattern: "v1.1.*2026-04-19|Current project stage"
    - from: "README.md milestone history"
      to: ".planning/milestones/v1.0-ROADMAP.md and .planning/milestones/v1.1-ROADMAP.md"
      via: "ground-truth debug workflow and cross-part refinement details"
      pattern: "ground_truth\\.json|accepted alternates|confidence_flags|33/35"
---

<objective>
Update README.md so it is an accurate, comprehensive, current description of Delta Preservation.

Purpose: The README is the main orientation document for the project. A reader should understand the domain problem, system surfaces, shipped debug workflow, current algorithm state, web workflow, constraints, operational commands, and known limitations without running the app first.

Output: One revised README.md plus the quick-workflow SUMMARY artifact. Do not update .planning/ROADMAP.md for this quick documentation task.
</objective>

<execution_context>
@/home/khoa2/.codex/get-shit-done/workflows/execute-plan.md
@/home/khoa2/.codex/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/milestones/v1.0-ROADMAP.md
@.planning/milestones/v1.1-ROADMAP.md
@AGENTS.md
@README.md

Key facts to preserve:

- Current state: milestone v1.1 "Cross-part Characteristic Matching Refinement" is completed and archived as of 2026-04-19; the project is ready for the next milestone.
- Core value: improve reconciliation accuracy across many parts through a consistent, evidence-rich debug workflow without part-specific hacks.
- Validated surfaces: standalone pipeline comparing Rev A/Rev B/Form 3 data, and FastAPI web workflow for review, sign-off, export, and debug inspection.
- v1.0 shipped deterministic ground-truth evaluation, exception-only debug review, accepted-alternate history, and debug_report.json output while keeping canonical ground_truth.json immutable.
- v1.1 shipped GD&T parser fixes, classification fixes, added-characteristic detection improvements, shared title-block exclusion, confidence_flags, sign-off gating on unresolved debug exceptions, signed debug snapshots for export fidelity, and live web run-to-review E2E coverage.
- Current known limitations/active next items include cross-part contradiction detection, benchmark trend summaries, overfitting warnings, Part 5 thread/countersink matching-layer support for indexes 16+17, Part 9 truth_index 42 false absorption, and composite GD&T positional-zone modifiers.
- Testing baseline from project context: 500 passed, 2 xfailed after v1.1.
- Ground truth files are canonical references and must not be auto-edited by the workflow.
- The debug workflow is developer/maintainer-facing, not a multi-user debug product.
</context>

<source_audit>
| Source Item | Coverage |
|-------------|----------|
| GOAL-01: Update README.md to match current project stage | Covered by Task 1 |
| GOAL-02: README must be extremely comprehensive and understandable without firsthand use | Covered by Task 1 and Task 2 |
| REQ-01: Single plan with 1-3 focused tasks | Covered by this plan; two tasks |
| REQ-02: No research phase | Covered; only local planning/codebase docs are used |
| REQ-03: Update GSD artifacts only as required by quick workflow, not ROADMAP.md | Covered by Task 2; files_modified excludes ROADMAP.md |
| CONTEXT-01: Generalization/no part-specific hacks | Covered by Task 1 README constraints/current-state sections |
| CONTEXT-02: ground_truth.json canonical and immutable | Covered by Task 1 debug workflow/out-of-scope sections |
| CONTEXT-03: Nonconforming results still require manual inspection | Covered by Task 1 review/debug sections |
| CONTEXT-04: Developer-only debug scope | Covered by Task 1 audience/scope sections |
| CONTEXT-05: Snippet tolerance favors visible context over exact center matching | Covered by Task 1 ground-truth evaluation details |
| RESEARCH: None provided and none required | Excluded by task constraints |
</source_audit>

<tasks>

<task type="auto">
  <name>Task 1: Rewrite README.md around the current shipped system</name>
  <files>README.md</files>
  <action>
Replace the stale README content with a comprehensive current project description. Keep the README reader-facing, not process-artifact-heavy, but include enough technical detail that a new maintainer understands the system without running it.

Required README structure:

1. Title and short summary describing Delta Preservation as a brownfield aerospace drawing-reconciliation system with two validated surfaces: standalone pipeline and web workflow.
2. "Current project stage" section with milestone v1.0 shipped 2026-04-12, milestone v1.1 shipped 2026-04-19, current status as planning next milestone, and the 500 passed / 2 xfailed test baseline from project context.
3. Domain/problem section explaining AS9102 Form 3, Rev A/Rev B drawings, characteristics, partial FAIR relevance, and why manual reconciliation is risky.
4. System overview section explaining the two surfaces and how they share the same packet/evidence model.
5. Core pipeline section covering inputs, outputs, delta_packet.json, snippets/debug artifacts, and the 8 pipeline stages from Form 3 parsing through output generation.
6. Algorithm/debug accuracy section covering GD&T semantic parsing, classification statuses, confidence_flags, adjacency bleed suppression, removed+added reconciliation, asymmetric tolerance detection, added-characteristic detection, title-block exclusion, and the 33/35 added-truth claim with the two Part 5 deferrals explicitly documented as not solved.
7. Ground-truth debug workflow section covering immutable assets/{part}/ground_truth.json, deterministic conformance evaluation, snippet tolerance, exception-only queue, debug_report.json, accepted alternates, same-part reuse only, and why canonical truth is not auto-edited.
8. Web workflow section covering setup/auth/roles, run submission and validation, Huey background processing, SSE progress, review items, admin debug review, sign-off gate on unresolved debug exceptions, signed debug snapshot, export artifacts, amendments, admin settings, and retention cleanup.
9. Architecture and persistence section covering delta_preservation/, shop/, SQLAlchemy/Alembic/SQLite, Huey, Jinja2 templates, WeasyPrint exports, and separation between core reconciliation logic and web state.
10. Repository structure section with current major directories and entry points.
11. Installation and usage section with uv install, sample CLI run, direct CLI invocation, local web app commands including Huey worker, Docker compose, CSS build command if relevant, and environment variables.
12. Testing section with pytest command, focused examples for algorithm/debug/web E2E test categories, and note that the suite includes corpus regression and live run-to-review coverage.
13. Constraints, scope boundaries, and known active next items section. Include generalization, ground-truth stability, human review boundary, developer-only debug scope, snippet tolerance, scanned/raster PDF rejection, current single-page pipeline path if still present in the existing README, no PLM/QMS integrations, cross-part contradiction detection, benchmark trends, overfitting warning, Part 5 indexes 16+17, Part 9 truth_index 42, and composite GD&T positional-zone modifiers.
14. Glossary section defining Rev A/Rev B, AS9102 Form 3, characteristic, delta packet, ground truth, accepted alternate, debug exception, confidence flag, and FAIR.

Do not modify .planning/ROADMAP.md. Do not state that known deferrals are complete. Do not suggest the workflow automatically edits ground_truth.json. Do not make this sound like a new end-user debug product; preserve the maintainer-facing scope from PROJECT.md.
  </action>
  <verify>
    <automated>cd /home/khoa2/delta-preservation && python -c 'from pathlib import Path; text=Path("README.md").read_text(); required=["Current project stage","v1.1","2026-04-19","ground_truth.json","accepted alternates","debug_report.json","confidence_flags","33/35","Part 5","truth_index 42","Huey","SQLAlchemy","pytest","Glossary"]; missing=[s for s in required if s not in text]; assert not missing, missing'</automated>
  </verify>
  <done>
    README.md is rewritten as a current, comprehensive project orientation document that covers shipped v1.0/v1.1 behavior, current limitations, architecture, workflows, setup, testing, and glossary without contradicting project constraints.
  </done>
</task>

<task type="auto">
  <name>Task 2: Source-check README.md and write the quick summary</name>
  <files>README.md, .planning/quick/260421-phx-i-want-you-to-update-the-readme-md-of-th/260421-phx-SUMMARY.md</files>
  <action>
Read the finished README.md once against the local source documents listed in the context block. Correct any stale claims, missing current-state facts, or accidental scope expansion. Confirm README.md does not mention updating .planning/ROADMAP.md and does not present deferred work as shipped.

Create `.planning/quick/260421-phx-i-want-you-to-update-the-readme-md-of-th/260421-phx-SUMMARY.md` with:

- What changed in README.md
- Sources used
- Verification command run and result
- Any remaining documentation caveats, if present

Do not create additional GSD artifacts beyond the summary file.
  </action>
  <verify>
    <automated>cd /home/khoa2/delta-preservation && test -f .planning/quick/260421-phx-i-want-you-to-update-the-readme-md-of-th/260421-phx-SUMMARY.md && python -c 'from pathlib import Path; text=Path("README.md").read_text(); forbidden=["automatically edits ground_truth.json","Part 5 thread/countersink support is complete","Part 9 truth_index 42 is fixed"]; hits=[s for s in forbidden if s in text]; assert not hits, hits'</automated>
  </verify>
  <done>
    README.md has been checked against STATE.md, PROJECT.md, ROADMAP.md, and milestone archives; quick SUMMARY exists; no ROADMAP.md update was made.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Local planning docs -> public README | Internal project state is summarized into a reader-facing repository document. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-260421-phx-01 | Information Disclosure | README.md | mitigate | Summarize architecture and workflow only; do not include secrets, private customer data, or full corpus ground-truth contents. |
| T-260421-phx-02 | Tampering | README.md project status | mitigate | Cross-check current-stage claims against STATE.md, PROJECT.md, ROADMAP.md, and v1.0/v1.1 milestone archives before writing summary. |
</threat_model>

<verification>
Run the task-level verification commands. Then inspect git diff for scope:

```bash
cd /home/khoa2/delta-preservation && git diff -- README.md .planning/quick/260421-phx-i-want-you-to-update-the-readme-md-of-th/260421-phx-SUMMARY.md
```
</verification>

<success_criteria>
- README.md is comprehensive enough for a reader to understand the project, current stage, architecture, workflows, commands, testing, constraints, and known limitations without running the app.
- README.md reflects shipped v1.0 and v1.1 state accurately and names deferred work as deferred.
- Canonical ground-truth stability, human review boundary, and maintainer-facing debug scope are explicit.
- Only README.md and the quick SUMMARY artifact are modified.
</success_criteria>

<output>
After completion, create `.planning/quick/260421-phx-i-want-you-to-update-the-readme-md-of-th/260421-phx-SUMMARY.md`.
</output>
