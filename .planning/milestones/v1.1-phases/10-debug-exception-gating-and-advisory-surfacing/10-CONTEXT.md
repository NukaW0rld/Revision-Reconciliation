# Phase 10: debug-exception-gating-and-advisory-surfacing - Context

**Gathered:** 2026-04-18 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Make unresolved debug exceptions and classifier advisory flags visible and enforceable at sign-off/export time in the existing maintainer workflow. Scope is the shared debug/review/export contract: define which rows count as unresolved debug blockers, surface packet-side advisory flags on maintainer-facing surfaces, and prevent sign-off or downstream audit/work-order exports from bypassing that state. This phase does not change classifier semantics, ground-truth logic, or Phase 11 web E2E automation.

</domain>

<decisions>
## Implementation Decisions

### Debug Blocker Contract
- **D-01:** Phase 10 gating uses the existing debug-exception set built by review services: packet rows with `evaluation.status == "review_needed"` plus synthetic missing-added-truth rows that do not yet have a saved debug verdict.
- **D-02:** Conforming rows and history-backed acceptable alternates are not promoted to blockers. Phase 10 closes integration around existing exception semantics rather than redefining classifier truth.

### Sign-Off and Export Enforcement
- **D-03:** Sign-off, audit packet export, and work-order export must all consult the same service-level debug summary and unresolved-exception state. UI-only disabled buttons or `run.status == "signed_off"` checks are insufficient.
- **D-04:** Hard blocking is the baseline contract. If planning introduces an acknowledgement bypass, it must be explicitly recorded and preserved in a durable contract that downstream exports and signed artifacts can read later.

### Advisory Surfacing Contract
- **D-05:** `DeltaItem.confidence_flags` is the authoritative advisory source and must be rendered directly from packet data on maintainer-facing review, debug, status, and export surfaces.
- **D-06:** Advisory surfacing is additive. Flags annotate the packet verdict and mismatch context, but do not replace `status`, `evaluation`, or semantic summaries as the core classification contract.

### the agent's Discretion
- Whether shared gating enforcement lives entirely in `shop/services/review.py` or is split between review services and export helpers, as long as the same unresolved-exception contract drives every gated route.
- Whether the UI presents advisory flags as badges, inline warning panels, or another compact maintainer-first treatment, as long as the exact packet flags remain visible on normal review surfaces and exported artifacts.

</decisions>

<specifics>
## Specific Ideas

- The existing unresolved debug set already combines `review_needed` packet rows with synthetic missing-added truth rows; planning should reuse that contract instead of inventing a parallel blocker definition.
- The exact Phase 5 advisory phrase currently emitted is `Rev B text may contain adjacent balloon content`; advisory surfacing should prefer packet-native flag text over template-only paraphrases.
- The current state is asymmetric: `/review/{run_id}/debug-report.json` already blocks on `debug_report_ready`, but sign-off and signed-artifact downloads do not consult the same unresolved-exception state. Phase 10 should unify those paths.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope and Gap Closure
- `.planning/ROADMAP.md` — Phase 10 goal, affected requirements, gap-closure ids, and success criteria.
- `.planning/v1.1-MILESTONE-AUDIT.md` — GAP-01, GAP-02, and FLOW-02 describe the exact cross-phase integration break that Phase 10 closes.

### Project Constraints
- `.planning/PROJECT.md` — Maintainer-only workflow scope, generalization constraint, immutable ground-truth rule, and debug-loop priorities.
- `.planning/REQUIREMENTS.md` — Active/out-of-scope milestone boundaries and the requirement ids Phase 10 protects (`CLS-01`, `ADD-01`, `ADD-02`, `SNP-01`, `VER-01`).

### Prior Phase Decisions
- `.planning/phases/06-added-characteristic-detection-and-snippet-accuracy/06-CONTEXT.md` — Shared missing-added truth and exclusion-contract decisions that feed the current debug exception set.
- `.planning/phases/07-regression-tests-and-verification/07-CONTEXT.md` — Algorithm-only verification and authoritative baseline decisions for maintainer-facing debug evidence.
- `.planning/phases/09-full-corpus-added-characteristic-closure/09-CONTEXT.md` — Explicit separation of Phase 9 algorithm/evidence work from Phase 10 gating/export integration work.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `shop/services/review.py::build_debug_queue_state` and `build_run_debug_summary` — already compute `exception_items`, synthetic missing-added rows, unresolved counts, and `debug_report_ready`.
- `shop/services/review.py::assemble_debug_report_payload`, `save_debug_verdict`, `save_missing_added_truth_verdict`, `load_debug_notes` — existing durable debug-side records and export-ready payload assembly.
- `delta_preservation/types.py::DeltaItem.confidence_flags` plus `delta_preservation/cli.py` packet conversion — existing persisted advisory field with backward-compatible defaults.
- `shop/templates/review/queue.html`, `shop/templates/review/_item_card_debug.html`, `shop/templates/review/_signoff_footer.html`, and `shop/templates/runs/status.html` — existing maintainer surfaces where blockers and advisory flags can be surfaced without inventing a new workflow.

### Established Patterns
- Service-first review state: routers render from review-service summaries rather than recomputing packet truth in templates.
- Signed artifact gating currently flows through router policy plus `attempt_sign_off` / `_get_signed_run`; Phase 10 should extend that contract rather than adding template-only safeguards.
- Debug verdicts and notes are persisted as run-scoped JSON sidecars instead of mutating canonical packet data or `ground_truth.json`.
- Packet schema changes are additive and backward-compatible; advisory surfacing should tolerate legacy packets that omit `confidence_flags`.

### Integration Points
- Review queue and sign-off routes in `shop/routers/review.py`.
- Debug summary, queue assembly, and export payload construction in `shop/services/review.py`.
- Signed audit/work-order download gates in `shop/routers/exports.py` and export generation in `shop/services/exports.py`.
- Packet-side advisory/evaluation source in `delta_preservation/reconcile/classify.py`, `delta_preservation/cli.py`, and `delta_preservation/types.py`.

</code_context>

<deferred>
## Deferred Ideas

None — analysis stayed within the Phase 10 boundary. Phase 11 web run-to-review E2E automation remains a separate downstream phase.

</deferred>
