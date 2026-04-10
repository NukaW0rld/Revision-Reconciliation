# Phase 2: Focused Debug Workflow - Context

**Gathered:** 2026-04-10
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase narrows the admin debug workflow to an exception-only manual-review loop for completed runs. The main debug queue should focus on rows whose Phase 1 evaluation ended in `review_needed`, while auto-passed rows remain available for inspection through run details and the exported `debug_report.json`.

This phase also reshapes the debug artifact contract so `debug_report.json` distinguishes canonical matches, acceptable alternate outcomes, unresolved review-needed rows, and confirmed algorithm errors without requiring manual verdict entry for auto-passed rows.

This phase does not persist accepted alternates in a durable history layer and does not change `ground_truth.json`. Structured alternate-history storage remains Phase 3 work.

</domain>

<decisions>
## Implementation Decisions

### Exception Queue Scope
- **D-01:** The main admin debug queue defaults to exception-only behavior and includes only rows whose evaluation status is `review_needed`.
- **D-02:** Auto-passed (`conforming`) rows do not appear in the main manual-review queue.
- **D-03:** Auto-passed rows remain inspectable from the run-details surface and from the exported `debug_report.json`, not from the main exception queue.
- **D-04:** If a run has zero exception rows, the debug path should stay on run details and show a clean all-conforming summary instead of routing the reviewer into an empty queue.

### Exception Card Presentation
- **D-05:** Exception cards are mismatch-first: ordered mismatch codes/messages and the review action come before deeper evidence browsing.
- **D-06:** The default expanded content on an exception card is the mismatch summary plus the review form; scores, semantic internals, bbox centers, and longer debug detail should be collapsed behind secondary disclosures.
- **D-07:** The review interaction remains one focused exception card at a time rather than changing to a dense list or grouped family browser in Phase 2.

### Reviewer Outcome Model
- **D-08:** Reviewer resolution vocabulary for exception rows should move from `correct / incorrect / partially_correct` to explicit exception outcomes centered on `algorithm_error` and `acceptable_alternate`.
- **D-09:** Resolving an `algorithm_error` requires a corrected classification plus reviewer rationale; corrected requirement text stays optional.
- **D-10:** Resolving an `acceptable_alternate` requires reviewer rationale only in Phase 2; normalized structured alternate fields are deferred until the history layer in Phase 3.

### Debug Report Contract
- **D-11:** `debug_report.json` becomes exportable when every exception row has been resolved; auto-passed rows must never block report readiness.
- **D-12:** `debug_report.json` must explicitly distinguish these row-level states: `canonical_match`, `acceptable_alternate`, `unresolved_review_needed`, and `algorithm_error`.
- **D-13:** Auto-passed rows remain full per-row entries in `debug_report.json`; they are not reduced to counts-only or summary-only data.

### the agent's Discretion
- Exact wording and field names for the new exception outcome vocabulary, as long as the semantics stay aligned with `algorithm_error` and `acceptable_alternate`
- Exact visual treatment for collapsed versus expanded diagnostic panels on exception cards
- Exact run-details summary layout for all-conforming runs and auto-pass counts
- Exact JSON nesting/field layout for the expanded `debug_report.json` contract, as long as the four required row states remain explicit and downstream-friendly

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope And Requirements
- `.planning/ROADMAP.md` — Phase 2 goal, success criteria, and plan breakdown
- `.planning/PROJECT.md` — milestone constraints, focused-review intent, and immutable-ground-truth rule
- `.planning/REQUIREMENTS.md` — Phase 2 requirement set (`DREV-*`, `RPT-*`)
- `.planning/STATE.md` — current project position and carry-forward concerns from Phase 1

### Prior Phase Decisions
- `.planning/phases/01-evaluation-foundation/01-CONTEXT.md` — locked Phase 1 evaluator contract, mismatch ordering, and snippet-tolerance decisions that Phase 2 must preserve

### Prior Research
- `.planning/research/SUMMARY.md` — recommended milestone sequencing and focused-review direction
- `.planning/research/ARCHITECTURE.md` — exception-only queue and debug-evaluation architecture guidance
- `.planning/research/FEATURES.md` — table-stakes definition for exception-only review and deterministic debug reporting
- `.planning/research/PITFALLS.md` — specific Phase 2 pitfalls, especially “manual review still too broad”

### Existing Integration Points
- `shop/services/review.py` — current queue seeding, debug verdict persistence, and `debug_report.json` assembly
- `shop/routers/review.py` — current review/debug route behavior and export readiness gate
- `shop/templates/review/queue.html` — current full-queue review surface and debug notes area
- `shop/templates/review/_item_card_debug.html` — current debug card layout and verdict form
- `shop/templates/review/_signoff_footer.html` — current debug export readiness UX
- `shop/templates/runs/status.html` — current run-details surface where auto-passed summaries can surface without reopening the main queue

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `shop/services/review.py::open_review_queue` — current `delta_packet.json` to `ReviewItem` seeding path that Phase 2 will need to narrow or reinterpret for exception-only review
- `shop/services/review.py::assemble_debug_report_payload` — current deterministic export assembler and the best place to shift export readiness from “all rows need verdicts” to “all exceptions resolved”
- `shop/services/review.py::debug_internals_by_char` — existing mismatch/evaluation/scores collector that can feed mismatch-first exception cards
- `shop/routers/review.py::review_queue` — existing route that already supports debug mode and can branch queue behavior by evaluation status
- `shop/templates/runs/status.html` — existing run-details entry point and CTA area that can host auto-pass counts and all-conforming summaries

### Established Patterns
- The web tier is packet-driven: review/debug screens derive from `delta_packet.json` plus persisted `ReviewItem` rows rather than recomputing reconciliation state in the UI
- Ordered machine-readable mismatch entries already exist on packet evaluation objects and should stay first-class instead of being reduced to prose-only explanations
- Debug-mode behavior is explicitly admin-only and should remain partitioned from the normal review surface
- Current debug export flow is deterministic and queue-order-aware; Phase 2 should preserve deterministic ordering while changing which rows require active resolution

### Integration Points
- Queue composition changes will touch `shop/services/review.py`, `shop/routers/review.py`, and the review templates together; the queue surface and export readiness gate cannot be changed independently
- Auto-passed visibility belongs on the run-details surface in `shop/templates/runs/status.html`, not in the main exception queue
- The new reviewer outcome vocabulary should bridge cleanly into Phase 3’s history layer without forcing full alternate-history storage in Phase 2

</code_context>

<specifics>
## Specific Ideas

- The manual-review surface should feel meaningfully narrower than the old workflow: only exceptions enter the queue.
- Exception cards should optimize for fast triage: mismatch summary first, then action, with heavy debug detail hidden until needed.
- Runs with no exceptions should end in a clean run-details summary rather than an empty review page.
- `debug_report.json` should stay rich enough for offline inspection and rerun comparison, including full row detail for canonical matches.

</specifics>

<deferred>
## Deferred Ideas

- Normalized structured storage for accepted alternate outcomes — Phase 3
- Queryable contradiction and cross-part overfitting analysis — later phase after the history layer exists
- Rich grouping or mismatch-family browsing beyond the one-card-at-a-time exception flow — future UX enhancement if needed

</deferred>

---

*Phase: 02-focused-debug-workflow*
*Context gathered: 2026-04-10*
