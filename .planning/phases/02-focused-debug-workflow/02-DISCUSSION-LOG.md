# Phase 2: Focused Debug Workflow - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `02-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-04-10
**Phase:** 02-Focused Debug Workflow
**Areas discussed:** Exception queue default, Queue card density, Reviewer outcome model, Report readiness + shape

---

## Exception Queue Default

| Option | Description | Selected |
|--------|-------------|----------|
| Exceptions only | Show only `review_needed` rows in the main queue; move auto-passed rows to a secondary surface | ✓ |
| Exceptions first, but expandable inline | Open on exceptions but allow conforming rows back into the same queue via toggle | |
| Full queue, better filters | Keep all rows in one queue and rely on filters/summaries | |

**User's choice:** Exceptions only
**Notes:** Auto-passed rows should be inspectable through run details and export, not through the main queue.

### Auto-passed row visibility

| Option | Description | Selected |
|--------|-------------|----------|
| Run details summary + export only | Keep auto-passed rows out of the queue; surface them on run details and in `debug_report.json` | ✓ |
| Separate auto-passed tab in review | Add a second review tab/list for conforming rows | |
| Inline drawer from exception queue | Reveal auto-passed rows through a collapsible drawer in the queue | |

**User's choice:** Run details summary + export only
**Notes:** The main manual-review path should stay clean.

### Zero-exception runs

| Option | Description | Selected |
|--------|-------------|----------|
| Stay on run details with all-conforming summary | Show `0 exceptions` and export readiness on run details | ✓ |
| Open an empty exception queue page | Route into review even when nothing needs review | |
| Open run details with explicit report link | Keep user on run details and emphasize the export artifact | |

**User's choice:** Stay on run details with all-conforming summary
**Notes:** No empty review detour.

---

## Queue Card Density

| Option | Description | Selected |
|--------|-------------|----------|
| Mismatch-first summary | Lead with mismatch codes/reasons and the review action | ✓ |
| Evidence-first | Lead with snippet images first | |
| Balanced split | Show mismatch summary and snippets side by side at the top | |

**User's choice:** Mismatch-first summary
**Notes:** Optimize for triage, not browsing.

### Expanded diagnostics

| Option | Description | Selected |
|--------|-------------|----------|
| Only mismatch summary expanded | Collapse scores, semantic internals, bbox centers, and long reason lists | ✓ |
| Mismatch + snippets expanded | Keep mismatch summary and snippets open by default | |
| Everything expanded | Preserve the current heavy-debug presentation | |

**User's choice:** Only mismatch summary expanded
**Notes:** Keep the queue sparse and fast to scan.

### Navigation model

| Option | Description | Selected |
|--------|-------------|----------|
| One focused card at a time | Preserve the focused-review feel for exceptions only | ✓ |
| Compact list, expand on demand | Start with a dense summary list | |
| Grouped by mismatch family | Cluster rows by mismatch type before review | |

**User's choice:** One focused card at a time
**Notes:** Lowest cognitive load per decision.

---

## Reviewer Outcome Model

| Option | Description | Selected |
|--------|-------------|----------|
| Algorithm error / acceptable alternate | Explicit exception-outcome vocabulary aligned with roadmap semantics | ✓ |
| Correct / incorrect / partially correct | Keep the current vocabulary and enrich fields behind it | |
| Approve alternate / reject alternate / needs more investigation | Phrase choices around truth disagreement instead of pipeline behavior | |

**User's choice:** Algorithm error / acceptable alternate
**Notes:** The current verdict vocabulary does not line up cleanly with Phase 2 and Phase 3 semantics.

### Algorithm error requirements

| Option | Description | Selected |
|--------|-------------|----------|
| Corrected classification + rationale | Structured corrected outcome plus explanation; requirement fields optional | ✓ |
| Corrected classification + corrected requirements + rationale | Require the full corrected interpretation every time | |
| Rationale only | Freeform explanation without structured correction | |

**User's choice:** Corrected classification + rationale
**Notes:** Enough structure for debugging without forcing unnecessary data entry.

### Acceptable alternate requirements

| Option | Description | Selected |
|--------|-------------|----------|
| Rationale only | Mark acceptable alternate and explain why; defer structured normalization | ✓ |
| Rationale + corrected classification | Capture one structured alternate field now | |
| Rationale + corrected classification + corrected requirements | Capture the full alternate interpretation now | |

**User's choice:** Rationale only
**Notes:** Full alternate-history normalization is deferred to Phase 3.

---

## Report Readiness + Shape

| Option | Description | Selected |
|--------|-------------|----------|
| When every exception row is resolved | Auto-passed rows never block export readiness | ✓ |
| Always exportable | Allow unresolved exceptions in the artifact at any time | |
| Only after explicit finalize action | Separate queue completion from export readiness | |

**User's choice:** When every exception row is resolved
**Notes:** Export readiness should follow the exception-only queue model.

### Explicit report row states

| Option | Description | Selected |
|--------|-------------|----------|
| canonical_match / acceptable_alternate / unresolved_review_needed / algorithm_error | Make all four states explicit in the export contract | ✓ |
| auto_pass / resolved_exception / unresolved_exception | Use broader buckets and rely on nested detail | |
| conforming / review_needed plus debug verdict fields | Keep Phase 1 vocabulary primary | |

**User's choice:** Explicit four-state contract
**Notes:** Best bridge into later history work while staying clear in Phase 2.

### Auto-passed row detail level

| Option | Description | Selected |
|--------|-------------|----------|
| Full row detail for all rows | Keep full packet/evaluation detail even for canonical matches | ✓ |
| Compact auto-pass rows, full exception rows | Trim canonical matches to a lighter summary | |
| Summary block only | No per-row canonical-match entries | |

**User's choice:** Full row detail for all rows
**Notes:** The report should remain useful for offline inspection and rerun comparison.

## the agent's Discretion

- Exact field names for the new exception outcome contract
- Exact UI mechanics for collapsed diagnostic panels
- Exact run-details summary presentation for all-conforming runs

## Deferred Ideas

- Durable alternate-outcome history storage
- Cross-part contradiction analysis
- Richer list/group navigation for exceptions beyond the focused one-card flow
