# Phase 1: Evaluation Foundation - Context

**Gathered:** 2026-04-10
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers deterministic post-run evaluation against immutable part-level ground truth for completed runs. It covers exact truth-fixture resolution, strict truth validation, requirement conformance checks, tolerant-but-deterministic snippet conformance, and structured mismatch output that later review/export steps can consume.

It does not change the review UI shape, narrow the queue, or persist acceptable alternate outcomes. Those remain in later phases.

</domain>

<decisions>
## Implementation Decisions

### Ground Truth Resolution And Validation
- **D-01:** A run must resolve to exactly one truth fixture by a strict mapping rule; there is no fuzzy or best-effort truth lookup.
- **D-02:** The fixture directory name is the canonical truth key, and the run must carry that exact fixture key.
- **D-03:** Missing, malformed, or structurally invalid `ground_truth.json` data hard-fails the whole run.
- **D-04:** Truth-vs-packet coverage mismatches are normal evaluation mismatches that become `review-needed`; they do not hard-fail the run by themselves.
- **D-05:** Canonical `added` entries are matched as an unordered pool by requirement plus snippet evidence, not by `char_no` ordering.
- **D-06:** Truth validation uses status-specific field rules rather than one uniform required-field contract for every characteristic.
- **D-07:** `requirement_revB` may be null for `removed`; `snippet_center_revA` may be null for `added` when the feature exists only in a new Rev B view; `snippet_center_revB` may be null for `removed` when the feature disappears with the removed view; Rev A requirement authority still comes from FAIR/Form 3 rather than ground truth.
- **D-08:** If truth is structurally valid but the evaluator still cannot match it unambiguously, the run stays alive and the affected rows become `review-needed`.

### Requirement Conformance
- **D-09:** Requirement comparison uses semantic equivalence when the existing parser can confidently prove the same engineering meaning; otherwise it falls back to normalized-text comparison.
- **D-10:** Missing semantic parsing alone does not fail conformance; fallback normalized-text comparison is allowed unless there is another mismatch.
- **D-11:** A larger extracted annotation can still conform when it clearly preserves the same engineering callout and does not change meaning.
- **D-12:** If truth expects a non-null `requirement_revB` but the packet has no usable requirement evidence, the row is `review-needed`.
- **D-13:** Requirement uncertainty blocks auto-pass even if classification and snippet evidence are otherwise strong.

### Snippet Conformance
- **D-14:** Snippet acceptance uses a deterministic containment rule with an edge guard, not exact center-coordinate equality.
- **D-15:** If a truth center exists for a side, that side must produce a snippet and that snippet must pass the deterministic rule.
- **D-16:** If a truth center is null for a side, no snippet is required or expected for that side.
- **D-17:** Notes blocks and grouped multi-span callouts use a looser but still deterministic grouped-callout snippet rule instead of the single-callout safe-zone rule.
- **D-18:** If the deterministic snippet rule fails, the row becomes `review-needed`; a human-plausible crop does not auto-pass it.

### Evaluation Mismatch Contract
- **D-19:** Nonconforming rows emit stable machine-readable mismatch entries and also keep human-readable `reasons`.
- **D-20:** Rows may carry multiple mismatch entries in deterministic order rather than collapsing to one dominant reason.
- **D-21:** Snippet mismatch entries must be side-specific and cause-specific.
- **D-22:** Hard truth-contract failures stay outside per-row evaluation output; rows that reach evaluation can still distinguish ordinary mismatches from evaluator ambiguity or fallback states.

### the agent's Discretion
- Exact edge-guard margin and grouped-callout safe-zone math for snippet acceptance
- Exact field names and serialization shape for the structured mismatch contract, as long as it remains stable and downstream-friendly
- Exact deterministic ordering of mismatch families after preserving the required separation between truth-contract failures, ordinary mismatches, and evaluator ambiguity/fallback

</decisions>

<specifics>
## Specific Ideas

- Snippet tolerance should favor useful visible context over exact center equality, but still remain deterministic enough for auto-pass decisions.
- If a truth center exists, there needs to be a snippet for that side; if the truth center is null, that side should not produce or require a snippet.
- Semantic requirement equivalence is allowed only when the existing parser can prove the same engineering meaning with confidence.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope And Requirements
- `.planning/ROADMAP.md` — Phase 1 goal, success criteria, and plan breakdown
- `.planning/PROJECT.md` — milestone constraints, immutable-ground-truth rule, and snippet-tolerance intent
- `.planning/REQUIREMENTS.md` — Phase 1 requirement set (`GTRU-*`, `EVAL-*`)
- `.planning/STATE.md` — current focus and open concerns carried into planning

### Domain Guidance
- `.planning/MANUFACTURING_QUALITY_KNOWLEDGE.md` — manufacturing semantics for GD&T, notes, fits, and surface-finish requirements

### Prior Research
- `.planning/research/SUMMARY.md` — recommended architecture, overfitting guardrails, and snippet-tolerance pitfalls

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `delta_preservation/types.py` — `DeltaItem` and `Evidence` already define the packet contract for status, reasons, requirement text, bbox metadata, and snippet image paths
- `delta_preservation/vision/bbox_utils.py` — bbox expansion, paired-size normalization, and coverage expansion already exist for deterministic snippet generation
- `delta_preservation/cli.py` — Stage 7 already computes classification, evidence bboxes, snippet files, and `requirement_revB`; Phase 1 evaluation should extend this flow rather than invent a parallel artifact path
- `shop/services/semantics.py` — existing export/review logic already derives compact summaries from stable reason prefixes, which is a useful precedent for structured mismatch output

### Established Patterns
- The review surface is packet-driven: `shop/services/review.py` seeds `ReviewItem` rows from `delta_packet.json` instead of recomputing reconciliation state in the web tier
- Human-readable `reasons` arrays are already first-class packet output and should remain available even if Phase 1 adds machine-readable mismatch structures
- Evidence is represented as `page + bbox + image_path`, with missing evidence expressed by omitting that side rather than inventing placeholder geometry
- Export/review code already relies on deterministic ordering between packet rows and persisted review items

### Integration Points
- Truth resolution and evaluation need to run after the pipeline produces `delta_packet.json` but before later debug workflow logic assumes every row needs manual review
- Structured mismatch details must be available to `shop/services/review.py` and `shop/services/exports.py` without depending on reviewer-entered debug verdicts
- Snippet conformance should build on the existing snippet bbox pipeline in `delta_preservation/cli.py`, not a separate image-analysis path

</code_context>

<deferred>
## Deferred Ideas

- Explicit projected-view recognition may be discussed later if grouped snippet evaluation proves too brittle for new-view annotations.

</deferred>

---

*Phase: 01-evaluation-foundation*
*Context gathered: 2026-04-10*
