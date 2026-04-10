# Phase 1: Evaluation Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `01-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-04-10
**Phase:** 01-evaluation-foundation
**Areas discussed:** Truth validation boundary, Requirement comparison strictness, Snippet acceptance rule, Mismatch reason contract

---

## Truth validation boundary

### What should happen when ground truth is missing, malformed, or has any characteristic-level contract issue?

| Option | Description | Selected |
|--------|-------------|----------|
| Fail the entire run if `ground_truth.json` is missing, malformed, or has any characteristic-level contract issue. | Strict truth validation. | ✓ |
| Fail only for file-level problems, but keep the run alive and mark affected rows as `review-needed` when individual truth entries are missing or malformed. | Mixed hard-fail / row-fail approach. | |
| Never fail evaluation; always downgrade truth problems into `review-needed` rows and warnings. | Fully soft-fail evaluation. | |

**User's choice:** Any structurally bad `ground_truth.json` entry hard-fails the whole run.
**Notes:** Structural truth quality is part of the benchmark contract, not a row-level disagreement.

### If the truth file is valid, but the algorithm output does not line up with it, what should happen?

| Option | Description | Selected |
|--------|-------------|----------|
| Treat that as a normal evaluation mismatch and surface it as `review-needed`. | Keep run alive for row-level disagreement. | ✓ |
| Treat any truth-vs-packet coverage mismatch as a hard run failure. | Escalate all coverage disagreement into hard failure. | |
| Other. | Freeform alternative. | |

**User's choice:** Truth-vs-packet coverage mismatches are normal evaluation mismatches and surface as `review-needed`.
**Notes:** Coverage mismatch is different from malformed truth.

### How should the evaluator pair algorithm-detected `added` rows to ground-truth `added` entries?

| Option | Description | Selected |
|--------|-------------|----------|
| Set-based matching: treat truth-added entries as an unordered pool and match them by requirement + snippet evidence, ignoring `char_no`. | Pool matching. | ✓ |
| Requirement-first matching: match only by `requirement_revB`; snippet is just a confirmation signal. | Requirement-dominant matching. | |
| Snippet-first matching: match mainly by snippet/evidence location; requirement is secondary. | Spatial-first matching. | |
| Other. | Freeform alternative. | |

**User's choice:** Treat canonical `added` entries as an unordered pool and match them by requirement plus snippet evidence, ignoring `char_no` ordering.
**Notes:** `added` rows should not assume stable numbering.

### Should truth validation enforce different required fields depending on the canonical classification?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes. Use status-specific validation rules per characteristic type. | Classification-aware truth contract. | ✓ |
| No. Keep one uniform required-field contract for every truth entry. | Single schema for all rows. | |
| Other. | Freeform alternative. | |

**User's choice:** Yes. Use status-specific validation rules per characteristic type.
**Notes:** Field nullability depends on whether the characteristic is unchanged, changed, removed, or added.

### What field-level nullability rules matter for truth validation?

| Option | Description | Selected |
|--------|-------------|----------|
| User-defined rules. | Explicit clarification captured directly. | ✓ |

**User's choice:** `requirement_revB` may be null for `removed`; `snippet_center_revA` may be null for `added` when the feature only exists in a new Rev B view; `snippet_center_revB` may be null for `removed` when the feature disappears with the removed view; Rev A requirement remains authoritative from FAIR/Form 3, not ground truth.
**Notes:** This clarification supplements the status-specific validation rule rather than replacing it.

### What if the truth file is structurally valid, but an evaluator still cannot match it unambiguously?

| Option | Description | Selected |
|--------|-------------|----------|
| A normal evaluation ambiguity, so the run stays alive and those rows become `review-needed`. | Soft-fail ambiguous evaluation. | ✓ |
| A truth-definition problem severe enough to hard-fail the run. | Escalate ambiguity into truth failure. | |
| Other. | Freeform alternative. | |

**User's choice:** Treat evaluator ambiguity as a normal evaluation ambiguity; keep the run alive and mark those rows as `review-needed`.
**Notes:** Ambiguity belongs in review, not truth validation failure.

### How should the evaluator decide which `ground_truth.json` belongs to a completed run?

| Option | Description | Selected |
|--------|-------------|----------|
| Exact fixture mapping only: a run must resolve to one specific truth file by a strict mapping rule, or the run hard-fails. | Strict fixture resolution. | ✓ |
| Best-effort fuzzy/alias lookup: try to guess the nearest truth file from part metadata. | Heuristic fixture resolution. | |
| Explicit truth target selection: the run/debug flow should name the truth fixture directly instead of inferring it. | Manually selected truth target. | |
| Other. | Freeform alternative. | |

**User's choice:** A run must resolve to one exact truth fixture by a strict mapping rule or hard-fail.
**Notes:** No fuzzy fixture lookup.

### What is the canonical key for resolving a run to exactly one `ground_truth.json`?

| Option | Description | Selected |
|--------|-------------|----------|
| The fixture directory name is the source of truth, and the run must carry that exact fixture key. | Directory-key mapping. | ✓ |
| The run’s `part_number` field must exactly match a truth-file identifier. | Part-number keying. | |
| A separate explicit mapping table/file should map run identifiers to fixture directories. | External map layer. | |
| Other. | Freeform alternative. | |

**User's choice:** The fixture directory name is the source of truth, and the run must carry that exact fixture key.
**Notes:** This locks the canonical lookup key for planning.

---

## Requirement comparison strictness

### What should count as conforming when requirement formatting differs but engineering meaning is the same?

| Option | Description | Selected |
|--------|-------------|----------|
| Exact raw-text equality only. | Strict string equality. | |
| Normalized-text equality for formatting-only differences, but not deeper semantic equivalence. | Text normalization only. | |
| Semantic equivalence when the existing parser can confidently prove the same engineering meaning; otherwise fall back to normalized text. | Semantic-first fallback comparison. | ✓ |
| Other. | Freeform alternative. | |

**User's choice:** Use semantic equivalence when the existing parser can confidently prove the same engineering meaning; otherwise fall back to normalized-text comparison.
**Notes:** Confidence in the parser matters; semantic fallback is not unconditional.

### What should happen when semantic parsing is unavailable for a pair?

| Option | Description | Selected |
|--------|-------------|----------|
| Fall back to normalized-text comparison only; do not penalize missing semantics by itself. | Text fallback. | ✓ |
| Any semantic-parser gap makes the requirement comparison `review-needed`. | Parser availability is mandatory. | |
| Treat parser `error` as `review-needed`, but `empty` / `not_implemented` can fall back to normalized text. | Mixed fallback. | |
| Other. | Freeform alternative. | |

**User's choice:** Fall back to normalized-text comparison only; do not penalize missing semantics by itself.
**Notes:** Missing semantics alone is not a failure.

### Should containment inside a larger extracted annotation count as conforming when the canonical requirement is clearly preserved?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, if the extra text is part of the same visible engineering callout and does not change the meaning. | Same-callout containment allowed. | ✓ |
| No, the extracted requirement must reduce to only the canonical requirement text/meaning. | Exact reduction only. | |
| Only for certain families like notes or grouped dimension stacks. | Conditional containment. | |
| Other. | Freeform alternative. | |

**User's choice:** Yes, if the extra text is part of the same visible engineering callout and does not change the meaning.
**Notes:** Grouped callouts can still conform when meaning is preserved.

### If truth expects a non-null `requirement_revB`, but the packet row has no usable `requirement_revB` text at all, what should happen?

| Option | Description | Selected |
|--------|-------------|----------|
| Mark it `review-needed` because requirement evidence is missing. | Missing requirement blocks auto-pass. | ✓ |
| Allow it to conform if classification and snippet both pass. | Requirement evidence optional. | |
| Allow that only for certain characteristic families. | Conditional allowance. | |
| Other. | Freeform alternative. | |

**User's choice:** Mark it `review-needed` because requirement evidence is missing.
**Notes:** Requirement evidence is mandatory when truth expects it.

### If the packet extracts a requirement that is semantically close but not confidently equivalent, should the evaluator ever lean conforming based on classification + snippet?

| Option | Description | Selected |
|--------|-------------|----------|
| Requirement uncertainty blocks auto-pass; it becomes `review-needed`. | Conservative conformance. | ✓ |
| If classification and snippet both pass strongly, allow conforming despite requirement uncertainty. | Majority-signal auto-pass. | |
| Only allow that for `unchanged`, not for `changed` or `added`. | Classification-specific compromise. | |
| Other. | Freeform alternative. | |

**User's choice:** Requirement uncertainty blocks auto-pass; it becomes `review-needed`.
**Notes:** Requirement ambiguity should not be outweighed by stronger signals elsewhere.

---

## Snippet acceptance rule

### What geometric pass rule should snippet evaluation use?

| Option | Description | Selected |
|--------|-------------|----------|
| Any containment | Pass if the truth center lands anywhere inside the generated snippet bbox. | |
| Containment with edge guard | Pass if the truth center is inside the snippet bbox and not just barely clipped at the edge. | ✓ |
| Fixed overlap window | Turn the truth center into a fixed expected box and require overlap with the generated snippet bbox. | |

**User's choice:** Use containment with an edge guard.
**Notes:** This keeps snippet acceptance tolerant without falling back to exact-center matching.

### How strict should snippet conformance be by classification?

| Option | Description | Selected |
|--------|-------------|----------|
| Both sides whenever possible | If both truth centers exist, both revs must pass. | ✓ |
| Classification-aware | `unchanged`/`changed` require both available sides; `removed` only requires Rev A; `added` only requires Rev B; any contract-allowed null side is ignored. | |
| One-side-is-enough | One good side can still auto-pass if classification and requirement match strongly. | |

**User's choice:** If a truth center exists, there needs to be a snippet; if a truth center is null, no snippet should be captured.
**Notes:** The user's clarification sharpened the rule beyond the original labels: truth-center presence controls snippet obligation per side.

### How should notes blocks and grouped multi-span callouts be treated?

| Option | Description | Selected |
|--------|-------------|----------|
| Same rule as everything else | Rely on the existing expanded snippet bbox only. | |
| Looser grouped-callout rule | Still deterministic, but allow a looser safe-zone rule for notes/grouped annotations. | ✓ |
| Never auto-pass them | Always send those to `review-needed`. | |

**User's choice:** Use a looser deterministic grouped-callout rule for notes and grouped annotations.
**Notes:** The rule remains deterministic; it is not a human-review override.

### If the center-based rule fails, but the correct annotation text is still visibly present to a human, should snippet still auto-pass?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes | Visible correct annotation is enough, even if the center missed. | |
| No | Snippet auto-pass still needs the deterministic rule to pass; otherwise it becomes `review-needed`. | ✓ |
| Only for grouped/notes callouts | Human-visible fallback only for complex callouts. | |

**User's choice:** No. If the deterministic snippet rule fails, the row does not auto-pass.
**Notes:** Human-plausible visibility belongs in manual review, not Phase 1 auto-pass logic.

---

## Mismatch reason contract

### What should the evaluator emit for nonconforming rows?

| Option | Description | Selected |
|--------|-------------|----------|
| Structured mismatch contract + human reasons | Add stable machine-readable mismatch entries, but keep human-readable `reasons` too. | ✓ |
| Human-readable reasons only | Keep relying on freeform reason strings. | |
| Single overall mismatch label | One top-level category plus optional notes. | |

**User's choice:** Structured mismatch contract plus human-readable `reasons`.
**Notes:** Phase 2 review/reporting should not need to infer everything from freeform text.

### If a row fails in multiple ways, how should that be represented?

| Option | Description | Selected |
|--------|-------------|----------|
| Primary reason only | Choose one dominant mismatch and suppress the rest. | |
| All triggered mismatches in deterministic order | Emit every triggered mismatch in a stable sequence. | ✓ |
| One visible reason plus hidden debug details | Split human and debug visibility. | |

**User's choice:** All triggered mismatches in deterministic order.
**Notes:** Multiple independent failures should remain visible to the debug workflow.

### How specific should snippet-related mismatch entries be?

| Option | Description | Selected |
|--------|-------------|----------|
| Side-specific and cause-specific | Distinguish missing snippet, outside safe zone, grouped-callout miss, etc. | ✓ |
| Generic snippet mismatch | One snippet reason regardless of side or cause. | |
| No separate snippet reason | Roll snippet failure into the overall mismatch summary only. | |

**User's choice:** Snippet mismatches should be side-specific and cause-specific.
**Notes:** Snippet tolerance is one of the core new evaluator behaviors in this phase.

### How should the contract distinguish evaluator problems from normal review-needed disagreements?

| Option | Description | Selected |
|--------|-------------|----------|
| Separate families | Hard truth-contract failures stay outside row evaluation; review-needed rows can still distinguish normal mismatch vs evaluator ambiguity/fallback. | ✓ |
| Collapse them together | Any non-pass is just a mismatch. | |
| Only distinguish hard-fail vs review-needed | No finer split inside review-needed rows. | |

**User's choice:** Keep separate families.
**Notes:** Structural truth failures hard-fail the run, while evaluator ambiguity still allows the run to continue with `review-needed` rows.

---

## the agent's Discretion

- Exact edge-guard threshold for snippet safe zones
- Exact grouped-callout safe-zone geometry
- Exact field names and ordering details for the structured mismatch contract

## Deferred Ideas

- Explicit projected-view recognition may be discussed later if snippet evaluation needs it.
