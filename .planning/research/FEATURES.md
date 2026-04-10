# Feature Research

**Domain:** Ground-truth-assisted debug workflow for reconciliation accuracy
**Researched:** 2026-04-09
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Ground truth contract loading per part | Without reliable truth data, there is no trustworthy automated debug loop | LOW | Validate schema and fail loudly when a part fixture is incomplete or inconsistent |
| Automatic conformance evaluation after each run | The whole point of the milestone is to stop manually judging every characteristic | MEDIUM | Compare classification, requirement text, and snippet tolerance against `ground_truth.json` |
| Exception-only manual review queue | Human attention should go only to disagreements or ambiguity | MEDIUM | Reuse the current debug UI but filter it to nonconforming rows |
| Deterministic `debug_report.json` export | Each run needs a portable artifact that explains what passed automatically and what still needs review | LOW | Extend the current debug export shape instead of inventing a new reporting format |
| Separate exceptions/history persistence | Alternate acceptable outcomes must be captured without mutating canonical truth | MEDIUM | This is the control point that prevents contradictory judgments from silently shaping the system |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Cross-part contradiction detection | Flags when accepted outcomes on one part risk teaching a rule that fails elsewhere | HIGH | This directly supports the project's core value of generalization over fixture-specific success |
| Tolerant snippet acceptance rules | Prevents noisy false mismatches when the annotation is clearly in view but the center moved | MEDIUM | Should favor visible annotation/context over exact coordinate equality |
| Alternate acceptable outcome modeling | Lets one mismatch be judged acceptable without weakening the truth baseline | HIGH | Useful for multi-answer cases such as equivalent reconciliation interpretations |
| Run-over-run debug trend summaries | Makes algorithm improvements measurable across repeated benchmark runs | MEDIUM | Helpful after the basic evaluator and history layer exist |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Auto-rewriting `ground_truth.json` from review decisions | Feels like the fastest way to keep truth up to date | It erodes the baseline and hides drift between runs | Keep immutable truth plus append-only exceptions/history |
| Auto-tuning the algorithm from debug feedback | Promises faster accuracy gains | It risks overfitting to the benchmark corpus before contradictions are understood | Stop at evidence + contradiction surfacing in this milestone |
| Per-part hardcoded acceptance rules inside classifier code | Makes bad fixtures pass immediately | These hacks make the algorithm brittle across other parts | Keep acceptance logic in evaluator/history layers, not core matching/classification |
| Pixel-perfect snippet matching | Feels objective | It rejects visually acceptable snippets and wastes review time | Use tolerance windows or region visibility checks |

## Feature Dependencies

```text
Ground truth schema validation
    └──requires──> Automatic conformance evaluator
                         └──requires──> Deterministic debug_report export

Exception/history persistence
    └──enables──> Alternate acceptable outcome modeling
                         └──enables──> Cross-part contradiction detection

Tolerant snippet acceptance
    └──enhances──> Automatic conformance evaluator
```

### Dependency Notes

- **Automatic conformance evaluator requires ground truth schema validation:** bad fixture inputs will otherwise create false contradictions and unreliable reports.
- **Alternate acceptable outcome modeling requires exception/history persistence:** the system needs a durable place to store acceptable deviations without mutating truth.
- **Cross-part contradiction detection depends on normalized history records:** contradiction logic is weak if judgments stay buried in freeform explanations.
- **Tolerant snippet acceptance enhances automatic conformance evaluation:** it prevents visually acceptable results from being treated as failures.

## MVP Definition

### Launch With (v1)

- [ ] Ground truth loader and validator for each part fixture — essential for deterministic evaluation
- [ ] Automatic evaluator that marks conforming characteristics as correct — core speed improvement
- [ ] Focused manual review path for nonconforming or ambiguous rows — preserves human judgment where it matters
- [ ] Extended `debug_report.json` that includes auto-pass results and review-needed rows — required artifact for iterative runs
- [ ] Separate exceptions/history layer for acceptable alternatives — required to avoid rewriting truth

### Add After Validation (v1.x)

- [ ] Cross-run contradiction views — add once history records exist and real conflicts appear
- [ ] Aggregate benchmark summaries across many parts — add once per-run reports are stable
- [ ] Helper tooling for seeding exception entries from reviewed mismatches — add once the review workflow settles

### Future Consideration (v2+)

- [ ] Automated suggestion ranking for likely acceptable alternatives — defer until contradiction handling is reliable
- [ ] Semi-automated parameter tuning experiments — defer until evaluation infrastructure proves it will not overfit the corpus
- [ ] Multi-user debug collaboration — not relevant for the current solo-developer workflow

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Automatic conformance evaluation | HIGH | MEDIUM | P1 |
| Exception-only manual review | HIGH | MEDIUM | P1 |
| Separate exceptions/history layer | HIGH | MEDIUM | P1 |
| Tolerant snippet acceptance | HIGH | MEDIUM | P1 |
| Cross-part contradiction detection | HIGH | HIGH | P2 |
| Run-over-run trend summaries | MEDIUM | MEDIUM | P2 |
| Review-assisted exception authoring | MEDIUM | MEDIUM | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | Generic QA diff tools | Generic labeling tools | Our Approach |
|---------|------------------------|------------------------|--------------|
| Automatic pass/fail against truth | Usually weak or absent for domain-specific drawing semantics | Usually available, but detached from reconciliation internals | Compare directly against drawing-specific truth plus pipeline evidence |
| Exception handling | Often manual notes only | Often supports adjudication but not domain-specific alternates | Store alternate acceptable outcomes separately from canonical truth |
| Cross-part consistency | Rarely addressed | Rarely addressed | Treat contradiction detection as a core quality guardrail |

## Sources

- `/home/khoa2/delta-preservation/.planning/PROJECT.md`
- `/home/khoa2/delta-preservation/README.md`
- `/home/khoa2/delta-preservation/shop/services/review.py`
- User-provided workflow description during project initialization

---
*Feature research for: ground-truth-assisted debug workflow*
*Researched: 2026-04-09*
