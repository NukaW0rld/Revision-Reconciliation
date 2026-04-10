# Pitfalls Research

**Domain:** Ground-truth-assisted debug workflow for reconciliation accuracy
**Researched:** 2026-04-09
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Benchmark Overfitting

**What goes wrong:**
The system starts passing a handful of benchmark parts by relying on fixture-specific assumptions instead of improving general reconciliation behavior.

**Why it happens:**
When a developer sees repeated failures on one part, the fastest fix is often a narrow heuristic or acceptance rule tailored to that fixture.

**How to avoid:**
Keep benchmark evaluation logic outside the classifier, record exceptions separately from truth, and add contradiction checks across many parts before accepting new alternates as guidance.

**Warning signs:**
One part's accuracy jumps after a tiny heuristic change while other parts stagnate or regress; explanations reference part names or one-off layout quirks.

**Phase to address:**
Phase 1 - evaluation foundation and Phase 4 - contradiction analysis

---

### Pitfall 2: Ground Truth Drift

**What goes wrong:**
`ground_truth.json` stops being a reliable baseline because it is edited to match current algorithm output instead of domain truth.

**Why it happens:**
Rewriting truth is easier than modeling acceptable alternates or disagreement history.

**How to avoid:**
Make `ground_truth.json` immutable in the workflow, validate it strictly, and store alternate acceptable outcomes in a separate history layer.

**Warning signs:**
Truth files change frequently after runs, old reports become hard to interpret, and there is no clear separation between canonical truth and reviewer judgment.

**Phase to address:**
Phase 1 - evaluation foundation

---

### Pitfall 3: Brittle Snippet Matching

**What goes wrong:**
Visually acceptable snippets are marked wrong because the center moved slightly or the crop geometry changed.

**Why it happens:**
It is tempting to compare exact coordinates because they are easy to compute, even though the user only cares that the target annotation and context are visible.

**How to avoid:**
Use tolerant snippet acceptance rules based on distance windows, expanded crop coverage, or visibility/overlap heuristics rather than exact center equality.

**Warning signs:**
Many mismatches show the right annotation in both snippets but still fail due to small coordinate offsets.

**Phase to address:**
Phase 1 - evaluation foundation

---

### Pitfall 4: Manual Review Still Too Broad

**What goes wrong:**
The new workflow still requires reviewing most rows, so the promised speed gain never materializes.

**Why it happens:**
Auto-evaluation remains too conservative or the UI still renders all items instead of only exceptions.

**How to avoid:**
Define clear auto-pass rules, explicitly mark review-needed reasons, and design the focused queue around only mismatches/ambiguity.

**Warning signs:**
`debug_report.json` exists, but the reviewer still spends roughly the same time per run and most rows remain in manual status.

**Phase to address:**
Phase 2 - focused exception workflow

---

### Pitfall 5: Contradiction History Without Structure

**What goes wrong:**
Alternate acceptable outcomes are recorded in freeform notes, making it impossible to detect conflicts across runs and parts.

**Why it happens:**
Freeform notes are easy to ship, while normalized contradiction analysis feels like extra work.

**How to avoid:**
Capture exception/history records with structured fields for accepted classification, accepted requirement corrections, rationale category, and part/run identifiers.

**Warning signs:**
The same type of disagreement is accepted differently on different parts, but there is no queryable record to surface the conflict.

**Phase to address:**
Phase 3 - structured exception/history layer and Phase 4 - contradiction analysis

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Keep history only in freeform debug notes | Fastest possible implementation | Contradictions and acceptable alternates become unqueryable | Only for a throwaway spike, not for this milestone |
| Add classifier exceptions keyed by part name | Makes one failing part pass quickly | Spreads brittle benchmark-specific logic into core reconciliation | Never |
| Recompute all snippet metadata from PDFs on every page load | Avoids persistence design up front | Slower review UI and repeated I/O cost across runs | Acceptable only until auto-evaluation rules are proven |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `ground_truth.json` | Assume every characteristic is present and well-formed | Validate schema and fail with precise errors before evaluation begins |
| `delta_packet.json` + review queue | Assume row order and counts always align without checks | Keep deterministic ordering and validate review-item / packet alignment before export |
| Exception/history storage | Mix canonical truth and reviewed alternates in the same file | Separate immutable truth from append-only history records |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Re-reading PDFs for every mismatch render | Review UI feels slow and CPU-heavy | Cache snippet metadata or derived tolerance inputs during evaluation | Breaks once repeated benchmark runs accumulate |
| Running contradiction scans over unstructured JSON blobs | Cross-run analysis becomes manual and error-prone | Normalize history records early if contradiction analysis becomes core | Breaks once many runs per part exist |
| Evaluating every row as manual by default | No noticeable speed improvement | Auto-mark conforming rows and render only exceptions | Breaks immediately against the milestone goal |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Trusting uploaded/output artifact paths without validation | Wrong-part truth or run artifacts could be compared accidentally | Resolve part/run paths deterministically and validate expected files exist |
| Treating debug-history payloads as arbitrary blobs | Corrupt or contradictory records become hard to audit | Validate history payloads with explicit schemas |
| Letting debug-only shortcuts bypass admin boundaries | Internal tooling leaks into normal review surfaces | Keep debug workflows admin-only and clearly partitioned from production review behavior |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Showing all rows even after auto-evaluation | The focused debug loop still feels like old manual review | Default to exceptions-only views with counts of auto-passed rows |
| Hiding why a row needs review | The reviewer cannot trust or correct the system efficiently | Include explicit mismatch reasons: classification mismatch, requirement mismatch, snippet tolerance failure, contradiction flag |
| Storing alternate acceptable outcomes only as prose | The reviewer cannot tell whether a similar case was already accepted elsewhere | Capture structured history alongside any narrative explanation |

## "Looks Done But Isn't" Checklist

- [ ] **Auto-evaluation:** Often missing per-row reasons for `needs_review` — verify every nonconforming row explains why it was not auto-passed
- [ ] **Exception history:** Often missing stable identifiers across part/run/item — verify records can be traced back unambiguously
- [ ] **Snippet tolerance:** Often missing realistic acceptance heuristics — verify visually correct crops do not fail on minor coordinate drift
- [ ] **Contradiction analysis:** Often missing normalized categories — verify accepted alternates can actually be compared across runs

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Benchmark overfitting | HIGH | Remove part-specific rules, replay benchmark corpus, and reintroduce only generalized evaluator/history logic |
| Ground truth drift | HIGH | Restore canonical truth from version control, replay history into a separate exceptions layer, and re-export reports |
| Brittle snippet matching | MEDIUM | Expand tolerance heuristics, regenerate affected reports, and compare before/after false mismatch rate |
| Unstructured contradiction history | MEDIUM | Backfill normalized records from existing notes/verdicts and add validation at write time |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Benchmark overfitting | Phase 1 / Phase 4 | New rules never reference a specific part and contradiction scans run across multiple parts |
| Ground truth drift | Phase 1 | Runs never auto-edit truth files; alternate outcomes land in a separate store |
| Brittle snippet matching | Phase 1 | Visually acceptable snippet cases auto-pass within tolerance |
| Manual review still too broad | Phase 2 | Most rows auto-pass and the UI defaults to exceptions only |
| Contradiction history without structure | Phase 3 / Phase 4 | Accepted alternates are queryable and contradictions can be surfaced automatically |

## Sources

- `/home/khoa2/delta-preservation/.planning/PROJECT.md`
- `/home/khoa2/delta-preservation/shop/services/review.py`
- `/home/khoa2/delta-preservation/tests/test_debug_verdicts.py`
- User-provided debugging workflow and constraints during initialization

---
*Pitfalls research for: ground-truth-assisted debug workflow*
*Researched: 2026-04-09*
