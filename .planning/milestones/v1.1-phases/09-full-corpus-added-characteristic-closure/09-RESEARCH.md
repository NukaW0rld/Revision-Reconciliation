# Phase 9: Full-Corpus Added Characteristic Closure - Research

**Researched:** 2026-04-18
**Domain:** Full-corpus added-characteristic closure, packet/evaluation accounting, matched-owner suppression, and verification-evidence refresh
**Confidence:** HIGH (grounded in current Phase 9 context, current code, the Phase 7 algorithm-only baseline, and the Phase 6 debug diagnoses)

<user_constraints>
## User Constraints

Phase 9 already has locked context in `09-CONTEXT.md`; this research translates that context into planning-ready implementation guidance.

### Locked Decisions

- **ADD-01:** Close the full current `missing_added_truth_indexes` bucket across all 9 parts with shared logic only. No part-specific overrides and no `ground_truth.json` edits.
- Preserve the Phase 6 grouped-evidence, duplicate-claim, and explained-by-match safety rails. Phase 9 may narrow or repair them, but must not revert to blunt proximity-only behavior.
- Treat `run.py` reruns plus `tests/fixtures/phase7_algorithm_only/` as the authoritative current baseline. Keep `assets/debug_report_part*.json` as frozen historical Phase 6 evidence.
- Refresh verification evidence after fixes land. Do not silently overwrite frozen historical fixtures and call that closure.
- Keep Phase 10 maintainer gating/export scope out of this phase.

### Out of Scope

- Web UI masking or debug-queue-only fixes that leave the standalone packet/evaluator contract incorrect
- Automatic ground-truth edits
- Re-baselining by weakening benchmark expectations without first fixing the underlying packet output
- Phase 10 debug exception gating or advisory surfacing

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Requirement | Research Support |
|----|-------------|------------------|
| ADD-01 | All ground-truth-added characteristics are present in current pipeline output across the 9-part debug corpus. | The accepted algorithm-only baseline still has misses on parts 1-5 and 9, but they are a mixed bucket with distinct root-cause families. |
| ADD-02 | No new false-positive added rows are introduced while closing the missing-row gap. | The existing suppressor and grouped-evidence contracts are valuable guardrails; Phase 9 should narrow owner signatures rather than remove suppression outright. |
| SNP-01 | Snippet/search evidence remains usable and tied to the real drawing annotation rather than boilerplate or unrelated neighboring content. | Evidence refresh must preserve the Phase 6 search-window and grouped-bbox behavior because added-truth claiming and benchmark evidence both depend on stable Rev B bbox ownership. |

</phase_requirements>

<architectural_responsibility_map>
## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Why it belongs there |
|------------|-------------|----------------|----------------------|
| Added-truth token/accounting correctness | `delta_preservation/evaluation/conformance.py` | `shop/services/review.py` | The evaluator decides which truth row a packet item claims; the review surface should consume that decision faithfully instead of reinterpreting it. |
| Requirement-format normalization for added rows | `delta_preservation/evaluation/conformance.py` | `delta_preservation/reconcile/normalize.py` | Phase 9 needs deterministic canonicalization shared by truth selection, not fuzzy or UI-only equivalence. |
| Added detection, grouping, and explained-by-match suppression | `delta_preservation/reconcile/classify.py` | `delta_preservation/reconcile/match.py` | The detector and suppressor already live in `classify.py`; matched-owner signatures depend on match metadata but should stay packet-facing and shared. |
| Final baseline refresh and parity documentation | `run.py` / `tests/fixtures/phase7_algorithm_only/` | `.planning/phases/06-*`, `.planning/phases/07-*` | The milestone claim is about current standalone behavior, so final evidence must come from reruns and refreshed algorithm-only fixtures. |

</architectural_responsibility_map>

<research_summary>
## Summary

Phase 9 is a closure-and-evidence phase, not a single detector tweak. The currently accepted algorithm-only baseline already says so: Part 1 `[38]`, Part 2 `[22]`, Part 3 `[19, 20]`, Part 4 `[11, 14, 15]`, Part 5 `[16, 17, 18]`, Part 9 `[42]`. The Phase 6 debug diagnoses confirm these misses are not one shared defect:

1. **Part 1 is an accounting defect, not a detection miss.** The packet already claims the canonical added row, but `_truth_match_token()` serializes it as integer `39` because the truth row still has `char_no=39`, while `_load_missing_added_truth_items()` only counts string tokens shaped like `added:<index>`. The result is a false missing-added row in maintainer/debug accounting even though the evaluator already matched it.
2. **Part 2 is a deterministic normalization miss.** The packet emits `0.635 / 0.615` while truth uses `.635 / .615`. `_normalize_requirement_text()` already strips harmless control-symbol spacing, but it does not normalize this leading-zero variation, so added-truth selection stays ambiguous even though the dimension is semantically the same.
3. **Parts 3-5 remain heterogeneous detection/grouping gaps.** The Phase 6 aggregate debug note explicitly calls out a mix of truncation and non-emission: surface-finish text shortened to `1000 Ra`, countersink or positional rows missing entirely, and plain-decimal / plain-integer annotations still not forming the canonical added row. These should be treated as a detector-family audit, not folded into the Part 9 suppressor bug.
4. **Part 9 is a real suppressor loss.** The missing `⏥ .01` row is present in the Rev B PDF text but is suppressed as “explained by” an unrelated matched annotation because the matched-owner signature sweep in `detect_added_characteristics()` pulls same-row unmatched spans within `±200 pt` into the owner's synthetic text/bbox. That turns a separate added flatness frame into a false content subset of matched char 8.

The safest planning split is therefore:

- **Plan A:** packet/evaluation accounting repair for Part 1 plus deterministic added-text normalization for Part 2.
- **Plan B:** detector/suppressor closure for the genuine detector-side misses, starting with the confirmed Part 9 suppressor bug and using the same audit pass to close Parts 3-5 without per-part conditionals.
- **Plan C:** fresh 9-part standalone rerun, refreshed algorithm-only fixtures, and Phase 06/07 evidence updates that prove the new closure without rewriting historical Phase 6 artifacts.

**Primary recommendation:** plan this phase as three waves that converge on one refreshed evidence set. Keep the early plans surgical and shared-path only; make the final wave responsible for fixture/doc refresh after the packet behavior is fixed.

</research_summary>

<standard_stack>
## Standard Stack

### Core

| Library / Tool | Purpose | Why standard here |
|----------------|---------|-------------------|
| `pytest` via `uv run pytest` | Fast regression coverage for detector, evaluator, and debug-row accounting behavior | Existing Phase 6/7 work already expresses added-row behavior through targeted tests and snapshot-backed benchmark assertions. |
| `run.py` | Authoritative standalone 9-part rerun entrypoint | Phase 9 success is defined against fresh standalone reruns, not web-export artifacts. |
| `rg` | Artifact verification and exact-string acceptance criteria | Planning and later verification rely on grep-verifiable planning/document outputs. |

### Supporting

| Artifact | Purpose | When to use |
|----------|---------|-------------|
| `.planning/phases/07-regression-tests-and-verification/07-04-ALGORITHM-ONLY-BASELINE.md` | Current accepted miss set and baseline provenance | Use to preserve or intentionally update the benchmark contract after fixes land. |
| `.planning/debug/phase06-aggregate-added-gaps.md` | Mixed-bucket diagnosis for parts 1-5 and 9 | Use to keep plan decomposition aligned to failure families instead of chasing one giant “added miss” theory. |
| `.planning/debug/phase06-part9-flatness-miss.md` | Confirmed suppressor root cause for truth index 42 | Use when narrowing the explained-by-match owner-signature construction. |
| `tests/test_phase7_benchmark.py` | Locked per-part benchmark expectations | Use to keep closure measurable and to refresh expectations only from fresh standalone evidence. |
| `tests/test_added_detection_phase6.py`, `tests/test_added_truth_selection.py`, `tests/test_debug_row_identity.py` | Existing Phase 6 regression harnesses for grouped evidence, truth claiming, and debug-row identity | Extend these rather than inventing a new Phase 9-specific harness unless a new family truly needs its own file. |

### Alternatives Considered

| Instead of | Could use | Tradeoff |
|------------|-----------|----------|
| Fixing Part 1 in evaluator/review accounting | Ignoring the false miss in review/debug surfaces | Faster, but invalid: benchmark and maintainer signals would still disagree with packet truth claims. |
| Deterministic normalization for Part 2 | Fuzzy matching or per-part alias tables | Might pass the current row, but violates the “no part-specific hacks” constraint and risks over-claiming other rows. |
| Narrowing matched-owner signatures for Part 9 | Disabling explained-by-match suppression broadly | Would likely reintroduce the Phase 6 false-positive added rows that Phase 9 is required not to regress. |
| Refreshing current evidence in place | Overwriting `assets/debug_report_part*.json` | Blurs historical Phase 6 evidence with current Phase 9 closure and breaks the existing Phase 6 asset-regression contract. |

</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Packet Truth First, Consumer Truth Second

**What:** Fix added-truth selection and token serialization in the evaluator/packet contract first, then keep debug/review consumers thin.

**Why it fits Phase 9:** Part 1 demonstrates that a packet can already be right while `missing_added_truth_indexes` is still wrong because `shop/services/review.py` reinterprets the token shape. The fix belongs in shared truth-token semantics, not in downstream suppression of the symptom.

### Pattern 2: Narrow the Owner Signature, Don’t Remove the Guardrail

**What:** Preserve explained-by-match suppression, but reduce how far a matched owner can sweep unrelated nearby spans into its synthetic annotation signature.

**Why it fits Phase 9:** The Part 9 flatness miss is caused by over-broad ownership construction, not by the existence of the suppressor itself. Removing the suppressor would likely regress ADD-02.

### Pattern 3: Use Detection-Family Audits for Heterogeneous Corpus Gaps

**What:** Treat remaining misses on Parts 3-5 as detector/grouping families (surface finish, countersink, plain decimal/plain integer, etc.) and close them through shared heuristics plus direct tests.

**Why it fits Phase 9:** The aggregate debug note already says these misses are heterogeneous. Planning should therefore include an audit-first task that groups them by detector family before code changes start, rather than hardcoding per-part exceptions.

### Pattern 4: Refresh Current Evidence Without Rewriting Historical Evidence

**What:** Keep Phase 6 historical assets intact; refresh the algorithm-only fixture set and downstream verification docs from new standalone reruns.

**Why it fits Phase 9:** The milestone claim is about current algorithm state. That should be proven by refreshed standalone outputs and documentation, while the old Phase 6 assets remain valid historical regression inputs.

### Anti-Patterns to Avoid

- **Ground-truth mutation as closure:** never edit `ground_truth.json` to make the benchmark pass.
- **UI-only masking:** do not “fix” missing-added reporting by filtering indices in the review layer while leaving packet truth claims inconsistent.
- **Suppressor amputation:** do not disable the explained-by-match suppressor globally just to surface the Part 9 flatness row.
- **Historical fixture overwrite:** do not overwrite `assets/debug_report_part*.json` and call that the refreshed baseline.

</architecture_patterns>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Treating every remaining miss as a detector problem

**What goes wrong:** Planning focuses only on `detect_added_characteristics()` and misses Part 1/Part 2, which are evaluator/accounting failures.
**How to avoid:** Separate packet/evaluation accounting and normalization from detector-side work in the plan decomposition.

### Pitfall 2: Solving Part 9 by broadening or deleting the suppressor

**What goes wrong:** The missing flatness row appears, but fragment false positives return elsewhere in the corpus.
**How to avoid:** Narrow matched-owner signature construction and prove the change with both positive and negative suppressor tests.

### Pitfall 3: Refreshing the benchmark before the packet contract is stable

**What goes wrong:** New fixtures get captured from a still-partially-broken state, locking the wrong baseline back into `tests/test_phase7_benchmark.py`.
**How to avoid:** Make the fixture/doc refresh the final wave, after the packet/evaluator and detector changes are green.

### Pitfall 4: Folding optional alias work into every path

**What goes wrong:** `↗` / `⌰` alias support gets widened across the pipeline without proving Phase 9 still needs it after the core closure work.
**How to avoid:** Treat alias support as conditional. Add it only if a post-fix rerun still shows a remaining miss or evidence mismatch that depends on those glyph forms.

</common_pitfalls>

<code_examples>
## Code Examples

### Part 1 accounting mismatch

```bash
rg -n "_truth_match_token|_load_missing_added_truth_items|ADDED_POOL_TOKEN_PREFIX" \
  delta_preservation/evaluation/conformance.py shop/services/review.py
```

Current finding: the evaluator can emit integer `matched_truth_char_no` values for canonical added rows with `char_no`, while the review-side missing-added collector only accepts string tokens shaped like `added:<index>`.

### Part 2 normalization boundary

```bash
uv run python - <<'PY'
from delta_preservation.evaluation.conformance import _normalize_requirement_text
for raw in [".635 / .615", "0.635 / 0.615", "⌖ ∅.015 D H", "⌖∅ .015 D H"]:
    print(raw, "=>", _normalize_requirement_text(raw))
PY
```

Use this to confirm whether leading-zero and harmless spacing variants normalize to the same canonical string before planning the Part 2 fix.

### Part 9 suppressor surface

```bash
rg -n "_expand_standard_added_span|_is_content_subset|matched_annotation_signatures|200\\.0|explained by an existing matched characteristic" \
  delta_preservation/reconcile/classify.py
```

Current finding: matched-owner signatures sweep same-row unmatched spans within `±200 pt`, which is broad enough to absorb the distinct block-46 `⏥ .01` frame into an unrelated matched owner.

### Final evidence refresh boundary

```bash
rg -n "part1 \\| |part9 \\| |Missing-Added" \
  .planning/phases/07-regression-tests-and-verification/07-04-ALGORITHM-ONLY-BASELINE.md \
  .planning/phases/07-regression-tests-and-verification/07-VERIFICATION.md
```

Use this to verify which artifact defines the currently accepted miss set before refreshing it from new standalone reruns.

</code_examples>

<validation_architecture>
## Validation Architecture

### Test Framework

- Framework: `pytest >=8` via `uv run pytest`
- Suggested quick command:
  `uv run pytest -q tests/test_added_detection_phase6.py tests/test_added_truth_selection.py tests/test_debug_row_identity.py tests/test_phase7_benchmark.py -x`
- Suggested full phase command:
  `uv run pytest -q tests/test_added_detection_phase6.py tests/test_added_truth_selection.py tests/test_debug_row_identity.py tests/test_phase7_benchmark.py tests/test_phase6_asset_regression.py -x`
- Standalone rerun proof:
  `uv run python run.py part1` through `uv run python run.py part9` (can be wrapped in a small shell loop during execution)

### Phase Requirements → Evidence Map

| Requirement | Current automated evidence | Gap Phase 9 must close |
|-------------|----------------------------|------------------------|
| ADD-01 | `tests/test_phase7_benchmark.py` records the accepted per-part miss ceilings; `tests/test_added_truth_selection.py` and `tests/test_debug_row_identity.py` cover current claiming/accounting behavior. | Reduce the accepted miss ceiling to zero across all 9 parts and refresh the algorithm-only evidence/docs to match. |
| ADD-02 | `tests/test_added_detection_phase6.py` already guards grouped evidence and suppressor behavior. | Keep those guards green while narrowing the owner-signature construction enough for Part 9 and any similar rows to survive correctly. |
| SNP-01 | `tests/test_phase6_asset_regression.py` and Phase 6 exclusion tests protect the search-window/title-block contract. | Preserve grouped bbox/snippet ownership during added-row closure so fresh evidence still points at the actual drawing callout. |

### Evidence Refresh Rule

- Do **not** refresh the algorithm-only fixture set until the packet/evaluator and detector-side fixes are green.
- Do **not** overwrite `assets/debug_report_part*.json`; update `tests/fixtures/phase7_algorithm_only/` and the Phase 06/07 verification docs instead.
- If `↗` / `⌰` alias support is still needed after the main closure work, add direct regression tests for the alias path before claiming it in refreshed documentation.

</validation_architecture>

<open_questions>
## Open Questions

1. **Do Parts 3-5 collapse into one shared detector-family fix or several small ones?**
   - What we know: the aggregate debug note labels them heterogeneous and names truncation/non-emission patterns.
   - Planning implication: the first detector-side task should include a read-only packet audit that groups these misses by detector family before code changes start.

2. **Is `↗` / `⌰` alias support still necessary after the packet/evaluator and suppressor fixes land?**
   - What we know: Phase 04 validation deferred this to Phase 9 only if closure still needs it.
   - Planning implication: keep it conditional; do not burn a whole plan on alias work unless fresh reruns still show an alias-shaped gap.

</open_questions>
