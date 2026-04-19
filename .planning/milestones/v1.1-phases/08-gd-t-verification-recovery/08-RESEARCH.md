# Phase 8: GD&T Verification Recovery - Research

**Researched:** 2026-04-17
**Domain:** Phase-gap recovery, GD&T parser verification, requirements traceability
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

No user constraints - all decisions at the agent's discretion.

</user_constraints>

<phase_requirements>
## Phase Requirements

| Requirement | Why Phase 8 owns it now |
|-------------|-------------------------|
| GDT-01 | Phase 04 shipped compact-token parsing, but the requirement is still orphaned because Phase 04 never produced a verification artifact. |
| GDT-02 | Phase 04 shipped word-form normalization, but verification never closed and the current codebase still has stale downstream type-gate behavior for parser-produced GD&T symbols. |
| GDT-03 | Phase 04 shipped composite-compartment capture and comparison, but the milestone audit still treats the requirement as orphaned because there is no substantive `04-VERIFICATION.md`. |

</phase_requirements>

<architectural_responsibility_map>
## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Rebuild missing verification artifact | API/Backend | — | Verification is a code-and-artifact truth exercise over parser, tests, and planning docs. |
| Close stale GD&T type-gate debt | API/Backend | Database/Storage | `classify_requirement_type()` feeds matching/classification decisions across the reconciliation pipeline. |
| Restore GDT requirement traceability | API/Backend | — | `.planning/REQUIREMENTS.md` and Phase 04 verification docs are the canonical milestone traceability sources. |
| Reconcile stale manual-only checks | API/Backend | — | The source of truth is current fixtures, current tests, and current parser behavior, not UI workflow state. |

</architectural_responsibility_map>

<research_summary>
## Summary

Phase 8 is a verification-recovery phase, not a new algorithm feature phase. The key failure is documentary and traceability-related: Phase 04 shipped two plans, two summaries, review-fix follow-ups, validation, security, and UAT, but it never produced `04-VERIFICATION.md`. That leaves `GDT-01`, `GDT-02`, and `GDT-03` orphaned in `.planning/v1.1-MILESTONE-AUDIT.md` even though the targeted Phase 04 regression suites pass today.

Current evidence shows the safest planning split is: first, close the one real residual code debt that still undermines verification claims; second, rebuild the verification chain and traceability from current artifacts. The residual debt is not the parser core itself. The parser can currently parse `○`, `⌿`, and `⟃` GD&T callouts through `extract_semantic_callout()`, but `classify_requirement_type()` still returns `other` for those parser-produced symbols. That stale gate is explicitly called out by the audit and should be closed before Phase 04 is re-verified.

The main scope trap is symbol-alias drift in the corpus. `assets/part8/ground_truth.json` still uses `↗` and `⌰` for runout and total runout, while the Phase 04 parser maps word-form GD&T controls to `⌿` and `⟃`. That alias gap overlaps the still-open added-characteristic closure work in Phase 9 (`ADD-01`) and should not silently expand Phase 8 unless execution proves it is required just to make truthful verification claims. Phase 8 should therefore close the stale type gate, rebuild `04-VERIFICATION.md`, and explicitly re-scope any corpus-symbol alias work that remains tied to missing added rows.

**Primary recommendation:** Split Phase 8 into two plans: one additive code/test cleanup for the stale GD&T type gate, then one documentation/traceability recovery pass that writes substantive verification evidence and updates `GDT-01/02/03` mappings.

</research_summary>

<standard_stack>
## Standard Stack

### Core

| Library / Tool | Purpose | Why standard here |
|----------------|---------|-------------------|
| `pytest` via `uv run pytest` | Re-run current Phase 04 and milestone regression suites | Existing phase verification in this repo is test-first and fixture-backed. |
| `rg` | Evidence and artifact grep checks | Existing planning/verification artifacts use exact-string acceptance criteria. |
| `gsd-tools.cjs` | Workflow-safe roadmap/frontmatter/plan verification helpers | This is the repo-local GSD CLI backing the missing `gsd-sdk` wrapper. |

### Supporting

| Artifact | Purpose | When to use |
|----------|---------|-------------|
| `.planning/phases/04-gd-t-parser-fixes/04-01-SUMMARY.md` | Records what Plan 04-01 shipped | Use when writing requirement-level evidence in `04-VERIFICATION.md`. |
| `.planning/phases/04-gd-t-parser-fixes/04-02-SUMMARY.md` | Records what Plan 04-02 shipped | Use when writing requirement-level evidence in `04-VERIFICATION.md`. |
| `.planning/phases/04-gd-t-parser-fixes/04-REVIEW-FIX.md` | Captures review fixes that landed after the original summaries | Must be incorporated so verification is based on current code, not stale pre-fix summaries. |
| `.planning/phases/07-regression-tests-and-verification/07-VERIFICATION.md` | Supplies current algorithm-only parity evidence and the authoritative fixture baseline | Use when Phase 8 cites current downstream evidence rather than re-running old snapshots. |

### Alternatives Considered

| Instead of | Could use | Tradeoff |
|------------|-----------|----------|
| Rebuilding verification from current artifacts | Writing a thin `04-VERIFICATION.md` from the old summaries only | Faster, but invalid: it would repeat the exact audit failure Phase 8 exists to close. |
| Narrow Phase 8 to docs only | Fix the stale type gate first, then verify | Slightly more work, but it prevents Phase 8 from certifying behavior the current downstream classifier still mishandles. |
| Expanding Phase 8 to all `↗` / `⌰` alias work | Deferring alias closure to Phase 9 unless Phase 8 execution proves otherwise | Keeps verification recovery bounded and aligned to the roadmap's existing ADD-01 closure phase. |

</standard_stack>

<architecture_patterns>
## Architecture Patterns

### System Architecture Diagram

```text
Phase 04 plans/summaries/review-fix
        |
        v
Current parser + classifier code
(`normalize.py`, `semantic_compare.py`, callers)
        |
        +--> Targeted regression suites
        |    (`test_semantic_extraction.py`,
        |     `test_semantic_comparison.py`,
        |     `test_semantic_types.py`,
        |     `test_pipeline_semantic_packet.py`,
        |     `test_phase7_regression.py`)
        |
        +--> Current corpus-derived evidence
        |    (`tests/fixtures/phase7_algorithm_only/`,
        |     `assets/part8/ground_truth.json`)
        |
        v
Phase 8 recovery artifacts
(`04-VALIDATION.md` reconciliation,
 `04-VERIFICATION.md`,
 `.planning/REQUIREMENTS.md` traceability)
```

### Pattern 1: Evidence-First Re-Verification

**What:** Verify the current code and tests first, then write `04-VERIFICATION.md` from those current truths.

**When to use:** Any time a phase shipped code but missed its verification artifact.

**Why it fits Phase 8:** The summaries are historical implementation notes, not sufficient verification evidence. Phase 8 must cite current code paths, current passing tests, and current fixture evidence.

### Pattern 2: Gap-Closure Split Between Code Debt and Traceability Debt

**What:** Separate any residual correctness fix from the doc/traceability rebuild.

**When to use:** The phase goal is mostly documentary, but one real code issue still undermines trustworthy verification.

**Why it fits Phase 8:** `classify_requirement_type()` still lags the expanded Phase 04 GD&T symbol set (`○`, `⌿`, `⟃`) even though the parser can emit those families. If left open, `04-VERIFICATION.md` would be certifying a partially stale downstream contract.

### Pattern 3: Explicit Deferred-Scope Handling

**What:** When a verification recovery phase discovers adjacent debt owned by another roadmap phase, document the boundary instead of absorbing it implicitly.

**When to use:** Evidence points at a larger unresolved issue (here, corpus symbol aliases tied to missing added rows) that would break milestone scope if pulled in silently.

**Why it fits Phase 8:** Part 8 ground truth uses `↗` and `⌰`, but the current Phase 04 parser coverage only proves `⌿` and `⟃`. That discrepancy overlaps the still-open Phase 9 added-row closure work and must be either explicitly handled or explicitly deferred with rationale.

### Anti-Patterns to Avoid

- **Summary-only verification:** Do not write `04-VERIFICATION.md` as a restatement of `04-01-SUMMARY.md` and `04-02-SUMMARY.md`.
- **Implicit scope creep:** Do not silently absorb all Part 8 / Part 9 added-row debt into Phase 8 because it mentions GD&T.
- **Trusting stale gates:** Do not certify Phase 04 while `classify_requirement_type()` still treats parser-produced `○`, `⌿`, and `⟃` callouts as `other`.

</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Plan/file sanity checks | Custom grep-only plan validation | `node ~/.codex/get-shit-done/bin/gsd-tools.cjs verify plan-structure` | The repo already has a schema-aware plan validator. |
| Traceability status discovery | Ad hoc manual file counting | Existing phase summaries, `07-VERIFICATION.md`, and `.planning/v1.1-MILESTONE-AUDIT.md` | The gap is already described precisely; Phase 8 should consume that evidence instead of rediscovering it. |
| New parser audit framework | A fresh fixture format or dashboard | Existing pytest suites plus current algorithm-only debug-report fixtures | The repo already has current corpus-derived evidence sources; a new framework would just delay the real close-out work. |

**Key insight:** Phase 8 needs synthesis and repair of existing evidence, not a new tooling stack.

</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Certifying the old Phase 04 state instead of the current codebase

**What goes wrong:** `04-VERIFICATION.md` cites only the original plan summaries and misses the follow-up fixes in `04-REVIEW-FIX.md`.
**Why it happens:** The missing verification artifact makes the summaries look like the obvious source of truth.
**How to avoid:** Treat the summaries as provenance only; cite current code paths, current tests, and review-fix commits in the verification report.
**Warning signs:** The draft verification file does not mention `04-REVIEW-FIX.md` or current test commands.

### Pitfall 2: Overstating what the corpus evidence proves

**What goes wrong:** Phase 8 claims full runout/total-runout corpus support even though the current parser only proves `⌿` / `⟃` symbol handling while the checked-in part-8 ground truth uses `↗` / `⌰`.
**Why it happens:** The phase is named "GD&T verification recovery", which makes adjacent GD&T debt look in-scope by default.
**How to avoid:** If execution does not add alias support, record the alias mismatch as explicitly deferred to Phase 9 / ADD-01 instead of quietly implying it was solved.
**Warning signs:** `04-VERIFICATION.md` mentions `↗` / `⌰` as verified without a corresponding code change or test.

### Pitfall 3: Updating traceability before verification is real

**What goes wrong:** `.planning/REQUIREMENTS.md` is marked complete before `04-VERIFICATION.md` exists or before the stale type gate is closed.
**Why it happens:** The planning/docs work feels administrative, so it's tempting to flip the requirement state early.
**How to avoid:** Gate traceability updates on a real `04-VERIFICATION.md` with requirement-level evidence.
**Warning signs:** GDT rows are checked off while `04-VERIFICATION.md` is still missing.

</common_pitfalls>

<code_examples>
## Code Examples

### Current stale type-gate reproduction

```bash
uv run python - <<'PY'
from delta_preservation.reconcile.normalize import classify_requirement_type
for text in ["○ 0.002", "⌿ .025 A-B", "⟃ .015 B"]:
    print(text, "=>", classify_requirement_type(text))
PY
```

Observed on 2026-04-17: all three currently return `other`, not `gdt`.

### Current parser behavior for Phase 04 parser-produced symbols

```bash
uv run python - <<'PY'
from delta_preservation.io.pdf import TextSpan
from delta_preservation.reconcile.normalize import extract_semantic_callout

def span(text: str) -> TextSpan:
    return TextSpan(text=text, bbox_pdf=(0.0, 0.0, 10.0, 5.0), font_size=10.0, block_id=0, line_id=0, span_id=0)

for text in ["○ 0.002", "⌿ .025 A-B", "⟃ .015 B"]:
    semantic = extract_semantic_callout(pdf_spans=[span(text)], form3_requirement=None)
    print(text, semantic.status.state, semantic.status.parser_family, None if semantic.gdt is None else semantic.gdt.control_type)
PY
```

Observed on 2026-04-17: all three parse as `gdt`, which proves the stale gap is downstream classification, not the main parser.

### Current corpus-symbol discrepancy check

```bash
rg -n "↗|⌰" assets/part8/ground_truth.json
```

Observed on 2026-04-17: `assets/part8/ground_truth.json` contains `↗ .025 A-B`, `↗ .035 A-B`, `⌰ .015 B`, and `⌰ .002 A`.

</code_examples>

<validation_architecture>
## Validation Architecture

### Test Framework

- Framework: `pytest` via `uv run pytest`
- Fast command: `uv run pytest -q tests/test_semantic_extraction.py tests/test_semantic_comparison.py tests/test_semantic_types.py tests/test_pipeline_semantic_packet.py -x`
- Recovery command: `uv run pytest -q tests/test_phase7_regression.py -x`
- Artifact checks: `rg -n` over Phase 04/08 planning docs and `.planning/REQUIREMENTS.md`

### Phase Requirements → Evidence Map

| Requirement | Current automated evidence | Gap Phase 8 must close |
|-------------|----------------------------|------------------------|
| GDT-01 | `tests/test_semantic_extraction.py` compact-token coverage; `tests/test_phase7_regression.py::test_cluster_gdt01_compact_token_covered` | Move from summary-only provenance to current Phase 04 verification evidence. |
| GDT-02 | `tests/test_semantic_extraction.py` word-form coverage; `tests/test_semantic_comparison.py` word-form vs symbol-form equality | Close stale downstream type gate and explicitly reconcile corpus-symbol alias scope. |
| GDT-03 | `tests/test_semantic_extraction.py` composite-frame coverage; `tests/test_semantic_comparison.py` compartment-aware comparison; `tests/test_semantic_types.py` default-factory coverage | Write a substantive verification artifact tying these current tests to the requirement and current code paths. |

### Manual vs Deferred Evidence Rule

- If the execution work does **not** add alias support for `↗` / `⌰`, Phase 8 must document those symbols as Phase 9 / ADD-01 territory instead of leaving a vague manual-only note behind.
- If execution **does** add alias support, Phase 8 must add direct tests for those aliases before claiming them in `04-VERIFICATION.md`.

</validation_architecture>

<open_questions>
## Open Questions

1. **Should Phase 8 add `↗` / `⌰` aliases or explicitly defer them?**
   - What we know: the current corpus uses `↗` / `⌰` in `assets/part8/ground_truth.json`, while the current parser/test contract proves `⌿` / `⟃`.
   - What's unclear: whether the alias gap is required just to make truthful Phase 04 verification claims, or whether it only matters for still-open added-row work already assigned to Phase 9.
   - Recommendation: treat alias support as deferred by default; only pull it into Phase 8 if execution proves the Phase 04 requirements cannot be honestly verified without it.

2. **Should `.planning/REQUIREMENTS.md` update only GDT rows or also other already-verified rows?**
   - What we know: the file currently leaves other verified rows unchecked as well.
   - What's unclear: whether updating only GDT rows is acceptable as a targeted gap closure.
   - Recommendation: update only `GDT-01`, `GDT-02`, and `GDT-03` in Phase 8 because they are the explicit orphaned requirements this phase is closing. Leave broader requirements hygiene to milestone-close work.

</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)

- `.planning/ROADMAP.md` — Phase 8 goal, dependency, and success criteria
- `.planning/REQUIREMENTS.md` — GDT-01/02/03 acceptance criteria and current traceability state
- `.planning/v1.1-MILESTONE-AUDIT.md` — orphaned requirement evidence, stale debt callouts, and FLOW-03 definition
- `.planning/phases/04-gd-t-parser-fixes/04-01-SUMMARY.md`
- `.planning/phases/04-gd-t-parser-fixes/04-02-SUMMARY.md`
- `.planning/phases/04-gd-t-parser-fixes/04-REVIEW.md`
- `.planning/phases/04-gd-t-parser-fixes/04-REVIEW-FIX.md`
- `.planning/phases/04-gd-t-parser-fixes/04-VALIDATION.md`
- `.planning/phases/04-gd-t-parser-fixes/04-UAT.md`
- `.planning/phases/07-regression-tests-and-verification/07-VERIFICATION.md`
- `delta_preservation/reconcile/normalize.py`
- `delta_preservation/reconcile/classify.py`
- `delta_preservation/reconcile/match.py`
- `tests/test_semantic_extraction.py`
- `tests/test_semantic_comparison.py`
- `tests/test_semantic_types.py`
- `tests/test_pipeline_semantic_packet.py`
- `tests/test_phase7_regression.py`
- `tests/fixtures/phase7_algorithm_only/part8-debug-report.json`
- `assets/part8/ground_truth.json`

### Secondary (MEDIUM confidence)

- `assets/debug_report_part8.json` — useful historical note about expected GD&T symbol families, but not authoritative verification evidence

</sources>

<metadata>
## Metadata

**Research scope:**
- Current Phase 04 implementation evidence
- Current stale parser/type-gate debt
- Current corpus/fixture evidence relevant to Phase 04 verification recovery
- Safe scope boundary against Phase 9 added-characteristic closure

**Confidence breakdown:**
- Artifact evidence: HIGH — directly read from checked-in docs and code
- Current runtime behavior: HIGH — reproduced locally with `uv run python` and `uv run pytest`
- Scope boundary for `↗` / `⌰`: MEDIUM — evidence is strong, but final ownership depends on execution findings

</metadata>
