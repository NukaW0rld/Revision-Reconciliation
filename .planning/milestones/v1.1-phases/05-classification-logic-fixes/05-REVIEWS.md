---
phase: 5
reviewers: [codex]
reviewed_at: 2026-04-16T19:15:00Z
plans_reviewed: [05-01-PLAN.md, 05-02-PLAN.md, 05-03-PLAN.md, 05-04-PLAN.md, 05-05-PLAN.md]
self_cli_skipped: claude
---

# Cross-AI Plan Review — Phase 5: Classification Logic Fixes

> **Context:** Running inside Claude Code CLI; `claude` skipped for independence. `gemini`, `opencode`, `qwen`, `cursor` not installed. Single external reviewer: **Codex**.

---

## Codex Review

The live edit surface is `classify.py`, `types.py`, `cli.py`, plus existing tests in `test_classify_bugfixes.py`, `test_pipeline_semantic_packet.py`, and `test_output_formatting.py`. Overall, the phase is implementable, but 05-03 and 05-05 need tightening, and 05-01 is missing some obvious fallout.

### Plan 05-01: Wave 0 Scaffold

**Summary:** Good additive scaffold. Adding `confidence_flags` is the right kind of schema change, but the plan underestimates how many test doubles and serialization paths will notice it.

**Strengths:**
- Additive field with `default_factory=list` is the right compatibility shape for both dataclass and Pydantic.
- Wiring through CLI at the packet boundary is the correct place to persist it.
- Backward-compat intent is correct.

**Concerns:**
- [HIGH] `cli.py` will read `delta_internal.confidence_flags`, but the fake internal items in existing tests do not define that attribute.
- [MEDIUM] I did not find checked-in `debug_report_part*.json` fixtures; the compatibility test may not be runnable as planned.
- [LOW] Empty stub test classes add churn without adding protection.

**Suggestions:**
- Either update the fake internal item classes immediately or make CLI read this field with a safe default.
- Point compatibility coverage at real checked-in fixtures, or generate old-shape payloads inline in tests.
- Skip empty stubs and land concrete failing tests instead.

**Risk Assessment: MEDIUM** — technically simple, but currently under-scoped around test fallout.

---

### Plan 05-02: CLS-01 Adjacency Bleed Suppression

**Summary:** This is the strongest plan. The integration point is correct, the heuristic is concrete, and the exemplar set is useful. The main risk is over-suppressing legitimate count changes.

**Strengths:**
- Hooks the right branch in `classify.py:~733`.
- Uses structural detection instead of part-specific strings.
- Includes both positive and negative corpus exemplars.

**Concerns:**
- [HIGH] Bleed detection can flatten a real count-added change unless it also proves the anchor's own content is present in one chunk.
- [MEDIUM] The plan does not say whether the exact phrase lives in `confidence_flags`, `reasons`, or both.
- [MEDIUM] It tests slash cases, but not a legitimate slash-separated changed case.

**Suggestions:**
- Require either primary-dimension presence or strong anchor-keyword overlap in one chunk before demoting to `unchanged`.
- Put the exact phrase in `confidence_flags`; optionally mirror it in `reasons`.
- Add one "real count change with slash content" negative test.

**Risk Assessment: MEDIUM** — good design, but heuristic precision matters.

---

### Plan 05-03: CLS-03 Asymmetric Tolerance Detection

**Summary:** Necessary fix, but the ordering is not tight enough. As written, it risks missing the real failure mode and duplicating logic that already exists.

**Strengths:**
- Correctly uses tolerance kind as the primary signal.
- Correctly adds a fallback for helper tests that bypass `tolerance_comparison`.
- Test list is directionally right.

**Concerns:**
- [HIGH] If the kind check runs only after the current `tolerances_differ` path, it misses the case where absolute limits match but formatting changed; current code boosts `unchanged` on `tolerances_match` first.
- [MEDIUM] `_ASYMMETRIC_SHAPE_RE` misses common leading-decimal forms like `+.005 / -.003`.
- [MEDIUM] The plan does not explain how it coexists with the existing tolerance-sized-diff logic already in the `primary_matches` path.

**Suggestions:**
- Make kind-transition detection a pre-check inside tolerance refinement, before both `tolerances_match` and `tolerances_differ`.
- Add a test where limits are equal but kind changes.
- Broaden the fallback regex or reuse parser output instead of relying on a narrow shape regex.

**Risk Assessment: HIGH** — the implementation order can nullify the fix.

---

### Plan 05-04: CLS-02 Removed+Added Reconciliation Post-Pass

**Summary:** A post-pass is the right architectural move for CLS-02, but the geometry and type inputs are under-specified. On the current data model, that matters.

**Strengths:**
- Keeps scope inside classifier/reconciliation, not IO or web.
- One-to-one matching and hard distance gates are sensible.
- Reusing the removed row preserves stable identity.

**Concerns:**
- [HIGH] Removed items do not carry their own bbox; the plan must specify `anchor.req_bbox` centroid with balloon fallback, or pairing will be inconsistent.
- [HIGH] Added items only carry a representative `added_span`; for grouped/GD&T added items, that can be the wrong text and the wrong center for type/distance gating.
- [MEDIUM] "Mutates removed item in place" is not enough; the plan needs explicit reason/confidence updates for auditability.
- [MEDIUM] Missing tests for `req_bbox=None`, cross-page candidates, and grouped added spans.

**Suggestions:**
- Define the removed-side point source explicitly (req_bbox with balloon fallback — already in the plan implementation, but should be a stated contract).
- Classify added type from grouped/expanded text, not only `added_span.text`.
- Add explicit reconciliation reasons and, if useful, a confidence flag.
- Add tests for missing `req_bbox`, grouped GD&T added items, and same-distance tie behavior.

**Risk Assessment: MEDIUM-HIGH** — sound direction, but the current internal shape is thin for safe reconciliation.

---

### Plan 05-05: Phase-5 Regression Harness

**Summary:** Good intent, weak mechanism. This does not really prove "no previously-passing characteristic regresses," and it appears to depend on fixtures that are not in the repo.

**Strengths:**
- Explicitly tries to protect generalization across all 9 parts.
- Cheap helper robustness tests are worth having.

**Concerns:**
- [HIGH] Could not find the referenced `debug_report_part*.json` corpus in the repo (may be gitignored or runtime-generated).
- [HIGH] Whitelist caps are a brittle proxy; they can pass with bad behavior and fail with harmless improvements.
- [MEDIUM] CLS-02 gets no corpus-style regression coverage even though it is the riskiest new logic.

**Suggestions:**
- Use explicit positive/negative exemplar assertions instead of count caps where possible.
- If snapshot coverage is required, check in a stable fixture corpus or generate it deterministically in-test.
- Add at least one packet-level synthetic regression for the reconcile post-pass.

**Risk Assessment: HIGH** — weak signal and likely blocked by missing fixtures.

---

## Consensus Summary

> Single reviewer (Codex). No divergent views to reconcile. Summary reflects Codex's full assessment.

### Agreed Strengths

- Additive schema extension via `default_factory=list` is the correct backward-compatible pattern (05-01)
- CLS-01 bleed heuristic hooks the right branch and uses structural rather than part-specific detection (05-02)
- Post-pass architecture for CLS-02 is the correct separation — not embedding reconciliation inside per-anchor classification (05-04)
- Kind-based primary signal for CLS-03 is the right approach (05-03)

### Top Concerns to Address Before Execution

1. **[HIGH — 05-03]** Ordering risk: kind-transition check placed AFTER `tolerances_match` boost could be neutralized when limits match but form changes. Move kind check to run before the match/differ branches.
2. **[HIGH — 05-02]** Over-suppression risk: bleed detection demotes to `unchanged` without verifying anchor content is actually present in the bleed span. Consider requiring anchor dimension overlap in one chunk.
3. **[HIGH — 05-04]** Added-item type-gating weakness: `added_span.text` may not represent the full grouped item; grouped GD&T added items could have wrong type and center for distance gating.
4. **[HIGH — 05-05]** Fixture availability: `debug_report_part*.json` files must exist at test time. Confirm they are present in the repo or generated by CI before Plan 05 lands.
5. **[MEDIUM — 05-01]** Test-doubles fallout: fake internal DeltaItem objects in existing tests will not have `confidence_flags`; executor should audit all test mocks before claiming Plan 01 complete.
6. **[MEDIUM — 05-03]** Regex gap: `_ASYMMETRIC_SHAPE_RE` does not match leading-decimal forms like `+.005 / -.003`. Broaden pattern or use the tolerance parser output.

### Divergent Views

N/A — single reviewer.

---

*To incorporate feedback: `/gsd-plan-phase 5 --reviews`*
