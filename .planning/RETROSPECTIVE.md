# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — Ground Truth Debug Workflow

**Shipped:** 2026-04-12
**Phases:** 3 | **Plans:** 8 | **Sessions:** 1

### What Was Built

- Deterministic ground-truth loading and packet-side evaluation for completed benchmark runs.
- Exception-only debug review and `debug_report.json` states that distinguish canonical matches from unresolved exceptions.
- Accepted-alternate history persistence and conservative same-part reuse without mutating canonical truth.

### What Worked

- Building packet-side evaluation before touching the UI kept Phase 2 focused on presentation and review flow instead of recomputing logic.
- Additive changes to review/export payloads preserved existing behavior while enabling the new debug workflow.
- Same-part accepted-alternate reuse stayed safe because the persistence layer and evaluator contract were separated cleanly.

### What Was Inefficient

- Phase 1 and Phase 2 never produced `VERIFICATION.md`, which forced the milestone audit to fail on documentation coverage instead of product behavior.
- Manual browser verification for the new status-page and exception-queue UX was never captured during the phase work.
- Some milestone planning artifacts remained outside clean commit history until close-out, which made final staging more selective than it should have been.

### Patterns Established

- Keep canonical truth immutable and layer evaluation, review state, and accepted alternates additively around it.
- Use packet order plus `ReviewItem.id` as the durable identity contract whenever `char_no` can be duplicate or null.
- Treat accepted-alternate reuse as a post-truth pass with exact same-part fingerprint matching rather than a replacement for canonical evaluation.

### Key Lessons

1. Phase-level `VERIFICATION.md` files need to be closed while the implementation context is fresh; leaving them for milestone close-out creates avoidable audit debt.
2. The safest way to extend this system is to keep the evaluation contract explicit and additive, then let downstream UI/report code consume that state directly.

### Cost Observations

- Model mix: not tracked in artifacts
- Sessions: 1 close-out session captured here
- Notable: delivering the milestone in 3 phases worked well because each phase had a narrow contract boundary and reused the packet evaluation model

---

## Milestone: v1.1 — Cross-part Characteristic Matching Refinement

**Shipped:** 2026-04-19
**Phases:** 9 | **Plans:** 32 | **Commits:** 155

### What Was Built

- GD&T parser extended to handle compact tokens, word-form names, and composite multi-compartment FCFs across the 9-part debug corpus.
- Classification logic fixes: adjacency bleed suppression with confidence flags, removed+added pair reconciliation, asymmetric tolerance detection.
- Added-characteristic detection improved from 7/35 to 33/35 corpus-wide truth rows claimed, with shared title-block exclusion contract and explained-by-match suppression.
- Cross-part regression benchmark locking per-part accuracy ceilings at 500 tests green.
- Sign-off gating on unresolved debug exceptions with signed debug snapshot for export fidelity.
- Live `/runs/new` → packet → review → debug → sign-off → export proven by E2E integration tests using real corpus assets.

### What Worked

- **Layered phase structure**: Phases 4-6 fixed algorithm errors, Phase 7 locked the baseline, Phases 8-9 closed coverage gaps, Phases 10-11 hardened the web workflow — each layer built cleanly on the previous.
- **Algorithm-only baseline fixtures**: Checking in per-part `delta_packet.json` fixtures in `tests/fixtures/phase7_algorithm_only/` made benchmark tests fast and deterministic without requiring live pipeline runs.
- **Shared exclusion contract**: Extracting `span_is_excluded_for_annotation_search` as a single source of truth eliminated duplicate title-block logic across 4+ call sites.
- **Companion walk replacing broad sweep**: The fixed-point `_spans_are_annotation_companions` walk was a targeted fix that resolved false absorption without needing a full algorithm redesign.
- **Phase 12 self-referential design**: Defining Phase 12 as "make the audit pass" worked cleanly — the audit itself became the verification artifact.

### What Was Inefficient

- **Phase 6 scope creep into Phase 9**: Added-characteristic detection was initially scoped to Parts 8/9 but the corpus-wide gap required 6 additional plans in Phase 9, nearly doubling the detection work.
- **Verification artifact debt carried forward**: Phase 8 existed solely to retroactively create `04-VERIFICATION.md` — a Phase 4 artifact that should have been written during Phase 4.
- **Stale STATE.md metadata**: The `stopped_at`, `current_focus`, and progress fields in STATE.md accumulated garbage values from the automated tooling; manual cleanup was needed at milestone close.
- **Debug sessions left open**: Three diagnosed-but-never-resolved debug sessions accumulated in `.planning/debug/` and needed manual triage at close-out.

### Patterns Established

- Use `confidence_flags` as packet-native advisory data that travels with the packet and is rendered directly, never re-derived.
- Key review-queue lookups by `ReviewItem.id` not `char_no` to handle duplicate/None characteristic numbers.
- Gate sign-off on strict `review_needed` exception contract from `build_debug_queue_state`, not the broader summary.
- Signed debug snapshot at sign-off time becomes the single source of truth for all export artifacts.
- Canonical `added:<index>` token format for added truth rows replaces legacy integer tokens.

### Key Lessons

1. **Scope added-detection work corpus-wide from the start.** Fixing only the failing parts and then discovering the rest need work too wastes a full re-planning cycle.
2. **Close verification artifacts during execution.** Phase 8 and Phase 12 both existed solely to pay back verification debt that should have been produced in the original phase.
3. **Resolved debug sessions should be moved to `resolved/` immediately**, not left in the active directory until milestone close forces triage.
4. **The explained-by-match suppressor needs a tighter ownership model.** The Part 9 flatness false-absorption (truth_index 42) is a known limitation that will require matching-layer changes.
5. **E2E tests using real corpus assets are worth the setup cost.** The Phase 11 tests caught real wiring issues that algorithm-only fixtures could not surface.

### Cost Observations

- Model mix: not tracked in artifacts
- Sessions: multiple across 7 days
- Notable: 9 phases / 32 plans in 7 days with 155 commits; the layered phase structure kept parallelism low but throughput high

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 1 | 3 | Shifted the debug loop from manual verdict entry to deterministic packet evaluation plus exception-only review |
| v1.1 | multiple | 9 | Algorithm accuracy across 9 parts, regression benchmarks, sign-off gating, and E2E web-workflow automation |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1.0 | `uv run pytest -q` green at close-out | Not tracked | 0 |
| v1.1 | 500 passed, 2 xfailed | Not tracked | 0 |

### Top Lessons (Verified Across Milestones)

1. Keep milestone verification artifacts current during execution, not only at archive time. (v1.0 lesson; confirmed again in v1.1 where Phases 8 and 12 existed solely for retroactive verification)
2. Preserve canonical truth and introduce new debug behavior as additive layers rather than mutating baseline data. (v1.0; reinforced by v1.1 ground-truth immutability constraint)
3. Scope algorithm fixes corpus-wide from the start to avoid re-planning cycles. (v1.1)
4. E2E tests with real corpus assets catch wiring issues that unit/algorithm-only fixtures cannot. (v1.1)
