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

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 1 | 3 | Shifted the debug loop from manual verdict entry to deterministic packet evaluation plus exception-only review |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1.0 | `uv run pytest -q` green at close-out | Not tracked | 0 |

### Top Lessons (Verified Across Milestones)

1. Keep milestone verification artifacts current during execution, not only at archive time.
2. Preserve canonical truth and introduce new debug behavior as additive layers rather than mutating baseline data.
