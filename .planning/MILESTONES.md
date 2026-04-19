# Milestones

## v1.0 Ground Truth Debug Workflow (Shipped: 2026-04-12)

**Phases completed:** 3 phases, 8 plans, 17 tasks

**Key accomplishments:**

- Strict ground truth fixture loading with status-aware validation and explicit benchmark run failure wiring
- Per-row canonical truth evaluation for classification and Rev B requirements, including unordered added-row matching and packet serialization
- Deterministic snippet-safe-zone evaluation with final conforming verdicts and downstream mismatch exposure
- Evaluation-driven admin debug queue state with stable packet-order row identity and zero-exception redirect behavior
- Phase 2 debug exports now distinguish canonical matches from reviewer-resolved exceptions, while strict verdict persistence rejects legacy vocabulary and requires explicit re-entry under the new outcome model.
- Admins now get an auto-pass-aware status-page summary and a mismatch-first exception queue that uses the new Phase 2 verdict vocabulary without letting canonical matches block debug export readiness.
- Accepted alternate outcomes now persist as DB-backed history snapshots with reversible verdict sync and immutable-truth regression coverage
- Later runs can now auto-conform through same-part accepted alternate history while preserving canonical-truth mismatches and explicit report provenance

**Accepted tech debt:** Phase 1 and Phase 2 verification artifacts were missing at close-out; see `milestones/v1.0-MILESTONE-AUDIT.md`.

---

## v1.1 Cross-part Characteristic Matching Refinement (Shipped: 2026-04-19)

**Phases completed:** 9 phases, 32 plans
**Timeline:** 7 days (2026-04-12 → 2026-04-19)
**Commits:** 155 | **Files changed:** 272 | **LOC:** +52,151 / −3,412
**Tests:** 500 passed, 2 xfailed | **Requirements:** 12/12 satisfied

**Key accomplishments:**

- GD&T parser handles all token forms — compact concatenated frames, word-form names, and composite multi-compartment FCFs parse correctly across the 9-part corpus
- Classification false positives eliminated — adjacency bleed suppression, removed+added pair reconciliation, and asymmetric tolerance detection ship with confidence flags
- Added-characteristic detection at near-full coverage — corpus-wide added-truth claims improved from 7/35 to 33/35 with false-positive suppression and title-block exclusion
- Regression tests lock the baseline — per-cluster parametrized tests plus cross-part benchmark guard aggregate accuracy at 500 tests green
- Sign-off gating on debug exceptions — unresolved debug items and classifier advisory flags block export until explicitly cleared
- Live web workflow proven end-to-end — `/runs/new` → packet → review → debug → sign-off → export automated with real corpus assets

**Known deferrals:**
- Part 5 indexes 16+17: thread/countersink matching-layer architectural deferrals
- Part 9 truth_index 42: explained-by-match suppressor false absorption
- SUMMARY.md frontmatter convention divergence (missing `requirements_completed` field)

**Archives:** [roadmap](milestones/v1.1-ROADMAP.md), [requirements](milestones/v1.1-REQUIREMENTS.md), [audit](milestones/v1.1-MILESTONE-AUDIT.md)

---
