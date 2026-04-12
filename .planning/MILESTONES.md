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
