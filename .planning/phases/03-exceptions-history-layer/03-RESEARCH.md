# Phase 03: Exceptions History Layer - Research

**Researched:** 2026-04-11  
**Domain:** Durable acceptable-alternate history and cross-run reuse on top of the Phase 2 debug workflow  
**Confidence:** HIGH

<project_constraints>
## Locked Constraints

- `ground_truth.json` remains canonical and must never be auto-edited by the review workflow.
- Phase 3 must improve reuse across later runs of the same part without teaching the classifier part-specific hacks.
- The history layer is for the solo maintainer's run -> review -> rerun loop; it does not need multi-user collaboration features.
- Cross-part contradiction analysis is explicitly deferred; this phase must leave space for it without implementing it now.
- Phase 2 behavior is already in production for this milestone: `acceptable_alternate` exists in `debug_verdicts.json`, `debug_report.json` exports `history_reference: null`, and later debug/report surfaces already distinguish canonical matches from exception resolutions.
</project_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| HIST-01 | System stores acceptable alternate outcomes in a separate exceptions/history layer instead of editing `ground_truth.json`. | Use a first-class SQLAlchemy/Alembic history model in `shop/`, not a mutation of truth fixtures or a second file beside `ground_truth.json`. |
| HIST-02 | Each exceptions/history record stores run identity, part identity, characteristic identity, reviewed outcome, and rationale. | Persist a normalized snapshot with `run_id`, `review_item_id`, `part_number`, stable truth/character keys, reviewed outcome fields, rationale, and audit timestamps. |
| HIST-03 | System can treat a previously approved acceptable alternate outcome for the same part and characteristic as conforming in a later run. | Add a conservative reuse contract that compares the current evaluated row against stored alternate outcomes for the same part and truth identity before marking the row conforming via history. |
</phase_requirements>

## Summary

Phase 2 already narrowed the debug surface and introduced the reviewer outcome vocabulary that Phase 3 needs. The repo now has three strong anchors:

1. `shop/services/review.py` persists exception resolutions in `debug_verdicts.json`, keyed by stable `ReviewItem.id`.
2. `delta_preservation/evaluation/conformance.py` computes deterministic canonical conformance and exposes `evaluation.matched_truth_char_no` for later matching.
3. `assemble_debug_report_payload()` and `build_run_debug_summary()` already reserve `history_reference` and separate canonical matches from unresolved exceptions.

The missing piece is durable, queryable history that can be reused on the next run without mutating truth fixtures. The safest brownfield design is:

- Keep durable history in the `shop` database through a new SQLAlchemy model + Alembic migration.
- Keep `delta_preservation` free of SQLAlchemy by introducing a pure data contract for approved alternates and applying it as an additive reuse step after canonical truth evaluation.
- Use a conservative matching key: same `part_number`, same truth identity (`evaluation.matched_truth_char_no` when present, otherwise a reviewed characteristic token), and same reviewed outcome fingerprint.
- Extend export/report state so a later run can be `conforming` because of accepted history while still being distinguishable from a canonical truth match.

**Primary recommendation:** Treat alternate-history reuse as an additive post-truth evaluation pass, not as a rewrite of `ground_truth.json` or an inline classifier heuristic.

## Current Brownfield Seams

### Existing integration points

- `shop/routers/review.py::save_debug_item_verdict` already validates and saves `acceptable_alternate` decisions.
- `shop/services/review.py::save_debug_verdict` writes the draft reviewer decision for each exception row.
- `delta_preservation/cli.py` runs the Phase 1 truth evaluation and serializes `DeltaItem.evaluation` into `delta_packet.json`.
- `delta_preservation/evaluation/conformance.py` is the canonical truth-matching layer and already exposes `matched_truth_char_no`.
- `shop/services/review.py::assemble_debug_report_payload` still hardcodes `history_reference: None`, which is the intended forward-compatibility slot for this phase.

### Carry-forward constraints from Phase 2

- `acceptable_alternate` currently requires only rationale. That is sufficient for current-run review, but insufficient for safe cross-run reuse. Phase 3 must store a richer reviewed outcome snapshot.
- Duplicate and `None` `char_no` rows are real cases. Any Phase 3 persistence scheme must not assume `char_no` alone is globally stable.
- The status page and `debug_report.json` already treat unresolved exceptions separately from canonical matches, so Phase 3 should add a third conforming source rather than invent a new review surface.

## Recommended Architecture

### Pattern 1: Separate DB-backed history layer

**What:** Add a dedicated SQLAlchemy model such as `AcceptedAlternateHistory` in `shop/models.py`, with an Alembic migration.

**Why:** Phase 3 needs durable, queryable records that remain separate from truth fixtures and are available to both web review logic and later evaluation reuse.

**Minimum persisted fields**

- `id`
- `run_id` - source run for audit traceability
- `review_item_id` - source review row when available
- `part_number`
- `matched_truth_char_no` - prefer the canonical token from `ItemEvaluation`
- `char_no` - packet/display characteristic number, nullable
- `reviewed_classification`
- `reviewed_requirement_revB_normalized` - normalized requirement text or `None`
- `mismatch_codes` - ordered or set-like fingerprint for conservative reuse
- `rationale`
- `created_at`
- Optional but recommended: `reviewed_by_id`, `source_debug_verdict`, `is_active`, `superseded_at`

**Key design rule:** Persist reviewed outcome data, not just "acceptable_alternate happened". Later runs need enough structure to compare current output against what was previously accepted.

### Pattern 2: Immediate sync from debug verdict save path

**What:** Synchronize the durable history row when an admin saves or updates an `acceptable_alternate` debug verdict.

**Why:** The milestone is about a tight rerun loop. Requiring a later export/sign-off step before reuse would make the workflow slower and less predictable for the single maintainer.

**Recommended behavior**

- When verdict becomes `acceptable_alternate`, upsert an active history record for that run/item.
- When verdict changes away from `acceptable_alternate`, deactivate or supersede the existing history record instead of silently leaving stale reusable state behind.
- Do not store history for `algorithm_error`.

This is slightly more stateful than a pure append-only log, but it keeps reuse immediate while preserving an audit trail if `is_active`/`superseded_at` is included.

### Pattern 3: Post-truth alternate reuse pass

**What:** After `evaluate_packet_against_truth()` returns canonical conformance results, run a second pass that checks unresolved rows against approved alternates for the same part and truth identity.

**Why:** This keeps canonical truth matching and alternate reuse distinct. `delta_preservation` still answers "does this match immutable truth?" first; Phase 3 then answers "has this exact non-canonical outcome already been accepted for this part/characteristic?"

**Recommended implementation shape**

- New pure-data helper in `delta_preservation/evaluation/` or a nearby additive module, for example `alternate_history.py`.
- New contract object for reusable alternates; avoid importing SQLAlchemy into `delta_preservation`.
- `shop` loads active history rows and passes normalized records into the evaluator or CLI orchestration path.
- If no history provider is supplied, evaluation behaves exactly as it does today.

**Conservative reuse fingerprint**

Reuse only when all of the following are true:

- `part_number` matches exactly
- truth identity matches:
  - prefer `evaluation.matched_truth_char_no`
  - fallback to stored `char_no` only when the row has no truth token and the history record was explicitly created for that same reviewable case
- current packet classification matches the stored reviewed classification
- normalized current requirement text matches the stored reviewed requirement text, when present
- current mismatch-code family matches the stored `mismatch_codes`

This avoids the dangerous shortcut of treating "same part + same char number" as blanket approval for all future mismatches on that row.

### Pattern 4: Distinguishable conforming source in evaluation/report output

**What:** Extend additive output so a later run can be conforming because of history while remaining visibly different from a canonical truth match.

**Recommended additive fields**

- In `ItemEvaluation` or a neighboring additive structure:
  - `conformance_source: "ground_truth" | "accepted_alternate"`
  - `history_reference` object or scalar id
- In `debug_report.json` rows:
  - keep `row_state`
  - emit `row_state == "acceptable_alternate"` for auto-conforming reuse via history
  - populate `history_reference` with the durable record id and basic provenance

This keeps Phase 2's explicit report semantics intact while letting later runs auto-pass through accepted history.

## Suggested Plan Split

### Plan 03-01: Design and persist the exceptions/history model

Deliver:

- new SQLAlchemy model + Alembic migration
- history sync service layered off debug verdict save/update
- validation rules for the stored reviewed outcome snapshot
- regression tests for persistence and supersede/deactivate behavior

Likely files:

- `shop/models.py`
- `alembic/versions/*.py`
- `shop/services/review.py`
- `tests/test_debug_history.py`
- possibly `shop/routers/review.py`

Primary requirements:

- HIST-01
- HIST-02

### Plan 03-02: Integrate history-backed acceptable alternates into evaluation and reports

Deliver:

- alternate-history lookup contract
- post-truth reuse pass
- `debug_report.json` / status-page history reference output
- regression tests proving a later run auto-conforms through history without mutating truth

Likely files:

- `delta_preservation/evaluation/conformance.py`
- `delta_preservation/cli.py`
- `delta_preservation/types.py`
- `shop/services/review.py`
- `tests/test_history_conformance.py`
- `tests/test_debug_internals.py`

Primary requirements:

- HIST-03

## Pitfalls Specific to Phase 3

### Pitfall 1: Reusing alternates by `char_no` alone

**What goes wrong:** A history record for one acceptable mismatch on a characteristic ends up auto-approving a different mismatch on the same characteristic later.

**Avoid by:** Matching on reviewed outcome fingerprint, not just `part_number + char_no`.

### Pitfall 2: Writing directly into `ground_truth.json`

**What goes wrong:** The canonical baseline drifts and the benchmark stops meaningfully distinguishing truth from accepted exceptions.

**Avoid by:** Keeping all Phase 3 writes in the database history layer only.

### Pitfall 3: Pulling SQLAlchemy into `delta_preservation`

**What goes wrong:** The standalone pipeline becomes coupled to the web app schema and is harder to test or run outside the web tier.

**Avoid by:** Passing a normalized alternate-history record list into the evaluator rather than querying the DB from `delta_preservation`.

### Pitfall 4: Draft verdicts becoming reusable forever

**What goes wrong:** A reviewer toggles or edits a debug verdict, but the old accepted alternate remains active and keeps auto-conforming future runs.

**Avoid by:** Making history sync idempotent and reversible when the saved debug verdict changes away from `acceptable_alternate`.

### Pitfall 5: Implementing contradiction analysis too early

**What goes wrong:** The phase sprawls into cross-part reasoning and analytics instead of delivering the narrow history-and-reuse loop.

**Avoid by:** Storing enough normalized fields for future contradiction work, but keeping lookup limited to same-part reuse only.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | `pytest` via `uv run pytest` |
| Config file | `pyproject.toml` |
| Quick run command | `uv run pytest -q tests/test_debug_history.py tests/test_history_conformance.py -x` |
| Full suite command | `uv run pytest -q` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| HIST-01 | Acceptable alternates are persisted in the DB history layer and never rewrite `ground_truth.json`. | integration | `uv run pytest -q tests/test_debug_history.py -x` | ❌ Wave 0 |
| HIST-02 | History records store run, part, characteristic, reviewed outcome, and rationale fields. | unit + integration | `uv run pytest -q tests/test_debug_history.py -x` | ❌ Wave 0 |
| HIST-03 | A later run for the same part/characteristic auto-conforms through history and emits a populated `history_reference`. | integration | `uv run pytest -q tests/test_history_conformance.py tests/test_debug_internals.py -x` | ❌ Wave 0 / ✅ existing `tests/test_debug_internals.py` |

### Sampling Rate

- Per task commit: run the narrowest history-focused smoke command for the task
- Per plan wave: `uv run pytest -q tests/test_debug_history.py tests/test_history_conformance.py tests/test_debug_internals.py -x`
- Phase gate: `uv run pytest -q`

### Wave 0 Gaps

- [ ] `tests/test_debug_history.py` - DB model, sync, and deactivation/supersede behavior
- [ ] `tests/test_history_conformance.py` - same-part reuse path and report/state integration

## Security Domain

### Applicable threat patterns

| Pattern | STRIDE | Mitigation |
|---------|--------|------------|
| Forged debug verdict or history payloads | Tampering | Keep explicit payload validation on the debug save path and validate stored outcome fields before activation. |
| Cross-run or cross-part history poisoning | Tampering / Spoofing | Require exact `part_number` and truth-identity matching before reuse. |
| Admin-route overreach | Elevation of Privilege | Preserve admin-only gating on debug verdict and report routes; do not add new public history endpoints. |
| Stale alternate history after verdict changes | Repudiation / Tampering | Make the history sync service idempotent and reversible per review item. |

## Sources

### Primary

- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`
- `.planning/research/SUMMARY.md`
- `.planning/research/ARCHITECTURE.md`
- `.planning/research/PITFALLS.md`
- `.planning/phases/02-focused-debug-workflow/02-CONTEXT.md`
- `.planning/phases/02-focused-debug-workflow/02-02-SUMMARY.md`
- `.planning/phases/02-focused-debug-workflow/02-03-SUMMARY.md`
- `shop/models.py`
- `shop/services/review.py`
- `shop/routers/review.py`
- `shop/routers/runs.py`
- `delta_preservation/types.py`
- `delta_preservation/evaluation/conformance.py`
- `delta_preservation/cli.py`

### Secondary

- `tests/test_debug_verdicts.py`
- `tests/test_debug_internals.py`
- `tests/test_run_status_debug_summary.py`
- `tests/test_debug_row_identity.py`

## Metadata

- Confidence: HIGH for brownfield seams and persistence boundary
- Biggest open planning choice: exact reviewed-outcome fingerprint used for safe reuse
- Valid for planning: until Phase 3 context or the review/evaluation contract changes

---

Phase 3 research complete. Planning can proceed without another discovery pass.
