# Phase 10: Debug Exception Gating and Advisory Surfacing - Research

**Researched:** 2026-04-18
**Domain:** Review-service debug gating, packet-native advisory surfacing, signed export consistency, and maintainer-facing sign-off enforcement
**Confidence:** HIGH (grounded in current Phase 10 context, the v1.1 milestone audit, current review/export code paths, and the existing debug-queue regression suite)

<user_constraints>
## User Constraints

Phase 10 already has locked context in `10-CONTEXT.md`; this research translates
that context into planning-ready implementation guidance.

### Locked Decisions

- **D-01 / D-02:** Unresolved debug blockers are the existing review-service
  exception set: packet rows where `evaluation.status == "review_needed"` plus
  synthetic missing-added-truth rows without a saved debug verdict. Conforming
  rows and accepted-history-backed alternates are not blockers.
- **D-03:** Sign-off, audit packet export, and work-order export must consult
  one shared service-level unresolved-exception contract. Template-only disabled
  buttons and `run.status == "signed_off"` checks are not sufficient.
- **D-04:** If an acknowledgement bypass exists it must be durable and
  export-visible. Phase 10 does not need that extra path to satisfy the roadmap;
  a hard block is the safer default.
- **D-05 / D-06:** `DeltaItem.confidence_flags` is the authoritative advisory
  source. Maintainer surfaces and exports must render those packet-native flags
  directly rather than reconstructing them from `reasons`, mismatch codes, or
  semantic summaries.
- Preserve the current maintainer-only workflow boundary: no multi-user
  collaboration features, no ground-truth mutation, and no classifier-semantics
  changes in this phase.

### Recommended Decisions For Planning

- **No acknowledgement bypass in Phase 10.** Hard blocking closes GAP-01/GAP-02
  without introducing a new persistence model for acknowledgements.
- **Keep post-sign-off debug editing available for admins, but snapshot the
  signed debug/export state at sign-off time.** This preserves the current
  maintainer debug loop while ensuring signed exports continue to represent the
  exact advisory/gating state that was cleared at sign-off.
- **Use `ReviewItem.id` as the join key for advisory and debug export state.**
  `char_no` is not safe because duplicate and `None` characteristic numbers are
  already supported in the debug export contract.

### Out of Scope

- New classifier heuristics, ground-truth evaluation changes, or Phase 11 live
  web E2E automation
- Replacing the existing debug verdict model with a new acknowledgement system
- Freezing admin debug editing after sign-off, unless execution proves snapshot
  capture is not enough to keep signed exports stable

</user_constraints>

<phase_requirements>
## Phase Requirements

| Source | Requirement / Gap | Research Support |
|--------|--------------------|------------------|
| GAP-01 | Unresolved debug exceptions must block sign-off and signed export creation. | `build_run_debug_summary()` already computes `unresolved_exception_count` and `debug_report_ready`, but `attempt_sign_off()` and `_get_signed_run()` do not consume that contract. |
| GAP-02 / FLOW-02 | `confidence_flags` must appear on maintainer-facing review/debug/status/export surfaces. | Packet items already persist `confidence_flags`, but the review templates and export renderers ignore them. |
| CLS-01 | The Phase 5 adjacency-bleed advisory must stay packet-native and visible to maintainers. | The exact user-facing string is already persisted in `delta_packet.json`; Phase 10 only needs to surface it. |
| ADD-01 / ADD-02 / SNP-01 | Missing-added and other review-needed exceptions must participate in one shared blocker path through review, sign-off, and exports. | The existing debug queue already synthesizes missing-added rows and tracks saved debug verdicts; the integration gap is at the sign-off/export boundary. |
| VER-01 | Signed artifacts must preserve the same evidence the maintainer cleared during review. | Current export generation is split between stored PDFs and live CSV/PDF rendering, so signed export consistency needs an explicit snapshot contract. |

</phase_requirements>

<architectural_responsibility_map>
## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Why it belongs there |
|------------|-------------|----------------|----------------------|
| Packet-native advisory extraction keyed by review row | `shop/services/review.py` | `shop/routers/review.py` | Review services already own packet-to-review-row pairing and avoid duplicate/`None` `char_no` collapse. |
| Maintainer-facing advisory rendering | `shop/templates/review/_item_card.html`, `shop/templates/review/_item_card_debug.html`, `shop/templates/runs/status.html` | review router context | These surfaces already consume review-service state; Phase 10 should keep them declarative. |
| Shared sign-off/export gate | `shop/services/review.py` | `shop/routers/review.py`, `shop/routers/exports.py` | The unresolved-exception contract already exists in service form. The gate should be a service preflight consumed by both routers. |
| Signed debug/export snapshot persistence | `shop/services/review.py` | `Run.packet_versions` JSON + `output_dir/packets/` | The project already versions signed audit packet files in `packet_versions`; Phase 10 can extend that JSON entry rather than introduce a new table. |
| Audit/work-order export rendering from signed state | `shop/services/exports.py` | `shop/templates/exports/*.html` | CSV/PDF exports are already generated centrally here and can merge signed snapshot data with review items. |

</architectural_responsibility_map>

<research_summary>
## Summary

Phase 10 is an integration-contract phase, not a classifier phase. The main
runtime pieces already exist:

1. **The unresolved debug contract already exists in review services.**
   `build_debug_queue_state()` and `build_run_debug_summary()` already collect
   `review_needed` packet rows, synthetic missing-added rows, resolved verdict
   counts, and `debug_report_ready`.
2. **The sign-off and export entry points do not use that contract.**
   `attempt_sign_off()` only checks `run.status`, then flips to `signing_off`
   and generates the audit packet. `sign_off_confirm()` only blocks on pending
   review decisions. `_get_signed_run()` only checks `run.status == "signed_off"`.
3. **The advisory source already exists in packet data.**
   `DeltaItem.confidence_flags` is persisted by the pipeline and carried through
   raw packet items plus `assemble_debug_report_payload()`, but the standard
   review item card, debug card, run status page, audit packet, and work-order
   exports do not render it.
4. **Signed artifact consistency is currently split.**
   `generate_and_store_audit_packet()` writes a versioned PDF at sign-off time,
   but `audit-packet.csv` and `work-order.*` are generated live from current DB
   rows and packet files. At the same time, admin debug verdict edits remain
   allowed after `signed_off`. Without a sign-off-time debug snapshot, later
   exports can drift away from the evidence the signed packet represents.

The safest planning split is:

- **Plan 01:** surface packet-native `confidence_flags` on normal review,
  debug, and status surfaces using `ReviewItem.id` keyed packet joins.
- **Plan 02:** introduce one sign-off/export preflight contract, expose that
  gate on the review queue, and capture a signed debug snapshot when sign-off
  succeeds.
- **Plan 03:** make audit/work-order exports render from the signed snapshot
  and include row-level advisory/debug state without turning synthetic missing
  truth rows into work-order tasks.

**Primary recommendation:** keep the current admin debug-edit loop, but make
sign-off produce a versioned signed debug snapshot (for example
`output_dir/packets/vN-debug-report.json`) and have all signed exports read
from that captured state. This closes the evidence drift without requiring a
new acknowledgement model or freezing debug edits outright.

</research_summary>

<standard_stack>
## Standard Stack

### Core

| Library / Tool | Purpose | Why standard here |
|----------------|---------|-------------------|
| FastAPI routers (`shop/routers/*.py`) | Review/sign-off/export HTTP entrypoints | The current gaps are at route enforcement and route-fed template context. |
| Jinja templates | Maintainer-facing review/status and export presentation | Phase 10 is mostly about surfacing existing service state rather than inventing new endpoints. |
| SQLAlchemy + JSON fields | Persist `Run.packet_versions` metadata | The signed packet version metadata already lives in a JSON column and can be extended without a new table. |
| `pytest` via `uv run pytest` | Fast regression coverage for sign-off, debug queue, status, and export behavior | The repo already has targeted tests for debug summaries, debug verdicts, sign-off rollback, and export CSVs. |

### Supporting

| Artifact | Purpose | When to use |
|----------|---------|-------------|
| `.planning/quick/260413-hfc-*.md` | Existing Phase 2 quick-task precedent for integrating missing-added rows into `build_run_debug_summary()` | Use as the pattern reference for keeping synthetic debug rows service-first and export-visible. |
| `tests/test_run_status_debug_summary.py` | Status-page debug summary assertions | Extend for advisory surfacing and sign-off blocker messaging. |
| `tests/test_review.py` | Sign-off and review queue behavior | Extend for unresolved-debug sign-off blocking and footer/modal gating. |
| `tests/test_debug_verdicts.py` | Admin debug queue behavior, signed-off debug editing, and debug export payload | Extend to prove advisory rendering and signed snapshot capture. |
| `tests/test_exports.py` | Signed audit/work-order export assertions | Extend for snapshot-backed export behavior and advisory/debug columns. |

### Alternatives Considered

| Instead of | Could use | Tradeoff |
|------------|-----------|----------|
| Rendering flags from `confidence_flags` | Re-deriving warnings from `reasons`, mismatches, or semantic summaries | Invalid for this phase: it can drift from packet truth and reintroduce part-specific heuristics. |
| Service-level sign-off/export gate | Template-only disabled button or `signed_off` check | Easy to bypass and already identified by GAP-01 as insufficient. |
| Signed debug snapshot in `packet_versions` + file sidecar | Freeze all admin debug edits after sign-off | Simpler consistency model, but it would remove an existing maintainer workflow affordance that the current debug tests explicitly preserve. |
| Showing synthetic missing-added rows as work-order tasks | Restricting synthetic rows to audit/debug summary surfaces | Work orders are actionable changed/added characteristics, not a second debug exception queue. Synthetic rows should stay summary-only. |

</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Service-First Gate, Template-Second Rendering

**What:** compute unresolved-exception state once in services, then feed it to
review/status/export routers and templates.

**Why it fits Phase 10:** the blocker definition already lives in
`build_run_debug_summary()`. Recomputing it in templates or in a route-local
branch would create a second contract and reintroduce the current asymmetry.

### Pattern 2: Join Advisory State By `ReviewItem.id`, Not `char_no`

**What:** whenever packet-derived advisory or debug state is merged onto review
rows, use the persisted review queue order and `ReviewItem.id`.

**Why it fits Phase 10:** `assemble_debug_report_payload()` already exists
because duplicate/`None` `char_no` values are real and cannot safely key the
signed export state.

### Pattern 3: Signed Snapshot Over Live Re-Derivation

**What:** capture the exact signed debug/export state once during sign-off and
store it alongside the versioned audit packet metadata.

**Why it fits Phase 10:** current signed exports are a mix of stored and live
artifacts. A signed snapshot keeps audit/work-order downloads aligned with the
evidence the maintainer actually cleared.

### Pattern 4: Additive Export Surfacing

**What:** add advisory/debug columns or sections to audit/work-order artifacts
without replacing the core classification, override, or semantic contract.

**Why it fits Phase 10:** `confidence_flags` are advisory context, not a
replacement for `pipeline_classification`, `reviewer_decision`, or semantic
readouts.

### Anti-Patterns to Avoid

- **UI-only gate:** disabling the sign-off button while leaving
  `attempt_sign_off()` and export routes blind to unresolved debug exceptions
- **Char-number keyed packet joins:** any advisory merge keyed only by `char_no`
  will corrupt duplicate or synthetic rows
- **Live export drift:** generating CSV/PDFs from mutable current debug verdicts
  after sign-off without a signed snapshot
- **Work-order debug leakage:** turning synthetic missing-added truth rows into
  work-order rows instead of keeping them in audit/debug summary context

</architecture_patterns>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Treating `debug_report_ready` as a debug-only concern

**What goes wrong:** the debug queue and status page know the run is blocked,
but sign-off and exports still succeed.
**How to avoid:** make sign-off and exports consume the same service-level gate
that drives the debug report readiness signal.

### Pitfall 2: Surfacing advisories from the wrong data source

**What goes wrong:** a template infers warnings from `reasons` or mismatch
codes, so the maintainer sees text that is not the packet truth.
**How to avoid:** surface `confidence_flags` directly from packet rows and
default legacy packets to `[]`.

### Pitfall 3: Losing row identity in normal review mode

**What goes wrong:** advisory flags appear on the wrong row or disappear for
duplicate/`None` characteristic numbers.
**How to avoid:** derive any normal-review advisory map from the existing packet
row order in `build_debug_queue_state(..., activate_review=False)`.

### Pitfall 4: Fixing gating without fixing signed export traceability

**What goes wrong:** sign-off becomes stricter, but later CSV/work-order
downloads still reflect mutable post-sign-off debug state rather than the
evidence that was actually signed.
**How to avoid:** capture a signed debug snapshot during sign-off and make
signed exports read from it.

</common_pitfalls>

<code_examples>
## Code Examples

### Existing unresolved-debug contract

```bash
rg -n "build_run_debug_summary|debug_report_ready|unresolved_exception_count|missing_added_truth_indexes" \
  shop/services/review.py shop/templates/runs/status.html
```

Current finding: the status page already receives a structured unresolved-debug
summary, but sign-off and export routes do not.

### Current sign-off/export blind spots

```bash
rg -n "attempt_sign_off|sign-off/confirm|_get_signed_run|signed_off" \
  shop/services/review.py shop/routers/review.py shop/routers/exports.py
```

Current finding: sign-off only checks pending review decisions, and exports only
check `run.status == "signed_off"`.

### Packet-native advisory path

```bash
rg -n "confidence_flags|assemble_debug_report_payload|packet_item" \
  delta_preservation shop/services/review.py shop/templates tests
```

Current finding: packet items and debug export payloads already carry
`confidence_flags`, but maintainer-facing templates and export renderers ignore
them.

### Signed artifact drift surface

```bash
rg -n "packet_versions|generate_and_store_audit_packet|work_order|audit_packet.csv|signed-off debug verdicts remain editable" \
  shop/services/exports.py shop/services/review.py shop/routers/exports.py tests
```

Current finding: the stored PDF is versioned at sign-off time, while CSV and
work-order exports are rendered live and signed-off debug verdicts remain
editable.

</code_examples>

<validation_architecture>
## Validation Architecture

### Test Framework

- Framework: `pytest >=8` via `uv run pytest`
- Suggested quick command:
  `uv run pytest -q tests/test_review.py tests/test_run_status_debug_summary.py tests/test_debug_verdicts.py tests/test_exports.py -x`
- Suggested phase-focused command split by wave:
  - Wave 1: `uv run pytest -q tests/test_review.py tests/test_run_status_debug_summary.py tests/test_debug_verdicts.py -k "confidence_flags or advisory" -x`
  - Wave 2: `uv run pytest -q tests/test_review.py tests/test_debug_verdicts.py tests/test_exports.py -k "sign_off or debug_exceptions or signed_snapshot" -x`
  - Wave 3: `uv run pytest -q tests/test_exports.py tests/test_review.py -k "audit_packet or work_order or signed_snapshot" -x`
- Suggested full verification command:
  `uv run pytest -q tests/test_review.py tests/test_run_status_debug_summary.py tests/test_debug_verdicts.py tests/test_exports.py tests/test_history.py tests/test_amendments.py -x`

### Phase Requirements → Evidence Map

| Requirement / Gap | Current automated evidence | Gap Phase 10 must close |
|-------------------|----------------------------|-------------------------|
| GAP-01 | `tests/test_run_status_debug_summary.py` and `tests/test_debug_verdicts.py` already prove the unresolved-debug summary and debug export payload exist. | Add sign-off/export preflight assertions so unresolved debug exceptions block the release path, not just debug export. |
| GAP-02 / FLOW-02 | `tests/test_classify_bugfixes.py` proves `confidence_flags` persist in packet items. | Add review/status/export surface assertions that the maintainer sees those exact packet-native flags. |
| VER-01 | `tests/test_review.py` covers sign-off rollback/immutability; `tests/test_exports.py` covers signed export basics. | Add signed snapshot assertions so exported artifacts stay aligned with signed review evidence. |

### Manual-Only Verification

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Audit packet PDF and work-order PDF advisory layout | GAP-02 / VER-01 | The automated suite primarily exercises CSVs and HTML-backed logic; PDF layout still needs a visual spot-check. | Sign off a representative run with at least one advisory-flagged row, download the signed audit packet PDF and work-order PDF, and verify the advisory/debug summary text is readable and attached to the correct rows. |

</validation_architecture>

<open_questions>
## Open Questions

1. **Should signed-off admin debug edits remain allowed once signed snapshot capture lands?**
   - Recommendation: yes, keep them, but make signed exports read from the
     captured snapshot so historical packet versions do not drift.
2. **How much debug state should the work order carry?**
   - Recommendation: include advisory flags plus a signed debug-summary header,
     but keep synthetic missing-added rows in the audit/debug summary surfaces
     instead of turning them into work-order actions.

</open_questions>
