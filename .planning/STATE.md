---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Cross-part Characteristic Matching Refinement
status: executing
stopped_at: Completed 11-02-PLAN.md
last_updated: "2026-04-19T04:15:35.935Z"
last_activity: 2026-04-19 -- Phase --phase execution started
progress:
  total_phases: 8
  completed_phases: 8
  total_plans: 29
  completed_plans: 29
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-13)

**Core value:** Improve reconciliation accuracy across many parts through a consistent, evidence-rich debug workflow that strengthens the algorithm without teaching it part-specific hacks.
**Current focus:** Phase --phase — 11

## Current Position

Phase: --phase (11) — EXECUTING
Plan: 1 of --name
Next: Phase 11 — Ready for execution
Status: Executing Phase --phase
Last activity: 2026-04-19 -- Phase --phase execution started

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**

- Total plans completed: 16
- Average duration: 4.7 min
- Total execution time: 0.2 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | 14 min | 4.7 min |
| 02 | 3 | 73 min | 24.3 min |
| 03 | 2 | - | - |
| 05 | 5 | - | - |
| 10 | 3 | - | - |

**Recent Trend:**

- Last 5 plans: 2 min, 3 min, 9 min
- Trend: Stable

| Phase 05 P01 | 2 | 3 tasks | 5 files |
| Phase 05 P02 | 3 | 2 tasks | 2 files |
| Phase 05 P03 | 5 | 2 tasks | 2 files |
| Phase 05 P04 | 4 | 3 tasks | 3 files |
| Phase 05 P05 | 129 | 2 tasks | 1 files |
| Phase 06 P01 | 6 | 2 tasks | 5 files |
| Phase 06 P02 | 11 | 3 tasks | 3 files |
| Phase 06 P03 | 4 | 2 tasks | 3 files |
| Phase 06 P04 | 3 | 1 tasks | 1 files |
| Phase 09 P01 | 3 | 2 tasks | 4 files |
| Phase 09 P02 | 28 | 2 tasks | 2 files |
| Phase 09-full-corpus-added-characteristic-closure P03 | 11 | 2 tasks | 15 files |
| Phase 10 P01 | 5 | 2 tasks | 8 files |
| Phase 10 P02 | 9 | 2 tasks | 6 files |
| Phase 10 P03 | 7 | 2 tasks | 5 files |
| Phase 11 P01 | 4 | 2 tasks | 1 files |
| Phase 11-web-run-to-review-e2e-automation P02 | 6 | 2 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Every algorithm fix must generalize across all 9 debug-corpus parts — no per-part hacks
- Ground truth files are canonical references and must never be auto-edited by the pipeline
- Phases 4-6 fix concrete algorithm errors; Phase 7 locks in regression coverage before close
- [Phase 05-01]: Use field(default_factory=list) on internal DeltaItem dataclass to guarantee independent mutable defaults per instance
- [Phase 05-01]: Use getattr(delta_internal, 'confidence_flags', []) in CLI to tolerate legacy/fake internal objects without the attribute
- [Phase 05]: Use whitespace-bounded slash split in _BLEED_SPLIT_RE to exclude embedded fractions from bleed detection
- [Phase 05]: Bleed detection requires parse_requirement numeric token matching for anchor-bearing chunk, not string substring, to handle float formatting
- [Phase 05]: Kind-transition reason always appended even when status already changed, for audit traceability in debug reports
- [Phase 05]: CLS-03 raw-text fallback uses _ASYMMETRIC_SHAPE_RE on anchor.requirement_raw vs candidate span when no tolerance_comparison is available
- [Phase 05]: Use getattr fallback for added_bbox/added_page/added_requirement_text in reconcile_removed_added_pairs to tolerate legacy fake DeltaItem objects without CLS-02 metadata fields
- [Phase 05]: CLS-02: req_bbox centroid with balloon_bbox fallback on removed side; added_bbox union centroid on added side; same-page gate and distance <= 150 pt gate; closest-wins one-to-one pairing
- [Phase 05]: Phase 05-05: Snapshot tests use exact requirement_revB strings as lookup keys, not count caps or index offsets
- [Phase 06-01]: Thin wrappers kept in match.py so existing internal call sites work unchanged after exclusion.py extraction
- [Phase 06-01]: estimate_page_dimensions uses portrait min_width=612/min_height=792 floors matching pre-existing match.py behavior
- [Phase 06-02]: Reduce Pass 0 GD&T proximity threshold from 40pt/50pt to 12pt; content-aware post-pass suppressor handles broader suppression cases
- [Phase 06-02]: Explained-by-match suppressor requires both content ownership (normalized text subset) AND bbox containment >= 0.3 before suppressing added fragment
- [Phase 06-02]: Use getattr fallback for added_requirement_text/added_bbox in cli.py to tolerate legacy _FakeInternalDeltaItem test objects
- [Phase 06-03]: Use 100.0 pt as ADDED_TRUTH_TIEBREAK_MAX_DISTANCE_PT — comfortable radius preventing cross-cluster confusion on aerospace drawings
- [Phase 06-03]: Two-stage tie-break: bbox containment first, then nearest-center within threshold; conservative ambiguity when evidence is missing or non-unique
- [Phase 06]: Phase-6 regression harness uses exact exemplar strings and canonical centers from checked-in assets, not vague count assertions (T-06-09 / T-06-10)
- [Phase 09-01]: _truth_match_token always returns added:<index> for classification=added rows, even when truth row carries legacy char_no
- [Phase 09-01]: _load_missing_added_truth_items accepts legacy integer tokens only when they resolve to added truth rows in the fixture
- [Phase 09-01]: Leading-zero stripping restricted to 0.<digits> tokens so whole-number integer parts remain unchanged
- Replace ±200 pt same-row sweep in matched-owner signatures with fixed-point _spans_are_annotation_companions walk
- Add ⌵ countersink symbol to GDT_ANCHOR_SYMBOLS so Pass 0 handles countersink callouts
- Surface-finish seeds use same-block vertical sweep (≤25 pt) to capture non-numeric prefix rows
- BASELINE_COUNTS for part5 set to max_missing_added=2 (indexes 16+17 are documented architectural deferrals from Plan 02)
- _MODIFIER_SPACING_RE collapses spaces between digit/decimal and GD&T material-modifier symbols (Ⓜ Ⓛ Ⓟ) so packet and Form 3 sources compare equal
- advisory_flags_by_item_id keys by ReviewItem.id (not char_no) to tolerate duplicate and None characteristic numbers
- confidence_flags rendered directly from packet data; no re-derivation from reasons or mismatch text on maintainer surfaces
- build_signoff_gate_state uses strict review_needed exception contract (build_debug_queue_state exception_items) rather than build_run_debug_summary to avoid false positives from items without evaluation
- _load_signed_debug_snapshot raises ValueError so callers can distinguish missing vs present snapshot
- _get_signed_run requires valid signed debug snapshot contract; raises 409 when snapshot unavailable
- Synthetic missing-added truth rows excluded from work-order action list via review_item_id=None filter
- Use pytest tmp_path instead of tempfile.TemporaryDirectory to keep pipeline output dirs alive across all post-run assertions in Phase 11 E2E tests
- part_number must be 'part6' in Phase 11 live run tests so GroundTruthContractError is not raised (loader normalizes to assets/part6 key)
- Use write_debug_verdicts (service-level) to clear debug exceptions in Phase 11 E2E clearance test — route path already exercised in Plan 01
- Companion seeded snapshot patches review_item_id after flush so advisory lookup by item.id resolves correctly in generate_audit_packet_csv
- Companion advisory assertion driven by seeded packet_item.confidence_flags in snapshot, not reconstructed from reasons text (T-11-04)

### Pending Todos

None yet.

### Blockers/Concerns

- None currently recorded.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260412-las | Fix ground truth evaluation failure in Docker: mount assets volume and normalize fixture key | 2026-04-12 | df8bcd3 | [260412-las-fix-ground-truth-evaluation-failure-in-d](./quick/260412-las-fix-ground-truth-evaluation-failure-in-d/) |
| 260413-hfc | Add missing added characteristics check to debug exception queue | 2026-04-13 | 33f27f4 | [260413-hfc-add-missing-added-characteristics-check-](./quick/260413-hfc-add-missing-added-characteristics-check-/) |

## Session Continuity

Last session: 2026-04-19T04:15:35.928Z
Stopped at: Completed 11-02-PLAN.md
Resume file: None

**Planned Phase:** 11 (web-run-to-review-e2e-automation) — 2 plans — 2026-04-19T02:29:54.055Z
