---
phase: 06-added-characteristic-detection-and-snippet-accuracy
plan: 02
subsystem: reconcile/classify + cli
tags: [added-detection, grouped-evidence, suppression, packet-assembly, tdd]
dependency_graph:
  requires: [06-01]
  provides: [grouped-added-evidence-contract, explained-by-match-suppressor, grouped-packet-assembly]
  affects: [delta_preservation/reconcile/classify.py, delta_preservation/cli.py]
tech_stack:
  added: []
  patterns:
    - Companion-span grouping for standard added Pass 2 (mirrors GD&T Pass 0 pattern)
    - Content-aware suppression requiring both text subset AND bbox overlap
    - Backward-compatible getattr access for legacy internal items
key_files:
  created:
    - tests/test_added_detection_phase6.py
  modified:
    - delta_preservation/reconcile/classify.py
    - delta_preservation/cli.py
decisions:
  - Reduce Pass 0 GD&T proximity threshold from 40pt/50pt to 12pt; content-aware post-pass suppressor handles broader cases
  - Suppressor requires both content ownership (normalized text subset) AND bbox containment ratio >= 0.3
  - Use getattr fallback for all new internal fields in cli.py to tolerate legacy _FakeInternalDeltaItem test objects
  - Extend GDT_ANCHOR_SYMBOLS to include circularity (◎), runout (⌰), and depth (↧) symbols
metrics:
  duration_min: 11
  completed_date: "2026-04-17"
  tasks_completed: 3
  files_modified: 3
---

# Phase 06 Plan 02: Grouped Added Evidence Contract and Suppressor Summary

**One-liner:** Grouped text + union bbox are now the canonical evidence for every added row in all three detection passes, and a content-aware suppressor prevents matched annotations from leaking out as false-positive added items.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 (RED) | Add grouped-evidence and suppression tests | 3c9f21e | tests/test_added_detection_phase6.py |
| 2+3 (GREEN) | Grouped evidence + suppressor + cli.py wiring | e658cb0 | classify.py, cli.py |

## What Was Built

### Task 1: Failing tests (RED)

Created `tests/test_added_detection_phase6.py` with three test classes covering the ADD-01 and ADD-02 requirements:

- `TestGroupedAddedEvidence` — 4 tests proving `◎ ∅.045 A` and `⌖ ∅.015 D H` produce full grouped text and union bbox
- `TestExplainedByMatchSuppression` — 3 tests proving content-aware suppression fires for `.045 A` fragment but not for `⌰ .015 B` or `⏥ .01`
- `TestAddedPacketAssembly` — 3 tests proving `cli.py` uses `added_requirement_text` and `added_bbox` for requirement_revB and snippet bbox

All 10 tests failed before the production edits, confirming RED gate.

### Tasks 2+3: Production implementation (GREEN)

**`delta_preservation/reconcile/classify.py`:**

1. Extended `GDT_ANCHOR_SYMBOLS` to include `◎` (circularity), `⌰` (runout), `↗`, `⏤`, `⌴`, `↧`, `⌖✢` — Pass 0 now handles all common GD&T anchor symbols.

2. Added `_expand_standard_added_span()` helper: groups companion spans on the same row (within 10pt vertically, 200pt horizontally) for the standard Pass 2. Returns `(grouped_text, union_bbox, all_group_spans)`.

3. Pass 2 now sets `added_requirement_text=grouped_text` and `added_bbox=grouped_union_bbox` for every emitted item, making the internal contract consistent with Pass 0 (GD&T) and Pass 1 (stacked pairs).

4. Added deduplication by grouped evidence identity (`seen_grouped_evidence` set) to prevent the same callout from generating multiple added items when companion spans are individually dimension-like.

5. Added `standard_consumed_keys` tracking so grouped companion spans are not processed again as separate seed spans.

6. Reduced Pass 0 early-exit proximity threshold from 40pt to 12pt (second-row companions only). The content-aware post-pass suppressor handles broader proximity cases correctly.

7. Added the explained-by-match suppressor (ADD-02) after all three passes:
   - Builds matched annotation signatures from all `matches` (grouped text + union bbox, including companion spans)
   - `_normalize_for_suppression()`: uppercase + collapse whitespace for text comparison
   - `_is_content_subset()`: direct substring check + token subset check (requires ≥2 tokens to avoid single-char false positives)
   - `_bbox_overlap_ratio()`: containment fraction of candidate bbox within matched bbox
   - Suppression requires BOTH gates (text subset AND bbox overlap ≥ 0.3)
   - Records suppression reason string for future debug export

**`delta_preservation/cli.py`:**

- Added row Rev B bbox: uses `getattr(delta_internal, "added_bbox", None)` as the base bbox before coverage expansion (not `added_span.bbox_pdf`)
- `requirement_revB` for added rows: uses `getattr(delta_internal, "added_requirement_text", None)` first, then falls back to span-expansion text
- `snippet_rule_family`: emits `"grouped_callout"` when `added_requirement_text` contains a space OR when `added_bbox` is wider than the seed span

## Test Results

| Suite | Before | After |
|-------|--------|-------|
| Full suite | 361 passed | 371 passed |
| test_added_detection_phase6.py | 0/10 (RED) | 10/10 |
| Regressions | 0 | 0 |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Reduce GD&T Pass 0 proximity threshold from 40pt/50pt to 12pt**
- **Found during:** Task 3 — `test_proximity_alone_does_not_suppress_added_row` failed because the 40pt threshold in Pass 0 suppressed a legitimate `⏥ .01` annotation at 15pt from the matched span center.
- **Issue:** Pass 0's `is_near_matched_span(threshold=40.0)` fired before the content-aware suppressor, incorrectly removing a semantically unrelated annotation.
- **Fix:** Reduced both Pass 0 proximity checks (early anchor guard and `group_near_match`) from 40pt/50pt to 12pt. The content-aware post-pass suppressor now handles cases that require semantic understanding.
- **Files modified:** `delta_preservation/reconcile/classify.py`
- **Commit:** e658cb0

**2. [Rule 3 - Blocking] Use getattr fallback for new internal fields in cli.py**
- **Found during:** Task 3 — `test_run_pipeline_omits_revA_evidence_for_added_characteristics` failed because `_FakeInternalDeltaItem` objects in the test fixture do not have `added_requirement_text` attribute.
- **Fix:** Changed direct attribute access to `getattr(delta_internal, "added_requirement_text", None)` in the `requirement_revB` derivation path and `getattr` for `added_bbox`/`added_span` in the `snippet_rule_family` path.
- **Files modified:** `delta_preservation/cli.py`
- **Commit:** e658cb0

## Known Stubs

None. All grouped evidence fields are wired to real span data.

## Threat Flags

No new security-relevant surfaces introduced.

## Self-Check: PASSED

- `tests/test_added_detection_phase6.py` exists and contains all three test classes
- Commits 3c9f21e and e658cb0 present in git log
- 371 tests pass, 0 failures
