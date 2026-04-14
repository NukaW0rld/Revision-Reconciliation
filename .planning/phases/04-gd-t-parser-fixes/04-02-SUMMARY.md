---
phase: 04-gd-t-parser-fixes
plan: 02
subsystem: parsing
tags: [gdt, composite-frame, compartments, semantic-comparison, regression-guard]

# Dependency graph
requires:
  - phase: 04-01
    provides: Word-form normalization and compact-token splitting that Plan 02 builds on top of
provides:
  - GdtCompartment model in types.py with control_type/tolerance_text/datum_refs/modifiers fields
  - compartments field on GdtSemanticPayload (default_factory=list, backward-compatible)
  - Composite '/' GD&T frame detection and capture in _parse_gdt_frame
  - Compartment-aware _compare_gdt: count mismatch → changed fragment, identical → match
  - Slash-family regression guards (1/8 FILLET stays weld, H7/p6 stays fit)
affects: [semantic-comparison, pipeline-packet, downstream comparison consumers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Composite GD&T detection: every '/' segment must begin with a GD&T control symbol
    - _allow_composite=False parameter prevents infinite recursion on recursive segment parsing
    - compartments=[] default means all existing single-compartment payloads need no migration
    - Compartment equality: ordered key tuples, normalize tolerance text before comparison

key-files:
  created: []
  modified:
    - delta_preservation/types.py
    - delta_preservation/reconcile/normalize.py
    - delta_preservation/reconcile/semantic_compare.py
    - tests/test_semantic_extraction.py
    - tests/test_semantic_comparison.py
    - tests/test_semantic_types.py
    - tests/test_pipeline_semantic_packet.py

key-decisions:
  - "GdtCompartment holds only the 4 comparison-relevant fields; no frame_text redundancy"
  - "_allow_composite=False on recursive calls prevents infinite loop — simpler than a separate helper"
  - "Composite split only when ALL segments start with a GD&T control symbol — weld fractions (1/8) and fit classes (H7/p6) safely excluded"
  - "_normalize_gdt_word_controls replaced all-occurrences loop to handle composite word-form frames"
  - "Integration test updated to expect compartments: [] in serialized GdtSemanticPayload"

patterns-established:
  - "Compartment comparison: count check before field-level diff for clearest reason fragment"
  - "Tolerance normalization reused from single-compartment path in compartment key function"

requirements-completed: [GDT-03]

# Metrics
duration: 20min
completed: 2026-04-14
---

# Plan 04-02: Composite GD&T Compartment Capture Summary

**Composite '/' GD&T frames now preserve every compartment as structured data, with compartment-aware semantic comparison and slash-family regression guards**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-04-14
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Added `GdtCompartment(BaseModel)` to `types.py` with 4 fields; added `compartments: List[GdtCompartment]` to `GdtSemanticPayload` with backward-compatible empty-list default
- Updated `_parse_gdt_frame` with composite detection: splits on `/` only when every stripped segment starts with a recognized GD&T control symbol; weld fractions and fit classes are excluded by design
- Fixed `_normalize_gdt_word_controls` to replace all occurrences per word (needed for composite frames with repeated word forms like "flatness 0.05 / flatness 0.01")
- Updated `_compare_gdt` for compartment-awareness: count mismatch returns `semantic GD&T changed: compartments N → M`, field-level diff returns compartment-specific fragment, identical ordered compartments count toward semantic match
- 63 tests pass across all test suites including integration and pipeline tests

## Task Commits

1. **Task 1: Types + composite parser** — `bd35170` (feat)
2. **Task 2: Comparison + regression guards** — `0386532` (feat)

## Files Created/Modified
- `delta_preservation/types.py` — GdtCompartment class, compartments field on GdtSemanticPayload
- `delta_preservation/reconcile/normalize.py` — field import, compartments on ParsedGdtFrame, composite detection in _parse_gdt_frame, all-occurrences word normalization fix
- `delta_preservation/reconcile/semantic_compare.py` — compartment-aware _compare_gdt
- `tests/test_semantic_extraction.py` — composite frame tests, slash-family regression guards
- `tests/test_semantic_comparison.py` — compartment count mismatch, composite match, single-compartment regression
- `tests/test_semantic_types.py` — compartments=[] default-factory test
- `tests/test_pipeline_semantic_packet.py` — updated expected GDT payload dicts to include compartments: []

## Decisions Made
- Used `_allow_composite=False` parameter on recursive calls instead of a separate helper to avoid code duplication
- `GdtCompartment` holds only the 4 fields needed for comparison (not frame_text) to avoid circular dependency with `_compare_gdt`
- Updated integration test fixtures rather than excluding the new field from serialization — the field belongs in the canonical JSON output

## Deviations from Plan
- Fixed `_normalize_gdt_word_controls` to replace all occurrences per word (was only replacing first occurrence). This was discovered when writing the composite word-form test — not anticipated in the plan, but a necessary correctness fix that is fully contained.

## Issues Encountered
- Integration test `test_run_pipeline_persists_semantic_callouts_in_delta_packet` and one other pipeline test had hardcoded GDT payload dicts without `compartments` — updated both to include `"compartments": []`.

## Next Phase Readiness
- All three GDT-01/02/03 requirements are now complete across Plans 04-01 and 04-02
- Phase 04 is ready for verification

---
*Phase: 04-gd-t-parser-fixes*
*Completed: 2026-04-14*
