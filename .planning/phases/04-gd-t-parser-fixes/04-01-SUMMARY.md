---
phase: 04-gd-t-parser-fixes
plan: 01
subsystem: parsing
tags: [gdt, normalize, regex, semantic-extraction]

# Dependency graph
requires: []
provides:
  - Word-form GD&T control name normalization (circularity, runout, total runout, position, flatness, perpendicularity → Unicode symbols)
  - Compact single-token GD&T frame splitting (⌖∅0.35ABC, ⌓0.5A, ⏥0.2 → structured ParsedGdtFrame)
  - Extended _GDT_CONTROL_MAP with circularity/circular_runout/total_runout symbol entries
affects: [04-02-composite-compartments, semantic-comparison, tests]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Word-control normalization applied before _parse_gdt_frame dispatch (not in I/O layer)
    - Compact single-token path exits _parse_gdt_frame early via _split_compact_gdt_remainder
    - Malformed-frame error path preserved for inputs with control symbol but no tolerance

key-files:
  created: []
  modified:
    - delta_preservation/reconcile/normalize.py
    - tests/test_semantic_extraction.py
    - tests/test_semantic_comparison.py

key-decisions:
  - "_normalize_gdt_word_controls wired into _extract_semantic_payload before _parse_gdt_frame, not in xlsx.py or I/O layer (D-03)"
  - "Compact path uses len(tokens)==1 guard so whitespace-tokenized path is completely unchanged"
  - "Datum suffix split into individual uppercase letters matching _GDT_DATUM_RE semantics"
  - "New symbols (○/⌿/⟃) added to _GDT_CONTROL_MAP so they survive the symbol lookup at line 710"

patterns-established:
  - "Word normalization: longest-first list match + lowered string comparison, no external regex library"
  - "Compact splitter: single fullmatch on anchored regex, returns None for non-matching remainders"

requirements-completed: [GDT-01, GDT-02]

# Metrics
duration: 15min
completed: 2026-04-14
---

# Plan 04-01: Compact & Word-Form GD&T Parsing Summary

**Word-form GD&T controls and compact whitespace-free FCF tokens now parse into structured payloads instead of falling through to gdt_malformed_frame**

## Performance

- **Duration:** ~15 min
- **Completed:** 2026-04-14
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added `_GDT_WORD_CONTROL_MAP` and `_normalize_gdt_word_controls()` for case-insensitive, longest-first substitution of spelled-out control names (circularity, runout, total runout, position, flatness, perpendicularity) to Unicode symbols
- Extended `_GDT_CONTROL_MAP` with three new entries: `○` (circularity), `⌿` (circular_runout), `⟃` (total_runout)
- Added `_GDT_COMPACT_REMAINDER_RE` and `_split_compact_gdt_remainder()` for single-pass splitting of compact remainders like `∅0.35ABC` → (diam_prefix, tolerance, datum_chars)
- Wired compact path into `_parse_gdt_frame` with an early-return for the single-token case; whitespace-tokenized path completely unaffected
- 31 tests pass including new word-form, compact-token, and cross-form semantic match tests

## Task Commits

1. **Task 1: Word-form GD&T control normalization** — `4167d61` (feat)
2. **Task 2: Compact single-token GD&T frame splitting** — `0e76c53` (feat)

## Files Created/Modified
- `delta_preservation/reconcile/normalize.py` — _GDT_WORD_CONTROL_MAP, _normalize_gdt_word_controls, _GDT_COMPACT_REMAINDER_RE, _split_compact_gdt_remainder, wired into _extract_semantic_payload and _parse_gdt_frame
- `tests/test_semantic_extraction.py` — test_extract_semantic_callout_gdt_parsed_word_name_controls, test_extract_semantic_callout_gdt_parsed_compact_token_variants
- `tests/test_semantic_comparison.py` — test_compare_semantic_callouts_gdt_word_form_and_symbol_form_report_semantic_match

## Decisions Made
- Compact path guarded by `len(tokens) == 1` to leave the multi-token whitespace path entirely unchanged
- Word normalization placed in `_extract_semantic_payload` (not in xlsx.py or I/O layer) per D-03
- `⌖ A` (control + datum, no tolerance) still returns `gdt_malformed_frame` — confirmed by regression test

## Deviations from Plan
None — plan executed exactly as written.

## Issues Encountered
A stale `assert result.reason_fragments == [...]` line appeared at the end of the comparison test file after an Edit operation — removed immediately before the test run.

## Next Phase Readiness
- Plan 04-02 (composite GD&T compartments) can proceed: the parser now correctly handles the single-compartment compact and word-form paths that are prerequisites for the multi-compartment path
- `_parse_gdt_frame` entry point is clean; the `/` slash compartment split in Plan 04-02 will slot in before the existing token loop

---
*Phase: 04-gd-t-parser-fixes*
*Completed: 2026-04-14*
