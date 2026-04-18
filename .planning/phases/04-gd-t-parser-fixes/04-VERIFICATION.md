---
phase: 04-gd-t-parser-fixes
verified: 2026-04-17T00:00:00Z
status: passed
score: 3/3 must-haves verified
overrides_applied: 0
re_verification: true
---

# Phase 04: GD&T Parser Fixes — Verification Report

**Phase Goal:** Fix GD&T parsing so compact single-token forms, word-form control names, and
composite multi-compartment frames are correctly captured as structured payloads instead of
falling through to `gdt_malformed_frame` errors.

**Verified:** 2026-04-17
**Status:** PASSED
**Re-verification:** Yes — Phase 04 shipped two plans and a review-fix pass but never produced
an initial `04-VERIFICATION.md`. This report is built from current code, current tests, and
current fixture evidence (not only the historical summaries).

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | GDT-01: Compact tokens (⌖∅0.35ABC, ⌓0.5A, ⏥0.2) split into structured ParsedGdtFrame without gdt_malformed_frame | VERIFIED | `test_extract_semantic_callout_gdt_parsed_compact_token_variants` — 3 compact cases pass; malformed regression confirms ⌖ A still errors |
| 2 | GDT-02: Word-form controls (circularity, runout, total runout, position, flatness, perpendicularity) normalize to Unicode equivalents; word-form and symbol-form compare equal | VERIFIED | `test_extract_semantic_callout_gdt_parsed_word_name_controls` — 5 cases including case-variation; `test_compare_semantic_callouts_gdt_word_form_and_symbol_form_report_semantic_match`; Phase 08 adds `test_classify_requirement_type_recognizes_phase4_gdt_symbols_and_word_forms` |
| 3 | GDT-03: Composite '/' GD&T frames populate `compartments` with every segment; slash-family non-GD&T inputs remain unaffected | VERIFIED | `test_extract_semantic_callout_gdt_parsed_composite_frame_preserves_all_compartments` — 2 compartments verified; slash-family guards confirm 1/8 FILLET and H7/p6 remain weld/fit |

**Score:** 3/3 must-haves verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `delta_preservation/reconcile/normalize.py` | _GDT_WORD_CONTROL_MAP, _normalize_gdt_word_controls, _GDT_CONTROL_MAP extensions, compact path in _parse_gdt_frame, composite detection, _GDT_ANCHOR_RE with ○⌿⟃ | VERIFIED | All present in current file; Phase 08 extended _GDT_ANCHOR_RE to include ○⌿⟃ |
| `delta_preservation/types.py` | GdtCompartment model, compartments field on GdtSemanticPayload | VERIFIED | GdtCompartment(BaseModel) with 4 fields; backward-compatible default |
| `delta_preservation/reconcile/semantic_compare.py` | Compartment-aware _compare_gdt with null-tolerance guard | VERIFIED | WR-03 fix from 04-REVIEW-FIX.md applied at commit 3f72225 |
| `tests/test_semantic_extraction.py` | Word-form, compact-token, composite-frame, slash-family, dashed-datum, Phase 08 type-gate tests | VERIFIED | 72 tests pass across all targeted suites |
| `tests/test_semantic_comparison.py` | Compartment-count mismatch, composite match, single-compartment regression, null-tolerance guard | VERIFIED | Passes in current suite |
| `tests/test_semantic_types.py` | GdtCompartment default-factory test | VERIFIED | Included in 72-test count |
| `tests/test_pipeline_semantic_packet.py` | Integration packet serialization with compartments field | VERIFIED | Updated at 04-02 to include "compartments": [] |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `classify.py` | `normalize.py` | `classify_requirement_type()` | WIRED | Phase 08 updated the gate to recognize ○⌿⟃; callers in classify.py use the same function |
| `match.py` | `normalize.py` | `classify_requirement_type()` | WIRED | match.py calls classify_requirement_type() for type-based filtering |
| `_extract_semantic_payload` | `_normalize_gdt_word_controls` | pre-parse normalization | WIRED | Called before _parse_gdt_frame dispatch in normalize.py |
| `_parse_gdt_frame` | `_split_compact_gdt_remainder` | single-token compact path | WIRED | len(tokens)==1 guard routes to compact path; whitespace path unchanged |
| `_compare_gdt` | `GdtCompartment.compartments` | ordered compartment key comparison | WIRED | Count check before field-level diff per 04-02-SUMMARY.md pattern |

---

## Evidence from 04-REVIEW-FIX.md

Phase 04 shipped follow-up review fixes in `04-REVIEW-FIX.md` that are part of the current
codebase and must be reflected in this verification. The three fixes are:

**WR-01 (commit 8cd18a7):** `_split_compact_gdt_remainder` now uses `_COMPACT_DATUM_RE` regex
to extract datum refs, preserving dashed compound refs like `A-B` as single tokens. Regression
tests `test_extract_semantic_callout_gdt_compact_dashed_datum_preserved` and
`test_extract_semantic_callout_gdt_compact_vs_whitespace_dashed_datum_match` pin this behavior.

**WR-02 (commit 8cd18a7):** `_normalize_gdt_word_controls` now uses `_GDT_WORD_RE` (word-boundary
regex) instead of a `.find()` loop, preventing false rewrites inside substrings like "positioning".
Regression tests `test_extract_semantic_callout_positioning_hole_not_gdt_malformed` and
`test_extract_semantic_callout_flatnessness_not_gdt_malformed` pin this behavior.

**WR-03 (commit 3f72225):** `_normalize_gdt_tolerance_text` now accepts `str | None` with an
early-return None guard, preventing crashes when schema-valid `GdtCompartment(tolerance_text=None)`
payloads are passed to `_compare_gdt`. Regression test
`test_compare_semantic_callouts_gdt_compartment_with_null_tolerance_does_not_crash` pins this.

---

## Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| GDT-01 | 04-01, 08-01 | Compact tokens split into structured ParsedGdtFrame | SATISFIED | `test_extract_semantic_callout_gdt_parsed_compact_token_variants`; `tests/test_phase7_regression.py` GDT-01 cluster (9 tests pass) |
| GDT-02 | 04-01, 08-01 | Word-form controls normalize to Unicode; word-form = symbol-form semantically | SATISFIED | `test_extract_semantic_callout_gdt_parsed_word_name_controls`; `test_compare_semantic_callouts_gdt_word_form_and_symbol_form_report_semantic_match`; `test_classify_requirement_type_recognizes_phase4_gdt_symbols_and_word_forms`; downstream type gate aligned in Phase 08 |
| GDT-03 | 04-02, 08-01 | Composite frames preserve all compartments; weld/fit slashes unaffected | SATISFIED | `test_extract_semantic_callout_gdt_parsed_composite_frame_preserves_all_compartments`; `test_extract_semantic_callout_gdt_parsed_composite_frame_word_normalized_variant`; slash-family guards |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase 04 regression suites | `uv run pytest -q tests/test_semantic_extraction.py tests/test_semantic_comparison.py tests/test_semantic_types.py tests/test_pipeline_semantic_packet.py -x` | 72 passed in 0.12s | PASS |
| Phase 07 GDT cluster tests | `uv run pytest -q tests/test_phase7_regression.py -x` | 9 passed in 0.02s | PASS |
| ○⌿⟃ type-gate alignment | `classify_requirement_type("○ 0.002")` / `"⌿ .025 A-B"` / `"⟃ .015 B"` | all return "gdt" | PASS |
| Word-form gate alignment | `classify_requirement_type("Circularity 0.002")` | returns "gdt" | PASS |
| Non-GDT stability | `classify_requirement_type("H7/p6")` / `"1/8 FILLET BOTH SIDES"` | "fit" / "weld" | PASS |

---

## Gaps Summary

### Verified gaps (in scope, closed by Phase 08)

- **Stale type gate for ○⌿⟃:** `classify_requirement_type()` previously returned "other" for
  the Phase 04 parser-produced GD&T symbols. Fixed in Plan 08-01 by expanding `_GDT_ANCHOR_RE`
  and adding word-form normalization before the gate. Now verified passing.

### Deferred gaps (out of scope for Phase 04 / Phase 08)

- **↗ / ⌰ corpus-symbol aliases:** `assets/part8/ground_truth.json` uses `↗` and `⌰` for
  runout/total runout while the current parser maps word-form controls to `⌿` and `⟃`. This
  alias gap is deferred to Phase 09 / ADD-01. See `04-VALIDATION.md ## Phase 8 Reconciliation`
  for the full rationale.

---

## Human Verification Required

None. All three GDT requirements are verifiable from the automated test suite and command outputs
documented above.

---

## Phase Closure Statement

**VERIFICATION PASSED — Re-verification Complete**

Phase 04 shipped compact-token parsing (GDT-01), word-form normalization (GDT-02), and composite
compartment capture (GDT-03) in Plans 04-01 and 04-02, with three review fixes (WR-01/02/03)
applied at commit 8cd18a7 and 3f72225. Phase 08 closed the one residual downstream debt item
(the stale type gate for ○⌿⟃). All three requirements are now verified against current code,
current tests, and current evidence.

The corpus-symbol alias gap (↗/⌰) is an adjacent but distinct problem assigned to Phase 09
and does not affect the correctness of the Phase 04 parser contract as verified here.

---

_Verified: 2026-04-17_
_Verifier: Claude Sonnet 4.6 (Phase 08 re-verification)_
