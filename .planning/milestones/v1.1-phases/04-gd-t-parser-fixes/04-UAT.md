---
status: complete
phase: 04-gd-t-parser-fixes
source: [04-01-SUMMARY.md, 04-02-SUMMARY.md]
started: 2026-04-14T00:00:00Z
updated: 2026-04-14T12:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Full Test Suite Passes
expected: Run `pytest` from the repo root. All 63 tests pass with no failures, errors, or unexpected skips. The suite covers word-form normalization, compact-token splitting, composite compartment parsing, compartment-aware comparison, slash-family regression guards, and pipeline integration tests.
result: issue
reported: "2 failed, 320 passed, 2 xfailed — test_baseline_upgrade_handles_repo_db_state (shop.db already at 0002, test expects ≤0001) and test_loader_uses_exact_fixture_key_without_fallback (DID NOT RAISE GroundTruthContractError)"
severity: major
root_cause: "Both failures were stale tests contradicting intentional post-phase behavior: (1) shop.db upgraded to 0002 in phase 03 but test precondition still checked ≤0001; (2) loader.py normalized keys to lowercase in commit df8bcd3 but test still expected strict-case error. Fixed: alembic precondition loosened; loader test renamed and inverted to assert normalization works."
resolved: true

### 2. Word-Form GD&T Controls Normalize to Symbols
expected: Spelled-out control names ("circularity", "runout", "total runout", "position", "flatness", "perpendicularity") are converted to their Unicode GD&T symbols before frame parsing. A callout like "circularity 0.05" should produce a parsed GDT payload (not gdt_malformed_frame) with the correct control type and tolerance value.
result: pass
verified_by: pytest tests/test_semantic_extraction.py -k word (2 passed)

### 3. Compact Single-Token GD&T Frame Splits Correctly
expected: A compact whitespace-free FCF token like ⌖∅0.35ABC splits into a structured ParsedGdtFrame with control=position, diam_prefix present, tolerance=0.35, and datum_refs=[A,B,C]. ⌓0.5A and ⏥0.2 also parse correctly. None of these produce gdt_malformed_frame.
result: pass
verified_by: pytest tests/test_semantic_extraction.py (25 passed)

### 4. Composite GD&T Frame Captures All Compartments
expected: A composite frame like "⌖0.1A / ⌖0.05AB" produces a GdtSemanticPayload with compartments=[{control_type, tolerance_text, datum_refs, modifiers}, ...] containing two compartments. Each compartment's fields match the respective segment's values.
result: pass
verified_by: pytest tests/test_semantic_extraction.py tests/test_semantic_types.py (32 passed)

### 5. Compartment Count Mismatch Detected as Change
expected: Comparing a single-compartment GDT callout against a two-compartment version of the same control produces a "semantic GD&T changed: compartments 1 → 2" reason fragment (or similar), not a match. The delta item is correctly flagged as changed.
result: pass
verified_by: pytest tests/test_semantic_comparison.py (18 passed)

### 6. Slash Fractions Not Misclassified as Composite GD&T
expected: Callouts containing slash fractions like "1/8 FILLET" (weld) or "H7/p6" (fit class) are NOT split as composite GD&T frames. They remain in their original form and produce the correct non-GDT semantic type (weld or fit), not gdt_malformed_frame or a spurious compartments list.
result: pass
verified_by: pytest tests/test_semantic_extraction.py (slash-family regression guards included)

### 7. Existing Single-Compartment GDT Payloads Backward-Compatible
expected: All pre-existing GDT callouts (single-compartment, whitespace-tokenized path) still produce correct GdtSemanticPayload with compartments=[] (empty list). No previously-passing semantic comparisons regress. Pipeline integration JSON output includes "compartments": [] on GDT payloads.
result: pass
verified_by: pytest tests/test_pipeline_semantic_packet.py (11 passed)

## Summary

total: 7
passed: 6
issues: 1
pending: 0
skipped: 0

## Gaps

[none yet]
