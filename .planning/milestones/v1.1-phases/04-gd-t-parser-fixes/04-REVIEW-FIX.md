---
phase: 04-gd-t-parser-fixes
fixed_at: 2026-04-14T20:15:00Z
review_path: .planning/phases/04-gd-t-parser-fixes/04-REVIEW.md
iteration: 1
fix_scope: critical_warning
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 04: Code Review Fix Report

**Fixed at:** 2026-04-14T20:15:00Z
**Source review:** .planning/phases/04-gd-t-parser-fixes/04-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 3
- Fixed: 3
- Skipped: 0

## Fixed Issues

### WR-01: Compact single-token parsing breaks dashed datum refs

**Files modified:** `delta_preservation/reconcile/normalize.py`, `tests/test_semantic_extraction.py`
**Commit:** 8cd18a7
**Applied fix:** Added `_COMPACT_DATUM_RE = re.compile(r"[A-Z](?:-[A-Z])?")` and replaced the character-by-character `[ch for ch in datum_suffix if ch.isupper()]` extraction in `_split_compact_gdt_remainder()` with `_COMPACT_DATUM_RE.findall(datum_suffix)`. This preserves dashed compound refs like `"A-B"` as a single token instead of splitting them into `["A", "B"]`. Added two regression tests: one asserting `"⌖∅0.35A-B"` produces `datum_refs == ["A-B"]`, and one comparison test confirming compact and whitespace-tokenized forms produce identical datum_refs.

### WR-02: Word-form normalization rewrites substrings inside unrelated words

**Files modified:** `delta_preservation/reconcile/normalize.py`, `tests/test_semantic_extraction.py`
**Commit:** 8cd18a7
**Applied fix:** Replaced the `lowered.find(word)` loop in `_normalize_gdt_word_controls()` with a compiled word-boundary regex `_GDT_WORD_RE` and a single `re.sub()` call. The regex uses `\b...\b` anchors so substrings inside other words (e.g. `"position"` inside `"positioning"`) are not matched. Added negative regression tests for `"positioning hole"` and `"flatnessness 0.1"` asserting neither produces `gdt_malformed_frame`.

Note: WR-01 and WR-02 were committed together (hash 8cd18a7) because both changes are in the same source file (`normalize.py`) and share the same test file (`test_semantic_extraction.py`), making separate atomic commits impractical without rewriting uncommitted state.

### WR-03: New compartment schema allows nulls, but `_compare_gdt()` crashes on them

**Files modified:** `delta_preservation/reconcile/semantic_compare.py`, `tests/test_semantic_comparison.py`
**Commit:** 3f72225
**Applied fix:** Changed `_normalize_gdt_tolerance_text(value: str) -> str` to `_normalize_gdt_tolerance_text(value: str | None) -> str | None` with an early-return `if value is None: return None` guard before the `.startswith()` call. This prevents `AttributeError` when schema-valid `GdtCompartment(tolerance_text=None, ...)` payloads are passed to `_compare_gdt()`. Added a regression test `test_compare_semantic_callouts_gdt_compartment_with_null_tolerance_does_not_crash` that constructs payloads with `tolerance_text=None` on both the top-level `GdtSemanticPayload` and on `GdtCompartment` entries, verifying no exception is raised.

---

_Fixed: 2026-04-14T20:15:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
