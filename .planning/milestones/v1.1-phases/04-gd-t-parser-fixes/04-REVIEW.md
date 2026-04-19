---
phase: 04-gd-t-parser-fixes
reviewed: 2026-04-14T19:44:25Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - delta_preservation/reconcile/normalize.py
  - delta_preservation/reconcile/semantic_compare.py
  - delta_preservation/types.py
  - tests/test_semantic_extraction.py
  - tests/test_semantic_comparison.py
  - tests/test_semantic_types.py
  - tests/test_pipeline_semantic_packet.py
findings:
  critical: 0
  warning: 3
  info: 0
  total: 3
status: issues_found
---

# Phase 04: Code Review Report

**Reviewed:** 2026-04-14T19:44:25Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Reviewed the Phase 04 GD&T parser changes in `normalize.py`, the new compartment schema in `types.py`, the comparison updates in `semantic_compare.py`, and the added regression tests. The new compact-token and word-form paths both introduced real behavioral regressions, and the new compartment API boundary can now crash on schema-valid input because the schema and comparator disagree on nullability.

## Warnings

### WR-01: Compact single-token parsing breaks dashed datum refs

**File:** `delta_preservation/reconcile/normalize.py:402-423`
**Issue:** `_split_compact_gdt_remainder()` extracts datum refs with `[ch for ch in datum_suffix if ch.isupper()]`, so compact inputs lose the existing dashed-datum behavior. I reproduced `extract_semantic_callout(["⌖∅0.35A-B"])` returning `datum_refs == ["A", "B"]`, while the long-standing whitespace path `["⌖", "∅0.35", "A-B"]` still returns `["A-B"]`. That makes semantically equivalent frames compare as changed depending only on PDF span tokenization.
**Fix:**
```python
_COMPACT_DATUM_RE = re.compile(r"[A-Z](?:-[A-Z])?")

datum_refs = _COMPACT_DATUM_RE.findall(datum_suffix)
```
Also add regression coverage in `tests/test_semantic_extraction.py:374-416` for compact `A-B` inputs and a comparison test that compact-vs-whitespace forms still match semantically.

### WR-02: Word-form normalization rewrites substrings inside unrelated words

**File:** `delta_preservation/reconcile/normalize.py:347-373`
**Issue:** `_normalize_gdt_word_controls()` uses `lowered.find(word)` in a loop without word boundaries, so any substring match is rewritten. I reproduced `extract_semantic_callout(["positioning hole"])` becoming a malformed GD&T frame and `extract_semantic_callout(["flatnessness 0.1"])` becoming `gdt_malformed_frame` instead of a non-GD&T/empty result. Because `_extract_semantic_payload()` treats any string result from `_parse_gdt_frame()` as authoritative GDT failure (`normalize.py:601-628`), these false positives block the normal weld/surface/fit/empty fallbacks.
**Fix:**
```python
_GDT_WORD_RE = re.compile(
    r"\\b(" + "|".join(re.escape(word) for word, _ in _GDT_WORD_CONTROL_MAP) + r")\\b",
    re.IGNORECASE,
)

def _normalize_gdt_word_controls(text: str) -> str:
    lookup = {word: symbol for word, symbol in _GDT_WORD_CONTROL_MAP}
    return _GDT_WORD_RE.sub(lambda m: lookup[m.group(1).lower()], text)
```
Add negative regression tests in `tests/test_semantic_extraction.py` for `"positioning hole"` and `"flatnessness 0.1"` to assert they do not produce `gdt_malformed_frame`.

### WR-03: New compartment schema allows nulls, but `_compare_gdt()` crashes on them

**File:** `delta_preservation/types.py:80-86`, `delta_preservation/reconcile/semantic_compare.py:121-128`
**Issue:** `GdtCompartment.tolerance_text` is declared `Optional[str]`, but `_compare_gdt()` unconditionally sends each compartment value into `_normalize_gdt_tolerance_text()`, which immediately calls `.startswith()`. Reproducing `compare_semantic_callouts()` with schema-valid `GdtCompartment(tolerance_text=None, ...)` raises `AttributeError: 'NoneType' object has no attribute 'startswith'`. This is a new phase-04 API boundary bug and is currently untested; the added tests only exercise fully populated compartments (`tests/test_semantic_comparison.py:491-564`) and the default empty-list case (`tests/test_semantic_types.py:326-336`).
**Fix:**
```python
def _normalize_gdt_tolerance_text(value: str | None) -> str | None:
    if value is None:
        return None
    ...
```
Or, if null compartments are not valid, make `GdtCompartment.control_type` and `tolerance_text` required instead of `Optional[...]`, then add a regression test that deserialized compartment payloads cannot crash comparison.

---

_Reviewed: 2026-04-14T19:44:25Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
