# Phase 4: GD&T Parser Fixes - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the analysis.

**Date:** 2026-04-13
**Phase:** 04-gd-t-parser-fixes
**Mode:** assumptions
**Areas analyzed:** GDT-01 Compact Token Splitting, GDT-02 Word-Name Normalization, GDT-03 Composite Frame Capture, Entry Point and Test Patterns

## Assumptions Presented

### GDT-01: Compact Token Splitting

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Fix must extend `_parse_gdt_frame` to regex-split remainder string into diameter-prefix + tolerance + datum-ref chars | Confident | `_GDT_TOLERANCE_RE` line 353 has `$` anchor rejecting suffix chars; token combination at line 735 only fires on separate next token; datum-ref regex line 364 matches individual letters only |

Alternative considered: Pre-processing normalization step in `_normalize_semantic_text` inserting spaces before datum-ref chars — rejected due to risk of corrupting non-compact uppercase contexts (e.g., `Ⓜ` modifier).

### GDT-02: Word-Name Normalization

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Add word-to-symbol mapping in normalize.py, applied before `_parse_gdt_frame` | Confident | `_GDT_CONTROL_MAP` lines 331-340 maps only Unicode symbols; grep for "circularity"/"runout" returns zero matches in production code; no preprocessing exists |

Alternative considered: Normalize inside `_parse_gdt_frame` itself — rejected because it mixes symbol resolution and parsing structure, making word-map behavior harder to test in isolation.

### GDT-03: Composite Frame Capture

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Split on `/` before parsing, return both compartments; extend `GdtSemanticPayload` with compartments field; update `_compare_gdt` | Likely | `|` stripped at line 702 but `/` hits explicit error return at line 769; `GdtSemanticPayload` has no list field; compartment comparison needed in `semantic_compare.py` |

Alternative considered: Parse first compartment only, attach second as `modifiers` entry — rejected because it silently loses the refinement zone tolerance, violating ROADMAP success criterion 3.

### Entry Point and Test Patterns

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| All fixes in `normalize.py`; tests written to `extract_semantic_callout` interface in `test_semantic_extraction.py` | Confident | All GD&T parsing flows through `_parse_gdt_frame` → `_extract_semantic_payload` → `extract_semantic_callout`; no other module duplicates this logic; test pattern established at lines 18-101 |

Alternative considered: GDT-02 normalization in `xlsx.py` — rejected because it creates two inconsistent code paths (inline Form 3 text would not be normalized).

## Corrections Made

No corrections — all assumptions confirmed by user.

## External Research

None required — all three fixes fully diagnosable from codebase evidence.
