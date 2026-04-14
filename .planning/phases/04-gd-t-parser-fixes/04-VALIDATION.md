---
phase: 04
slug: gd-t-parser-fixes
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-14
audited: 2026-04-14
---

# Phase 04 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Plan / Wave Graph

| Plan | Wave | Depends On | Validation Focus |
|------|------|------------|------------------|
| `04-01` | 1 | — | Word-form normalization plus compact-token parsing in `normalize.py` with extraction/comparison regression coverage |
| `04-02` | 2 | `04-01` | Composite compartment capture, `GdtCompartment` schema support, and compartment-aware comparison/regression guards |

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest >=8` via `uv run pytest` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest -q tests/test_semantic_extraction.py tests/test_semantic_comparison.py -x` |
| **Full suite command** | `uv run pytest -q` |
| **Estimated quick runtime** | ~10-25 seconds |

---

## Sampling Rate

- After every task commit: run the narrowest task-local smoke command from the verification map below.
- After every plan wave:
  - Wave 1: `uv run pytest -q tests/test_semantic_extraction.py tests/test_semantic_comparison.py -x`
  - Wave 2: `uv run pytest -q tests/test_semantic_extraction.py tests/test_semantic_comparison.py tests/test_semantic_types.py tests/test_reconcile_semantic_integration.py tests/test_pipeline_semantic_packet.py -x`
- Before `/gsd-verify-work`: full suite must be green.
- Max feedback latency: target under 25 seconds for task-level smoke runs.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 04-01 | 1 | GDT-02 | T-04-01, T-04-02 | Word-form controls normalize inside `normalize.py` before GD&T parsing and compare semantically to symbol-form annotations. | unit | `uv run pytest -q tests/test_semantic_extraction.py tests/test_semantic_comparison.py -x` | ✅ existing files extended by task | ✅ green |
| 04-01-02 | 04-01 | 1 | GDT-01 | T-04-01, T-04-03 | Compact control-symbol-leading tokens split into tolerance and datum refs only inside the GD&T parser path and preserve malformed-frame errors for bad inputs. | unit | `uv run pytest -q tests/test_semantic_extraction.py -x` | ✅ existing file extended by task | ✅ green |
| 04-02-01 | 04-02 | 2 | GDT-03 | T-04-01, T-04-02 | Composite `/`-separated GD&T frames populate structured `compartments` data without stealing weld or fit slash inputs. | unit | `uv run pytest -q tests/test_semantic_extraction.py tests/test_semantic_types.py -x` | ✅ existing files extended by task | ✅ green |
| 04-02-02 | 04-02 | 2 | GDT-03 | T-04-02, T-04-03 | `_compare_gdt` treats compartment mismatches as semantic changes while single-compartment equality remains stable. | unit + integration | `uv run pytest -q tests/test_semantic_comparison.py tests/test_semantic_types.py tests/test_reconcile_semantic_integration.py tests/test_pipeline_semantic_packet.py -x` | ✅ existing files extended by task | ✅ green |

*Status: ✅ green · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] Extend `tests/test_semantic_extraction.py` with parametrized coverage for compact-token forms (`⌖∅0.35ABC`, `⌓0.5A`, `⏥0.2`).
- [x] Extend `tests/test_semantic_extraction.py` with parametrized coverage for word-name forms (`circularity`, `runout`, `total runout`, `perpendicularity`) including at least one case-variation check.
- [x] Extend `tests/test_semantic_extraction.py` with composite-frame coverage for `/`-separated GD&T compartments plus slash-family guards for `1/8 FILLET` and `H7/p6`.
- [x] Extend `tests/test_semantic_comparison.py` and `tests/test_semantic_types.py` with compartment-aware equality/change coverage.
- [x] Existing `uv run pytest` infrastructure already satisfies the Nyquist baseline; no new framework install is required.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Sampled debug-corpus parser output no longer falls back to `gdt_malformed_frame` for known compact, word-name, and composite Phase 4 patterns | GDT-01, GDT-02, GDT-03 | Requires inspecting representative real-corpus semantic extraction output, not just synthetic unit fixtures | Run a representative Phase 4 debug part through the pipeline or parser inspection tooling, capture the affected callouts, and verify the output now contains structured GD&T payloads instead of `gdt_malformed_frame` errors for the known Phase 4 token shapes. |
| Word-form Form 3 text compares equal to symbol-form drawing annotations using the actual codepoints emitted by the corpus PDF extractor | GDT-02 | Unicode codepoint choice depends on real extractor output and is risky to validate from synthetic fixtures alone | Inspect one corpus example for each new word-name mapping, confirm the symbol emitted by PDF extraction matches the normalization table, and verify the semantic comparison reports equality rather than fallback or changed. |

---

## Threat References

| Threat ID | Category | Concern |
|-----------|----------|---------|
| T-04-01 | Tampering / parser bleed | GD&T-only parsing logic accidentally captures weld or fit slash patterns, or broad regexes mis-parse non-GD&T tokens. |
| T-04-02 | Integrity | New `compartments` schema or comparator logic changes equality semantics for previously passing single-compartment GD&T payloads. |
| T-04-03 | Availability | Overly broad or backtracking-heavy regex logic degrades parser performance or turns malformed inputs into slow paths. |

---

## Validation Sign-Off

- [x] All planned work areas have automated verification commands
- [x] Sampling continuity is preserved across waves
- [x] Wave 0 gaps are explicit and bounded to semantic parser/comparison test surfaces
- [x] No watch-mode flags are used
- [x] Task-level feedback latency targets under 25 seconds
- [x] `nyquist_compliant: true` is set in frontmatter

**Approval:** 2026-04-14 — all tasks green, Wave 0 complete, 68 tests pass.

---

## Validation Audit 2026-04-14

| Metric | Count |
|--------|-------|
| Gaps found | 1 |
| Resolved | 1 |
| Escalated | 0 |

**Gap resolved:** Added case-variation case (`"Circularity .05 A"`) to `test_extract_semantic_callout_gdt_parsed_word_name_controls` — Wave 0 required at least one case-variation check; normalization uses `.lower()` at line 372 of `normalize.py`, confirmed working.
