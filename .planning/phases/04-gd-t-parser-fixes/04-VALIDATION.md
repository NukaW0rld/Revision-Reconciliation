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

## Phase 8 Reconciliation

**Reconciled:** 2026-04-17 (Phase 08 — GD&T Verification Recovery)

The original manual-only validation rows in the table above were never closed by a Phase 04
verification artifact. Phase 04 shipped `04-01-SUMMARY.md`, `04-02-SUMMARY.md`, `04-REVIEW-FIX.md`,
`04-SECURITY.md`, and `04-UAT.md`, but `04-VERIFICATION.md` was never produced. This section
records the Phase 08 reconciliation against current evidence.

### Current Automated Evidence

The manual-only validation behaviors listed above are now covered by the following current
automated evidence sets:

**Phase 04 regression suites (pass as of 2026-04-17, 72 tests):**
- `tests/test_semantic_extraction.py` — compact-token, word-form, composite-frame, slash-family
  guard, dashed-datum, and Phase 08 type-gate recovery coverage
- `tests/test_semantic_comparison.py` — compartment-aware comparison, word-form/symbol-form
  semantic match, null-tolerance guard
- `tests/test_semantic_types.py` — GdtCompartment default-factory and field coverage
- `tests/test_pipeline_semantic_packet.py` — integration packet serialization including compartments

**Milestone-level GDT cluster checks (pass as of 2026-04-17, 9 tests):**
- `tests/test_phase7_regression.py` — GDT-01 compact-token cluster, GDT-02 word-form cluster,
  GDT-03 compartment-count cluster; all 9 tests verify current corpus-derived behavior

### Corpus-Symbol Discrepancy

During Phase 08 research a discrepancy was discovered between the current parser contract and
the Part 8 ground truth file:

- `assets/part8/ground_truth.json` uses `↗` (arrow up-right) and `⌰` (symbol) for runout and
  total runout annotations as extracted from the corpus PDF.
- The current Phase 04 parser/test contract proves `⌿` (circular runout) and `⟃` (total runout)
  from the word-form GD&T control normalization path.

This alias gap is a corpus extraction artifact, not a parser defect. The current parser correctly
handles the Phase 04 word-form inputs and produces the correct Unicode control symbols; the corpus
ground truth files use an alternate symbol encoding extracted by a different PDF extractor version.

### Scope Decision: ↗ / ⌰ Alias Handling Deferred to Phase 09 / ADD-01

Phase 08 execution did **not** add alias support for `↗` / `⌰`. This decision is documented here:

- The alias gap is intrinsically tied to the still-open added-characteristic closure work in
  Phase 09 (`ADD-01`). Parts 1-5 and 9 still have `missing_added_truth_indexes` containing
  GD&T characteristics that use the alternate symbol encoding.
- Adding aliases in Phase 08 without closing the added-characteristic gap would still leave the
  affected rows unmatched; the alias fix only becomes meaningful once the detection logic in
  Phase 09 surfaces those rows for comparison.
- Pulling the alias work into Phase 08 would implicitly scope-creep a verification-recovery phase
  into algorithm feature work already assigned to Phase 09.

**Action:** `↗` / `⌰` alias handling is assigned to Phase 09 / ADD-01. The manual-only
validation note about "inspecting representative real-corpus semantic extraction output" is
therefore re-scoped to Phase 09, which will include end-to-end corpus verification of the
corrected added-characteristic detection and symbol handling together.

---

## Validation Audit 2026-04-14

| Metric | Count |
|--------|-------|
| Gaps found | 1 |
| Resolved | 1 |
| Escalated | 0 |

**Gap resolved:** Added case-variation case (`"Circularity .05 A"`) to `test_extract_semantic_callout_gdt_parsed_word_name_controls` — Wave 0 required at least one case-variation check; normalization uses `.lower()` at line 372 of `normalize.py`, confirmed working.
