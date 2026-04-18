---
phase: 06-added-characteristic-detection-and-snippet-accuracy
verified: 2026-04-17T14:42:59Z
status: human_needed
score: 2/4 roadmap success criteria verified via code; 2 deferred to Phase 7 (end-to-end pipeline run)
overrides_applied: 0
deferred:
  - truth: "After a clean pipeline run, missing_added_truth_indexes is empty for parts 8 and 9"
    addressed_in: "Phase 7"
    evidence: "Phase 7 SC3: 'All 9 parts are re-run through the pipeline and their ground-truth evaluation results show equal or better conforming count versus the pre-fix baseline'; Phase 7 requirement VER-01 owns the full 9-part rerun"
  - truth: "Aggregate added-characteristic count across all 9 parts matches or exceeds the ground-truth added count with no false-positive increase"
    addressed_in: "Phase 7"
    evidence: "Phase 7 SC2/SC3 and VER-01 cover cross-part benchmark and full rerun; 06-VALIDATION.md explicitly defers 'Full 9-part corpus rerun' to Phase 7"
human_verification:
  - test: "Run the full pipeline on parts 8 and 9 with the Phase 6 code and inspect the output missing_added_truth_indexes field"
    expected: "missing_added_truth_indexes is empty (or at minimum reduced) for both parts after the grouped added evidence, suppressor, and tie-break logic is applied"
    why_human: "Cannot run the full pipeline without real PDF assets and the complete reconciliation stack. The regression harness pins the frozen pre-fix debug corpus state deliberately (read-only contract); it does not assert the post-fix empty-missing-indexes outcome."
  - test: "Run the full pipeline across all 9 debug corpus parts and compare aggregate added-characteristic counts against ground truth"
    expected: "Total added rows emitted >= total ground truth added rows, and no new false-positive added rows introduced versus the pre-Phase-6 baseline"
    why_human: "Cross-part aggregate count comparison requires a full pipeline run with all 9 PDF assets. Phase 6 unit tests verify individual mechanisms but not end-to-end aggregate output."
---

# Phase 6: Added Characteristic Detection and Snippet Accuracy — Verification Report

**Phase Goal:** All ground-truth-added characteristics are present in pipeline output for every part, false-positive added rows are suppressed, and title block regions are reliably excluded from search windows so snippet matches land on actual drawing annotations.
**Verified:** 2026-04-17T14:42:59Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | After a clean pipeline run, `missing_added_truth_indexes` is empty for parts 8 and 9 | VERIFIED (Phase 9) | Phase 9 Plan 03: standalone reruns at HEAD 8a611c9 confirm empty `missing_added_truth_indexes` for parts 8 and 9, and for parts 1-4 and 6-7 as well. Part 5 indexes 16+17 are architectural deferrals documented in 09-02-SUMMARY.md. The algorithm-only fixture set and benchmark baseline (`tests/test_phase7_benchmark.py`) encode the zero-miss counts. |
| 2 | A raw PDF span whose content is fully explained by an existing matched characteristic does not produce a spurious added-characteristic row | VERIFIED | `_is_content_subset()` + `_bbox_overlap_ratio()` suppressor in `classify.py:1969` — requires both text ownership AND bbox overlap >= 0.3. Covered by `TestExplainedByMatchSuppression` (3 tests) and `TestPhase6Part8Exemplars.test_fragment_only_form_does_not_survive_when_ownership_exists`. All pass. |
| 3 | No characteristic search window captures text from the title block or revision table; `snippet_outside_revA` cases caused by title-block capture are resolved | VERIFIED | `exclusion.py` exposes `span_is_excluded_for_annotation_search()` used by anchors.py (L147), match.py (L281), classify.py (L318, L1221, L1256, L1603). `TestSharedExclusionContract` covers title-block, revision-table, bottom-center boilerplate, and legitimate edge-annotation negative control. 11 tests pass. |
| 4 | Aggregate added-characteristic count across all 9 parts matches or exceeds the ground-truth added count with no false-positive increase | DEFERRED | Phase 7 SC2/SC3 and VER-01 cover cross-part benchmark and full rerun. 06-VALIDATION.md explicitly defers this to Phase 7. |

**Score (automatable truths):** 2/2 verifiable truths confirmed. 2 deferred to Phase 7.

---

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | `missing_added_truth_indexes` empty for parts 8 and 9 after pipeline run | Phase 7 → **CLOSED Phase 9** | Phase 9 Plan 03: standalone reruns at HEAD 8a611c9 confirm empty `missing_added_truth_indexes` for parts 8 and 9 (and all other parts except Part 5 deferred items). Algorithm-only fixture set in `tests/fixtures/phase7_algorithm_only/` refreshed. |
| 2 | Aggregate added count matches or exceeds ground truth across all 9 parts | Phase 7 → **CLOSED Phase 9** | Phase 9 Plans 01-03 closed the token contract, leading-zero normalization, detector-side misses, and material-modifier spacing. 8 of 9 parts now have zero missing-added rows; Part 5 has 2 architectural deferrals. |

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `delta_preservation/reconcile/exclusion.py` | Shared page-dimension estimation and annotation-search exclusion helpers | VERIFIED | 178 lines. Contains `def estimate_page_dimensions` (L97), `def is_boilerplate_candidate_text` (L69), `def span_is_excluded_for_annotation_search` (L126). |
| `delta_preservation/reconcile/anchors.py` | Rev A anchor filtering wired to shared exclusion helpers | VERIFIED | Imports from `exclusion` (L20), calls `span_is_excluded_for_annotation_search` (L147). |
| `delta_preservation/reconcile/classify.py` | Keyword rescue and added detection wired to shared exclusion helpers; grouped added-evidence contract | VERIFIED | 2090 lines. Imports exclusion (L64), calls `estimate_page_dimensions` (L313), `span_is_excluded_for_annotation_search` at 4 call sites. Contains `added_requirement_text` field, `_expand_standard_added_span()`, suppressor, `added_bbox` and `added_page` set in all three passes. |
| `delta_preservation/cli.py` | Packet assembly using grouped added evidence contract | VERIFIED | 974 lines. Uses `delta_internal.added_requirement_text` (L840) for `requirement_revB`, `getattr(delta_internal, "added_bbox", None)` for Rev B bbox, emits `snippet_rule_family="grouped_callout"` (L874, L877). |
| `delta_preservation/evaluation/conformance.py` | Deterministic duplicate added-truth tie-break using packet Rev B bbox evidence | VERIFIED | 481 lines. `ADDED_TRUTH_TIEBREAK_MAX_DISTANCE_PT = 100.0` (L31), `_coerce_packet_bbox`, `_bbox_center`, `_point_inside_bbox`, `_distance` helpers. Two-stage tie-break in `select_truth_row_for_item`. Conservative ambiguity path retained (L261). |
| `tests/test_phase6_exclusion.py` | Phase-6 unit coverage for boilerplate and title-block exclusion | VERIFIED | 283 lines. Contains `class TestSharedExclusionContract` (L51) and `class TestRescueAndAddedDetectionExclusion` (L144). |
| `tests/test_added_detection_phase6.py` | Mechanism and packet-assembly regression coverage for ADD-01/ADD-02 | VERIFIED | 566 lines. Contains `class TestGroupedAddedEvidence` (L85), `class TestExplainedByMatchSuppression` (L222), `class TestAddedPacketAssembly` (L379). |
| `tests/test_added_truth_selection.py` | Focused evaluator coverage for unique, duplicate, and ambiguous added-row selection | VERIFIED | 311 lines. Contains `class TestDuplicateAddedTruthSelection` (L97). |
| `tests/test_phase6_asset_regression.py` | Read-only Part 8/Part 9 regression harness keyed to checked-in debug artifacts and ground truth | VERIFIED | 601 lines. Contains `class TestPhase6AssetInvariants` (L62), `class TestPhase6Part8Exemplars` (L183), `class TestPhase6Part9DuplicateAddedRows` (L347). |
| `tests/test_debug_row_identity.py` | Queue-facing regression for duplicate added-truth claiming | VERIFIED | `test_duplicate_added_truth_rows_claim_distinct_indexes_from_revb_evidence` (L300) passes. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `anchors.py` | `exclusion.py` | `from delta_preservation.reconcile.exclusion import` | WIRED | L20 import, L147 call site |
| `match.py` | `exclusion.py` | `from delta_preservation.reconcile.exclusion import` | WIRED | L27 import, L281 call site (thin wrapper) |
| `classify.py` | `exclusion.py` | `from delta_preservation.reconcile.exclusion import` | WIRED | L64 import, 4 call sites (L313, L318, L1221, L1256, L1603) |
| `cli.py` | `classify.py` added evidence fields | `getattr(delta_internal, "added_requirement_text", None)` | WIRED | L835-840 in cli.py; field defined L85 in classify.py internal type |
| `cli.py` | `classify.py` added bbox field | `getattr(delta_internal, "added_bbox", None)` | WIRED | L863-877 in cli.py; `snippet_rule_family="grouped_callout"` emitted |
| `conformance.py` | packet `revB.bbox` | `item.revB.bbox` → `_coerce_packet_bbox()` | WIRED | Two-stage tie-break wired to packet bbox; verified by `TestDuplicateAddedTruthSelection` |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `classify.py` `detect_added_characteristics()` | `added_requirement_text` | `_expand_standard_added_span()` groups companion spans from real PDF spans | Yes — grouping uses actual span geometry and text | FLOWING |
| `cli.py` added row `requirement_revB` | `delta_internal.added_requirement_text` | Populated by classify.py from real PDF span data | Yes — getattr with fallback; real text when present | FLOWING |
| `conformance.py` tie-break | `truth_row.snippet_center_revB` | Ground-truth JSON loaded from `assets/part9/ground_truth.json` | Yes — loaded from checked-in fixture at runtime | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All Phase 6 test files pass | `uv run pytest tests/test_phase6_exclusion.py tests/test_added_detection_phase6.py tests/test_added_truth_selection.py tests/test_phase6_asset_regression.py` | 63 passed in 0.05s | PASS |
| Cross-file smoke (exclusion + multishift) | `uv run pytest tests/test_alignment_multishift.py::test_generate_candidates_excludes_tolerance_block_boilerplate tests/test_debug_row_identity.py::test_duplicate_added_truth_rows_claim_distinct_indexes_from_revb_evidence` | 2 passed | PASS |
| Full test suite — no regressions | `uv run pytest` | 415 passed, 2 xfailed, 0 failures | PASS |
| All Phase 6 commits present in git log | `git log --oneline --all \| grep -E "f9a5f45\|443a29d\|3c9f21e\|e658cb0\|fdc55bd\|26f88a7\|03f7be6"` | All 7 commits found | PASS |

---

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|---------------|-------------|--------|----------|
| ADD-01 | 06-02, 06-03, 06-04, 09-01, 09-02, 09-03, 09-04, 09-05 | All ground-truth-added characteristics present in pipeline output for every part | VERIFIED (Phase 9 full-corpus closure, parts 1–9 at zero missing-added) | Grouped evidence contract and duplicate tie-break mechanism implemented and unit-tested in Phase 6. End-to-end closure confirmed in Phase 9: refreshed standalone reruns at HEAD db265b5 show zero missing_added_truth_indexes for all 9 parts including Part 5. Plans 04+05 closed Part 5 indexes 16 (zone-aware boilerplate filter) and 17 (dimensional-incompatibility guard + CLS-02 threshold tightening). Updated Phase 7 algorithm-only baseline in tests/fixtures/phase7_algorithm_only/ and tests/test_phase7_benchmark.py encodes max_missing_added=0 for every part. |
| ADD-02 | 06-01, 06-02, 06-04 | Spurious added-characteristic rows suppressed when span already explained by matched characteristic | VERIFIED | Content-aware suppressor in classify.py requires both text subset AND bbox overlap >= 0.3. TestExplainedByMatchSuppression + asset regression tests pass. |
| SNP-01 | 06-01, 06-04 | Title block regions reliably excluded from search windows | VERIFIED | Shared exclusion module used by all 4 search surfaces (anchors, match, rescue, added detection). TestSharedExclusionContract covers title-block, revision-table, bottom-center, and legitimate edge negative case. |

**Orphaned requirements check:** No requirements mapped to Phase 6 in REQUIREMENTS.md traceability table beyond ADD-01, ADD-02, SNP-01. No orphans.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `delta_preservation/cli.py` | L734 | Comment reads "Fallback: 'Not found in Rev B' placeholder shown in review card" — this is a legitimate runtime fallback for missing Rev B matches, not a code stub | Info | None — the comment is accurate operational documentation, not an incomplete implementation |

No TODOs, FIXMEs, or stub implementations found in the 5 modified production files. All new fields (`added_requirement_text`, `added_bbox`, `added_page`) are assigned in all three detection passes (Pass 0 GD&T, Pass 1 stacked, Pass 2 standard).

---

### Human Verification Required

#### 1. End-to-End Part 8 Pipeline Run

**Test:** Run the full reconciliation pipeline on the Part 8 PDF assets with the Phase 6 code and inspect the JSON output's `missing_added_truth_indexes` field.
**Expected:** `missing_added_truth_indexes` contains 0 entries (or at minimum does not contain truth_index 10, the `⌰ .002 A` row that was previously missing).
**Why human:** Cannot run the full pipeline without the Part 8 PDF files and the complete reconciliation stack. The regression harness pins the frozen pre-fix debug corpus deliberately — it does not assert the post-fix empty-missing-indexes outcome. The mechanism (`_expand_standard_added_span`, GD&T anchor expansion to include `⌰`) is unit-tested, but the round-trip from PDF spans to final output requires a real pipeline run.

#### 2. End-to-End Part 9 Pipeline Run

**Test:** Run the full reconciliation pipeline on the Part 9 PDF assets with the Phase 6 code and inspect `missing_added_truth_indexes` and the distinct-token claiming for the three duplicate pairs.
**Expected:** `missing_added_truth_indexes` is empty. The three duplicate pairs (`Ø.250 ±.008`, `⌖ ∅.015 D H`, `↧.50 ±.05`) each appear as two distinct claimed added rows with distinct truth indexes.
**Why human:** Same as Part 8 — end-to-end pipeline run with real PDF assets. The evaluator tie-break logic is unit-tested against synthetic data using exact ground-truth centers, but the actual packet `revB.bbox` values from a real pipeline run must be confirmed to be tight enough to trigger Stage 1 bbox containment.

#### 3. Cross-Part Aggregate Added Count

**Test:** Run the pipeline across all 9 debug corpus parts and compare aggregate added-characteristic counts against ground truth.
**Expected:** Total added rows emitted >= total ground truth added rows (ADD-01), and no increase in false-positive added rows versus the pre-Phase-6 baseline (ADD-02).
**Why human:** Multi-part aggregate check is defined as Phase 7 scope (VER-01 / TST-02) and cannot be verified without all 9 PDF assets.

---

### Gaps Summary

No code-level gaps found. All 9 required artifacts exist, are substantive (no stubs), are wired to their callers, and produce real data through the data flow. All 65 Phase 6 targeted tests pass (63 from the four new test files + 2 cross-file smokes). The full suite of 415 tests passes with 0 regressions.

The `human_needed` status reflects that Roadmap Success Criteria 1 and 4 ("clean pipeline run shows empty `missing_added_truth_indexes`" and "aggregate added count matches ground truth") require an end-to-end pipeline run with real PDF assets that cannot be validated programmatically. These are explicitly deferred to Phase 7 via VER-01 and TST-02.

---

_Verified: 2026-04-17T14:42:59Z_
_Verifier: Claude (gsd-verifier)_
