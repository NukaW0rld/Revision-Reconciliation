---
phase: 09-full-corpus-added-characteristic-closure
verified: 2026-04-18T00:00:00Z
status: gaps_found
score: 2/3 must-haves verified
gaps:
  - truth: "Fresh evaluation across all 9 debug parts yields empty missing_added_truth_indexes, including the current misses on parts 1-5 and 9."
    status: partial
    reason: "Part 5 truth_indexes 16 ('800') and 17 ('3X Ø18 ↧30') remain in missing_added_truth_indexes=[16,17]. Plans 02-03 explicitly deferred these as architectural deferrals (matching-layer mis-assignment and near-boilerplate proximity filter), but the roadmap SC requires empty for all parts including part5. No later milestone phase (10 or 11) covers these items."
    artifacts:
      - path: "tests/fixtures/phase7_algorithm_only/part5-debug-report.json"
        issue: "missing_added_truth_indexes=[16,17] — not empty"
      - path: "tests/test_phase7_benchmark.py"
        issue: "BASELINE_COUNTS['part5']['max_missing_added']=2 — not 0 as roadmap SC requires"
    missing:
      - "Close Part 5 truth_index 16 ('800') — near-boilerplate proximity threshold needs tightening or the '800' token needs shared-path handling that distinguishes it from boilerplate"
      - "Close Part 5 truth_index 17 ('3X Ø18 ↧30') — matching mis-assignment consumes spans before detection; matching-layer fix required"
---

# Phase 9: Full-Corpus Added Characteristic Closure — Verification Report

**Phase Goal:** Close added-characteristic missing-added gaps across all 9 corpus parts — zero missing_added_truth_indexes in algorithm-only fixture set, refreshed Phase 06/07 evidence, ADD-01 traceability complete.
**Verified:** 2026-04-18
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Fresh evaluation across all 9 debug parts yields empty `missing_added_truth_indexes`, including the current misses on parts 1-5 and 9. | PARTIAL | Parts 1-4 and 6-9 have empty `missing_added_truth_indexes` (confirmed in fixture JSON). Part 5 has `[16, 17]` remaining. Plans 02-03 document these as architectural deferrals but no later phase addresses them. |
| 2 | Added-characteristic output does not introduce new false positives relative to the current Phase 06/07 baseline while closing the missing-row gap. | VERIFIED | `tests/test_added_detection_phase6.py` (51 tests) and `tests/test_phase6_asset_regression.py` all pass. `assets/debug_report_part*.json` frozen assets unmodified (git diff confirms no output). Explained-by-match suppressor guardrails remain intact after owner-signature narrowing. |
| 3 | Phase 06 verification/validation artifacts and any downstream milestone evidence are refreshed to reflect the final full-corpus result. | VERIFIED | `06-VALIDATION.md` has Phase 9 follow-up note. `06-VERIFICATION.md` has ADD-01 updated from DEFERRED to VERIFIED. `07-VERIFICATION.md` has Phase 9 Closure Note. `REQUIREMENTS.md` has `- [x] **ADD-01**` checked and traceability row `\| ADD-01 \| Phase 9 \| 09-01-PLAN.md, 09-02-PLAN.md, 09-03-PLAN.md \|`. |

**Score:** 2/3 truths verified (SC #1 partially met — 8 of 9 parts at zero, Part 5 has 2 remaining)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `delta_preservation/evaluation/conformance.py` | `def _truth_match_token(` + canonical added-row token + `_LEADING_ZERO_RE` + `_MODIFIER_SPACING_RE` | VERIFIED | All four markers present. `_truth_match_token` returns `f"{ADDED_POOL_TOKEN_PREFIX}:{truth_index}"` when `truth_row.classification == "added"`. |
| `shop/services/review.py` | `def _load_missing_added_truth_items(` | VERIFIED | Present. Backward-compatible legacy integer token acceptance logic present (lines 549+). |
| `tests/test_added_truth_selection.py` | `test_added_truth_match_token_uses_added_pool_prefix_even_when_truth_row_has_char_no` + `test_unique_exact_text_accepts_leading_zero_variant_for_stacked_limits` | VERIFIED | Both test functions present at lines 338 and 377 respectively. |
| `tests/test_debug_row_identity.py` | `test_missing_added_accepts_legacy_int_token_for_added_truth_row_char_no` | VERIFIED | Present at line 300. |
| `delta_preservation/reconcile/classify.py` | `matched_annotation_signatures` + `span_is_excluded_for_annotation_search(` + no `ox0 > mx1 + 200.0` | VERIFIED | `matched_annotation_signatures` present (line 2143). `span_is_excluded_for_annotation_search(` present (8 occurrences). Old `±200 pt` sweep condition absent. |
| `tests/test_added_detection_phase6.py` | `test_part9_flatness_survives_when_unrelated_same_row_match_exists` + `test_part8_fragment_still_suppressed_after_owner_signature_narrowing` + exemplar strings for Parts 3-5 | VERIFIED | All required functions and exemplar strings present (lines 468, 534; exemplars at lines 837, 887, 919, 947, 980). |
| `tests/test_phase6_asset_regression.py` | Plan-specified test name `test_live_part8_packet_preserves_separate_runout_and_diameter_rows` | PARTIAL | Artifact exists and is substantive. The plan's `contains` value specifies a test name that does not exist in the file. Actual function is `test_live_part8_pipeline_keeps_added_rows_distinct` (line 365) which covers identical intent (Part 8 runout and diameter rows kept distinct). This was a pre-existing Phase 6 test; the plan specified the wrong expected name. The regression guard is functionally present. |
| `tests/test_phase7_benchmark.py` | `BASELINE_COUNTS` with `max_missing_added=0` for all parts | PARTIAL | `BASELINE_COUNTS` present with all 9 parts. Parts 1-4 and 6-9 have `max_missing_added=0`. Part 5 has `max_missing_added=2`. Plan 03 decision: set 2 to avoid false confidence. Does not meet roadmap SC #1 which requires zero for all parts. |
| `tests/fixtures/phase7_algorithm_only/part{1..9}-debug-report.json` | All 9 fixture files present with empty `missing_added_truth_indexes` | PARTIAL | All 9 files exist. Parts 1-4 and 6-9 have `missing_added_truth_indexes: []`. Part 5 has `[16, 17]`. |
| `.planning/phases/07-regression-tests-and-verification/07-04-ALGORITHM-ONLY-BASELINE.md` | Contains "Missing-Added (indexes)", "historical Phase 6", "part9" | VERIFIED | All three strings present. Document rewritten with Phase 9 post-closure provenance. |
| `.planning/phases/07-regression-tests-and-verification/07-VERIFICATION.md` | Contains "Phase 9" | VERIFIED | Phase 9 Closure Note section present. |
| `.planning/REQUIREMENTS.md` | `\| ADD-01 \| Phase 9 \| 09-01-PLAN.md, 09-02-PLAN.md, 09-03-PLAN.md \|` | VERIFIED | Traceability row present at line 101. ADD-01 checkbox checked at line 44. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `delta_preservation/evaluation/conformance.py` | `shop/services/review.py` | `ADDED_POOL_TOKEN_PREFIX` token contract | VERIFIED | `ADDED_POOL_TOKEN_PREFIX = "added"` defined in conformance.py. `_load_missing_added_truth_items` in review.py consumes tokens matching `added:<index>` pattern as confirmed by test `test_missing_added_accepts_legacy_int_token_for_added_truth_row_char_no`. |
| `delta_preservation/reconcile/classify.py` | `tests/test_added_detection_phase6.py` | `detect_added_characteristics` / explained-by-match suppressor | VERIFIED | `def detect_added_characteristics(` present in classify.py. Tests reference the suppressor and companion-walk behavior directly. All 51 tests pass. |
| `tests/test_phase7_benchmark.py` | `tests/fixtures/phase7_algorithm_only/` | `FIXTURES_DIR` / `phase7_algorithm_only` | VERIFIED | `FIXTURES_DIR = Path(__file__).parent / "fixtures" / "phase7_algorithm_only"` at line 26. All 9 fixture files present. 20 benchmark tests pass. |

### Behavioral Spot-Checks

| Behavior | Command / Check | Result | Status |
|----------|-----------------|--------|--------|
| Leading-zero normalization | `_normalize_requirement_text("0.635 / 0.615") == _normalize_requirement_text(".635 / .615")` | True | PASS |
| Leading-zero unchanged for non-zero integers | `_normalize_requirement_text("10.000±.001")` unchanged | True | PASS |
| All fixtures have expected missing_added state | Python loop over 9 fixture files | parts 1-4,6-9: `[]`; part5: `[16,17]` | PARTIAL |
| Full test suite | `uv run pytest tests/` | 482 passed, 2 xfailed, 0 failures | PASS |
| Phase 06/07 regression tests | `uv run pytest tests/test_added_detection_phase6.py tests/test_phase6_asset_regression.py` | 51 passed | PASS |
| Benchmark tests | `uv run pytest tests/test_phase7_benchmark.py` | 20 passed | PASS |
| Historical Phase 6 assets unmodified | `git diff -- assets/debug_report_part*.json` | No output | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ADD-01 | 09-01, 09-02, 09-03 | All ground-truth-added characteristics present in pipeline output for every part | PARTIALLY SATISFIED | Parts 1-4 and 6-9 verified at zero. Part 5 indexes 16+17 remain. REQUIREMENTS.md marks ADD-01 checked with caveat. Roadmap SC #1 requires full closure including Part 5. |
| ADD-02 | 09-02, 09-03 (affected, not primary) | Spurious added rows suppressed when span explained by matched characteristic | SATISFIED | Explained-by-match suppressor unchanged from Phase 6 contract. TestExplainedByMatchSuppression passes (included in 51 tests). Owner-signature narrowing does not weaken suppression for genuine fragments (test_part8_fragment_still_suppressed_after_owner_signature_narrowing confirms). |
| SNP-01 | 09-02, 09-03 (affected, not primary) | Title block regions excluded from search windows | SATISFIED | `span_is_excluded_for_annotation_search(` present in classify.py (8 occurrences). Exclusion contract unchanged. Full suite passes. |

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `tests/fixtures/phase7_algorithm_only/part5-debug-report.json` | `missing_added_truth_indexes: [16, 17]` | BLOCKER | Part 5 does not meet the roadmap SC #1 zero-miss requirement. Intentional deferral but not waived in roadmap. |
| `tests/test_phase7_benchmark.py` | `BASELINE_COUNTS['part5']['max_missing_added'] = 2` | WARNING | Benchmark allows 2 misses for Part 5; matches fixture state but deviates from roadmap zero-miss contract. |

### Gaps Summary

**One substantive gap** prevents full roadmap SC #1 verification:

Part 5 has two remaining `missing_added_truth_indexes` ([16, 17]) corresponding to `800` (truth_index 16, near-boilerplate false-proximity) and `3X Ø18 ↧30` (truth_index 17, matching-layer mis-assignment). Phase 9 Plans 02 and 03 explicitly documented these as architectural deferrals — they require changes at the matching layer (for index 17) and threshold tuning (for index 16) that were judged out-of-scope for Phase 9's Plans.

The roadmap success criteria says "including the current misses on parts 1-5 and 9" — Part 5 is named explicitly. No later milestone phase (10 or 11) covers these items in their success criteria.

There is also a minor naming mismatch: the 09-02-PLAN `contains` check for `test_phase6_asset_regression.py` specifies `test_live_part8_packet_preserves_separate_runout_and_diameter_rows` but the actual pre-existing function is `test_live_part8_pipeline_keeps_added_rows_distinct`. The functional intent is identical and the test was not supposed to be created in Phase 9 (it is a pre-existing Phase 6 test). This does not block goal achievement but represents a plan spec inaccuracy.

**Three items are fully verified:**
- Token/accounting contract (Plan 01) — canonical `added:<index>` token enforced, legacy integer accepted backward-compatibly
- Leading-zero and material-modifier normalization (Plans 01, 03) — deterministic, tested
- Phase 06/07 evidence refresh and ADD-01 traceability (Plan 03) — all documents updated, benchmark and asset regression tests pass

---

_Verified: 2026-04-18T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
