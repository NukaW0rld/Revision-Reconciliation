---
phase: 09-full-corpus-added-characteristic-closure
plan: "01"
subsystem: evaluation/conformance + debug-accounting
tags: [added-row-token, leading-zero-normalization, debug-accounting, evaluation]
dependency_graph:
  requires: []
  provides:
    - Canonical added-row truth-token contract in conformance.py
    - Backward-compatible legacy integer token acceptance in review.py
    - Deterministic leading-zero normalization in _normalize_requirement_text
  affects:
    - delta_preservation/evaluation/conformance.py
    - shop/services/review.py
    - tests/test_added_truth_selection.py
    - tests/test_debug_row_identity.py
tech_stack:
  added: []
  patterns:
    - Canonical token-first, consumer-backward-compatible pattern
    - Deterministic token-level rewrite after semantic normalization
key_files:
  created:
    - tests/test_added_truth_selection.py (extended — 3 new tests)
    - tests/test_debug_row_identity.py (extended — 1 new test)
  modified:
    - delta_preservation/evaluation/conformance.py
    - shop/services/review.py
decisions:
  - "_truth_match_token always returns added:<index> for classification=added rows, even when truth row carries legacy char_no"
  - "_load_missing_added_truth_items accepts integer tokens only when they resolve to added truth rows"
  - "Leading-zero stripping restricted to tokens matching 0.<digits> so whole-number parts (10.000, 1.250) are unchanged"
metrics:
  duration: "3 min"
  completed_date: "2026-04-18"
  tasks_completed: 2
  files_modified: 4
requirements: [ADD-01]
---

# Phase 9 Plan 01: Canonical Added-Row Token Contract and Leading-Zero Normalization

One-liner: Canonical added:<index> token enforced for all added truth rows and leading-zero decimal variants normalized deterministically, closing Part 1 false missing-added accounting and Part 2 truth-selection miss.

## What Was Built

**Task 1 — Unify canonical added-row token contract**

Changed `_truth_match_token()` in `delta_preservation/evaluation/conformance.py` so that
any truth row whose `classification == "added"` always returns `f"added:{truth_index}"`,
regardless of whether the row also carries a legacy `char_no`.

Updated `_load_missing_added_truth_items()` in `shop/services/review.py` to also accept a
legacy integer `matched_truth_char_no` token as a valid claim when that integer equals the
`char_no` of an added truth row in the loaded fixture.  Non-added integer tokens are
ignored exactly as before.

Added two regression tests:
- `test_added_truth_match_token_uses_added_pool_prefix_even_when_truth_row_has_char_no` —
  asserts that an added truth row with `char_no=39` at pool index 0 yields `"added:0"`.
- `test_missing_added_accepts_legacy_int_token_for_added_truth_row_char_no` — asserts that
  a packet with `matched_truth_char_no=39` (integer) results in an empty
  `missing_added_truth_indexes` when the truth fixture has an added row with `char_no=39`.

**Task 2 — Normalize leading-zero decimal variants**

Added `_LEADING_ZERO_RE = re.compile(r"(?<![.\d])([+-]?)0(\.\d+)")` to
`delta_preservation/evaluation/conformance.py` and applied it as a final step inside
`_normalize_requirement_text()`.  The substitution strips the leading `0` only when the
integer part is exactly zero (e.g., `0.635` → `.635`, `+0.50` → `+.50`).  Tokens where
the integer part is greater than zero (`10.000`, `1.250`, `800`) are unchanged.

Added regression test:
- `test_unique_exact_text_accepts_leading_zero_variant_for_stacked_limits` — asserts that
  a packet item using `0.635 / 0.615` selects the same truth row as canonical `.635 / .615`.

## Verification

- `uv run pytest tests/test_added_truth_selection.py tests/test_debug_row_identity.py` → 25 passed
- `_normalize_requirement_text("0.635 / 0.615") == _normalize_requirement_text(".635 / .615")` → True
- `_normalize_requirement_text("+0.50 ± 0.05") == _normalize_requirement_text("+.50 ± .05")` → True
- `_normalize_requirement_text("10.000±.001") == _normalize_requirement_text("10.000±.001")` → True

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1    | 4a3d5ff | feat(09-01): unify canonical added-row token contract between evaluator and debug accounting |
| 2    | ae11931 | feat(09-01): normalize leading-zero decimal variants in requirement text for Part 2 closure |

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None — changes are confined to normalization and token serialization paths; no new network endpoints, auth paths, file access patterns, or schema changes introduced.

## Self-Check: PASSED

- `delta_preservation/evaluation/conformance.py` contains `truth_row.classification == "added"` inside `_truth_match_token` ✓
- `delta_preservation/evaluation/conformance.py` contains `f"{ADDED_POOL_TOKEN_PREFIX}:{truth_index}"` ✓
- `tests/test_added_truth_selection.py` contains `test_added_truth_match_token_uses_added_pool_prefix_even_when_truth_row_has_char_no` ✓
- `tests/test_debug_row_identity.py` contains `test_missing_added_accepts_legacy_int_token_for_added_truth_row_char_no` ✓
- `tests/test_added_truth_selection.py` contains `test_unique_exact_text_accepts_leading_zero_variant_for_stacked_limits` ✓
- Commits 4a3d5ff and ae11931 exist in git log ✓
