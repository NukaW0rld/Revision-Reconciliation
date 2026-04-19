# 09-05 Summary: Dimensional-Incompatibility Guard

**Status:** COMPLETE
**Commit:** db265b5

## What Changed

### Task 1: `_candidate_is_dimensionally_compatible` guard in `match.py`

Added a new helper function that blocks cross-characteristic mis-assignments
where a **grouped** candidate's numeric payload has zero overlap with the
anchor's numeric tokens. The guard is intentionally narrow — it only fires for
grouped candidates (those with `source_spans`) to avoid breaking legitimate
GD&T tolerance matches on single-span candidates.

Inserted the guard in both the greedy edge-acceptance loop and the shared-span
fallback path within `assign_matches`.

### Task 2: Source-span pruning

Replaced the unconditional `candidate_source_keys_set` construction with a
pruned computation that only marks dimensionally-compatible sub-spans as
consumed. Foreign sub-spans that were merged by `_group_candidate_spans` are
freed for `detect_added_characteristics()` so they can surface as added rows.

### Task 3: CLS-02 primary-value threshold tightening

Tightened the CLS-02 removed+added reconciliation primary-value threshold from
15% to 10%. This blocks `Ø35.2 / Ø34.8` (primary=35.2) from merging with
`3X Ø18 ↧30` (primary=30) where the relative gap was 14.8% — just under the
old 15% threshold.

### Task 4: Focused unit coverage

Created `tests/test_matching_dimensional_guard.py` with 3 tests:
- Rejection: `Ø35.2 / Ø34.8` anchor rejects `3X Ø18 ↧30` grouped candidate
- Acceptance: anchor accepts grouped candidate that carries its primary value
- Notes permissiveness: non-numeric anchor accepts any candidate

## Closed Gap

- **Part 5 truth_index 17 (`3X Ø18 ↧30`)**: The three source spans (`3X Ø18`,
  `↧`, `30`) were previously consumed by char 1 (`Ø35.2 / Ø34.8`) in the
  matching layer before added detection could claim them. The dimensional guard
  blocks the mis-assignment and source-span pruning frees the foreign sub-spans.

## Test Results

487 passed, 2 xfailed, 0 failures.
