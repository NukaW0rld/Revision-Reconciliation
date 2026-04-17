---
phase: 06-added-characteristic-detection-and-snippet-accuracy
plan: "05"
subsystem: testing
tags: [part8, part9, added-detection, evaluation, regression, ground-truth]
dependency_graph:
  requires: [06-01, 06-02, 06-03, 06-04]
  provides: [phase-06-gap-closure, fresh-part8-part9-verification, refreshed-9-part-baseline]
  affects: [07-verification-and-generalization]
tech_stack:
  added: []
  patterns: [location-aware added dedupe, semantic ownership gates, tolerant truth-token normalization]
key_files:
  created:
    - .planning/phases/06-added-characteristic-detection-and-snippet-accuracy/06-05-SUMMARY.md
  modified:
    - delta_preservation/reconcile/classify.py
    - delta_preservation/reconcile/match.py
    - delta_preservation/evaluation/conformance.py
    - tests/test_added_detection_phase6.py
    - tests/test_added_truth_selection.py
    - tests/test_classify_bugfixes.py
    - tests/test_part5_grouping_bugfix.py
    - tests/test_phase6_asset_regression.py
    - .planning/debug/phase06-added-e2e-regressions.md
decisions:
  - "Treat Part 8/Part 9 reruns as the source of truth for gap closure rather than relying on frozen pre-fix debug snapshots"
  - "Keep added-row detection geometry-driven and generic; no part-specific string or coordinate hardcoding was introduced"
  - "Fix evaluator-side spacing tolerance once the rerun showed duplicate rows were emitted but not claimable"
metrics:
  duration: "60 min"
  completed: "2026-04-17"
  tasks_completed: 4
  files_modified: 9
requirements_completed: [ADD-01, ADD-02]
---

# Phase 06 Plan 05: Added-Row Verification Gap Closure Summary

Real pipeline verification now separates and claims the targeted Part 8 and Part 9 added rows instead of collapsing or stealing them before ground-truth evaluation.

## What Changed

- Tightened same-row grouping in `delta_preservation/reconcile/classify.py` so local GD&T companions stay together without absorbing distant annotations.
- Replaced text-only added-row dedupe with a `(grouped_text, rounded_bbox)` identity so duplicate callouts at different locations survive packet assembly.
- Added semantic-control, datum-set, and primary-value gates in `delta_preservation/reconcile/match.py` and `delta_preservation/reconcile/classify.py` to block weak ownership and removed+added reconciliation leaks.
- Hardened `delta_preservation/evaluation/conformance.py` so harmless spacing variants like `⌖∅ .015 D H` and `↧ .50 ±.05` still match canonical truth rows and reach the bbox tie-break.
- Added live-pipeline regression coverage for Part 8 and Part 9, plus focused selector tests for spacing-tolerant duplicate truth claiming.

## Verification

- Targeted suite: `uv run pytest tests/test_added_truth_selection.py tests/test_phase6_asset_regression.py tests/test_added_detection_phase6.py tests/test_part5_grouping_bugfix.py tests/test_classify_bugfixes.py -x` → `108 passed`
- Final Part 8 rerun: `/tmp/phase6-plan05-part8-final/part8_2026-04-17T11-01-35_49ee5833`
- Final Part 9 rerun: `/tmp/phase6-plan05-part9-final/part9_2026-04-17T11-01-35_f898d419`
- Final 9-part rerun root: `/tmp/phase6-plan05-all-final-kai96fdz`

Key outcomes:

- Part 8 now claims all canonical added truth indexes: `[8, 9, 10]`
- Part 9 now claims duplicate added truth indexes distinctly: `[35, 36, 37, 38, 39, 40, 41]`
- Corpus-wide aggregate improved from `21/35` emitted added rows and `28` missing truth rows to `34/35` emitted added rows and `11` missing truth rows
- Aggregate false-positive added rows dropped from `14` to `10`

## Commits

- `a9f60e1` — code and regression fixes for Part 8/Part 9 added-row verification gaps

## Deviations from Plan

One small evaluator-side fix was added after the first fresh rerun.

- The detector and packet assembly were already preserving duplicate Part 9 rows, but `select_truth_row_for_item()` still refused to claim them because its normalization treated spacing variants as different requirements.
- This was corrected by canonicalizing harmless control-symbol spacing in `delta_preservation/evaluation/conformance.py` and adding focused tests in `tests/test_added_truth_selection.py` plus a live packet regression in `tests/test_phase6_asset_regression.py`.

This did not widen scope beyond the plan. The fix was required to make the real rerun satisfy the “distinct claimed truth indexes” outcome instead of stopping at row emission.

## Remaining Gaps

Plan 05 closed the targeted regressions, but the corpus is not fully solved yet.

- `part9` still misses `truth_index 42` (`⏥ .01`)
- `part1` through `part5` still carry older missing added-truth rows
- Aggregate emitted added rows improved to `34/35`, which is materially better but still short of full ground-truth coverage

## Next Phase Readiness

Phase 07 now has a truthful post-fix baseline to build from.

- The stale “Part 8 merged row / Part 9 duplicate collapse” failure mode is no longer the blocker.
- The remaining work is narrower and explicit: recover the final unclaimed Part 9 flatness row and keep reducing the older corpus misses without reintroducing the fragment false positives that Phase 06 just removed.
