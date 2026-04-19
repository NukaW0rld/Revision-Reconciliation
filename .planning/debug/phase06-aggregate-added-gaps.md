---
status: diagnosed
trigger: "Phase 06 aggregate added-row coverage remains 34/35 emitted with 11 missing truth rows across the 9-part corpus after the gap-closure fixes. Diagnose whether this is one shared remaining algorithm gap or a collection of older unresolved misses outside the targeted Part 8/9 regressions."
created: 2026-04-17T00:00:00-05:00
updated: 2026-04-17T00:47:00-05:00
---

## Current Focus

hypothesis: confirmed: the aggregate 11 missing truth rows are a mixed bucket, not one shared remaining algorithm regression
test: complete
expecting: n/a
next_action: return diagnosis in root-cause-only format

## Symptoms

expected: Aggregate added-row coverage should reflect the Phase 06 gap-closure fixes without leaving unexplained corpus-wide missing truth rows.
actual: The 9-part rerun reports truth_added_rows=35, emitted_added_rows=34, claimed_added_rows=24, missing_added_rows=11, false_positive_added_rows=10, with per-part missing indexes part1 [38], part2 [22], part3 [19,20], part4 [11,14,15], part5 [16,17,18], part6 [], part7 [], part8 [], part9 [42].
errors: No explicit runtime error; the failure is a benchmark/debug coverage gap in added-row detection.
reproduction: Inspect the live rerun root /tmp/phase6-uat-all-CukVkX and compare aggregate/per-part added-row results against the Phase 06 UAT expectations and benchmark logic.
started: After the Phase 06 gap-closure fixes, during aggregate 9-part rerun validation.

## Eliminated

## Evidence

- timestamp: 2026-04-17T00:10:00-05:00
  checked: .planning/debug/knowledge-base.md
  found: The only keyword overlap was an old debug-export UI mismatch involving hidden missing_added_truth_indexes, not an algorithmic added-detection issue.
  implication: No strong knowledge-base match exists for the current aggregate coverage gap; investigate the pipeline outputs directly.

- timestamp: 2026-04-17T00:12:00-05:00
  checked: .planning/phases/06-added-characteristic-detection-and-snippet-accuracy/06-UAT.md and .planning/debug/phase06-added-e2e-regressions.md
  found: The refreshed Phase 06 verification explicitly says Part 8 is resolved, Part 9 preserves duplicate rows but still misses only truth index 42, and Parts 1-5 retain the same missing added truth rows in the final 9-part rerun.
  implication: The aggregate failure already looked like one residual Part 9 miss plus older unresolved misses before this new diagnosis started.

- timestamp: 2026-04-17T00:14:00-05:00
  checked: .planning/phases/06-added-characteristic-detection-and-snippet-accuracy/06-05-PLAN.md and 06-05-SUMMARY.md
  found: Plan 05 scope targeted Part 8 over-grouping, Part 9 duplicate collapse, weak ownership, and evaluator spacing tolerance; the summary states the remaining gaps are Part 9 truth index 42 and pre-existing misses in Parts 1-5.
  implication: If the live rerun matches those same per-part misses, the aggregate shortfall is not a new shared regression introduced after the fixes.

- timestamp: 2026-04-17T00:17:00-05:00
  checked: tests/test_phase7_benchmark.py and .planning/phases/07-regression-tests-and-verification/07-CONTEXT.md
  found: The Phase 7 benchmark locks per-part max_missing_added ceilings of part1=1, part2=1, part3=2, part4=3, part5=3, part6=0, part7=0, part8=0, part9=1, which sum to 11 missing truth rows.
  implication: The accepted algorithm-only baseline already encodes the current aggregate result as legacy unresolved misses, not as a post-fix regression.

- timestamp: 2026-04-17T00:26:00-05:00
  checked: /tmp/phase6-uat-all-CukVkX delta_packet.json files re-evaluated against assets/*/ground_truth.json
  found: The live rerun reproduces the same per-part missing added rows as the Phase 7 algorithm-only fixtures: part1 [38], part2 [22], part3 [19,20], part4 [11,14,15], part5 [16,17,18], part6 [], part7 [], part8 [], part9 [42].
  implication: The current aggregate numbers are the existing post-Plan-05 baseline, not a new corpus-wide regression.

- timestamp: 2026-04-17T00:31:00-05:00
  checked: part1 live packet, assets/part1/ground_truth.json, delta_preservation/evaluation/conformance.py, shop/services/review.py
  found: Part1 emits an added row with exact requirement text `2 x 5 X 45.0°`, and evaluation matches it to the canonical added truth row at index 38; because that truth row still has `char_no=39`, `_truth_match_token()` serializes the match as integer `39` instead of `added:38`, and `_load_missing_added_truth_items()` ignores non-string tokens when computing claimed added rows.
  implication: Part1's reported missing added truth row is a debug/evaluation accounting artifact, not an added-detection miss.

- timestamp: 2026-04-17T00:35:00-05:00
  checked: part2 and part3 live packets compared with conformance normalization
  found: Part2 emits `0.635 / 0.615` but truth expects `.635 / .615`, and current normalization preserves that difference so the row stays `truth_ambiguity`; Part3 emits truncated surface-finish text `1000 Ra` relative to truth `FINISH TURN 1000 Ra`, while its countersink row `⌵ Ø.531 X 82.0°` is absent entirely.
  implication: Parts2-3 do not share one common failure mode; one is claim/normalization, the other mixes truncation and non-emission.

- timestamp: 2026-04-17T00:38:00-05:00
  checked: part4, part5, and part9 live packets versus truth rows
  found: Part4 still lacks `⌖ ∅.005Ⓜ A B C` and both `1.250` rows; Part5 still lacks `800`, `3X Ø18 ↧30`, and `M20x2.5 − 6H ↧6`; Part9 still never emits the canonical `⏥ .01` row even though all duplicate pairs and `⌓ .02 A B C` are now claimed correctly.
  implication: The remaining misses are heterogeneous legacy gaps plus one residual Part9 non-emission, not a single shared post-fix algorithm defect.

## Resolution

root_cause: The aggregate 11-row shortfall is not one shared remaining algorithm gap. It is a mixed bucket composed of (1) a Part1 debug/evaluation accounting false miss, (2) a Part2 truth-selection normalization miss, (3) older heterogeneous Parts3-5 added-detection/grouping misses that predate the targeted Part8/9 fixes, and (4) one real remaining Part9 non-emitted row (`⏥ .01`).
fix: None applied; diagnose-only mode.
verification: Verified by re-evaluating the live 9-part rerun packets against ground truth and comparing the result to the locked Phase 7 algorithm-only baseline plus packet-level added-row contents.
files_changed: []
