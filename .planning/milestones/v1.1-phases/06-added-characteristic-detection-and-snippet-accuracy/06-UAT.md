---
status: complete
phase: 06-added-characteristic-detection-and-snippet-accuracy
source: [06-05-SUMMARY.md, 06-VERIFICATION.md]
started: 2026-04-17T18:55:25-05:00
updated: 2026-04-17T18:55:25-05:00
---

## Current Test

[testing complete]

## Tests

### 1. End-to-End Part 8 Pipeline Run
expected: `missing_added_truth_indexes` is empty (or at minimum does not contain truth_index 10, the `⌰ .002 A` row that was previously missing) after running the full reconciliation pipeline on Part 8 PDF assets with Phase 6 code.
result: pass

### 2. End-to-End Part 9 Pipeline Run
expected: `missing_added_truth_indexes` is empty and the three duplicate pairs (`Ø.250 ±.008`, `⌖ ∅.015 D H`, `↧.50 ±.05`) each appear as two distinct claimed added rows with distinct truth indexes.
result: issue
reported: "Live rerun preserved both instances of the three duplicate pairs and claimed truth indexes [35, 36, 37, 38, 39, 40, 41], but still left missing_added_truth_indexes=[42] for the canonical added `⏥ .01` row."
severity: major

### 3. Cross-Part Aggregate Added Count
expected: Total added rows emitted >= total ground truth added rows, and no increase in false-positive added rows versus the pre-Phase-6 baseline across all 9 debug corpus parts.
result: issue
reported: "Live 9-part rerun emitted 34 added rows for 35 truth-added rows and left 11 missing truth rows overall; false-positive added rows stayed improved at 10, but aggregate added-row coverage is still short of ground truth."
severity: major

## Summary

total: 3
passed: 1
issues: 2
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "`missing_added_truth_indexes` is empty and the three duplicate pairs (`Ø.250 ±.008`, `⌖ ∅.015 D H`, `↧.50 ±.05`) each appear as two distinct claimed added rows with distinct truth indexes."
  status: failed
  reason: "Live rerun preserved both instances of the three duplicate pairs and claimed truth indexes [35, 36, 37, 38, 39, 40, 41], but still left missing_added_truth_indexes=[42] for the canonical added `⏥ .01` row."
  severity: major
  test: 2
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""

- truth: "Total added rows emitted >= total ground truth added rows, and no increase in false-positive added rows versus the pre-Phase-6 baseline across all 9 debug corpus parts."
  status: failed
  reason: "Live 9-part rerun emitted 34 added rows for 35 truth-added rows and left 11 missing truth rows overall; false-positive added rows stayed improved at 10, but aggregate added-row coverage is still short of ground truth."
  severity: major
  test: 3
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""
