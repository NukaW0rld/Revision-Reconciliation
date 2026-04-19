---
status: resolved
trigger: "Phase 06 Part 9 still misses canonical added truth_index 42 (`⏥ .01`) after the Phase 06 gap-closure fixes."
created: 2026-04-17T19:00:00-05:00
updated: 2026-04-17T19:39:00-05:00
---

## Current Focus

hypothesis: confirmed root cause
test: completed
expecting: n/a
next_action: return compact diagnosis without applying fixes

## Symptoms

expected: Part 9 should emit canonical added truth_index 42 (`⏥ .01`) as an added item in the packet/run outputs
actual: The rerun misses added truth_index 42; observed added rows are `⌓ .02 A B C`, two `⌖∅ .015 D H`, two `↧ .50 ±.05`, and two `Ø.250 ±.008`, while the packet still contains an unchanged char 9 with requirement_revB `⏥ .01` at truth_index 8
errors: missing_added_truth_indexes: [42]
reproduction: Inspect live rerun directory `/tmp/phase6-uat-part9-3YS8OH/part9_2026-04-17T18-53-18_f898d419` against `assets/part9/ground_truth.json`; truth_index 42 is canonical added `⏥ .01` at snippet_center_revB [726, 446]
started: Still reproducible after the Phase 06 gap-closure fixes

## Eliminated

- hypothesis: The truth-42 flatness row is lost by early unmatched-span filters such as matched-span ownership, matched-group containment, Rev A carryover, exclusion-zone filtering, or near-match rejection.
  evidence: Reconstructed live Part 9 `matches` state shows the block-46 `⏥` and `.01` spans are not in `matched_span_keys`, not inside a matched group, not present in Rev A, not excluded by `span_is_excluded_for_annotation_search`, and not near any matched span center even at 25 pt.
  timestamp: 2026-04-17T19:30:00-05:00

## Evidence

- timestamp: 2026-04-17T19:06:00-05:00
  checked: knowledge base and Phase 06 UAT/summary artifacts
  found: The only knowledge-base overlap is a UI/export blocker session, not a detection/matching diagnosis. Phase 06 Plan 05 explicitly reports that Part 9 duplicate collapse and evaluator truth-claim spacing issues were fixed, leaving only `truth_index 42` (`⏥ .01`) unclaimed.
  implication: The remaining miss is likely narrower than the already-fixed duplicate-text and evaluator-normalization failures.

- timestamp: 2026-04-17T19:06:00-05:00
  checked: `.planning/debug/phase06-added-e2e-regressions.md` and `06-05-SUMMARY.md`
  found: Fresh reruns claim Part 9 truth indexes `[35, 36, 37, 38, 39, 40, 41]` and preserve the known duplicate groups, but still miss only `42`.
  implication: The system is no longer broadly collapsing duplicate added rows; this points to a flatness-specific ownership or detection path.

- timestamp: 2026-04-17T19:09:00-05:00
  checked: live rerun `delta_packet.json` item for `char_no=9`
  found: The packet contains an unchanged `char 9` with `requirement_revB: "⏥ .01"`, grouped-callout evidence, semantic-family flatness parsing, and evaluator match `truth_index: 8`.
  implication: One `⏥ .01` definitely survives into the packet as an existing match; the remaining question is whether the second physical flatness callout is never emitted as added or is emitted and then reconciled away.

- timestamp: 2026-04-17T19:12:00-05:00
  checked: `delta_preservation/reconcile/match.py` and `delta_preservation/reconcile/classify.py`
  found: `assign_matches()` allows only one owner per grouped source-span set and `detect_added_characteristics()` builds GD&T added rows from unmatched spans only, after excluding anything already matched or considered inside a matched group.
  implication: If the added flatness callout is already consumed or geometrically filtered before the added pass, the evaluator will never see a separate added `⏥ .01` row.

- timestamp: 2026-04-17T19:16:00-05:00
  checked: `assets/part9/ground_truth.json`, `assets/debug_report_part9.json`, and live packet geometry
  found: Part 9 has two distinct `⏥ .01` truths: unchanged `char_no 9` centered at `[542, 588]` and added `truth_index 42` centered at `[726, 446]`. The live packet only contains the unchanged one; there is no added packet row with `⏥ .01`.
  implication: The remaining miss is a true detection/assembly loss for the second physical flatness callout, not an evaluator tie-break between two packet rows.

- timestamp: 2026-04-17T19:22:00-05:00
  checked: raw PDF spans from `assets/part9/revB.pdf` near truth center `[726, 446]`
  found: The missing truth is present as clean PDF text in block `46`: line `0` is `⌓ .02 A B C` and line `1` is `⏥ .01`, with the flatness symbol and tolerance split into spans centered near `[712, 447]` and `[738, 447]`.
  implication: OCR/text extraction is not the issue; the loss happens after extraction inside the added-detection filters or suppressor.

- timestamp: 2026-04-17T19:26:00-05:00
  checked: raw PDF spans around the surviving unchanged `char 9` and around truth `42`
  found: The surviving unchanged flatness frame is block `33` near `[542, 588]`, while the missing added flatness frame is block `46` near `[726, 446]`; they are distinct physical annotations, not the same span claimed two different ways.
  implication: The remaining failure is not a simple evaluator ownership swap between the two flatness rows. The second frame is disappearing earlier in the added-detection pipeline.

- timestamp: 2026-04-17T19:30:00-05:00
  checked: reconstructed live Part 9 `matches` state against block `46` spans
  found: The block-46 `⏥` and `.01` spans are not in `matched_span_keys`, not inside any matched group, not present in Rev A, not excluded by `span_is_excluded_for_annotation_search`, and not near any matched span center even at `25 pt`.
  implication: None of the early unmatched-span filters explain the miss. The loss must happen later in `detect_added_characteristics()` or immediately after it.

- timestamp: 2026-04-17T19:34:00-05:00
  checked: `detect_added_characteristics()` and `reconcile_removed_added_pairs()` on reconstructed Part 9 state
  found: `detect_added_characteristics()` returns exactly 7 added rows and none are `⏥ .01`; `reconcile_removed_added_pairs()` does not consume any flatness added row afterward.
  implication: The missing `⏥ .01` is already gone by the end of `detect_added_characteristics()`, so the remaining suspect is its internal explained-by-match suppression stage.

- timestamp: 2026-04-17T19:39:00-05:00
  checked: explained-by-match suppressor signatures reconstructed from the live Part 9 `matches` state
  found: The suppressor creates a matched signature for existing `char 8` with text `4X INDIVIDUALLY 2X CR.50±.02 ⏥ .01` and bbox `(335.94, 438.41, 746.69, 456.05)` by sweeping same-row unmatched spans within ±200 pt. The missing flatness candidate `⏥ .01` has content-subset match against that signature and overlap ratio `1.0`, so it is suppressed as “explained by an existing matched characteristic.”
  implication: The independent added flatness frame at truth_index 42 is being falsely absorbed by the explained-by-match suppressor for `char 8`, not by matching or evaluator truth selection.

## Resolution

root_cause:
  The missing Part 9 added `⏥ .01` row is falsely removed by `detect_added_characteristics()`'s explained-by-match suppressor. When building matched-annotation ownership signatures, the suppressor expands matched `char 8` (`2X CR.50±.02`) by sweeping in unmatched same-row spans within 200 pt, which incorrectly pulls the separate block-46 flatness frame into `char 8`'s synthetic owner text/bbox (`4X INDIVIDUALLY 2X CR.50±.02 ⏥ .01`). The later subset-plus-overlap check then suppresses the real added flatness candidate as already explained.
fix:
verification:
files_changed: []
