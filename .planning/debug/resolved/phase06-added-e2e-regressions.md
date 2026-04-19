---
phase: 06-added-characteristic-detection-and-snippet-accuracy
created: 2026-04-17T15:04:42Z
updated: 2026-04-17T16:04:29Z
kind: verification-refresh
code_commit: a9f60e1
---

# Phase 06 Added-Detection E2E Regressions

## Scope

Fresh verification was rerun against the real pipeline outputs after the Plan 05 fixes landed.

Commands run:

```bash
uv run python run.py part8 --out_dir /tmp/phase6-plan05-part8-final
uv run python run.py part9 --out_dir /tmp/phase6-plan05-part9-final
for n in 1 2 3 4 5 6 7 8 9; do
  uv run python run.py "part${n}" --out_dir /tmp/phase6-plan05-all-final-kai96fdz
done
```

Verification outputs:

- Part 8 run: `/tmp/phase6-plan05-part8-final/part8_2026-04-17T11-01-35_49ee5833`
- Part 9 run: `/tmp/phase6-plan05-part9-final/part9_2026-04-17T11-01-35_f898d419`
- 9-part rerun root: `/tmp/phase6-plan05-all-final-kai96fdz`

## Final Results

### Part 8

The Part 8 over-grouping failure is resolved.

- Emitted added rows: `6`
- Claimed added truth indexes: `[8, 9, 10]`
- Missing added truth indexes: `[]`
- Canonical added rows now present as distinct packet rows:
  - `⌰ .015 B` → `added:9`
  - `⌰ .002 A` → `added:10`
  - `Ø10.000±.001` → `added:8`
- The merged row `⌰ .015 B Ø10.000±.001` no longer appears

Residual Part 8 added rows that remain review-needed are now false-positive fragments rather than missed canonical rows:

- `◎∅ .045 A`
- `⚪ .005`
- `⟂ .010 D`

### Part 9

The duplicate-collapse and weak-reconciliation failures are resolved.

- Emitted added rows: `7`
- Claimed added truth indexes: `[35, 36, 37, 38, 39, 40, 41]`
- Missing added truth indexes: `[42]`
- Duplicate groups now survive packet assembly and claim distinct truth tokens:
  - `⌖∅ .015 D H` → `added:39`, `added:36`
  - `↧ .50 ±.05` → `added:40`, `added:37`
  - `Ø.250 ±.008` → `added:38`, `added:35`
- Unique added profile row remains claimable:
  - `⌓ .02 A B C` → `added:41`

Only one canonical added row remains unclaimed in Part 9:

- `truth_index 42` → `⏥ .01`

### 9-Part Aggregate

The corpus-wide rerun materially improves the pre-fix snapshot.

| Metric | Pre-fix diagnostic | Final rerun |
|--------|--------------------|-------------|
| Truth-added rows across parts 1-9 | 35 | 35 |
| Emitted added rows | 21 | 34 |
| Claimed added truth rows | 7 | 24 |
| False-positive added rows | 14 | 10 |
| Missing added truth rows | 28 | 11 |

Per-part final missing indexes:

- `part1`: `[38]`
- `part2`: `[22]`
- `part3`: `[19, 20]`
- `part4`: `[11, 14, 15]`
- `part5`: `[16, 17, 18]`
- `part6`: `[]`
- `part7`: `[]`
- `part8`: `[]`
- `part9`: `[42]`

## Root Causes Verified

The fresh reruns confirm the four Plan 05 hypotheses were real:

1. Pass-0 same-row GD&T grouping was absorbing distant annotations into one added row.
2. Standard added-row dedupe was collapsing physical duplicates by text only.
3. Weak existing-match ownership allowed semantic mismatches to steal legitimate added rows.
4. Added-row truth selection still failed on harmless spacing variants, which prevented duplicate rows from claiming their canonical truth indexes even after detection was fixed.

## Resolution Applied

The finished fix set in `a9f60e1` closed those gaps by:

- replacing broad same-row sweeps with local companion walks in `delta_preservation/reconcile/classify.py`
- deduping added rows by `(grouped_text, rounded_bbox)` instead of text only
- adding semantic-control, datum-set, and primary-value guards in `delta_preservation/reconcile/match.py` and `delta_preservation/reconcile/classify.py`
- canonicalizing harmless control-symbol spacing during added-row truth selection in `delta_preservation/evaluation/conformance.py`

## Remaining Gaps

Plan 05 closed the targeted Part 8 and Part 9 regressions, but it did not finish the entire corpus-wide added-detection problem set.

Still open after the final rerun:

- `part9` still misses `truth_index 42` (`⏥ .01`)
- `part1` through `part5` still have pre-existing missing added truth rows
- aggregate emitted added rows improved to `34/35`, but the corpus is not yet fully ground-truth complete

## Takeaway

Phase 06 no longer fails for the reasons that triggered the gap plan.

- Part 8 now claims all three canonical added rows.
- Part 9 now preserves and claims both duplicate position and depth rows instead of collapsing them.
- The 9-part rerun moves the corpus from `21/35` emitted added rows and `28` missing truth rows to `34/35` emitted added rows and `11` missing truth rows without increasing false positives.
