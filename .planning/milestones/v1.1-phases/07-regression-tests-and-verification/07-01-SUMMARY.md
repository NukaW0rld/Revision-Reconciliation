# Plan 07-01 Summary: Per-Cluster Audit and CLS-03 Gap-Filling

**Completed:** 2026-04-17
**Plan:** 07-01
**Phase:** 07-regression-tests-and-verification
**Requirements:** TST-01

---

## Task Outcomes

### Task 1: Per-cluster audit document

**Status:** Complete

Created `.planning/phases/07-regression-tests-and-verification/07-01-AUDIT.md` with a 9-row
cluster audit table. All existing exemplar test names were verified with `rg` before writing.

**Verdict summary:**

| Cluster | Verdict |
|---------|---------|
| GDT-01 compact token splitting | covered |
| GDT-02 word-name normalization | covered |
| GDT-03 composite FCF capture | covered |
| CLS-01 adjacency bleed suppression | covered |
| CLS-02 removed+added reconciliation | covered |
| CLS-03 asymmetric tolerance detection | gap_filling_required |
| ADD-01 missing added rows | covered |
| ADD-02 false-positive suppression | covered |
| SNP-01 title block / revision exclusion | covered |

Eight of nine clusters have pre-fix-failing exemplars. One gap identified: CLS-03 leading-decimal
asymmetric shape form was not explicitly pinned at the regex level.

### Task 2: CLS-03 leading-decimal gap-filling

**Status:** Complete

Added two parametrized test methods to
`tests/test_classify_phase5_regression.py::TestPhase5SnapshotExemplars`:

- `test_asymmetric_shape_re_matches_leading_decimal_variants` — 3 parametrized inputs
  (`+.3°/−.1°` variants); asserts `_ASYMMETRIC_SHAPE_RE.search(revB_text) is not None`.
- `test_asymmetric_shape_re_does_not_match_plain_fractional_ratios` — 2 parametrized inputs
  (`.3 / .1`, `70 / 30`); asserts the regex does NOT match plain numeric ratios.

All 6 new parametrized cases pass:
```
uv run pytest tests/test_classify_phase5_regression.py -k "asymmetric_shape_re" -x
→ 6 passed
```

### Task 3: Additional gap-filling from audit

**Status:** Skipped (zero additional gaps)

The Task 1 audit identified only the CLS-03 leading-decimal case as `gap_filling_required`.
All other 8 clusters were `covered`. Task 2 closed the sole gap. No further gap-filling was
required.

---

## Pre-existing test failures (not caused by this plan)

The full suite has two pre-existing failures that existed before this plan's changes:

1. `tests/test_classify_phase5_regression.py::TestPhase5SyntheticReconciliation::test_grouped_compatible_added_near_removed_becomes_changed`
   — `reconcile_removed_added_pairs` returns `"removed"` instead of `"changed"` for the grouped
   added item case; pre-existing regression unrelated to Phase 7 work.

2. `tests/test_ground_truth_evaluation.py::test_normalized_text_fallback_allows_equivalent_requirement_when_semantics_unavailable`
   — `requirement_conforms` is `False` when expected `True`; pre-existing evaluation layer
   regression unrelated to Phase 7 work.

Both verified by `git stash` + targeted re-run before restoring changes.

---

## Verification commands (all pass)

```bash
# New asymmetric shape regex tests
uv run pytest tests/test_classify_phase5_regression.py -k "asymmetric_shape_re" -x
# → 6 passed

# Audit document has required section
rg -n "^## Cluster audit table" .planning/phases/07-regression-tests-and-verification/07-01-AUDIT.md
# → 7:## Cluster audit table

# Audit has exactly 9 data rows
python3 -c "..."
# → ok — 9 data rows found

# Gap-filling tests are present in target file
rg -n "test_asymmetric_shape_re_matches_leading_decimal_variants" tests/test_classify_phase5_regression.py
rg -n "test_asymmetric_shape_re_does_not_match_plain_fractional_ratios" tests/test_classify_phase5_regression.py
```

---

## Artifacts produced

| Artifact | Status |
|----------|--------|
| `.planning/phases/07-regression-tests-and-verification/07-01-AUDIT.md` | Created |
| `tests/test_classify_phase5_regression.py` | Modified (2 new test methods, 5 new parametrize cases) |
| `.planning/phases/07-regression-tests-and-verification/07-01-SUMMARY.md` | Created (this file) |
