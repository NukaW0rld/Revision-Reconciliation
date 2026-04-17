# Plan 07-03 Summary: VER-01 Scaffold and Manual Re-Run Checkpoint

**Status:** CHECKPOINT REACHED — awaiting developer pipeline runs
**Plan:** 07-03
**Phase:** 07-regression-tests-and-verification
**Requirements:** VER-01

---

## Task Outcomes

### Task 1: Scaffold VER-01 artifact directory and VERIFICATION.md template

**Status:** Complete — committed at `e56e8a8`

Created:
- `.planning/phases/07-regression-tests-and-verification/07-03-VER-ARTIFACTS/README.md`
- `.planning/phases/07-regression-tests-and-verification/07-VERIFICATION.md` (template with 49 TODO_FILL placeholders)

Acceptance criteria verified:
- `## Summary table` present with 9 data rows ✅
- 9 `### Part N` sections ✅
- `## Phase closure statement` section present ✅
- 49 `<TODO_FILL>` placeholders (template intentionally unfilled) ✅

---

## CHECKPOINT REACHED

Task 2 is a **manual developer action** (`autonomous: false`, CONTEXT.md D-06).
The agent must not run the pipeline. The developer must produce the 9 raw run
captures before Task 3 (populating VERIFICATION.md) can proceed.

### Developer run script (copy-pasteable)

```bash
set -e
mkdir -p .planning/phases/07-regression-tests-and-verification/07-03-VER-ARTIFACTS
for i in 1 2 3 4 5 6 7 8 9; do
  echo "=== part${i} ==="
  python run.py part${i} 2>&1 \
    | tee .planning/phases/07-regression-tests-and-verification/07-03-VER-ARTIFACTS/part${i}-run.txt
done
```

### What to check after each run

For each `partN-run.txt`, verify:
1. No `Traceback (most recent call last)` lines (or document any exception explicitly)
2. The output contains `conforming` (evaluation summary is present)
3. The pipeline ran against `assets/part<N>/revA.pdf` + `assets/part<N>/revB.pdf`

### Resuming Task 3

Once all 9 captures exist in `07-03-VER-ARTIFACTS/`, start a new session and
ask Cascade to resume plan 07-03, Task 3: populate `07-VERIFICATION.md` from
the raw captures.

Task 3 will:
1. Run the derivation script against `out/part{1..9}/debug_report.json`
2. Fill the `(post)` columns from fresh pipeline output
3. Fill the `(pre)` columns from pre-Phase-4 baseline (Option B: git-historic snapshots)
4. Compute the `Previously-passing regressions` column per part
5. Fill the Phase closure statement if zero regressions, or emit `## VERIFICATION BLOCKED` if any regression is found

---

## Artifacts produced

| Artifact | Status |
|----------|--------|
| `.planning/phases/07-regression-tests-and-verification/07-03-VER-ARTIFACTS/README.md` | Created |
| `.planning/phases/07-regression-tests-and-verification/07-VERIFICATION.md` | Created (template) |
| `.planning/phases/07-regression-tests-and-verification/07-03-SUMMARY.md` | Created (this file) |
