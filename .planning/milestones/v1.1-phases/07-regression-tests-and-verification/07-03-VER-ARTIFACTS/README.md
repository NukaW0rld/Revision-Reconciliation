# VER-01 Per-Part Run Artifacts

Each `partN-run.txt` file is the verbatim stdout + stderr capture of:

```bash
python run.py part<N> 2>&1 | tee 07-03-VER-ARTIFACTS/part<N>-run.txt
```

These captures are the evidence trail for `07-VERIFICATION.md`. Do not
edit them after capture. If a part needs to be re-run, overwrite the file
with the new full capture and re-derive the summary table.

The captures intentionally include the run's:
- pipeline stage log lines
- per-part conforming / review_needed / missing_added_truth_indexes output
- any exception traces or warnings

Do not redact. Do not truncate.
