---
phase: 11-web-run-to-review-e2e-automation
reviewed: 2026-04-18T00:00:00Z
depth: standard
files_reviewed: 1
files_reviewed_list:
  - tests/test_web_run_review_e2e.py
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 11: Code Review Report

**Reviewed:** 2026-04-18T00:00:00Z
**Depth:** standard
**Files Reviewed:** 1
**Status:** issues_found

## Summary

The file `tests/test_web_run_review_e2e.py` implements four test cases covering the full live web-run-to-review E2E flow for Phase 11. The overall structure is sound: it uses `huey_immediate`, proper `with patch(...)` context managers, in-memory SQLite via `db_engine`, and exercises real corpus assets from `assets/part6/`. The tests are reasonably self-contained and well-documented.

Four warnings were found involving incorrect docstring return type, a patch scope leak, unguarded dict key access on external data, and a potential `run_id` extraction crash. Three info items cover a file read outside the patch scope, a redundant variable read query, and the late import style.

## Warnings

### WR-01: `_submit_live_run` docstring claims wrong return type

**File:** `tests/test_web_run_review_e2e.py:72`
**Issue:** The docstring states "Returns the (response, run_id) tuple." but the function returns only `resp` (a single response object). This mismatch will mislead any reader who trusts the docstring to decide whether to unpack the return value.
**Fix:** Update the docstring to match the actual return value:
```python
# Change:
#   Returns the (response, run_id) tuple.
# To:
#   Returns the HTTP response from POST /runs/new.
```

---

### WR-02: Huey task patches exit before the task executes (patch scope leak)

**File:** `tests/test_web_run_review_e2e.py:77-95`
**Issue:** The `with patch(...)` block exits at line 95 after `client.post()` returns. With `huey_immediate=True` the task runs synchronously inside `.post()`, so the patches are active during task execution — this works today. However if Huey's immediate mode ever runs the task in a deferred or thread-pool fashion (or if the fixture order changes), `shop.tasks.SessionLocal` and `shop.tasks.OUT_DIR` will no longer be patched when the task executes. The pattern is fragile because correctness is implicitly coupled to the synchronous-execution guarantee of `huey_immediate`.

The downstream tests that call `_submit_live_run` and then immediately query `run.output_dir` assume the pipeline has fully finished before the patches exit. If the guarantee breaks, tests will query with a `None` output_dir and produce confusing failures.

**Fix:** Add an explicit assertion or comment that documents the synchronous-execution contract, and/or move the patches outside the submission call so they remain active for the full test duration:
```python
# In each test that calls _submit_live_run, hoist the patches to the test scope:
with patch("shop.tasks.SessionLocal", Session), \
     patch("shop.services.runs.UPLOADS_DIR", str(uploads_path)), \
     patch("shop.tasks.OUT_DIR", str(out_path)):
    resp = _submit_live_run(client, db_engine, engineer_user, uploads_path, out_path)
    # ... rest of the test assertions here
```
This makes the patch lifetime explicit and survives any future change to Huey's task scheduling.

---

### WR-03: `run_id` extracted with no guard against malformed redirect

**File:** `tests/test_web_run_review_e2e.py:137`
**Issue:** `run_id = int(location.split("/")[-1])` will raise `ValueError` if the redirect location ends with a trailing slash, is an error page URL, or contains a query string. For example `/runs/42?created=1` would produce `"42?created=1"` which cannot be cast to `int`. The prior assertion only checks `location.startswith("/runs/")`, which allows these variants through.

The same unguarded pattern appears at lines 222 and 354.

**Fix:**
```python
# Replace bare split/int with a more robust extraction:
parts = location.rstrip("/").split("/")
assert parts[-1].isdigit(), f"Could not extract run_id from redirect location: {location!r}"
run_id = int(parts[-1])
```

---

### WR-04: `packet.get("items", [])` iterates external JSON without field-presence guards

**File:** `tests/test_web_run_review_e2e.py:282-286`
**Issue:** The sign-off test reads the live delta packet and calls `.get("evaluation", {}).get("status")` on each item dict. This is safe as written. However, the outer `packet.get("items", [])` trusts that `items_data` is always a list — if the packet is malformed or a future pipeline change emits `items: null`, `any(... for item in items_data)` will raise `TypeError: 'NoneType' is not iterable`.

**Fix:**
```python
items_data = packet.get("items") or []
has_debug_exceptions = any(
    item.get("evaluation", {}).get("status") == "review_needed"
    for item in items_data
)
```

---

## Info

### IN-01: `_corpus_files()` reads PDF bytes outside the `with patch(...)` scope

**File:** `tests/test_web_run_review_e2e.py:51-59` (called at line 91)
**Issue:** `_corpus_files()` calls `REVA_PDF.read_bytes()`, `REVB_PDF.read_bytes()`, and `FAIR_XLSX.read_bytes()` at call time inside `_submit_live_run`, which is already inside the patch block, so there is no functional problem. The note is that if the assets directory is missing (e.g., in CI without the corpus), all three `read_bytes()` calls will raise `FileNotFoundError` with no informative message. The module-level path definitions (lines 29-32) do not guard against missing files.
**Fix:** Add a `pytest.skip` or `pytest.importorskip`-style guard at module or fixture level:
```python
def _corpus_files():
    for p in (REVA_PDF, REVB_PDF, FAIR_XLSX):
        if not p.exists():
            pytest.skip(f"Corpus asset not found: {p}. Skipping Phase 11 live tests.")
    return { ... }
```

---

### IN-02: Redundant DB query after sign-off to retrieve `output_dir`

**File:** `tests/test_web_run_review_e2e.py:270-275`
**Issue:** In `test_live_run_blocks_signoff_until_debug_queue_is_cleared`, `output_dir` is fetched from the DB in a second query after sign-off (lines 270-275). The same run was already queried on lines 234-239 and its `output_dir` is stable (it is set once during pipeline execution). Storing the value from the first query would eliminate the second session open/close cycle.
**Fix:** Capture `output_dir` at the first DB read and reuse it.

---

### IN-03: Late imports inside test functions

**File:** `tests/test_web_run_review_e2e.py:119`, `208`, `335`, `633`
**Issue:** `from shop.models import Run`, `from shop.models import ReviewItem`, `from shop.services.review import ...`, `import csv as csv_mod`, and `import io` appear inside test function bodies. This is a common pytest pattern to defer imports that may fail at collection time (the conftest.py comment on line 8 explains this rationale), but `csv` and `io` are stdlib modules with no conditional-import risk. Moving stdlib-only imports to the module top level would reduce clutter without any collection-time risk.
**Fix:** Move `import csv as csv_mod` and `import io` to the top of the file (they are stdlib and always available). Leave the `shop.*` deferred imports as-is per the established project pattern.

---

_Reviewed: 2026-04-18T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
