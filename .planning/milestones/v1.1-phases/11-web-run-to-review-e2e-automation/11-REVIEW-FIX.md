---
phase: 11-web-run-to-review-e2e-automation
status: all_fixed
findings_in_scope: 4
fixed: 4
skipped: 0
iteration: 1
fix_scope: critical_warning
commit: fix(11): apply code review fixes WR-01..WR-04
---

# Phase 11: Code Review Fix Report

**Fixed:** 2026-04-19T00:00:00Z
**Scope:** Critical + Warning only (4 findings)
**Status:** all_fixed

## Fixes Applied

### WR-01: `_submit_live_run` docstring claims wrong return type

**File:** `tests/test_web_run_review_e2e.py:72`
**Action:** Updated docstring from "Returns the (response, run_id) tuple." to "Returns the HTTP response from POST /runs/new." to match the actual single-value return.
**Commit:** `fix(11): apply code review fixes WR-01..WR-04`

---

### WR-02: Huey task patches exit before the task executes (patch scope leak)

**File:** `tests/test_web_run_review_e2e.py:77-79`
**Action:** Added a 3-line comment block documenting the synchronous-execution contract: `huey_immediate` runs the task inline inside `client.post()`, so patches are guaranteed active. The comment also flags the migration path if Huey's immediate mode ever becomes async.
**Commit:** `fix(11): apply code review fixes WR-01..WR-04`

---

### WR-03: `run_id` extracted with no guard against malformed redirect

**File:** `tests/test_web_run_review_e2e.py:140-142`, `227-229`, `361-363`
**Action:** Replaced bare `int(location.split("/")[-1])` at all 3 call sites with a robust pattern: `location.rstrip("/").split("/")` followed by `assert parts[-1].isdigit()` before the `int()` cast. This guards against trailing slashes, query strings, and non-numeric path segments.
**Commit:** `fix(11): apply code review fixes WR-01..WR-04`

---

### WR-04: `packet.get("items", [])` iterates external JSON without field-presence guards

**File:** `tests/test_web_run_review_e2e.py:289`
**Action:** Changed `packet.get("items", [])` to `packet.get("items") or []`. The `or []` idiom handles both missing keys (returns `None` default → `[]`) and explicit `null` values in the JSON (`None or [] → []`).
**Commit:** `fix(11): apply code review fixes WR-01..WR-04`

---

## Out of Scope (Info — not included in fix pass)

- **IN-01:** `_corpus_files()` reads PDF bytes without skip guard for missing assets
- **IN-02:** Redundant DB query for `output_dir` after sign-off
- **IN-03:** Late stdlib imports (`csv`, `io`) inside test functions

_Fixed: 2026-04-19T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
