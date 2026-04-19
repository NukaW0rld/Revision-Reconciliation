---
phase: 10-debug-exception-gating-and-advisory-surfacing
reviewed: 2026-04-18T00:00:00Z
depth: standard
files_reviewed: 14
files_reviewed_list:
  - shop/services/review.py
  - shop/routers/review.py
  - shop/services/exports.py
  - shop/routers/exports.py
  - shop/templates/review/_item_card.html
  - shop/templates/review/_item_card_debug.html
  - shop/templates/review/_signoff_footer.html
  - shop/templates/review/_signoff_footer_oob.html
  - shop/templates/runs/status.html
  - shop/templates/exports/audit_packet.html
  - shop/templates/exports/work_order.html
  - tests/test_review.py
  - tests/test_debug_verdicts.py
  - tests/test_run_status_debug_summary.py
  - tests/test_exports.py
findings:
  critical: 0
  warning: 5
  info: 4
  total: 9
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-04-18
**Depth:** standard
**Files Reviewed:** 14
**Status:** issues_found

## Summary

This phase implements debug exception gating, signed debug snapshots, and advisory flag
surfacing across the review, export, and status-page surfaces. The overall architecture is
sound: the two-phase sign-off, atomic file writes with `NamedTemporaryFile`/`replace`, and
the strict/lenient dual-loader pattern for verdicts are all well-structured.

Five warnings were found. The most consequential are a bare `except Exception` that silently
swallows all errors in the sign-off gate (causing the gate to appear clear when it should not),
unhandled `OSError` / `json.JSONDecodeError` in two hot paths that read JSON without a
try/except, a double-invocation of expensive services in a single request cycle, and a
path-traversal gap in the snippet-serving route that only checks for `..` but not for other
encoded separators. No critical security vulnerabilities or data-loss bugs were found.

---

## Warnings

### WR-01: Bare `except Exception` in `build_signoff_gate_state` silently clears the gate on any error

**File:** `shop/services/review.py:746`
**Issue:** The `except Exception` block at line 746 falls through to
`unresolved_exception_count = 0`, which means **any** runtime error (e.g. a corrupt
`delta_packet.json`, a DB inconsistency, an import failure) will cause the gate to appear
clear and allow sign-off even when exceptions genuinely remain unresolved. This is the
opposite of the intended fail-safe: errors should block sign-off, not permit it.

```python
    try:
        queue_state = build_debug_queue_state(db, run, activate_review=False)
        ...
    except Exception:
        # If debug summary can't be built (no packet yet), treat as 0 unresolved
        unresolved_exception_count = 0   # <-- gate is cleared on ANY error
```

**Fix:** Narrow the exception to the known "packet not yet present" case
(`DebugVerdictValidationError`) and re-raise or block on all others:

```python
    try:
        queue_state = build_debug_queue_state(db, run, activate_review=False)
        ...
    except DebugVerdictValidationError:
        # Packet not yet available — treat as no exceptions (pre-pipeline run)
        unresolved_exception_count = 0
    except Exception:
        # Unknown error: block sign-off conservatively
        unresolved_exception_count = 1
```

---

### WR-02: `load_debug_verdicts` reads a JSON file without exception handling — crashes if file is unreadable

**File:** `shop/services/review.py:127`
**Issue:** `load_debug_verdicts` calls `json.loads(verdicts_path.read_text())` with no
try/except. If the file exists but is corrupted, locked, or truncated during an in-progress
atomic write (race window between `NamedTemporaryFile` creation and `replace`), the call
raises `OSError` or `json.JSONDecodeError` and propagates uncaught to callers such as
`save_debug_verdict`, `assemble_debug_report_payload`, and `write_signed_debug_snapshot`,
causing a 500 response or a failed sign-off.

```python
    raw_data = json.loads(verdicts_path.read_text())   # line 127 — no try/except
```

The lenient variant (`load_debug_verdicts_for_render`) correctly wraps the same read in a
try/except; the strict variant should do the same and re-raise as `DebugVerdictValidationError`:

```python
    try:
        raw_data = json.loads(verdicts_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DebugVerdictValidationError(
            f"debug_verdicts.json could not be read: {exc}"
        ) from exc
```

---

### WR-03: `_load_delta_packet` (in `exports.py`) reads JSON without exception handling — crashes on corrupt packet

**File:** `shop/services/exports.py:97`
**Issue:** `_load_delta_packet_items` calls `json.loads(packet_path.read_text())` at line 97
with no try/except. A corrupt `delta_packet.json` on a signed-off run causes an unhandled
`json.JSONDecodeError` that propagates through `semantic_contracts_by_char`, which is called
from both `generate_audit_packet_csv` and `render_audit_packet_pdf`, crashing both export routes
with a 500 instead of a meaningful error.

```python
    packet_data = json.loads(packet_path.read_text())   # line 97 — no try/except
```

**Fix:**

```python
    try:
        packet_data = json.loads(packet_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
```

---

### WR-04: `_render_debug_item_card` invokes `debug_internals_by_item_id` and `semantic_contracts_by_item_id` independently — both rebuild `build_debug_queue_state` internally, causing triple invocation per verdict POST

**File:** `shop/routers/review.py:164-171`
**Issue:** `_render_debug_item_card` calls:
1. `build_debug_queue_state(db, run, ...)` directly (line 165)
2. `semantic_contracts_by_item_id(db, run)` — which also calls `build_debug_queue_state` internally
3. `debug_internals_by_item_id(db, run)` — which also calls `build_debug_queue_state` internally
4. `advisory_flags_by_item_id(db, run)` — which also calls `build_debug_queue_state` internally

Each `build_debug_queue_state` call reads `delta_packet.json` from disk, queries all
`ReviewItem` rows, and may query `load_ground_truth_packet`. A single verdict POST triggers
four disk reads and four DB query bursts for a packet that does not change between calls.

**Fix:** Pass the pre-built `queue_state` into the helper functions rather than
re-building it on each invocation. Alternatively, factor a single `build_debug_queue_state`
call at the top of `_render_debug_item_card` and thread it through to each sub-call.

---

### WR-05: Snippet-serving route path traversal — `..` check is bypassable on some OS/URL-decode combinations

**File:** `shop/routers/review.py:653`
**Issue:** The guard on the snippet route is:

```python
if ".." in filename or "/" in filename or not filename.endswith(".png"):
```

This checks for literal `..` and `/`, but does not check for `\` (Windows backslash) or
URL-encoded variants (`%2F`, `%2e%2e`). On Linux, FastAPI decodes path parameters before the
handler receives them, so `%2F` in a path *segment* would be rejected by the router before
reaching Python. However, `\` is not treated as a separator on Linux, allowing a filename
like `foo\..\..\etc\passwd.png` to pass the guard and attempt a path join. The resulting
`Path(run.output_dir) / "snippets" / filename` would not escape the snippets directory on
Linux because `\` is a literal character in the filename, so the actual exploit risk is low.
But a safer guard is more explicit and future-proof:

```python
safe_name = Path(filename).name
if safe_name != filename or not filename.endswith(".png"):
    raise HTTPException(status_code=400, detail="Invalid filename")
```

`Path(filename).name` strips any directory components regardless of separator, and the
equality check ensures no separator characters were present.

---

## Info

### IN-01: Dead code — `semantic_contract_for_item` is defined but never called

**File:** `shop/services/review.py:488-491`
**Issue:** `semantic_contract_for_item` is defined at line 488 and is not imported anywhere
in the reviewed files. It duplicates logic that callers implement inline. If intentionally
retained for future use it should be documented; otherwise it should be removed.

---

### IN-02: Duplicate footer template — `_signoff_footer.html` and `_signoff_footer_oob.html` are identical except for `hx-swap-oob`

**File:** `shop/templates/review/_signoff_footer.html`, `shop/templates/review/_signoff_footer_oob.html`
**Issue:** Both templates contain 46 identical lines of logic. Any future logic change must be
applied in both files. Consider extracting a shared `_signoff_footer_body.html` macro and
having both templates include it, differing only in the wrapping `<div>` attributes.

---

### IN-03: `assemble_debug_report_payload` increments `debug_submitted` only for exception items, not for `missing_added_truth_items` that were resolved

**File:** `shop/services/review.py:400-401, 440-441`
**Issue:** The logic for incrementing `debug_submitted` in the main loop (line 400) requires
`item.id in exception_item_ids`, so conforming rows with a verdict do not count. For missing
added rows (lines 440-441), `debug_submitted += 1` runs unconditionally when `verdict_payload`
is not None. These two counters use inconsistent rules. For the main loop, the intent is
correct (only exception rows count toward the submitted total). For the missing-added loop,
the same intent is expressed without the `exception_item_ids` guard — which is also correct
since all missing-added rows are exceptions by definition. No actual bug exists here, but the
asymmetric code style makes the intent harder to verify. A clarifying comment stating "all
missing-added rows are exceptions; no guard needed" would make this safer to maintain.

---

### IN-04: `test_review.py` has unreachable test body at lines 503-528

**File:** `tests/test_review.py:503-528`
**Issue:** The block starting at line 503 contains a docstring-like string literal
(`"""GET /review/{run_id}...`), followed by test code that uses `run_id` and `char_no`
variables that do not exist in scope. This code is not inside any function — it is at the
module level after `test_review_queue_surfaces_fallback_reason_without_empty_semantic_scaffolding`
ends. The code will never execute as a test. It appears to be a test body that was accidentally
left outside its function definition (the function name and `def` line were removed or
never added). This test case is silently lost.

**Fix:** Wrap the body in a properly named test function:

```python
def test_review_queue_renders_all_items_and_transitions_to_reviewing(
    client: TestClient, engineer_user, db_engine, tmp_path
):
    """GET /review/{run_id} returns 200 with all ReviewItems listed."""
    _login_engineer(client, db_engine, engineer_user)
    run_id = _make_run_with_packet(tmp_path, db_engine, n_items=3, status="completed")
    # ... rest of the body
```

---

_Reviewed: 2026-04-18_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
