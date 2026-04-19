---
phase: 10-debug-exception-gating-and-advisory-surfacing
fixed_at: 2026-04-18T00:00:00Z
fix_scope: critical_warning
findings_in_scope: 5
fixed: 5
skipped: 0
iteration: 1
status: all_fixed
commits:
  - dfb9214
  - f0b3958
  - e9f1a06
---

# Phase 10: Code Review Fix Report

**Fix scope:** critical + warning (default)
**Findings in scope:** 5
**Fixed:** 5
**Skipped:** 0
**Status:** all_fixed

---

## Fixes Applied

### WR-01 — Narrowed bare `except Exception` in `build_signoff_gate_state`

**File:** `shop/services/review.py`
**Commit:** dfb9214

The bare `except Exception` was silently clearing `unresolved_exception_count = 0` on any
runtime error, allowing sign-off when the gate should have been blocked. Fixed by splitting
into two branches: `DebugVerdictValidationError` (expected "no packet yet" case) keeps
`unresolved_exception_count = 0`; all other unknown exceptions now set it to `1`, blocking
sign-off conservatively.

```python
except DebugVerdictValidationError:
    # Packet not yet available — treat as no exceptions (pre-pipeline run)
    unresolved_exception_count = 0
except Exception:
    # Unknown error: block sign-off conservatively rather than silently clearing the gate
    unresolved_exception_count = 1
```

---

### WR-02 — Added `OSError`/`JSONDecodeError` guard to `load_debug_verdicts`

**File:** `shop/services/review.py`
**Commit:** dfb9214

The bare `json.loads(verdicts_path.read_text())` at line 127 could raise `OSError` or
`json.JSONDecodeError` on a corrupt or mid-write file, propagating uncaught to callers.
Wrapped in a try/except that re-raises as `DebugVerdictValidationError`, consistent with the
lenient variant's pattern.

```python
try:
    raw_data = json.loads(verdicts_path.read_text())
except (OSError, json.JSONDecodeError) as exc:
    raise DebugVerdictValidationError(
        f"debug_verdicts.json could not be read: {exc}"
    ) from exc
```

---

### WR-03 — Added `OSError`/`JSONDecodeError` guard to `_load_delta_packet_items`

**File:** `shop/services/exports.py`
**Commit:** f0b3958

The bare `json.loads(packet_path.read_text())` at line 97 raised an unhandled
`json.JSONDecodeError` on a corrupt packet, crashing both export routes with a 500. Wrapped
in a try/except that returns `[]`, matching the existing "no file" early-return pattern.

```python
try:
    packet_data = json.loads(packet_path.read_text())
except (OSError, json.JSONDecodeError):
    return []
```

---

### WR-04 — Eliminated redundant `build_debug_queue_state` invocations in `_render_debug_item_card`

**Files:** `shop/services/review.py`, `shop/routers/review.py`
**Commit:** e9f1a06 (router), dfb9214 (service)

Each of `semantic_contracts_by_item_id`, `debug_internals_by_item_id`, and
`advisory_flags_by_item_id` internally called `build_debug_queue_state`, causing four total
disk reads and DB query bursts per verdict POST (the direct call in `_render_debug_item_card`
plus one per helper). Fixed by adding an optional keyword-only `queue_state: dict | None = None`
parameter to all three service functions. When supplied, the internal rebuild is skipped.
`_render_debug_item_card` now passes its pre-built `debug_queue_state` through to all three
helpers, reducing invocations from 4 to 1.

---

### WR-05 — Hardened snippet-serving route against path-traversal

**File:** `shop/routers/review.py`
**Commit:** e9f1a06

The original `".." in filename or "/" in filename` guard did not cover backslash separators
or other encoded variants. Replaced with a `Path(filename).name` equality check, which strips
any directory component regardless of separator type and is safe across OS/encoding
combinations.

```python
safe_name = _Path(filename).name
if safe_name != filename or not filename.endswith(".png"):
    raise HTTPException(status_code=400, detail="Invalid filename")
```

---

## Info Findings (not in scope)

The following Info findings were not addressed in this pass (default scope is critical + warning):

- **IN-01** — Dead code `semantic_contract_for_item` never called (`shop/services/review.py:488`)
- **IN-02** — Duplicate footer templates (`_signoff_footer.html` / `_signoff_footer_oob.html`)
- **IN-03** — Asymmetric `debug_submitted` increment style in `assemble_debug_report_payload`
- **IN-04** — Unreachable test body at `tests/test_review.py:503-528` (missing `def` line)

---

_Fixed: 2026-04-18_
_Fixer: Cascade (gsd-code-fixer)_
_Tests passing: 70/70_
