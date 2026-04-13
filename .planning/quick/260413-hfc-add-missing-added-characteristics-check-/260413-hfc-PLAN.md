---
phase: 260413-hfc
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - shop/services/review.py
  - tests/test_debug_row_identity.py
autonomous: true
requirements:
  - QUICK-260413-hfc
must_haves:
  truths:
    - "build_debug_queue_state returns missing_added_truth_indexes listing truth 'added' row indexes unclaimed by any packet row"
    - "debug_total in build_debug_queue_state counts both review_needed packet rows AND missing added truth rows"
    - "build_run_debug_summary includes missing added truth rows in exception_rows so debug_report_ready is False while any unclaimed added rows exist"
    - "If ground truth is unavailable (no fixture), missing_added check is skipped silently (no error)"
  artifacts:
    - path: "shop/services/review.py"
      provides: "Updated build_debug_queue_state and build_run_debug_summary with missing added coverage"
      contains: "missing_added_truth_indexes"
    - path: "tests/test_debug_row_identity.py"
      provides: "Regression tests for missing added truth detection"
      contains: "missing_added_truth_indexes"
  key_links:
    - from: "shop/services/review.py build_debug_queue_state"
      to: "delta_preservation.evaluation.loader.load_ground_truth_packet"
      via: "packet_data['inputs']['truth_fixture_key'] or run.part_number"
      pattern: "load_ground_truth_packet"
    - from: "ItemEvaluation.matched_truth_char_no"
      to: "claimed truth added index set"
      via: "parse 'added:N' tokens from packet row evaluations"
      pattern: "added:"
---

<objective>
Add a missing "added" characteristics check to the exception review queue.

Purpose: The exception queue currently surfaces only packet rows where evaluation.status == "review_needed". It misses the case where ground truth contains N "added" characteristics but the algorithm only captured M < N — the uncaptured ones are invisible. This gap means debug sign-off can succeed even when the algorithm missed added characteristics.

Output: Updated build_debug_queue_state returning missing_added_truth_indexes; updated build_run_debug_summary propagating those indexes as exception rows; regression tests covering the detection.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

Key facts established during planning:

- Ground truth is loaded via load_ground_truth_packet(key) from delta_preservation.evaluation.loader
- truth_fixture_key is stored in delta_packet.json under inputs["truth_fixture_key"]
- Fallback: run.part_number (normalized lowercase, strip spaces) is the fixture key
- Added truth rows have no char_no; they are matched to packet rows via "added:N" token in ItemEvaluation.matched_truth_char_no
- ADDED_POOL_TOKEN_PREFIX = "added" in conformance.py; token format is f"added:{truth_index}"
- build_debug_queue_state currently only adds ReviewItem to exception_items when evaluation.status == "review_needed"
- build_run_debug_summary derives debug_report_ready from unresolved_exception_count == 0
- GroundTruthContractError raised when fixture is missing/invalid — must be caught and silently skipped

<interfaces>
From delta_preservation.evaluation.loader:
```python
def load_ground_truth_packet(truth_fixture_key: str, repo_root: Path | None = None) -> GroundTruthPacket
# Raises GroundTruthContractError if fixture not found or invalid
```

From delta_preservation.evaluation.contracts:
```python
class GroundTruthCharacteristic(BaseModel):
    char_no: int | None
    classification: TruthClassification  # "unchanged"|"changed"|"removed"|"added"
    requirement_revB: str | None
    snippet_center_revA: tuple[float, float] | None
    snippet_center_revB: tuple[float, float] | None

class GroundTruthPacket(BaseModel):
    part_name: str
    general_notes: str
    characteristics: list[GroundTruthCharacteristic]
```

From delta_preservation.evaluation.conformance:
```python
ADDED_POOL_TOKEN_PREFIX = "added"  # token format: f"added:{truth_index}"
```

From delta_preservation.types:
```python
class ItemEvaluation(BaseModel):
    status: Literal["conforming", "review_needed"]
    matched_truth_char_no: int | str | None  # "added:N" for added pool matches
    ...

class DeltaItem(BaseModel):
    evaluation: ItemEvaluation | None
    status: str  # "unchanged"|"changed"|"removed"|"added"|"uncertain"
    ...
```

From shop.models:
```python
class Run(Base):
    part_number: str  # used as truth_fixture_key fallback
```

Current build_debug_queue_state return dict keys:
  all_items, exception_items, packet_items_by_item_id,
  raw_packet_items_by_item_id, packet_rows, packet_declares_items, debug_total
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Detect unclaimed truth added rows in build_debug_queue_state</name>
  <files>shop/services/review.py, tests/test_debug_row_identity.py</files>
  <behavior>
    - When ground truth has 2 "added" rows and the packet matched both (via "added:0" and "added:1"), missing_added_truth_indexes is []
    - When ground truth has 3 "added" rows and the packet only matched indexes 0 and 2 (skipped index 1), missing_added_truth_indexes is [1]
    - When ground truth has no "added" rows, missing_added_truth_indexes is []
    - When truth fixture is missing (GroundTruthContractError), missing_added_truth_indexes is [] and no exception is raised
    - debug_total = len(exception_items) + len(missing_added_truth_indexes)
    - build_run_debug_summary appends one synthetic exception_row dict per missing added truth index (row_state="missing_added_truth", char_no=None, queue_index continues sequence)
    - debug_report_ready = unresolved_exception_count == 0 still works correctly when missing added rows are counted
  </behavior>
  <action>
In shop/services/review.py:

1. Add import at top: `from delta_preservation.evaluation.contracts import GroundTruthContractError` and `from delta_preservation.evaluation import load_ground_truth_packet`.

2. Add helper function `_load_missing_added_truth_indexes` just before `build_debug_queue_state`:

```python
def _load_missing_added_truth_indexes(run: Run, packet_data: dict, packet_rows: list) -> list[int]:
    """Return truth added row indexes not claimed by any packet row.

    Silently returns [] when the ground truth fixture is unavailable.
    """
    inputs = packet_data.get("inputs") or {}
    truth_fixture_key = inputs.get("truth_fixture_key") or run.part_number
    if not truth_fixture_key:
        return []
    try:
        truth_packet = load_ground_truth_packet(truth_fixture_key)
    except (GroundTruthContractError, Exception):
        return []

    # Collect truth added row indexes
    truth_added_indexes = {
        i for i, ch in enumerate(truth_packet.characteristics)
        if ch.classification == "added"
    }
    if not truth_added_indexes:
        return []

    # Collect matched added token indexes from packet evaluations
    claimed: set[int] = set()
    for _item, delta_item in packet_rows:
        if delta_item.evaluation is None:
            continue
        token = delta_item.evaluation.matched_truth_char_no
        if not isinstance(token, str):
            continue
        if token.startswith(f"{ADDED_POOL_TOKEN_PREFIX}:"):
            try:
                idx = int(token.split(":", 1)[1])
                claimed.add(idx)
            except (ValueError, IndexError):
                pass

    return sorted(truth_added_indexes - claimed)
```

3. In `build_debug_queue_state`, after building `packet_rows`, add:

```python
    missing_added_truth_indexes = _load_missing_added_truth_indexes(run, packet_data, packet_rows)
```

4. Update the return dict to include it and adjust `debug_total`:

```python
    return {
        "all_items": all_items,
        "exception_items": exception_items,
        "missing_added_truth_indexes": missing_added_truth_indexes,
        "packet_items_by_item_id": packet_items_by_item_id,
        "raw_packet_items_by_item_id": raw_packet_items_by_item_id,
        "packet_rows": packet_rows,
        "packet_declares_items": "items" in packet_data,
        "debug_total": len(exception_items) + len(missing_added_truth_indexes),
    }
```

5. In `build_run_debug_summary`, after the loop over `packet_rows`, add synthetic exception rows for each missing added truth index:

```python
    for truth_index in queue_state.get("missing_added_truth_indexes", []):
        queue_index = len(conforming_rows) + len(exception_rows) + 1
        exception_rows.append({
            "queue_index": queue_index,
            "review_item_id": None,
            "char_no": None,
            "pipeline_classification": None,
            "requirement_revB": None,
            "saved_verdict": None,
            "mismatches": [{"code": "missing_added_characteristic", "message": f"ground truth added row {truth_index} was not captured by the algorithm"}],
            "packet_item": None,
            "row_state": "missing_added_truth",
        })
```

Note: `resolved_exception_count` does NOT increment for missing_added rows (no verdict possible) — their presence alone blocks `debug_report_ready`.

Also update `assemble_debug_report_payload` to pass `missing_added_truth_indexes` from queue_state into the returned payload under key `"missing_added_truth_indexes"`.

Write tests in tests/test_debug_row_identity.py:
- `test_missing_added_truth_indexes_detected_when_packet_misses_a_truth_added_row`: seed a packet with 1 "added" item matching truth index 0, but ground truth has 2 added rows (use tmp_path to write a fake ground_truth.json fixture). Verify missing_added_truth_indexes == [1] and debug_total == exception_count + 1.
- `test_no_missing_added_when_all_truth_added_rows_are_claimed`: ground truth has 1 added row, packet has 1 added item matching "added:0". Verify missing_added_truth_indexes == [].
- `test_missing_added_silently_skipped_when_no_fixture`: packet_data has no inputs field and run.part_number doesn't match any fixture. Verify missing_added_truth_indexes == [] and no exception.

For tests using a real fixture file: write a minimal ground_truth.json to `tmp_path / "assets" / normalized_key / "ground_truth.json"` and pass `repo_root=tmp_path` by monkey-patching `load_ground_truth_packet` or by using the loader directly in a helper. Since `_load_missing_added_truth_indexes` calls `load_ground_truth_packet` without repo_root, the easiest approach is to use `unittest.mock.patch` to mock `shop.services.review.load_ground_truth_packet` returning a controlled GroundTruthPacket.
  </action>
  <verify>
    <automated>cd /home/khoa2/delta-preservation && python -m pytest tests/test_debug_row_identity.py -x -q 2>&1 | tail -20</automated>
  </verify>
  <done>
    - build_debug_queue_state returns missing_added_truth_indexes (list of int) in its result dict
    - debug_total = len(exception_items) + len(missing_added_truth_indexes)
    - build_run_debug_summary appends synthetic rows for unclaimed truth added indexes
    - Missing fixture causes silent [] return, not an exception
    - All three new tests pass alongside existing test_duplicate_and_none_char_rows_keep_distinct_review_item_ids
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| filesystem → service | ground_truth.json read from assets/ directory |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-hfc-01 | Information Disclosure | _load_missing_added_truth_indexes | accept | Fixture file is read-only repo asset; GroundTruthContractError caught broadly; no user-controlled path traversal since key comes from packet_data inputs written by the pipeline, not from HTTP request |
</threat_model>

<verification>
Run full test suite to confirm no regressions:

```bash
cd /home/khoa2/delta-preservation && python -m pytest tests/test_debug_row_identity.py tests/test_debug_verdicts.py tests/test_focused_debug_queue.py tests/test_run_status_debug_summary.py -x -q
```
</verification>

<success_criteria>
- build_debug_queue_state["missing_added_truth_indexes"] is present and correct
- debug_total accounts for both review_needed rows and missing added truth rows
- build_run_debug_summary exception_rows includes synthetic missing_added_truth rows
- debug_report_ready is False while unclaimed added truth rows exist
- No fixture = silent [] (no crash)
- All existing debug queue tests continue to pass
</success_criteria>

<output>
After completion, create `.planning/quick/260413-hfc-add-missing-added-characteristics-check-/260413-hfc-01-SUMMARY.md`
</output>
