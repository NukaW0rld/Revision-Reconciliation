# Phase 6: Added Characteristic Detection and Snippet Accuracy — Pattern Map

**Mapped:** 2026-04-16
**Files analyzed:** 9
**Analogs found:** 9 / 9 — every planned touchpoint has a clear in-repo analog.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `delta_preservation/reconcile/exclusion.py` (NEW) | shared utility | pure helper / search guardrail | `delta_preservation/reconcile/match.py::_is_boilerplate_candidate_text`, `_span_is_excluded_for_matching` | strong |
| `delta_preservation/reconcile/anchors.py` (MODIFY) | anchor builder | Rev A text candidate filtering | existing inline page-edge filter in same file | exact |
| `delta_preservation/reconcile/match.py` (MODIFY) | candidate generation | Rev B candidate prefilter + grouping | existing exclusion helpers and `_group_candidate_spans` | exact |
| `delta_preservation/reconcile/classify.py` (MODIFY) | rescue scan + added detection | grouped added evidence and suppression | `detect_added_characteristics()`, `reconcile_removed_added_pairs()` | exact |
| `delta_preservation/cli.py` (MODIFY) | packet assembly | internal DeltaItem -> packet DeltaItem | current matched-annotation text formatting path | exact |
| `delta_preservation/evaluation/conformance.py` (MODIFY) | canonical truth selection | packet item -> truth row | `select_truth_row_for_item()` current exact-text path | exact |
| `tests/test_alignment_multishift.py` (MODIFY) | focused unit test | exclusion smoke for matching | existing `test_generate_candidates_excludes_tolerance_block_boilerplate` | exact |
| `tests/test_debug_row_identity.py` (MODIFY) | evaluator behavior test | missing-added and synthetic truth coverage | existing missing-added tests | exact |
| `tests/test_phase6_asset_regression.py` / other new phase-6 tests (NEW) | regression harness | read-only asset-backed assertions | `tests/test_classify_phase5_regression.py` | strong |

## Pattern Assignments

### Shared exclusion helper

**Analog:** `delta_preservation/reconcile/match.py` lines around `_is_boilerplate_candidate_text()` and `_span_is_excluded_for_matching()`

**Pattern to preserve**

- normalize text with `" ".join(text.strip().upper().split())`
- keep exact/contains/pattern boilerplate checks in one place
- apply geometric zones using span centers, not raw `x0`/`y0`
- estimate page dimensions with minimum page-size floors

**Planning implication**

Move these rules into a dedicated helper module and let `match.py` import them back, instead of making other modules import private helpers from `match.py`.

### Added grouping contract

**Analog:** `delta_preservation/reconcile/match.py::_group_candidate_spans()`

**Pattern to preserve**

- group nearby companion spans into one synthetic callout
- keep a representative seed span plus union bbox plus ordered text
- dedupe by span key and preserve stable ordering

**Planning implication**

Phase 6 should reuse the same grouping idea for the standard added pass so packet rows stop collapsing to raw fragments.

### Internal added evidence fields

**Analog:** `delta_preservation/reconcile/classify.py::DeltaItem`

**Pattern to preserve**

- keep internal-only evidence on the mutable dataclass
- populate packet-facing evidence in `cli.py`
- avoid storing implementation-only metadata in the persisted Pydantic model unless the packet contract truly needs it

**Planning implication**

Prefer `added_requirement_text` and `added_bbox` as the canonical internal evidence for added rows. Use `added_span` only as the representative seed / provenance handle.

### Packet-side annotation formatting

**Analog:** `delta_preservation/cli.py` matched-item path using `_format_annotation_text(...)` and expanded annotation spans

**Pattern to preserve**

- format requirement text from grouped annotation spans, not from a seed span alone
- derive snippet bbox from the full annotation extent before expansion
- keep snippet rule family explicit (`single_callout` vs `grouped_callout`)

**Planning implication**

Added rows should follow the same packet assembly pattern instead of taking `span.text` directly.

### Duplicate truth selection

**Analog:** `delta_preservation/evaluation/conformance.py::select_truth_row_for_item()`

**Pattern to preserve**

- deterministic selection first
- conservative ambiguity when evidence is insufficient
- reserve added truth indexes only after selection succeeds

**Secondary analog:** `delta_preservation/evaluation/snippet_rules.py`

**Useful idea to reuse**

- bbox coercion
- center-in-bbox checks
- conservative fallback when geometry is missing or invalid

### Phase-local regression harness

**Analog:** `tests/test_classify_phase5_regression.py`

**Pattern to preserve**

- use checked-in assets as read-only fixtures
- assert exact exemplar strings / coordinates instead of vague counts
- keep phase-local harness deterministic and fast

## Recommended File Set For Plans

### Plan 01

- `delta_preservation/reconcile/exclusion.py`
- `delta_preservation/reconcile/anchors.py`
- `delta_preservation/reconcile/match.py`
- `delta_preservation/reconcile/classify.py`
- `tests/test_phase6_exclusion.py`
- `tests/test_alignment_multishift.py`

### Plan 02

- `delta_preservation/reconcile/match.py`
- `delta_preservation/reconcile/classify.py`
- `delta_preservation/cli.py`
- `tests/test_added_detection_phase6.py`

### Plan 03

- `delta_preservation/evaluation/conformance.py`
- `tests/test_added_truth_selection.py`
- `tests/test_debug_row_identity.py`

### Plan 04

- `tests/test_phase6_asset_regression.py`

## Anti-Patterns To Avoid

- importing private `_span_is_excluded_for_matching()` directly from `match.py` into unrelated modules
- teaching the evaluator about Part 9 by hardcoding truth indexes
- emitting added packet rows from raw `span.text` when grouped evidence is available
- suppressing added rows by proximity alone without checking content ownership
