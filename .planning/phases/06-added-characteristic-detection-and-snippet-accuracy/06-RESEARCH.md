# Phase 6: Added Characteristic Detection and Snippet Accuracy - Research

**Researched:** 2026-04-16
**Domain:** Shared reconcile-path fixes for added-characteristic detection, duplicate added-truth claiming, and snippet/search-window exclusion in `delta_preservation/reconcile/*`, `delta_preservation/cli.py`, and `delta_preservation/evaluation/conformance.py`.
**Confidence:** HIGH (findings verified against current source, Phase 6 context, and the checked-in Part 8 / Part 9 debug artifacts)

<user_constraints>
## User Constraints

Phase 6 already has locked context in `06-CONTEXT.md`; this research translates that context into planning-ready implementation guidance.

### Locked Decisions

- **ADD-01:** All canonical added rows must be emitted by the shared pipeline path. Parts 8 and 9 are the current failing corpus members and cannot be solved with per-part overrides.
- **ADD-02:** False-positive added rows must be suppressed only when an existing matched/grouped characteristic already explains the same content. A blunt proximity-only suppressor is not acceptable.
- **SNP-01:** Title block, revision table, and general-tolerance boilerplate must be excluded consistently across anchor lookup, candidate generation, rescue scans, and added detection.
- `ground_truth.json` is canonical. Phase 6 must improve packet output and evaluator selection without editing truth files.
- The fix belongs in the shared reconcile / packet-generation / evaluation path. `shop/services/review.py` remains a consumer.
- Duplicate added requirements in Part 9 must be claimable deterministically from richer packet evidence, not manual mappings.

### Out of Scope

- Web UI or review workflow changes
- Automatic ground-truth edits
- Part-specific hardcoded mappings
- Phase 7's full 9-part rerun gate (`VER-01`) beyond phase-local regression harnesses

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Requirement | Research Support |
|----|-------------|------------------|
| ADD-01 | All ground-truth-added characteristics appear in pipeline output for every debug-corpus part. | Part 8 currently leaves `missing_added_truth_indexes=[10]`; Part 9 leaves `[35..42]`. Root causes split across added detection and duplicate truth claiming. |
| ADD-02 | Added false positives are suppressed when the unmatched fragment is already fully explained by an existing matched characteristic. | Current standard added pass works span-by-span and emits fragment rows such as `.045 A` and truncated `⌖∅`; suppression logic is inconsistent and mostly proximity-based. |
| SNP-01 | Title block / revision table / boilerplate regions are excluded consistently from search windows and snippet targeting. | `anchors.py`, `classify.py`, and `detect_added_characteristics()` each carry their own exclusion heuristics; only `match.py` has the richer boilerplate-aware contract. |

</phase_requirements>

## Summary

Phase 6 is not one bug. It is four interacting gaps in the current pipeline contract:

1. `match.py` already has a reasonably good title-block / boilerplate exclusion helper, but `anchors.py`, the `classify_delta()` rescue scans, and `detect_added_characteristics()` still use simpler, duplicated geometry-only rules. The same drawing region can therefore be filtered in one stage and accepted in another.
2. `detect_added_characteristics()` preserves grouped evidence for GD&T and stacked-limit rows, but the standard pass still emits rows from single raw spans. That is why Part 8 can surface `.045 A` instead of `◎ ∅.045 A`, and Part 9 can surface `⌖∅` instead of `⌖ ∅.015 D H`.
3. The current standard added detector knows how to skip spans inside matched bboxes or near matched centers, but it does not ask whether a surviving unmatched/grouped span is already semantically explained by an existing matched characteristic. That leaves the Part 8-style false-positive added rows alive.
4. `select_truth_row_for_item()` in `evaluation/conformance.py` only resolves added rows by normalized requirement text. When two or more canonical added rows share the same normalized text, it returns `truth_ambiguity` immediately instead of using the packet's Rev B bbox / snippet evidence to disambiguate them.

These gaps line up directly with the observed debug corpus failures:

- **Part 8**
  - `missing_added_truth_indexes=[10]`
  - false-positive added rows for the changed balloon-5 / balloon-6 cluster
  - a missing canonical added row `⌰ .002 A` even though related text is present elsewhere in the packet
- **Part 9**
  - eight missing canonical added rows
  - a truncated added row `⌖∅`
  - duplicate added requirements (`Ø.250 ±.008`, `⌖ ∅.015 D H`, `↧.50 ±.05`) that the evaluator cannot currently claim deterministically

The clean planning split is:

- shared exclusion contract
- grouped added-evidence contract plus explained-by-match suppression
- deterministic duplicate added-truth claiming
- asset-backed phase-local regression coverage

## Verified Failure Evidence

### Part 8

- `assets/debug_report_part8.json` records `missing_added_truth_indexes: [10]`.
- The checked-in Part 8 ground truth has three added rows:
  - `Ø10.000±.001`
  - `⌰ .015 B`
  - `⌰ .002 A`
- The debug report also shows:
  - false-positive added row for `⚪ .005` that actually belongs to changed balloon 5
  - false-positive added row for `.045 A` / `◎ ∅.045 A` that actually belongs to changed balloon 6
  - a changed row whose `requirement_revB` was incorrectly filled with `⌰ .002 A`

### Part 9

- `assets/debug_report_part9.json` records `missing_added_truth_indexes: [35, 36, 37, 38, 39, 40, 41, 42]`.
- The checked-in Part 9 ground truth has eight added rows:
  - two `Ø.250 ±.008`
  - two `⌖ ∅.015 D H`
  - two `↧.50 ±.05`
  - one `⌓ .02 A B C`
  - one `⏥ .01`
- The debug report shows at least one truncated added packet row `⌖∅`, which proves the packet currently loses grouped evidence before evaluation.

## Root Cause Breakdown

### Cluster 1: Shared exclusion logic is fractured

**Current state**

- `delta_preservation/reconcile/match.py`
  - `_is_boilerplate_candidate_text()`
  - `_span_is_excluded_for_matching()`
  - page-dimension estimation with 612x792 floors
- `delta_preservation/reconcile/anchors.py`
  - only skips `y0 > page_height * 0.85` or `x0 > page_width * 0.80`
  - no shared boilerplate-text contract
- `delta_preservation/reconcile/classify.py`
  - keyword rescue scan hardcodes its own keyword list
  - `detect_added_characteristics()` defines a local `is_in_exclusion_zone()` that only knows top-right / bottom-right corners

**Impact**

- A span can be excluded during candidate generation but still appear in added detection.
- Bottom-center tolerance-block text is explicitly filtered in `match.py` but not in `anchors.py` or `detect_added_characteristics()`.
- Phase 6 cannot solve `snippet_outside_*` or search-window drift consistently while these rules remain split.

**Planning implication**

Create one shared exclusion utility and make all four search surfaces import it:

- `build_revA_anchors()`
- `generate_candidates()`
- `classify_delta()` rescue scans
- `detect_added_characteristics()`

### Cluster 2: Standard added detection still reasons from raw seed spans

**Current state**

- GD&T pass records grouped text + union bbox.
- stacked-limit pass records pair text + union bbox.
- standard pass records only:
  - `added_span=span`
  - `added_requirement_text=span.text`
  - `added_bbox=span.bbox_pdf`

**Impact**

- Split annotations collapse to fragments in the packet.
- Added rows that should be full callouts become unclaimable or ambiguous.
- CLI builds added `requirement_revB` from expanded `added_span` context, not from a canonical grouped evidence contract; that makes Phase 6 vulnerable to both truncation and over-expansion.

**Planning implication**

The standard added pass needs the same kind of grouped-evidence contract already used elsewhere:

- grouped text
- union bbox
- representative seed span only as a fallback handle
- deterministic formatting path into packet `requirement_revB`

### Cluster 3: False-positive added rows are not checked against matched ownership

**Current state**

- `detect_added_characteristics()` suppresses some cases by geometry:
  - span key already matched
  - span inside matched group bbox
  - near matched center
  - stacked pair already present in matched numeric pairs
- It does **not** answer the stronger question:
  - "Is this unmatched/grouped span already fully explained by a matched characteristic's grouped annotation?"

**Impact**

- Part 8-style fragments survive as false-positive added rows.
- The pipeline can simultaneously:
  - assign an added-looking fragment to a changed/matched item
  - emit another added row from the companion fragment

**Planning implication**

Add an explained-by-match suppressor that compares grouped text + bbox ownership against already matched annotations. It should be content-aware first, geometry-aware second.

### Cluster 4: Duplicate added truth rows are ambiguous because selection stops at normalized text

**Current state**

`select_truth_row_for_item()` currently does:

1. gather unmatched canonical added rows
2. normalize `item.requirement_revB`
3. match exact normalized requirement text
4. if one row matches: accept
5. if multiple rows match: return `truth_ambiguity`

**Impact**

- Part 9's duplicate added rows can never become conforming even when packet bbox evidence clearly points at one location.
- Missing-added queues stay inflated even after the detector finds the right strings.

**Planning implication**

Use packet-side Rev B evidence for a second-stage tie-break:

- first choice: unique truth center inside packet `revB.bbox`
- second choice: unique nearest truth center to packet bbox center within a bounded tolerance
- otherwise: keep current conservative ambiguity outcome

## Recommended Fix Clusters

### Fix Cluster A: Shared exclusion module

**Recommended location:** `delta_preservation/reconcile/exclusion.py`

**Why**

- `match.py` already owns the best rules.
- `anchors.py` and `classify.py` need the same contract, but importing private helpers from `match.py` would deepen coupling.

**Functions worth centralizing**

- `estimate_page_dimensions(spans, *, min_width=612.0, min_height=792.0)`
- `is_boilerplate_candidate_text(text)`
- `span_is_excluded_for_annotation_search(span, *, page_width, page_height)`

**Callers**

- `build_revA_anchors()`
- `generate_candidates()`
- keyword rescue in `classify_delta()`
- `detect_added_characteristics()`

### Fix Cluster B: Added grouping and explained-by-match suppression

**Recommended ownership:** `delta_preservation/reconcile/classify.py`

**Why**

- The added detector already owns all three passes and the `DeltaItem` contract.
- The matched-item ownership data it needs is already assembled there from `matches`.

**Key behaviors**

- standard pass should build grouped candidates, not raw seed spans
- packet-facing evidence should prefer `added_requirement_text` / `added_bbox`
- a surviving added row must prove it is not already explained by a matched grouped annotation

### Fix Cluster C: Duplicate added truth tie-break by packet evidence

**Recommended ownership:** `delta_preservation/evaluation/conformance.py`

**Why**

- Canonical selection belongs at evaluation time, not during detection.
- Packet rows already carry `revB.bbox`, `requirement_revB`, and snippet rule family.

**Conservative rule set**

- unique exact normalized text match -> accept (keep current behavior)
- multiple exact matches:
  - if exactly one truth `snippet_center_revB` lies inside packet `revB.bbox`, accept it
  - else if exactly one truth center is the unique nearest to packet bbox center within a bounded threshold, accept it
  - else ambiguity

### Fix Cluster D: Asset-backed phase-local regression harness

**Recommended location:** `tests/test_phase6_asset_regression.py`

**Why**

- Phase 7 will own the full corpus rerun.
- Phase 6 still needs a deterministic, read-only harness tied to the actual failure exemplars.

**Use read-only sources**

- `assets/debug_report_part8.json`
- `assets/debug_report_part9.json`
- `assets/part8/ground_truth.json`
- `assets/part9/ground_truth.json`

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Why |
|------------|--------------|----------------|-----|
| shared exclusion contract | `delta_preservation/reconcile/exclusion.py` | callers in `anchors.py`, `match.py`, `classify.py` | one source of truth for geometry + boilerplate rejection |
| grouped added-evidence contract | `delta_preservation/reconcile/classify.py` | `delta_preservation/cli.py` | detector owns evidence construction; CLI owns packet formatting |
| explained-by-match suppressor | `delta_preservation/reconcile/classify.py` | existing grouped-match metadata from `matches` | suppression depends on detector-local ownership analysis |
| duplicate added truth tie-break | `delta_preservation/evaluation/conformance.py` | `delta_preservation/evaluation/snippet_rules.py` patterns as analogs | canonical truth selection belongs in evaluation |
| queue / report surfacing | `shop/services/review.py` | none | should improve automatically once packet + evaluator behavior improve |

## Standard Stack

| Component | Use in Phase 6 | Notes |
|-----------|----------------|-------|
| `pytest` via `uv run pytest` | phase-local fast verification | already standard in repo |
| `re` / `math` | shared exclusion and distance tie-breaks | already used heavily in reconcile/evaluation code |
| `jq` / checked-in JSON assets | regression exemplars only | read-only fixture usage |
| `pydantic` packet models | no schema redesign required | packet already carries Rev B bbox evidence |

## Validation Architecture

Phase 6 can satisfy Nyquist without a full corpus rerun by using a narrow, layered test stack:

1. **Shared exclusion smoke**
   - `uv run pytest tests/test_phase6_exclusion.py tests/test_alignment_multishift.py::test_generate_candidates_excludes_tolerance_block_boilerplate -x`
2. **Added detection mechanism**
   - `uv run pytest tests/test_added_detection_phase6.py -x`
3. **Added truth selection**
   - `uv run pytest tests/test_added_truth_selection.py tests/test_debug_row_identity.py -x`
4. **Asset-backed regression harness**
   - `uv run pytest tests/test_phase6_asset_regression.py -x`
5. **Phase gate**
   - `uv run pytest -x`

The phase-local harness should stay read-only with respect to `assets/` and `ground_truth.json`. The full 9-part pipeline rerun remains Phase 7 territory.

## Planning Guidance

The smallest defensible Phase 6 plan set is four plans:

1. shared exclusion contract
2. grouped added evidence + explained-by-match suppression
3. deterministic duplicate added-truth claiming
4. asset-backed regression harness

Trying to combine all four into one plan would blur responsibilities and make verification too coarse.

## Risks To Address In Plans

- Over-excluding legitimate annotations near the page edge
- Suppressing true added rows because the explained-by-match rule is too loose
- Assigning duplicate added truth rows incorrectly instead of preserving ambiguity
- Letting grouped added text over-expand into neighboring annotations or note blocks

## Non-Recommendations

- Do not add per-part lookup tables keyed by ground-truth index.
- Do not let the evaluator mutate or "reserve" truth rows by anything other than packet evidence.
- Do not solve Part 9 duplicate added rows by editing `ground_truth.json`.
- Do not patch `shop/services/review.py` to hide missing-added truth rows; that would only mask the packet defect.
