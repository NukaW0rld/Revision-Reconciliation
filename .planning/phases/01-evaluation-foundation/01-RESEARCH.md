# Phase 1 Research: Evaluation Foundation

**Phase:** 1
**Date:** 2026-04-10
**Status:** Ready for planning

## Research Goal

Answer: what needs to be true for Phase 1 planning to stay deterministic, avoid part-specific hacks, and fit the existing pipeline and web workflow without inventing a parallel debug stack.

## Key Findings

### 1. The fixture key can stay exact and file-system-backed

- `run.py` already treats `part_name` as the exact asset directory under `assets/`.
- `shop/tasks.py` already passes the submitted `part_number` through to `run_pipeline(..., part_name=part_name)`.
- The safest Phase 1 contract is to treat that existing string as the exact truth fixture key for benchmarked runs. No alias table, no fuzzy lookup, no fallback search.
- The loader should resolve only `assets/{truth_fixture_key}/ground_truth.json` and fail immediately on missing directories or missing files.

### 2. The existing `ground_truth.json` fixtures already imply a stable schema

Observed top-level fields across current fixtures:

- `part_name`
- `general_notes`
- `characteristics`

Observed characteristic fields:

- `classification`
- `requirement_revB`
- `snippet_center_revA`
- `snippet_center_revB`
- `char_no` for non-`added` entries

Status-aware validation should enforce:

- `classification` in `{"unchanged", "changed", "removed", "added"}`
- non-`added` entries require `char_no`
- `removed` entries may set `requirement_revB` to `null`
- `added` entries may omit `char_no`
- `snippet_center_revA` may be `null` for `added`
- `snippet_center_revB` may be `null` for `removed`

That matches the locked Phase 1 decisions and avoids one global required-field rule that would reject valid truth rows.

### 3. Evaluation should extend `DeltaPacket`, not create a sidecar artifact

- `shop/services/review.py` and export paths already load `delta_packet.json` as the source of truth for review seeding and debug payload assembly.
- `delta_preservation/types.py` already defines the stable packet contract shared by the pipeline and the web tier.
- The lowest-risk integration is to add an additive `evaluation` envelope per `DeltaItem` and persist the exact truth fixture key in `DeltaPacket.inputs`.
- That keeps the pipeline as the only producer and preserves the current packet-driven review model.

### 4. Requirement conformance should reuse the semantic pipeline before falling back to text

- `delta_preservation/reconcile/semantic_compare.py` already defines a bounded semantic equivalence contract for GD&T, weld, surface finish, and fit callouts.
- `delta_preservation/reconcile/normalize.py` already exposes `parse_requirement()` for deterministic normalized-text fallback.
- Phase 1 should compare requirements in this order:
  1. semantic equality when both sides are parsed and comparable
  2. normalized-text equality when semantic comparison is unavailable
  3. explicit mismatch when neither path proves equivalence
- Missing semantic parsing alone must not fail the row. Missing requirement evidence still must block auto-pass when truth expects a non-null Rev B requirement.

### 5. Added-characteristic truth matching must be unordered

- Existing fixtures already contain canonical `added` rows without `char_no`.
- That means evaluation cannot depend on packet `char_no` ordering for added items.
- The evaluator needs a separate matching path that consumes an unordered pool of truth-side `added` expectations and pairs them using normalized requirement plus snippet evidence.
- Ambiguous matching should not hard-fail the run. It should emit `review_needed` evaluation output for the affected rows.

### 6. Snippet tolerance needs deterministic geometry, not center equality

The packet already stores reviewer-facing evidence bboxes in PDF coordinates, so Phase 1 does not need to invent a second image-analysis path.

Recommended rules:

- Single-callout rule: truth center must fall inside the evidence bbox after contracting each edge by `max(12.0, 0.10 * min(width, height))` PDF points.
- Edge guard: even when inside the bbox, reject if the truth center is within `6.0` PDF points of any surviving edge.
- Grouped-callout or notes rule: use the union bbox that already absorbs companion spans and contract each edge by `6.0` PDF points only.
- Null-truth rule: if the truth center is `null` for a side, that side is not required for conformance.
- Missing-evidence rule: if truth expects a center and the packet has no evidence on that side, emit a side-specific mismatch instead of a hard failure.

This stays deterministic and aligns with the project constraint that visually acceptable context matters more than exact center equality.

### 7. Downstream consumption should stay additive in Phase 1

- Phase 2 owns the focused queue and `debug_report.json` behavior changes.
- Phase 1 only needs to make mismatch details available, not redesign the review surface.
- The right additive move is:
  - store machine-readable mismatch entries on each packet row
  - keep existing human-readable `reasons`
  - expose the raw evaluation envelope through review/debug helpers so later phases can consume it without recomputing

## Recommended Implementation Shape

| Concern | Module | Planned role |
|---------|--------|--------------|
| Truth schema | `delta_preservation/evaluation/contracts.py` | Pydantic models and contract errors for fixture validation |
| Truth loading | `delta_preservation/evaluation/loader.py` | Exact fixture resolution and `ground_truth.json` loading |
| Classification + requirement evaluation | `delta_preservation/evaluation/conformance.py` | Match packet rows to truth rows, compare classification and requirements |
| Snippet tolerance | `delta_preservation/evaluation/snippet_rules.py` | Deterministic bbox-vs-center acceptance rules |
| Packet contract | `delta_preservation/types.py` | Additive `evaluation` and `mismatches` models |
| Pipeline wiring | `delta_preservation/cli.py` | Load truth, evaluate rows, serialize evaluation into packet |
| Run failure surface | `shop/tasks.py` | Fail clearly on missing or malformed truth fixtures |
| Downstream access | `shop/services/review.py` | Surface packet evaluation details without recomputation |

## Sequencing Recommendation

### Plan 01-01

- Define status-aware ground-truth contracts
- Add strict loader keyed by exact fixture directory name
- Fail runs clearly when truth data is missing or malformed

### Plan 01-02

- Add packet-side evaluation models
- Implement classification and requirement conformance
- Match canonical `added` truth rows as an unordered pool

### Plan 01-03

- Implement deterministic snippet tolerance
- Finalize per-row `conforming` vs `review_needed`
- Persist ordered machine-readable mismatch entries and expose them to review helpers

## Validation Architecture

### Test infrastructure

- Framework: `pytest` via `uv run pytest`
- Existing config: `pyproject.toml`
- Existing test anchors:
  - `tests/test_pipeline_task.py`
  - `tests/test_output_formatting.py`
  - `tests/test_debug_internals.py`

### Fast feedback loop

- Contract/loader work: `uv run pytest tests/test_ground_truth_loader.py -q`
- Evaluator work: `uv run pytest tests/test_ground_truth_evaluation.py -q`
- Snippet/review integration: `uv run pytest tests/test_snippet_evaluation.py tests/test_debug_internals.py -q`
- Per-wave safety check: `uv run pytest -q`

### Nyquist implications

- Every Phase 1 task should carry an automated `uv run pytest ...` command.
- No Wave 0 plan is needed if each plan creates or extends its own targeted test file before verifying.
- Full-suite feedback should stay under the existing project threshold for local iteration; targeted commands should stay well under 30 seconds.

## Open Questions (RESOLVED)

1. **Where should the truth key live?**  
   RESOLVED: use the exact existing `part_name` / `Run.part_number` value as the canonical fixture key for benchmark runs in Phase 1. Do not add fuzzy lookup or aliasing.

2. **Where should evaluation output be stored?**  
   RESOLVED: embed it into each `DeltaItem` as an additive `evaluation` block and preserve the fixture key in `DeltaPacket.inputs`.

3. **How should canonical `added` rows be matched?**  
   RESOLVED: treat truth-side `added` expectations as an unordered pool matched by normalized requirement text plus snippet evidence, never by packet ordering.

4. **How should snippet tolerance stay deterministic?**  
   RESOLVED: use bbox containment with explicit edge-guard math rather than exact center equality or subjective crop review.

---

Phase 1 research complete. Planning can proceed without another discovery pass.
