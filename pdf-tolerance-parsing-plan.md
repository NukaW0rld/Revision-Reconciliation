# PDF Drawing Tolerance Parsing Plan

This plan adds tolerance extraction for characteristics detected from PDF text spans (including stacked formats) and threads the parsed tolerance metadata through the matching pipeline, without changing the match/classify decision logic until tolerance extraction is reliable.

## Why this is needed (current state)
- Rev A "authoritative requirement" comes from Form 3 (`Anchor.requirement_raw`).
- Rev B candidates come from PDF `TextSpan.text` and `TextSpan.bbox_pdf`.
- Current PDF parsing treats tolerances as just "more numbers", and matching/classification uses a **primary numeric heuristic** (largest number) that often ignores tolerance-only changes.
- Form 3 parsing already produces `upper_tolerance`/`lower_tolerance` (recently added in `io/xlsx.py`).

## Goals / non-goals
- **Goal**: Extract tolerance information from the PDF-detected characteristic annotation near a balloon / matched span.
- **Goal**: Support the 5 cases:
  1. No tolerance
  2. Bilateral tolerance as `±value` (or `+/-`)
  3. Bilateral tolerance presented as stacked **max over min** limit values (e.g., ".160 .140" means nominal 0.150 ±0.01)
  4. Unilateral tolerance presented as stacked **upper over lower** (signed) (e.g., "Ø12 +0.3 +0.1" means +0.3/+0.1 tolerance)
  5. **NEW**: Bilateral tolerance with stacked opposite-sign values (e.g., "8.0 +0.3 -0.1" with +0.3 over -0.1)
- **Non-goal (for this iteration)**: Changing `generate_candidates()` / `classify_delta()` to incorporate tolerances. That comes after extraction is stable.

## (COMPLETED) Milestone 1 — Add a PDF tolerance extraction module (text + geometry)
**Deliverable**: A deterministic function that accepts a set of `TextSpan` objects (an “annotation group”) and returns tolerance info.

- **New types** (updated based on clarifications):
  - `PdfTolerance` dataclass with:
    - `kind`: `none | plus_minus | bilateral_stacked | unilateral_stacked | limits_stacked`
    - `upper_limit`, `lower_limit`: Optional[float] (absolute max/min values for ALL cases)
    - `nominal_value`: Optional[float] (the base dimension value)
    - `confidence`: float
    - `source_spans`: list of span keys / bboxes for debugging

- **Kind meanings**:
  - `none`: No tolerance detected
  - `plus_minus`: Inline `±0.1` or `+/-0.1` format
  - `bilateral_stacked`: Stacked opposite-sign values like `+0.3` over `-0.1` next to dimension
  - `unilateral_stacked`: Stacked same-sign values like `+0.3` over `+0.1` after dimension
  - `limits_stacked`: Stacked unsigned values like `.160` over `.140` (absolute limits)

- **Core parsing primitives**:
  - `parse_inline_plus_minus(text: str)` → bilateral tolerance from `±` or `+/-`
  - `parse_inline_unilateral(text: str)` → unilateral from `+0.3/-0.1` or similar
  - `parse_stacked_tolerance_values(spans: list[TextSpan])` → detect stacked numeric spans using geometry
  - `interpret_stacked_values(upper_val, lower_val, has_signs, nominal)` → convert to absolute limits

- **Absolute limits approach**:
  - For `±0.1` with nominal `8.0` → `upper_limit=8.1, lower_limit=7.9`
  - For stacked `+0.3/+0.1` with nominal `12` → `upper_limit=12.3, lower_limit=12.1`
  - For stacked `.160/.140` → `upper_limit=0.160, lower_limit=0.140, nominal=0.150`

## Milestone 2 — Build “annotation groups” from PDF spans
**Deliverable**: A function that, given a “seed” span (or bbox), produces the set of spans representing a single characteristic annotation.

Rationale: tolerance information is frequently split across multiple spans (or vertically stacked), and today most of the pipeline assumes a single `TextSpan`.

- **Grouping strategy** (incremental, conservative):
  - Start from an initial span selected by existing logic (Rev A: `best_span`; Rev B: a candidate span).
  - Merge **horizontal neighbors** on the same baseline (reuse the ideas in `vision/bbox_utils.expand_bbox_with_adjacent_spans`).
  - Additionally merge **vertical stack neighbors** when:
    - x-overlap is high
    - y distance between centers is small
    - font sizes are similar
    - both spans look numeric-ish (contain digits and optional sign)
  - Output:
    - `group_spans: list[TextSpan]`
    - `group_bbox: union bbox`
    - `group_text`: normalized concatenation for debugging (but geometry remains authoritative for stacked parsing)

## Milestone 3 — Wire tolerance extraction into the pipeline (data plumbing only)
**Deliverable**: Tolerance metadata is computed and carried along for later comparison, without changing match scoring yet.

- **Rev A**:
  - When building anchors in `build_revA_anchors()`, for `best_span` (requirement annotation), build an annotation group and compute:
    - `revA_pdf_tolerance` (from the PDF text itself)
  - Keep Form 3 tolerance (authoritative) separate from PDF tolerance (observed).

- **Rev B**:
  - For **ALL candidates** during `generate_candidates()`, build an annotation group around each `candidate.span` and compute:
    - `revB_pdf_tolerance`
  - This means tolerance parsing happens for every candidate during ranking, not just the final match.

- **Storage location**:
  - Add optional tolerance fields to:
    - `Anchor` (Rev A observed tolerance)
    - `Candidate` (Rev B observed tolerance)
  - If ballooned feature on Rev A PDF has no tolerance in PDF but Form 3 has tolerance, ignore PDF tolerance for matching (set to `kind=none`).

## Milestone 4 — Debug artifacts + acceptance checks
**Deliverable**: Easily inspect tolerance parsing output per characteristic, and ensure behavior is stable before changing comparison logic.

- **Debug output** (into the existing `out/<run_id>/debug/` dir):
  - For each anchor and its best Rev B match, write a JSON record:
    - char_no
    - Rev A: requirement text + revA group spans + parsed tolerance
    - Rev B: matched span text + revB group spans + parsed tolerance
    - confidence + reasons

- **Fixture-driven checks**:
  - Add a small set of “golden” examples (strings + synthetic bboxes) to validate:
    - `±` parsing
    - signed unilateral parsing
    - stacked unsigned limit parsing
    - stacked signed unilateral parsing

## Milestone 5 — (Next step) incorporate tolerance into comparison
**Not implemented in this phase**, but once milestones 1–4 are stable, update matching/classification:
- Treat tolerance-only change as `changed` (or at least reduce confidence of `unchanged`) when primary dimension matches but tolerance differs.
- Prefer comparing:
  - `upper_tol/lower_tol` when available
  - otherwise compare `upper_limit/lower_limit`

## Answers received from user
- **Q1**: Use absolute limits for ALL cases (not just limits-style). Convert everything to `upper_limit`/`lower_limit`.
- **Q2**: For unilateral stacked tolerances, `upper_tolerance` and `lower_tolerance` are always same-sign (e.g., `+0.3/+0.1` or `-0.1/-0.3`). They can never be opposite signs (that would be bilateral).
- **Q3**: Parse tolerances for ALL candidates during ranking.

## Additional clarifications received
- **Case 3 clarification**: Stacked `.160/.140` means nominal=0.150 with ±0.01 tolerance (the stacked values are the final limits after tolerance applied).
- **Case 4 clarification**: Stacked `+0.3/+0.1` after `Ø12` means the characteristic can be as large as Ø12.3 or as small as Ø12.1 (tolerance limits).
- **New Case 5**: Bilateral stacked opposite-sign like `8.0 +0.3 -0.1` with `+0.3` over `-0.1` (uneven bilateral tolerance).
- **Rev A PDF tolerance**: If ballooned feature has no tolerance in PDF but Form 3 has tolerance, ignore PDF tolerance for matching.
