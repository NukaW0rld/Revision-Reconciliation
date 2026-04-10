# Phase 02: Focused Debug Workflow - Research

**Researched:** 2026-04-10  
**Domain:** Exception-only admin debug review and deterministic debug report generation on top of Phase 1 evaluation outputs  
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
### Exception Queue Scope
- **D-01:** The main admin debug queue defaults to exception-only behavior and includes only rows whose evaluation status is `review_needed`.
- **D-02:** Auto-passed (`conforming`) rows do not appear in the main manual-review queue.
- **D-03:** Auto-passed rows remain inspectable from the run-details surface and from the exported `debug_report.json`, not from the main exception queue.
- **D-04:** If a run has zero exception rows, the debug path should stay on run details and show a clean all-conforming summary instead of routing the reviewer into an empty queue.

### Exception Card Presentation
- **D-05:** Exception cards are mismatch-first: ordered mismatch codes/messages and the review action come before deeper evidence browsing.
- **D-06:** The default expanded content on an exception card is the mismatch summary plus the review form; scores, semantic internals, bbox centers, and longer debug detail should be collapsed behind secondary disclosures.
- **D-07:** The review interaction remains one focused exception card at a time rather than changing to a dense list or grouped family browser in Phase 2.

### Reviewer Outcome Model
- **D-08:** Reviewer resolution vocabulary for exception rows should move from `correct / incorrect / partially_correct` to explicit exception outcomes centered on `algorithm_error` and `acceptable_alternate`.
- **D-09:** Resolving an `algorithm_error` requires a corrected classification plus reviewer rationale; corrected requirement text stays optional.
- **D-10:** Resolving an `acceptable_alternate` requires reviewer rationale only in Phase 2; normalized structured alternate fields are deferred until the history layer in Phase 3.

### Debug Report Contract
- **D-11:** `debug_report.json` becomes exportable when every exception row has been resolved; auto-passed rows must never block report readiness.
- **D-12:** `debug_report.json` must explicitly distinguish these row-level states: `canonical_match`, `acceptable_alternate`, `unresolved_review_needed`, and `algorithm_error`.
- **D-13:** Auto-passed rows remain full per-row entries in `debug_report.json`; they are not reduced to counts-only or summary-only data.

### Claude's Discretion
- Exact wording and field names for the new exception outcome vocabulary, as long as the semantics stay aligned with `algorithm_error` and `acceptable_alternate`
- Exact visual treatment for collapsed versus expanded diagnostic panels on exception cards
- Exact run-details summary layout for all-conforming runs and auto-pass counts
- Exact JSON nesting/field layout for the expanded `debug_report.json` contract, as long as the four required row states remain explicit and downstream-friendly

### Deferred Ideas (OUT OF SCOPE)
- Normalized structured storage for accepted alternate outcomes — Phase 3
- Queryable contradiction and cross-part overfitting analysis — later phase after the history layer exists
- Rich grouping or mismatch-family browsing beyond the one-card-at-a-time exception flow — future UX enhancement if needed
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DREV-01 | Admin debug reviewer can open a queue focused on only nonconforming or ambiguous characteristics for a run. `[CITED: .planning/REQUIREMENTS.md]` | Drive queue membership from `delta_packet.json -> item.evaluation.status == "review_needed"` instead of from `ReviewItem.reviewer_decision`. `[VERIFIED: repo grep]` |
| DREV-02 | Admin debug reviewer can still inspect auto-passed characteristics in the run details or exported debug report. `[CITED: .planning/REQUIREMENTS.md]` | Add run-details evaluation summary and export all packet rows even when no debug verdict exists for conforming rows. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md][VERIFIED: repo grep]` |
| DREV-03 | Admin debug reviewer can record whether a nonconforming characteristic is an algorithm error or an acceptable alternate outcome. `[CITED: .planning/REQUIREMENTS.md]` | Replace the old `correct/incorrect/partially_correct` validator and form options with exception-focused outcomes. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md][VERIFIED: repo grep]` |
| DREV-04 | Admin debug reviewer can attach rationale for any nonconforming characteristic that is not simply marked correct. `[CITED: .planning/REQUIREMENTS.md]` | Require rationale for `algorithm_error` and `acceptable_alternate`; require corrected classification for `algorithm_error`, but keep corrected requirement text optional. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]` |
| RPT-01 | System generates a `debug_report.json` for each evaluated run without requiring manual verdict entry for auto-passed characteristics. `[CITED: .planning/REQUIREMENTS.md]` | Build report rows from packet evaluation first; only unresolved exception rows should block export readiness. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md][VERIFIED: repo grep]` |
| RPT-02 | `debug_report.json` distinguishes canonical ground-truth matches, acceptable alternate matches, and unresolved review-needed rows. `[CITED: .planning/REQUIREMENTS.md]` | Add explicit row-state derivation logic with `canonical_match`, `acceptable_alternate`, `unresolved_review_needed`, and `algorithm_error`. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]` |
| RPT-03 | `debug_report.json` includes mismatch reasons and any linked exceptions/history references for rows that need review. `[CITED: .planning/REQUIREMENTS.md]` | Preserve Phase 1 ordered `evaluation.mismatches` and keep any future history reference field nullable/empty in Phase 2 so Phase 3 persistence does not leak early. `[VERIFIED: repo grep][CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]` |
</phase_requirements>

## Summary

Phase 1 already gives Phase 2 the core signal it needs: every packet row can carry `evaluation.status` (`conforming` or `review_needed`) and ordered machine-readable mismatches inside `delta_packet.json`. `[VERIFIED: repo grep]` The current web layer does not use that signal to decide queue membership, run-details summaries, or export readiness yet; it still assumes every packet row becomes a manual `ReviewItem` and every `ReviewItem` needs a debug verdict before `debug_report.json` can be exported. `[VERIFIED: repo grep]`

The safest Phase 2 plan is therefore a brownfield refactor, not a new subsystem. `[VERIFIED: repo grep]` Keep FastAPI, SQLAlchemy, Jinja, and the Phase 1 packet contract. `[VERIFIED: repo grep]` Move debug queue composition, progress counts, and export readiness to packet-evaluation semantics; keep normal review/sign-off fields intact; and do not add Phase 3 history persistence yet. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md][VERIFIED: repo grep]`

The highest-risk implementation trap is row identity. `assemble_debug_report_payload()` already avoids collapsing duplicate or `None` `char_no` rows by using queue order, but `semantic_contracts_by_char()` and `debug_internals_by_char()` still key debug data by `char_no`, which can silently collide for duplicate characteristics or added rows. `[VERIFIED: repo grep]` Phase 2 should standardize on a stable per-row key such as packet index or `ReviewItem.id` whenever packet rows are joined to rendered debug data. `[VERIFIED: repo grep]`

**Primary recommendation:** Treat `DeltaItem.evaluation` as the source of truth for Phase 2 debug state, and make `debug_report.json` exportable from packet rows plus exception-only resolutions rather than from “all `ReviewItem`s have verdicts.” `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md][VERIFIED: repo grep]`

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | Project pin `0.135.2`; PyPI latest `0.135.3` published `2026-04-01T16:23:59Z`. `[VERIFIED: repo grep + PyPI]` | Review routes, run-details route, debug export route. `[VERIFIED: repo grep]` | The existing debug and run-status surfaces already live in FastAPI routers; Phase 2 is a route/service/template refactor, not a framework change. `[VERIFIED: repo grep]` |
| SQLAlchemy | Project pin `2.0.48`; PyPI latest `2.0.49` published `2026-04-03T17:12:34Z`. `[VERIFIED: repo grep + PyPI]` | `Run` and `ReviewItem` persistence, route queries, sign-off state. `[VERIFIED: repo grep]` | Existing review/sign-off state is already persisted through ORM models; Phase 2 should reuse those models and only change queue semantics. `[VERIFIED: repo grep]` |
| Pydantic | Project pin `2.12.5`; PyPI latest `2.12.5`. `[VERIFIED: repo grep + PyPI]` | `DeltaItem`, `ItemEvaluation`, `EvaluationMismatch`, and packet validation. `[VERIFIED: repo grep]` | Phase 1 already serializes the evaluation contract through Pydantic models; Phase 2 should reuse that contract instead of introducing ad hoc JSON parsing. `[VERIFIED: repo grep]` |
| Jinja2 | Project pin `3.1.6`; PyPI latest `3.1.6`. `[VERIFIED: repo grep + PyPI]` | Server-rendered review cards, footer, and run-details summary. `[VERIFIED: repo grep]` | The admin debug UI is already rendered through Jinja partials with HTMX updates; Phase 2 should stay inside that pattern. `[VERIFIED: repo grep]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | Project pin `9.0.2`; PyPI latest `9.0.3` published `2026-04-07T17:16:18Z`. `[VERIFIED: repo grep + PyPI]` | Regression coverage for queue logic, verdict validation, and export payloads. `[VERIFIED: repo grep]` | Use for every Phase 2 behavioral change; existing debug tests already cover export ordering and validator behavior. `[VERIFIED: repo grep]` |
| httpx | Project pin `0.28.1`; PyPI latest `0.28.1`. `[VERIFIED: repo grep + PyPI]` | FastAPI `TestClient` and route-level integration tests. `[VERIFIED: repo grep]` | Use for queue and run-details HTTP tests. `[VERIFIED: repo grep]` |
| Tailwind CSS | Project dependency `^4.0.0`; npm latest `4.2.2`, modified `2026-04-07`. `[VERIFIED: repo grep + npm registry]` | Existing CSS utility layer for template changes. `[VERIFIED: repo grep]` | Use only for incremental layout/state styling in review and run-details templates. `[VERIFIED: repo grep]` |
| DaisyUI | Project dependency `^5.0.0`; npm latest `5.5.19`, modified `2026-02-20`. `[VERIFIED: repo grep + npm registry]` | Existing badge/button/panel styling in the debug UI. `[VERIFIED: repo grep]` | Reuse existing component classes for the exception-only UX rather than adding a new UI library. `[VERIFIED: repo grep]` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Reusing FastAPI + Jinja partials | A new SPA/debug frontend | A new frontend would duplicate state and routing for a workflow that already exists server-side. `[VERIFIED: repo grep]` |
| Reading packet evaluation directly | Recomputing conformance inside the web tier | Recomputing risks divergence from the Phase 1 packet contract and duplicates logic already written in the pipeline. `[VERIFIED: repo grep]` |
| Deferring alternate history storage to Phase 3 | Adding a new exception-history table in Phase 2 | Phase 2 context explicitly defers durable alternate/history storage; adding it now leaks Phase 3 scope into this phase. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]` |

**Installation:** Existing project dependencies are sufficient for Phase 2; no new framework/library is required. `[VERIFIED: repo grep]`

```bash
uv sync --locked
npm install
```

## Architecture Patterns

### Recommended Change Surface
```text
delta_preservation/
└── cli.py                          # keep writing evaluation into delta_packet.json
shop/
├── services/
│   └── review.py                   # queue composition, verdict validation, export assembly
├── routers/
│   ├── review.py                   # exception-only queue route and debug export gate
│   └── runs.py                     # run-details auto-pass summary / all-conforming branch
└── templates/
    ├── review/
    │   ├── queue.html
    │   ├── _item_card_debug.html
    │   ├── _progress_bar.html
    │   └── _signoff_footer.html
    └── runs/status.html
tests/
├── test_debug_verdicts.py
├── test_debug_internals.py
├── test_review.py
└── new Phase 2 queue/status tests
```

### Pattern 1: Packet Evaluation Drives Debug Queue
**What:** Treat `DeltaItem.evaluation.status` as the authoritative debug queue membership signal. `[VERIFIED: repo grep]`  
**When to use:** Always for Phase 2 exception-only behavior. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]`  
**Why:** Phase 1 already serializes `conforming` vs `review_needed` per row, while the current review service still seeds and counts all packet rows equally. `[VERIFIED: repo grep]`  
**Example:**

```python
# Source: delta_preservation/types.py + shop/services/review.py
packet_rows = [
    (packet_index, DeltaItem.model_validate(raw_item))
    for packet_index, raw_item in enumerate(sorted(raw_items, key=sort_key))
]
exception_rows = [
    (packet_index, item)
    for packet_index, item in packet_rows
    if item.evaluation is not None and item.evaluation.status == "review_needed"
]
```

### Pattern 2: Stable Row Identity Beats `char_no`
**What:** Join packet rows, debug internals, and verdict payloads with a stable per-row key such as packet index or `ReviewItem.id`, not `char_no`. `[VERIFIED: repo grep]`  
**When to use:** Any time the UI or export needs to address a specific packet row. `[VERIFIED: repo grep]`  
**Why:** `assemble_debug_report_payload()` already preserves queue order to avoid duplicate/`None` `char_no` collisions, but the debug helper maps still collapse by `char_no`. `[VERIFIED: repo grep]`  
**Example:**

```python
# Source: shop/services/review.py (ordering requirement) + tests/test_debug_verdicts.py
rows = []
for queue_index, (review_item, raw_item) in enumerate(zip(review_items, ordered_packet_items), start=1):
    rows.append(
        {
            "queue_index": queue_index,
            "review_item_id": review_item.id,
            "packet_index": queue_index - 1,
            "packet_item": raw_item,
        }
    )
```

### Pattern 3: Export Readiness Is Exception-Scoped
**What:** `debug_report.json` should be ready when every exception row is resolved, not when every packet row has a saved debug verdict. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]`  
**When to use:** In footer readiness counts, download gate logic, and run-details summary CTAs. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]`  
**Why:** Current export logic raises if any `ReviewItem` lacks a verdict, which directly contradicts Phase 2 requirement `RPT-01`. `[VERIFIED: repo grep]`  
**Example:**

```python
# Source: 02-CONTEXT decisions D-11/D-12
if evaluation.status == "conforming":
    row_state = "canonical_match"
elif verdict == "acceptable_alternate":
    row_state = "acceptable_alternate"
elif verdict == "algorithm_error":
    row_state = "algorithm_error"
else:
    row_state = "unresolved_review_needed"
```

### Pattern 4: Run Details Must Be Able To Render Without Opening The Queue
**What:** Add a packet-summary path on `/runs/{id}` so all-conforming runs can show debug counts and export readiness without sending the reviewer into `/review/{id}?debug=1`. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md][VERIFIED: repo grep]`  
**When to use:** For completed runs and especially runs with zero exception rows. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]`  
**Why:** `run_status()` currently renders only `Run` metadata, and `open_review_queue()` mutates `run.status` to `reviewing` on first open. `[VERIFIED: repo grep]`

### Anti-Patterns to Avoid
- **Driving the debug queue from `reviewer_decision` fields:** those fields belong to the normal review/sign-off flow and do not encode Phase 1 conformance results. `[VERIFIED: repo grep]`
- **Keying debug internals or semantic contracts by `char_no`:** duplicate `char_no` values and `None` rows can display the wrong card details. `[VERIFIED: repo grep]`
- **Making all-conforming runs open an empty review page:** Phase 2 explicitly requires the debug path to remain on run details for zero-exception runs. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]`
- **Adding durable acceptable-alternate history in Phase 2:** Phase 2 only records rationale in the debug artifact layer; Phase 3 owns durable history. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]`

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Exception detection | A second conformance evaluator in `shop/` | The existing `DeltaItem.evaluation` envelope from `delta_packet.json` | Phase 1 already serializes ordered mismatches and final conformance status. `[VERIFIED: repo grep]` |
| Debug row identity | `char_no`-only lookup maps | Stable packet index / `ReviewItem.id` joins | Duplicate and `None` `char_no` rows are already a known case in tests and export assembly. `[VERIFIED: repo grep]` |
| Export readiness | `len(saved_verdicts) == len(review_items)` | Exception-only readiness counts | Auto-passed rows must never block export in Phase 2. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]` |
| Acceptable-alternate persistence | A new history table or fake history IDs | Report-local resolution payload with nullable future `history_ref` | Phase 2 must not leak Phase 3 storage concerns into current behavior. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]` |

**Key insight:** The safest Phase 2 implementation is to reuse the Phase 1 packet as the canonical debug data source and only add the minimum exception-resolution layer needed to explain review-needed rows. `[VERIFIED: repo grep][CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]`

## Common Pitfalls

### Pitfall 1: Exception-Only UI With All-Row Backend Semantics
**What goes wrong:** The screen shows only exception rows, but progress counts and export readiness still measure every `ReviewItem`. `[VERIFIED: repo grep]`  
**Why it happens:** `debug_verdict_state()` is currently passed the full `open_review_queue()` result, and `_signoff_footer.html` derives readiness from `debug_total`/`debug_submitted` only. `[VERIFIED: repo grep]`  
**How to avoid:** Compute debug totals from the exception subset and derive report rows from packet evaluation plus optional exception resolutions. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md][VERIFIED: repo grep]`  
**Warning signs:** A run with many `conforming` rows still says most debug verdicts are missing. `[VERIFIED: repo grep]`

### Pitfall 2: Duplicate-Row Diagnostics Point At The Wrong Characteristic
**What goes wrong:** The wrong mismatch summary or semantic contract shows on a card when two rows share the same `char_no` or have `char_no=None`. `[VERIFIED: repo grep]`  
**Why it happens:** Current debug helper maps are keyed by `char_no`, not by a stable row identifier. `[VERIFIED: repo grep]`  
**How to avoid:** Build all debug-row helper maps from packet index or `ReviewItem.id`. `[VERIFIED: repo grep]`  
**Warning signs:** Duplicate-characteristic tests pass in export code but card-level debug data looks inconsistent. `[VERIFIED: repo grep]`

### Pitfall 3: All-Conforming Runs Still Need Queue Seeding
**What goes wrong:** An admin cannot see the clean all-conforming summary or export until the queue has been opened and seeded first. `[VERIFIED: repo grep]`  
**Why it happens:** `run_status()` does not load packet evaluation data, and `assemble_debug_report_payload()` currently depends on `ReviewItem` rows being present for every packet row. `[VERIFIED: repo grep]`  
**How to avoid:** Add a run-details summary path that reads packet evaluation directly and make export assembly independent from “every packet row has a persisted `ReviewItem`.” `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md][VERIFIED: repo grep]`  
**Warning signs:** Clicking debug on an all-conforming run changes status to `reviewing` or lands on “No items match” instead of a summary. `[VERIFIED: repo grep]`

### Pitfall 4: New Verdict Vocabulary Still Behaves Like Old Manual QA
**What goes wrong:** The form labels change, but validation rules and stored payload shape still assume `correct/incorrect/partially_correct`. `[VERIFIED: repo grep]`  
**Why it happens:** `VALID_DEBUG_VERDICTS` and `validate_debug_verdict_payload()` are still built around the old vocabulary. `[VERIFIED: repo grep]`  
**How to avoid:** Redefine the validator around Phase 2 semantics and make “corrected classification” mean “reviewer-accepted classification” so `algorithm_error` can cover false-positive queueing as well as true misclassification. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md][VERIFIED: repo grep]`  
**Warning signs:** Reviewers cannot resolve evaluator false positives without pretending a row was `incorrect`. `[VERIFIED: repo grep]`

### Pitfall 5: Phase 3 History Leakage
**What goes wrong:** Phase 2 writes fake exception-history identifiers or normalized alternate structures that later need migration. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]`  
**Why it happens:** `RPT-03` mentions future history references, and it is easy to prematurely introduce storage scaffolding. `[CITED: .planning/REQUIREMENTS.md]`  
**How to avoid:** Reserve a nullable field such as `history_ref: null` if forward-compatibility matters, but do not create durable history persistence in this phase. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]`  
**Warning signs:** Phase 2 plans start adding new tables or mutable truth aliases for accepted alternates. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]`

## Code Examples

Verified patterns from project sources:

### Exception Queue Selection
```python
# Source: delta_preservation/types.py, delta_preservation/cli.py, shop/services/review.py
packet_data = _load_delta_packet(run)
ordered_packet_items = sorted(
    packet_data.get("items", []),
    key=lambda x: (x.get("char_no") is None, x.get("char_no") or 0),
)

exception_rows = []
for packet_index, raw_item in enumerate(ordered_packet_items):
    delta_item = DeltaItem.model_validate(raw_item)
    if delta_item.evaluation and delta_item.evaluation.status == "review_needed":
        exception_rows.append((packet_index, delta_item))
```

### Report Row-State Derivation
```python
# Source: 02-CONTEXT.md decisions D-11/D-12
def derive_row_state(evaluation_status: str, resolution: str | None) -> str:
    if evaluation_status == "conforming":
        return "canonical_match"
    if resolution == "acceptable_alternate":
        return "acceptable_alternate"
    if resolution == "algorithm_error":
        return "algorithm_error"
    return "unresolved_review_needed"
```

### Exception-Scoped Export Readiness
```python
# Source: 02-CONTEXT.md decisions D-01/D-11
debug_total = len(exception_review_items)
debug_submitted = sum(
    1
    for item in exception_review_items
    if item.id in verdicts_by_item_id
)
debug_export_ready = debug_submitted >= debug_total
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Debug resolution vocabulary = `correct`, `incorrect`, `partially_correct`. `[VERIFIED: repo grep]` | Debug resolution vocabulary should be centered on `algorithm_error` and `acceptable_alternate`. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]` | Phase 2 | Validator, templates, tests, and report row-state derivation all change together. `[VERIFIED: repo grep][CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]` |
| Export readiness requires a saved verdict for every `ReviewItem`. `[VERIFIED: repo grep]` | Export readiness should require resolution only for exception rows; auto-passed rows are included without blocking export. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]` | Phase 2 | `assemble_debug_report_payload()`, footer counts, and run-details CTA logic must be refactored together. `[VERIFIED: repo grep]` |
| Admin debug UI still renders all queue rows and exposes heavy debug internals expanded by default. `[VERIFIED: repo grep]` | Admin debug UI should become exception-only and mismatch-first, with diagnostics collapsed behind secondary disclosure. `[CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]` | Phase 2 | Template structure and partial update targets need a focused-card redesign rather than a filter-only tweak. `[VERIFIED: repo grep]` |

**Deprecated/outdated:**
- `VALID_DEBUG_VERDICTS = {"correct", "incorrect", "partially_correct"}` is outdated for Phase 2. `[VERIFIED: repo grep][CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]`
- `debug_internals_by_char()` / `semantic_contracts_by_char()` row joins are too weak for an exception-only debug surface that must preserve duplicate-row identity. `[VERIFIED: repo grep]`

## Assumptions Log

All claims in this research were verified from the repo, the phase context, or package registries during this session. No `[ASSUMED]` claims remain.

## Open Questions (RESOLVED)

1. **Legacy verdict compatibility**
   - Resolution: Phase 2 will **not** auto-migrate legacy `debug_verdicts.json` payloads that still use `correct`, `incorrect`, or `partially_correct`.
   - Encoded behavior: best-effort render loaders may ignore unsupported legacy entries so stale runs remain viewable, but strict write/export paths must fail with a clear Phase 2 re-entry message instructing the reviewer to re-save the affected rows under `algorithm_error` / `acceptable_alternate`.
   - Planning impact: Plan 02-02 should add regression coverage for stale legacy payload handling and implement explicit stale-payload rejection rather than a lossy mapper.

2. **Meaning of `corrected_classification` for `algorithm_error`**
   - Resolution: `corrected_classification` is the reviewer-accepted classification and **may equal** the pipeline classification.
   - Encoded behavior: validation must require the field for `algorithm_error` but must not require it to differ from the pipeline label, because `algorithm_error` can represent evaluator false positives or wrong queue/report semantics even when the classification itself was already acceptable.
   - Planning impact: Plan 02-02 should add regression coverage proving matching pipeline/reviewer classifications are valid for `algorithm_error` and should document that semantics explicitly in the validator/report contract.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | App code, route tests, service tests | ✓ | `3.12.3` `[VERIFIED: local command]` | — |
| `uv` | Locked dependency install and test execution | ✓ | `0.11.1` `[VERIFIED: local command]` | — |
| Node.js | CSS rebuild after template changes | ✓ | `v25.8.2` `[VERIFIED: local command]` | — |
| npm | Tailwind/DaisyUI build command | ✓ | `11.12.0` `[VERIFIED: local command]` | — |
| `pytest` global CLI | Direct test invocation | ✗ | — `[VERIFIED: local command]` | Use `uv run pytest` (`pytest 9.0.2`). `[VERIFIED: local command]` |
| `sqlite3` CLI | Optional local DB inspection only | ✗ | — `[VERIFIED: local command]` | Use SQLAlchemy session queries/tests; Phase 2 does not require the CLI. `[VERIFIED: repo grep]` |

**Missing dependencies with no fallback:**
- None. `[VERIFIED: local command]`

**Missing dependencies with fallback:**
- Global `pytest` CLI is absent, but `uv run pytest` works and matches the project lockfile. `[VERIFIED: local command]`

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest 9.0.2` via `uv run pytest`; project config in `pyproject.toml`. `[VERIFIED: repo grep + local command]` |
| Config file | `pyproject.toml` `[VERIFIED: repo grep]` |
| Quick run command | `uv run pytest -q tests/test_debug_verdicts.py tests/test_debug_internals.py -x` `[VERIFIED: repo grep + local command]` |
| Full suite command | `uv run pytest -q` `[VERIFIED: repo grep + local command]` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DREV-01 | Only `review_needed` rows appear in the admin debug queue; zero-exception runs stay on run details. `[CITED: .planning/REQUIREMENTS.md][CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]` | integration | `uv run pytest -q tests/test_focused_debug_queue.py -x` | ❌ Wave 0 |
| DREV-02 | Auto-passed rows remain visible in run details and in exported report output. `[CITED: .planning/REQUIREMENTS.md]` | integration | `uv run pytest -q tests/test_run_status_debug_summary.py -x` | ❌ Wave 0 |
| DREV-03 | Debug verdict validation and persistence use the new exception vocabulary. `[CITED: .planning/REQUIREMENTS.md]` | unit + integration | `uv run pytest -q tests/test_debug_verdicts.py -x` | ✅ |
| DREV-04 | Rationale is required for non-canonical exception resolutions. `[CITED: .planning/REQUIREMENTS.md]` | unit + integration | `uv run pytest -q tests/test_debug_verdicts.py -x` | ✅ |
| RPT-01 | `debug_report.json` exports without conforming-row verdicts. `[CITED: .planning/REQUIREMENTS.md]` | unit + integration | `uv run pytest -q tests/test_debug_verdicts.py tests/test_debug_internals.py -x` | ✅ |
| RPT-02 | Export row states distinguish canonical matches, acceptable alternates, unresolved exceptions, and algorithm errors. `[CITED: .planning/REQUIREMENTS.md][CITED: .planning/phases/02-focused-debug-workflow/02-CONTEXT.md]` | unit + integration | `uv run pytest -q tests/test_debug_verdicts.py tests/test_debug_internals.py -x` | ✅ |
| RPT-03 | Export preserves ordered mismatch reasons and keeps history references empty/null in Phase 2. `[CITED: .planning/REQUIREMENTS.md]` | unit | `uv run pytest -q tests/test_debug_internals.py -x` | ✅ |

### Sampling Rate
- **Per task commit:** `uv run pytest -q tests/test_debug_verdicts.py tests/test_debug_internals.py -x` `[VERIFIED: repo grep + local command]`
- **Per wave merge:** `uv run pytest -q tests/test_debug_verdicts.py tests/test_debug_internals.py tests/test_review.py tests/test_output_formatting.py -x` `[VERIFIED: repo grep]`
- **Phase gate:** `uv run pytest -q` `[VERIFIED: repo grep + local command]`

### Wave 0 Gaps
- [ ] `tests/test_focused_debug_queue.py` — exception-only filtering, zero-exception redirect/summary, exception-scoped progress counts. `[VERIFIED: repo grep]`
- [ ] `tests/test_run_status_debug_summary.py` — run-details auto-pass counts and all-conforming summary/CTA behavior. `[VERIFIED: repo grep]`
- [ ] `tests/test_debug_row_identity.py` — duplicate/`None` `char_no` rows keep correct per-row diagnostics and resolutions. `[VERIFIED: repo grep]`

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | yes `[VERIFIED: repo grep]` | Existing session authentication in `shop/services/auth.py` and `get_current_user()`. `[VERIFIED: repo grep]` |
| V3 Session Management | yes `[VERIFIED: repo grep]` | `UserSession` persistence plus session-cookie enforcement. `[VERIFIED: repo grep]` |
| V4 Access Control | yes `[VERIFIED: repo grep]` | Debug queue and debug export stay admin-only in `shop/routers/review.py`. `[VERIFIED: repo grep]` |
| V5 Input Validation | yes `[VERIFIED: repo grep]` | Pydantic packet models plus `validate_debug_verdict_payload()`. `[VERIFIED: repo grep]` |
| V6 Cryptography | yes `[VERIFIED: repo grep]` | Existing `pwdlib[bcrypt]`, `bcrypt`, and token/signature helpers remain the auth boundary; Phase 2 should not replace them. `[VERIFIED: repo grep]` |

### Known Threat Patterns for This Phase

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Run/item ID tampering on admin debug routes | Elevation of Privilege / Tampering | Keep admin-role checks and the existing `item.run_id == run_id` guard on debug POST routes. `[VERIFIED: repo grep]` |
| Malformed or hand-edited `debug_verdicts.json` payloads | Tampering | Keep strict validation on write/export paths and best-effort ignoring on render paths. `[VERIFIED: repo grep]` |
| Reviewer rationale or corrected requirement text rendered unsafely | Tampering / XSS | Keep server-side templating with default escaping; never render debug rationale as trusted HTML. `[VERIFIED: repo grep]` |
| Packet/output path confusion on run-details summaries | Spoofing / Tampering | Continue loading packet/debug artifacts only from `run.output_dir` plus fixed filenames. `[VERIFIED: repo grep]` |

## Sources

### Primary (HIGH confidence)
- `.planning/phases/02-focused-debug-workflow/02-CONTEXT.md` — locked Phase 2 decisions and deferred scope
- `.planning/REQUIREMENTS.md` — DREV/RPT requirement wording
- `.planning/ROADMAP.md` — phase goal, success criteria, and plan breakdown
- `.planning/phases/01-evaluation-foundation/01-CONTEXT.md` — Phase 1 evaluator guarantees that Phase 2 must preserve
- `delta_preservation/types.py` — `ItemEvaluation`, `EvaluationMismatch`, and packet contract
- `delta_preservation/cli.py` — point where Phase 1 writes evaluation into `delta_packet.json`
- `shop/services/review.py` — current queue seeding, verdict validation, and debug export behavior
- `shop/routers/review.py` — current debug queue/export route wiring
- `shop/routers/runs.py` — current run-details route data surface
- `shop/templates/review/queue.html` — current queue rendering behavior
- `shop/templates/review/_item_card_debug.html` — current verdict form and debug panel defaults
- `shop/templates/review/_signoff_footer.html` — current export readiness gate
- `shop/templates/runs/status.html` — current run-details CTA surface
- `tests/test_debug_verdicts.py` — current verdict/export coverage
- `tests/test_debug_internals.py` — current evaluation/mismatch export coverage
- `tests/test_review.py` — current `open_review_queue()` idempotence and status mutation
- PyPI JSON API (`fastapi`, `sqlalchemy`, `jinja2`, `pydantic`, `alembic`, `huey`, `weasyprint`, `pytest`, `httpx`) — current package versions
- npm registry (`tailwindcss`, `daisyui`, `@tailwindcss/cli`) — current frontend package versions

### Secondary (MEDIUM confidence)
- `.planning/research/SUMMARY.md` — milestone sequencing and focused-review framing
- `.planning/research/ARCHITECTURE.md` — existing research recommendation to keep a focused review surface
- `.planning/research/FEATURES.md` — milestone feature table-stakes
- `.planning/research/PITFALLS.md` — prior pitfall framing, especially “manual review still too broad”

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all recommended libraries are already in the brownfield stack and verified in `uv.lock` / `package.json` plus registries.
- Architecture: HIGH — the existing code clearly shows where queue composition, export gating, run-details summaries, and row-identity joins must change.
- Pitfalls: HIGH — the major risks are visible directly in the current service/template code and Phase 2 decisions.

**Research date:** 2026-04-10  
**Valid until:** 2026-05-10 for planning purposes unless the phase context or dependency lockfile changes first.
