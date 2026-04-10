# Architecture Research

**Domain:** Ground-truth-assisted debug infrastructure for a brownfield reconciliation system
**Researched:** 2026-04-09
**Confidence:** HIGH

## Standard Architecture

### System Overview

```text
┌──────────────────────────────────────────────────────────────────────┐
│                      Existing Run Pipeline Layer                     │
├──────────────────────────────────────────────────────────────────────┤
│  Run submission  Background task  delta_packet.json  Snippet assets │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                     Debug Evaluation Layer                           │
├──────────────────────────────────────────────────────────────────────┤
│  Ground truth loader  Conformance evaluator  Snippet tolerance      │
│  rule set             report assembler                              │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                   Exception / History Layer                          │
├──────────────────────────────────────────────────────────────────────┤
│  Accepted alternates  Review rationale  Contradiction records       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                       Focused Review Surface                         │
├──────────────────────────────────────────────────────────────────────┤
│  Exception-only queue  Run notes  debug_report.json export          │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Ground truth catalog | Read immutable part-level truth data and validate it before comparison | File-backed JSON contracts under `assets/<part>/ground_truth.json` with Pydantic validation |
| Conformance evaluator | Compare each pipeline row to canonical truth and produce auto-pass / needs-review decisions | New service module layered on top of `delta_packet.json`, `ReviewItem`, and snippet evidence |
| Exception/history store | Persist alternate acceptable outcomes, rationale, and future contradiction-analysis inputs | Append-only JSON or SQLAlchemy-backed records kept separate from `ground_truth.json` |
| Focused debug UI/export | Show only nonconforming rows, preserve manual notes, and emit deterministic reports | Extend current review/debug routes and export payload assembly in `shop/services/review.py` |

## Recommended Project Structure

```text
delta_preservation/
├── debug/                    # Evaluation primitives for debug comparison
│   ├── ground_truth.py       # Ground truth schema + loading
│   ├── compare.py            # Classification/requirement/snippet comparison rules
│   └── tolerance.py          # Snippet visibility / tolerance heuristics
shop/
├── services/
│   ├── debug_evaluator.py    # Run-level auto-evaluation orchestration
│   ├── debug_history.py      # Exception/history persistence + contradiction checks
│   └── review.py             # Existing export/UI integration points
├── routers/
│   └── review.py             # Focused exception queue and report endpoints
tests/
├── test_debug_ground_truth.py
├── test_debug_history.py
└── test_debug_review_flow.py
```

### Structure Rationale

- **`delta_preservation/debug/`:** keeps comparison logic close to the reconciliation domain without polluting the core classifier with benchmark-specific acceptance rules.
- **`shop/services/`:** keeps run orchestration, history persistence, and UI/report integration inside the existing web application boundary.
- **`tests/`:** fixture-driven regression coverage is essential because the whole milestone exists to improve repeatability across many parts.

## Architectural Patterns

### Pattern 1: Immutable Truth + Append-Only History

**What:** Keep canonical truth separate from reviewed exceptions.
**When to use:** Always, because alternate acceptable outcomes must not silently redefine the benchmark.
**Trade-offs:** Slightly more complexity than rewriting one JSON file, but dramatically better auditability and contradiction analysis.

### Pattern 2: Deterministic Post-Run Evaluation

**What:** Evaluate completed pipeline artifacts after the run, rather than modifying classification behavior inline.
**When to use:** For this milestone, where the goal is faster debugging and evidence, not automatic learning.
**Trade-offs:** Adds one more processing stage, but avoids coupling benchmark logic directly to the classifier.

### Pattern 3: Exception-Only Human Review

**What:** Auto-accept conforming rows and route only disagreements or ambiguity to the UI.
**When to use:** Once the conformance evaluator is trusted enough to reduce manual workload.
**Trade-offs:** Requires careful mismatch reporting so reviewers trust what was auto-passed.

## Data Flow

### Request Flow

```text
Run completes
    ↓
delta_packet.json + snippet artifacts
    ↓
Ground truth loader -> Conformance evaluator -> auto-evaluation results
    ↓
Focused review queue for mismatches/ambiguity
    ↓
Manual decisions -> exception/history layer
    ↓
debug_report.json + contradiction signals
```

### State Management

```text
Ground truth (immutable assets)
    ↓
Per-run evaluation result
    ↓
Focused review decisions
    ↓
Exception/history records
    ↓
Cross-run contradiction analysis
```

### Key Data Flows

1. **Run evaluation:** `delta_packet.json` is compared against `ground_truth.json` and existing exceptions/history to determine auto-pass vs review-needed rows.
2. **Exception capture:** a reviewed nonconforming row produces a durable alternate-acceptable or rejected outcome record without editing truth.
3. **Contradiction analysis:** history records are scanned across runs and parts to detect when accepted alternates imply conflicting algorithm expectations.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 0-100 benchmark runs | File-backed truth plus lightweight history artifacts are acceptable |
| 100-1k benchmark runs | Normalize exception/history records in the database for querying and contradiction checks |
| 1k+ benchmark runs | Add indexed history queries and aggregated benchmark summaries before considering any service split |

### Scaling Priorities

1. **First bottleneck:** contradiction analysis across many ad hoc files; move history into structured records before adding analytics.
2. **Second bottleneck:** snippet/tolerance evaluation cost; cache derived geometry or normalized snippet metadata instead of re-reading PDFs repeatedly.

## Anti-Patterns

### Anti-Pattern 1: Teach the classifier benchmark-specific hacks

**What people do:** fold per-part correction logic directly into matching/classification code.
**Why it's wrong:** it raises accuracy on one fixture while quietly degrading behavior on others.
**Do this instead:** keep evaluator and exception handling as separate layers that expose contradictions rather than hiding them.

### Anti-Pattern 2: Treat every mismatch as a truth update

**What people do:** rewrite the ground truth whenever the algorithm and benchmark disagree.
**Why it's wrong:** it destroys the baseline and makes run-to-run comparisons meaningless.
**Do this instead:** store reviewed alternates and rationale in a separate history layer.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Asset fixtures under `assets/` | Direct file reads | Each part already contains FAIR, Rev A, Rev B, and `ground_truth.json` |
| Existing run output directory | Direct file reads/writes | Reuse `delta_packet.json`, snippet artifacts, and current debug-report export location |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `delta_preservation` -> `shop` | File artifacts + service orchestration | Keep reconciliation output stable and let the web layer evaluate it afterward |
| `shop.services.review` -> new debug evaluator/history services | Direct service calls | Extend the current debug/export workflow without replacing it wholesale |

## Sources

- `/home/khoa2/delta-preservation/.planning/codebase/ARCHITECTURE.md`
- `/home/khoa2/delta-preservation/shop/services/review.py`
- `/home/khoa2/delta-preservation/.planning/PROJECT.md`
- User-provided debugging workflow details during initialization

---
*Architecture research for: ground-truth-assisted debug infrastructure*
*Researched: 2026-04-09*
