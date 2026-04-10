# Stack Research

**Domain:** Ground-truth-assisted debug infrastructure for a drawing-reconciliation system
**Researched:** 2026-04-09
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.10-3.12 | Core evaluation, report generation, comparison rules | The existing reconciliation engine, review services, and artifact generation already live in Python, so this milestone should extend the current monolith instead of introducing a second runtime |
| FastAPI + Jinja2 | Current project stack | Admin/debug workflow endpoints and focused review UI | The existing review/debug flow already uses FastAPI, server-rendered templates, and HTMX-style interactions; keeping the same surface minimizes integration cost |
| SQLAlchemy | 2.x | Run metadata, future exception/history records, contradiction analysis inputs | The app already persists runs and review items through SQLAlchemy, making it the natural home for structured exception history rather than ad hoc files |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| PyMuPDF | Current project dependency | Interpret snippet coordinates and optionally normalize snippet windows | Use for evaluating whether generated snippets still capture the intended annotation/context from the PDFs |
| Pydantic | >=2.5,<3.0 | Schema validation for ground truth, evaluation output, and exception payloads | Use for immutable contract enforcement around `ground_truth.json`, auto-evaluation rows, and history-layer records |
| OpenCV + NumPy | Current project dependency set | Reuse existing image-space utilities where snippet tolerance needs geometric heuristics | Use only when bbox overlap or image-derived tolerance checks are needed beyond simple center-distance windows |
| pytest + httpx | Current project test stack | Regression coverage for evaluator logic, export payloads, and focused debug UI/API flows | Use for deterministic fixture-based tests against known parts and contradiction cases |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `uv` | Python dependency and test environment management | Keep this as the primary workflow to match the rest of the repository |
| Alembic | Schema changes for exception/history persistence | Add migrations only when the history layer moves from file-only to database-backed state |
| Existing asset fixtures | Known part pairs plus `ground_truth.json` contracts | Treat these as the benchmark corpus for iterative validation rather than one-off examples |

## Installation

```bash
# Core
uv sync --locked

# Supporting UI assets
npm install

# Dev/test workflow
pytest
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Extend the current Python/FastAPI monolith | Build a separate evaluation microservice | Only if evaluation throughput or ownership boundaries justify another deployable, which this solo-developer workflow does not |
| SQLAlchemy-backed exception/history layer | JSON-only history files per run/part | Acceptable for a very short-lived prototype, but it becomes weak once contradiction analysis spans many runs |
| Pydantic-validated contracts | Loosely shaped dicts everywhere | Only for throwaway experiments; this milestone needs stable schemas to avoid contradictory debug data |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Auto-editing `ground_truth.json` during runs | It destroys the stable baseline needed for cross-run comparison | Keep `ground_truth.json` immutable and write alternate acceptable outcomes to a separate history layer |
| Part-specific correction rules inside the classifier | They improve one fixture while weakening generalization across other parts | Keep evaluation/history logic separate from core reconciliation behavior |
| Pixel-perfect snippet checks | The user only needs the target annotation and surrounding context in view, not exact center matches | Use tolerant windows and visibility heuristics |

## Stack Patterns by Variant

**If the exception/history layer stays lightweight at first:**
- Use append-only JSON artifacts per run plus normalized services to read them
- Because this lets the milestone ship quickly while preserving an upgrade path to DB-backed analysis

**If contradiction analysis becomes cross-run and query-heavy:**
- Promote exceptions/history into first-class SQLAlchemy tables
- Because trend inspection and conflict detection are easier with normalized records than with scattered JSON files

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| FastAPI (current project version) | Pydantic 2.x | Already proven in the existing app; keep new evaluator models on the same contract system |
| SQLAlchemy 2.x | Alembic 1.18.x | Existing migration flow supports additive history-layer tables |
| PyMuPDF + OpenCV | NumPy current project version | Already co-exist in the reconciliation pipeline; reuse rather than re-implement geometry logic |

## Sources

- `/home/khoa2/delta-preservation/pyproject.toml` — current Python dependency set
- `/home/khoa2/delta-preservation/package.json` — current UI build tooling
- `/home/khoa2/delta-preservation/.planning/codebase/STACK.md` — existing stack summary
- `/home/khoa2/delta-preservation/.planning/PROJECT.md` — milestone goals and constraints

---
*Stack research for: ground-truth-assisted debug infrastructure*
*Researched: 2026-04-09*
