# Delta Preservation

Automated revision reconciliation for AS9102 Form 3 characteristics, with both a standalone CLI pipeline and a production-style review/sign-off web application.

---

## Table of Contents

1. [What this project is (non-technical)](#what-this-project-is-non-technical)
2. [What problem it solves](#what-problem-it-solves)
3. [How the core reconciliation engine works](#how-the-core-reconciliation-engine-works)
   - [Inputs](#inputs)
   - [Outputs](#outputs)
   - [The 8-stage pipeline](#the-8-stage-pipeline)
   - [How classification decisions are made](#how-classification-decisions-are-made)
   - [Semantic callouts (GD&T, weld, finish, fit)](#semantic-callouts-gdt-weld-finish-fit)
4. [Web application capabilities](#web-application-capabilities)
   - [Setup flow](#setup-flow)
   - [Run submission and validation](#run-submission-and-validation)
   - [Run lifecycle states](#run-lifecycle-states)
   - [Review workflow](#review-workflow)
   - [Sign-off, amendments, and packet versioning](#sign-off-amendments-and-packet-versioning)
   - [Exports](#exports)
   - [Admin features](#admin-features)
5. [Data model and persistence](#data-model-and-persistence)
6. [Repository structure](#repository-structure)
7. [Installation and usage](#installation-and-usage)
8. [Configuration and runtime environment](#configuration-and-runtime-environment)
9. [Testing](#testing)
10. [Limitations and current scope boundaries](#limitations-and-current-scope-boundaries)
11. [Who this is for](#who-this-is-for)

---

## What this project is (non-technical)

In aerospace and defense manufacturing, engineering drawings are revised frequently (Rev A → Rev B, etc.). Every drawing characteristic that appears on AS9102 Form 3 must stay traceable across those revisions.

**Delta Preservation** helps quality teams answer one question with evidence:

> “For each previously inspected characteristic, did it stay the same, change, disappear, or get replaced?”

Instead of manually diffing two complex drawings, the system:

- reads the old FAIR Form 3 spreadsheet,
- finds where those characteristics live on Rev A,
- aligns Rev A to Rev B,
- finds likely Rev B counterparts,
- classifies the delta,
- and emits auditable evidence (JSON + snippets + review/sign-off artifacts).

It is designed to support **partial FAIR** decision-making, while keeping a reviewer-in-the-loop workflow for accountability.

---

## What problem it solves

Manual drawing revision reconciliation is expensive and error-prone because:

- annotation locations move,
- views shift independently,
- notation formats change,
- title blocks and noise interfere with naive image comparison,
- and a missed characteristic can cause either risk (under-inspection) or waste (full re-FAIR).

This project addresses those failure modes with a hybrid of:

- PDF text extraction,
- computer vision alignment,
- numeric/text/context scoring,
- typed semantic parsing,
- and explicit human review controls.

---

## How the core reconciliation engine works

The core library lives under `delta_preservation/` and can be run directly (CLI) or via the web task queue.

### Inputs

Required inputs for each run:

1. **Rev A PDF** (old drawing revision)
2. **Rev B PDF** (new drawing revision)
3. **Form 3 XLSX** (existing FAIR characteristic list)

Expected Form 3 content is discovered by scanning header rows for fields equivalent to:

- characteristic number,
- reference location,
- characteristic designator,
- requirement.

### Outputs

Each run creates a deterministic output directory:

```text
out/<part>_<timestamp>_<hash>/
├── delta_packet.json
├── snippets/
│   ├── char_###_revA_p0.png
│   └── char_###_revB_p0.png
├── debug/
│   ├── form3_chars.json
│   ├── tolerance_*.json
│   └── ...
└── packets/
    └── v1.pdf (or later versions if amended)
```

`delta_packet.json` is the machine-readable source of truth for downstream review/export.

### The 8-stage pipeline

The orchestrator is `run_pipeline()` in `delta_preservation/cli.py`.

1. **Form 3 parsing**
   - Reads and validates Form 3 rows.
   - Parses tolerance hints from requirement text.
2. **Balloon detection (Rev A)**
   - Uses PDF text + CV methods to find ballooned characteristic IDs.
3. **Text extraction (Rev A)**
   - Extracts all spans with PDF-space bounding boxes.
4. **Anchor building**
   - Maps Form 3 characteristics to Rev A text/balloon anchors.
   - Optionally narrows search using inferred drawing grid zones from Form 3 reference locations.
5. **Alignment**
   - Computes ORB/RANSAC image transform between Rev A and Rev B.
   - Computes text-span-based transform candidates.
   - Chooses transform strategy based on inlier behavior (including near-identity ORB guardrails).
6. **Candidate matching**
   - Searches Rev B around predicted anchor locations.
   - Scores by location/text/context.
   - Assigns with one-to-one style constraints and grouped-span handling.
7. **Classification**
   - Emits `unchanged`, `changed`, `removed`, `added`, or `uncertain`.
   - Includes tolerance and semantic comparison hooks.
   - Applies rescue logic for alignment misses and keyword-only anchors.
8. **Output / evidence generation**
   - Builds snippet crops with normalization/coverage expansion.
   - Writes packet JSON and debug artifacts.

### How classification decisions are made

Classification uses layered logic, not just one score.

- **Match unavailable** → initially tends toward `removed`, but with rescue scans that can downgrade to `uncertain` if plausible off-window matches are found.
- **Match exists** → compares:
  - location confidence,
  - numeric/text compatibility,
  - contextual neighborhood,
  - tolerance deltas,
  - requirement-type compatibility.
- **New Rev B spans not claimed** become `added` characteristics.

The engine also protects against common false positives (notes blocks, title block text, boilerplate tolerances, incompatible callout families).

### Semantic callouts (GD&T, weld, finish, fit)

Beyond raw numeric matching, the packet supports a typed semantic envelope per item:

- GD&T
- Weld
- Surface finish
- Fit/class callouts

Each semantic callout includes:

- provenance metadata,
- parser state (`parsed`, `empty`, `error`, etc.),
- normalized text,
- family-specific payload.

Semantic comparison can override formatting-only differences and improve changed/unchanged decisions when both sides parse cleanly.

---

## Web application capabilities

The web app in `shop/` wraps the pipeline into a complete controlled workflow using FastAPI + Jinja2 + HTMX + Huey + SQLite.

### Setup flow

First-run setup currently uses a **3-step wizard**:

1. shop name,
2. admin username,
3. admin password.

When setup is incomplete, middleware forces navigation to setup routes.

### Run submission and validation

Authenticated users can create runs by uploading Rev A, Rev B, and Form 3.

Validation includes:

- PDF raster detection (scanned-image PDFs are rejected),
- PDF page count reporting,
- XLSX readability and Form 3-preview parsing.

Files are persisted to an uploads directory, then a Huey task is enqueued.

### Run lifecycle states

Run status values in the current system include:

- `queued`
- `running`
- `completed`
- `warning`
- `failed`
- `reviewing`
- `signing_off`
- `signed_off`

Stage-level progress (`Form 3 parsing` through `Output`) is streamed via SSE for near-real-time UI updates.

### Review workflow

When the pipeline succeeds, review items are materialized from `delta_packet.json`.

For each item, reviewer can:

- **Approve** pipeline result,
- **Override** classification (note required),
- **Reset** an item back to pending before sign-off.

Filtering is available by item decision state and pipeline classification.

#### Admin-only debug review mode

There is an additional debug layer for admins:

- per-item debug verdict capture,
- corrected fields/explanations,
- run-level debug notes,
- downloadable `debug_report.json` once complete.

This is designed for model/heuristic evaluation and adjudication workflows without mutating official reviewer decisions.

### Sign-off, amendments, and packet versioning

Sign-off behavior:

- blocked until all items are reviewed,
- transitions run through `signing_off` to `signed_off`,
- writes immutable packet PDF version metadata to `packet_versions`.

Amendment behavior:

- only allowed on `signed_off` runs,
- clones run metadata and review items into a new run,
- preserves packet version chain (v2, v3, ...),
- allows decision corrections while original signed run remains immutable.

### Exports

For signed-off runs:

- audit packet PDF (`/exports/{run_id}/audit-packet.pdf?version=N`),
- audit packet CSV,
- work-order PDF,
- work-order CSV.

Work orders include only effective `changed` and `added` items (respecting reviewer overrides).

### Admin features

Admins can:

- manage users (create/deactivate),
- update shop name,
- set retention policy (days),
- reassign reviewers for non-signed-off runs,
- access debug review tooling.

A scheduled Huey periodic task purges old runs/files for deletable statuses according to retention settings.

---

## Data model and persistence

Primary ORM entities (`shop/models.py`):

- `User`, `UserSession`
- `ShopConfig`
- `Run`
- `RunAlert`
- `ReviewItem`

Persistence and infrastructure:

- SQLite for app state,
- SQLAlchemy ORM,
- Alembic migrations,
- SqliteHuey for background jobs,
- WeasyPrint for packet/work-order PDF generation.

---

## Repository structure

```text
.
├── delta_preservation/     # Core reconciliation engine
│   ├── cli.py              # Pipeline orchestrator
│   ├── io/                 # PDF/XLSX ingestion
│   ├── vision/             # Balloons, alignment, grid, snippet utilities
│   ├── reconcile/          # Anchors, matching, normalization, classification
│   └── types.py            # Pydantic packet schema
├── shop/                   # Web product (FastAPI app)
│   ├── routers/            # HTTP routes
│   ├── services/           # Business logic
│   ├── templates/          # Jinja templates
│   ├── middleware/         # Setup guard
│   └── tasks.py            # Huey tasks and cleanup cron
├── tests/                  # Pipeline + web tests
├── assets/                 # Sample fixtures
├── docker/                 # Container runtime definition
├── run.py                  # CLI convenience wrapper for assets/<part>
└── run_web.py              # Docker-oriented startup/seed script
```

---

## Installation and usage

### Requirements

- Python `>=3.10,<3.13`
- `uv` recommended (or pip)
- system packages required by OpenCV/WeasyPrint depending on platform

### Install

```bash
uv sync
```

(or)

```bash
pip install -e .
```

### Run sample fixture from CLI

```bash
uv run python run.py part1
```

Optional:

```bash
uv run python run.py part1 --dpi 150 --out_dir ./out
```

### Run direct module CLI

```bash
python -m delta_preservation.cli \
  --revA_pdf /path/revA.pdf \
  --revB_pdf /path/revB.pdf \
  --form3_xlsx /path/FAIR.xlsx \
  --part_name demo \
  --out_dir ./out \
  --dpi 300
```

### Run web app locally

In one terminal:

```bash
uv run uvicorn shop.app:app --reload --port 8000
```

In another terminal:

```bash
uv run huey_consumer.py shop.tasks.huey --workers=1 --worker-type=thread
```

Open <http://localhost:8000>.

### Run via Docker

```bash
cd docker
docker compose up --build
```

This uses `run_web.py` to run migrations, seed defaults, and start uvicorn.

---

## Configuration and runtime environment

Common environment variables:

- `DATABASE_URL` (default local SQLite path)
- `HUEY_DB`
- `OUT_DIR`
- `UPLOADS_DIR`
- `ADMIN_EMAIL` (used as default admin username seed)
- `ADMIN_DEFAULT_PASSWORD`

Notes:

- Task/output paths include writable-dir fallbacks when `/app/...` is unavailable.
- Password hashing uses bcrypt via `pwdlib`.
- Session duration is currently 8 hours.

---

## Testing

Test suite is in `tests/` and covers:

- pipeline logic and classification edge cases,
- semantic comparison/extraction,
- task-state transitions,
- RBAC/auth/setup/admin flows,
- review/sign-off/amendment behavior,
- export generation.

Run tests with:

```bash
pytest -q
```

---

## Limitations and current scope boundaries

Current known limitations:

1. **Single-page processing path in pipeline** (page index 0 for core extraction/alignment/cropping).
2. **Vector PDFs strongly preferred**; scanned/raster drawings are rejected by web validation.
3. **Ballooning assumptions** (numbered/circled characteristics) still drive anchoring quality.
4. **Heuristic-heavy matching** means some edge cases intentionally surface as `uncertain` for human adjudication.
5. **No external PLM/QMS integrations** are included in this repository.

This codebase is currently best viewed as a robust prototype / pre-product platform with production-like workflow controls.

---

## Who this is for

This project is useful for:

- quality engineers managing AS9102 revision deltas,
- manufacturing teams needing defensible partial FAIR scoping,
- developers building audit-ready “human + automation” review systems around drawing changes.

