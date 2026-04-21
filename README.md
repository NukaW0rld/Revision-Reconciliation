# Delta Preservation

Delta Preservation is a brownfield aerospace drawing-reconciliation system. It has two validated surfaces that use the same reconciliation packet and evidence model:

- a standalone pipeline for comparing Rev A drawings, Rev B drawings, and AS9102 Form 3 data
- a FastAPI web workflow for run submission, review, sign-off, export, and maintainer debug inspection

The current project focus is not a new end-user product. The focus is a faster, more consistent maintainer debug loop that compares each run against stable part-level ground truth, improves the algorithm across many drawing pairs, and avoids part-specific hacks.

## Current project stage

Delta Preservation has completed two shipped milestones:

- `v1.0` Ground Truth Debug Workflow shipped on 2026-04-12.
- `v1.1` Cross-part Characteristic Matching Refinement shipped on 2026-04-19.

The repository is currently between milestones and ready for the next planning cycle. The validated baseline after `v1.1` is 500 passed tests and 2 expected failures.

Shipped `v1.0` capabilities include deterministic ground-truth evaluation, exception-only debug review, accepted-alternate history, and `debug_report.json` output while keeping canonical `ground_truth.json` fixtures stable.

Shipped `v1.1` capabilities include GD&T parser fixes, classification fixes, added-characteristic detection improvements, shared title-block exclusion, `confidence_flags`, sign-off gating on unresolved debug exceptions, signed debug snapshots for export fidelity, and live web run-to-review end-to-end coverage.

The active next focus is cross-part consistency tooling: contradiction detection between accepted alternates, benchmark trend summaries, and overfitting warnings before a maintainer accepts a new alternate.

## Domain problem

AS9102 Form 3 is the characteristic accountability form used in First Article Inspection Reporting. It records drawing characteristics such as dimensions, tolerances, GD&T frames, surface finishes, and notes that must be inspected or accounted for.

When an aerospace drawing moves from Rev A to Rev B, a quality engineer needs to know what happened to each characteristic already represented in the prior FAIR:

- the characteristic may be unchanged
- the requirement may have changed
- the annotation may have moved or been reformatted
- the characteristic may have been removed
- a new Rev B characteristic may have been added

This matters for partial FAIR decisions. If only some characteristics changed, a shop may be able to limit inspection work to the affected characteristics. Manual reconciliation is risky because complex drawings contain dense callouts, reused values, nearby notes, title-block noise, and view shifts that make visual diffing unreliable. A missed characteristic can create under-inspection risk; a false change can create unnecessary reinspection work.

Delta Preservation keeps a human reviewer in the loop. The system generates evidence-rich classifications, but nonconforming and ambiguous results still require manual inspection because more than one outcome may be acceptable in some cases.

## System overview

The standalone pipeline and the web workflow share the same core output:

- `delta_packet.json`: structured classifications, confidence, reasons, component scores, semantic callouts, evidence locations, snippet references, and advisory flags
- snippet images: Rev A and Rev B crops used to inspect the evidence behind a row
- debug artifacts: intermediate JSON and reports used to understand why a row matched, failed, or needs review

The core reconciliation code lives in `delta_preservation/`. The web application in `shop/` persists uploaded files, run state, review decisions, debug verdicts, sign-off metadata, and exports. The web tier delegates the actual algorithm work to the same pipeline used by the standalone CLI.

## Core pipeline

The pipeline compares three inputs:

1. Rev A PDF: the prior drawing revision
2. Rev B PDF: the new drawing revision
3. Form 3 XLSX: the prior AS9102 Form 3 characteristic list

Typical output is written under `out/<part>_<timestamp>_<hash>/`:

```text
out/<part>_<timestamp>_<hash>/
├── delta_packet.json
├── snippets/
│   ├── char_<id>_revA_p0.png
│   └── char_<id>_revB_p0.png
└── debug/
    ├── form3_chars.json
    ├── tolerance_*.json
    └── other intermediate diagnostics
```

`delta_packet.json` is the primary machine-readable result. The web review queue, export services, debug review, and regression tests all consume packet data rather than re-running classification logic.

The pipeline is orchestrated by `run_pipeline()` in `delta_preservation/cli.py` and follows eight stages:

1. Form 3 parsing: reads the spreadsheet, locates characteristic columns, normalizes rows, and extracts tolerance hints.
2. Rev A balloon detection: finds characteristic balloons from PDF text and computer-vision detections.
3. Rev A text extraction: extracts text spans with PDF-space bounding boxes.
4. Anchor building: links Form 3 rows to Rev A evidence using balloon numbers, text spans, reference locations, and grid hints.
5. Rev B extraction and alignment: extracts Rev B spans and estimates Rev A to Rev B alignment with image and text-based transforms.
6. Candidate matching: searches near predicted Rev B positions and scores candidate spans by location, text, and context.
7. Classification and snippets: assigns `unchanged`, `changed`, `removed`, `added`, or `uncertain`, then generates evidence crops.
8. Packet output: writes `delta_packet.json` and debug artifacts for review and export.

## Algorithm and debug accuracy

Classification is deliberately layered. Location alone is not enough, and text equality alone is not enough. The classifier combines:

- spatial proximity after alignment
- text and numeric similarity
- contextual neighborhood checks
- tolerance parsing
- semantic requirement type compatibility
- evidence quality and rescue scans

The packet classification statuses are:

- `unchanged`: the Rev B evidence is materially equivalent to Rev A/Form 3
- `changed`: the characteristic still exists but the requirement changed
- `removed`: the prior characteristic was not found in Rev B
- `added`: new Rev B evidence appears to be a new characteristic not claimed by a Rev A/Form 3 row
- `uncertain`: the system found plausible evidence but not enough certainty for an automated classification

`v1.1` improved the algorithm in several general-purpose areas:

- GD&T semantic parsing handles compact concatenated frames, word-form names such as "circularity", and composite multi-compartment feature-control frames.
- `confidence_flags` are packet-native advisory data surfaced in review and export, rather than warnings re-derived from free-text reasons.
- Adjacency bleed suppression reduces false `changed` results when nearby balloon text leaks into a Rev B span.
- Removed-plus-added reconciliation can merge close compatible pairs into a single `changed` characteristic instead of two unrelated rows.
- Asymmetric tolerance detection identifies cases such as symmetric `+/-T` changing to `+a/-b`.
- Added-characteristic detection uses grouped evidence, title-block exclusion, explained-by-match suppression, canonical `added:<index>` tokens, leading-zero normalization, and deterministic duplicate claiming.
- A shared `span_is_excluded_for_annotation_search` contract keeps title-block regions out of anchor search, matching, rescue scans, and added detection.

The current added-characteristic benchmark claim is 33/35 ground-truth-added rows detected across the debug corpus. The remaining two are explicit deferrals for Part 5 indexes 16 and 17; they require thread/countersink matching-layer support and are not considered solved.

Known deferred algorithm items also include Part 9 `truth_index 42`, where an explained-by-match suppressor can falsely absorb a flatness row, and composite GD&T cases with positional-zone modifiers.

## Ground-truth debug workflow

The maintainer debug loop is centered on immutable fixtures:

```text
assets/<part>/ground_truth.json
```

Those files are canonical references. Runs may read them for evaluation, but the workflow does not rewrite them. Keeping the truth stable prevents accidental benchmark drift and keeps algorithm changes honest across reruns.

The `v1.0` workflow added deterministic conformance evaluation:

- completed packet rows are compared with the part's canonical `ground_truth.json`
- requirement, status, and evidence expectations are checked with ordered mismatch codes
- snippet evaluation favors useful visible context over exact center-coordinate matching
- conforming rows auto-pass
- nonconforming or ambiguous rows enter an exception-only review queue
- `debug_report.json` records conforming rows, unresolved exceptions, accepted alternates, mismatch reasons, and notes

Accepted alternates are stored in a separate history layer. They are useful when a row is not an exact canonical match but a maintainer decides the result is still acceptable. Alternate reuse is intentionally narrow: later same-part reruns can auto-conform when the reviewed fingerprint matches, but the canonical `ground_truth.json` file remains unchanged and alternates are not generalized across unrelated parts.

This debug workflow is for one maintainer improving the algorithm. It is not a multi-user debug product and it is not an automatic self-tuning loop.

## Web workflow

The web application wraps the pipeline with a reviewable shop workflow.

Setup and access:

- FastAPI serves the app from `shop.app:app`.
- First-run setup captures shop name, admin username, and admin password.
- Authentication uses session tokens backed by `UserSession`.
- Roles separate admin functions from engineer review/submission work.

Run submission and processing:

- authenticated users upload Rev A PDF, Rev B PDF, and Form 3 XLSX
- upload validation checks PDF integrity, rejects scanned/raster PDFs, reports page counts, and previews Form 3 parsing
- files are stored under the configured uploads directory
- a Huey background task runs the pipeline
- Server-Sent Events stream stage progress from Form 3 parsing through packet output
- failures are recorded on the run with stage and alert details

Review:

- completed runs materialize `ReviewItem` rows from `delta_packet.json`
- reviewers approve, override, or reset individual decisions
- filters support pending, approved, overridden, and classification-focused review
- packet `confidence_flags` appear on normal review cards and debug surfaces

Admin debug review:

- admins see exception-focused debug queues based on ground-truth evaluation
- conforming rows stay visible in summaries but do not require verdicts
- per-row debug verdicts, corrected fields, explanations, and run-level notes feed `debug_report.json`
- unresolved debug exceptions block sign-off

Sign-off and export:

- sign-off is blocked until normal review and debug exception requirements are satisfied
- sign-off writes a signed debug snapshot so exported artifacts preserve the same advisory/debug state the maintainer reviewed
- exports include audit packet PDF/CSV and work-order PDF/CSV
- work-order exports focus on effective `changed` and `added` rows after reviewer decisions

Amendments and administration:

- signed-off runs can be amended by cloning review state into a new packet version
- the original signed run remains immutable
- admins can manage users, shop settings, reviewer assignment, and retention policy
- Huey cleanup removes old runs/files when they satisfy the configured retention rules

## Architecture and persistence

Delta Preservation separates the reconciliation engine from web workflow state:

- `delta_preservation/` contains PDF/XLSX ingestion, vision utilities, semantic parsing, matching, classification, evaluation, and packet generation.
- `shop/` contains FastAPI routers, Jinja2 templates, SQLAlchemy models, services, middleware, exports, and Huey tasks.
- `run.py` is a convenience CLI wrapper for checked-in `assets/<part>/` fixtures.
- `run_web.py` is the Docker startup entry point that runs migrations, seeds the default admin when needed, and starts Uvicorn.

Persistence and runtime pieces:

- SQLAlchemy models store users, sessions, shop configuration, runs, alerts, review items, debug state, accepted alternates, and packet versions.
- Alembic manages schema migrations.
- SQLite is the default database, configurable via `DATABASE_URL`.
- SqliteHuey stores background task state, configurable via `HUEY_DB`.
- Jinja2 templates render the server-side UI.
- WeasyPrint generates PDF audit packets and work orders.
- Tailwind CSS and DaisyUI provide the compiled stylesheet.

The core algorithm does not depend on FastAPI. The web tier reads and writes run state, but it delegates reconciliation to `delta_preservation/` and consumes the resulting packet.

## Repository structure

```text
.
├── delta_preservation/       # Core reconciliation, evaluation, packet generation
│   ├── cli.py                # run_pipeline() and direct CLI entry point
│   ├── evaluation/           # Ground-truth conformance evaluation
│   ├── io/                   # PDF rendering/text extraction and XLSX parsing
│   ├── reconcile/            # Anchors, matching, normalization, classification
│   ├── vision/               # Balloons, alignment, grids, snippets, bbox helpers
│   └── types.py              # Pydantic packet and evidence models
├── shop/                     # FastAPI web workflow
│   ├── routers/              # HTTP route modules
│   ├── services/             # Auth, runs, review, exports, amendments, history
│   ├── templates/            # Jinja2 templates
│   ├── middleware/           # Setup and request middleware
│   ├── models.py             # SQLAlchemy ORM models
│   ├── database.py           # Engine/session setup
│   └── tasks.py              # Huey tasks and retention cleanup
├── tests/                    # Algorithm, evaluation, web, export, and E2E tests
├── assets/                   # Debug corpus fixtures and ground_truth.json files
├── alembic/                  # Database migrations
├── docker/                   # Dockerfile, compose file, supervisord config
├── static/                   # Tailwind input and compiled CSS
├── run.py                    # Convenience runner for assets/<part>
├── run_web.py                # Container startup and admin seeding
├── pyproject.toml            # Python metadata, dependencies, pytest config
├── package.json              # CSS build dependencies and scripts
└── uv.lock                   # Locked Python dependency graph
```

## Installation and usage

Requirements:

- Python 3.10 through 3.12
- `uv` for locked dependency installation
- Node.js 22 or compatible npm runtime if rebuilding CSS outside Docker
- Docker Engine if running the containerized stack
- SQLite for the default local database
- platform libraries required by OpenCV and WeasyPrint

Install Python dependencies:

```bash
uv sync --locked
```

Build CSS when template or style changes require it:

```bash
npm install
npm run build:css
```

Run a checked-in sample part:

```bash
uv run python run.py part1
```

Run a sample part with explicit rendering/output options:

```bash
uv run python run.py part1 --dpi 150 --out_dir ./out
```

Run the direct module CLI:

```bash
uv run python -m delta_preservation.cli \
  --revA_pdf /path/to/revA.pdf \
  --revB_pdf /path/to/revB.pdf \
  --form3_xlsx /path/to/FAIR.xlsx \
  --part_name demo \
  --out_dir ./out \
  --dpi 300
```

Run the web app locally:

```bash
uv run alembic upgrade head
uv run uvicorn shop.app:app --reload --port 8000
```

In a second terminal, run the worker:

```bash
uv run huey_consumer.py shop.tasks.huey --workers=1 --worker-type=thread
```

Open `http://localhost:8000`.

Run with Docker Compose:

```bash
cd docker
docker compose up --build
```

The Docker service runs migrations on startup, starts Uvicorn on port 8000, and runs the Huey worker under Supervisor. The compose file mounts `data/`, `out/`, and read-only `assets/`.

Common environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `DATABASE_URL` | SQLAlchemy database URL | `sqlite:///./shop.db` locally, `/app/data/shop.db` in Docker |
| `HUEY_DB` | Huey queue SQLite path | `/app/data/huey.db` in Docker |
| `OUT_DIR` | Pipeline output directory | `/app/out` in Docker |
| `UPLOADS_DIR` | Uploaded PDF/XLSX storage | `/app/data/uploads` in Docker |
| `ADMIN_EMAIL` | Seed admin username | `admin@shop.local` |
| `ADMIN_DEFAULT_PASSWORD` | Seed admin password | `changeme` |

Change the default admin credentials for any non-local deployment.

## Testing

Run the full suite:

```bash
uv run pytest
```

The `pyproject.toml` pytest configuration already adds `-q`, so this is equivalent to a quiet run over `tests/`.

Focused examples:

```bash
uv run pytest tests/test_classify_bugfixes.py
uv run pytest tests/test_debug_verdicts.py
uv run pytest tests/test_web_run_review_e2e.py
uv run pytest tests/test_phase6_asset_regression.py
```

The suite covers:

- Form 3 parsing and packet serialization
- GD&T, tolerance, surface-finish, weld, and fit semantic parsing
- classification bug fixes and regression cases
- ground-truth conformance evaluation and accepted alternates
- debug queue behavior and `debug_report.json` export
- sign-off gating, signed debug snapshots, and export fidelity
- web auth, setup, review, amendments, admin settings, retention cleanup
- live corpus run-to-review coverage using real assets
- cross-part benchmark guards for added-characteristic detection

Current documented baseline: 500 passed, 2 xfailed after `v1.1`.

## Constraints, scope boundaries, and active next items

Hard constraints:

- Generalization: fixes must improve behavior across many parts, not encode one drawing pair's quirks.
- Ground-truth stability: `assets/<part>/ground_truth.json` is canonical and manually curated.
- Human review boundary: nonconforming rows still require inspection when multiple outcomes may be acceptable.
- Developer-only debug scope: the debug loop is optimized for one maintainer's run, review, rerun cycle.
- Snippet tolerance: useful visible context matters more than exact center-coordinate matching.

Current scope boundaries:

- scanned/raster PDFs are rejected by web validation; vector PDFs are expected
- the main extraction/alignment/snippet path is currently centered on page index 0
- no PLM, QMS, ERP, or customer-system integrations are included
- no automatic algorithm self-tuning from review history
- no multi-user debug collaboration workflow
- no guarantee that every `uncertain` row can be resolved without a knowledgeable reviewer

Known active next items:

- cross-part contradiction detection between accepted alternates
- benchmark trend summaries across runs
- overfitting warning before accepting a new alternate
- Part 5 thread/countersink matching-layer support for indexes 16 and 17
- Part 9 `truth_index 42` false absorption in explained-by-match suppression
- composite GD&T positional-zone modifiers
- better reporting around long-term benchmark movement and regression risk

## Glossary

Rev A / Rev B: The prior and new drawing revisions being compared.

AS9102 Form 3: The FAIR characteristic accountability form that lists drawing characteristics, requirements, and inspection results.

Characteristic: A drawing requirement that must be accounted for, such as a dimension, tolerance, GD&T frame, note, surface finish, weld callout, or fit/class requirement.

Delta packet: The structured `delta_packet.json` output containing classifications, evidence, reasons, scores, snippets, semantic callouts, and advisory flags.

Ground truth: The canonical expected result for a part, stored in `assets/<part>/ground_truth.json` and used for deterministic evaluation.

Accepted alternate: A reviewed outcome that differs from canonical truth but is acceptable for the same part and fingerprint; stored in history without changing ground truth.

Debug exception: A nonconforming or ambiguous packet row that needs maintainer review in the debug workflow.

Confidence flag: Packet-native advisory data, stored in `confidence_flags`, that explains a known risk or caveat without changing the classification by itself.

FAIR: First Article Inspection Report, the aerospace quality record that demonstrates a part was inspected against drawing and specification requirements.
