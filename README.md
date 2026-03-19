# Delta Preservation

**Automated inspection characteristic tracking across engineering drawing revisions — built for AS9102 FAIR compliance in aerospace and defense manufacturing.**

---

## Table of Contents

- [What This Does (Plain Language)](#what-this-does-plain-language)
- [The Problem It Solves](#the-problem-it-solves)
- [How It Works](#how-it-works)
  - [The 8-Stage Pipeline](#the-8-stage-pipeline)
  - [Pipeline Walkthrough](#pipeline-walkthrough)
- [What Goes In, What Comes Out](#what-goes-in-what-comes-out)
  - [Inputs](#inputs)
  - [Outputs](#outputs)
  - [Delta Packet Structure](#delta-packet-structure)
- [Classification Statuses](#classification-statuses)
- [Web Application](#web-application)
  - [Setup Wizard](#setup-wizard)
  - [Submitting a Run](#submitting-a-run)
  - [Review Queue](#review-queue)
  - [Sign-Off and Audit Packets](#sign-off-and-audit-packets)
  - [Amendments](#amendments)
  - [Work Orders](#work-orders)
  - [Admin Features](#admin-features)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Running via Command Line](#running-via-command-line)
  - [Running the Web Application](#running-the-web-application)
  - [Docker Deployment](#docker-deployment)
- [Command-Line Reference](#command-line-reference)
- [Repository Structure](#repository-structure)
- [Technical Architecture](#technical-architecture)
  - [Core Pipeline (`delta_preservation/`)](#core-pipeline-delta_preservation)
  - [Web Application (`shop/`)](#web-application-shop)
  - [Key Algorithms](#key-algorithms)
- [Background on AS9102 and FAIRs](#background-on-as9102-and-fairs)
- [Limitations](#limitations)
- [Support](#support)

---

## What This Does (Plain Language)

In aerospace manufacturing, every part has an engineering drawing that specifies exact dimensions, tolerances, and requirements. These are called **design characteristics** — things like "this hole must be Ø6.0 ± 0.1 mm" or "this edge radius must be R3.5 mm."

Before a part goes into production, an inspector physically measures **every single one** of these characteristics and records the results in a formal report called a **FAIR** (First Article Inspection Report). This is an AS9102 standard requirement across the aerospace industry.

When the engineering drawing gets revised (say, Rev A → Rev B), the quality team needs to figure out:

- Which characteristics **stayed the same** — those can keep their existing inspection data
- Which characteristics **changed** — those need to be re-measured
- Which characteristics **were removed** — those get taken off the report
- Which characteristics **were added** — those are new inspection requirements

**Delta Preservation automates this comparison.** It takes the old drawing (Rev A), the new drawing (Rev B), and the existing inspection report (Form 3), then uses computer vision and text analysis to figure out what changed. Instead of an engineer spending hours manually cross-referencing two drawings, the system does it in seconds and produces a structured report with visual evidence for every decision.

---

## The Problem It Solves

Traditional approaches to revision comparison fail because:

- **Layout changes**: When drawings are revised, annotations move. The same dimension that was at the top-left of Rev A might be in the center of Rev B.
- **Projection shifts**: View groups (front view, side view, detail views) can shift independently.
- **Formatting differences**: The same tolerance value "0.150 ± 0.010" might appear as stacked limits ".160 / .140" in a different revision.
- **Scale of the task**: A single drawing can have 20–100+ characteristics. Manually comparing two revisions is tedious, error-prone, and expensive.

When traditional comparison fails, quality teams are forced to do a **complete re-FAIR** — re-inspecting every single characteristic from scratch, even if only one dimension changed. This costs significant time, equipment availability, and money.

Delta Preservation enables **partial FAIRs** (per AS9102 Section 4.6) by precisely identifying which characteristics require re-inspection.

---

## How It Works

### The 8-Stage Pipeline

```
Rev A PDF ──┐                           ┌── delta_packet.json
             │                           │   (structured results)
Form 3 XLSX ─┼──→ [8-Stage Pipeline] ──→┤
             │                           │   snippets/
Rev B PDF ──┘                           └── (visual evidence images)
```

| Stage | Name | What It Does |
|-------|------|-------------|
| 1 | **Form 3 Parsing** | Reads the Excel spreadsheet and extracts every inspection characteristic (number, requirement text, tolerances) |
| 2 | **Balloon Detection** | Finds the circled numbers on Rev A that label each characteristic (uses PDF text extraction + OpenCV circle detection) |
| 3 | **Text Extraction** | Extracts every text annotation from Rev A with precise X/Y coordinates |
| 4 | **Anchor Building** | Links each Form 3 requirement to its physical location on Rev A by matching balloon numbers to nearby annotation text |
| 5 | **Alignment Estimation** | Figures out how Rev B's layout shifted relative to Rev A using ORB feature matching and text-span correlation |
| 6 | **Candidate Matching** | For each Rev A characteristic, searches the predicted location in Rev B and scores potential matches using location + text similarity + spatial context |
| 7 | **Delta Classification** | Makes the final call: unchanged, changed, removed, added, or uncertain — with confidence scores and human-readable reasoning |
| 8 | **Evidence Generation** | Crops side-by-side image snippets from both PDFs for every characteristic, writes the structured JSON output |

### Pipeline Walkthrough

**Stage 1 – Form 3 Parsing**: The system opens the AS9102 Form 3 Excel file, auto-detects the header row by scanning for keywords like "Char No", "Requirement", etc., and reads each row into a structured record. Tolerances (bilateral like `±0.2` and unilateral like `+0.3/+0.1`) are parsed out separately.

**Stage 2 – Balloon Detection**: Engineering drawings use circled numbers ("balloons") to tag inspection points. The system first tries extracting these from the PDF's text layer (fast, reliable for vector PDFs), validating each candidate with OpenCV's Hough circle detection. If too few are found, it falls back to pure computer vision — rendering the page as an image and detecting circles with OCR.

**Stage 3 – Text Extraction**: Using PyMuPDF, every text span on Rev A is extracted with its exact bounding box in PDF coordinate space (points at 72 DPI). This includes dimension values, tolerances, symbols (Ø, R, ±), notes, and all other annotations.

**Stage 4 – Anchor Building**: For each Form 3 characteristic that has a detected balloon, the system searches nearby text spans for the matching annotation. It uses token overlap (Jaccard similarity), distance scoring, and size penalties to find the best match. Each "anchor" records the balloon location, annotation location, requirement text, and a local context window (all nearby text spans within ~2 inches).

**Stage 5 – Alignment**: The system renders both PDFs as images and runs ORB (Oriented FAST and Rotated BRIEF) feature detection with RANSAC homography estimation to compute how Rev B's content shifted. Because engineering drawings have large static title blocks that can fool image-based alignment, there's a parallel text-span alignment that matches identical annotation strings (like "6.0" or "R3.5") between revisions and computes the median displacement. When ORB produces a near-identity transform but text-span alignment detects a real shift, the text-based transform wins.

**Stage 6 – Candidate Matching**: For each anchored characteristic, the system applies the alignment transform to predict where it should appear in Rev B, then searches within a 4-inch radius for candidate text spans. Each candidate is scored on three axes:
- **Location** (35%): How close it is to the predicted position
- **Text** (50%): How well the numeric values, symbols, and patterns match — the primary dimension value is the key discriminator
- **Context** (15%): How similar the surrounding text neighborhood is

A greedy bipartite assignment ensures each characteristic gets at most one match and each Rev B span is used at most once (with a special shared-span fallback for combined annotations like "10 x 90°" that encode multiple characteristics).

**Stage 7 – Classification**: Each matched pair (or unmatched anchor) gets classified:
- **Unchanged**: Key numeric values match, tolerance structure is consistent
- **Changed**: The matched span has different numeric values or tolerance ranges
- **Removed**: No viable candidate found within the search window
- **Added**: Text spans in Rev B that don't correspond to any Rev A characteristic
- **Uncertain**: Ambiguous cases where the system isn't confident

Tolerance comparison is done at the PDF level — extracting upper/lower limit values from the actual drawing annotations and comparing them numerically, handling both `±` notation and stacked limits notation.

**Stage 8 – Evidence Generation**: For every characteristic, the system crops a region from both Rev A and Rev B centered on the annotation, including the balloon marker and enough surrounding context. The paired snippets are normalized to the same size and expanded to 2× coverage for context. These are saved as PNG files alongside a `delta_packet.json` containing all results with full traceability.

---

## What Goes In, What Comes Out

### Inputs

| File | Format | Description |
|------|--------|-------------|
| **Rev A PDF** | PDF (vector preferred) | The previous revision of the engineering drawing, with balloon annotations marking each inspection characteristic |
| **Rev B PDF** | PDF (vector preferred) | The new revision of the engineering drawing (may or may not have balloon annotations) |
| **Form 3 XLSX** | Excel (.xlsx) | The AS9102 Form 3 spreadsheet from the existing FAIR, containing characteristic numbers, requirements, and tolerances |

The Form 3 Excel file must contain a worksheet named "Form3" or "F3" with headers for Char No, Reference Location, Characteristic Designator, and Requirement. The parser auto-detects column positions by scanning for these keywords.

### Outputs

Each pipeline run produces a timestamped directory:

```
out/<part_name>_<timestamp>_<hash>/
├── delta_packet.json             # Complete structured results
├── snippets/                     # Visual evidence images
│   ├── char_001_revA_p0.png     # Rev A snippet for characteristic 1
│   ├── char_001_revB_p0.png     # Rev B snippet for characteristic 1
│   ├── char_002_revA_p0.png
│   ├── char_002_revB_p0.png
│   └── ...
└── debug/                        # Debug artifacts
    ├── form3_chars.json          # Parsed Form 3 data
    └── tolerance_parsing_tests.json
```

### Delta Packet Structure

The `delta_packet.json` is the primary output — a structured JSON file containing every classification decision with full evidence:

```json
{
  "run_id": "part1_2024-01-29T16-12-59_95484c30",
  "inputs": {
    "revA_pdf": "/path/to/revA.pdf",
    "revB_pdf": "/path/to/revB.pdf",
    "form3_xlsx": "/path/to/FAIR.xlsx",
    "dpi": "300"
  },
  "items": [
    {
      "char_no": 17,
      "status": "unchanged",
      "confidence": 0.92,
      "reasons": [
        "Primary dimension matches: 6.0",
        "Numeric values match (100% overlap)",
        "High location agreement after global alignment"
      ],
      "scores": {
        "location": 0.88,
        "text": 0.95,
        "context": 0.42
      },
      "revA": {
        "page": 0,
        "bbox": [284.2, 156.8, 324.1, 178.3],
        "image_path": "snippets/char_017_revA_p0.png"
      },
      "revB": {
        "page": 0,
        "bbox": [291.5, 162.1, 331.4, 183.6],
        "image_path": "snippets/char_017_revB_p0.png"
      }
    }
  ]
}
```

Each item contains:
- **char_no**: The characteristic number from the Form 3
- **status**: The classification decision
- **confidence**: A 0–1 score indicating how certain the system is
- **reasons**: Human-readable explanations (for audit trail)
- **scores**: Breakdown of location, text, and context similarity
- **revA / revB**: Page number, bounding box (PDF coordinates), and path to the cropped evidence image

---

## Classification Statuses

| Status | Meaning | FAIR Impact |
|--------|---------|-------------|
| **Unchanged** | The characteristic exists in Rev B with the same requirement | Carry forward existing inspection data — no re-measurement needed |
| **Changed** | The characteristic exists in Rev B but the requirement differs (different dimension, tolerance, or specification) | Requires re-measurement on the next production article |
| **Removed** | The characteristic from Rev A no longer appears in Rev B | Remove from the updated Form 3 |
| **Added** | A new characteristic exists in Rev B that was not in Rev A | New inspection requirement — add to Form 3 and measure |
| **Uncertain** | The system cannot confidently determine the status | Flagged for manual review by the engineer |

---

## Web Application

The `shop/` directory contains a full web application (FastAPI + Jinja2 + HTMX + Tailwind CSS + DaisyUI) that wraps the pipeline for use by quality teams. It provides file upload, real-time progress tracking, engineer review and sign-off workflows, audit packet generation, and role-based access control.

### Setup Wizard

On first launch, a setup wizard guides the administrator through initial configuration:

1. **Shop Name** — The organization or facility name (appears on audit packets)
2. **Admin Account** — Create the initial admin username and password
3. **Engineer Account** — Create the first engineer user
4. **Form 3 Column Mapping** — Upload a sample Form 3 Excel file to verify the column detection works correctly

Until setup completes, all routes redirect to the wizard (enforced by `SetupGuardMiddleware`).

### Submitting a Run

Engineers upload three files through the browser:

1. **Rev A PDF** — The ballooned drawing (previous revision)
2. **Rev B PDF** — The new drawing revision
3. **Form 3 Excel** — The existing AS9102 Form 3 inspection report

They also enter metadata: part number, revision labels (e.g., "A" and "B"), customer name, and job number. Optionally, they assign a reviewer from the active user list.

The system validates uploaded files immediately:
- PDFs are checked for raster-only content (scanned drawings are flagged as unsupported)
- Excel files are validated for Form 3 structure
- Page counts are reported for multi-page PDFs (user selects which page to analyze)

After submission, the pipeline runs **asynchronously** via a Huey task queue (SQLite-backed, no Redis or external services needed). The browser shows real-time progress via Server-Sent Events (SSE), updating a stage checklist as each of the 8 stages completes.

### Review Queue

Once the pipeline finishes, the run enters a **review queue** where an engineer examines every characteristic classification:

- Each item shows the pipeline's classification (unchanged/changed/removed/added/uncertain) with its confidence score
- Side-by-side Rev A and Rev B evidence snippets are displayed
- The engineer can **approve** the pipeline's decision or **override** it with a different classification and a mandatory note explaining why

Items can be filtered by review status (pending / approved / overridden) and by classification (unchanged / changed / removed / added / uncertain).

A progress bar at the top tracks how many items have been reviewed. All items must be decided before sign-off is allowed.

### Sign-Off and Audit Packets

Once every item is reviewed, the engineer can **sign off** on the run. This:

1. Transitions the run to `signing_off` status
2. Records the signer's identity and timestamp
3. Generates a **PDF audit packet** using WeasyPrint containing:
   - Run metadata (part number, revisions, customer, job number)
   - Shop name and sign-off information
   - Every characteristic with its classification, confidence, Rev A/Rev B snippets, reviewer decision, and any override notes
4. Stores the PDF in the output directory (`packets/v1.pdf`)
5. Transitions to `signed_off` status — the run is now **immutable**

The audit packet PDF is designed for regulatory compliance and can be included in the FAIR documentation package.

A CSV export of the review decisions is also available at any time during or after review.

### Amendments

After a run is signed off, if a correction is needed, an authorized user can create an **amendment**:

- The amendment clones all review items from the original run with their existing decisions pre-filled
- The engineer can change specific decisions as needed
- On sign-off, the amendment generates `v2.pdf` (or `v3.pdf`, etc.), continuing the version chain from the original packet
- The original signed-off run remains immutable

### Work Orders

For characteristics classified as **changed** or **added**, the system can generate a **work order** — a document listing only the characteristics that need re-measurement. Available as both CSV and PDF, it includes:

- Characteristic number
- Priority (RE-MEASURE for changed, NEW for added)
- Requirement text
- Drawing reference (balloon number)

### Admin Features

Administrators can:

- **Manage users**: Create, deactivate, or reactivate engineer and admin accounts
- **Configure settings**: Set the shop name and data retention period (days before old completed/failed runs are auto-cleaned)
- **Reassign runs**: Transfer a run's reviewer to a different engineer

### Run Lifecycle

```
queued → running → completed → reviewing → signing_off → signed_off
                 ↘ failed                                      ↓
                 ↘ warning → reviewing → ...              [amendment]
```

| Status | Description |
|--------|-------------|
| `queued` | Submitted, waiting for Huey worker |
| `running` | Pipeline is executing (stage progress visible via SSE) |
| `completed` | Pipeline finished successfully |
| `failed` | Pipeline error — failure stage and message recorded, alert sent to reviewer |
| `warning` | Pipeline completed but >50% of characteristics have low location confidence |
| `reviewing` | Engineer is reviewing characteristic classifications |
| `signing_off` | Audit packet PDF is being generated |
| `signed_off` | Immutable — fully reviewed and signed off with audit packet |

---

## Getting Started

### Requirements

- **Python 3.10–3.12**
- **uv** (recommended) or pip for dependency management
- **Node.js** (only needed if modifying Tailwind CSS styles)
- System libraries: OpenCV dependencies are handled by `opencv-python-headless`

### Installation

```bash
# Clone repository
git clone <repository-url>
cd delta-preservation

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Running via Command Line

Test data is included in `assets/part1/` and `assets/part2/`:

```bash
# Run on included test fixtures
uv run python run.py part1
uv run python run.py part2

# With custom DPI (default: 300)
uv run python run.py part1 --dpi 150

# Full manual invocation with custom paths
python -m delta_preservation.cli \
  --revA_pdf /path/to/revA.pdf \
  --revB_pdf /path/to/revB.pdf \
  --form3_xlsx /path/to/FAIR.xlsx \
  --part_name "my_part" \
  --out_dir ./out
```

Output appears in `out/<part_name>_<timestamp>_<hash>/`.

### Running the Web Application

```bash
# Start the web server (development)
uv run uvicorn shop.app:app --reload --port 8000

# In a separate terminal, start the Huey task worker
uv run huey_consumer.py shop.tasks.huey --workers=1 --worker-type=thread
```

Open http://localhost:8000. The setup wizard will guide you through initial configuration.

### Docker Deployment

The Docker setup runs both the web server and the Huey worker in a single container using supervisord:

```bash
cd docker/
docker compose up --build
```

This starts the application on port 8000 with persistent storage:
- `data/` volume: SQLite databases (app DB + Huey queue), uploaded files
- `out/` volume: Pipeline output directories

Default admin credentials (change immediately):
- Email: `admin@shop.local`
- Password: `changeme`

Override via environment variables `ADMIN_EMAIL` and `ADMIN_DEFAULT_PASSWORD` in `docker-compose.yml`.

---

## Command-Line Reference

### `run.py` (convenience wrapper)

```
uv run python run.py <part_name> [--dpi N] [--out_dir PATH]
```

| Argument | Description |
|----------|-------------|
| `part_name` | Directory name under `assets/` (e.g., `part1`, `part2`) |
| `--dpi` | Image rendering DPI (default: 300) |
| `--out_dir` | Output base directory (default: `./out`) |

### `delta_preservation.cli` (direct invocation)

```
python -m delta_preservation.cli --revA_pdf PATH --revB_pdf PATH --form3_xlsx PATH [options]
```

| Argument | Description |
|----------|-------------|
| `--revA_pdf` | Path to Rev A PDF (required) |
| `--revB_pdf` | Path to Rev B PDF (required) |
| `--form3_xlsx` | Path to Form 3 Excel file (required) |
| `--out_dir` | Output directory (default: `./out`) |
| `--dpi` | Rendering DPI (default: 300) |
| `--part_name` | Part identifier for naming the output directory (default: `part`) |

---

## Repository Structure

```
delta-preservation/
├── run.py                            # CLI convenience entry point
├── run_web.py                        # Web app entry point (seeds admin, starts uvicorn)
├── pyproject.toml                    # Python project config and dependencies
├── package.json                      # Node.js config (Tailwind CSS build)
│
├── delta_preservation/               # Core pipeline package
│   ├── cli.py                        # Pipeline orchestration (8 stages)
│   ├── types.py                      # Pydantic models (DeltaPacket, DeltaItem, Evidence)
│   ├── config.py                     # Default parameters and thresholds
│   ├── io/
│   │   ├── pdf.py                    # PDF rendering and text extraction (PyMuPDF)
│   │   └── xlsx.py                   # AS9102 Form 3 Excel parsing (OpenPyXL)
│   ├── vision/
│   │   ├── balloons.py               # Balloon detection (PDF text + OpenCV Hough circles)
│   │   ├── alignment.py              # Image alignment (ORB + RANSAC homography, text-span fallback)
│   │   ├── snippets.py               # Evidence image cropping and saving
│   │   └── bbox_utils.py             # Bounding box manipulation utilities
│   └── reconcile/
│       ├── anchors.py                # Anchor building (Form 3 → Rev A spatial links)
│       ├── match.py                  # Candidate generation and bipartite assignment
│       ├── classify.py               # Delta classification logic
│       ├── normalize.py              # Requirement text parsing and fingerprinting
│       └── tolerance_pdf.py          # PDF-level tolerance extraction and comparison
│
├── shop/                             # Web application (FastAPI + Jinja2)
│   ├── app.py                        # FastAPI app factory and configuration
│   ├── database.py                   # SQLAlchemy engine and session setup
│   ├── models.py                     # Database models (User, Run, ReviewItem, etc.)
│   ├── dependencies.py               # FastAPI dependency injection (auth, DB session)
│   ├── tasks.py                      # Huey async task queue (pipeline execution, cleanup cron)
│   ├── middleware/
│   │   └── setup_guard.py            # Redirects to setup wizard until configured
│   ├── routers/
│   │   ├── auth.py                   # Login/logout routes
│   │   ├── setup.py                  # First-run setup wizard (4 steps)
│   │   ├── runs.py                   # Run submission, status, SSE progress
│   │   ├── review.py                 # Review queue, approve/override/reset, sign-off
│   │   ├── admin.py                  # User management, settings
│   │   └── exports.py                # CSV/PDF audit packets and work orders
│   ├── services/
│   │   ├── auth.py                   # Password hashing (bcrypt), session management
│   │   ├── runs.py                   # File upload handling, PDF/Excel validation
│   │   ├── review.py                 # Review queue creation, sign-off logic
│   │   ├── exports.py                # Audit packet and work order generation (WeasyPrint)
│   │   ├── amendments.py             # Amendment cloning from signed-off runs
│   │   └── form3.py                  # Excel preview parsing for setup wizard
│   └── templates/                    # Jinja2 HTML templates (HTMX-powered)
│       ├── base.html                 # Main layout with navigation
│       ├── dashboard.html            # Landing page / run list
│       ├── auth/                     # Login page
│       ├── setup/                    # Setup wizard steps
│       ├── runs/                     # Run submission, status, SSE progress
│       ├── review/                   # Review queue, item cards, sign-off
│       ├── admin/                    # User management, settings
│       └── exports/                  # Audit packet and work order PDF templates
│
├── docker/                           # Docker deployment
│   ├── Dockerfile                    # Multi-stage build (Tailwind → Python deps → runtime)
│   ├── docker-compose.yml            # Single-container with volume mounts
│   └── supervisord.conf              # Process manager (uvicorn + huey_consumer)
│
├── static/                           # Frontend assets
│   ├── dist/output.css               # Compiled Tailwind CSS (DaisyUI components)
│   ├── js/htmx.min.js               # HTMX for HTML-over-the-wire interactivity
│   ├── js/htmx-sse.js               # HTMX SSE extension for real-time progress
│   └── src/input.css                 # Tailwind source CSS
│
├── tests/                            # Test suite (pytest)
│   ├── conftest.py                   # Shared fixtures (in-memory DB, test client)
│   ├── test_auth.py                  # Authentication and session tests
│   ├── test_setup.py                 # Setup wizard tests
│   ├── test_runs.py                  # Run submission and lifecycle tests
│   ├── test_runs_service.py          # Run service layer tests
│   ├── test_pipeline_task.py         # Huey task and stage callback tests
│   ├── test_review.py               # Review queue, approve/override/sign-off tests
│   ├── test_exports.py              # Audit packet and work order export tests
│   ├── test_amendments.py           # Amendment creation and versioning tests
│   ├── test_admin.py                # Admin routes and user management tests
│   ├── test_rbac.py                 # Role-based access control tests
│   └── ...
│
└── assets/                           # Test fixtures and sample data
    ├── part1/                        # Stable layout test case
    │   ├── revA.pdf
    │   ├── revB.pdf
    │   └── FAIR.xlsx
    └── part2/                        # Major layout shift test case
        ├── revA.pdf
        ├── revB.pdf
        └── FAIR.xlsx
```

---

## Technical Architecture

### Core Pipeline (`delta_preservation/`)

The pipeline is a pure Python library with no web dependencies. It can be used standalone via the CLI or embedded in other applications via `run_pipeline()`.

**Key dependencies:**
- **PyMuPDF (fitz)**: PDF rendering to images and structured text extraction with bounding boxes
- **OpenCV**: ORB feature detection, Hough circle transform for balloon detection, homography estimation with RANSAC
- **NumPy**: Matrix operations for coordinate transforms
- **OpenPyXL**: Excel file parsing for AS9102 Form 3
- **RapidFuzz**: Fuzzy string matching (used in text comparison)
- **Pydantic**: Validated data models for the output delta packet

**Coordinate systems:**
- PDF space: Points at 72 DPI, origin at top-left
- Image space: Pixels at configurable DPI (default 300), origin at top-left
- Conversion: `scale = dpi / 72.0`

### Web Application (`shop/`)

The web app follows a service-layer architecture:

- **Routers** handle HTTP requests and template rendering
- **Services** contain business logic (validation, review workflows, export generation)
- **Models** define the SQLAlchemy ORM schema
- **Tasks** run the pipeline asynchronously via Huey

**Frontend stack:**
- **HTMX**: HTML-over-the-wire — server returns HTML fragments, not JSON. Enables interactive UIs without a JavaScript framework.
- **HTMX-SSE**: Server-Sent Events for real-time pipeline progress
- **Tailwind CSS + DaisyUI**: Utility-first CSS with pre-built components
- **WeasyPrint**: Server-side PDF rendering for audit packets and work orders

**Database:** SQLite via SQLAlchemy ORM. Tables: `users`, `sessions`, `shop_config`, `runs`, `run_alerts`, `review_items`.

**Authentication:** Username/password with bcrypt hashing. Sessions are stored in the database with expiration timestamps. Two roles: `admin` (full access) and `engineer` (run submission, review, sign-off).

**Task queue:** Huey with SQLite storage. One worker thread processes pipeline runs sequentially. A daily cron job (`cleanup_old_runs`) deletes old runs beyond the configured retention period.

### Key Algorithms

**Hybrid alignment strategy**: ORB feature matching works well for general image alignment, but engineering drawings have large identical title blocks that dominate the feature pool and can produce a spurious near-identity transform. The text-span alignment method matches identical annotation strings between revisions (filtered to dimension-like content only) and computes median displacement. When ORB says "nothing moved" but text-span analysis says "content shifted 300+ points", the text-based transform is used instead.

**Multi-component scoring**: Each candidate match is scored on location (distance from predicted position), text (numeric value overlap with emphasis on the primary dimension value), and context (Jaccard similarity of neighboring text spans). Primary dimension mismatches incur strong penalties, preventing "Ø6.0" from matching "Ø8.0" even when they're in the same location.

**Greedy bipartite assignment**: All candidate edges are sorted by score and greedily assigned, ensuring no characteristic maps to two spans and no span maps to two characteristics. A shared-span fallback handles combined annotations (like "10 x 90°") that encode multiple characteristics in a single text span.

**Tolerance comparison**: The system extracts tolerance values directly from PDF text spans (not from the Form 3 requirement text), handling both bilateral notation (±0.2) and stacked limits notation (.160 / .140). This catches cases where the nominal dimension is unchanged but the tolerance was tightened or loosened.

---

## Background on AS9102 and FAIRs

**AS9102** is the aerospace standard published by SAE International that defines requirements for First Article Inspection (FAI). It applies across aviation, space, and defense industries.

A **FAIR** (First Article Inspection Report) is the documented evidence that a manufacturer's production processes can produce parts conforming to engineering requirements. It consists of three forms:

- **Form 1**: Part number accountability — identifies the part, revision, manufacturer, and whether it's a full or partial FAI
- **Form 2**: Product accountability — materials, special processes, and functional testing
- **Form 3**: Characteristic accountability — every design characteristic with its requirement, measured result, pass/fail status, and any nonconformance references

**Form 3 is the core of the FAIR** and what this system works with. Each row represents one design characteristic (a dimension, tolerance, surface finish, material property, etc.) identified by a balloon number on the engineering drawing.

**Partial FAI** (AS9102 Section 4.6): When a drawing is revised, the standard allows a partial FAI that only re-inspects characteristics affected by the change, provided all other characteristics were conforming on the previous FAI. This is what Delta Preservation enables — identifying exactly which characteristics changed, so only those need re-measurement.

**Key terms:**
- **Balloon**: A circled number on a drawing that uniquely identifies a design characteristic
- **Design characteristic**: Any measurable feature (dimension, tolerance, surface finish, material property) specified on the engineering drawing
- **Variable data**: Actual measured values (required when the drawing specifies numerical limits)
- **Attribute data**: Pass/fail results (allowed when no numerical measurement technique exists)

---

## Limitations

- **Single-page analysis**: Currently processes one page per PDF. Multi-page drawings require selecting which page to analyze.
- **Vector PDFs preferred**: The system works best with vector PDFs where text is extractable. Scanned/raster PDFs are detected and flagged but cannot be processed.
- **English-only text parsing**: Engineering symbols (Ø, R, ±) and numeric patterns are language-agnostic, but keyword matching for pattern classification (e.g., "NOTES", "THREAD") assumes English.
- **Balloon format assumptions**: Expects circled integer numbers 1–200 as characteristic markers. Non-standard balloon formats may not be detected.
- **Title block interference**: The alignment algorithm has special handling for title blocks, but very unusual title block layouts may affect alignment quality.

---

## Support

For questions, issues, or contributions, please email james@handymechanics.com.
