# Architecture

**Analysis Date:** 2026-04-09

## Pattern Overview

**Overall:** Dual-layer hybrid architecture combining a standalone **reconciliation engine** (CLI-driven data processing pipeline) and a **web application tier** (FastAPI-based review/sign-off system).

**Key Characteristics:**
- **Separation of concerns**: Core reconciliation logic is independent of web framework
- **Async task processing**: Web tier delegates long-running reconciliation to background job queue (Huey)
- **Database-driven state**: Web tier maintains run lifecycle, user sessions, review state in SQLAlchemy ORM
- **Evidence-rich output**: Pipeline generates structured deltas with visual snippets and confidence scores for human review
- **Event streaming**: Web tier uses Server-Sent Events (SSE) to stream real-time pipeline progress to UI

## Layers

**Core Reconciliation Engine:**
- Purpose: Autonomous revision comparison and characteristic classification
- Location: `delta_preservation/` package
- Contains: 8-stage pipeline orchestrated by CLI
- Depends on: PDF/Excel I/O, vision utilities, semantic parsing
- Used by: CLI entry point (`run.py`), web tier via task queue

**I/O Layer:**
- Purpose: Extract and normalize structured data from PDFs and Excel
- Location: `delta_preservation/io/` (pdf.py, xlsx.py)
- Contains: PDF rendering, text extraction with coordinate mapping, Form 3 parsing
- Depends on: PyMuPDF (fitz), openpyxl
- Used by: CLI pipeline stages, vision module for coordinate transformation

**Vision Layer:**
- Purpose: Computer vision analysis for balloon detection and spatial matching
- Location: `delta_preservation/vision/` (balloons.py, alignment.py, bbox_utils.py, snippets.py, grid.py)
- Contains: Balloon detection, image alignment (homography), bounding box utilities, evidence snippet generation
- Depends on: OpenCV, NumPy, PyMuPDF for coordinate systems
- Used by: CLI stages 2-7 (Balloon detection through Classification)

**Reconciliation Layer:**
- Purpose: Characteristic identity matching and delta classification
- Location: `delta_preservation/reconcile/` (anchors.py, match.py, classify.py, normalize.py, tolerance_pdf.py, semantic_compare.py)
- Contains: Form 3 anchor building, candidate matching, classification logic, semantic parsing
- Depends on: Reconciliation types, vision layer for spatial matching
- Used by: CLI stage 6-7 and web tier for re-analysis

**Web Application Tier:**
- Purpose: Multi-user review/approval workflow with persistence
- Location: `shop/` package
- Contains: FastAPI app, Jinja2 templates, SQLAlchemy ORM models, routers, services
- Depends on: Reconciliation engine (dynamically imported), SQLAlchemy, Huey job queue
- Layers:
  - **Routers** (`shop/routers/`): HTTP request handlers
  - **Services** (`shop/services/`): Business logic (run lifecycle, review workflow, exports)
  - **Models** (`shop/models.py`): SQLAlchemy ORM (User, Run, ReviewItem, etc.)
  - **Middleware** (`shop/middleware/`): SetupGuardMiddleware enforces onboarding flow
  - **Templates** (`shop/templates/`): Server-side Jinja2 HTML rendering

**Background Task Layer:**
- Purpose: Async job execution for long-running pipeline runs
- Location: `shop/tasks.py` with Huey (SqliteHuey)
- Contains: Task definitions for pipeline execution, result persistence
- Depends on: Core reconciliation engine
- Used by: Runs router to queue jobs

## Data Flow

**CLI Mode (Standalone):**

1. **Input**: `run.py` parses args (part_name, revA PDF, revB PDF, Form 3 XLSX)
2. **Pipeline** (8 stages orchestrated by `delta_preservation/cli.py:run_pipeline()`):
   - Stage 1: Form 3 parsing → List of Characteristics (char_no, requirement, etc.)
   - Stage 2: Balloon detection → Balloons with coordinates in revA PDF
   - Stage 3: Text extraction → TextSpans with spatial coordinates from both PDFs
   - Stage 4: Anchor building → Map each characteristic to revA location (Anchor)
   - Stage 5: Alignment → Compute homography from revA to revB
   - Stage 6: Candidate matching → For each revA Anchor, find revB TextSpan candidates
   - Stage 7: Classification → Classify each match as unchanged/changed/removed
   - Stage 8: Output → Write DeltaPacket (JSON) + evidence snippets + tolerance report
3. **Output**: `out/{part_name}/` directory with:
   - `delta.json` (DeltaPacket with all classification results)
   - `snippets/` (image evidence for each characteristic)
   - `tolerance_debug.json` (tolerance comparison details)

**Web Mode (Review/Sign-Off):**

1. **Setup Flow**:
   - Admin creates ShopConfig via onboarding wizard at `/setup`
   - Form 3 column mapping configured (char_no, reference_location, requirement, designator)
   - Admin created via `run_web.py` seed step

2. **Run Submission**:
   - Engineer uploads revA PDF, revB PDF, Form 3 XLSX at `/runs` → POST
   - Metadata validation (PDF integrity, page counts)
   - Run record created with status="queued"
   - Job queued to Huey task system

3. **Run Execution** (background task `execute_run_task`):
   - Huey worker invokes core pipeline with uploaded files
   - Progress reported via Run.current_stage updates
   - Completion → Run.status = "completed" or "failed"
   - ReviewItems created (one per DeltaItem) with initial status

4. **Review & Sign-Off**:
   - Engineer navigates to `/review/{run_id}`
   - Loads ReviewItems for the run (fetches DeltaPacket from disk)
   - Can filter by status (all/uncertain/changed/added/removed)
   - For each item: view evidence snippets, semantic callout, confidence scores
   - Submit verdict (approved/rejected/needs_review) → ReviewItem.verdict updated
   - Admin can sign off entire run → Run.status="signed_off"

5. **Exports**:
   - Generate Form 3 (XLSX) with delta classifications filled in
   - Generate PDF report with decision narrative

**State Management:**

- **Pipeline State**: Transient per-run JSON structures (Anchor, Match, DeltaItem) in memory
- **Web State**: Persistent in SQLAlchemy DB (shop.db or PostgreSQL via DATABASE_URL):
  - User sessions (UserSession)
  - Run metadata + current status (Run)
  - Review verdicts per characteristic (ReviewItem)
  - Run alerts for failures (RunAlert)
  - Column mapping for Form 3 parsing (ShopConfig)
- **File State**: Pipeline outputs persisted to disk:
  - Uploaded PDFs/Excel in UPLOADS_DIR
  - Output artifacts in configured out_dir
  - Evidence snippets in run-specific subdirectories

## Key Abstractions

**Anchor (reconcile/anchors.py):**
- Purpose: Bridges Form 3 characteristics to spatial locations in revA PDF
- Attributes: char_no, requirement, visual evidence (balloon position, text spans)
- Pattern: Built from Form 3 + vision detections; input to matching algorithm

**Match (reconcile/match.py):**
- Purpose: Represents a candidate or confirmed pairing between revA and revB characteristics
- Attributes: anchor, candidate span, match score, component scores (location, text, context)
- Pattern: Assigned by `assign_matches()` algorithm; input to classification

**DeltaItem (reconcile/classify.py for internal, types.py for output):**
- Purpose: Encapsulates final classification result (unchanged/changed/removed/added) with confidence
- Attributes: char_no, status, confidence, reasons, component_scores, revA/revB Evidence
- Pattern: Output by `classify_delta()`; serialized as DeltaPacket for web/export

**DeltaPacket (types.py):**
- Purpose: Top-level output container for complete delta analysis
- Attributes: run_id, inputs (file paths), items (list of DeltaItems)
- Pattern: JSON serializable; consumed by review UI, exports, auditing

**Characteristic (io/xlsx.py):**
- Purpose: Represents a single row from Form 3
- Attributes: char_no, reference_location, requirement, designator
- Pattern: Parsed from XLSX; input to anchor building

**TextSpan (io/pdf.py):**
- Purpose: Represents extracted text with spatial coordinates
- Attributes: text, page, bbox (PDF points), confidence
- Pattern: Extracted from PDF text layer; matched across revisions

**Balloon (vision/balloons.py):**
- Purpose: Detected characteristic callout symbol on drawing
- Attributes: center, radius, number, page, method (pdf_text or cv2_detection)
- Pattern: Detected by vision module; helps locate anchors

**SemanticCallout (types.py):**
- Purpose: Typed semantic representation of characteristic requirement (GD&T, weld, surface finish, fit)
- Attributes: provenance, status, raw_text, normalized_text, gdt/weld/surface_finish/fit payloads
- Pattern: Extracted and compared across revisions; enriches classification decisions

## Entry Points

**CLI Entry Point (run.py):**
- Location: `/home/khoa2/delta-preservation/run.py`
- Triggers: `python run.py <part_name>`
- Responsibilities:
  - Resolve asset directory from part_name
  - Validate file existence
  - Call `delta_preservation.cli.run_pipeline()`
  - Outputs to `./out/{part_name}/`

**Web Entry Point (run_web.py):**
- Location: `/home/khoa2/delta-preservation/run_web.py`
- Triggers: Docker startup or `uvicorn shop.app:app`
- Responsibilities:
  - Seed default admin user from env vars
  - Run Alembic migrations
  - Create ShopConfig singleton
  - Start uvicorn server on 0.0.0.0:8000

**FastAPI App Creation (shop/app.py:create_app):**
- Location: `shop/app.py`
- Triggers: Module import (`from shop.app import app`)
- Responsibilities:
  - Wire template filters for Jinja2
  - Add SetupGuardMiddleware (enforces setup completion)
  - Mount static files
  - Register routers (auth, admin, setup, runs, review, exports)

**Pipeline CLI Orchestration (delta_preservation/cli.py:run_pipeline):**
- Location: `delta_preservation/cli.py`
- Triggers: Called by run.py or `execute_run_task` (web task)
- Responsibilities:
  - Execute 8-stage pipeline
  - Manage file I/O and progress reporting
  - Produce DeltaPacket and evidence snippets
  - Returns output directory path

## Error Handling

**Strategy:** Layered validation with graceful degradation.

**Pipeline Errors:**
- Stage failures recorded in Run.failure_stage + Run.failure_message
- Run.status set to "failed"; RunAlert created for user
- Evidence partial: stages 1-5 may succeed even if 6+ fail

**Web Validation:**
- PDF integrity check before Run creation (`validate_pdf_bytes()`)
- Form 3 parsing errors trigger Run.status="failed" + helpful alert
- Database integrity via SQLAlchemy constraints (ForeignKey, unique)

**Review Errors:**
- ReviewItem verdict validation in `validate_debug_verdict_payload()`
- Sign-off blocked if uncertain items remain (configurable per org policy)

**Patterns:**
- Try-except at stage boundaries in CLI
- HTTPException(status_code=400) for user input validation in routers
- Structured error messages in RunAlert for admin/engineer visibility

## Cross-Cutting Concerns

**Logging:** 
- Approach: Python logging module with basicConfig in run_web.py + CLI
- Level: DEBUG in dev, INFO in production
- Used for: Pipeline progress, task queue events, HTTP request tracing

**Validation:** 
- Approach: Pydantic BaseModel for types.py (DeltaItem, DeltaPacket, SemanticCallout)
- SQLAlchemy ORM for web models with column constraints
- File validation functions in shop/services/runs.py (PDF, Excel integrity)
- Custom validators in reconcile modules (tolerance parsing, semantic extraction)

**Authentication:** 
- Approach: Custom session-based auth in shop/services/auth.py
- Bcrypt password hashing (pwdlib)
- UserSession table with token + expiry
- Middleware/dependency: `get_current_user()` in shop/dependencies.py

**Permissions:**
- Admin: Access /admin, /setup, edit ShopConfig, sign-off runs
- Engineer: Submit runs, review own/shared runs
- Enforced via role check in routers + middleware

---

*Architecture analysis: 2026-04-09*
