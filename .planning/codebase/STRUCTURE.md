# Codebase Structure

**Analysis Date:** 2026-04-09

## Directory Layout

```
delta-preservation/
├── delta_preservation/          # Core reconciliation engine (pipeline-agnostic)
│   ├── __init__.py
│   ├── cli.py                   # 8-stage pipeline orchestration & entry point
│   ├── types.py                 # Pydantic models (DeltaItem, DeltaPacket, SemanticCallout)
│   ├── config.py                # Default parameters & file validation constants
│   ├── io/                       # Input/output handling
│   │   ├── pdf.py               # PDF rendering, text extraction, coordinate mapping
│   │   └── xlsx.py              # Form 3 (AS9102) Excel parsing
│   ├── vision/                   # Computer vision for drawing analysis
│   │   ├── balloons.py          # Characteristic callout detection (PDF text + CV)
│   │   ├── alignment.py         # Revision homography estimation
│   │   ├── grid.py              # Grid/zone inference from drawing
│   │   ├── bbox_utils.py        # Spatial bounding box utilities
│   │   └── snippets.py          # Evidence snippet extraction & saving
│   └── reconcile/               # Characteristic matching & classification
│       ├── anchors.py           # Form 3 → revA spatial location mapping
│       ├── match.py             # Candidate generation & matching assignment
│       ├── classify.py          # Delta classification (unchanged/changed/removed/added)
│       ├── normalize.py         # Requirement text parsing & normalization
│       ├── semantic_compare.py  # Semantic callout comparison (GD&T, weld, etc.)
│       └── tolerance_pdf.py     # Tolerance extraction & PDF debug reporting
│
├── shop/                         # Web application tier (FastAPI + review workflow)
│   ├── __init__.py
│   ├── app.py                   # FastAPI app factory, template filters
│   ├── database.py              # SQLAlchemy engine & session factory
│   ├── models.py                # ORM models (User, Run, ReviewItem, etc.)
│   ├── dependencies.py          # Dependency injection (get_db, get_current_user)
│   ├── utils.py                 # Utilities (utcnow)
│   ├── tasks.py                 # Huey job queue task definitions
│   ├── middleware/
│   │   └── setup_guard.py       # Middleware enforcing onboarding completion
│   ├── routers/                 # HTTP endpoint handlers
│   │   ├── auth.py              # Login, logout, session management
│   │   ├── setup.py             # Onboarding wizard (admin user, column mapping)
│   │   ├── runs.py              # Run submission, list, SSE progress streaming
│   │   ├── review.py            # Review queue, verdict submission, sign-off
│   │   ├── exports.py           # Form 3 export, PDF report generation
│   │   └── admin.py             # Admin dashboard, user management, config
│   ├── services/                # Business logic modules
│   │   ├── auth.py              # Password hashing, session validation
│   │   ├── runs.py              # Run creation, file upload handling
│   │   ├── form3.py             # Form 3 parsing for preview, export generation
│   │   ├── review.py            # Review queue state, verdict management
│   │   ├── exports.py           # Output file generation (XLSX, PDF)
│   │   ├── amendments.py        # Run amendment/resubmission handling
│   │   └── semantics.py         # Semantic callout formatting for display
│   └── templates/               # Jinja2 HTML templates
│       ├── base.html            # Base layout with navbar, session, alerts
│       ├── dashboard.html       # User dashboard
│       ├── auth/                # Login, logout templates
│       ├── setup/               # Onboarding wizard templates
│       ├── runs/                # Run list, detail, upload templates
│       ├── review/              # Review queue & characteristic detail templates
│       ├── exports/             # Export preview templates
│       └── admin/               # Admin tools templates
│
├── static/                       # Frontend assets (CSS, JS)
│   ├── src/
│   │   └── input.css            # Tailwind input (compiled by npm build:css)
│   ├── dist/
│   │   └── output.css           # Compiled Tailwind output
│   └── js/                       # Client-side JavaScript
│
├── tests/                        # Pytest test suite
│   ├── test_*.py                # Unit & integration tests
│   ├── conftest.py              # Pytest fixtures (if present)
│   └── __init__.py
│
├── alembic/                      # SQLAlchemy schema migrations
│   ├── versions/                 # Migration scripts
│   ├── env.py                    # Alembic environment setup
│   └── script.py.mako            # Migration template
│
├── assets/                       # Test fixtures (engineering drawings)
│   ├── part1/                    # revA.pdf, revB.pdf, FAIR.xlsx
│   ├── part2/
│   ├── part3/
│   ├── part4/
│   ├── part5_NIST_CTC1/
│   ├── part6_NIST_CTC3/
│   └── part7_NIST_CTC4/
│
├── data/                         # Runtime data directory
│   ├── uploads/                  # Uploaded PDFs & Excel files (by run UUID)
│   ├── out/                      # Pipeline output artifacts (deltas, snippets)
│   └── huey.db                   # Huey job queue database (SQLite)
│
├── out/                          # CLI pipeline output (same structure as data/out)
│   └── {part_name}/              # Results for each run
│       ├── delta.json            # DeltaPacket (all classification results)
│       ├── snippets/             # Evidence images (revA_*.png, revB_*.png)
│       └── tolerance_debug.json  # Tolerance comparison metadata
│
├── docker/                       # Docker build artifacts
│   └── Dockerfile
│
├── run.py                        # CLI entry point (standalone mode)
├── run_web.py                    # Web server startup (Docker entry point)
├── pyproject.toml                # Python project metadata & dependencies
├── package.json                  # NPM config for Tailwind CSS build
├── alembic.ini                   # Alembic configuration
├── README.md                     # Project documentation
└── .gitignore                    # Git exclusions
```

## Directory Purposes

**delta_preservation/:**
- Purpose: Standalone pipeline for revision reconciliation
- Contains: Core business logic independent of web framework
- Key files: `cli.py` (orchestration), `types.py` (data contracts)
- Generated outputs: None (stateless library)

**delta_preservation/io/:**
- Purpose: File I/O abstraction for PDFs and Excel
- Contains: PDF text/image extraction, Form 3 parsing
- Key files: `pdf.py` (coordinate system integration), `xlsx.py` (Form 3 schema)
- Dependencies: PyMuPDF, openpyxl

**delta_preservation/vision/:**
- Purpose: Computer vision for engineering drawing analysis
- Contains: Balloon detection, alignment, spatial utilities
- Key files: `balloons.py` (detection algorithms), `alignment.py` (homography)
- Dependencies: OpenCV, NumPy

**delta_preservation/reconcile/:**
- Purpose: Characteristic identity matching and classification
- Contains: Matching algorithms, semantic parsing, tolerance comparison
- Key files: `match.py` (bipartite matching), `classify.py` (decision logic)
- Dependencies: Reconcile types, vision utilities

**shop/:**
- Purpose: Multi-user web application with review/sign-off workflow
- Contains: HTTP handlers, ORM models, business services
- Stateful: Persists to database and filesystem
- Key files: `app.py` (FastAPI setup), `models.py` (database schema)

**shop/routers/:**
- Purpose: HTTP request handling and response rendering
- Pattern: One router per feature area (auth, runs, review, etc.)
- Each router: Handles GET/POST, renders templates, calls services
- No business logic: Routes → Services → Models

**shop/services/:**
- Purpose: Business logic encapsulation, database queries, file operations
- Pattern: One service per domain (runs, review, exports, auth)
- Each service: Stateless functions that operate on models
- No HTTP handling: Services are router-agnostic

**shop/models.py:**
- Purpose: SQLAlchemy ORM schema (single file, ~127 lines)
- Contains: User, UserSession, Run, ReviewItem, RunAlert, ShopConfig, Characteristic
- Pattern: Each class = one table; relationships encoded as Mapped annotations
- No business logic: Pure data model

**shop/templates/:**
- Purpose: Server-side Jinja2 HTML rendering
- Pattern: Base layout + feature-specific subdirectories
- Filters: Custom filters defined in app.py (status_badge_class, etc.)
- Static: Links to compiled CSS (static/dist/output.css)

**static/:**
- Purpose: Frontend assets (CSS, minimal JS)
- CSS: Tailwind + DaisyUI (config in tailwindcss.config.js, compiled by npm)
- JS: Minimal client-side scripts (form validation, SSE streaming)
- Built: `npm run build:css` generates dist/output.css

**tests/:**
- Purpose: Pytest test suite
- Organization: test_*.py files, one per module/feature
- Patterns: Fixtures in conftest.py, mocks of FastAPI dependencies
- Run: `pytest` or `pytest -v` from repo root

**alembic/:**
- Purpose: SQLAlchemy schema version control
- Versions/: Timestamped migration scripts (e.g., 001_initial_schema.py)
- Workflow: `alembic revision --autogenerate -m "message"` then `alembic upgrade head`

**assets/:**
- Purpose: Test fixtures (engineering drawings for part1-7)
- Structure: Each part/{revA.pdf, revB.pdf, FAIR.xlsx, ground_truth.json}
- Used by: CLI tests and demo; can be run via `python run.py part1`

## Key File Locations

**Entry Points:**
- `run.py`: CLI standalone mode (resolves assets directory, calls cli.py)
- `run_web.py`: Web server startup (seeds DB, runs migrations, starts uvicorn)
- `shop/app.py`: FastAPI app factory (called by uvicorn at startup)
- `delta_preservation/cli.py`: 8-stage pipeline orchestration

**Configuration:**
- `delta_preservation/config.py`: Default parameters (DPI, search radius, thresholds)
- `shop/database.py`: SQLAlchemy engine setup from DATABASE_URL env var
- `alembic.ini`: Alembic config (DB URL, script location)
- `pyproject.toml`: Python dependencies, pytest config
- `package.json`: Tailwind build script

**Core Logic:**
- `delta_preservation/reconcile/classify.py`: Main classification algorithm (1477 lines)
- `delta_preservation/reconcile/match.py`: Candidate matching (1154 lines)
- `shop/services/review.py`: Review workflow state machine
- `shop/services/form3.py`: Form 3 parsing and export generation
- `shop/tasks.py`: Background job definitions (execute_run_task)

**Testing:**
- `tests/test_*.py`: Test modules (pytest discovery)
- `tests/conftest.py`: Shared fixtures (FastAPI TestClient, DB session)
- `assets/part{1-7}/`: Test data (PDFs, Excel)

**Database:**
- `shop/models.py`: ORM schema
- `alembic/versions/`: Migration scripts
- `shop.db`: SQLite default (can use PostgreSQL via DATABASE_URL)

## Naming Conventions

**Files:**
- Python: snake_case (e.g., `classify.py`, `run_web.py`)
- HTML templates: snake_case with .html (e.g., `review_queue.html`)
- CSS: input.css (Tailwind source), output.css (compiled)
- Tests: test_{module_name}.py (e.g., `test_classify_bugfixes.py`)

**Python Modules:**
- Public functions: snake_case (e.g., `run_pipeline()`, `classify_delta()`)
- Classes: PascalCase (e.g., `DeltaItem`, `Anchor`, `User`)
- Private functions: leading underscore (e.g., `_homography_is_near_identity()`)
- Constants: UPPER_SNAKE_CASE (e.g., `DEFAULT_DPI`, `FORM3_HEADER_KEYWORDS`)

**Directories:**
- Python packages: snake_case (delta_preservation, shop)
- Feature areas: snake_case (io, vision, reconcile, routers, services, templates)
- Part fixtures: snake_case with numbers (part1, part5_NIST_CTC1)

**Database Tables:**
- Table names: snake_case plural (users, sessions, runs, review_items)
- Columns: snake_case with underscores (char_no, reference_location, hashed_password)
- Foreign keys: {table_name}_id (e.g., user_id, run_id, reviewer_id)

**URLs/Routes:**
- snake_case paths (e.g., `/review/{run_id}`, `/runs/{run_id}/detail`)
- POST form endpoints: same path as GET (e.g., POST /runs for submit)

## Module Organization

**delta_preservation → shop dependency flow:**

```
delta_preservation (core pipeline)
    └─ used by shop/tasks.py (via execute_run_task)
    └─ used by shop/services/review.py (loads DeltaPacket from disk)
    └─ used by shop/services/semantics.py (formats SemanticCallout for display)

shop (web tier)
    ├─ shop/routers/*.py → shop/services/*.py
    ├─ shop/services/*.py → shop/models.py (SQLAlchemy queries)
    ├─ shop/routers/*.py → shop/app.py (templates, dependencies)
    └─ shop/tasks.py → delta_preservation.cli (pipeline execution)
```

**No circular imports:** delta_preservation does not depend on shop.

**Dependency injection in shop:**
- `get_db()`: Returns SQLAlchemy Session from SessionLocal
- `get_current_user()`: Validates session token, returns User
- Used in route handlers via `Depends()`

## Where to Add New Code

**New Pipeline Feature (e.g., new matching algorithm):**
- Primary code: `delta_preservation/reconcile/new_feature.py`
- Export in: `delta_preservation/reconcile/__init__.py`
- Integration: Call from `delta_preservation/cli.py` stage handler
- Tests: `tests/test_new_feature.py`

**New Web Route (e.g., new admin tool):**
- Implementation: `shop/routers/new_feature.py`
- Register in: `shop/app.py:create_app()` via `app.include_router()`
- Service layer: `shop/services/new_feature.py`
- Templates: `shop/templates/new_feature/`
- Tests: `tests/test_new_feature.py` (FastAPI TestClient)

**New Web Service:**
- Implementation: `shop/services/new_feature.py`
- Consumed by: `shop/routers/` that import from services
- Database: Add models to `shop/models.py` if new tables needed
- Migrations: `alembic revision --autogenerate -m "message"`

**New Vision Algorithm:**
- Implementation: `delta_preservation/vision/new_algorithm.py`
- Export in: `delta_preservation/vision/__init__.py`
- Integration: Call from `delta_preservation/cli.py` stage handler
- Tests: `tests/test_new_algorithm.py` (use assets for fixture images)

**Utilities/Helpers:**
- Shared Python utilities: `delta_preservation/` subfolder or as new module in existing folder
- Web utilities: `shop/utils.py` (currently minimal)
- Template filters: Define in `shop/app.py`, apply to `templates.env.filters`

## Special Directories

**data/uploads/:**
- Purpose: Temporary storage of uploaded PDFs and Excel files
- Structure: `{run_uuid}/{filename}` (one dir per Run)
- Generated: By `shop/services/runs.py:save_upload()`
- Cleanup: Subject to ShopConfig.retention_days (default 30 days)
- Committed: No (in .gitignore)

**data/out/ and out/:**
- Purpose: Pipeline output artifacts (deltas, snippets, reports)
- Structure: `{part_name}/` (CLI mode) or `{run_id}/` (web mode)
- Generated: By `delta_preservation/cli.py:run_pipeline()`
- Cleanup: Manual or periodic (not auto-purged)
- Committed: No (in .gitignore)

**.planning/codebase/:**
- Purpose: Architecture and structure documentation
- Generated: By /gsd-map-codebase orchestrator
- Files: ARCHITECTURE.md, STRUCTURE.md (this file), STACK.md, INTEGRATIONS.md (if tech focus), CONVENTIONS.md, TESTING.md, CONCERNS.md (if quality/concerns focus)
- Committed: No (in .gitignore per PR #2)

**alembic/versions/:**
- Purpose: Schema migration history
- Generated: By `alembic revision --autogenerate`
- Pattern: {timestamp}_{message}.py (e.g., 001_initial_schema.py)
- Committed: Yes (part of version control)

---

*Structure analysis: 2026-04-09*
