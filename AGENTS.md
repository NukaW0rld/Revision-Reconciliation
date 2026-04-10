<!-- GSD:project-start source:PROJECT.md -->
## Project

**Delta Preservation**

Delta Preservation is a brownfield aerospace drawing-reconciliation system with two validated surfaces: a standalone pipeline that compares Rev A and Rev B drawings against AS9102 Form 3 data, and a web workflow for review, sign-off, and debug inspection. The current project focus is not a new end-user feature, but a faster and more consistent internal debug loop that compares each run against stable part-level ground truth and helps one developer improve the algorithm across many drawing pairs without overfitting to any single part.

**Core Value:** Improve reconciliation accuracy across many parts through a consistent, evidence-rich debug workflow that strengthens the algorithm without teaching it part-specific hacks.

### Constraints

- **Generalization**: Debug decisions must improve behavior across many parts — the system cannot be shaped around one drawing pair at the expense of others
- **Ground truth stability**: `ground_truth.json` files are canonical references and must not be auto-edited by the workflow
- **Human review boundary**: Nonconforming results still require manual inspection because more than one outcome may be acceptable in some cases
- **Developer-only scope**: This milestone is optimized for one maintainer's iterative run -> review -> rerun loop, not a multi-user debug product
- **Snippet tolerance**: Snippet evaluation should favor useful visible context over exact center-coordinate matching
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.10–3.12 - Backend pipeline, web API, database models
- JavaScript/TypeScript - Frontend (minimal, CSS build only)
- SQL - Database schema via SQLAlchemy ORM
- Jinja2 Templates - HTML templating
- CSS/Tailwind - Styling with DaisyUI components
## Runtime
- Python 3.11 (Docker runtime, `/home/khoa2/delta-preservation/docker/Dockerfile` specifies `python:3.11-slim`)
- Node.js 22 (Alpine, used for Tailwind CSS compilation in Docker multi-stage build)
- `uv` (Astral package manager) - Primary dependency resolver (`/home/khoa2/delta-preservation/uv.lock`)
- npm - JavaScript/CSS dependencies (`/home/khoa2/delta-preservation/package.json`)
## Frameworks & Libraries
- FastAPI 0.* - REST API framework with automatic OpenAPI docs
- Uvicorn - ASGI server (runs on port 8000)
- Jinja2 - Server-side template rendering
- SQLAlchemy 2.* - SQL toolkit and ORM (`shop.database.py`, `shop.models.py`)
- Alembic 1.18.4 - Database schema migrations (`/home/khoa2/delta-preservation/alembic/`)
- SQLite - Default file-based database (configurable via `DATABASE_URL` env var)
- Huey (SqliteHuey) - Lightweight task queue backed by SQLite (`shop/tasks.py`)
- Supervisor - Process manager for uvicorn + huey_worker (see `docker/supervisord.conf`)
- PyMuPDF (fitz) - PDF parsing and rendering
- WeasyPrint - HTML-to-PDF generation for audit packet exports
- OpenPyXL - Excel file reading (Form 3 parsing)
- OpenCV (opencv-python-headless) - Image processing, balloon detection, feature matching
- NumPy - Numerical computations
- pwdlib[bcrypt] - Password hashing library
- bcrypt ≥4.0.0,<5.0.0 - Bcrypt hashing algorithm
- itsdangerous - Token generation for session management
- Pydantic ≥2.5,<3.0 - Data validation, model definitions (`delta_preservation/types.py`)
- RapidFuzz - Fuzzy string matching for characteristic reconciliation
- python-multipart - Form data parsing in FastAPI
- Tailwind CSS 4.0.0 - Utility-first CSS framework
- DaisyUI 5.0.0 - Component library built on Tailwind
- pytest ≥8 - Testing framework
- httpx ≥0.28.1 - Async HTTP client for tests
## Configuration
- `DATABASE_URL` - SQLAlchemy connection string (default: `sqlite:///./shop.db`)
- `HUEY_DB` - Path to Huey task queue SQLite database (default: `/app/data/huey.db`)
- `OUT_DIR` - Output directory for pipeline artifacts (default: `/app/out`)
- `UPLOADS_DIR` - File upload storage directory (default: `/app/data/uploads`)
- `ADMIN_EMAIL` - Default admin username (default: `admin@shop.local`)
- `ADMIN_DEFAULT_PASSWORD` - Default admin password (default: `changeme`)
- `pyproject.toml` - Python project metadata, dependencies, test configuration
- `.tailwindrc` - Tailwind CSS configuration (auto-generated or inferred from HTML)
- `tsconfig.json` - Not present (no TypeScript compilation)
- Alembic configuration: `alembic.ini`
- Migrations run automatically on container startup via `docker/Dockerfile` line 46: `alembic upgrade head`
## Build & Bundling
- Tailwind CLI - Compiles `static/src/input.css` → `static/dist/output.css`
- Build command: `npm run build:css` (defined in `package.json`)
- Executed in Docker multi-stage build before Python runtime image is created
- `uv sync --locked` - Installs frozen dependencies from `uv.lock`
- No compiled extensions (relies on pre-built wheels for NumPy, OpenCV, bcrypt)
## Infrastructure
- Docker 3-stage build (`docker/Dockerfile`):
- Supervisor (`docker/supervisord.conf`) manages two concurrent processes:
- Docker Compose (`docker/docker-compose.yml`) - Single service `web` with:
- `supervisor` - Process manager
- `libpango-1.0-0`, `libpangoft2-1.0-0`, `libharfbuzz-subset0` - Required by WeasyPrint for font rendering
## Platform Requirements
- Python 3.10+ locally (with `uv` or `pip` for installation)
- Node.js 22+ for CSS compilation (if building outside Docker)
- SQLite3 (typically pre-installed on Linux/macOS)
- Docker Engine (for containerized deployment)
- Persistent storage for `/app/data` (SQLite DB, Huey queue, uploads)
- Writable `/app/out` directory for pipeline output
- Port 8000 exposed (or reverse proxy)
- Minimum 2 OS threads (one for uvicorn, one for huey_worker)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Language Style
- Indentation: 4 spaces per level (standard PEP 8)
- String quotes: Double quotes (`"`) are primary
- Semicolons: Not used (standard Python)
- Type hints: Required for function signatures and class attributes
- Line length: Appears flexible, generally follows PEP 8 (~100-120 chars)
- Docstrings: Triple-quoted, comprehensive module/class/function docstrings with Args/Returns sections
- Use SQLAlchemy 2.0+ mapping style with `Mapped` type hints
- Example from `shop/models.py`: `id: Mapped[int] = mapped_column(Integer, primary_key=True)`
- Relationships defined with `Mapped[list["ClassName"]]` for collections
## Naming Patterns
- `snake_case.py` for module files (e.g., `bbox_utils.py`, `semantic_compare.py`)
- `test_<module>.py` for test files (e.g., `test_grid.py`, `test_auth.py`)
- `__init__.py` for package markers
- `snake_case` for public functions: `detect_balloons()`, `render_page()`, `extract_text_spans()`
- `_snake_case` prefix for internal/private functions: `_span_key()`, `_dedupe_spans()`, `_normalize_span_text()`
- Single leading underscore indicates module-private scope
- `snake_case` for all variable names (local, module, class attributes)
- `ALL_CAPS_CONST` for module-level constants: `DEFAULT_DPI`, `MIN_ALIGNMENT_INLIERS`, `FORM3_DEFAULT_COLUMNS`
- Abbreviated names avoided except for well-established domain terms: `bbox` (bounding box), `dpi` (dots per inch), `revA`/`revB` (revision A/B)
- `PascalCase` for all class names
- Dataclasses with `@dataclass` decorator: `TextSpan`, `Balloon`, `DeltaItem`, `Candidate`, `Match`
- Pydantic models inherit from `BaseModel`: `DeltaPacket`, `Evidence`, `SemanticCallout`
- Frozen dataclasses used when immutability needed: `@dataclass(frozen=True)` in `GroupedSpan`
- Type hints use standard Python typing: `Optional[T]`, `List[T]`, `Dict[K, V]`, `Tuple[...]`
- Literal types for fixed string values: `Literal["unchanged", "changed", "removed", "added", "uncertain"]`
- Enum classes for method families: `DetectionMethod.PDF_TEXT`, `DetectionMethod.CV`
- Revision references: `revA`, `revB` (standardized across codebase)
- Characteristic number: `char_no` (from AS9102 Form 3)
- Bounding box: `bbox`, `bbox_pdf` (PDF coordinates), `bbox_img` (image coordinates)
- Coordinate tuples: `(x0, y0, x1, y1)` for bboxes, `(x, y)` for points, `(cx, cy)` for centers
## Code Style
- No automatic formatter configured (no `.prettierrc`, `.black`, or `pyproject.toml` [tool.black] section)
- Manual formatting follows PEP 8 with exceptions noted above
- Imports organized in standard Python style: stdlib, third-party, local
- No linter config files detected (no `.eslintrc`, `.flake8`, `pylint.rc`)
- Linting appears to be implicit compliance rather than enforced
- No path aliases configured in `pyproject.toml`
- Absolute imports from package root: `from delta_preservation.types import DeltaItem`
## Error Handling
- Explicit exception raising for validation failures: `raise FileNotFoundError()`, `raise IndexError()`
- Error messages include context: `f"PDF not found: {pdf_path}"`, `f"Page index {page_index} out of range (0-{page_count-1})"`
- No global error handlers; exceptions propagate to caller for handling
- Functions validate inputs at entry points (path existence, index bounds)
- Optional match results: return `None` instead of raising (e.g., `match_or_none: Optional[Match]`)
- Classification uses status strings rather than exceptions: `status: str # "unchanged", "changed", "removed", "added", "uncertain"`
- Pydantic `Field()` used for validation and description: `Field(..., ge=0.0, le=1.0)` for ranges
- Optional fields use `Optional[T]` with `default=None` in Field
## Comments
- Module-level docstrings explain purpose and key concepts: See `delta_preservation/types.py`, `delta_preservation/cli.py`
- Function docstrings describe parameters, return values, raises, and notes
- Inline comments explain non-obvious logic or complex transformations
- Examples from codebase:
- Python docstrings follow Google/NumPy style with Args/Returns sections
- Pydantic Field descriptions serve as inline documentation: `description="Bounding box coordinates [x0, y0, x1, y1] in PDF points"`
## Function Design
- Functions vary in length; longer functions have clear logical sections marked by comments
- Example: `run_pipeline()` in `cli.py` is ~300 lines with 8 marked pipeline stages
- Type hints required on all parameters
- Default values used for configuration: `dpi: int = 300`, `search_radius: float = 144.0`
- Callback parameters accepted: `stage_callback: Optional[Callable[[int, str], None]] = None`
- Explicit return types required
- Multiple returns: use dataclass or tuple
- Example: `Candidate` dataclass contains `span`, `total_score`, `location_score`, `text_score`, `context_score`, `reasons`
- `None` used for optional results
## Module Design
- Public API defined by module-level imports and direct function/class definitions
- No explicit `__all__` lists observed (imports handle visibility)
- Internal functions prefixed with `_` excluded from intended public API
- No barrel export files observed (e.g., no index.ts)
- `__init__.py` files exist but appear minimal/empty
- Imports use direct module paths: `from delta_preservation.io.pdf import TextSpan`
## Pydantic/SQLAlchemy Specifics
- Used for structured data validation and serialization
- Example from `delta_preservation/types.py`: `DeltaPacket` with nested models
- Field validation: `confidence: float = Field(..., ge=0.0, le=1.0, description="...")`
- Modern declarative syntax with `Mapped` type hints
- Foreign key relationships with `ForeignKey` and `relationship()`
- Cascade options used: `cascade="all, delete-orphan"`
- Example from `shop/models.py`: `sessions: Mapped[list["UserSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")`
## Anti-patterns to Avoid
- Ambiguous single-letter variable names (except loop counters in tight scopes)
- Silent failures; exceptions preferred over returning incorrect values
- Circular dependencies between modules
- Mixing camelCase and snake_case in same scope
- Global mutable state (use dependency injection via `SessionLocal` or passed parameters)
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- **Separation of concerns**: Core reconciliation logic is independent of web framework
- **Async task processing**: Web tier delegates long-running reconciliation to background job queue (Huey)
- **Database-driven state**: Web tier maintains run lifecycle, user sessions, review state in SQLAlchemy ORM
- **Evidence-rich output**: Pipeline generates structured deltas with visual snippets and confidence scores for human review
- **Event streaming**: Web tier uses Server-Sent Events (SSE) to stream real-time pipeline progress to UI
## Layers
- Purpose: Autonomous revision comparison and characteristic classification
- Location: `delta_preservation/` package
- Contains: 8-stage pipeline orchestrated by CLI
- Depends on: PDF/Excel I/O, vision utilities, semantic parsing
- Used by: CLI entry point (`run.py`), web tier via task queue
- Purpose: Extract and normalize structured data from PDFs and Excel
- Location: `delta_preservation/io/` (pdf.py, xlsx.py)
- Contains: PDF rendering, text extraction with coordinate mapping, Form 3 parsing
- Depends on: PyMuPDF (fitz), openpyxl
- Used by: CLI pipeline stages, vision module for coordinate transformation
- Purpose: Computer vision analysis for balloon detection and spatial matching
- Location: `delta_preservation/vision/` (balloons.py, alignment.py, bbox_utils.py, snippets.py, grid.py)
- Contains: Balloon detection, image alignment (homography), bounding box utilities, evidence snippet generation
- Depends on: OpenCV, NumPy, PyMuPDF for coordinate systems
- Used by: CLI stages 2-7 (Balloon detection through Classification)
- Purpose: Characteristic identity matching and delta classification
- Location: `delta_preservation/reconcile/` (anchors.py, match.py, classify.py, normalize.py, tolerance_pdf.py, semantic_compare.py)
- Contains: Form 3 anchor building, candidate matching, classification logic, semantic parsing
- Depends on: Reconciliation types, vision layer for spatial matching
- Used by: CLI stage 6-7 and web tier for re-analysis
- Purpose: Multi-user review/approval workflow with persistence
- Location: `shop/` package
- Contains: FastAPI app, Jinja2 templates, SQLAlchemy ORM models, routers, services
- Depends on: Reconciliation engine (dynamically imported), SQLAlchemy, Huey job queue
- Layers:
- Purpose: Async job execution for long-running pipeline runs
- Location: `shop/tasks.py` with Huey (SqliteHuey)
- Contains: Task definitions for pipeline execution, result persistence
- Depends on: Core reconciliation engine
- Used by: Runs router to queue jobs
## Data Flow
- **Pipeline State**: Transient per-run JSON structures (Anchor, Match, DeltaItem) in memory
- **Web State**: Persistent in SQLAlchemy DB (shop.db or PostgreSQL via DATABASE_URL):
- **File State**: Pipeline outputs persisted to disk:
## Key Abstractions
- Purpose: Bridges Form 3 characteristics to spatial locations in revA PDF
- Attributes: char_no, requirement, visual evidence (balloon position, text spans)
- Pattern: Built from Form 3 + vision detections; input to matching algorithm
- Purpose: Represents a candidate or confirmed pairing between revA and revB characteristics
- Attributes: anchor, candidate span, match score, component scores (location, text, context)
- Pattern: Assigned by `assign_matches()` algorithm; input to classification
- Purpose: Encapsulates final classification result (unchanged/changed/removed/added) with confidence
- Attributes: char_no, status, confidence, reasons, component_scores, revA/revB Evidence
- Pattern: Output by `classify_delta()`; serialized as DeltaPacket for web/export
- Purpose: Top-level output container for complete delta analysis
- Attributes: run_id, inputs (file paths), items (list of DeltaItems)
- Pattern: JSON serializable; consumed by review UI, exports, auditing
- Purpose: Represents a single row from Form 3
- Attributes: char_no, reference_location, requirement, designator
- Pattern: Parsed from XLSX; input to anchor building
- Purpose: Represents extracted text with spatial coordinates
- Attributes: text, page, bbox (PDF points), confidence
- Pattern: Extracted from PDF text layer; matched across revisions
- Purpose: Detected characteristic callout symbol on drawing
- Attributes: center, radius, number, page, method (pdf_text or cv2_detection)
- Pattern: Detected by vision module; helps locate anchors
- Purpose: Typed semantic representation of characteristic requirement (GD&T, weld, surface finish, fit)
- Attributes: provenance, status, raw_text, normalized_text, gdt/weld/surface_finish/fit payloads
- Pattern: Extracted and compared across revisions; enriches classification decisions
## Entry Points
- Location: `/home/khoa2/delta-preservation/run.py`
- Triggers: `python run.py <part_name>`
- Responsibilities:
- Location: `/home/khoa2/delta-preservation/run_web.py`
- Triggers: Docker startup or `uvicorn shop.app:app`
- Responsibilities:
- Location: `shop/app.py`
- Triggers: Module import (`from shop.app import app`)
- Responsibilities:
- Location: `delta_preservation/cli.py`
- Triggers: Called by run.py or `execute_run_task` (web task)
- Responsibilities:
## Error Handling
- Stage failures recorded in Run.failure_stage + Run.failure_message
- Run.status set to "failed"; RunAlert created for user
- Evidence partial: stages 1-5 may succeed even if 6+ fail
- PDF integrity check before Run creation (`validate_pdf_bytes()`)
- Form 3 parsing errors trigger Run.status="failed" + helpful alert
- Database integrity via SQLAlchemy constraints (ForeignKey, unique)
- ReviewItem verdict validation in `validate_debug_verdict_payload()`
- Sign-off blocked if uncertain items remain (configurable per org policy)
- Try-except at stage boundaries in CLI
- HTTPException(status_code=400) for user input validation in routers
- Structured error messages in RunAlert for admin/engineer visibility
## Cross-Cutting Concerns
- Approach: Python logging module with basicConfig in run_web.py + CLI
- Level: DEBUG in dev, INFO in production
- Used for: Pipeline progress, task queue events, HTTP request tracing
- Approach: Pydantic BaseModel for types.py (DeltaItem, DeltaPacket, SemanticCallout)
- SQLAlchemy ORM for web models with column constraints
- File validation functions in shop/services/runs.py (PDF, Excel integrity)
- Custom validators in reconcile modules (tolerance parsing, semantic extraction)
- Approach: Custom session-based auth in shop/services/auth.py
- Bcrypt password hashing (pwdlib)
- UserSession table with token + expiry
- Middleware/dependency: `get_current_user()` in shop/dependencies.py
- Admin: Access /admin, /setup, edit ShopConfig, sign-off runs
- Engineer: Submit runs, review own/shared runs
- Enforced via role check in routers + middleware
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
