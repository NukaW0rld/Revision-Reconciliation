# Technology Stack

**Analysis Date:** 2026-04-09

## Languages

**Primary:**
- Python 3.10–3.12 - Backend pipeline, web API, database models
- JavaScript/TypeScript - Frontend (minimal, CSS build only)
- SQL - Database schema via SQLAlchemy ORM

**Supporting:**
- Jinja2 Templates - HTML templating
- CSS/Tailwind - Styling with DaisyUI components

## Runtime

**Environment:**
- Python 3.11 (Docker runtime, `/home/khoa2/delta-preservation/docker/Dockerfile` specifies `python:3.11-slim`)
- Node.js 22 (Alpine, used for Tailwind CSS compilation in Docker multi-stage build)

**Package Manager:**
- `uv` (Astral package manager) - Primary dependency resolver (`/home/khoa2/delta-preservation/uv.lock`)
- npm - JavaScript/CSS dependencies (`/home/khoa2/delta-preservation/package.json`)

## Frameworks & Libraries

**Web Framework:**
- FastAPI 0.* - REST API framework with automatic OpenAPI docs
- Uvicorn - ASGI server (runs on port 8000)
- Jinja2 - Server-side template rendering

**Database & ORM:**
- SQLAlchemy 2.* - SQL toolkit and ORM (`shop.database.py`, `shop.models.py`)
- Alembic 1.18.4 - Database schema migrations (`/home/khoa2/delta-preservation/alembic/`)
- SQLite - Default file-based database (configurable via `DATABASE_URL` env var)

**Task Queue:**
- Huey (SqliteHuey) - Lightweight task queue backed by SQLite (`shop/tasks.py`)
- Supervisor - Process manager for uvicorn + huey_worker (see `docker/supervisord.conf`)

**PDF & Document Processing:**
- PyMuPDF (fitz) - PDF parsing and rendering
- WeasyPrint - HTML-to-PDF generation for audit packet exports
- OpenPyXL - Excel file reading (Form 3 parsing)

**Computer Vision & Image Processing:**
- OpenCV (opencv-python-headless) - Image processing, balloon detection, feature matching
- NumPy - Numerical computations

**Authentication & Security:**
- pwdlib[bcrypt] - Password hashing library
- bcrypt ≥4.0.0,<5.0.0 - Bcrypt hashing algorithm
- itsdangerous - Token generation for session management

**Data Validation & Types:**
- Pydantic ≥2.5,<3.0 - Data validation, model definitions (`delta_preservation/types.py`)
- RapidFuzz - Fuzzy string matching for characteristic reconciliation

**Utilities:**
- python-multipart - Form data parsing in FastAPI

**Styling & UI Components:**
- Tailwind CSS 4.0.0 - Utility-first CSS framework
- DaisyUI 5.0.0 - Component library built on Tailwind

**Development & Testing:**
- pytest ≥8 - Testing framework
- httpx ≥0.28.1 - Async HTTP client for tests

## Configuration

**Environment Variables:**
- `DATABASE_URL` - SQLAlchemy connection string (default: `sqlite:///./shop.db`)
- `HUEY_DB` - Path to Huey task queue SQLite database (default: `/app/data/huey.db`)
- `OUT_DIR` - Output directory for pipeline artifacts (default: `/app/out`)
- `UPLOADS_DIR` - File upload storage directory (default: `/app/data/uploads`)
- `ADMIN_EMAIL` - Default admin username (default: `admin@shop.local`)
- `ADMIN_DEFAULT_PASSWORD` - Default admin password (default: `changeme`)

**Build Configuration:**
- `pyproject.toml` - Python project metadata, dependencies, test configuration
- `.tailwindrc` - Tailwind CSS configuration (auto-generated or inferred from HTML)
- `tsconfig.json` - Not present (no TypeScript compilation)

**Database Migrations:**
- Alembic configuration: `alembic.ini`
- Migrations run automatically on container startup via `docker/Dockerfile` line 46: `alembic upgrade head`

## Build & Bundling

**CSS Compilation:**
- Tailwind CLI - Compiles `static/src/input.css` → `static/dist/output.css`
- Build command: `npm run build:css` (defined in `package.json`)
- Executed in Docker multi-stage build before Python runtime image is created

**Python Packaging:**
- `uv sync --locked` - Installs frozen dependencies from `uv.lock`
- No compiled extensions (relies on pre-built wheels for NumPy, OpenCV, bcrypt)

## Infrastructure

**Containerization:**
- Docker 3-stage build (`docker/Dockerfile`):
  1. **Stage 1 (tailwind-builder):** Node.js 22-Alpine, compiles CSS
  2. **Stage 2 (python-builder):** Python 3.11, resolves dependencies via `uv sync`
  3. **Stage 3 (runtime):** Python 3.11-slim, installs supervisor, runs migrations + services

**Process Management:**
- Supervisor (`docker/supervisord.conf`) manages two concurrent processes:
  - `uvicorn` (web server, priority 10)
  - `huey_worker` (background task queue consumer, priority 20)

**Orchestration:**
- Docker Compose (`docker/docker-compose.yml`) - Single service `web` with:
  - Port mapping: `8000:8000`
  - Volume mounts: `/app/data` (persistent DB), `/app/out` (pipeline output)
  - Restart policy: `unless-stopped`

**System Dependencies (in Docker):**
- `supervisor` - Process manager
- `libpango-1.0-0`, `libpangoft2-1.0-0`, `libharfbuzz-subset0` - Required by WeasyPrint for font rendering

## Platform Requirements

**Development:**
- Python 3.10+ locally (with `uv` or `pip` for installation)
- Node.js 22+ for CSS compilation (if building outside Docker)
- SQLite3 (typically pre-installed on Linux/macOS)

**Production:**
- Docker Engine (for containerized deployment)
- Persistent storage for `/app/data` (SQLite DB, Huey queue, uploads)
- Writable `/app/out` directory for pipeline output
- Port 8000 exposed (or reverse proxy)
- Minimum 2 OS threads (one for uvicorn, one for huey_worker)

---

*Stack analysis: 2026-04-09*
