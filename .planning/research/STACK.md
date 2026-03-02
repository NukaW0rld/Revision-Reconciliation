# Stack Research

**Domain:** Python-native web application wrapping a compute-heavy CV pipeline; multi-user review UI with image display; PDF audit artifact generation; Docker LAN deployment
**Researched:** 2026-03-01
**Confidence:** HIGH (all versions verified against PyPI; patterns verified via official docs and multiple sources)

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| FastAPI | 0.135.1 | HTTP framework, REST endpoints, file upload, Jinja2 templates | Async-native (required to not block on pipeline jobs), Pydantic v2 already in pipeline's dependency tree, best DX for API + SSR hybrid, 70K+ GitHub stars overtaking Flask in 2025 active adoption |
| Uvicorn | 0.41.0 | ASGI server (dev + prod worker) | FastAPI's native ASGI runner; used directly in dev, managed by Gunicorn in prod |
| Gunicorn | 25.1.0 | Production process manager | Manages N Uvicorn workers, handles crashes and restarts; standard FastAPI prod pattern; `gunicorn -k uvicorn.workers.UvicornWorker` |
| SQLAlchemy | 2.0.47 | ORM and query layer | Mature, battle-tested, async support (`asyncio` engine), direct Pydantic v2 interop; do NOT use SQLModel (see below) |
| Alembic | 1.18.4 | Database schema migrations | Standard SQLAlchemy migration tool; autogenerate diffs from ORM models; rollback-safe |
| SQLite | 3.x (system) | Primary database | Single-file, zero-config, no separate server to run in Docker; O(10) engineers and O(100s) of runs/year never approaches write concurrency limits; WAL mode required for concurrent reads during writes |
| Dramatiq | 2.0.1 | Background job queue for pipeline execution | Actively maintained (v2.0.1 released Jan 2026), Redis-backed, simple decorator API, async middleware available, substantially faster than RQ in load tests; preferred over ARQ (maintenance-only) and Celery (overkill) |
| Redis | 7.2.1 (client) | Job queue broker and result backend for Dramatiq | Dramatiq Redis broker; simple Docker service; also stores job status for polling |
| Jinja2 | 3.1.6 | Server-side HTML templates | FastAPI has native Jinja2/Starlette integration; no build step, no SPA overhead; right choice for O(10) internal users with server-driven review UI |
| HTMX | 2.0.8 (CDN) | Partial-page updates without SPA framework | Server-rendered HTML fragments for review queue interactions (approve, override, status polling); ~14KB; no build toolchain; natural complement to Jinja2 |
| Alpine.js | 3.15.8 (CDN) | Client-side reactivity for local UI state | Required for: confidence score display, toggle override note visibility, form validation feedback; ~15KB; no build toolchain; does NOT replace server communication (that is HTMX's role) |
| WeasyPrint | 68.1 | HTML+CSS to PDF generation | No headless browser; Jinja2 templates drive both web UI and PDF output (reuses same layout code); handles image embedding via `file://` URIs for evidence PNG snippets; system deps: `libpango`, `libharfbuzz-subset0`, `libjpeg-dev` (add to Dockerfile) |
| pwdlib | 0.3.0 | Password hashing | Modern, maintained replacement for passlib (last passlib release: 2020, breaks on Python 3.13); supports Argon2id (OWASP 2025 recommendation) and bcrypt; requires Python 3.10+ (matches pipeline constraint) |
| python-jose | 3.5.0 | JWT creation and validation | Standard FastAPI auth pattern; used with OAuth2PasswordBearer for session tokens |
| python-multipart | 0.0.22 | Multipart form + file upload parsing | Required by FastAPI for `UploadFile`; handles PDF and Excel file uploads |
| python-dotenv | 1.2.2 | Environment variable configuration | Load secrets and config from `.env` in development; Pydantic `BaseSettings` reads them in production |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| aiofiles | 24.x | Async file I/O | Reading snippet PNGs from disk to serve in review UI without blocking the event loop |
| python-magic | 0.4.27 | MIME type validation via file magic bytes | Validate uploaded PDFs and Excel files server-side; do not trust Content-Type header alone |
| Pydantic Settings | 2.x (bundled with Pydantic) | Config model with env var loading | Type-safe configuration for secrets, thresholds, Redis URL, DB path |
| httpx | 0.28.x | Async HTTP test client | Used in pytest with `AsyncClient` for testing FastAPI endpoints; replaces `requests` in test suite |
| pytest-asyncio | 0.26.x | Async test support | Required when testing `async def` FastAPI routes; configure with `asyncio_mode = "auto"` |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| uv | Dependency management and virtualenv | Already in use by pipeline; add web deps to same `pyproject.toml` |
| Docker + docker-compose | Container build and shop deployment | `docker-compose.yml` with three services: `web`, `worker` (Dramatiq), `redis`; SQLite DB file on a named volume |
| pytest | Test runner | Already in use; extend with `pytest-asyncio` and `httpx.AsyncClient` for route testing |
| Ruff | Linting + formatting | Replaces flake8/black in 2025; single tool, fast, uv-compatible |

---

## Installation

```bash
# Add to pyproject.toml [project.dependencies] (uv manages the rest)

# Core web framework
fastapi==0.135.1
uvicorn[standard]==0.41.0
gunicorn==25.1.0
python-multipart==0.0.22
jinja2==3.1.6
aiofiles

# Database
sqlalchemy==2.0.47
alembic==1.18.4

# Job queue
dramatiq[redis]==2.0.1
redis==7.2.1

# Auth
pwdlib[argon2]==0.3.0
python-jose[cryptography]==3.5.0

# PDF generation
weasyprint==68.1

# Config
python-dotenv==1.2.2

# Dev/test
httpx
pytest-asyncio
ruff

# uv sync installs everything from lockfile
uv sync
```

**Dockerfile system dependencies for WeasyPrint** (add to your `apt-get install` layer):
```
libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 libjpeg-dev libopenjp2-7-dev libffi-dev
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| FastAPI | Django | If you need Django Admin for content management, or the team has deep Django expertise; Django's ORM and admin save months on CRUD-heavy apps; not chosen here because pipeline is async-heavy, FastAPI's DX is better for API-first design, and Django admin is not relevant to this use case |
| FastAPI | Flask | If you want zero learning curve and don't need native async; Flask 3.x has ASGI support but it's bolted on; not chosen because async-first matters for non-blocking pipeline dispatch |
| SQLite | PostgreSQL | If: write concurrency exceeds SQLite WAL limits (i.e., more than a few concurrent writes/sec), or if you need full-text search, or if the shop scales to multiple Docker hosts; PostgreSQL requires an additional service in docker-compose; SQLite is right for the described scale (O(10) users, O(100s) runs/year) |
| SQLAlchemy | SQLModel | SQLModel is built on SQLAlchemy but trades performance for convenience; benchmarks show significantly slower query performance vs raw SQLAlchemy; at O(100s) of runs it would not matter, but SQLAlchemy's ecosystem is larger and better documented; SQLModel is acceptable if team prefers the unified model approach |
| Dramatiq | ARQ | ARQ is maintenance-only as of 2026; use ARQ only if you need true asyncio-native tasks without any sync bridge; Dramatiq 2.0 has async middleware that covers the FastAPI integration case |
| Dramatiq | Celery | Celery is the right choice at high task volume with complex workflow DAGs (chains, groups, chords); for this use case (serial pipeline jobs, O(10) concurrent users) Celery adds operational complexity with no benefit |
| WeasyPrint | ReportLab | ReportLab gives programmatic control over every PDF element (useful for complex charts); choose ReportLab if WeasyPrint CSS rendering proves inadequate for the audit packet layout; WeasyPrint is chosen here because it lets you reuse the same Jinja2 template for both web preview and PDF export |
| WeasyPrint | Playwright/Puppeteer | Chromium-based tools produce pixel-perfect PDFs but add a ~300MB binary to the Docker image and introduce a separate process; not appropriate for a LAN-deployed shop tool |
| HTMX + Alpine.js + Jinja2 | React/Vue SPA | SPA frameworks require a separate build step (Node, webpack/vite), a separate deployment artifact, and CORS configuration; for O(10) internal users a SPA is architectural overhead; server-rendered HTML with HTMX covers the partial-update needs without a JS build pipeline |
| pwdlib | passlib | passlib's last release was 2020; it raises warnings and will break on Python 3.13; do not use passlib for new code in 2025+ |
| python-jose | PyJWT | Both are fine; python-jose is already the pattern in the FastAPI official docs and tutorial; PyJWT is an acceptable alternative with a simpler API |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| passlib | Last release 2020; issues with Python 3.13 crypt module removal; unmaintained | pwdlib 0.3.0 with Argon2id |
| SQLModel | Performance slower than raw SQLAlchemy due to double Pydantic validation on every query; ORM model and API schema conflation creates coupling problems in non-trivial apps | SQLAlchemy 2.0 + Pydantic schemas separately |
| FastAPI `BackgroundTasks` for pipeline execution | Built-in BackgroundTasks runs in the same process as the web server; a pipeline job (OpenCV, PyMuPDF, NumPy homography) will block the event loop or exhaust workers; not crash-safe (in-flight jobs lost on restart) | Dramatiq + Redis; enqueue job, return job ID, poll for status |
| ARQ | Maintenance-only as of v0.27.0; the project acknowledges it will not receive new features | Dramatiq 2.0 with async middleware |
| Celery | Requires separate result backend config, broker tuning, and flower monitoring for operational visibility; overkill for O(10) users | Dramatiq with Redis broker |
| Playwright/Puppeteer for PDF | Adds ~300MB Chromium binary to Docker image; network process isolation issues in some LAN environments; heavier than needed for structured report PDFs | WeasyPrint 68.1 |
| React/Vue SPA frontend | Requires Node.js build toolchain in CI, separate static file hosting or SPA routing config, and CORS setup; no benefit over HTMX for an internal tool with <10 users | HTMX + Alpine.js + Jinja2 |
| OAuth / SSO | Out of scope per PROJECT.md; email+password is explicitly scoped for pilot | pwdlib + python-jose JWT |
| PostgreSQL | Correct at scale, but adds a second stateful service to docker-compose, requires connection pooling config, and needs backup procedures; SQLite in WAL mode is production-viable at this scale | SQLite with WAL mode enabled |

---

## Stack Patterns by Variant

**If the pipeline run takes >30 seconds (likely: ORB feature matching on high-DPI pages):**
- Enqueue via Dramatiq; return job ID immediately from the POST endpoint
- Poll job status via HTMX-driven GET with `hx-trigger="every 2s"` until status is `complete` or `failed`
- Do NOT use FastAPI `BackgroundTasks` — they block the same worker process

**If WeasyPrint system dependencies cause Docker build issues on Alpine base image:**
- Switch Dockerfile base from `python:3.12-alpine` to `python:3.12-slim-bookworm` (Debian Bookworm)
- All WeasyPrint system deps are straightforward `apt-get install` on Debian; Alpine requires musl-specific package names that change between Alpine versions

**If SQLite write contention appears (symptom: `database is locked` errors):**
- Enable WAL mode at startup: `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`
- Add `check_same_thread=False` to SQLAlchemy engine and use connection pool with `StaticPool`
- If contention persists at scale, migrate to PostgreSQL (Alembic migrations carry forward unchanged)

**If shop runs on Windows Server instead of Linux:**
- WeasyPrint is supported on Windows but system dep installation is more complex (requires GTK runtime)
- Docker deployment (Linux container on Docker Desktop for Windows) sidesteps this — recommend staying with Docker
- Do not attempt native Windows pip install of WeasyPrint for production

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| FastAPI 0.135.1 | Pydantic 2.x, Starlette 0.46.x | Pydantic v1 is not supported in FastAPI ≥0.100; pipeline already uses Pydantic 2.12.5 — no conflict |
| SQLAlchemy 2.0.47 | Alembic 1.18.4 | SQLAlchemy 2.0 API required; Alembic 1.x tracks SQLAlchemy 2.x; do not mix with SQLAlchemy 1.4 patterns |
| Dramatiq 2.0.1 | Redis 7.x client, Python 3.10+ | Dramatiq 2.0 dropped Python <3.10; matches pipeline constraint |
| WeasyPrint 68.1 | Python 3.10–3.13 | Matches pipeline constraint; requires Pango ≥1.44 on host |
| pwdlib 0.3.0 | Python 3.10–3.14, Pydantic 2.x | Argon2 extra: `pwdlib[argon2]`; bcrypt extra: `pwdlib[bcrypt]` |
| python-jose 3.5.0 | Python 3.9–3.13 | cryptography backend: `python-jose[cryptography]` |
| HTMX 2.0.8 | No IE11 | HTMX 2.x dropped IE support; not a concern for shop LAN |

---

## Sources

- FastAPI official docs — https://fastapi.tiangolo.com/ — framework capabilities, Jinja2 integration, file upload, security/JWT pattern (HIGH confidence)
- PyPI FastAPI 0.135.1 — https://pypi.org/project/fastapi/ — version verified (HIGH confidence)
- PyPI SQLAlchemy 2.0.47 — https://pypi.org/project/SQLAlchemy/ — version verified (HIGH confidence)
- PyPI Alembic 1.18.4 — https://pypi.org/project/alembic/ — version verified (HIGH confidence)
- PyPI Uvicorn 0.41.0 — https://pypi.org/project/uvicorn/ — version verified (HIGH confidence)
- PyPI Gunicorn 25.1.0 — https://pypi.org/project/gunicorn/ — version verified (HIGH confidence)
- PyPI Dramatiq 2.0.1 — https://pypi.org/project/dramatiq/ — version verified, Jan 2026 release (HIGH confidence)
- PyPI WeasyPrint 68.1 — https://pypi.org/project/weasyprint/ — version verified (HIGH confidence)
- WeasyPrint docs — https://doc.courtbouillon.org/weasyprint/stable/first_steps.html — system deps on Linux verified (HIGH confidence)
- PyPI pwdlib 0.3.0 — https://pypi.org/project/pwdlib/ — version verified, Oct 2025 release (HIGH confidence)
- PyPI python-jose 3.5.0 — https://pypi.org/project/python-jose/ — version verified (HIGH confidence)
- PyPI python-multipart 0.0.22 — https://pypi.org/project/python-multipart/ — version verified (HIGH confidence)
- PyPI redis 7.2.1 — https://pypi.org/project/redis/ — version verified (HIGH confidence)
- PyPI arq 0.27.0 — https://pypi.org/project/arq/ — maintenance-only status confirmed (HIGH confidence)
- PyPI Jinja2 3.1.6 — https://pypi.org/project/jinja2/ — version verified (HIGH confidence)
- PyPI python-dotenv 1.2.2 — https://pypi.org/project/python-dotenv/ — version verified (HIGH confidence)
- ARQ GitHub issue #437 "Future plan for Arq" — https://github.com/python-arq/arq/issues/437 — maintenance-only status confirmed (MEDIUM confidence, GitHub issue)
- HTMX 2.0.8 release — https://htmx.org/ and https://github.com/bigskysoftware/htmx/releases — current stable version confirmed (HIGH confidence)
- Alpine.js 3.15.8 — https://alpinejs.dev/essentials/installation — current stable CDN version confirmed (HIGH confidence)
- passlib PyPI warehouse issue — https://github.com/pypi/warehouse/issues/15454 — maintenance/3.13 breakage confirmed (MEDIUM confidence)
- JetBrains 2025 framework comparison — https://blog.jetbrains.com/pycharm/2025/02/django-flask-fastapi/ — FastAPI positioning confirmed (MEDIUM confidence, third-party)
- FastAPI deployment docs — https://fastapi.tiangolo.com/deployment/server-workers/ — Gunicorn+Uvicorn worker pattern (HIGH confidence)
- SQLite "Appropriate Uses" — https://sqlite.org/whentouse.html — production suitability guidance (HIGH confidence, official)
- pwdlib introduction — https://www.francoisvoron.com/blog/introducing-pwdlib-a-modern-password-hash-helper-for-python — maintained passlib alternative rationale (MEDIUM confidence)
- Dramatiq docs — https://dramatiq.io/ — async middleware, Redis broker (HIGH confidence, official)

---
*Stack research for: Delta Preservation web application layer*
*Researched: 2026-03-01*
