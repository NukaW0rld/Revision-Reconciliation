# Architecture Research

**Domain:** Web application wrapping a blocking compute pipeline — aerospace QC review tool
**Researched:** 2026-03-01
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        Browser (LAN)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Upload UI   │  │ Review Queue │  │  Run History / Audit │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
└─────────┼────────────────┼──────────────────────┼───────────────┘
          │  HTTP + SSE    │  HTTP (polling/SSE)  │  HTTP
┌─────────▼────────────────▼──────────────────────▼───────────────┐
│                     FastAPI (ASGI, uvicorn)                       │
│  ┌───────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  /api/runs    │  │ /api/review  │  │  /api/export           │ │
│  │  POST upload  │  │ PATCH items  │  │  GET packet/workorder  │ │
│  │  GET status   │  │ POST sign-off│  │  /static/snippets/...  │ │
│  └───────┬───────┘  └──────┬───────┘  └────────────────────────┘ │
│          │ enqueue         │ read/write state                     │
│  ┌───────▼───────┐  ┌──────▼────────────────────────────────┐   │
│  │  Job Queue    │  │              SQLite DB                 │   │
│  │  (SQLite via  │  │  runs, review_items, users, sessions,  │   │
│  │   Huey/file)  │  │  audit_events, signed_packets          │   │
│  └───────┬───────┘  └───────────────────────────────────────┘   │
│          │ task consumed by                                        │
│  ┌───────▼───────────────────────────────────────────────────┐   │
│  │         Pipeline Worker (background thread/process)        │   │
│  │                                                            │   │
│  │  run_pipeline() → writes out/<run_id>/ to filesystem       │   │
│  │  (delta_packet.json, snippets/*.png, debug/*.json)         │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │          Filesystem (Docker volume-mounted)                │   │
│  │  uploads/<run_id>/  (input PDFs, xlsx — write-once)        │   │
│  │  out/<run_id>/      (pipeline output — write-once)         │   │
│  │  exports/<run_id>/  (generated PDF/CSV — write-once)       │   │
│  └───────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
```

Everything runs inside a single Docker container managed by supervisord (two processes: `uvicorn` and `huey worker`). The only storage is SQLite (one file) + the local filesystem (one mounted volume). No external services.

### Component Responsibilities

| Component | Responsibility | Communicates With |
|-----------|----------------|-------------------|
| FastAPI app | HTTP routing, auth, request validation, SSE progress streams, static file serving | Browser, SQLite, Job Queue, Filesystem |
| Job Queue (Huey + SQLite) | Accepts pipeline run tasks, exposes task status; decouples HTTP from blocking compute | FastAPI (enqueue), Worker (consume) |
| Pipeline Worker | Calls `run_pipeline()` synchronously; writes output to filesystem; updates run status in DB | Filesystem, SQLite (status updates), Existing pipeline code |
| SQLite DB | Authoritative state for runs, review items, users, sessions, audit events, signed packets | FastAPI, Worker |
| Filesystem (volume) | Stores input files (uploaded PDFs/xlsx), pipeline output (snippets/JSON), signed exports | FastAPI (serve), Worker (write) |
| Export Generator (WeasyPrint) | Renders audit packet PDF and work order PDF from review data + snippets | FastAPI (on-demand), Filesystem (reads snippets, writes PDF) |

---

## Recommended Project Structure

```
delta_preservation/          # existing pipeline (unchanged)

web/
├── main.py                  # FastAPI app factory, mounts routers and static
├── config.py                # web-layer config (DB path, upload dir, data dir)
├── db/
│   ├── session.py           # SQLAlchemy engine + session factory (aiosqlite + WAL)
│   ├── models.py            # ORM models: Run, ReviewItem, User, Session, AuditEvent, SignedPacket
│   └── migrations/          # Alembic or manual migration scripts
├── routers/
│   ├── auth.py              # Login, logout, session cookie
│   ├── runs.py              # Create run (upload), get run status, SSE progress
│   ├── review.py            # Get queue items, patch item (approve/override), sign-off
│   ├── export.py            # Generate + download audit packet and work order
│   └── admin.py             # Shop config, user management, retention policy
├── tasks/
│   ├── queue.py             # Huey instance configured with SqliteStorage
│   └── pipeline_task.py     # @huey.task wrapping run_pipeline(); writes progress to DB
├── services/
│   ├── export_service.py    # WeasyPrint PDF rendering, CSV generation
│   ├── run_service.py       # Business logic: create run record, validate uploads
│   └── review_service.py    # State machine transitions, sign-off atomicity
├── schemas/
│   └── *.py                 # Pydantic request/response schemas (separate from DB models)
├── static/                  # Compiled frontend assets (JS bundle, CSS)
├── templates/
│   └── export/              # Jinja2 HTML templates rendered → WeasyPrint → PDF
└── tests/
    ├── test_runs.py
    ├── test_review.py
    └── test_export.py

docker/
├── Dockerfile
├── supervisord.conf          # Manages uvicorn + huey worker processes
└── docker-compose.yml

data/                        # Volume mount point (outside container)
├── db/
│   └── app.db               # SQLite database file
├── uploads/                 # Input files (write-once after run creation)
│   └── <run_id>/
│       ├── revA.pdf
│       ├── revB.pdf
│       └── form3.xlsx
├── out/                     # Pipeline output (write-once, pipeline writes here)
│   └── <run_id>/
│       ├── delta_packet.json
│       ├── snippets/
│       └── debug/
└── exports/                 # Generated audit packets (write-once, immutable)
    └── <run_id>/
        ├── audit_packet_v1.pdf
        └── audit_packet_v1.csv
```

### Structure Rationale

- **web/ vs delta_preservation/:** Clean boundary between existing pipeline (untouched) and new web layer. The pipeline is imported as a library by `pipeline_task.py`, not modified.
- **tasks/:** Isolated job queue layer. Huey's `SqliteStorage` uses the same `data/db/` volume but a separate file (`huey.db`), keeping queue state distinct from application state.
- **services/:** Business logic kept out of routers; routers are thin HTTP adapters. `review_service.py` owns all state machine transitions.
- **data/ volume:** Single mount point for all persistent data. Backup = `rsync data/` or `tar data/`. No cloud storage, no NFS, no complexity.
- **templates/export/:** HTML-based PDF generation via WeasyPrint is simpler to maintain than low-level ReportLab. Jinja2 templates can embed base64-encoded snippet images inline.

---

## Architectural Patterns

### Pattern 1: Blocking Pipeline Bridged via Process-Isolated Worker

**What:** The existing `run_pipeline()` function is CPU-bound (OpenCV ORB/RANSAC, PyMuPDF rendering), GIL-holding, and not async-aware. It cannot run inside the FastAPI event loop without blocking all HTTP handling. The correct bridge is to enqueue jobs in Huey (SQLite-backed), and have a separate `huey worker` process consume them. The web process never calls `run_pipeline()` directly.

**When to use:** Any time an existing synchronous, CPU-heavy function must be integrated into an async web framework without modification.

**Why Huey + SQLite over alternatives:**
- No Redis, no RabbitMQ, no external broker — the SQLite storage file lives on the same Docker volume as the rest of the data.
- For O(10) concurrent users submitting O(100s) of runs/year, Huey's throughput is more than sufficient.
- Huey supports task result storage in the same SQLite DB, enabling status polling without a separate mechanism.
- Worker crashes are surfaced via supervisord restart policy; the job stays queued and is retried.

**Implementation sketch:**
```python
# web/tasks/queue.py
from huey import SqliteHuey
huey = SqliteHuey(filename="data/db/huey.db")

# web/tasks/pipeline_task.py
from web.tasks.queue import huey
from web.db.session import get_sync_session
from delta_preservation.cli import run_pipeline

@huey.task()
def run_pipeline_task(run_id: str, paths: dict):
    # runs in huey worker process — not the FastAPI process
    with get_sync_session() as db:
        db.execute("UPDATE runs SET status='running' WHERE id=:id", {"id": run_id})
        db.commit()
    try:
        result = run_pipeline(**paths)
        with get_sync_session() as db:
            db.execute(
                "UPDATE runs SET status='complete', packet_path=:p WHERE id=:id",
                {"p": result.packet_path, "id": run_id}
            )
            db.commit()
    except Exception as exc:
        with get_sync_session() as db:
            db.execute(
                "UPDATE runs SET status='failed', error=:e WHERE id=:id",
                {"e": str(exc), "id": run_id}
            )
            db.commit()
```

**Trade-offs:** Adds a second process (huey worker) to the container. This is handled by supervisord with two stanzas. The tradeoff is worthwhile: the alternative (threading or BackgroundTasks) risks blocking the event loop and is harder to restart on crash.

### Pattern 2: SSE for Pipeline Progress

**What:** After submitting a run, the browser polls or holds an open SSE connection to `/api/runs/{run_id}/progress`. The FastAPI endpoint reads run status from SQLite and yields events. FastAPI 0.135.0+ has native `EventSourceResponse` built into `fastapi.sse`.

**When to use:** One-directional server-to-client streaming of status updates. Simpler than WebSocket when the client does not need to send messages back.

**Implementation sketch:**
```python
from fastapi.sse import EventSourceResponse, ServerSentEvent
from collections.abc import AsyncIterable

@router.get("/runs/{run_id}/progress", response_class=EventSourceResponse)
async def run_progress(run_id: str) -> AsyncIterable[ServerSentEvent]:
    while True:
        run = await db.get_run(run_id)
        yield ServerSentEvent(data=run.model_dump_json(), event="status")
        if run.status in ("complete", "failed"):
            return
        await asyncio.sleep(1.5)
```

**Trade-offs:** SSE keeps a persistent HTTP connection open per tab. For O(10) concurrent users, this is negligible. The polling interval (1.5s) keeps DB load low while providing a responsive UI.

### Pattern 3: Review State Machine in SQLite

**What:** Each review item has a status column constrained to a defined set: `pending | approved | overridden | skipped`. State transitions are validated in `review_service.py` before writing. Sign-off is only available when no items remain in `pending` state. Sign-off and packet generation are wrapped in a single DB transaction with filesystem write; if the WeasyPrint render fails, the transaction is rolled back.

**When to use:** Any workflow where items must reach terminal states before a gate action (sign-off), and where the audit trail must record who did what when.

**State table:**
```
pending → approved       (engineer clicks Approve)
pending → overridden     (engineer overrides with required note)
approved → overridden    (engineer changes mind; still requires note)
overridden → approved    (engineer accepts original classification)
```

**Amendment handling:** A finalized run's `SignedPacket` row is never updated. Amendment creates a new `Run` row with `amendment_of = <original_run_id>` and copies review items to a new set, allowing re-review. The original signed packet file remains on disk, untouched.

**Trade-offs:** State machine logic in application code (not DB triggers) is easier to test and extend. The cost is that invalid transitions can only be caught at the service layer, not the DB constraint layer — acceptable for this scale.

### Pattern 4: Filesystem as Write-Once Evidence Store

**What:** Pipeline outputs (snippets, delta_packet.json) and signed exports (audit PDFs, CSVs) are treated as write-once artifacts on the filesystem. Once written, they are never modified in place — amendments create new files under a new run_id. FastAPI's `StaticFiles` mount serves snippet images directly from the `data/out/` directory.

**When to use:** When output immutability is a requirement (audit integrity) and the data is file-based (images, PDFs) rather than structured records.

**Implementation sketch:**
```python
# web/main.py
from fastapi.staticfiles import StaticFiles
app.mount("/static/out", StaticFiles(directory="data/out"), name="out")
app.mount("/static/exports", StaticFiles(directory="data/exports"), name="exports")
app.mount("/static", StaticFiles(directory="web/static"), name="static")
```

The browser loads snippet images via `/static/out/<run_id>/snippets/<char_no>_revA.png` — no API endpoint needed for image serving.

**Trade-offs:** No access control on snippet images served via `StaticFiles` (anyone with the URL can view them). For a shop LAN with authenticated sessions, this is acceptable. If tighter control is needed, serve snippets through an authenticated API endpoint instead.

---

## Data Flow

### Upload → Pipeline → Review → Export

```
1. Engineer submits upload form
   Browser → POST /api/runs (multipart: revA.pdf, revB.pdf, form3.xlsx, metadata)
          → FastAPI saves files to data/uploads/<run_id>/
          → Creates Run row in SQLite (status=queued)
          → Enqueues run_pipeline_task(run_id) in Huey
          → Returns {run_id, status_url}

2. Pipeline execution
   Huey Worker → dequeues task
              → Updates Run.status = running
              → Calls run_pipeline(revA=..., revB=..., form3=..., out_dir=data/out/<run_id>/)
              → Pipeline writes: data/out/<run_id>/delta_packet.json + snippets/*.png
              → Worker reads delta_packet.json
              → Writes ReviewItem rows to SQLite (one per DeltaItem in packet)
              → Updates Run.status = review_pending
              → Signals failure → Run.status = failed + Run.error = message

3. Progress streaming
   Browser → GET /api/runs/{run_id}/progress (SSE, stays open)
   FastAPI → polls SQLite Run.status every 1.5s
           → yields ServerSentEvent on each change
           → closes stream when status = review_pending | failed

4. Review
   Browser → GET /api/runs/{run_id}/review  → list of ReviewItem with snippet URLs
   Engineer approves/overrides each item
   Browser → PATCH /api/review/{item_id}    → {action: approve|override, note?: string}
   FastAPI → validate state transition in review_service
           → UPDATE ReviewItem.status, ReviewItem.reviewer_note, ReviewItem.reviewed_by
           → UPDATE ReviewItem.reviewed_at = now()
           → INSERT AuditEvent(run_id, item_id, action, actor, timestamp)

5. Sign-off
   Browser → POST /api/runs/{run_id}/signoff
   FastAPI → verify all ReviewItems are terminal (none pending)
           → begin DB transaction
           → render audit_packet PDF via WeasyPrint + Jinja2 template
              (reads: SQLite review data, data/out/<run_id>/snippets/*.png as base64)
           → write to data/exports/<run_id>/audit_packet_v1.pdf + .csv
           → INSERT SignedPacket row (run_id, path, sha256, signed_by, signed_at)
           → UPDATE Run.status = finalized
           → commit transaction
           → if WeasyPrint fails → rollback transaction → return 500, run stays review_pending

6. Work order (on demand, post-sign-off)
   Browser → POST /api/runs/{run_id}/workorder
   FastAPI → filter ReviewItems where status = changed | added
           → render work_order PDF + CSV
           → write to data/exports/<run_id>/workorder_v1.pdf + .csv
           → return download URL
```

### State Flow (Run)

```
queued → running → review_pending → finalized
                 ↘ failed
```

### State Flow (ReviewItem)

```
pending → approved
pending → overridden
approved ↔ overridden   (re-review allowed until run is finalized)
```

---

## Build Order

The web layer has hard dependencies that determine build order:

```
Phase 1: Foundation (no dependencies)
  ├── Docker + supervisord setup
  ├── FastAPI app skeleton + uvicorn
  ├── SQLite DB schema + SQLAlchemy models
  └── Auth (session cookie, user roles)
        ↓
Phase 2: Pipeline Bridge (depends on Phase 1)
  ├── Huey + SqliteStorage job queue
  ├── File upload endpoint → saves to data/uploads/<run_id>/
  ├── pipeline_task.py wrapping run_pipeline()
  └── Run status polling (SSE progress endpoint)
        ↓
Phase 3: Review Queue (depends on Phase 2 — needs real pipeline output to review)
  ├── ReviewItem population from delta_packet.json
  ├── Review API (GET queue, PATCH item)
  ├── State machine transitions + AuditEvent logging
  ├── Snippet serving via StaticFiles
  └── Sign-off gate + atomic packet generation
        ↓
Phase 4: Export + Admin (depends on Phase 3)
  ├── WeasyPrint audit packet PDF + CSV
  ├── Work order PDF + CSV
  ├── Admin panel (user management, shop config, retention)
  └── Run history + amendment model
```

**Why this order:**
- Auth must come first — every subsequent endpoint needs identity.
- Pipeline bridge must come before review — you cannot build the review UI until you have real pipeline output to review.
- Sign-off must come before export — the export references the signed state.
- Admin and amendment are independent of each other and can be parallelized in Phase 4.

---

## Anti-Patterns

### Anti-Pattern 1: Calling run_pipeline() Inside the FastAPI Event Loop

**What people do:** Use `asyncio.run_in_executor(None, run_pipeline, ...)` or FastAPI's `BackgroundTasks` with a sync function, calling the pipeline directly from the web process.

**Why it's wrong:** `run_pipeline()` takes 10–60 seconds, holds the GIL during OpenCV operations, and blocks other requests during that time. With a single uvicorn worker (which is the standard for a LAN deployment), this makes the UI unresponsive for all users during a run.

**Do this instead:** Enqueue the task into Huey. The Huey worker is a separate process that owns all CPU time during the run. The FastAPI process stays responsive.

### Anti-Pattern 2: Storing Pipeline Output in the Database

**What people do:** Read delta_packet.json and inline all snippet image bytes as BLOBs in SQLite.

**Why it's wrong:** PNG snippets are typically 50–200 KB each; a 100-characteristic part has 200 snippets (RevA + RevB per item) = 10–40 MB per run. Storing this in SQLite bloats the database and slows all queries. SQLite is optimized for structured records, not file storage.

**Do this instead:** Keep snippet files on the filesystem under `data/out/<run_id>/snippets/`. Store only the relative path in the ReviewItem row. Serve images via `StaticFiles` mount.

### Anti-Pattern 3: Allowing Overwrite of Signed Packets

**What people do:** Re-run sign-off and overwrite the existing PDF if the engineer spots a mistake.

**Why it's wrong:** AS9102 audit records must be immutable once signed. Overwriting destroys the audit chain.

**Do this instead:** Signed packets are never overwritten. Correction = create an amendment (new Run row with `amendment_of` foreign key). Amendment produces a new `SignedPacket` under a new version slug (`audit_packet_v2.pdf`).

### Anti-Pattern 4: In-Process SQLite Writer Contention

**What people do:** Both the FastAPI process and the Huey worker process connect to the same SQLite file with default journal mode.

**Why it's wrong:** With default journal mode, a write in one process can produce `database is locked` errors in the other under concurrent access.

**Do this instead:** Enable WAL mode on the SQLite file at startup (`PRAGMA journal_mode=WAL`). WAL allows simultaneous readers and one writer, eliminating lock contention in the typical pattern where the web process reads frequently and the worker writes infrequently.

```python
# web/db/session.py
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine("sqlite+aiosqlite:///data/db/app.db")

@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_conn, _):
    dbapi_conn.execute("PRAGMA journal_mode=WAL")
    dbapi_conn.execute("PRAGMA synchronous=NORMAL")
    dbapi_conn.execute("PRAGMA busy_timeout=5000")
```

### Anti-Pattern 5: Silent Review Overrides

**What people do:** Allow engineers to change a classification without recording a reason, to make the UI faster.

**Why it's wrong:** AS9102 audit integrity requires that every human correction to a system classification is documented. An override without a note has no audit value and cannot support a nonconformance investigation.

**Do this instead:** The API rejects PATCH requests with `action=override` and no `note` field. The UI enforces this with a required textarea before the override button is enabled.

---

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| FastAPI ↔ Huey Worker | SQLite (Huey tables) — enqueue + task result | Both processes read/write the same DB volume; WAL mode required |
| FastAPI ↔ Pipeline Output | Filesystem read — delta_packet.json, snippets/*.png | FastAPI reads after worker writes; no locking needed (write-once) |
| Huey Worker ↔ Pipeline | Direct Python function call (`from delta_preservation.cli import run_pipeline`) | Worker process imports existing pipeline; no network hop |
| FastAPI ↔ Export Generator | In-process function call (`export_service.generate_packet(run_id)`) | WeasyPrint runs in the FastAPI process; acceptable since export is user-triggered, not concurrent with pipeline |
| Browser ↔ Snippets | HTTP GET `/static/out/<run_id>/snippets/*.png` via StaticFiles | No auth on static mount; acceptable for LAN deployment |

### Docker Volume Layout

```yaml
# docker-compose.yml (single-container, two processes)
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data          # persistent: db, uploads, out, exports
    environment:
      - DATA_DIR=/app/data
      - DATABASE_URL=sqlite+aiosqlite:///app/data/db/app.db
      - HUEY_DB=/app/data/db/huey.db
```

Supervisord runs two programs:
1. `uvicorn web.main:app --host 0.0.0.0 --port 8000`
2. `huey_consumer web.tasks.queue.huey --workers 1`

`--workers 1` on the huey consumer ensures only one pipeline job runs at a time, which is correct: the pipeline is CPU-saturating and concurrent runs would compete for the same CPU cores.

---

## Scalability Considerations

This system is sized for shop-scale (O(10) engineers, O(100s) runs/year). Scalability beyond this scope is explicitly out of scope for v1.

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 1–10 engineers, 100s of runs/year | Current architecture — SQLite + single container + 1 Huey worker |
| 10–50 engineers, 1000s of runs/year | Increase Huey workers to 2–3 (CPU-permitting); move to PostgreSQL; add Nginx reverse proxy |
| 50+ engineers | Break into separate web + worker containers; dedicated job broker (Redis/RabbitMQ); shared NFS for files |

**First bottleneck:** Pipeline throughput. With 1 worker, runs are serialized. If multiple engineers submit simultaneously, they queue. For v1 this is acceptable — runs complete in 30–120 seconds and the queue is visible.

**Second bottleneck:** SQLite write contention. WAL mode handles the current 2-process pattern well. Beyond ~5 concurrent writers, contention may appear — at that point, migrate to PostgreSQL.

---

## Sources

- FastAPI SSE built-in (0.135.0+): https://fastapi.tiangolo.com/tutorial/server-sent-events/
- FastAPI async/CPU-bound guidance: https://fastapi.tiangolo.com/async/
- FastAPI file uploads: https://fastapi.tiangolo.com/tutorial/request-files/
- FastAPI Docker deployment: https://fastapi.tiangolo.com/deployment/docker/
- Huey SQLite task queue: https://huey.readthedocs.io/en/latest/api.html
- Huey GitHub (SQLiteStorage, WAL mode): https://github.com/coleifer/huey
- SQLAlchemy SQLite WAL + aiosqlite: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html
- SQLite WAL mode: https://sqlite.org/wal.html
- Docker multi-process container: https://docs.docker.com/engine/containers/multi-service_container/
- supervisord in containers: https://www.advantch.com/blog/how-to-run-multiple-processes-in-a-single-docker-container
- WeasyPrint PDF generation: https://dev.to/claudeprime/generate-pdfs-in-python-weasyprint-vs-reportlab-ifi
- FastAPI BackgroundTasks blocking issue: https://github.com/fastapi/fastapi/discussions/11210
- ProcessPoolExecutor vs run_in_executor: https://sentry.io/answers/fastapi-difference-between-run-in-executor-and-run-in-threadpool/
- Immutable audit trail patterns: https://www.hubifi.com/blog/immutable-audit-log-basics
- SQLite in production (single-server): https://shivekkhurana.com/blog/sqlite-in-production/

---
*Architecture research for: delta-preservation web layer*
*Researched: 2026-03-01*
