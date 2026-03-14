# Project Research Summary

**Project:** Delta Preservation — Web Application Layer
**Domain:** Aerospace QC audit tool — web wrapper around a Python compute pipeline (AS9102 FAIR compliance)
**Researched:** 2026-03-01
**Confidence:** HIGH (stack fully verified; architecture patterns established; pitfalls confirmed via multiple independent sources)

## Executive Summary

Delta Preservation is a domain-specific, audit-sensitive internal tool: a web layer wrapping an existing 8-stage computer vision pipeline that compares engineering drawing revisions for AS9102 FAIR compliance. The product is not a general CRUD app — it is a quality management system whose primary output (a signed audit packet) has legal and contractual implications. Experts building this category of tool prioritize audit integrity (immutable records, mandatory justifications, individual accountability on every action) above UX convenience, because non-compliance in aerospace QC carries customer and regulatory consequences that no amount of good UI design can undo.

The recommended approach is a Python-native web stack (FastAPI + Jinja2 + HTMX) with a process-isolated job queue (Huey + SQLite), all delivered as a single Docker container with supervisord managing two processes: the uvicorn web server and the Huey worker. This stack makes zero external service dependencies — no Redis, no cloud, no external auth — which is required for air-gapped shop deployment. The existing pipeline is imported as a library by the worker; the pipeline code is never modified. All persistent state lives in a single SQLite file and a single Docker volume, making backup and migration trivially simple at shop scale (O(10) engineers, O(100s) runs/year).

The key risks fall into two categories: audit integrity failures and infrastructure reliability failures. Audit integrity risks (mutable signed records, silent review overrides, missing reviewer identity) must be addressed in the data model design before any review UI is built — they cannot be retrofitted. Infrastructure reliability risks (sign-off / PDF generation not being atomic, job queue race conditions, session state loss mid-review, disk accumulation from orphaned runs) are predictable and well-documented; each has a clear prevention pattern. The architecture research provides explicit, verified implementations for all of them.

## Key Findings

### Recommended Stack

The stack is fully Python-native with no build toolchain required. FastAPI 0.135.1 serves as the HTTP layer (async-native, Pydantic v2 already in the pipeline's dependency tree). Jinja2 + HTMX + Alpine.js provide a server-rendered UI with partial-page update capability — no SPA framework, no Node.js build step, no CORS. SQLite in WAL mode is the database; it is production-viable at this scale. Huey with SqliteStorage is the job queue (no Redis needed). WeasyPrint generates audit packet PDFs from the same Jinja2 templates used for the web UI. All dependencies are managed via the existing `uv` workflow.

Notably: passlib is explicitly excluded (unmaintained since 2020, breaks on Python 3.13); pwdlib 0.3.0 replaces it. SQLModel is excluded (performance and coupling concerns); raw SQLAlchemy 2.0 + separate Pydantic schemas is the correct pattern. FastAPI BackgroundTasks is excluded for pipeline execution — it blocks the event loop and loses tasks on restart.

**Core technologies:**
- FastAPI 0.135.1: HTTP framework, async-native, Pydantic v2 already a dependency — avoids conflict
- Jinja2 + HTMX + Alpine.js: server-rendered UI with partial updates — no build toolchain, right-sized for O(10) users
- SQLAlchemy 2.0 + Alembic: ORM and migrations — mature, battle-tested, migration-safe
- SQLite (WAL mode): primary database — zero-config, single-file, sufficient at shop scale
- Huey + SqliteStorage: job queue — no Redis dependency, sufficient throughput, crash-recoverable via supervisord
- WeasyPrint 68.1: HTML+CSS to PDF — reuses Jinja2 templates for both web and PDF output
- pwdlib 0.3.0 + python-jose 3.5.0: password hashing and JWT — current, maintained, OWASP-aligned
- Docker + supervisord: single container, two processes (uvicorn + huey worker), one volume mount

### Expected Features

The feature research draws a clear distinction between what is legally/contractually required for any FAIR tool (table stakes) and what differentiates this product from cloud competitors (differentiators). The most important table-stakes finding is that audit integrity features — mandatory override notes, immutable signed packets, individual reviewer identity on every record — are compliance requirements, not UX preferences. They must be treated as hard system constraints, not soft UX guardrails.

The primary competitive differentiator is the combination of air-gapped on-premises deployment (competitors Net-Inspect, GroundControl, InspectionXpert are all cloud-only) with visual evidence snippets per item and an explicit partial FAI work order output. The partial FAI work order (telling the shop exactly which characteristics need re-inspection after a revision) does not exist as a distinct output in any competitor tool — this is the core cost-saving claim.

**Must have (table stakes):**
- Per-item review card with Rev A + Rev B snippet images — visual evidence is audit-critical; image-free review is audit-worthless
- Mandatory non-empty override note — compliance requirement; silent overrides fail audit
- Hard gate: all items terminal before sign-off is available — partial sign-off creates invalid audit records
- Immutable signed packet with amendment model — AS9102 records must survive post-sign-off; overwrites are a compliance violation
- Reviewer identity (name + timestamp) on every audit record — AS9102 Form 3 accountability requirement
- Atomic sign-off + packet generation with rollback — if PDF fails, run must return to reviewable state
- Async pipeline with stage-by-stage progress — 8-stage CV pipeline takes 30–120 seconds; a spinner erodes engineer trust
- File upload with format validation (magic bytes, raster PDF detection, multi-page selector)
- User accounts with admin/engineer roles + session management
- Setup wizard (undismissable until complete) for shop config and Form 3 column mapping
- Server-side session persistence for partial review — QC engineers are frequently interrupted
- PDF + CSV audit packet export — the formal deliverable; PDF is what goes to the customer
- Partial FAI work order (PDF + CSV) on demand after finalization
- Run history with filter, reopen, and re-download
- Docker deployment for air-gapped LAN shops

**Should have (competitive differentiators):**
- Pipeline confidence score + reasons per item (already computed by pipeline; surfacing in UI is a display decision only)
- Confidence distribution warning screen before review starts
- Rev A vs Rev B requirement text diff inline in review card
- Amendment model with visible version trail
- Multi-page PDF page selector at upload
- Run assignment to non-submitter engineer
- Admin-configurable confidence threshold for auto-classification

**Defer (v2+):**
- GD&T semantic parsing — multi-month investment; string matching catches value changes; uncertain GD&T items route to manual review
- Cross-revision timeline view (Rev A→B→C history analytics)
- QMS direct API integration (ETQ, IQS) — CSV export is the v1 integration surface
- Email notifications — in-app alerts sufficient at shop scale
- OAuth/SSO — incompatible with air-gapped constraint

### Architecture Approach

The architecture is a single Docker container running two supervised processes: a FastAPI ASGI server handling HTTP and an Huey worker consuming pipeline jobs. The pipeline is always called from the worker process — never from the web process. All persistent state is in one SQLite file and one filesystem volume. The data flow is strictly linear: upload → queue → pipeline execution → review → sign-off → export, with each stage gated by completion of the previous. This is a well-understood pattern for wrapping blocking compute pipelines in async web servers, and the research provides complete, tested implementations for each integration point.

**Major components:**
1. FastAPI app — HTTP routing, auth, SSE progress streams, static file serving; thin routers delegating to services
2. Huey job queue (SqliteStorage) — decouples HTTP from blocking compute; holds task state across restarts
3. Pipeline worker (separate process) — calls `run_pipeline()` synchronously; writes output to filesystem; updates run status in DB
4. SQLite database — authoritative state for runs, review items, users, sessions, audit events, signed packets
5. Filesystem volume — write-once artifact store: uploads, pipeline output (snippets/JSON), signed exports
6. WeasyPrint export service — renders audit packet and work order PDFs from Jinja2 templates + snippet images

**Key patterns:**
- Huey + SqliteStorage as the pipeline bridge (not FastAPI BackgroundTasks, not Redis-backed Dramatiq — SQLite-only, no external broker)
- SSE for pipeline progress streaming (FastAPI 0.135+ native `EventSourceResponse`)
- Review state machine in SQLite with service-layer transition validation
- Write-once filesystem for all artifact outputs — never modify in place; amendments create new files under new run_id
- SQLite WAL mode required from day one (two processes sharing one DB file)

### Critical Pitfalls

1. **Sign-off / PDF generation not atomic** — Write `signing_in_progress`, generate PDF to temp path, atomically move + flip to `signed` in one transaction; if WeasyPrint fails, roll back status to `reviewable`. Never set `signed` before the packet file exists.

2. **Job dispatched inside open DB transaction** — Always enqueue the Huey task after the transaction commits (post-commit hook or transactional outbox). Never call `huey.enqueue(task, run_id)` inside a `with session.begin():` block; the worker can pick up the job before the run record is visible.

3. **FastAPI BackgroundTasks for CPU-bound pipeline work** — BackgroundTasks shares the event loop with the web server. The CV pipeline (ORB, RANSAC, PyMuPDF) holds the GIL and blocks all HTTP handling. Use Huey worker in a separate process.

4. **Review state lost on session expiry** — Persist every approve/override decision to the DB immediately on user action (per-decision PATCH endpoint), not on form submit. Use a sliding session window sized for realistic review sessions (4 hours). Show auto-save confirmation after each decision.

5. **Soft audit immutability (application-only enforcement)** — Application-level refusal to update signed rows is insufficient; a direct SQL UPDATE bypasses it. Revoke UPDATE/DELETE on audit-sensitive tables from the application DB role. Use an append-only `audit_events` table. Validate at the DB constraint layer, not only in service code.

6. **Uploaded files stored with user-provided filenames** — Path traversal and concurrent-run name collisions. Always store uploads as `uploads/{run_id}/{uuid}.original_ext`; record original filename in DB only; validate type by magic bytes.

7. **Pipeline run artifacts accumulate until disk is full** — Implement cleanup from first deployment: delete failed/abandoned run artifacts after admin-configured N days; delete intermediate render images (300 DPI page renders) after packet generation; log disk usage on each cleanup run.

## Implications for Roadmap

The architecture research explicitly derives a build order from hard dependency analysis. The phase structure below follows it directly, enriched by feature and pitfall findings.

### Phase 1: Foundation — Infrastructure and Identity

**Rationale:** Auth and database schema must come first. Every subsequent feature requires authenticated sessions, and the audit data model (append-only events, immutable signed-packet rows, DB-level access restrictions) must be designed before any data is written — it cannot be retrofitted without data migration.

**Delivers:** Running Docker container with FastAPI + SQLite + auth; admin and engineer accounts; setup wizard; schema with audit-safe constraints from day one.

**Addresses:** User accounts, role-based access, setup wizard (Form 3 column mapping), Docker deployment.

**Avoids:** Pitfall 5 (soft immutability) — DB-level access restrictions and append-only `audit_events` table designed here, before any run data exists.

**Research flag:** Standard patterns — well-documented FastAPI auth, SQLAlchemy schema design, Docker + supervisord. No additional research phase needed.

### Phase 2: Pipeline Bridge — Job Queue and Upload

**Rationale:** The review queue cannot be built until real pipeline output exists to review. The job queue infrastructure (Huey + SqliteStorage + supervisord worker) and file upload endpoint must be built and verified before the review UI.

**Delivers:** File upload endpoint; per-run UUID-namespaced storage; Huey task queue with pipeline worker; SSE progress streaming; run status transitions (queued → running → review_pending / failed); in-app failure alerts.

**Addresses:** Async pipeline execution with stage-by-stage progress, file upload with validation (magic bytes, raster PDF detection, multi-page selector), in-app run failure alerts.

**Avoids:** Pitfall 2 (job dispatched inside DB transaction — post-commit enqueue pattern), Pitfall 3 (BackgroundTasks for CPU-bound work), Pitfall 6 (user-provided upload filenames).

**Research flag:** Standard patterns — Huey SqliteStorage, FastAPI SSE, multipart upload. Architecture research provides complete implementation sketches.

### Phase 3: Review Queue — Core UX and Sign-Off

**Rationale:** The review UI is the product's core value delivery. It depends on real pipeline output from Phase 2. Sign-off atomicity must be built here — it cannot be bolted on after the fact — and the full state machine (review item transitions, sign-off gate, atomic PDF generation with rollback) belongs in this phase.

**Delivers:** Per-item review card with Rev A + Rev B snippets, system classification, confidence score with reasons, requirement text; Approve / Override controls with mandatory non-empty note; server-side persistence of every decision immediately on action; hard gate (sign-off blocked until all items terminal); atomic sign-off + PDF/CSV audit packet generation with rollback on WeasyPrint failure; amendment model for post-sign-off corrections.

**Addresses:** Per-item review card, mandatory override note, all-items gate, immutable signed packet, reviewer identity in audit records, atomic sign-off, session persistence for partial review, PDF/CSV audit packet.

**Avoids:** Pitfall 1 (sign-off atomicity — two-phase write with rollback), Pitfall 4 (review state lost on session expiry — per-decision persistence), Pitfall 9 (concurrent review lost updates — optimistic locking with version column), Pitfall 10 (override note whitespace bypass — minimum length after strip + inline validation).

**Research flag:** Needs deeper research on WeasyPrint behavior with real engineering drawing snippets (multi-page, large PNGs). Recommend testing WeasyPrint with actual pipeline output before finalizing the PDF template design.

### Phase 4: Export, History, and Admin

**Rationale:** Partial FAI work order, run history, and admin panel all depend on finalized runs from Phase 3. They are independent of each other within this phase and can be parallelized.

**Delivers:** Partial FAI work order (PDF + CSV, post-finalization on demand); run history with filter by part number and date, reopen, and re-download; amendment model with version trail surfaced in UI; admin panel (user management, shop config, retention policy, disk usage monitoring); cleanup job for orphaned run artifacts.

**Addresses:** Partial FAI work order, run history, amendment model, role-based admin, data retention, disk usage management.

**Avoids:** Pitfall 7 (artifact accumulation — cleanup job with admin-configurable retention implemented here before first production use).

**Research flag:** Partial FAI work order format may benefit from validation against a real aerospace QC engineer. The specific fields and layout are inferred from AS9102C requirements — confirm against an actual customer-facing work order format during pilot.

### Phase 5: Differentiators — UX Enhancements

**Rationale:** These features are high-value differentiators that are independent of core compliance functionality and should be added after pilot validation confirms the v1 workflow is correct.

**Delivers:** Confidence distribution warning screen before review starts; Rev A vs Rev B requirement text diff inline in review card; run assignment to non-submitter engineer; admin-configurable confidence threshold for auto-classification; multi-page PDF page selector at upload.

**Addresses:** P2 features from the feature prioritization matrix.

**Research flag:** Text diff rendering for GD&T-heavy requirement strings (e.g., stacked tolerances) may need exploration. Standard Python `difflib` may produce confusing diffs on formatted tolerance text — worth a spike before committing to the implementation approach.

### Phase Ordering Rationale

- Auth must precede everything — no endpoint has value without identity.
- DB schema with audit constraints must precede any data write — retrofitting immutability is expensive and risky.
- Pipeline bridge must precede review — the review UI needs real pipeline output to display; building it against mocked data defers the hardest integration problems.
- Sign-off atomicity must be designed in Phase 3, not added in Phase 4 — the two-phase write pattern is a schema and service design decision, not a feature toggle.
- Cleanup job must ship before first production use (Phase 4) — disk accumulation is silent and catastrophic; deferring it to "later" is a pattern that consistently results in production incidents.
- Differentiators are last because they deliver no compliance value and their implementation quality depends on feedback from the core review workflow in pilot use.

### Research Flags

Phases needing additional research during planning:
- **Phase 3 (Review + Sign-off):** WeasyPrint behavior with real engineering drawing snippets. The pipeline produces PNG crops of 300 DPI engineering drawings — large files with technical line art. WeasyPrint's rendering of these in a PDF template at real part complexity has not been tested. A proof-of-concept before template design is strongly recommended.
- **Phase 4 (Partial FAI Work Order):** Format and field requirements for the partial FAI work order output. The feature exists as a differentiator without a directly comparable prior art in competitor tools. Field content and structure should be validated with a real QC engineer during pilot.

Phases with standard patterns (skip additional research phase):
- **Phase 1 (Foundation):** FastAPI auth, SQLAlchemy schema, Docker + supervisord — extensively documented with official sources.
- **Phase 2 (Pipeline Bridge):** Huey SqliteStorage, FastAPI SSE, multipart upload — architecture research provides complete, verified implementation sketches.
- **Phase 5 (Differentiators):** All are incremental additions to an established UI pattern; no novel integration problems.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions verified against PyPI; alternatives explicitly researched and ruled out with rationale; version compatibility matrix provided |
| Features | MEDIUM | Table-stakes and differentiators verified against three competitor products (Net-Inspect, GroundControl, InspectionXpert) and regulatory references (21 CFR Part 11, AS9102C); anti-features drawn from adjacent domains with lower direct citation |
| Architecture | HIGH | Patterns verified against official FastAPI, Huey, SQLAlchemy, SQLite docs; complete implementation sketches provided for all critical integration points; anti-patterns documented with specific sources |
| Pitfalls | MEDIUM-HIGH | Core pitfalls (BackgroundTasks blocking, job-in-transaction race, sign-off atomicity, session loss) verified via official community reports and practitioner post-mortems; aerospace-specific compliance pitfalls extrapolated from AS9102C and adjacent QMS domain research |

**Overall confidence:** HIGH

### Gaps to Address

- **WeasyPrint with real pipeline output:** PDF generation is verified against WeasyPrint docs and simple examples. Behavior with 50–200 KB engineering drawing PNG crops in a multi-item template is unverified. Validate this before finalizing the Phase 3 audit packet template design — a rendering failure at sign-off is the highest-severity recoverable pitfall.

- **Partial FAI work order field format:** The work order is described as a differentiator but has no direct competitor analog. The specific fields (char number, requirement, drawing reference, priority flag) are inferred from AS9102C section 4.6. A real QC engineer should validate the format before Phase 4 implementation.

- **Pipeline stage progress reporting:** The existing `run_pipeline()` function does not currently emit stage-by-stage progress events. The architecture assumes progress can be reported by the worker via DB updates (e.g., `Run.stage = "alignment"`). This requires either modifying the pipeline to accept a progress callback or polling a known output structure. Confirm the approach during Phase 2 design.

- **Session timeout for review sessions:** Research recommends a 4-hour sliding session window for review sessions. This is longer than typical security guidance (30 minutes). The right balance between security and usability for a LAN-deployed shop tool should be confirmed with actual users during pilot; 4 hours is a starting recommendation, not a verified best practice for this specific deployment context.

## Sources

### Primary (HIGH confidence)
- FastAPI official docs (fastapi.tiangolo.com) — framework capabilities, Jinja2/SSE/file upload/auth patterns
- PyPI package pages — all version numbers verified: FastAPI 0.135.1, SQLAlchemy 2.0.47, Alembic 1.18.4, Uvicorn 0.41.0, Gunicorn 25.1.0, Dramatiq 2.0.1, WeasyPrint 68.1, pwdlib 0.3.0, python-jose 3.5.0, python-multipart 0.0.22, redis 7.2.1, Jinja2 3.1.6
- SQLite official documentation (sqlite.org) — WAL mode, production suitability guidance
- Dramatiq official docs (dramatiq.io) — async middleware, Redis broker
- HTMX 2.0.8 (htmx.org) — CDN version, IE11 drop confirmed
- Alpine.js 3.15.8 (alpinejs.dev) — CDN version
- Huey docs (huey.readthedocs.io) — SqliteStorage API, supervisor integration
- OWASP File Upload Cheat Sheet — file upload security patterns
- FastAPI BackgroundTasks blocking issue (GitHub Discussion #11210) — CPU-bound blocking confirmed
- Confluent: The Dual-Write Problem — job-in-transaction race condition patterns
- SQLAlchemy SQLite WAL + aiosqlite docs — WAL mode pragma implementation
- FastAPI Docker deployment docs — Gunicorn + Uvicorn worker pattern

### Secondary (MEDIUM confidence)
- JetBrains 2025 framework comparison — FastAPI positioning vs Django/Flask
- Net-Inspect, GroundControl, InspectionXpert feature pages — competitor feature analysis
- 21 CFR Part 11 compliance summaries — electronic record / audit trail requirements
- Leapcell: FastAPI async task management pitfalls — BackgroundTasks behavior
- YunoJuno: Django-RQ race conditions post-mortem — job-in-transaction race confirmation
- WeasyPrint dev.to comparison article — WeasyPrint vs ReportLab for structured PDFs
- Better Stack: Docker image size reduction guide — multi-stage build patterns
- binaryigor: Optimistic vs pessimistic locking — version column pattern
- WorkOS: Session management best practices — sliding session window guidance
- Nielsen Norman Group: Designing for waits and interruptions — progress indicator UX guidance
- AS9102 partial FAI standards (as9100store.com) — characteristic accountability scope
- Ideagen AS9102 — audit trail documentation patterns
- Immutable audit log patterns (hubifi.com, designgurus.io) — append-only storage, hash chain

### Tertiary (LOW confidence — needs validation)
- Partial FAI work order field format — inferred from AS9102C section 4.6 structure; no direct published example found
- 4-hour session timeout recommendation — extrapolated from QC workflow interruption patterns; not a verified standard for shop-LAN tools

---
*Research completed: 2026-03-01*
*Ready for roadmap: yes*

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

# Feature Research

**Domain:** Document review and sign-off web application for aerospace QC (AS9102 FAIR compliance)
**Researched:** 2026-03-01
**Confidence:** MEDIUM — core audit/sign-off patterns verified against industry tools (GroundControl, Net-Inspect, InspectionXpert) and regulatory references (21 CFR Part 11, AS9102C); specific UX anti-patterns drawn from adjacent domains (content moderation, pharma QMS, eQMS) with lower direct citation

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete or legally insufficient.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Per-item review card with visual evidence | QC engineers cannot classify without seeing what they're approving; image-free review is audit-worthless | MEDIUM | Side-by-side Rev A / Rev B snippets are the core interaction; layout must be unambiguous about which is which |
| Mandatory reason on override | Every FAI-aware QC tool requires documented justification for any deviation from the system's finding; silent overrides fail audit | LOW | Single required text field per override; must block submit if empty |
| All items must reach terminal state before sign-off | Partial sign-off creates ambiguous coverage claims; industry standard for FAIR is 100% characteristic accountability | MEDIUM | Hard gate, not a soft warning; show remaining count prominently |
| Immutable signed packet | Post-sign-off modification of a FAIR is a compliance violation in aerospace QMS; records must be tamper-evident | HIGH | Append-only audit log + amendment model; original record must survive amendments |
| Reviewer identity in every audit record | AS9102 Form 3 requires accountable reviewer identity; anonymous approvals are non-compliant | LOW | Name + timestamp on every approve/override action; sourced from authenticated session |
| Run history with reopen and download | QC teams regularly reference prior FAIRs during audits and corrective actions | MEDIUM | Filterable by part number and date; download must re-export the exact signed packet, not regenerate |
| Role-based access (admin vs engineer) | Shop config (column mapping, retention) must be protected; engineer-level changes to quality decisions must be traceable | LOW | Two roles sufficient for v1; admin manages accounts and shop settings, engineer reviews and signs |
| File upload with format validation | Users upload wrong file types regularly; a raster PDF or wrong Excel layout must be caught before compute is wasted | LOW | Validate file type at upload; detect raster vs vector PDF before job submission |
| Async pipeline with stage-by-stage progress | An 8-stage pipeline running computer vision on a large PDF will take 30–120 seconds; a spinner with no feedback erodes trust | MEDIUM | Show stage name + completion count per stage; distinguish queued / running / failed states |
| In-app alert on run failure | Engineers need to know when a run fails without polling the history page | LOW | In-app notification when logged in; no email needed for v1 |
| Session persistence for partial review | A 50-item queue cannot be completed in one sitting; losing progress on browser close is unacceptable | MEDIUM | Persist approve/override state server-side as each decision is made; allow resume from any device |
| Setup wizard for first admin login | If the shop config (Excel column mapping, admin account) can be skipped, the system produces garbage output silently | MEDIUM | Wizard must be undismissable until complete; cover shop name, admin password, first engineer account, Form 3 column mapping |
| PDF + CSV export of audit packet | QC engineers deliver FAIR packages to customers; PDF is the expected format; CSV enables QMS import | HIGH | PDF must include cover page with part/revision/reviewer/timestamp, summary table, per-item cards with inline snippets; CSV for programmatic use |
| Partial FAI work order output | The primary cost-saving claim of the product — telling the shop which characteristics need re-inspection — must be exportable | MEDIUM | PDF + CSV; contains char number, requirement, drawing reference, priority flag |
| Atomic sign-off + packet generation | If packet generation fails after sign-off is recorded, the run is in an ambiguous state that cannot be resolved without manual intervention | MEDIUM | Roll back sign-off if PDF generation fails; surface clear error; run stays in reviewable state |

---

### Differentiators (Competitive Advantage)

Features that set this product apart. Not expected by default, but high value for the domain.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Pipeline confidence score per item with reasons | Existing FAI tools show results, not reasoning; surfacing "matched via location score 0.91, text overlap 0.87" gives the engineer the right frame for disagreeing or agreeing | LOW | Already produced by the pipeline; surfacing in UI is a display decision, not new computation |
| Auto-classification with confidence-gated manual escalation | System auto-classifies high-confidence items; only uncertain items require active review; reduces queue from N to uncertain-N | MEDIUM | Threshold configurable by admin; transparent about what was auto-classified and why; distinguishes "auto-classified" from "reviewed by engineer" in audit record |
| Rev A vs Rev B annotation diff inline in review card | Engineers can see exactly which text tokens changed (requirement text A vs B) rather than free-reading two snippet images | MEDIUM | Text diff on requirement string; not a pixel diff; highlights added/removed/changed substrings |
| Confidence distribution warning before review | Low alignment confidence across a run is more useful to surface before review starts than per-item; prevents engineer from reviewing 50 items then discovering the homography was garbage | LOW | Pre-review summary screen showing confidence histogram; explicit "proceed" / "abort" decision |
| Amendment model with version trail | Most FAI tools either lock records completely or allow silent overwrites; an explicit amendment model with preserved original is audit-correct and user-visible | MEDIUM | Show version history on any finalized run; amendment creates new versioned packet, original is locked |
| Air-gapped / on-premises deployment | Net-Inspect and GroundControl require cloud access; ITAR-sensitive shops cannot use cloud-hosted tools; Docker on shop server removes the dependency | HIGH | All dependencies bundled in Docker image; no external API calls at runtime; documented pull-based update path |
| Shop-owned data with configurable retention | Cloud tools hold customer data; a shop-hosted tool means the shop controls what survives and when it is deleted | LOW | Admin sets retention period for unfinished/failed runs; finalized records retained indefinitely by default |
| Explicit partial FAI scope output | Existing FAI software completes FAIRs end-to-end; the work order that says "only re-inspect characteristics 4, 7, 23" does not exist as a distinct output in competitor tools | MEDIUM | Separate action from sign-off; customer-facing format with drawing reference and priority |
| Multi-page PDF page selector at upload | Most tools assume single-page drawings; aerospace drawings are commonly multi-page; page selection at upload time prevents a wrong-page run | LOW | Detect multi-page at upload; show page thumbnails with selection before job submission |
| Duplicate balloon / count token tracking | "4X Ø 0.5" is one balloon but four Form 3 rows; treating count changes as a classification signal catches a class of changes competitors miss | MEDIUM | Already in pipeline; surfaced in UI as count token diff on changed items |

---

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem helpful but undermine audit integrity or user trust.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Bulk approve / select-all approve | Speeds up review when engineer "knows" items are all fine | Destroys individual accountability; audit trail shows a single approval action for N items; removes the forced pause per item that catches errors; 21 CFR Part 11 analog tools explicitly prohibit this pattern | Surface items grouped by system classification; let engineer move quickly through a sorted queue; keyboard shortcuts for approve/next acceptable if each item is individually confirmed |
| Auto-sign-off when all items are approved | One fewer click; feels like a natural workflow completion | Conflates "review complete" with "sign-off authorized"; sign-off is a formal attestation requiring deliberate intent; unintended sign-off creates immutable records that require amendments to correct | Explicit sign-off button that is enabled only when queue is fully resolved; clear label "Sign and finalize — this creates an immutable audit record" |
| Editable requirement text in the review UI | Engineer sees typo or OCR error in requirement text and wants to fix it | Editing requirements in the review interface creates ambiguity about what was actually on the drawing vs what the system read; requirement text should reflect what the pipeline extracted from the actual document | Display raw extracted text alongside the Form 3 requirement; if the pipeline extracted wrong text, that is a pipeline issue to fix in a new run, not an in-review edit |
| Email notifications for every action | Looks like a "complete" feature; QMS tools often include this | Adds external dependency (SMTP server, email deliverability) disproportionate to v1 scale (O(10) engineers, O(100s) runs/year); in-app alerts are sufficient for same-team use | In-app notification panel; email deferred to v1.x when pilot validation confirms need |
| Concurrent multi-user review of same run | Looks like a collaboration feature; teams may want parallel review | State conflicts when two engineers approve the same item simultaneously require locking or merge logic; one reviewer per run keeps the audit record unambiguous about who reviewed what | Run assignment model; submitter defaults as reviewer; can explicitly reassign to another engineer |
| Real-time collaborative cursor / annotation on snippets | Borrowed from design review tools (Figma, Drawboard); looks premium | Adds WebSocket complexity for no audit value; aerospace QC review is an individual attestation, not a collaborative annotation session | Override note field captures the engineer's reasoning; comments belong in the audit record, not on the image |
| OAuth / SSO login | Looks enterprise-ready | SSO requires external identity provider which may not exist at a small machine shop; adds dependency, config complexity, and security surface area for marginal benefit at O(10) user scale | Simple email + password accounts; admin manages accounts; no external dependencies |
| Auto-update via internet pull | Looks like good DevOps hygiene | Air-gapped shops cannot reach the internet; auto-update breaks the no-external-dependency constraint; unexpected updates in a production quality environment are a compliance risk (what changed in the tool?) | Documented manual `docker compose pull` process; change log published per release; update is a shop IT decision, not automatic |
| GD&T semantic parser in v1 | Engineers want symbol/datum/modifier interpretation, not string matching | Semantic GD&T parsing is a multi-month engineering investment; string matching on tolerance values catches the most consequential changes (value changes); semantic failures surface as uncertain items routed to manual review | Opaque string matching in v1; flag uncertain GD&T items clearly; defer semantic parsing to v2 after pilot validates scope |

---

## Feature Dependencies

```
User accounts (admin + engineer)
    └──requires──> Role-based access control
                       └──requires──> Session management / auth

File upload with validation
    └──requires──> User accounts
    └──enables──> Async pipeline execution

Async pipeline execution
    └──requires──> File upload
    └──enables──> Review queue (pipeline output is queue input)
    └──requires──> Stage-by-stage progress display

Review queue
    └──requires──> Async pipeline execution (delta_packet.json + snippets)
    └──requires──> Session persistence
    └──requires──> Mandatory override notes
    └──enables──> Sign-off

Sign-off
    └──requires──> Review queue fully resolved (hard gate)
    └──requires──> Atomic packet generation
    └──enables──> Run history / download
    └──enables──> Amendment model

Audit packet (PDF + CSV)
    └──requires──> Sign-off trigger
    └──requires──> Snippet images from pipeline run

Partial FAI work order
    └──requires──> Finalized run (signed)
    └──independent from──> Audit packet (separate action)

Amendment model
    └──requires──> Immutable original signed packet
    └──requires──> Finalized run

Run history
    └──requires──> Completed runs (any state)
    └──requires──> User accounts (filter by user / assignment)

Setup wizard
    └──requires──> Admin account creation (bootstraps system)
    └──enables──> Excel column mapping
    └──enables──> Engineer account creation

Air-gapped deployment
    └──conflicts──> OAuth / SSO
    └──conflicts──> Auto-update
    └──conflicts──> Email notifications (SMTP dependency)
```

### Dependency Notes

- **Review queue requires pipeline output:** The review queue cannot be populated until the 8-stage pipeline completes and writes `delta_packet.json` + snippet images. Stage-by-stage progress is required to make the wait tolerable.
- **Sign-off requires atomic packet generation:** Packet generation failure must roll back the sign-off state; these two operations must execute in a single transaction boundary.
- **Amendment model requires immutable originals:** Amendment is meaningless without a preserved, locked original. The immutability constraint must be designed into the data model before sign-off is built, not added after.
- **Setup wizard enables column mapping:** Without Excel column mapping, the pipeline will fail to parse Form 3 correctly on every run. The wizard is not optional UX polish — it's a prerequisite for any run succeeding.
- **Air-gapped deployment conflicts with external dependencies:** OAuth/SSO, email notifications, and auto-update all introduce outbound network calls. These are incompatible with the air-gapped shop network constraint and must be explicitly excluded.

---

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed for a QC team to use this in their actual workflow and produce a legally recognizable audit artifact.

- [ ] User accounts (admin + engineer roles) with session management — without identity, audit records have no value
- [ ] Setup wizard (shop config, Form 3 column mapping, engineer accounts) — without this, no run will parse correctly
- [ ] File upload with validation (raster PDF detection, multi-page selector, Excel format check) — garbage in = garbage out
- [ ] Async pipeline execution with stage-by-stage progress display — computer vision pipeline takes time; progress required for trust
- [ ] Review queue: per-item card with Rev A / Rev B snippets, system classification, confidence score, requirement text, Approve / Override controls, required override note — this is the core UX
- [ ] Session persistence for partial review — engineers cannot complete a 50-item queue in one sitting
- [ ] Hard gate: all items terminal before sign-off available — partial sign-off creates invalid audit records
- [ ] Atomic sign-off + PDF/CSV audit packet generation (with rollback on failure) — the product output; what gets delivered to the customer
- [ ] Immutable signed packet; amendment model for corrections post-sign-off — audit records must survive
- [ ] Partial FAI work order (PDF + CSV) generated on demand after finalization — the primary cost-saving output
- [ ] Run history with filter, reopen, and re-download — teams reference past FAIRs during audits
- [ ] In-app alert on run failure — engineers need failure notification without polling
- [ ] Docker deployment via `docker-compose.yml` — the only viable delivery mechanism for air-gapped shops

### Add After Validation (v1.x)

Features to add once the pilot validates the core workflow.

- [ ] Confidence distribution warning screen before review starts — useful but not blocking for v1; engineers can read per-item confidence scores
- [ ] Rev A vs Rev B requirement text diff inline in review card — reduces review time but engineers can read two text strings without highlighting
- [ ] Run assignment to non-submitter engineer — submitter-as-reviewer is sufficient for pilot; assignment needed when shop has dedicated QC reviewers vs job submitters
- [ ] Admin-configurable confidence threshold for auto-classification — 0.9 hardcoded is reasonable for pilot; configurability deferred until pilot shows the threshold needs tuning

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] GD&T semantic parsing (feature control frame symbol/datum/modifier interpretation) — significant engineering investment; string matching is sufficient for pilot
- [ ] Cross-revision timeline view (Rev A→B, B→C history) — analytics feature; individual run comparisons sufficient for v1
- [ ] QMS integration (ETQ, IQS, direct API connectors) — CSV export is the v1 integration surface; direct connectors require per-customer work
- [ ] Email notifications — in-app alerts sufficient for small shop scale; add when shop grows or pilot asks for it
- [ ] OAuth / SSO — incompatible with air-gapped constraint; only relevant if shops move to cloud-hosted deployment

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Per-item review card with snippets | HIGH | MEDIUM | P1 |
| Mandatory override note | HIGH | LOW | P1 |
| All-items gate before sign-off | HIGH | LOW | P1 |
| Immutable signed packet | HIGH | HIGH | P1 |
| Reviewer identity in audit record | HIGH | LOW | P1 |
| Atomic sign-off + packet generation | HIGH | MEDIUM | P1 |
| Async pipeline with stage progress | HIGH | MEDIUM | P1 |
| File upload with validation | HIGH | LOW | P1 |
| User accounts + roles | HIGH | MEDIUM | P1 |
| Setup wizard | HIGH | MEDIUM | P1 |
| Session persistence for partial review | HIGH | MEDIUM | P1 |
| PDF + CSV audit packet | HIGH | HIGH | P1 |
| Partial FAI work order | HIGH | MEDIUM | P1 |
| Run history + reopen + download | MEDIUM | MEDIUM | P1 |
| In-app run failure alert | MEDIUM | LOW | P1 |
| Docker deployment | HIGH | LOW | P1 |
| Confidence score + reasons per item | HIGH | LOW | P2 |
| Confidence distribution pre-review warning | MEDIUM | LOW | P2 |
| Amendment model with version trail | HIGH | MEDIUM | P2 |
| Rev A vs Rev B text diff inline | MEDIUM | MEDIUM | P2 |
| Multi-page PDF page selector | MEDIUM | LOW | P2 |
| Run assignment to other engineer | MEDIUM | LOW | P2 |
| Admin-configurable confidence threshold | LOW | LOW | P2 |
| GD&T semantic parsing | HIGH | HIGH | P3 |
| Cross-revision timeline view | LOW | HIGH | P3 |
| QMS direct API integration | MEDIUM | HIGH | P3 |
| Email notifications | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

---

## Competitor Feature Analysis

| Feature | Net-Inspect | GroundControl | InspectionXpert | Delta Preservation |
|---------|-------------|---------------|-----------------|-------------------|
| Electronic sign-off on Form 3 | Yes | Yes | Yes | Yes — with atomic packet generation |
| Per-characteristic color-coding | Yes (out-of-tolerance) | Yes | Yes | Yes — plus confidence score and reasons |
| Auto-balloon extraction | Partial (requires 2D/3D source) | Yes (AI-assisted) | Yes | Yes — existing pipeline |
| Audit trail | Basic | Basic | Basic | Append-only, tamper-evident |
| Amendment / version history | Unknown | Unknown | Unknown | Explicit version trail |
| Air-gapped / on-premises | No (cloud) | No (AWS GovCloud) | No (cloud) | Yes — Docker on shop server |
| Partial FAI work order output | No | No (full FAI focus) | No | Yes — explicit scope output |
| Visual evidence snippets per item | No | No | No | Yes — Rev A + Rev B PNG crops |
| Session persistence for partial review | Basic | Basic | Basic | Server-side per-decision persistence |
| Setup wizard for first-time config | Unknown | Unknown | Unknown | Yes — undismissable until complete |
| Required override justification | Unknown | Unknown | Unknown | Yes — mandatory field, no silent overrides |

---

## What QC/Audit-Focused Tools Get Right That General-Purpose Apps Miss

Based on research across FAI software (GroundControl, Net-Inspect, InspectionXpert), pharma QMS tools (SimplerQMS, QT9), and audit trail standards (21 CFR Part 11, AS9102C):

**1. Identity is permanent, not optional.** Every action in an audit-worthy record must carry who did it and when. General-purpose apps treat identity as a filter; QC tools treat it as the primary key of every record.

**2. Reversibility is controlled, not free.** General-purpose apps allow edit/delete liberally. QC tools allow amendments but never overwrites. The original record must survive any correction.

**3. Required fields enforce compliance, not just UX.** In general apps, required fields are UX guardrails. In QC tools, an empty override justification is a compliance failure, not a missing input. The system must treat them differently — blocking, not warning.

**4. Progress must be conservative.** General-purpose apps show optimistic progress ("Almost done!"). QC tools must show conservative, factual progress ("3 of 50 items confirmed; 47 pending"). Overconfident UI leads engineers to sign off prematurely.

**5. Slow actions need explanation.** Computer vision pipelines take time. Without stage-by-stage progress, engineers assume the system crashed and submit again. Specific microcopy ("Running cross-revision alignment — step 5 of 8") prevents double-submission.

**6. Export format is not a bonus feature.** In general apps, PDF export is nice to have. In aerospace QC, the PDF is the deliverable — what goes to the customer and what survives an audit. It must carry formal structure (cover page, reviewer identity, immutable timestamp), not just a print of the screen.

---

## Sources

- [Net-Inspect FAI Software](https://www.net-inspect.com/solutions/first-article-inspection-software/) — feature set, industry market position
- [GroundControl AS9102 Software](https://www.gndctl.com/) — AI-supporting-humans workflow model, ITAR/compliance positioning
- [InspectionXpert AS9102](https://www.inspectionxpert.com/fai/as9102) — competitor feature comparison
- [Ideagen AS9102](https://www.ideagen.com/standards/as9102) — audit trail documentation patterns
- [21 CFR Part 11 compliance requirements](https://www.cognidox.com/blog/what-is-fda-21-cfr-part-11) — electronic record / e-signature requirements informing audit trail design
- [Immutable Audit Log patterns](https://www.hubifi.com/blog/immutable-audit-log-basics) — append-only storage, hash chain verification
- [UI patterns for async workflows](https://blog.logrocket.com/ui-patterns-for-async-workflows-background-jobs-and-data-pipelines/) — stage-by-stage progress display, partial failure handling
- [Electronic signature audit trail](https://www.gonitro.com/resources/esignature-audit-trials) — immutability requirements, capture what/who/when/why
- [AS9102 partial FAI standards](https://as9100store.com/aerospace-standards-explained/what-is-as9102-first-article-inspection/) — characteristic accountability, partial FAI re-accomplishment scope
- [Quality audit best practice](https://www.compliancequest.com/blog/quality-audit-best-practice/) — documented justification requirements for corrective actions

---
*Feature research for: Document review and sign-off web application (aerospace QC, AS9102 FAIR compliance)*
*Researched: 2026-03-01*

# Pitfalls Research

**Domain:** Web application wrapping a Python compute pipeline for aerospace quality audit (AS9102 FAIR compliance)
**Researched:** 2026-03-01
**Confidence:** MEDIUM-HIGH (core pitfalls verified across multiple sources; aerospace-specific extrapolations from domain context)

---

## Critical Pitfalls

### Pitfall 1: Sign-Off State / PDF Generation Are Not Atomic

**What goes wrong:**
The sign-off action writes the reviewer identity and timestamp to the database, then calls the PDF generator. If PDF generation fails (OOM, missing font, corrupt snippet image, WeasyPrint CSS bug on a real drawing), the database already shows `signed`. The run is now in a permanently broken state: the audit record says it is signed but there is no packet, and the engineer cannot re-sign because the system thinks it is done. Recovery requires direct database edits, which themselves are an audit integrity problem.

**Why it happens:**
Developers treat the database write and the file system write as two cheap operations and do them sequentially without a rollback plan. PDF generation feels like a side effect, not a transaction participant. The issue is invisible during development because test PDFs with small, clean snippets never trigger edge-case rendering failures.

**How to avoid:**
Implement sign-off as a two-phase operation:
1. Write `status = "signing_in_progress"` and `reviewer_id` to the database in a transaction.
2. Generate the PDF to a temp path.
3. If generation succeeds: atomically move the file to its final path and update status to `signed` with the packet path in the same transaction.
4. If generation fails: update status back to `reviewable` and surface the error to the user.

Never set `status = signed` until the packet file exists at its expected path and has been fsync'd. Validate that the packet is readable (not just that the generator returned successfully) before committing the status flip. Implement a background integrity check that alerts the admin if any `signed` run lacks a readable packet file.

**Warning signs:**
- PDF generation is called outside a database transaction
- No rollback path in the sign-off endpoint
- Test suite only covers happy-path PDF generation
- No integrity check that `signed` runs have readable packet files
- Sign-off code does `db.update(status=signed)` before `pdf_generator.render()`

**Phase to address:** File upload / pipeline integration phase (before review UI); must be designed atomically from day one.

---

### Pitfall 2: Job Queue Tasks Dispatched Inside Open Database Transactions

**What goes wrong:**
The web handler opens a transaction, saves the run record (status `queued`), enqueues the pipeline job, then commits. The worker picks up the job before the transaction commits and cannot find the run record — it gets a "run not found" error and fails or retries indefinitely. This is the classic Django-RQ race condition: the queue message travels faster than the database commit.

**Why it happens:**
The ordering feels natural: save the thing, then tell the worker about it. Developers who are not deep in async queue semantics do not realize that the message broker and the database are separate systems with no shared transaction boundary. Under load (or on a fast machine), the worker always wins the race.

**How to avoid:**
Always enqueue after the transaction commits, never inside it. In Django, use the `transaction.on_commit()` hook:
```python
def save_run_and_enqueue(run):
    run.save()
    transaction.on_commit(lambda: enqueue_pipeline_job.delay(run.id))
```
In other frameworks, use explicit post-commit hooks or the Transactional Outbox pattern (write a `pending_jobs` table row in the same transaction; a separate poller picks it up after commit). Never call `queue.enqueue(job, run_id)` inside a `with transaction.atomic():` block.

**Warning signs:**
- Worker job dispatched inside a transaction block
- Intermittent "run not found" errors in worker logs under concurrent load
- Test suite uses SQLite (no real transaction isolation) and misses this race
- No `on_commit` hook or outbox table

**Phase to address:** Job queue infrastructure phase.

---

### Pitfall 3: FastAPI BackgroundTasks Used for the Pipeline (Event Loop Blocking)

**What goes wrong:**
FastAPI's built-in `BackgroundTasks` runs in the same event loop as the web server. The delta preservation pipeline is CPU-bound and I/O-heavy (PDF rendering, OpenCV, NumPy). Using `BackgroundTasks` for the pipeline blocks every other request while the job runs. With 2–3 concurrent pipeline runs (realistic for a shop with multiple engineers), the entire web application becomes unresponsive. Additionally, if the server restarts mid-job (Docker container restart, `docker compose pull` update), the background task disappears with no recovery path.

**Why it happens:**
`BackgroundTasks` is the most visible async mechanism in FastAPI docs and is correct for lightweight post-request work (sending an email, updating a counter). Developers reach for it because it is zero-infrastructure. The distinction between I/O-bound and CPU-bound is not obvious in FastAPI's documentation for the background tasks feature.

**How to avoid:**
Use a real worker process with persistent job state: Redis Queue (RQ) or Celery backed by Redis. The pipeline job must run in a separate process that does not share the event loop with the web server. The job record in the database is the source of truth for status — not in-memory state. Configure Redis with a non-evicting memory policy (do not use `allkeys-lru`) to prevent task loss under memory pressure.

**Warning signs:**
- Pipeline enqueued via `background_tasks.add_task(run_pipeline, ...)`
- Server becomes slow while a job is running
- No job status in the database (status only tracked in memory)
- Redis `maxmemory-policy` not set explicitly

**Phase to address:** Job queue infrastructure phase.

---

### Pitfall 4: Review State Lost on Session Expiry Mid-Review

**What goes wrong:**
A QC engineer is partway through reviewing 40 characteristics. A phone call interrupts them for 30 minutes. They return, click "Approve" on the next item — and the session has expired. The application redirects them to login. Upon return, all their in-progress review decisions are gone because they were stored in the server-side session (or in-memory) rather than persisted to the database. They restart the review from the beginning, furious.

**Why it happens:**
Developers persist review decisions only on "save" or "submit" button clicks. The implicit assumption is that users complete workflows in one sitting. QC engineers in a machine shop are frequently interrupted (floor calls, supervisor questions, part inspections). The session timeout is set at a security-sensible 30 minutes without considering the workflow context.

**How to avoid:**
Persist every review decision to the database immediately on user action — not on session save, not on form submit. Implement auto-save: each "Approve" or "Override" click fires a `PATCH /runs/{id}/items/{item_id}/decision` endpoint that writes immediately to the `review_decisions` table. Use optimistic locking on the decision record (version column) to detect double-writes. Extend the session sliding window to match the longest realistic review session (e.g., 4 hours), or use refresh-token-based auth that can renew transparently. Show a "saved" indicator after each decision so engineers trust the persistence.

**Warning signs:**
- Review decisions held in session or in-memory between form submissions
- No `review_decisions` table (decisions only written on final submit)
- Session timeout ≤ 30 minutes with no sliding window
- No auto-save feedback in the UI

**Phase to address:** Review UI phase.

---

### Pitfall 5: Audit Records Are Mutable — "Soft Immutability" That Fails Under a SQL Client

**What goes wrong:**
The application enforces immutability in code (the sign-off endpoint refuses to update a signed run), but the underlying database rows are fully mutable. A direct SQL UPDATE or a future migration that touches those rows silently corrupts the audit record. In an AS9102 audit, the customer or certifying body asks to see the original reviewed evidence — and the record no longer matches what was actually reviewed.

**Why it happens:**
Immutability feels handled once the application logic refuses updates. Developers do not think about the database layer, future migrations, or other processes that share the same database. The gap between application-level and database-level immutability is invisible until an audit challenge.

**How to avoid:**
Enforce immutability at the database level, not just in application code:
- For signed runs: revoke UPDATE and DELETE privileges on the `runs` table from the application database role; only allow INSERT and the specific status column transitions (via stored procedure or trigger).
- Append-only audit events: write all state transitions to an `audit_events` table (insert-only). Never update or delete audit events.
- Hash chaining: include a hash of the previous event in each audit event row; a broken chain is detectable.
- For amendments: write a new record referencing the original, never modify the original.
- Before every migration: check whether it touches `signed` run rows.

**Warning signs:**
- Application database user has full UPDATE/DELETE on run and decision tables
- No `audit_events` or equivalent append-only log table
- No migration guard against touching finalized records
- `signed` status enforced only in application code

**Phase to address:** Database schema phase (before any data is written).

---

### Pitfall 6: Uploaded Files Are Not Isolated Per Run — Path Traversal and Name Collisions

**What goes wrong:**
Two engineers submit runs for the same part number simultaneously. Both upload `revA.pdf`. The files land at `uploads/revA.pdf` and one overwrites the other. The first run picks up the second engineer's file, produces wrong results, and the audit record references a file that is not what was inspected. Alternatively, a crafted filename (`../../etc/passwd`) causes a path traversal write.

**Why it happens:**
Developers store uploaded files with the user-provided filename in a shared directory. This works fine in development with one developer and test files. The collision and security issues only surface with concurrent users using real filenames.

**How to avoid:**
Generate a UUID for every upload, store it at `uploads/{run_id}/{uuid}.pdf`, and record the original filename in the database as metadata only. Never use the user-provided filename in any filesystem path. Validate file type by reading the magic bytes (not just the extension or Content-Type header). Enforce a max upload size at both the web server (Nginx/Caddy) and application layers. Verify the resolved path stays within the upload root before writing.

**Warning signs:**
- Upload path includes any portion of the user-provided filename
- No per-run upload subdirectory
- File type validated only by extension or Content-Type header
- No max size limit enforced server-side

**Phase to address:** File upload phase.

---

### Pitfall 7: Pipeline Run Orphans Fill Disk Until the Server Dies

**What goes wrong:**
Every pipeline run produces: the original uploaded PDFs (2–5 MB each), rendered page images at 300 DPI (~20–50 MB per page), evidence snippet PNGs, debug JSON, and the final delta packet. Failed runs (balloon detection failure, alignment failure) accumulate all intermediate artifacts. After 100 failed or abandoned runs, the Docker volume is full and the server stops accepting uploads.

**Why it happens:**
Developers think about the happy path output (`out/<run_id>/delta_packet.json`) and not about intermediate artifacts. The admin-configurable retention policy is planned for later, and "later" gets deferred indefinitely. Docker volumes grow silently until a crisis.

**How to avoid:**
Implement a cleanup policy from the first deployment:
- Failed and abandoned runs: delete all artifacts after admin-configured N days (default 7 days). Run cleanup on a schedule (cron or APScheduler inside the worker).
- Finalized runs: retain the signed PDF packet and CSV; delete intermediate render artifacts (page images) after packet generation.
- Monitor disk usage: expose a `/admin/disk-usage` endpoint or log it on each cleanup run.
- Set Docker volume size limits or mount a dedicated volume for uploads/output.
- Never store large intermediate render images in the database; only store paths.

**Warning signs:**
- No cleanup job scheduled
- All run artifacts retained indefinitely by default
- No disk usage monitoring or alerting
- Intermediate page render images (300 DPI, multi-page) stored alongside final outputs with no lifecycle

**Phase to address:** Deployment / infrastructure phase; must be present before first production use.

---

### Pitfall 8: Docker Image Includes Pipeline Build Dependencies — 2+ GB Image

**What goes wrong:**
The pipeline depends on OpenCV, NumPy, PyMuPDF, and potentially Pillow. A naive `pip install -r requirements.txt` in a Debian or Ubuntu base image results in a 1.5–2.5 GB Docker image. First deployment on a shop LAN server (often a repurposed office PC with limited internet) takes 15–30 minutes over a slow connection. Updates via `docker compose pull` hit the same wall. Engineers lose confidence in the update process and stop pulling updates.

**Why it happens:**
The image size problem is invisible in development because the developer's machine already has layers cached. The shop's first `docker compose up` is the first time the full pull size is felt.

**How to avoid:**
Use multi-stage Docker builds: compile/build stage uses a full image; runtime stage uses `python:3.11-slim`. Install OpenCV headless variant (`opencv-python-headless`) which excludes GUI dependencies. Pre-compile Python packages to wheels in the build stage, copy only wheels to runtime stage. Aim for under 500 MB final image size. Publish the image to a registry that supports layer caching so updates only pull changed layers. Include image size in CI output and set a size budget alert.

**Warning signs:**
- Single-stage Dockerfile using `ubuntu:latest` or `python:3.11` (full)
- `opencv-python` (not headless) in requirements
- Image size not tracked in CI
- No multi-stage build

**Phase to address:** Deployment / Docker phase.

---

### Pitfall 9: Concurrent Review Decisions Produce Lost Updates (No Optimistic Locking)

**What goes wrong:**
The project spec allows only one reviewer per run, but engineers share machines or one engineer has the run open in two browser tabs. Both tabs load the same review item. The engineer approves in tab 1 (writes to DB), then adds an override note and re-approves in tab 2 — which loaded the pre-approval state. Tab 2's write overwrites tab 1's approval with stale data. For audit purposes, the final state is the tab 2 write — which is wrong.

**Why it happens:**
The "one reviewer per run" constraint is enforced by policy, not by the data model. Developers assume single-tab, single-session use, which is not realistic for engineers who open links from email or copy URLs.

**How to avoid:**
Add a version column to every review decision row. Each update must include the current version number; if the version in the database does not match, reject the update with HTTP 409 and prompt the user to reload. This is standard optimistic locking. Also track the last-modified timestamp per item; detect stale reads in the UI and show a "this item was updated by another session" banner.

**Warning signs:**
- No `version` or `etag` column on review decision rows
- UPDATE statements do not check a version/timestamp condition
- No conflict response (HTTP 409) in the review API
- No stale-data detection in the frontend

**Phase to address:** Review UI phase.

---

### Pitfall 10: Override Notes Accepted as Empty Strings — Silent Audit Hole

**What goes wrong:**
The project spec requires override notes to be non-empty. The API endpoint validates presence of the `note` field, but accepts `note: "   "` (whitespace only) or `note: "ok"` as valid. Engineers under time pressure learn the minimum input to satisfy the validator. Three months later, the audit reveals overrides with no meaningful justification, which is a compliance finding.

**Why it happens:**
Developers validate field presence (required field check) but not field content quality. The distinction between "has a value" and "has a meaningful value" is a UX and domain problem, not just a technical one.

**How to avoid:**
Validate that override notes are non-empty after stripping whitespace (minimum 10–20 characters). Surface the validation in real-time in the UI (character count, inline message) rather than only on submit. Preserve the note alongside the override in every audit event row — not just in a mutable `notes` column. In the audit PDF, display override reasons inline with the decision evidence so the paper trail is self-evident.

**Warning signs:**
- `note.strip()` not called before validation
- No minimum length requirement
- Override note stored in a mutable column without an audit event copy
- Override note not printed in the audit PDF

**Phase to address:** Review UI phase.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Dispatch pipeline job directly in request handler (no queue) | Zero infrastructure, works immediately | Blocks web server, no recovery on crash, no status visibility | Never — use a worker from day one |
| Store uploaded files with user-provided filename | Simple code | Path traversal risk, race condition collisions | Never |
| Sign-off writes status then generates PDF (no rollback) | Simple sequential code | Permanently broken `signed` runs with no packet | Never |
| Soft immutability (application-only, not DB-level) | Easier schema | Audit records mutable via SQL client or migration | Only for non-audit tables |
| FastAPI BackgroundTasks for pipeline | No Redis dependency | Event loop blocked, tasks lost on restart | Never for CPU-bound long tasks |
| Skip cleanup policy (retain everything) | Simple storage logic | Disk exhaustion in production | Acceptable in MVP dev only — must be added before first production use |
| Single-container Docker (web + worker in one process) | Single `docker-compose` service | Worker crash takes web server down; cannot scale separately | Acceptable only if absolutely no alternative; document the risk |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Python pipeline → web layer | Calling `run_pipeline()` synchronously in a web request | Serialize run inputs to disk, enqueue job ID, worker calls pipeline in separate process |
| Redis as job broker | Using default `maxmemory-policy allkeys-lru` | Set `maxmemory-policy noeviction` so tasks are never silently dropped |
| WeasyPrint PDF generation | Assuming it handles all CSS edge cases | Test with actual snippet images and multi-page layouts early; have fallback error path if render fails |
| PyMuPDF in Docker | Missing system dependencies (libmupdf, freetype) in slim images | Test PDF rendering in the Docker image explicitly, not just locally |
| SQLite in Docker | Using SQLite with a Docker volume | SQLite + Docker volume is fine for single-shop scale, but WAL mode must be enabled; use `PRAGMA journal_mode=WAL` on every connection |
| Session persistence across Docker restarts | Session secret not set via environment variable | Always externalize `SECRET_KEY` / session secret; if it changes on restart, all sessions invalidate and engineers lose in-progress work |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Rendering 300 DPI page images on every request | 2–5 second response times for snippet viewing | Cache rendered images on first pipeline run; serve from file system on subsequent requests | At 5+ concurrent review sessions |
| Loading all run history with joined evidence records | History list page takes 10+ seconds | Paginate history list; load evidence only on run detail view | At 200+ runs in history |
| Serving large snippet PNG files from Django/FastAPI | High memory usage in web process | Serve static files via Nginx or Caddy (or equivalent); never serve large files through Python | At 10+ concurrent users downloading snippets |
| SQLite without WAL mode under concurrent reads/writes | Random "database is locked" errors during review | Enable WAL mode; if concurrent writes become frequent, consider PostgreSQL | At 3+ concurrent review sessions |
| No timeout on pipeline subprocess | Zombie pipeline processes consuming CPU forever | Set a maximum job timeout (e.g., 10 minutes); kill and mark job failed if exceeded | Any drawing that causes an infinite alignment loop |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Application DB user has UPDATE/DELETE on signed run rows | Direct DB edit or future migration silently corrupts audit records | Separate DB roles: app role INSERT-only on audit/event tables; migration role has full access but is never used by the running app |
| No CSRF protection on state-changing API endpoints | Malicious page could approve/override review items on behalf of an authenticated engineer | Enable CSRF tokens for session-based auth; use SameSite=Strict cookies |
| Pipeline output directory accessible via web URL | Unauthenticated access to evidence snippets and audit packets | Serve snippet images and packets through an authenticated endpoint, not as static files at a predictable URL |
| Storing uploaded PDFs in a web-accessible directory | Path traversal or direct URL access to competitor drawings | Store uploads outside the web root; serve only through authenticated download endpoints |
| Weak or default admin password accepted at setup | Shop admin account trivially compromised | Enforce password strength at setup wizard; reject default/weak passwords; log all admin actions |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Session expires mid-review with no warning | Engineer loses all progress; restarts entire review; high frustration | Show session expiry countdown at 5 minutes remaining; auto-extend on any user action; persist every decision immediately |
| "All items reviewed" gate is a soft warning, not a hard block | Engineers skip uncertain items under time pressure; audit packet is incomplete | Hard block: sign-off button is not rendered until all items reach terminal state; no bypass |
| Override reason validation only on submit | Engineer types a long note, submits, gets a validation error, loses the text | Validate inline with real-time feedback; never clear a note field on validation error |
| No distinction between "unchanged" (auto-classified) and "unchanged" (engineer confirmed) | Auditor cannot tell which classifications were reviewed | Store and display both system classification and engineer decision as separate fields; the audit packet must show both |
| Pipeline failure surfaced only in logs | Engineer submits a run and never finds out it failed | Show run failure as an in-app alert; include the failure reason (raster PDF detected, balloon detection failed, etc.) in plain language |
| No progress indicator during pipeline run | Engineer re-submits thinking the run failed | Show job status (queued / running / stage N of 8 / complete / failed) with polling or WebSocket |

---

## "Looks Done But Isn't" Checklist

- [ ] **Sign-off atomicity:** Does the system roll back the signed status if PDF generation fails? Verify by injecting a PDF generation failure and confirming the run returns to reviewable state.
- [ ] **Job queue recovery:** Stop the worker mid-job. Restart it. Does the job resume or re-queue? Verify the run does not get stuck in `running` forever.
- [ ] **Review persistence:** Begin reviewing, approve 5 items, kill the browser tab. Reopen. Are all 5 approvals still there? Verify with a fresh browser session.
- [ ] **File upload collision:** Submit two runs simultaneously with identically named PDFs. Verify they do not share storage and each run references the correct files.
- [ ] **Disk cleanup:** Run the cleanup job on a folder with a mix of aged failed, aged abandoned, and recent finalized runs. Verify only the correct runs are deleted.
- [ ] **Docker image size:** Build the production image. Check `docker image inspect`; verify under 500 MB before first deployment.
- [ ] **Audit immutability:** After sign-off, issue a direct SQL UPDATE on the signed run row. The application DB role should be denied. Verify the error.
- [ ] **Session expiry on long review:** Set session timeout to 1 minute for testing. Begin a review. Wait 2 minutes without submitting. Resume. Verify decisions are not lost.
- [ ] **Override note whitespace:** Submit an override with `note = "   "`. Verify this is rejected with a clear validation message.
- [ ] **Pipeline timeout:** Submit a deliberately malformed PDF that would cause the alignment stage to loop. Verify the job is killed after the configured timeout and marked failed.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Signed run with no packet (generation failure) | MEDIUM | Admin updates status back to `reviewable` via admin tool; engineer re-signs; root cause PDF failure investigated |
| Review decisions lost on session expiry | LOW | Engineer re-reviews; if decisions were not persisted, loss is total for that session |
| Disk full (artifact accumulation) | HIGH | Stop new submissions; run emergency cleanup targeting oldest failed runs; restore service; post-mortem on why cleanup was not running |
| Job stuck in `running` after worker crash | LOW | Admin tool to reset status to `queued`; worker re-picks up |
| Audit record mutated (discovered in audit) | CRITICAL | Cannot undo if no hash chain or append-only log; must disclose to customer; re-inspect affected characteristics; this is a regulatory finding |
| Docker image too large for shop network | MEDIUM | Rebuild with multi-stage; push optimized image; provide offline tarball (`docker save`) for shops with constrained connectivity |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Sign-off / PDF generation not atomic | File upload + pipeline integration | Integration test: inject PDF generation failure; verify status rollback |
| Job dispatched inside DB transaction | Job queue infrastructure | Load test with concurrent submissions; check worker logs for "not found" errors |
| FastAPI BackgroundTasks for CPU-bound work | Job queue infrastructure | Submit a run; confirm web server responds during pipeline execution |
| Review state lost on session expiry | Review UI | Manual test: approve items, expire session, reload, confirm persistence |
| Soft audit immutability | Database schema design | Attempt direct SQL UPDATE as app DB role; expect permission denied |
| Uploaded file naming / collision | File upload phase | Concurrent upload test with identical filenames; check storage isolation |
| Disk artifact accumulation | Deployment / infrastructure | Run cleanup job after seeding failed/abandoned runs; verify correct deletion |
| Docker image size | Docker / deployment phase | `docker build` + `docker image ls`; size check in CI |
| Lost updates in concurrent review | Review UI | Two-tab test: approve same item in both tabs; verify HTTP 409 on second write |
| Override note whitespace bypass | Review UI | Automated API test: POST override with whitespace-only note; expect 400 |

---

## Sources

- [YunoJuno: Django-RQ race conditions — jobs dispatched inside transactions](http://tech.yunojuno.com/django-rq-and-race-conditions) (MEDIUM confidence — real post-mortem, specific to Django-RQ)
- [Confluent: The Dual-Write Problem](https://www.confluent.io/blog/dual-write-problem/) (HIGH confidence — official Confluent engineering blog)
- [Design Gurus: Enforcing immutability and append-only audit trails](https://www.designgurus.io/answers/detail/how-do-you-enforce-immutability-and-appendonly-audit-trails) (MEDIUM confidence — multiple sources agree on DB-level enforcement)
- [Leapcell: FastAPI async task management pitfalls](https://leapcell.io/blog/understanding-pitfalls-of-async-task-management-in-fastapi-requests) (HIGH confidence — FastAPI-specific, verified against FastAPI BackgroundTasks behavior)
- [FastAPI BackgroundTasks blocks application (GitHub Discussion)](https://github.com/fastapi/fastapi/discussions/11210) (HIGH confidence — official FastAPI community report)
- [Vinta Software: Advanced Celery for Django — fixing unreliable background tasks](https://www.vintasoftware.com/blog/guide-django-celery-tasks) (MEDIUM confidence — practitioner blog)
- [OWASP: File Upload Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html) (HIGH confidence — authoritative security standard)
- [Better Stack: Reducing Docker image sizes from 1.2GB to 150MB](https://betterstack.com/community/guides/scaling-docker/reducing-docker-image-size/) (MEDIUM confidence — verified against Docker multi-stage build docs)
- [WorkOS: Session management best practices](https://workos.com/blog/session-management-best-practices) (MEDIUM confidence)
- [NN/g: Designing for long waits and interruptions in complex workflows](https://www.nngroup.com/articles/designing-for-waits-and-interruptions/) (HIGH confidence — Nielsen Norman Group, authoritative UX research)
- [binaryigor: Optimistic vs pessimistic locking](https://binaryigor.com/optimistic-vs-pessimistic-locking.html) (MEDIUM confidence — practical engineering analysis)
- [David Seddonym: The trouble with transaction.atomic](https://seddonym.me/2020/11/19/trouble-atomic/) (MEDIUM confidence — Django-specific post-mortem)

---
*Pitfalls research for: Aerospace FAIR pipeline web wrapper — audit-sensitive state, async jobs, file upload, PDF generation, Docker LAN deployment*
*Researched: 2026-03-01*