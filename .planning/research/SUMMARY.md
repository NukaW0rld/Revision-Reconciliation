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
