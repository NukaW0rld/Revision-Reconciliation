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
