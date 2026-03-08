# Phase 2: Pipeline Bridge - Research

**Researched:** 2026-03-03
**Domain:** Async task execution (Huey + SqliteHuey), FastAPI SSE, file upload validation, DB-polled progress streaming
**Confidence:** HIGH (core stack verified via official docs and package inspection)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Upload form structure:** Single-page form at `/runs/new` — all 3 file pickers + all part metadata fields on one page
- **Multi-page PDF page selection:** Inline on the form via HTMX partial; submit button disabled until page selected; stays on same page
- **Upload CTA:** Prominent "Submit New Run" button on dashboard (not nav bar link)
- **After submission:** Auto-redirect to run status page via standard HTML POST (not HTMX POST)
- **Stage display:** Numbered checklist — completed stages show checkmark, current stage shows spinner, pending stages are dim; current stage name visually highlighted
- **Stage order:** Form 3 parsing → Balloon detection → Text extraction → Anchor building → Alignment → Candidate matching → Classification → Output
- **Run states:** queued → running (with current stage) → completed / failed / warning
- **Dashboard run list:** 5-10 most recent runs with compact cards + "View all runs" link to `/runs`
- **Run card content:** part number (prominent), revision letters (A→B), status badge, submitted date, assigned reviewer name
- **Nav bar (added Phase 2):** Persistent top nav in `base.html` — shop name/logo, "New Run" link, "My Runs" link, alert bell indicator, logged-in user email + logout
- **All engineers see all runs** (consistent with HISTORY-01); filterable by part number and date
- **Alerts are personal:** Only the assigned reviewer sees failure notifications for their runs
- **PIPE-05 (Rev B balloon failure):** Inline yellow warning banner; "Open Review Queue" replaced with "Acknowledge and Open Review"; no modal
- **PIPE-06 (alignment low confidence):** Inline warning section showing plain-language confidence summary + two buttons: "Proceed to Review" / "Abort Run"
- **PIPE-04 (Rev A balloon failure — hard fail):** "Failed" badge + stage checklist with failing stage highlighted red + plain-language error message; no recovery action
- **Alert surface:** Bell icon with red badge count in nav bar; unread failure alerts as dismissible banner at top of dashboard
- **Alert content:** Part number, run ID, failure stage + "View run" link; clicking banner dismisses and clears badge
- **Standard HTML POST** for form submissions that result in redirects (not HTMX POST)
- **HTMX partials** for inline UI updates that stay on the same page
- **Router imports deferred inside `create_app()` body** to break circular imports
- **`get_current_user()`** auth dependency on every new route
- **Phase 2 adds `Run` and `RunAlert` models** to `shop/models.py`

### Claude's Discretion

- Exact DaisyUI component choices for the stage checklist (steps component, timeline, or custom list)
- Confidence distribution display format (bar chart, percentage list, or text summary — choose whichever is clearest without adding a charting library)
- Exact run assignment UI placement (submitter defaults as reviewer; explicit reassignment is Phase 3)
- File storage location for uploaded PDFs and Excel files within the Docker container
- Huey task definition structure and queue configuration details

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| UPLOAD-01 | Engineer uploads Rev A PDF, Rev B PDF, Form 3 Excel as single submission with part metadata | FastAPI UploadFile + Form fields pattern; shutil.copyfileobj to disk |
| UPLOAD-02 | System detects raster (scanned) PDFs at upload time and displays a clear error | PyMuPDF image-coverage heuristic: `page.get_images()` area vs. `page.rect` area ≥ 95% |
| UPLOAD-03 | System detects multi-page PDFs at upload time and prompts page selection inline | `fitz.open(bytes).page_count > 1` triggers HTMX partial page-selector response |
| UPLOAD-04 | System validates Excel is readable and contains recognizable sheet structure | `parse_excel_preview()` already implemented in `shop/services/form3.py` — reuse directly |
| UPLOAD-05 | Revision labels (Rev A letter, Rev B letter) captured and stored as part of run record | Text fields on upload form stored in `Run` model columns `rev_a_label` / `rev_b_label` |
| PIPE-01 | Pipeline executes asynchronously in a worker process separate from web server | Huey 2.6.0 + SqliteHuey — worker started via `huey_consumer.py shop.tasks.huey` under supervisord |
| PIPE-02 | Stage-by-stage progress display through all 8 stages | Worker writes `current_stage` + `current_stage_index` to `Run` row; SSE endpoint polls DB every 1 s |
| PIPE-03 | Distinguishes run states: queued, running, completed, failed | `Run.status` enum column with values: queued/running/completed/failed/warning |
| PIPE-04 | Rev A balloon detection failure → hard fail with clear error message | Worker catches `BalloonDetectionError`; sets `Run.status='failed'`, `Run.failure_stage='Balloon detection'`, `Run.failure_message=...` |
| PIPE-05 | Rev B balloon detection failure → partial result warning; engineer must acknowledge | Worker sets `Run.status='warning'`, `Run.warning_type='revB_balloon'`; status page gated by acknowledge button |
| PIPE-06 | Alignment uniformly low confidence → warning state with confidence summary + explicit proceed/abort | Worker computes confidence distribution; sets `Run.status='warning'`, `Run.warning_type='low_confidence'`, `Run.confidence_summary=JSON` |
| PIPE-07 | Characteristics not located in Rev B: auto-classified removed if confidence ≥ 0.9, else unresolved | Pipeline classify logic already handles this; no new code, just correct thresholds wired through |
| PIPE-08 | Characteristics with balloon found but text extraction fails: low-confidence unchanged or unresolved | Pipeline classify logic already handles this; driven by existing confidence thresholds |
| PIPE-09 | Duplicate balloon numbers treated as one logical characteristic; count token changes tracked | Pipeline already implements this in `normalize.py` `parse_requirement()` count_tokens field |
| PIPE-10 | GD&T feature control frames matched as opaque strings only | Existing classify logic already matches as strings; no additional code needed — document as intentional |
| PIPE-11 | Assigned reviewer receives in-app alert when run they're responsible for fails | Worker creates `RunAlert` row after setting `Run.status='failed'`; nav bar reads unread alert count |
</phase_requirements>

---

## Summary

Phase 2 connects the existing delta preservation pipeline (`delta_preservation/cli.py:run_pipeline()`) to the web UI via an async task queue. Engineers submit a run via a standard HTML POST form, the server enqueues a Huey task, and a Huey worker process runs the pipeline in the background. A FastAPI SSE endpoint streams stage-by-stage progress to the status page by polling the SQLite DB every 1 second. Upload-time validation (scanned PDF detection, multi-page PDF page selection, Excel readability) prevents bad submissions before the pipeline starts.

The critical architectural constraint is the **process boundary**: Huey's consumer runs in a separate OS process from the FastAPI web server. This means asyncio queues cannot carry progress events between them. The established pattern is for the **worker to write stage progress to the shared SQLite DB**, and the **SSE endpoint to poll that DB row in an async loop**. This is simple, reliable, and requires no additional infrastructure beyond what already exists (SQLite + SQLAlchemy).

Phase 1 has established all key patterns this phase builds on: standard HTML POST for form submissions that result in redirects, HTMX partials for inline UI updates, deferred router imports inside `create_app()`, and DaisyUI components (badge, card, btn, alert). All 21 existing tests pass.

**Primary recommendation:** Use Huey 2.6.0 + SqliteHuey (stdlib-only, no peewee) with a separate `shop/tasks.py` module; run both uvicorn and `huey_consumer.py` under supervisord inside the existing Docker container; use DB-polling SSE for progress streaming.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| huey | 2.6.0 | Async task queue + SqliteHuey backend | Chosen at roadmap; zero external service dependency; stdlib sqlite3 only; supports `immediate=True` for tests |
| FastAPI native SSE | 0.135.1 (installed) | SSE via `fastapi.sse.EventSourceResponse` | Already installed; native support added in 0.135.0; no extra dependency needed |
| htmx-sse extension | 2.x (already bundled) | HTMX-side SSE attribute binding | Already in `/static/js/htmx-sse.js`; extension already loaded in `base.html` |
| PyMuPDF (fitz) | already installed | Upload-time PDF validation (page count, raster detection) | Already a pipeline dependency |
| openpyxl | already installed | Excel validation at upload time via `parse_excel_preview()` | Already a pipeline dependency |
| supervisord | system package | Run uvicorn + huey_consumer in same Docker container | Standard pattern for multi-process single-container deployments |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| python-multipart | already installed | Form + file upload parsing for FastAPI | Required for `UploadFile`; already in pyproject.toml |
| shutil (stdlib) | stdlib | Stream UploadFile to disk without full memory load | Use `shutil.copyfileobj(file.file, dest)` for all file saves |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| DB-polling SSE | asyncio.Queue-based SSE | Queue pattern is cleaner but doesn't cross process boundaries — DB poll is the correct choice when worker is a separate OS process |
| supervisord | Docker multi-service compose | Two-container compose adds complexity and a shared-volume SSE event file; single container with supervisord is simpler for air-gapped shop |
| Native FastAPI SSE (`fastapi.sse`) | sse-starlette | Both work; native FastAPI SSE is already available (v0.135.1 installed) — no extra dependency needed |

**Installation (new packages only):**
```bash
# Add to pyproject.toml dependencies and run:
uv add huey
# supervisord added to Dockerfile via apt-get install -y supervisor
```

---

## Architecture Patterns

### Recommended Project Structure (Phase 2 additions)

```
shop/
├── routers/
│   └── runs.py              # /runs/new, /runs/{id}, /runs/{id}/sse, /runs, alerts
├── services/
│   └── runs.py              # File save, run record creation, alert creation
├── tasks.py                 # SqliteHuey instance + @huey.task() pipeline_task()
├── models.py                # Add Run, RunAlert models
└── templates/
    ├── runs/
    │   ├── new.html          # Upload form (single page)
    │   ├── status.html       # Stage progress + warnings + alerts
    │   ├── list.html         # /runs — all runs, filterable
    │   └── page_selector.html   # HTMX partial for multi-page PDF
    └── base.html             # Add persistent nav bar here
docker/
├── supervisord.conf         # Runs uvicorn + huey_consumer
└── Dockerfile               # Add apt-get install -y supervisor + COPY supervisord.conf
```

### Pattern 1: Huey Task Module

**What:** Centralized `shop/tasks.py` defines the `SqliteHuey` instance and the pipeline task. Both the web app and the worker consumer import from this module.

**When to use:** This is the only pattern that works — the Huey instance must be importable as a dotted path for `huey_consumer.py`.

**Example:**
```python
# shop/tasks.py
from huey import SqliteHuey

# Filename must match the huey.db path used by the consumer startup command
huey = SqliteHuey(filename="/app/data/huey.db")

@huey.task()
def run_pipeline_task(run_id: int, revA_path: str, revB_path: str,
                       form3_path: str, part_name: str, dpi: int = 300):
    """
    Huey task wrapping delta_preservation.cli.run_pipeline().
    Updates Run.current_stage and Run.status in the shop DB at each stage.
    """
    from shop.database import SessionLocal
    from shop.models import Run, RunAlert
    db = SessionLocal()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        run.status = "running"
        db.commit()

        # Stage-by-stage execution with DB updates between stages
        # Worker calls the pipeline with a progress_callback kwarg
        from delta_preservation.cli import run_pipeline
        out_dir = run_pipeline(
            revA_pdf=revA_path,
            revB_pdf=revB_path,
            form3_xlsx=form3_path,
            part_name=part_name,
            dpi=dpi,
            stage_callback=lambda stage_idx, stage_name: _update_stage(db, run, stage_idx, stage_name),
        )
        run.status = "completed"
        run.output_dir = str(out_dir)
        db.commit()
    except BalloonDetectionRevAError as e:
        run.status = "failed"
        run.failure_stage = "Balloon detection"
        run.failure_message = str(e)
        db.commit()
        _create_alert(db, run)
    except Exception as e:
        run.status = "failed"
        run.failure_stage = "Unknown"
        run.failure_message = str(e)
        db.commit()
        _create_alert(db, run)
    finally:
        db.close()
```

**Note on `stage_callback`:** `run_pipeline()` in `cli.py` currently uses `print()` for stage progress. The task wrapper must either (a) add a `stage_callback` parameter to `run_pipeline()`, or (b) wrap each of the 8 stages manually in `run_pipeline_task()` by calling the stage functions directly. Option (b) is safer since it avoids modifying the CLI contract.

### Pattern 2: DB-Polling SSE Endpoint

**What:** The SSE endpoint opens an async generator that polls the `Run` DB row every 1 second and emits an SSE event with the current stage + status. The HTMX status page listens and swaps the stage checklist partial on each event.

**When to use:** Always — this is the correct pattern when the progress producer (Huey worker) is a separate OS process. Cross-process asyncio queues are not possible.

**Example:**
```python
# shop/routers/runs.py
import asyncio
from collections.abc import AsyncIterable
from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent
from shop.models import Run
from shop.dependencies import get_db, get_current_user

router = APIRouter(prefix="/runs")

@router.get("/{run_id}/sse", response_class=EventSourceResponse)
async def run_sse(
    run_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AsyncIterable[ServerSentEvent]:
    """Poll DB every 1s; emit SSE events with current stage; close when terminal."""
    terminal = {"completed", "failed", "warning"}
    while True:
        if await request.is_disconnected():
            break
        run = db.query(Run).filter(Run.id == run_id).first()
        if run is None:
            break
        payload = {
            "status": run.status,
            "current_stage": run.current_stage,
            "current_stage_index": run.current_stage_index,
            "failure_stage": run.failure_stage,
            "failure_message": run.failure_message,
            "warning_type": run.warning_type,
        }
        yield ServerSentEvent(data=payload, event="stage_update")
        if run.status in terminal:
            yield ServerSentEvent(event="close", data="done")
            break
        await asyncio.sleep(1.0)
```

**HTMX side (status.html):**
```html
<!-- SSE connection on the status container -->
<div hx-ext="sse"
     sse-connect="/runs/{{ run.id }}/sse"
     sse-close="close">

  <!-- Swap stage checklist on each update -->
  <div sse-swap="stage_update"
       hx-target="#stage-checklist"
       hx-swap="outerHTML">
    {% include "runs/_stage_checklist.html" %}
  </div>

  <div id="stage-checklist">
    {% include "runs/_stage_checklist.html" %}
  </div>
</div>
```

**Note:** The `sse-close="close"` attribute on the parent div closes the SSE connection when the server sends an event named `close`. This prevents the browser from reconnecting after the run completes.

### Pattern 3: Standard HTML POST Upload → Redirect to Status Page

**What:** Upload form uses `<form method="post" action="/runs/new">` (no HTMX). FastAPI handler saves files, creates `Run` record, enqueues Huey task, returns `RedirectResponse("/runs/{id}", 302)`. Browser follows redirect natively.

**When to use:** All form submissions that result in page redirects. This is an established pattern from Phase 1 (login, wizard steps).

**Example:**
```python
@router.post("/new")
async def submit_run(
    request: Request,
    revA_pdf: UploadFile = File(...),
    revB_pdf: UploadFile = File(...),
    form3_xlsx: UploadFile = File(...),
    part_number: str = Form(...),
    rev_a_label: str = Form(...),
    rev_b_label: str = Form(...),
    customer: str = Form(...),
    job_number: str = Form(...),
    revA_page: int = Form(0),      # Set by HTMX page selector partial
    revB_page: int = Form(0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # Validate PDFs (raster check, page count already done on file pick)
    # Save files to /app/data/uploads/{run_uuid}/
    # Create Run record
    # Enqueue Huey task
    # Return RedirectResponse to /runs/{run_id}
```

### Pattern 4: HTMX Partial for Multi-Page PDF Page Selector

**What:** After the engineer picks a PDF file, an HTMX `hx-trigger="change"` on the file input POSTs the file bytes to `/runs/validate-pdf`, which returns an HTML partial containing a page selector if `page_count > 1`, or an empty response if single-page.

**When to use:** Inline validation/selection that must stay on the same page.

**Example:**
```python
@router.post("/validate-pdf")
async def validate_pdf_partial(
    request: Request,
    file: UploadFile = File(...),
    field: str = Form(...),  # "revA" or "revB" — drives the hidden input name
):
    content = await file.read()
    doc = fitz.open(stream=content, filetype="pdf")
    page_count = doc.page_count
    is_raster = _detect_raster(doc)
    doc.close()
    if is_raster:
        return templates.TemplateResponse(request, "runs/pdf_error.html",
            {"error": "This PDF appears to be a scanned image. Upload a vector PDF."})
    if page_count > 1:
        return templates.TemplateResponse(request, "runs/page_selector.html",
            {"page_count": page_count, "field": field})
    return HTMLResponse("")  # Single-page, valid — clear the section
```

**Raster detection function:**
```python
def _detect_raster(doc: fitz.Document, threshold: float = 0.95) -> bool:
    """Return True if first page is predominantly a raster image (scanned PDF)."""
    page = doc.load_page(0)
    page_area = abs(page.rect)
    if page_area == 0:
        return False
    images = page.get_images(full=True)
    if not images:
        return False
    total_image_area = 0.0
    for img in images:
        xref = img[0]
        rects = page.get_image_rects(xref)
        for rect in rects:
            total_image_area += abs(rect)
    return (total_image_area / page_area) >= threshold
```

**Source:** Based on PyMuPDF maintainer recommendation from https://github.com/pymupdf/PyMuPDF/discussions/1653 — compare image bbox area vs. page rect area.

### Pattern 5: DaisyUI Steps Component for Stage Checklist

**What:** Use the DaisyUI `steps steps-vertical` component. Completed stages get `step-success`, current stage gets `step-primary` with a spinner icon via `data-content`, pending stages have no color class (dim/default).

**Example:**
```html
<!-- runs/_stage_checklist.html -->
<ul class="steps steps-vertical w-full" id="stage-checklist">
  {% for stage in stages %}
  <li class="step {% if stage.index < current_stage_index %}step-success
                   {% elif stage.index == current_stage_index %}step-primary
                   {% endif %}"
      {% if stage.index == current_stage_index %}data-content="..."{% endif %}>
    <span class="flex items-center gap-2">
      {% if stage.index == current_stage_index %}
        <span class="loading loading-spinner loading-xs"></span>
      {% elif stage.index < current_stage_index %}
        <span>✓</span>
      {% endif %}
      {{ stage.name }}
    </span>
  </li>
  {% endfor %}
</ul>
```

### Anti-Patterns to Avoid

- **HTMX POST for the upload form:** HTMX intercepts 302 redirects as partial DOM swaps, not full-page navigation. This breaks the redirect-to-status-page pattern. Use standard HTML POST.
- **asyncio.Queue for cross-process progress:** The Huey worker is a separate OS process. asyncio queues only work within a single process. The worker must write to the DB; SSE must read from the DB.
- **Calling `run_pipeline()` in the FastAPI request handler (without Huey):** Blocks the ASGI event loop. Always use the Huey task.
- **Re-reading uploaded file bytes into memory for validation after save:** Read the bytes once during the `validate-pdf` HTMX partial request; save to disk during the final POST submission. Don't save then re-read.
- **Importing `shop.tasks.huey` at module load in `shop/app.py`:** Import inside `create_app()` body following the established deferred-import pattern. Otherwise circular imports arise since `tasks.py` imports from `shop.database`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Task queue + worker | Custom subprocess + IPC | Huey + SqliteHuey | Retry, result storage, concurrency, graceful shutdown already handled |
| Raster PDF detection | Custom image-vs-text heuristic | PyMuPDF `page.get_images()` area check | Already installed; maintainer-endorsed approach |
| Excel validation | Custom openpyxl parsing | `parse_excel_preview()` in `shop/services/form3.py` | Already written and tested — reuse directly |
| SSE formatting | Manual `text/event-stream` response | `fastapi.sse.EventSourceResponse` + `ServerSentEvent` | Native FastAPI 0.135.x; correct headers, keep-alive, and reconnect handled |
| Multi-process management | Custom shell scripts or subprocess.Popen | supervisord | Standard Docker pattern; handles restart, log forwarding to stdout |
| Stage checklist UI | Custom CSS/JS progress bar | DaisyUI `steps` component | Already in project's Tailwind config; no additional CSS needed |

**Key insight:** The pipeline (`delta_preservation/cli.py`) is already complete and battle-tested. Phase 2's job is purely plumbing — queue, status persistence, and UI wiring. Avoid touching the pipeline internals.

---

## Common Pitfalls

### Pitfall 1: File Upload HTMX vs Standard POST Confusion

**What goes wrong:** Developer uses `hx-post` on the upload form, which causes HTMX to intercept the 302 redirect as a partial swap, resulting in the status page HTML appearing inside the current page div rather than navigating to it.

**Why it happens:** HTMX treats all responses as partial swaps by default; only full-page navigation bypasses this.

**How to avoid:** The upload form must be `<form method="post" action="/runs/new">` with no HTMX attributes on the form element itself. Only use HTMX on the file input's `change` event for the inline page-selector validation.

**Warning signs:** After form submit, page URL does not change but page content is partially replaced.

### Pitfall 2: SSE Generator Blocking on DB Query in Async Context

**What goes wrong:** Using a synchronous `db.query()` inside an `async def` SSE generator blocks the ASGI event loop, preventing other requests from being served during the 1-second sleep.

**Why it happens:** SQLAlchemy's synchronous Session is blocking I/O.

**How to avoid:** Use `asyncio.sleep(1.0)` (not `time.sleep(1)`) and keep the DB query fast (single row by primary key). Alternatively, use `run_in_executor` to offload the DB read. For this application's scale (O(10) engineers), the synchronous query + asyncio.sleep pattern is acceptable since the query is a single indexed primary key lookup.

**Warning signs:** Other page loads become slow or timeout during active run monitoring.

### Pitfall 3: Huey SQLite File Location Mismatch

**What goes wrong:** `SqliteHuey(filename="/app/data/huey.db")` in `shop/tasks.py` but supervisord starts the consumer with a different working directory, causing the consumer to create `huey.db` in a different path than the web app expects.

**Why it happens:** Relative paths in SqliteHuey resolve relative to the working directory of whichever process opens the file.

**How to avoid:** Always use an absolute path for the Huey DB filename that matches the Docker volume mount. Set it via an environment variable consistent with `DATABASE_URL`:

```python
import os
HUEY_DB = os.environ.get("HUEY_DB", "/app/data/huey.db")
huey = SqliteHuey(filename=HUEY_DB)
```

**Warning signs:** "No tasks processed" in consumer logs while the web app enqueues successfully; consumer exits immediately.

### Pitfall 4: SSE Connection Not Closed After Terminal State

**What goes wrong:** After the run reaches `completed` / `failed` / `warning`, the browser EventSource reconnects every 3 seconds indefinitely (SSE default reconnect behavior), hammering the DB with polling.

**Why it happens:** SSE reconnects automatically unless the server sends a `close` event or the client explicitly calls `eventSource.close()`.

**How to avoid:** Send a named `close` event from the SSE generator after the terminal payload, and bind `sse-close="close"` on the HTMX SSE container element. This closes the EventSource from the HTMX side.

**Warning signs:** DB connections accumulate; server logs show repeated SSE connections for completed runs.

### Pitfall 5: Raster PDF Detection False Positives

**What goes wrong:** A legitimate vector PDF with a large background image (e.g., title block logo) is rejected as raster.

**Why it happens:** Image area threshold set too low (e.g., 80%); even vector drawings sometimes have decorative raster elements.

**How to avoid:** Set the threshold at 95% (`threshold=0.95`) per the PyMuPDF maintainer recommendation. Add a secondary check: if text can be extracted from the page (`.get_text()` returns non-empty), it is likely a vector PDF even if it has background images.

**Warning signs:** Engineers report valid drawings being rejected.

### Pitfall 6: Huey `immediate=True` Required in Tests

**What goes wrong:** Tests that submit the upload form hang indefinitely waiting for a background task that never executes, because the Huey consumer process is not running in the test environment.

**Why it happens:** Huey tasks are enqueued but no consumer is running in the test process.

**How to avoid:** In the test fixture, set `huey.immediate = True` before the test and `huey.immediate = False` after. Immediate mode executes tasks synchronously in the calling thread, making them testable without a consumer process.

```python
# conftest.py addition
@pytest.fixture(autouse=False)
def huey_immediate():
    from shop.tasks import huey
    huey.immediate = True
    yield
    huey.immediate = False
    huey.storage.flush_all()  # clean up between tests
```

**Warning signs:** Tests time out or hang on run submission.

---

## Code Examples

Verified patterns from official sources:

### SqliteHuey Instantiation (Huey 2.6.0)

```python
# Source: https://huey.readthedocs.io/en/latest/api.html#SqliteHuey
from huey import SqliteHuey

# stdlib sqlite3 only — no peewee required as of Huey 2.0+
huey = SqliteHuey(
    filename="/app/data/huey.db",
    fsync=False,         # True = durable writes; slower but safer on power loss
)
```

### Consumer Startup Command (for supervisord.conf)

```ini
; Source: https://huey.readthedocs.io/en/latest/consumer.html
[program:huey_worker]
directory=/app
command=huey_consumer.py shop.tasks.huey --workers=1 --worker-type=thread
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stdout
stderr_logfile_maxbytes=0
autostart=true
autorestart=true
```

**Why `--worker-type=thread`:** The pipeline is CPU + I/O bound (PyMuPDF, OpenCV, NumPy). Thread workers share process memory (which matters for SQLAlchemy connections). Process workers would require separate DB connection setup. For a single-run-at-a-time shop workflow, thread with `--workers=1` is sufficient.

### FastAPI Native SSE (FastAPI 0.135.x)

```python
# Source: https://fastapi.tiangolo.com/tutorial/server-sent-events/
from fastapi.sse import EventSourceResponse, ServerSentEvent
from collections.abc import AsyncIterable

@router.get("/{run_id}/sse", response_class=EventSourceResponse)
async def run_progress_sse(
    run_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AsyncIterable[ServerSentEvent]:
    terminal = {"completed", "failed", "warning"}
    while True:
        if await request.is_disconnected():
            break
        run = db.query(Run).filter(Run.id == run_id).first()
        data = {
            "status": run.status,
            "current_stage_index": run.current_stage_index,
            "current_stage": run.current_stage,
        }
        yield ServerSentEvent(data=data, event="stage_update")
        if run.status in terminal:
            yield ServerSentEvent(event="close", data="done")
            break
        await asyncio.sleep(1.0)
```

### UploadFile to Disk (FastAPI + shutil)

```python
# Source: https://fastapi.tiangolo.com/tutorial/request-files/
import shutil
import uuid
from pathlib import Path
from fastapi import UploadFile

def save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    """Stream UploadFile to dest_dir using a stable UUID filename."""
    suffix = Path(upload.filename).suffix.lower()
    filename = f"{uuid.uuid4().hex}{suffix}"
    dest = dest_dir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest
```

### Raster PDF Detection (PyMuPDF)

```python
# Source: https://github.com/pymupdf/PyMuPDF/discussions/1653
import fitz

def is_raster_pdf(pdf_bytes: bytes, threshold: float = 0.95) -> bool:
    """
    Return True if the first page is predominantly covered by raster images.
    Also returns False if any extractable text is found (secondary heuristic).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)

    # Secondary: if text extraction returns content, it's likely vector
    if page.get_text().strip():
        doc.close()
        return False

    page_area = abs(page.rect)
    if page_area == 0:
        doc.close()
        return False

    images = page.get_images(full=True)
    if not images:
        doc.close()
        return False

    total_image_area = sum(
        abs(rect)
        for img in images
        for rect in page.get_image_rects(img[0])
    )
    doc.close()
    return (total_image_area / page_area) >= threshold
```

### HTMX SSE Binding with Close (htmx 2.x + htmx-sse extension)

```html
<!-- Source: https://htmx.org/extensions/sse/ -->
<div hx-ext="sse"
     sse-connect="/runs/{{ run.id }}/sse"
     sse-close="close">

  <div id="stage-checklist"
       sse-swap="stage_update"
       hx-target="#stage-checklist"
       hx-swap="outerHTML">
    {% include "runs/_stage_checklist.html" %}
  </div>
</div>
```

### DaisyUI Steps Vertical (stage checklist)

```html
<!-- Source: https://daisyui.com/components/steps/ -->
<ul class="steps steps-vertical">
  <li class="step step-success" data-content="✓">Form 3 parsing</li>
  <li class="step step-primary">
    <span class="loading loading-spinner loading-xs mr-1"></span>
    Balloon detection
  </li>
  <li class="step">Text extraction</li>
  <li class="step">Anchor building</li>
  <li class="step">Alignment</li>
  <li class="step">Candidate matching</li>
  <li class="step">Classification</li>
  <li class="step">Output</li>
</ul>
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| sse-starlette (external) for FastAPI SSE | `fastapi.sse.EventSourceResponse` (native) | FastAPI 0.135.0 (2025) | No extra dependency; already available in this project |
| Huey pre-2.0 SQLite required peewee | Huey 2.0+ SqliteHuey uses stdlib sqlite3 only | Huey 2.0 (2019) | No extra dependency; just `uv add huey` |
| `from huey.contrib.sqlitedb import SqliteHuey` | `from huey import SqliteHuey` | Huey 2.0 | Updated import path |
| htmx SSE with manual EventSource JS | `hx-ext="sse"` + `sse-connect` attributes | htmx 1.9+ | Declarative; no custom JS needed; already bundled |

**Deprecated/outdated:**
- `from huey.contrib.sqlitedb import SqliteHuey`: Old import path (pre-2.0). Use `from huey import SqliteHuey`.
- `sse-starlette`: Was the de-facto standard before FastAPI 0.135.0. Still works but unnecessary here since FastAPI native SSE is available and already installed.

---

## Open Questions

1. **Pipeline `stage_callback` integration**
   - What we know: `run_pipeline()` currently uses `print()` for stage progress; no callback parameter exists
   - What's unclear: Whether to add a `stage_callback` parameter to `run_pipeline()` (cleaner) or have the Huey task call each of the 8 pipeline stage functions individually (avoids modifying the CLI)
   - Recommendation: Add an optional `stage_callback: Callable[[int, str], None] | None = None` parameter to `run_pipeline()` — called before each stage with `(stage_index, stage_name)`. This is the minimal-invasive change and keeps the CLI fully backward-compatible.

2. **PIPE-06 confidence distribution threshold**
   - What we know: `MIN_ALIGNMENT_INLIERS=40`, `MIN_ALIGNMENT_RATIO=0.15` exist in `config.py`; PIPE-06 triggers when alignment is "uniformly low confidence"
   - What's unclear: The exact condition — is it based on alignment inliers/ratio, or on the distribution of per-characteristic confidence scores in the delta items?
   - Recommendation: Trigger PIPE-06 when alignment `inlier_ratio < MIN_ALIGNMENT_RATIO` (transform is unreliable) AND the run has not already hard-failed on PIPE-04. Store the warning type and a JSON summary of confidence stats in `Run.confidence_summary`.

3. **File storage path within Docker container**
   - What we know: `/app/data` is already mounted as a Docker volume (for `shop.db`); `/app/out` is mounted for pipeline output
   - What's unclear: Whether to store uploaded files under `/app/data/uploads/` (alongside the DB) or `/app/out/uploads/`
   - Recommendation: Store uploads under `/app/data/uploads/{run_uuid}/` — they are inputs, not outputs, and belong with the persistent data volume. The pipeline output stays in `/app/out/{run_id}/`.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.x (installed) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/test_runs.py -x -q` |
| Full suite command | `uv run pytest -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| UPLOAD-01 | Upload form accepts 3 files + metadata, creates Run record, redirects to status page | integration | `pytest tests/test_runs.py::test_upload_creates_run -x` | Wave 0 |
| UPLOAD-02 | Raster PDF at validate-pdf endpoint returns error partial | unit | `pytest tests/test_runs.py::test_raster_pdf_rejected -x` | Wave 0 |
| UPLOAD-03 | Multi-page PDF at validate-pdf returns page selector partial | unit | `pytest tests/test_runs.py::test_multipage_pdf_page_selector -x` | Wave 0 |
| UPLOAD-04 | Unreadable Excel rejected with error message | unit | `pytest tests/test_runs.py::test_invalid_excel_rejected -x` | Wave 0 |
| UPLOAD-05 | Rev labels stored in Run.rev_a_label / rev_b_label after submission | integration | `pytest tests/test_runs.py::test_rev_labels_stored -x` | Wave 0 |
| PIPE-01 | Huey task enqueued after submission; worker executes pipeline | integration | `pytest tests/test_runs.py::test_pipeline_task_enqueued -x` (huey_immediate) | Wave 0 |
| PIPE-02 | Run.current_stage and current_stage_index update during execution | unit | `pytest tests/test_runs.py::test_stage_progress_updates -x` | Wave 0 |
| PIPE-03 | Run.status transitions from queued→running→completed | integration | `pytest tests/test_runs.py::test_run_status_lifecycle -x` | Wave 0 |
| PIPE-04 | Rev A balloon failure sets status=failed, failure_stage populated | unit | `pytest tests/test_runs.py::test_revA_balloon_failure -x` | Wave 0 |
| PIPE-05 | Rev B balloon failure sets status=warning, warning_type=revB_balloon | unit | `pytest tests/test_runs.py::test_revB_balloon_warning -x` | Wave 0 |
| PIPE-06 | Low alignment confidence sets status=warning, warning_type=low_confidence | unit | `pytest tests/test_runs.py::test_low_confidence_warning -x` | Wave 0 |
| PIPE-07 | Removed classification applied when spatial confidence ≥ 0.9 | unit | Covered by existing pipeline unit tests in delta_preservation | Exists |
| PIPE-08 | Low-confidence unchanged applied when spatial confidence ≥ 0.9 | unit | Covered by existing pipeline unit tests in delta_preservation | Exists |
| PIPE-09 | Count token changes tracked for duplicate balloon numbers | unit | Covered by existing pipeline unit tests in delta_preservation | Exists |
| PIPE-10 | GD&T matched as opaque string (no semantic parsing) | unit | Covered by existing pipeline unit tests in delta_preservation | Exists |
| PIPE-11 | RunAlert created when reviewer's run fails | unit | `pytest tests/test_runs.py::test_failure_alert_created -x` | Wave 0 |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/test_runs.py -x -q`
- **Per wave merge:** `uv run pytest -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_runs.py` — all run/upload/pipeline tests (does not exist yet)
- [ ] `shop/tasks.py` — huey instance needed before tests can import it
- [ ] `shop/models.py` addition of `Run` and `RunAlert` — needed by all run tests
- [ ] `conftest.py` addition: `huey_immediate` fixture for test isolation
- [ ] Framework install: `uv add huey` — not yet in pyproject.toml
- [ ] `shop/routers/runs.py` — router stub needed before route tests can collect

---

## Sources

### Primary (HIGH confidence)

- `https://huey.readthedocs.io/en/latest/api.html` — SqliteHuey API, constructor params, stdlib-only dependency confirmed
- `https://huey.readthedocs.io/en/latest/consumer.html` — consumer startup command, worker type options
- `https://huey.readthedocs.io/en/latest/signals.html` — signal system API (SIGNAL_COMPLETE, SIGNAL_ERROR, etc.)
- `https://fastapi.tiangolo.com/tutorial/server-sent-events/` — native SSE in FastAPI 0.135.0, EventSourceResponse, ServerSentEvent
- `https://htmx.org/extensions/sse/` — hx-ext="sse", sse-connect, sse-swap, sse-close attributes
- `https://daisyui.com/components/steps/` — steps component HTML structure, color classes
- `fastapi==0.135.1` installed in .venv — confirms native SSE available without sse-starlette
- `huey==2.6.0` PyPI — current version confirmed (January 2026 release)
- `pyproject.toml` — confirmed python-multipart, PyMuPDF, openpyxl all already installed

### Secondary (MEDIUM confidence)

- `https://github.com/pymupdf/PyMuPDF/discussions/1653` — raster PDF detection via image-area coverage; maintainer recommendation; 95% threshold verified
- `https://www.advantch.com/blog/how-to-run-multiple-processes-in-a-single-docker-container` — supervisord.conf pattern for uvicorn + huey_consumer in single container

### Tertiary (LOW confidence)

- WebSearch results on huey + FastAPI DB-polling SSE pattern — no single authoritative source; conclusion derived from combining Huey process boundary constraints with FastAPI SSE polling pattern; flagged for implementation validation

---

## Metadata

**Confidence breakdown:**
- Standard stack (Huey, SSE, DaisyUI): HIGH — versions confirmed from PyPI, official docs, and installed packages
- Architecture patterns (DB-polling SSE, standard POST redirect, HTMX partial validation): HIGH — patterns derived from Phase 1 established conventions + FastAPI docs
- Raster PDF detection: MEDIUM — PyMuPDF maintainer recommendation; "no failsafe answer" caveat acknowledged; secondary text extraction check added
- Huey SQLite concurrency (single writer): MEDIUM — docs state it's safe for multi-thread/process; single worker (`--workers=1`) eliminates the concern entirely
- Pitfalls: HIGH — most derived from Phase 1 codebase decisions and official HTMX/Huey documentation

**Research date:** 2026-03-03
**Valid until:** 2026-04-03 (stable libraries; Huey and FastAPI release cadences are slow)
