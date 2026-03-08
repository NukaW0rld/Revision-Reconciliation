# Phase 4: Exports, History, and Amendments - Research

**Researched:** 2026-03-07
**Domain:** PDF/CSV export (WeasyPrint), run history UI, amendment versioning model, DB schema evolution, Huey periodic tasks
**Confidence:** HIGH (stack verified via official docs and existing codebase inspection)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Audit packet PDF structure:**
- Cover page: formal header block — part number, revision levels (A→B), customer, job number, run ID, reviewer name, sign-off timestamp, shop name
- Summary table (after cover): one row per characteristic showing char #, pipeline classification, reviewer decision, override note (truncated if long); reviewer name and timestamp in the header block, not repeated per row
- Per-item detail section: one item per page — Rev A snippet and Rev B snippet + a compact text block (char #, requirement text Rev A and Rev B, pipeline classification + confidence, reviewer decision, override note if any); reviewer name and timestamp not repeated per card
- Audit packet CSV: one row per characteristic — char number, Rev A requirement, Rev B requirement, system classification, reviewer decision, override note, reviewer name, timestamp

**Work order format and trigger:**
- Trigger: "Generate Work Order" button on the run summary page, alongside the "Download Audit Packet" button — available for all signed-off runs on demand
- Generation: synchronous — instant download response (no queue step; work order has no image crops and is lightweight)
- Work order lists only changed and added characteristics (WORK-02)
- Priority flag: text label "RE-MEASURE" for changed, "NEW" for added — in a dedicated Priority column, not color-coded (must be unambiguous on monochrome shop printers)
- Table columns: Char #, Priority, Requirement (Rev B), Drawing Reference (balloon number/zone)
- Work order CSV: same structure as PDF table, one row per characteristic

**Amendment UX and versioning:**
- Trigger: "Amend" button on the run summary page; clicking shows a confirmation modal ("This will create an amendment. The original signed packet is preserved unchanged.")
- Scope: any finalized (signed_off) run can be amended — no restriction to only the most recent version
- Amendment scope: review decisions only; input files (Rev A PDF, Rev B PDF, Form 3 Excel) are locked and cannot be changed
- Review queue when amendment is active: pre-filled with original reviewer decisions (approved/overridden state preserved)
- Amendment requires its own sign-off, producing a new versioned packet
- Versioning display: labeled download links — e.g., "v1 (original, 2026-03-08)" / "v2 (amendment, 2026-03-10)"; linear list, no tabs
- Both original and all amendment packets remain accessible from the run record permanently

**Admin retention controls:**
- Location: existing admin settings panel — a new "Run Retention" section with a number-of-days input
- Deletion scope: files + DB record are fully removed (uploaded PDFs/Excel, pipeline output directory with snippets and delta_packet.json, and the Run DB row)
- Eligible statuses: queued, running, failed — completed runs awaiting review are also eligible; signed_off runs are always exempt
- Default period: 30 days
- No dry-run/preview step

### Claude's Discretion
- Exact WeasyPrint HTML template layout for per-item PDF pages (one item per page is locked; exact CSS/spacing within that constraint is Claude's call)
- How to store and reference audit packet PDF path in the Run DB record (new column, separate table, or derived from output_dir)
- How to model amendments in the DB (separate Amendment model, a parent_run_id FK on Run, or another approach)
- Polling/cleanup job mechanism for auto-retention (Huey periodic task, or on-request scan)
- Exact DaisyUI component choices for version list, settings form fields, and amendment modal

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PACKET-01 | Audit packet exported as PDF: cover page, summary table, per-item detail cards | WeasyPrint HTML→PDF with base_url for PNG snippets; Jinja2 template drives cover+cards |
| PACKET-02 | Audit packet exported as CSV: one row per characteristic | Python csv module + StringIO + StreamingResponse with Content-Disposition header |
| PACKET-03 | Signed audit packet re-downloadable from run history at any time | Store PDF path on Run (new column); FileResponse from stored path OR re-render on demand |
| WORK-01 | Engineer can generate partial FAI work order on demand after finalization | New button on `/runs/{id}` status page; synchronous POST → StreamingResponse |
| WORK-02 | Work order lists only changed and added characteristics | Filter `delta_packet.json` items where `status in ("changed", "added")` |
| WORK-03 | Work order rows: char #, requirement, drawing reference, priority flag | Priority = "RE-MEASURE" (changed) or "NEW" (added); WeasyPrint table template |
| WORK-04 | Work order exported as PDF and CSV | Same StreamingResponse pattern as audit packet; no image crops needed |
| HISTORY-01 | Engineer can view all runs, filterable by part number and date | Existing list_runs route already filters by part_number; add date filter |
| HISTORY-02 | Engineer can reopen any finalized run in read-only view | Review router guard: if `signed_off` and no active amendment, render queue read-only |
| HISTORY-03 | Finalized runs retained indefinitely | `signed_off` runs exempt from auto-cleanup; no TTL applied |
| HISTORY-04 | Admin configures auto-cleanup period for unfinished/failed runs | New `retention_days` column on `ShopConfig`; Huey `@periodic_task` or on-request scan |
| AMEND-01 | Engineer can reopen finalized run to create amendment; original packet preserved | New Amendment model or `parent_run_id` FK; new ReviewItem rows cloned from original |
| AMEND-02 | Amendment scope limited to review decisions; input files locked | Amendment routes reject file changes; revA/revB/form3 paths copied from parent |
| AMEND-03 | Amendment requires own sign-off; both original and amendment packets accessible | Versioned packet path storage; version list rendered on run summary page |
</phase_requirements>

## Summary

Phase 4 is primarily a PDF/CSV generation and data-management phase built entirely on top of the existing FastAPI + SQLite + Huey stack. The main new dependency is WeasyPrint for HTML-to-PDF rendering. WeasyPrint 68.1 is the current stable release and requires three system-level apt packages (libpango-1.0-0, libpangoft2-1.0-0, libharfbuzz-subset0) that must be added to the Dockerfile's runtime stage. The open blocker — whether WeasyPrint can render 300 DPI engineering PNG crops cleanly — must be resolved in Wave 0 of planning before committing to the per-item card template layout.

The amendment model involves the most architectural risk. The cleanest approach for this codebase is a `parent_run_id` nullable FK on the existing `Run` model, turning a "run" into an amendment simply by pointing it at its parent. This avoids a separate Amendment table while still being queryable. All amendment runs share the same `part_number`/`customer`/`job_number` and inherit their input file paths from the parent; the review queue clones ReviewItems from the parent's final decisions as starting state. Each signed-off run (parent or amendment) generates its own versioned audit packet stored in the run's `output_dir`.

The DB schema uses `Base.metadata.create_all()` without Alembic — this means new columns on existing tables require a migration workaround. The established pattern in this project is to use SQLite's `ALTER TABLE ADD COLUMN` in a startup migration script, or to add columns with `nullable=True` / `server_default` so `create_all` can handle schema drift. Both approaches are documented below.

**Primary recommendation:** Add WeasyPrint + apt dependencies to Dockerfile; implement amendment as `parent_run_id` FK on Run; store versioned packet paths in a new JSON column on Run; use a Huey `@periodic_task` with `crontab(hour="0")` for daily retention cleanup.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| WeasyPrint | 68.1 | HTML+CSS → PDF rendering | Established in Phase 3 decision; air-gapped compatible, no browser dependency |
| Python csv | stdlib | CSV generation | No extra dependency; StringIO + DictWriter is the idiomatic pattern |
| FastAPI StreamingResponse | existing | Serve generated files as downloads | Already used throughout; `Content-Disposition: attachment` makes browser download |
| SQLAlchemy 2.0 | existing | ORM + schema evolution via `text()` DDL | Already in use; `create_all` handles new tables; manual DDL handles new columns |
| Huey SqliteHuey | existing | Periodic cleanup task | Already running; `@periodic_task(crontab(...))` with no external dependencies |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| io.BytesIO | stdlib | In-memory PDF bytes buffer | WeasyPrint `write_pdf(target=None)` returns bytes; wrap in BytesIO for StreamingResponse |
| io.StringIO | stdlib | In-memory CSV buffer | csv.DictWriter writes to StringIO; seek(0) then StreamingResponse |
| Jinja2 | existing | PDF HTML template rendering | Same template engine used in web app; render to string, pass to WeasyPrint |
| pathlib.Path | stdlib | File path management for packet storage | Already used throughout; `run.output_dir / "packets" / "v1.pdf"` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| WeasyPrint | ReportLab | ReportLab is lower-level Python API; harder to maintain layout; no HTML/CSS input |
| WeasyPrint | pdfkit/wkhtmltopdf | wkhtmltopdf is EOL; not air-gapped friendly; binary dep outside venv |
| `parent_run_id` FK on Run | Separate Amendment model | Separate model is more normalized but adds join complexity; FK is simpler for this scale |
| Huey periodic_task | Startup-time scan on request | On-request scan only cleans when someone logs in; periodic task runs reliably on schedule |
| SQLite ALTER TABLE | Alembic | Alembic not in project; adding it is overkill; manual DDL + `try/except` is the established pattern |

**Installation (pyproject.toml dependency + Dockerfile apt):**
```bash
# pyproject.toml — add to dependencies list:
"weasyprint",

# Dockerfile runtime stage — add to apt-get install line:
libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0
```

## Architecture Patterns

### Recommended Project Structure Additions
```
shop/
├── routers/
│   ├── exports.py          # NEW: audit packet, work order download routes
│   └── history.py          # NEW: run history list (or extend runs.py)
├── services/
│   └── exports.py          # NEW: PDF/CSV generation logic
├── templates/
│   ├── exports/
│   │   ├── audit_packet.html   # WeasyPrint template: cover + summary + cards
│   │   └── work_order.html     # WeasyPrint template: work order table
│   └── runs/
│       └── status.html         # MODIFY: add Download/Work Order/Amend buttons
```

### Pattern 1: WeasyPrint PDF generation (synchronous, in-memory)
**What:** Render a Jinja2 HTML string with WeasyPrint and return bytes
**When to use:** Audit packet PDF, work order PDF — both are synchronous instant downloads

```python
# Source: https://doc.courtbouillon.org/weasyprint/stable/api_reference.html
import io
from weasyprint import HTML
from fastapi.responses import StreamingResponse

def generate_pdf_bytes(html_string: str, base_url: str) -> bytes:
    """Render HTML to PDF bytes. base_url resolves relative <img> paths."""
    return HTML(string=html_string, base_url=base_url).write_pdf()

# In FastAPI route:
pdf_bytes = generate_pdf_bytes(html, base_url=str(run.output_dir))
return StreamingResponse(
    io.BytesIO(pdf_bytes),
    media_type="application/pdf",
    headers={"Content-Disposition": f'attachment; filename="audit_packet_{run_id}.pdf"'},
)
```

**Critical:** `base_url` must point to the directory containing snippet PNGs so WeasyPrint resolves `<img src="snippets/char_001_revA.png">` correctly. Use `str(Path(run.output_dir))` — WeasyPrint accepts `pathlib.Path` objects directly.

### Pattern 2: CSV generation (in-memory, no file on disk)
**What:** Write rows to StringIO, return as StreamingResponse
**When to use:** Audit packet CSV, work order CSV

```python
# Source: Python csv stdlib docs
import csv, io

def generate_csv_bytes(rows: list[dict], fieldnames: list[str]) -> io.StringIO:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)  # CRITICAL: reset position or response is empty
    return output

# In FastAPI route:
buf = generate_csv_bytes(rows, fieldnames=[...])
return StreamingResponse(
    buf,
    media_type="text/csv",
    headers={"Content-Disposition": f'attachment; filename="audit_packet_{run_id}.csv"'},
)
```

### Pattern 3: Amendment model via parent_run_id FK
**What:** A `parent_run_id` nullable FK on `Run` marks a run as an amendment of its parent
**When to use:** Any time `AMEND-01` amendment flow is triggered

DB schema addition:
```python
# In models.py — add to Run:
parent_run_id: Mapped[Optional[int]] = mapped_column(
    ForeignKey("runs.id"), nullable=True
)
# No relationship needed unless we want to query amendments from the parent
```

Amendment creation logic:
```python
# In services/exports.py or a new services/amendments.py
def create_amendment(db: Session, parent_run: Run, initiator_id: int) -> Run:
    """Clone parent run as a new amendment; pre-fill ReviewItems from parent decisions."""
    amendment = Run(
        part_number=parent_run.part_number,
        rev_a_label=parent_run.rev_a_label,
        rev_b_label=parent_run.rev_b_label,
        customer=parent_run.customer,
        job_number=parent_run.job_number,
        status="reviewing",
        output_dir=parent_run.output_dir,   # same pipeline output; no re-run
        revA_path=parent_run.revA_path,
        revB_path=parent_run.revB_path,
        form3_path=parent_run.form3_path,
        reviewer_id=initiator_id,
        parent_run_id=parent_run.id,
    )
    db.add(amendment)
    db.flush()   # get amendment.id before cloning items
    # Clone ReviewItems with existing decisions pre-populated
    for item in parent_run.review_items:
        clone = ReviewItem(
            run_id=amendment.id,
            char_no=item.char_no,
            pipeline_classification=item.pipeline_classification,
            confidence=item.confidence,
            requirement_revA=item.requirement_revA,
            requirement_revB=item.requirement_revB,
            revA_snippet_path=item.revA_snippet_path,
            revB_snippet_path=item.revB_snippet_path,
            revA_bbox=item.revA_bbox,
            revB_bbox=item.revB_bbox,
            reviewer_decision=item.reviewer_decision,      # pre-filled
            override_classification=item.override_classification,
            override_note=item.override_note,
            reviewed_by_id=item.reviewed_by_id,
            reviewed_at=item.reviewed_at,
        )
        db.add(clone)
    db.commit()
    return amendment
```

### Pattern 4: Versioned packet path storage
**What:** Store a JSON list of `{version, type, path, signed_at}` records per run
**When to use:** Audit packet generation at sign-off; re-download; amendment sign-off

Option A (recommended — Claude's Discretion): New JSON column `packet_versions` on `Run`:
```python
# In models.py — add to Run:
packet_versions: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
# Value: [{"version": 1, "type": "original", "path": "/app/out/.../packets/v1.pdf", "signed_at": "..."}]
```

Option B: Derive from `output_dir` + a fixed naming convention (`packets/v1.pdf`, `packets/v2.pdf`). Simpler but harder to query. Option A is preferred because it stores `signed_at` metadata without extra DB queries.

### Pattern 5: Huey periodic_task for retention cleanup
**What:** `@periodic_task(crontab(hour="0", minute="0"))` runs daily; deletes old unfinished runs
**When to use:** HISTORY-04 auto-cleanup

```python
# Source: https://huey.readthedocs.io/en/latest/guide.html
from huey import crontab

@huey.periodic_task(crontab(hour="0", minute="0"))
def cleanup_old_runs():
    """Delete runs older than retention_days that are not signed_off."""
    from shop.database import SessionLocal
    from shop.models import Run, ShopConfig
    import shutil
    from datetime import datetime, timedelta

    db = SessionLocal()
    try:
        config = db.query(ShopConfig).filter(ShopConfig.id == 1).first()
        retention_days = getattr(config, "retention_days", 30) or 30
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        DELETABLE_STATUSES = {"queued", "running", "failed", "completed"}
        old_runs = (
            db.query(Run)
            .filter(Run.status.in_(DELETABLE_STATUSES), Run.submitted_at < cutoff)
            .all()
        )
        for run in old_runs:
            if run.output_dir and Path(run.output_dir).exists():
                shutil.rmtree(run.output_dir, ignore_errors=True)
            # Delete uploaded files
            for path in [run.revA_path, run.revB_path, run.form3_path]:
                if path and Path(path).exists():
                    Path(path).unlink(missing_ok=True)
            db.delete(run)
        db.commit()
    finally:
        db.close()
```

**Critical:** Periodic tasks are only executed by the `huey_consumer` process. The supervisord config already manages `huey_consumer` — no changes needed to supervisord.

### Pattern 6: DB schema evolution without Alembic
**What:** Adding new nullable columns to existing SQLite tables
**When to use:** `parent_run_id` on Run; `packet_versions` on Run; `retention_days` on ShopConfig

The project uses `Base.metadata.create_all()` on startup, which creates missing tables but does NOT add new columns to existing tables. For existing deployments with a real `shop.db`, new columns require an explicit DDL migration.

Established pattern — run at app startup in `database.py`:
```python
# Source: SQLite docs — ALTER TABLE ADD COLUMN
from sqlalchemy import text, inspect

def run_schema_migrations(engine):
    """Add new columns to existing tables if not present."""
    with engine.connect() as conn:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("runs")}
        if "parent_run_id" not in columns:
            conn.execute(text("ALTER TABLE runs ADD COLUMN parent_run_id INTEGER REFERENCES runs(id)"))
        if "packet_versions" not in columns:
            conn.execute(text("ALTER TABLE runs ADD COLUMN packet_versions JSON"))
        columns_cfg = {c["name"] for c in inspector.get_columns("shop_config")}
        if "retention_days" not in columns_cfg:
            conn.execute(text("ALTER TABLE shop_config ADD COLUMN retention_days INTEGER DEFAULT 30"))
        conn.commit()
```

Call `run_schema_migrations(engine)` in `create_app()` after `Base.metadata.create_all(bind=engine)`.

### Pattern 7: Audit packet generation integrated into sign-off atomicity
**What:** The existing `attempt_sign_off` has a Phase 3 placeholder comment "Phase 4 PDF generation would go here"
**When to use:** The two-phase write pattern must wrap PDF generation so sign-off rollback works

```python
# Extend attempt_sign_off in services/review.py:
def attempt_sign_off(db: Session, run: Run, reviewer_id: int) -> bool:
    if run.status == "signed_off":
        return False
    run.status = "signing_off"
    db.commit()
    try:
        # Phase 2: generate audit packet PDF + persist path
        from shop.services.exports import generate_and_store_audit_packet
        generate_and_store_audit_packet(db, run)  # raises on failure
        run.signed_at = datetime.utcnow()
        run.signed_by_id = reviewer_id
        run.status = "signed_off"
        db.commit()
        return True
    except Exception:
        db.rollback()
        run = db.query(Run).filter(Run.id == run.id).first()
        if run is not None:
            run.status = "reviewing"
            db.commit()
        return False
```

### Anti-Patterns to Avoid
- **Generating PDF in a Huey task for sign-off:** The two-phase atomicity pattern requires PDF generation in the sign-off request, not in a background task, so rollback is possible.
- **Storing packet PDFs outside `output_dir`:** All per-run artifacts belong under `out/<run_id>/`. Use `output_dir/packets/v1.pdf` so the entire run artifact set is deleted atomically when cleanup runs.
- **Using color to convey priority in work order:** The decision is "RE-MEASURE"/"NEW" text labels. Do not use colored cells or badge CSS that disappears on monochrome printers.
- **Re-running the pipeline for amendments:** Amendments reuse the existing `delta_packet.json` and snippet files from the parent run. No pipeline re-execution.
- **Forgetting `output.seek(0)` for CSV StreamingResponse:** The StringIO cursor is at the end after writing; seek(0) is mandatory or the response body is empty.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTML → PDF conversion | Custom PDF writer | WeasyPrint 68.1 | CSS page breaks, image embedding, fonts, print media all handled |
| CSV serialization | String concatenation | Python `csv.DictWriter` | Handles quoting, escaping, encoding edge cases |
| Schedule-based cleanup | Custom cron subprocess | `huey.periodic_task(crontab(...))` | Already-running huey_consumer handles scheduling; no new process needed |
| File streaming | Read-all-then-send | `StreamingResponse(BytesIO(...))` | Proper HTTP chunked transfer; allows large files |
| Amendment versioning UI | Custom version tracking table | JSON column on Run | Sufficient for linear versioning at this scale |

**Key insight:** WeasyPrint's print CSS (`@page`, `page-break-after: always`) handles pagination automatically. Every per-item card gets its own page with a single CSS rule — no manual page count math needed.

## Common Pitfalls

### Pitfall 1: WeasyPrint cannot find snippet PNGs
**What goes wrong:** `<img src="snippets/char_001.png">` renders as broken image in PDF
**Why it happens:** WeasyPrint's `base_url` defaults to CWD, not the run's output directory
**How to avoid:** Always pass `base_url=str(Path(run.output_dir))` to `HTML(string=..., base_url=...)`
**Warning signs:** PDF renders but images are missing/broken; check WeasyPrint stderr for "Failed to load image"

### Pitfall 2: Large 300 DPI PNGs cause slow PDF generation or memory issues
**What goes wrong:** Each engineering drawing crop can be 2–5 MB; 50+ items = 100–250 MB in a single PDF
**Why it happens:** WeasyPrint loads all images into memory before writing the PDF
**How to avoid:** Use WeasyPrint's `dpi` option to cap embedded image resolution at 150 DPI for PDF output (visual quality sufficient for review; original 300 DPI PNG preserved on disk). POC in Wave 0 must verify acceptable quality.
**Warning signs:** Sign-off request times out; memory usage spikes during sign-off

### Pitfall 3: Amendment creates ReviewItems on top of existing ones
**What goes wrong:** Amendment run_id conflicts with existing ReviewItems for the parent
**Why it happens:** `open_review_queue()` checks `ReviewItem.run_id == run.id`; amendment has a new `run_id` so it's safe, BUT if the amendment references the parent's `output_dir`, and someone calls `open_review_queue()` on the amendment, it would re-read from `delta_packet.json` instead of using the cloned items
**How to avoid:** The `create_amendment()` service must clone ReviewItems at amendment creation time AND set amendment's initial ReviewItem count > 0 so `open_review_queue()`'s idempotency check short-circuits immediately

### Pitfall 4: SQLite `ALTER TABLE ADD COLUMN` limitations
**What goes wrong:** Attempting to add a NOT NULL column without a DEFAULT to a table with existing rows fails in SQLite
**Why it happens:** SQLite's ALTER TABLE only supports adding nullable columns or columns with a DEFAULT
**How to avoid:** All new columns must be `nullable=True` OR have an explicit `DEFAULT` value in the DDL statement. `parent_run_id` is nullable (fine). `retention_days INTEGER DEFAULT 30` is fine. `packet_versions JSON` (nullable) is fine.
**Warning signs:** `OperationalError: Cannot add a NOT NULL column with no default value`

### Pitfall 5: `signed_off` runs accidentally deleted by cleanup
**What goes wrong:** Retention cleanup deletes signed-off runs
**Why it happens:** Query filters on `submitted_at < cutoff` without checking status
**How to avoid:** The cleanup query must explicitly filter `Run.status.in_({"queued", "running", "failed", "completed"})` — never include `reviewing`, `signing_off`, or `signed_off`

### Pitfall 6: WeasyPrint not installed in Docker runtime
**What goes wrong:** `ImportError: No module named 'weasyprint'` at sign-off time
**Why it happens:** `weasyprint` is not yet in `pyproject.toml`; system deps not in Dockerfile
**How to avoid:** Wave 0 plan must add `"weasyprint"` to `pyproject.toml` dependencies AND add `libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0` to the Dockerfile runtime `apt-get install` line

### Pitfall 7: History list date filter — submitted_at is UTC, UI is local time
**What goes wrong:** Date filter shows wrong results for engineers in non-UTC timezones
**Why it happens:** `Run.submitted_at` is stored as UTC; date comparison in SQL uses UTC midnight
**How to avoid:** For v1, filter by date string substring on `submitted_at` displayed in UTC, with a UI note "dates shown in UTC". Full timezone support is out of scope for v1.

## Code Examples

### WeasyPrint PDF to bytes
```python
# Source: https://doc.courtbouillon.org/weasyprint/stable/api_reference.html
from weasyprint import HTML
from pathlib import Path

def render_pdf(html_string: str, base_url: str | Path) -> bytes:
    """Render HTML string to PDF bytes. Returns bytes directly (no temp file)."""
    return HTML(string=html_string, base_url=str(base_url)).write_pdf()
```

### FastAPI download route pattern
```python
# Source: FastAPI custom response docs
import io
from fastapi.responses import StreamingResponse

@router.get("/{run_id}/download/audit-packet.pdf")
def download_audit_packet(run_id: int, ...):
    pdf_bytes = render_pdf(html, base_url=run.output_dir)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="audit_packet_{run_id}_v{version}.pdf"'
        },
    )
```

### CSV download route pattern
```python
# Source: Python csv stdlib + FastAPI StreamingResponse
import csv, io
from fastapi.responses import StreamingResponse

@router.get("/{run_id}/download/audit-packet.csv")
def download_audit_csv(run_id: int, ...):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "char_no", "requirement_revA", "requirement_revB",
        "pipeline_classification", "reviewer_decision",
        "override_note", "reviewer_name", "reviewed_at",
    ])
    writer.writeheader()
    for item in run.review_items:
        writer.writerow({...})
    output.seek(0)  # must seek before streaming
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="audit_{run_id}.csv"'},
    )
```

### Huey periodic task registration
```python
# Source: https://huey.readthedocs.io/en/latest/guide.html
from huey import crontab
from shop.tasks import huey  # existing SqliteHuey instance

@huey.periodic_task(crontab(hour="0", minute="0"))
def cleanup_old_runs():
    # runs daily at midnight UTC in the huey_consumer process
    ...
```

### WeasyPrint CSS for page-per-item layout
```css
/* Source: CSS paged media spec — WeasyPrint supports @page and page-break */
@page {
    size: Letter;
    margin: 1in;
}

.item-card {
    page-break-after: always;
}

.item-card:last-child {
    page-break-after: avoid;
}

img.snippet {
    max-width: 100%;
    max-height: 3in;
    object-fit: contain;
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| wkhtmltopdf | WeasyPrint | ~2020 (wkhtmltopdf EOL) | WeasyPrint has no headless browser dep; pure Python + Pango |
| ReportLab programmatic PDF | WeasyPrint HTML templates | Project decision Phase 3 | HTML/CSS layout is maintainable; Jinja2 templates are familiar |
| Alembic for migrations | Manual DDL via `ALTER TABLE` | Project convention | This project skips Alembic; startup migration function is the pattern |
| Separate scheduler process | Huey `@periodic_task` | Phase 1 decision (Huey chosen) | huey_consumer already running; zero new infrastructure needed |

**Deprecated/outdated:**
- wkhtmltopdf: EOL, not recommended for new projects
- pdfkit (wkhtmltopdf wrapper): same EOL concern
- passlib: replaced by pwdlib in this project (already done in Phase 1)

## Open Questions

1. **WeasyPrint 300 DPI PNG rendering quality**
   - What we know: WeasyPrint supports Pillow-format PNGs including large files; `dpi` parameter can cap embedded resolution
   - What's unclear: Whether engineering drawing crops (potentially large, high-contrast technical images) degrade visibly at PDF render time; whether memory usage is acceptable for 50+ item runs
   - Recommendation: Wave 0 must include a POC task: install WeasyPrint, render a 5-item test packet with real snippets from `out/`, measure PDF file size and visual quality

2. **Where to store versioned audit packet PDFs**
   - What we know: `Run.output_dir` points to `out/<run_id>/`; amendment runs share the same `output_dir` as parent
   - What's unclear: If amendment shares parent's `output_dir`, versioned PDFs go in `<parent_output_dir>/packets/v1.pdf` and `v2.pdf` — but the amendment Run's `output_dir` must also point to the parent's directory for this to work
   - Recommendation: Store versioned packets under the parent run's `output_dir/packets/`; amendment's `output_dir` = parent's `output_dir`; `packet_versions` JSON column tracks all version metadata

3. **Date filter for HISTORY-01**
   - What we know: `Run.submitted_at` is a DateTime (UTC); existing list_runs route filters by `part_number` already
   - What's unclear: Whether a date range picker or simple year/month dropdowns is sufficient for v1
   - Recommendation: Add a simple `date_from` query parameter (YYYY-MM-DD string) and filter `Run.submitted_at >= date_from`; single input, no complex date picker needed

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/test_exports.py -q` |
| Full suite command | `pytest -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PACKET-01 | PDF bytes generated with cover + items | unit | `pytest tests/test_exports.py::test_audit_packet_pdf_bytes -x` | Wave 0 |
| PACKET-02 | CSV rows match ReviewItems | unit | `pytest tests/test_exports.py::test_audit_packet_csv_rows -x` | Wave 0 |
| PACKET-03 | Re-download returns stored PDF bytes | integration | `pytest tests/test_exports.py::test_audit_packet_redownload -x` | Wave 0 |
| WORK-01 | Work order button visible on signed-off run page | integration | `pytest tests/test_exports.py::test_work_order_button_visible -x` | Wave 0 |
| WORK-02 | Work order contains only changed/added items | unit | `pytest tests/test_exports.py::test_work_order_filters_status -x` | Wave 0 |
| WORK-03 | Work order rows contain RE-MEASURE/NEW labels | unit | `pytest tests/test_exports.py::test_work_order_priority_labels -x` | Wave 0 |
| WORK-04 | Work order PDF and CSV both downloadable | integration | `pytest tests/test_exports.py::test_work_order_pdf_csv -x` | Wave 0 |
| HISTORY-01 | History list filterable by part number and date | integration | `pytest tests/test_history.py::test_history_filters -x` | Wave 0 |
| HISTORY-02 | Signed-off run opens in read-only queue view | integration | `pytest tests/test_history.py::test_signed_off_readonly_view -x` | Wave 0 |
| HISTORY-03 | Signed-off runs exempt from cleanup deletion | unit | `pytest tests/test_history.py::test_cleanup_exempt_signed_off -x` | Wave 0 |
| HISTORY-04 | Cleanup deletes old failed/queued runs | unit | `pytest tests/test_history.py::test_cleanup_deletes_old_runs -x` | Wave 0 |
| AMEND-01 | Amendment creates new Run with pre-filled items | unit | `pytest tests/test_amendments.py::test_create_amendment -x` | Wave 0 |
| AMEND-02 | Amendment input files locked (revA/revB/form3 unchanged) | unit | `pytest tests/test_amendments.py::test_amendment_files_locked -x` | Wave 0 |
| AMEND-03 | Amendment sign-off produces versioned packet; both versions accessible | integration | `pytest tests/test_amendments.py::test_amendment_versioned_packet -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_exports.py tests/test_history.py tests/test_amendments.py -q`
- **Per wave merge:** `pytest -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_exports.py` — covers PACKET-01..03, WORK-01..04
- [ ] `tests/test_history.py` — covers HISTORY-01..04
- [ ] `tests/test_amendments.py` — covers AMEND-01..03
- [ ] WeasyPrint POC: install + render test packet to verify 300 DPI PNG quality
- [ ] Add `"weasyprint"` to `pyproject.toml` dependencies
- [ ] Add `libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0` to Dockerfile runtime apt-get
- [ ] Add schema migration function for new columns: `parent_run_id`, `packet_versions`, `retention_days`

## Sources

### Primary (HIGH confidence)
- WeasyPrint 68.1 official API docs (https://doc.courtbouillon.org/weasyprint/stable/api_reference.html) — HTML class, write_pdf(), base_url, dpi parameter
- WeasyPrint 68.1 first steps docs (https://doc.courtbouillon.org/weasyprint/stable/first_steps.html) — Debian/Docker system dependencies
- Huey 2.6.0 official docs (https://huey.readthedocs.io/en/latest/guide.html) — periodic_task, crontab, SqliteHuey
- Python stdlib csv docs — DictWriter, StringIO pattern
- Existing codebase: `shop/models.py`, `shop/services/review.py`, `shop/tasks.py`, `shop/routers/runs.py` — established patterns for StreamingResponse, two-phase sign-off, Huey task structure

### Secondary (MEDIUM confidence)
- FastAPI custom response docs (https://fastapi.tiangolo.com/advanced/custom-response/) — StreamingResponse with BytesIO
- Alembic batch docs for SQLite (https://alembic.sqlalchemy.org/en/latest/batch.html) — ALTER TABLE ADD COLUMN behavior in SQLite

### Tertiary (LOW confidence)
- WeasyPrint GitHub issues #248, #872 — image size and DPI behavior (community-reported; not verified against 68.1 API)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — WeasyPrint 68.1 confirmed in official docs; all other libraries already in use
- Architecture: HIGH — patterns derived from existing codebase code; WeasyPrint API verified
- Pitfalls: HIGH for SQLite/streaming issues (stdlib behavior); MEDIUM for WeasyPrint DPI/memory (unverified POC)

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (WeasyPrint stable; stack stable)
