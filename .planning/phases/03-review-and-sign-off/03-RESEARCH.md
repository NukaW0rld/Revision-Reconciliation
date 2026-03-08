# Phase 3: Review and Sign-Off - Research

**Researched:** 2026-03-07
**Domain:** FastAPI + SQLAlchemy + HTMX + DaisyUI review queue with atomic sign-off
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Queue ordering:** char_no ascending — matches engineering drawing and Form 3; no sidebar/jump-to nav; single scroll with filter
- **Filtering:** two combined dropdowns — review status (all / pending / approved / overridden) and pipeline classification (all / unchanged / changed / removed / uncertain / added)
- **Sign-off gate counts all items regardless of active filter** — no filter can hide a pending item
- **Sign-off flow:** Confirmation modal → redirect to generating status page (SSE/polling) → redirect to run summary with download link; on failure: error alert on review queue, run stays reviewable
- **Card traversal:** scrollable list; all items visible simultaneously; full snippets inline (no collapse/expand)
- **Resolved cards:** green/muted left border + status badge replaces action buttons; HTMX partial swap after each save
- **Removed items:** still show Rev B snippet cropped at homography-predicted location; engineer sees that the balloon is gone
- **Missing/failed snippets:** "Image unavailable" placeholder; removed items with no predicted bbox get "Not found in Rev B" gray placeholder
- **Snippet interaction:** click any snippet to open full-size in a modal overlay
- **Standard HTML POST** for major navigation (sign-off confirmation → redirect) — per Phase 01 decision
- **HTMX POST** for incremental saves (approve/override per item) — per Phase 02 pattern
- `db.expire(run) + db.refresh(run)` required inside SSE polling loops to bypass SQLAlchemy identity map cache
- `reviewer_id` is already set on Run at submission time

### Claude's Discretion
- Exact DaisyUI component choices for filter dropdowns and modal
- Polling interval and timeout for the generating status page
- Exact CSS treatment of resolved card border (color intensity, width)
- Override dropdown option ordering within the right panel
- Loading spinner placement during HTMX saves

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| REVIEW-01 | Engineer can open review queue for any completed run; see all characteristics with pipeline classification and confidence score | DB model, router, template patterns from Phases 1-2 |
| REVIEW-02 | Each item shows Rev A snippet (top) / Rev B snippet (bottom) stacked; char_no, classification, confidence, requirement text; Approve button and Override dropdown with required note | HTMX partial swap pattern; DaisyUI card layout; snippet file serving |
| REVIEW-03 | Engineer can approve a classification; saved immediately to server | HTMX POST to review router; ReviewItem upsert pattern |
| REVIEW-04 | Engineer can override to any valid state; override note required and non-empty | HTML5 required validation; server-side validation; HTMX partial swap |
| REVIEW-05 | Review decisions persisted server-side; engineer can close browser and resume from same state on any device | ReviewItem DB model with per-item persist |
| REVIEW-06 | Run can be assigned to any shop engineer; submitter is default; admin can reassign | Run.reviewer_id already exists; admin reassign UI is new |
| REVIEW-07 | Review queue shows live count of items by state (pending / approved / overridden) | Count queries on ReviewItem; HTMX OOB swap or full partial re-render |
| SIGNOFF-01 | Sign-off only available when all items are terminal; button disabled with remaining count while any pending | Count check in router; DaisyUI disabled button with counter badge |
| SIGNOFF-02 | Sign-off + audit packet generation are atomic: if PDF fails, sign-off rolled back; no signed-but-no-packet state | Two-phase write pattern; SQLAlchemy transaction; Run.status rollback |
| SIGNOFF-03 | Signed audit packet is immutable; original packet never overwritten after sign-off | File write with existence check; Run.status gates re-write attempts |
</phase_requirements>

---

## Summary

Phase 3 adds the human review layer between pipeline completion and the final audit artifact. Engineers open a queue showing every characteristic card — classification, evidence snippets, and action controls — then approve or override each one. When all items are resolved, a sign-off produces an immutable audit packet atomically.

The stack is already established: FastAPI + SQLAlchemy + Jinja2 + DaisyUI + HTMX. Phase 3 extends it with a new `ReviewItem` DB model, a `review` router mounted at `/review`, new HTML templates following the `status.html` two-column card pattern, and a two-phase write for sign-off atomicity. The SSE pattern from `runs/status.html` is reused verbatim for the sign-off generating status page.

One critical pipeline gap must be resolved in Wave 0: the current pipeline sets `revB=None` on removed `DeltaItem` records because no predicted bbox is computed. The review card's "removed" snippet requirement depends on a predicted Rev B bbox derived from the anchor location + homography. A pipeline patch is required to populate this before the review UI can display it.

**Primary recommendation:** Build the `ReviewItem` model and HTMX review queue as a single router under `/review`, implement the two-phase sign-off write within a single SQLAlchemy transaction boundary, and patch the pipeline's removed-item snippet generation as a Wave 0 prerequisite.

---

## Standard Stack

All dependencies are already in `pyproject.toml`. No new packages are required for the review queue.

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | current | Router + request handling | Established in Phases 1-2 |
| SQLAlchemy 2.0 | current | ORM + transaction management | Established; two-phase write uses `db.rollback()` |
| Jinja2 | current | Template rendering | Established; DaisyUI card patterns already in use |
| HTMX 2.x | bundled locally | Incremental DOM swaps for approve/override saves | Already bundled at `/static/js/htmx.min.js` |
| htmx-sse | bundled locally | SSE for sign-off generating status page | Already bundled at `/static/js/htmx-sse.js` |
| DaisyUI | compiled via Tailwind | Component library for card, modal, badge, button | Established; `status_badge_class` filter already in `app.py` |

### No New Dependencies Required

The audit packet PDF (SIGNOFF-02) is referenced in STATE.md as WeasyPrint, but WeasyPrint is **not in `pyproject.toml`**. SIGNOFF-02 requires only atomicity of sign-off + packet generation. The actual PDF generation is Phase 4 (PACKET-01). Phase 3 scope is: mark run as signed, write an immutable placeholder/marker, make the sign-off rollback if the Phase 4 packet write fails. Since Phase 4 is not implemented yet, the sign-off in Phase 3 should: (1) transition Run.status to `signed_off`, (2) write an immutable `signed_at` timestamp to the Run record, and (3) treat the audit packet as a Phase 4 concern. The SIGNOFF-02 atomicity guarantee is architectural — enforced by the two-phase write pattern — but the actual PDF generation is deferred.

**Revised interpretation for Phase 3:** "audit packet generation" in SIGNOFF-02 means generating a record/marker that a future Phase 4 PDF export can reference. The immutability guarantee (SIGNOFF-03) is achieved by writing a `signed_at` timestamp and gating any re-write attempts on `Run.status == 'signed_off'`.

### Installation
```bash
# No new packages needed — all in pyproject.toml already
uv sync
```

---

## Architecture Patterns

### Recommended Project Structure (additions to existing)
```
shop/
├── routers/
│   ├── runs.py            # existing
│   └── review.py          # NEW: /review/{run_id} routes
├── services/
│   ├── runs.py            # existing
│   └── review.py          # NEW: ReviewItem CRUD, sign-off atomicity
├── models.py              # ADD: ReviewItem model + Run.status extensions
└── templates/
    └── review/
        ├── queue.html          # Full review queue page
        ├── _item_card.html     # Single ReviewItem card (HTMX swap target)
        ├── _progress_bar.html  # Pending/approved/overridden count bar
        └── generating.html     # Sign-off generating status page (SSE poll)
```

### Pattern 1: ReviewItem DB Model
**What:** One row per (run_id, char_no) pair. Created in bulk when the review queue is first opened. Persisted on each approve/override action.
**When to use:** Always — this is the single source of truth for review state.

```python
# Source: established SQLAlchemy 2.0 mapped_column pattern from shop/models.py
class ReviewItem(Base):
    __tablename__ = "review_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    char_no: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pipeline_classification: Mapped[str] = mapped_column(String)  # unchanged/changed/removed/added/uncertain
    confidence: Mapped[float] = mapped_column(Float)
    requirement_revA: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    requirement_revB: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    revA_snippet_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    revB_snippet_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    revA_bbox: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    revB_bbox: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    reviewer_decision: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # None=pending, approved, overridden
    override_classification: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    override_note: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    reviewed_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    run: Mapped["Run"] = relationship(back_populates="review_items")
    reviewed_by: Mapped[Optional["User"]] = relationship()
```

### Pattern 2: Run Status Extensions
**What:** Phase 3 adds new Run.status values: `reviewing` (queue open), `signing_off` (generating), `signed_off` (terminal).
**When to use:** Status transitions gate UI actions.

```
Completed/Warning → "reviewing" (first GET /review/{run_id} — idempotent)
reviewing → "signing_off" (POST /review/{run_id}/sign-off — confirmation)
signing_off → "signed_off" (after successful atomicity check)
signing_off → "reviewing" (rollback on any failure during sign-off)
```

### Pattern 3: HTMX Partial Swap for Approve/Override
**What:** Each review card has a unique `id="review-item-{char_no}"`. HTMX POST returns the updated card partial, which replaces the card in place. Resolved state (green border + badge) replaces action buttons.
**When to use:** Per-item saves — approve and override.

```html
<!-- In _item_card.html: unresolved state -->
<div id="review-item-{{ item.char_no }}" class="card bg-base-100 shadow border-l-4 border-base-300">
  <form hx-post="/review/{{ run_id }}/items/{{ item.char_no }}/approve"
        hx-target="#review-item-{{ item.char_no }}"
        hx-swap="outerHTML"
        hx-indicator="#spinner-{{ item.char_no }}">
    <button type="submit" class="btn btn-success btn-sm">Approve</button>
  </form>
</div>

<!-- Resolved card returned by HTMX POST -->
<div id="review-item-{{ item.char_no }}" class="card bg-base-100 shadow border-l-4 border-success">
  <span class="badge badge-success">Approved</span>
</div>
```

### Pattern 4: Override Note Validation
**What:** The override note field has `required` HTML attribute. Server also validates non-empty before saving. If empty, return 422 with re-rendered card showing error.
**When to use:** Override POST handler.

```python
# Source: consistent with Phase 1-2 validation pattern (validate → 422 + re-render)
@router.post("/{run_id}/items/{char_no}/override", response_class=HTMLResponse)
def override_item(run_id: int, char_no: int, ...):
    if not override_note or not override_note.strip():
        return templates.TemplateResponse(request, "review/_item_card.html",
            {"item": item, "error": "Override note is required.", "run_id": run_id},
            status_code=422)
```

### Pattern 5: Two-Phase Sign-Off Atomicity (SIGNOFF-02)
**What:** Sign-off writes Run.status = 'signing_off', attempts packet marker write, then either commits to 'signed_off' or rolls back to 'reviewing'.
**When to use:** POST /review/{run_id}/sign-off/confirm — after confirmation modal.

```python
# Source: STATE.md architectural decision — two-phase write pattern
def attempt_sign_off(db: Session, run: Run, reviewer_id: int) -> bool:
    """Atomic sign-off. Returns True on success, False on rollback."""
    run.status = "signing_off"
    db.commit()
    try:
        # Phase 4 PDF generation would go here. For Phase 3, write signed_at marker.
        run.signed_at = datetime.utcnow()
        run.signed_by_id = reviewer_id
        run.status = "signed_off"
        db.commit()
        return True
    except Exception:
        db.rollback()
        run = db.query(Run).filter(Run.id == run.id).first()
        run.status = "reviewing"
        db.commit()
        return False
```

### Pattern 6: Sign-Off Generating Status Page (SSE Reuse)
**What:** After confirmation POST → redirect to `/review/{run_id}/generating` → page polls with same SSE pattern as `runs/status.html`. Polls Run.status; when `signed_off`, redirects to run summary with download link; when `reviewing` (rollback), shows error and redirects back to queue.
**When to use:** Post-confirmation flow.

```python
# Source: run_sse() pattern from shop/routers/runs.py — db.expire + db.refresh required
@router.get("/{run_id}/sse")
async def review_sign_off_sse(run_id: int, ...) -> AsyncIterable[ServerSentEvent]:
    terminal = {"signed_off", "reviewing"}  # reviewing = rolled back
    run = db.query(Run).filter(Run.id == run_id).first()
    while True:
        db.expire(run)
        db.refresh(run)
        payload = _json.dumps({"status": run.status})
        yield ServerSentEvent(raw_data=payload, event="status_update")
        if run.status in terminal:
            yield ServerSentEvent(event="close", raw_data="done")
            break
        await asyncio.sleep(2.0)
```

### Pattern 7: Bulk ReviewItem Population
**What:** First GET /review/{run_id} reads `delta_packet.json`, creates ReviewItem rows for all DeltaItems, transitions Run to `reviewing`. Subsequent GETs use existing ReviewItem rows (idempotent).
**When to use:** Queue entry point.

```python
def open_review_queue(db: Session, run: Run) -> list[ReviewItem]:
    """Idempotent: create ReviewItems from delta_packet if not yet created."""
    existing = db.query(ReviewItem).filter(ReviewItem.run_id == run.id).count()
    if existing > 0:
        return db.query(ReviewItem).filter(ReviewItem.run_id == run.id)\
            .order_by(ReviewItem.char_no).all()
    # Load delta_packet.json
    packet_path = Path(run.output_dir) / "delta_packet.json"
    packet = DeltaPacket.model_validate_json(packet_path.read_text())
    items = []
    for delta in sorted(packet.items, key=lambda x: x.char_no or 0):
        item = ReviewItem(
            run_id=run.id,
            char_no=delta.char_no,
            pipeline_classification=delta.status,
            confidence=delta.confidence,
            revA_snippet_path=delta.revA.image_path if delta.revA else None,
            revB_snippet_path=delta.revB.image_path if delta.revB else None,
            revA_bbox=delta.revA.bbox if delta.revA else None,
            revB_bbox=delta.revB.bbox if delta.revB else None,
        )
        db.add(item)
        items.append(item)
    run.status = "reviewing"
    db.commit()
    return items
```

### Anti-Patterns to Avoid
- **Reading delta_packet.json on every request:** Load once into ReviewItem rows on queue open; all subsequent requests query ReviewItem table only.
- **Using HTMX POST for major navigation:** Sign-off confirmation → redirect must be standard HTML POST (Phase 01 decision) — HTMX intercepts redirects as partial swaps.
- **Serving snippet PNGs through a Jinja template read:** Mount output directory as static files or stream via a dedicated `/snippets/{run_id}/{filename}` route.
- **Bulk approve shortcut:** REQUIREMENTS.md explicitly prohibits bulk approve — each item must be individually confirmed. Never implement select-all.
- **Re-opening a signed_off run for re-review:** SIGNOFF-03 requires immutability. Block any status change from signed_off in the router.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Snippet file serving | Custom file streaming endpoint | FastAPI StaticFiles mount on output_dir | Already solved in Phases 1-2 for uploads; same pattern |
| Per-item save confirmation | Custom JS localStorage | HTMX POST + server-side ReviewItem row | SSR persistence is the requirement (REVIEW-05) |
| Sign-off atomicity | Application-level saga | SQLAlchemy transaction + rollback | DB transaction handles rollback atomically |
| Override note validation | Client-only JS validation | HTML `required` attribute + server 422 | Server is authoritative; client is convenience |
| Filter state management | Custom JS URL state manager | HTML form GET with query params | Browser handles state via URL naturally |

**Key insight:** Every piece of review state lives server-side in ReviewItem rows. The browser is a thin view layer. HTMX swaps the DOM; the server is the source of truth.

---

## Common Pitfalls

### Pitfall 1: Snippet Path Resolution
**What goes wrong:** ReviewItem stores `image_path` as relative string from `DeltaPacket` (e.g., `"snippets/char_01_revA_p0.png"`). Template tries to render this as an `<img src>` and gets a 404 because the output directory is not mounted as static.
**Why it happens:** The output directory path (`Run.output_dir`) is an absolute filesystem path, not a URL path.
**How to avoid:** Add a dedicated route `/review/{run_id}/snippets/{filename}` that reads from `Path(run.output_dir) / "snippets" / filename` and returns `FileResponse`. Or mount each run's output dir as a sub-path of static. The route approach is simpler and avoids dynamic StaticFiles mounts.
**Warning signs:** `<img>` tags returning 404 in browser devtools.

### Pitfall 2: HTMX Swapping the Progress Counter
**What goes wrong:** After an approve/override save, the pending count in the header doesn't update — the engineer sees "15 pending" when only 14 remain.
**Why it happens:** HTMX `hx-target` is set to the individual card; the counter is outside the swap region.
**How to avoid:** Use HTMX out-of-band swap (`hx-swap-oob="true"`) on a progress counter partial that is included in every approve/override response. Alternatively, include the counter inside each card's swap region at the top of the queue (re-render the header partial as OOB).
**Warning signs:** Counter stale after HTMX saves.

### Pitfall 3: Run Status Transition Race
**What goes wrong:** Two browser tabs open the same review queue simultaneously; both POST approve actions; the second POST transitions the Run to `signing_off` before the first finishes, causing a conflict.
**Why it happens:** No mutex on Run status transitions in SQLite.
**How to avoid:** REQUIREMENTS.md already scopes to one reviewer per run (concurrent multi-user review is explicitly out of scope). Document this assumption. For the sign-off POST specifically, check `run.status == "reviewing"` before transitioning; return 409 if already signing_off or signed_off.
**Warning signs:** Unexpected 500 or duplicate sign-off submissions.

### Pitfall 4: Removed Item Rev B Snippet is Always Null
**What goes wrong:** The pipeline sets `revB=None` for removed items because no match was found and no predicted bbox is computed. The review card cannot display a Rev B snippet.
**Why it happens:** Confirmed by reading `delta_preservation/cli.py` lines 260-414: `revB_bbox_pdf` remains `None` when `delta_internal.match is None` and `delta_internal.added_span is None` (both are None for removed items).
**How to avoid:** Pipeline must be patched in Wave 0 to compute a predicted Rev B bbox for removed items using the homography transform applied to the Rev A anchor bbox. The transform `H` is available in `run_pipeline()` scope. The predicted bbox is `apply_transform_bbox(revA_bbox_pdf, transform.H)`. This gives the homography-mapped location on Rev B — exactly where the balloon would be if it were still present.
**Warning signs:** All "removed" cards showing "Not found in Rev B" placeholder even when the engineer expects to see the Rev B drawing region.

### Pitfall 5: SQLAlchemy Relationship on ReviewItem Not Added to Run
**What goes wrong:** `Run.review_items` relationship is missing, so `db.query(ReviewItem).filter(ReviewItem.run_id == run.id)` works but cascade delete on Run deletion doesn't.
**Why it happens:** Adding the FK column without updating both sides of the relationship.
**How to avoid:** Add `review_items: Mapped[list["ReviewItem"]] = relationship(back_populates="run", cascade="all, delete-orphan")` to `Run` model when adding the `ReviewItem` model.

### Pitfall 6: Sign-Off Generates a "Signed But No Packet" State During Testing
**What goes wrong:** A test verifies that `run.status == "signed_off"` without checking whether the atomic write succeeded. In production, if sign-off commits but a subsequent packet write step fails, the state is corrupt.
**Why it happens:** The two-phase write isn't exercised end-to-end in tests.
**How to avoid:** Write a test that patches the packet-write step to raise an exception and asserts that Run.status is rolled back to "reviewing". This is the primary SIGNOFF-02 test.

---

## Code Examples

Verified patterns from existing codebase:

### Queue Entry Route (GET /review/{run_id})
```python
# Source: runs.py run_status() pattern — GET + template render + nav context
@router.get("/{run_id}", response_class=HTMLResponse)
def review_queue(run_id: int, request: Request,
                 status_filter: str = "all", classification_filter: str = "all",
                 db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    run = db.query(Run).filter(Run.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404)
    items = open_review_queue(db, run)  # idempotent: creates ReviewItems if needed
    # Apply filters
    if status_filter != "all":
        items = [i for i in items if i.reviewer_decision == status_filter]
    if classification_filter != "all":
        items = [i for i in items if i.pipeline_classification == classification_filter]
    # Count totals (unfiltered)
    all_items = db.query(ReviewItem).filter(ReviewItem.run_id == run_id).all()
    pending = sum(1 for i in all_items if i.reviewer_decision is None)
    approved = sum(1 for i in all_items if i.reviewer_decision == "approved")
    overridden = sum(1 for i in all_items if i.reviewer_decision == "overridden")
    nav = _get_nav_context(db, user)
    return templates.TemplateResponse(request, "review/queue.html", {
        "run": run, "items": items, "user": user,
        "pending": pending, "approved": approved, "overridden": overridden,
        "total": len(all_items), "status_filter": status_filter,
        "classification_filter": classification_filter, **nav,
    })
```

### Approve Route (HTMX POST, returns card partial)
```python
# Source: Phase 02 alert dismiss pattern — HTMX POST returns partial HTML
@router.post("/{run_id}/items/{char_no}/approve", response_class=HTMLResponse)
def approve_item(run_id: int, char_no: int, request: Request,
                 db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    item = db.query(ReviewItem).filter(
        ReviewItem.run_id == run_id, ReviewItem.char_no == char_no
    ).first()
    if item is None:
        raise HTTPException(status_code=404)
    item.reviewer_decision = "approved"
    item.reviewed_by_id = user.id
    item.reviewed_at = datetime.utcnow()
    db.commit()
    # Return updated card partial + OOB counter update
    all_items = db.query(ReviewItem).filter(ReviewItem.run_id == run_id).all()
    pending = sum(1 for i in all_items if i.reviewer_decision is None)
    return templates.TemplateResponse(request, "review/_item_card.html", {
        "item": item, "run_id": run_id, "pending": pending, "total": len(all_items),
        "oob_update": True,
    })
```

### Sign-Off Confirmation POST (standard HTML POST → redirect)
```python
# Source: Phase 01 decision — standard HTML POST for major navigation
@router.post("/{run_id}/sign-off/confirm")
def sign_off_confirm(run_id: int, request: Request,
                     db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    run = db.query(Run).filter(Run.id == run_id).first()
    all_items = db.query(ReviewItem).filter(ReviewItem.run_id == run_id).all()
    pending = [i for i in all_items if i.reviewer_decision is None]
    if pending:
        return RedirectResponse(f"/review/{run_id}?error=pending_items", status_code=302)
    success = attempt_sign_off(db, run, user.id)
    if success:
        return RedirectResponse(f"/review/{run_id}/generating", status_code=302)
    else:
        return RedirectResponse(f"/review/{run_id}?error=sign_off_failed", status_code=302)
```

### Pipeline Patch for Removed Item Rev B Bbox
```python
# Source: cli.py lines 360-414 pattern — add this branch for removed items
# Add AFTER the match/added_span bbox computation blocks:
elif delta_internal.status == "removed" and anchor is not None and revA_bbox_pdf is not None:
    # Compute predicted Rev B location via homography transform
    from delta_preservation.vision.alignment import apply_transform_bbox
    try:
        revB_bbox_pdf = apply_transform_bbox(revA_bbox_pdf, transform.H)
    except Exception:
        revB_bbox_pdf = None  # Fallback: "Not found in Rev B" placeholder
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Server-side session storage for review state | DB rows (ReviewItem table) | Phase 3 design | Resume from any device (REVIEW-05) |
| Full page reload on each review action | HTMX partial swap | Phase 2 pattern established | No page reload; faster UX |
| Pipeline always sets revB=None for removed items | Pipeline patch to compute predicted bbox | Phase 3 Wave 0 | Enables removed-item snippet in review card |

**Deprecated/outdated:**
- `Run.status` values `queued/running/completed/failed/warning` were the full set in Phase 2. Phase 3 adds `reviewing/signing_off/signed_off`. The `status_badge_class` filter in `app.py` must be extended to handle these new values.

---

## Open Questions

1. **Requirement text source for ReviewItem**
   - What we know: `DeltaPacket.inputs` contains file paths but not requirement text. `DeltaItem` in `types.py` has `reasons` (list of strings) but not a structured `requirement` field. The Form 3 `Characteristic` objects have the requirement text.
   - What's unclear: Is requirement text (Rev A and Rev B) stored anywhere in `delta_packet.json`? The `DeltaItem.reasons` field contains classification reasoning, not the raw requirement string.
   - Recommendation: Add requirement text to the ReviewItem population step by re-reading `form3_chars.json` (already at `<output_dir>/debug/form3_chars.json`) when creating ReviewItem rows. Alternatively, add `requirement_revA` to the `DeltaItem` Pydantic type in a pipeline patch. The debug file approach avoids a pipeline change but adds a dependency on the debug output existing.

2. **Admin reassign UI scope (REVIEW-06)**
   - What we know: `Run.reviewer_id` already exists; the admin router is at `/admin`. REVIEW-06 requires admin can reassign.
   - What's unclear: Whether reassignment UI belongs in the admin panel or on the review queue page itself.
   - Recommendation: Put a simple reassign form on the run status page (`/runs/{run_id}`) accessible only to admin role, consistent with how admin-gated actions are handled in Phase 1.

3. **Snippet route vs. StaticFiles mount**
   - What we know: `Run.output_dir` is an absolute path like `/app/data/out/part1_2026-...`. StaticFiles requires a known directory at startup; run output directories are dynamic.
   - What's unclear: Whether dynamically mounting new StaticFiles routes per run is acceptable in FastAPI.
   - Recommendation: Use a dedicated `/review/{run_id}/snippets/{filename}` route that resolves to `Path(run.output_dir) / "snippets" / filename` and returns `FileResponse`. This is simpler and avoids dynamic mount complexity. Validate the filename is a `.png` to prevent path traversal.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | `pyproject.toml` — `[tool.pytest.ini_options]` testpaths=["tests"] addopts="-q" |
| Quick run command | `pytest tests/test_review.py -x -q` |
| Full suite command | `pytest -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REVIEW-01 | GET /review/{run_id} returns 200 with all items listed | integration | `pytest tests/test_review.py::test_review_queue_loads -x` | Wave 0 |
| REVIEW-02 | Review card HTML contains snippet img, char_no, classification, approve/override controls | integration | `pytest tests/test_review.py::test_review_item_card_html -x` | Wave 0 |
| REVIEW-03 | POST /review/{run_id}/items/{char_no}/approve saves ReviewItem.reviewer_decision='approved' | integration | `pytest tests/test_review.py::test_approve_item -x` | Wave 0 |
| REVIEW-04 | POST override with empty note returns 422; non-empty note saves override_classification | integration | `pytest tests/test_review.py::test_override_requires_note -x` | Wave 0 |
| REVIEW-05 | ReviewItem rows survive across requests; second GET returns same state | integration | `pytest tests/test_review.py::test_review_state_persisted -x` | Wave 0 |
| REVIEW-06 | Admin can reassign reviewer; engineer cannot | integration | `pytest tests/test_review.py::test_admin_can_reassign -x` | Wave 0 |
| REVIEW-07 | Queue response contains pending/approved/overridden counts | integration | `pytest tests/test_review.py::test_review_counts -x` | Wave 0 |
| SIGNOFF-01 | Sign-off button disabled when pending > 0; enabled when all resolved | integration | `pytest tests/test_review.py::test_sign_off_gate -x` | Wave 0 |
| SIGNOFF-02 | If sign-off packet write fails, Run.status rolls back to 'reviewing'; no signed_off state exists | integration | `pytest tests/test_review.py::test_sign_off_rollback -x` | Wave 0 |
| SIGNOFF-03 | Run.signed_at is immutable after sign-off; second sign-off attempt is blocked | integration | `pytest tests/test_review.py::test_signed_off_immutable -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_review.py -x -q`
- **Per wave merge:** `pytest -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_review.py` — all 10 requirement tests listed above
- [ ] `shop/models.py` — `ReviewItem` model + `Run.review_items` relationship + `Run.signed_at`/`Run.signed_by_id` columns
- [ ] Pipeline patch in `delta_preservation/cli.py` — predicted Rev B bbox for removed items

*(conftest.py has all needed fixtures: `client`, `db_engine`, `engineer_user`, `admin_user`, `huey_immediate`)*

---

## Sources

### Primary (HIGH confidence)
- Direct codebase read: `shop/models.py` — existing Run, User, RunAlert models; SQLAlchemy 2.0 `mapped_column` pattern confirmed
- Direct codebase read: `shop/routers/runs.py` — `_get_nav_context()`, SSE pattern, `run_sse()`, HTMX POST pattern
- Direct codebase read: `shop/templates/runs/status.html` — two-column card layout, SSE JS listener
- Direct codebase read: `delta_preservation/cli.py` lines 250-490 — Evidence construction for all DeltaItem statuses; confirmed `revB_bbox_pdf = None` for removed items
- Direct codebase read: `delta_preservation/types.py` — `DeltaItem`, `Evidence`, `DeltaPacket` Pydantic models
- Direct codebase read: `tests/conftest.py` — test fixture inventory
- Direct codebase read: `.planning/phases/03-review-and-sign-off/03-CONTEXT.md` — locked decisions
- Direct codebase read: `.planning/REQUIREMENTS.md` — REVIEW-01..07, SIGNOFF-01..03
- Direct codebase read: `.planning/STATE.md` — architectural decisions, two-phase write requirement

### Secondary (MEDIUM confidence)
- HTMX 2.x OOB swap behavior: documented in htmx.org (not re-verified via WebFetch, but `hx-swap-oob` is a stable HTMX 1.x+ feature used in prior phases)
- DaisyUI component names (card, badge, btn, modal): consistent with what's already compiled in the project's Tailwind output

### Tertiary (LOW confidence)
- None — all critical claims verified from codebase or authoritative project documents

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies already in pyproject.toml; confirmed from codebase
- Architecture: HIGH — patterns derived directly from existing Phase 1-2 code; no external library API questions
- Pitfalls: HIGH — Pitfall 4 (removed item snippet) is directly confirmed by reading cli.py; others are derived from established SQLAlchemy/HTMX behavior patterns
- Pipeline gap: HIGH — confirmed by code reading that `revB_bbox_pdf` is `None` for removed items

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable stack; no fast-moving external dependencies introduced)
