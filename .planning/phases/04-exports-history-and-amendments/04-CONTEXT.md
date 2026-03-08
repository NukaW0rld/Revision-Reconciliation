# Phase 4: Exports, History, and Amendments - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Download formal deliverables (audit packet PDF/CSV + partial FAI work order PDF/CSV), browse past runs with read-only review access, admin-configurable retention controls for unfinished/failed runs, and an amendment model that allows correcting finalized reviews while preserving the original signed packet unchanged. Input files (Rev A, Rev B, Form 3) cannot be changed in an amendment; a new run is required for that.

</domain>

<decisions>
## Implementation Decisions

### Audit packet PDF structure
- Cover page: formal header block — part number, revision levels (A→B), customer, job number, run ID, reviewer name, sign-off timestamp, shop name
- Summary table (after cover): one row per characteristic showing char #, pipeline classification, reviewer decision, override note (truncated if long); reviewer name and timestamp in the header block, not repeated per row
- Per-item detail section: one item per page — Rev A snippet and Rev B snippet + a compact text block (char #, requirement text Rev A and Rev B, pipeline classification + confidence, reviewer decision, override note if any); reviewer name and timestamp not repeated per card
- Audit packet CSV: one row per characteristic — char number, Rev A requirement, Rev B requirement, system classification, reviewer decision, override note, reviewer name, timestamp

### Work order format and trigger
- Trigger: "Generate Work Order" button on the run summary page, alongside the "Download Audit Packet" button — available for all signed-off runs on demand
- Generation: synchronous — instant download response (no queue step; work order has no image crops and is lightweight)
- Work order lists only changed and added characteristics (WORK-02)
- Priority flag: text label "RE-MEASURE" for changed, "NEW" for added — in a dedicated Priority column, not color-coded (must be unambiguous on monochrome shop printers)
- Table columns: Char #, Priority, Requirement (Rev B), Drawing Reference (balloon number/zone)
- Work order CSV: same structure as PDF table, one row per characteristic

### Amendment UX and versioning
- Trigger: "Amend" button on the run summary page alongside the Download button; clicking shows a confirmation modal ("This will create an amendment. The original signed packet is preserved unchanged.")
- Scope: any finalized (signed_off) run can be amended — no restriction to only the most recent version
- Amendment scope: review decisions only; input files (Rev A PDF, Rev B PDF, Form 3 Excel) are locked and cannot be changed
- Review queue when amendment is active: pre-filled with original reviewer decisions (approved/overridden state preserved); engineer modifies only the items that need correcting, then signs off
- Amendment requires its own sign-off, producing a new versioned packet
- Versioning display on run summary page: labeled download links listed below the Download header — e.g., "v1 (original, 2026-03-08)" / "v2 (amendment, 2026-03-10)"; linear list, no tabs
- Both original and all amendment packets remain accessible from the run record permanently

### Admin retention controls
- Location: existing admin settings panel (alongside shop name and Form 3 column mapping) — a new "Run Retention" section with a number-of-days input and a plain-language description of what gets cleaned up
- Deletion scope: files + DB record are fully removed (uploaded PDFs/Excel, pipeline output directory with snippets and delta_packet.json, and the Run DB row)
- Eligible statuses: queued, running, failed — completed runs awaiting review are also eligible; signed_off runs are always exempt and never deleted by auto-cleanup
- Default period: 30 days
- No dry-run/preview step — Claude's Discretion on exact UI layout within the settings panel

### Claude's Discretion
- Exact WeasyPrint HTML template layout for per-item PDF pages (one item per page is locked; exact CSS/spacing within that constraint is Claude's call)
- How to store and reference audit packet PDF path in the Run DB record (new column, separate table, or derived from output_dir)
- How to model amendments in the DB (separate Amendment model, a parent_run_id FK on Run, or another approach)
- Polling/cleanup job mechanism for auto-retention (Huey periodic task, or on-request scan)
- Exact DaisyUI component choices for version list, settings form fields, and amendment modal

</decisions>

<specifics>
## Specific Ideas

- The WeasyPrint proof-of-concept with real 300 DPI engineering PNG crops is still unverified (STATE.md blocker). This must be resolved before finalizing the audit packet PDF template — if WeasyPrint can't render large PNGs inline without distortion, the per-item card layout may need adjustment.
- "RE-MEASURE" / "NEW" text labels in the work order Priority column — these are the exact strings to use, not abbreviations. They must be readable on a black-and-white shop printer.
- Amendment packet versioning: the label format "v1 (original, date)" / "v2 (amendment, date)" should be consistent across the run summary page and the history list.
- The amendment confirmation modal should clearly state the immutability guarantee so engineers understand what "preserved unchanged" means before proceeding.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `shop/services/review.py`: `open_review_queue()` loads ReviewItems from delta_packet.json; amendment can reuse this service with a "pre-fill from prior decisions" mode
- `shop/routers/review.py` + `shop/templates/review/queue.html`: the review queue template can be rendered read-only (controls hidden) for HISTORY-02 and reused for amendment queue with controls active
- `Run.output_dir` (models.py): points to `out/<run_id>/` which contains `delta_packet.json` and `snippets/` — audit packet PDF generation reads snippets from here
- `Run.signed_at`, `Run.signed_by_id`: already exist on the Run model for cover page metadata
- `_get_nav_context()` (runs.py): nav bar helper, reuse in exports/history routes
- DaisyUI modal pattern (from review sign-off confirmation): reuse for amendment confirmation modal

### Established Patterns
- Standard HTML POST (not HTMX) for major actions that result in redirects — amendment initiation follows this pattern
- HTMX POST for incremental saves (review item decisions in the amendment queue)
- WeasyPrint is the PDF engine (declared in Phase 3) — but `weasyprint` is not yet installed; POC is required before the audit packet PDF plan is finalized
- Synchronous FastAPI route returning `StreamingResponse` with `media_type="application/pdf"` for instant download — appropriate for the work order; audit packet re-download from run summary also follows this pattern
- Admin settings routes already exist (`shop/routers/admin.py`) — retention settings extend this router

### Integration Points
- `Run.status == "signed_off"`: the guard condition for exposing Download, Generate Work Order, and Amend buttons
- `delta_packet.json` → `DeltaItem` list → source data for both audit packet detail section and work order (filter `status in ("changed", "added")` for work order)
- `ReviewItem` records (from Phase 3): source for reviewer decisions, override notes, reviewer name, reviewed_at timestamps in the audit packet
- `ShopConfig` (models.py): holds shop name for the audit packet cover page header block
- New DB fields/models needed: audit packet PDF path storage on `Run`, amendment relationship (parent_run_id or separate model), retention period setting in `ShopConfig`

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 04-exports-history-and-amendments*
*Context gathered: 2026-03-07*
