# Phase 3: Review and Sign-Off - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Per-item review queue with image evidence, approve/override controls with mandatory notes, and atomic sign-off that produces an immutable signed audit packet. Creating, downloading, or re-exporting the audit packet as a formal deliverable is Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Queue ordering and filtering
- Default order: by char_no ascending — matches the engineering drawing and Form 3; engineers think in balloon numbers
- Filtering: two combined dropdowns — one for review status (all / pending / approved / overridden), one for pipeline classification (all / unchanged / changed / removed / uncertain / added)
- Sign-off gate always counts all items regardless of active filter — no filter can hide a pending item from the completeness check
- No sidebar or jump-to nav — single scroll; filter reduces the visible list

### Sign-off flow
- Confirmation modal before triggering sign-off — modal shows run summary (part number, revision, reviewer, item count); engineer clicks Confirm to proceed
- After confirmation: redirect to a generating status page that polls for completion (same SSE/polling pattern as pipeline progress)
- After successful packet generation: redirect to run summary page with a prominent download link for the audit packet
- If PDF generation fails and sign-off is rolled back: error alert shown on the review queue; queue remains fully interactive; run stays reviewable

### Item traversal and card state
- Scrollable list — all items visible simultaneously; full Rev A and Rev B snippets inline in each card (no collapse/expand)
- Resolved cards show green/muted left border + status badge (e.g., "Approved" or "Overridden") in place of action buttons
- Unresolved cards show Approve button and Override dropdown with required note field
- Card state updates via HTMX partial swap after each approve/override save

### Snippet display — removed items
- "Removed" characteristics still show a Rev B snippet — cropped at the homography-predicted location on the Rev B PDF, so engineers can see with their own eyes that the balloon is gone from the drawing
- The predicted location is already computed by the pipeline; the snippet must be generated at that predicted Rev B bbox even when no match was found

### Snippet display — missing/failed snippets
- If a snippet PNG is missing or unreadable: simple "Image unavailable" placeholder (muted box, no coordinates)
- For removed items where no predicted bbox exists: gray placeholder with label "Not found in Rev B"

### Snippet interaction
- Click any snippet image to open it full-size in a modal overlay — engineering drawing crops at 300 DPI contain detail that card thumbnails cannot show

### Claude's Discretion
- Exact DaisyUI component choices for the filter dropdowns and modal
- Polling interval and timeout for the generating status page
- Exact CSS treatment of the resolved card border (color intensity, width)
- Override dropdown option ordering within the right panel
- Loading spinner placement during HTMX saves

</decisions>

<specifics>
## Specific Ideas

- The status.html two-column layout (left panel / right panel with cards) is the established pattern — review queue extends this
- The SSE polling pattern from the pipeline progress page is the right model for the sign-off generating status page
- Resolved card state: green border + badge replaces buttons (not a subtle background shift) — the visual change should be unambiguous so engineers don't accidentally re-review a completed item

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Run` model (`shop/models.py`): has `reviewer_id`, `status`, `output_dir` — `output_dir` points to `delta_packet.json` which is the source of all ReviewItem data
- `_get_nav_context()` (`shop/routers/runs.py`): reusable helper for nav bar queries — use in review router
- DaisyUI card pattern from `runs/status.html`: established two-column card layout (left panel + right panel)
- HTMX SSE pattern from `runs/status.html` + `run_sse()`: reusable for sign-off generating status polling
- `runs/_stage_checklist.html` include pattern: reusable HTMX partial include approach

### Established Patterns
- Standard HTML POST (not HTMX) for major navigation actions (sign-off confirmation → redirect) — per Phase 01 decision
- HTMX POST for incremental saves (approve/override per item) — consistent with Phase 02 alert dismiss pattern
- `db.expire(run) + db.refresh(run)` required inside SSE polling loops to bypass SQLAlchemy identity map cache
- `reviewer_id` is already set on Run at submission time (submitter = default reviewer)

### Integration Points
- `GET /review/{run_id}` — placeholder route already referenced in `runs/status.html` and `runs/status.html` warning acknowledgment buttons; Phase 3 implements this route
- `Run.output_dir` → `<output_dir>/delta_packet.json` → `DeltaItem` list → source data for ReviewItem records
- New DB model needed: `ReviewItem` (char_no, run_id, pipeline_classification, confidence, reviewer_decision, override_note, reviewed_by_id, reviewed_at)
- New `Run.status` values needed: `reviewing` (queue open), `signing_off` (generating packet), `signed_off` (terminal)
- Snippet files already at `<output_dir>/snippets/*.png` — review card reads these directly
- Predicted Rev B bbox for removed items must be preserved in `delta_packet.json` or `DeltaItem` — verify pipeline output includes this before planning the snippet generation task

### Concerns to Verify Before Planning
- WeasyPrint rendering with actual 300 DPI engineering PNG crops is unverified (noted in STATE.md blockers) — run a proof-of-concept before finalizing the audit packet PDF plan
- Pipeline `DeltaItem` output for removed characteristics: confirm whether `revB` Evidence field is populated with the predicted bbox or left null — this drives the removed-item snippet approach

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-review-and-sign-off*
*Context gathered: 2026-03-07*
