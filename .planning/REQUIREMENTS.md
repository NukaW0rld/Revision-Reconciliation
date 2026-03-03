# Requirements: Delta Preservation

**Defined:** 2026-03-01
**Core Value:** Every characteristic classification is confirmed by a human engineer with image evidence before the change packet becomes an audit artifact — the system accelerates the review, never bypasses it.

---

## v1 Requirements

Requirements for initial release. Each maps to a roadmap phase.

### Authentication (AUTH)

- [x] **AUTH-01**: User can create an account with email and password (admin-initiated; no self-registration)
- [x] **AUTH-02**: User can log in and maintain a session across browser refresh
- [x] **AUTH-03**: User can log out from any page
- [x] **AUTH-04**: Admin can create, view, and deactivate engineer accounts
- [x] **AUTH-05**: Admin role can access shop configuration; engineer role cannot
- [x] **AUTH-06**: Reviewer name and timestamp are permanently attached to every approve, override, and sign-off action

### Shop Setup (SETUP)

- [x] **SETUP-01**: First admin login triggers an undismissable setup wizard (shop name, admin password, first engineer account, Form 3 column mapping sample upload)
- [x] **SETUP-02**: Admin can upload a sample Form 3 Excel and map columns to expected fields (char_no, requirement, reference location) via a UI mapping interface
- [x] **SETUP-03**: Fatal upload errors (unreadable file, empty sheet) are caught before the mapping UI is shown; minor column-mapping issues surface in the mapping UI itself
- [x] **SETUP-04**: System accepts non-contiguous or non-standard characteristic numbering without normalization

### File Upload & Input (UPLOAD)

- [ ] **UPLOAD-01**: Engineer can upload Rev A PDF, Rev B PDF, and Form 3 Excel as a single submission with part metadata (part number, revision level for each PDF, customer name, job number)
- [ ] **UPLOAD-02**: System detects raster (scanned) PDFs at upload time and displays a clear error before the run is submitted
- [ ] **UPLOAD-03**: System detects multi-page PDFs at upload time and prompts the engineer to select which page to analyze before the run starts
- [ ] **UPLOAD-04**: System validates that the Excel file is readable and contains a recognizable sheet structure before accepting the upload
- [ ] **UPLOAD-05**: Revision labels (Rev A letter, Rev B letter) are captured and stored as part of the run record

### Pipeline Execution (PIPE)

- [ ] **PIPE-01**: Engineer can submit a run from the upload form; pipeline executes asynchronously in a worker process separate from the web server
- [ ] **PIPE-02**: Run status displays stage-by-stage progress (Form 3 parsing → balloon detection → text extraction → anchor building → alignment → candidate matching → classification → output) with current stage name and completion indicator
- [ ] **PIPE-03**: System distinguishes run states: queued, running (with current stage), completed, failed
- [ ] **PIPE-04**: If Rev A balloon detection fails entirely, the run hard-fails with a clear error message identifying the failure stage
- [ ] **PIPE-05**: If Rev B balloon detection fails entirely, the run completes with a partial-result warning; engineer decides whether the output is usable before reviewing
- [ ] **PIPE-06**: If alignment produces uniformly low confidence across all characteristics, the run enters a warning state; engineer sees the confidence distribution and explicitly chooses to proceed or abort before the review queue opens
- [ ] **PIPE-07**: Characteristics not located in Rev B are auto-classified as removed if spatial match confidence ≥ 0.9; below 0.9 they appear in the review queue as unresolved, requiring engineer judgment
- [ ] **PIPE-08**: Characteristics where a balloon is found in Rev B but text extraction fails are classified as low-confidence unchanged if spatial confidence ≥ 0.9; if spatial confidence is also below 0.9, they appear as unresolved
- [ ] **PIPE-09**: Duplicate balloon numbers (e.g., "4X Ø 0.5") are treated as one logical characteristic; count token changes between revisions are tracked and surfaced as a classification signal
- [ ] **PIPE-10**: GD&T feature control frames are matched as opaque strings; semantic parsing of symbols, datum references, and modifiers is not performed in v1
- [ ] **PIPE-11**: Assigned reviewer receives an in-app alert when a run they are responsible for fails

### Review Queue (REVIEW)

- [ ] **REVIEW-01**: Engineer can open the review queue for any completed run and see all characteristics listed with their pipeline classification and confidence score
- [ ] **REVIEW-02**: Each review queue item displays: Rev A snippet (top) and Rev B snippet (bottom) stacked vertically with clear "Rev A" / "Rev B" labels on the left panel; characteristic number, system classification, confidence score, requirement text (Rev A → Rev B), Approve button, and Override dropdown with required note field on the right panel
- [ ] **REVIEW-03**: Engineer can approve a characteristic classification (confirming the system's call); approval is saved immediately to the server
- [ ] **REVIEW-04**: Engineer can override a classification to any valid state (unchanged, changed, removed, added, uncertain); an override note is required and must be non-empty before the override can be saved
- [ ] **REVIEW-05**: Review decisions are persisted server-side as each decision is made; engineer can close the browser and resume the review from the same state on any device
- [ ] **REVIEW-06**: Run can be assigned to any engineer in the shop; submitter is the default assignee; admin can reassign
- [ ] **REVIEW-07**: Review queue shows a live count of items by state (pending, approved, overridden) so the engineer knows how much remains

### Sign-Off (SIGNOFF)

- [ ] **SIGNOFF-01**: Sign-off action is only available when every characteristic has a terminal state (approved or overridden); the sign-off button is disabled and shows remaining count while any item is pending
- [ ] **SIGNOFF-02**: Sign-off and audit packet generation are atomic: if PDF generation fails, the sign-off is rolled back and the run remains in reviewable state with a clear error; no run can exist in a signed-but-no-packet state
- [ ] **SIGNOFF-03**: Signed audit packet is immutable; the original packet file is never overwritten after sign-off

### Audit Packet Export (PACKET)

- [ ] **PACKET-01**: Audit packet is exported as PDF with: formal cover page (part number, revision levels, customer, run ID, reviewer name, sign-off timestamp), summary table of all characteristics, and a detail section with one card per characteristic (system classification, reviewer decision, override note if any, Rev A snippet, Rev B snippet)
- [ ] **PACKET-02**: Audit packet is also exported as CSV with one row per characteristic (char number, Rev A requirement, Rev B requirement, system classification, reviewer decision, override note, reviewer name, timestamp)
- [ ] **PACKET-03**: Signed audit packet can be re-downloaded from the run history at any time after finalization

### Partial FAI Work Order (WORK)

- [ ] **WORK-01**: After a run is finalized, engineer can generate a partial FAI work order on demand as a separate action from sign-off
- [ ] **WORK-02**: Work order lists only characteristics classified as changed or added (the scope requiring new measurement)
- [ ] **WORK-03**: Each work order row includes: characteristic number, requirement text (from drawing), drawing reference (balloon number, page, zone), and a priority flag distinguishing changed (re-measure) from added (new measurement)
- [ ] **WORK-04**: Work order is exported as PDF and CSV

### Run History & Management (HISTORY)

- [ ] **HISTORY-01**: Engineer can view a list of all runs for the shop, filterable by part number and date
- [ ] **HISTORY-02**: Engineer can reopen any finalized run to view its review decisions (read-only unless an amendment is started)
- [ ] **HISTORY-03**: Finalized runs are retained indefinitely on the server
- [ ] **HISTORY-04**: Admin can configure an auto-cleanup period for unfinished and failed runs (e.g., delete after 30 days)

### Amendment Model (AMEND)

- [ ] **AMEND-01**: Engineer can reopen a finalized run to create an amendment; the original signed packet is preserved unchanged
- [ ] **AMEND-02**: Amendment scope is limited to review decisions only; input files (Rev A PDF, Rev B PDF, Form 3 Excel) cannot be changed in an amendment
- [ ] **AMEND-03**: Amendment requires its own sign-off, producing a new versioned audit packet; both the original and amendment packet are accessible from the run record

### Deployment (DEPLOY)

- [x] **DEPLOY-01**: Application runs via `docker compose up` from a single `docker-compose.yml`; no external services required beyond what is bundled in the Docker image
- [ ] **DEPLOY-02**: All pipeline dependencies (PyMuPDF, OpenCV, NumPy) and web dependencies (FastAPI, Dramatiq, WeasyPrint) are bundled in the Docker image
- [x] **DEPLOY-03**: Application functions in an air-gapped shop network with no outbound internet access required at runtime

---

## v2 Requirements

Deferred to a future release. Tracked but not in the current roadmap.

### Workflow Enhancements

- **UX-01**: Pre-review confidence distribution screen showing a histogram of all characteristics' confidence scores before the review queue opens, with explicit proceed/abort decision
- **UX-02**: Inline text diff on requirement text in the review card (highlight added/removed/changed substrings between Rev A and Rev B requirement strings)
- **UX-03**: Explicit run assignment UI allowing submitter to assign a run to any engineer at submission time

### Admin Controls

- **ADMIN-01**: Admin-configurable confidence threshold for the auto-classification gate (currently hardcoded at 0.9); surfaced in admin settings panel after pilot calibration

---

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| GD&T semantic parsing (feature control frame symbol/datum/modifier) | Multi-month engineering investment; string matching catches value changes; defer to v2 after pilot validates scope |
| Concurrent multi-user review of the same run | State conflict complexity; one reviewer per run keeps audit record unambiguous |
| Email notifications for run failure or completion | Adds SMTP dependency; in-app alerts sufficient for O(10) engineer shop |
| Auto-update mechanism | Incompatible with air-gapped networks; unexpected updates in a production QC environment are a compliance risk |
| Cross-revision timeline view (Rev A→B, B→C, C→D history) | Analytics feature; individual run comparisons cover v1 need |
| QMS direct API integration (ETQ, IQS, etc.) | CSV export is v1 integration surface; direct connectors require per-customer work |
| OAuth / SSO login | Requires external identity provider; incompatible with air-gapped shop constraint |
| Bulk approve / select-all | Destroys per-item accountability; audit trail must record individual confirmation of each characteristic |
| Editable requirement text in review UI | Creates ambiguity between what the drawing says and what the system read; extraction errors require a new run |
| Auto-sign-off when all items approved | Sign-off is a formal attestation requiring deliberate intent; must be an explicit action |

---

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| AUTH-01 through AUTH-06 | Phase 1 — Foundation | Pending |
| SETUP-01 through SETUP-04 | Phase 1 — Foundation | Pending |
| DEPLOY-01 through DEPLOY-03 | Phase 1 — Foundation | Pending |
| UPLOAD-01 through UPLOAD-05 | Phase 2 — Pipeline Bridge | Pending |
| PIPE-01 through PIPE-11 | Phase 2 — Pipeline Bridge | Pending |
| REVIEW-01 through REVIEW-07 | Phase 3 — Review and Sign-Off | Pending |
| SIGNOFF-01 through SIGNOFF-03 | Phase 3 — Review and Sign-Off | Pending |
| PACKET-01 through PACKET-03 | Phase 4 — Exports, History, and Amendments | Pending |
| WORK-01 through WORK-04 | Phase 4 — Exports, History, and Amendments | Pending |
| HISTORY-01 through HISTORY-04 | Phase 4 — Exports, History, and Amendments | Pending |
| AMEND-01 through AMEND-03 | Phase 4 — Exports, History, and Amendments | Pending |

**Coverage:**
- v1 requirements: 53 total
- Mapped to phases: 53
- Unmapped: 0

**By phase:**
- Phase 1 (Foundation): AUTH-01..06, SETUP-01..04, DEPLOY-01..03 = 13 requirements
- Phase 2 (Pipeline Bridge): UPLOAD-01..05, PIPE-01..11 = 16 requirements
- Phase 3 (Review and Sign-Off): REVIEW-01..07, SIGNOFF-01..03 = 10 requirements
- Phase 4 (Exports, History, and Amendments): PACKET-01..03, WORK-01..04, HISTORY-01..04, AMEND-01..03 = 14 requirements

---
*Requirements defined: 2026-03-01*
*Last updated: 2026-03-01 after roadmap creation*
