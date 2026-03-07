# Roadmap: Delta Preservation

## Overview

The existing Python pipeline is the computation core. This roadmap builds the production shell around it: four phases that take the project from a running Docker container with authenticated users (Phase 1) through pipeline job execution and upload (Phase 2), the core review and sign-off workflow (Phase 3), and finally exports, history, and the amendment model (Phase 4). Each phase delivers a coherent, testable capability. Nothing in Phase N requires anything from Phase N+1.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Running Docker container with FastAPI, SQLite, auth, roles, setup wizard, and deployment configuration (completed 2026-03-03)
- [x] **Phase 2: Pipeline Bridge** - File upload, validation, async job queue, stage-by-stage pipeline progress, and run status tracking (completed 2026-03-04)
- [ ] **Phase 3: Review and Sign-Off** - Per-item review queue with image evidence, approve/override with mandatory notes, hard sign-off gate, and atomic audit packet generation
- [ ] **Phase 4: Exports, History, and Amendments** - Partial FAI work order, run history with reopen and re-download, admin retention controls, and amendment model with versioned packets

## Phase Details

### Phase 1: Foundation
**Goal**: Engineers and admins can securely access a running shop instance with the correct role permissions and shop configuration established
**Depends on**: Nothing (first phase)
**Requirements**: AUTH-01, AUTH-02, AUTH-03, AUTH-04, AUTH-05, AUTH-06, SETUP-01, SETUP-02, SETUP-03, SETUP-04, DEPLOY-01, DEPLOY-02, DEPLOY-03
**Success Criteria** (what must be TRUE):
  1. Admin can log in with email and password; session persists across browser refresh; admin can log out from any page
  2. Admin can create an engineer account; that engineer can log in and access only engineer-facing pages; admin pages are inaccessible to engineer accounts
  3. First admin login triggers the setup wizard; the wizard cannot be dismissed until shop name, admin password, first engineer account, and Form 3 column mapping are complete
  4. Admin can upload a sample Form 3 Excel and map columns to char_no, requirement, and reference location via a UI; fatal file errors surface before the mapping screen; minor mapping issues surface in the mapping UI
  5. Shop runs via `docker compose up` with no external services and no outbound internet access required; all dependencies are bundled in the image
**Plans**: 8 plans

Plans:
- [x] 01-01-PLAN.md — Test infrastructure scaffold (pytest config, conftest, test stubs)
- [x] 01-02-PLAN.md — Core shop package: DB models, auth service, app factory, dependencies
- [ ] 01-03-PLAN.md — Auth routes: login/logout/session + HTMX base template + static assets
- [ ] 01-04-PLAN.md — Admin user management: create/view/deactivate engineers + RBAC enforcement
- [ ] 01-05-PLAN.md — Setup wizard middleware guard + steps 1-3 (shop name, password, engineer)
- [ ] 01-06-PLAN.md — Wizard step 4: Form 3 column mapping UI + admin settings reconfiguration
- [ ] 01-07-PLAN.md — Docker multi-stage build: Tailwind/DaisyUI compile + uv runtime + docker-compose
- [ ] 01-08-PLAN.md — Deploy checkpoint: automated suite + human Docker verification

### Phase 2: Pipeline Bridge
**Goal**: Engineers can submit a drawing comparison run and watch it progress stage by stage to completion or failure, with upload validation catching bad files before the pipeline starts
**Depends on**: Phase 1
**Requirements**: UPLOAD-01, UPLOAD-02, UPLOAD-03, UPLOAD-04, UPLOAD-05, PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05, PIPE-06, PIPE-07, PIPE-08, PIPE-09, PIPE-10, PIPE-11
**Success Criteria** (what must be TRUE):
  1. Engineer can upload Rev A PDF, Rev B PDF, and Form 3 Excel with part metadata (part number, revision letters, customer, job number) as a single submission; revision labels are stored with the run record
  2. Uploading a raster (scanned) PDF shows a clear error before submission; uploading a multi-page PDF prompts the engineer to select which page before the run starts; uploading an unreadable Excel file is rejected with a clear error
  3. After submission, the run status page shows stage-by-stage progress through all 8 pipeline stages with the current stage name highlighted; the page reflects queued, running, completed, and failed states
  4. If Rev A balloon detection fails, the run hard-fails with a clear error message identifying the stage; if Rev B balloon detection fails, the run completes with a partial-result warning the engineer must acknowledge; if alignment is uniformly low confidence, the engineer sees the confidence distribution and explicitly proceeds or aborts before the review queue opens
  5. The assigned reviewer receives an in-app alert when a run they are responsible for fails
**Plans**: 8 plans

Plans:
- [x] 02-01-PLAN.md — DB foundation: Run + RunAlert models, Huey install, shop/tasks.py scaffold, test stubs
- [x] 02-02-PLAN.md — Upload form: /runs/new, PDF validation partials, Excel validation, file save, run creation
- [x] 02-03-PLAN.md — Pipeline task worker: stage_callback in cli.py, full run_pipeline_task() implementation
- [x] 02-04-PLAN.md — Nav bar + dashboard: persistent nav in base.html, recent runs, Submit CTA, /runs list
- [x] 02-05-PLAN.md — Run status page + SSE: stage checklist, real-time progress, three warning state UIs
- [x] 02-06-PLAN.md — In-app alerts: alert banners on dashboard, bell badge, HTMX dismiss, alert routes
- [x] 02-07-PLAN.md — Docker update: supervisord.conf, Dockerfile supervisor install, compose env vars
- [x] 02-08-PLAN.md — Phase 2 gate: implement test stubs, full suite green, Docker e2e human verification

### Phase 3: Review and Sign-Off
**Goal**: Engineers can review every characteristic with visual evidence, approve or override each classification with a documented reason, and produce an immutable signed audit packet when all items are resolved
**Depends on**: Phase 2
**Requirements**: REVIEW-01, REVIEW-02, REVIEW-03, REVIEW-04, REVIEW-05, REVIEW-06, REVIEW-07, SIGNOFF-01, SIGNOFF-02, SIGNOFF-03
**Success Criteria** (what must be TRUE):
  1. Engineer can open a completed run's review queue and see all characteristics listed with their pipeline classification, confidence score, and a live count of items by state (pending, approved, overridden)
  2. Each review item shows Rev A snippet (top) and Rev B snippet (bottom) stacked with clear labels on the left panel; characteristic number, system classification, confidence score, requirement text (Rev A then Rev B), Approve button, and Override dropdown with a required non-empty note field on the right panel
  3. Engineer can approve or override each item; every decision is saved immediately to the server; engineer can close the browser and resume the review from the same state on any device
  4. Sign-off is only available when every item has a terminal state (approved or overridden); the sign-off button is disabled and shows the remaining pending count while any item is pending
  5. Sign-off and audit packet generation are atomic: if PDF generation fails, the sign-off is rolled back and the run stays reviewable; a signed-but-no-packet state cannot exist; the signed packet is immutable after successful generation
**Plans**: 6 plans

Plans:
- [ ] 03-01-PLAN.md — Test stubs + ReviewItem DB model + pipeline patch for removed-item Rev B bbox
- [ ] 03-02-PLAN.md — Review queue GET route: open_review_queue service, queue.html, progress bar partial
- [ ] 03-03-PLAN.md — Per-item approve/override HTMX actions, item card template, snippet serving route
- [ ] 03-04-PLAN.md — Sign-off gate UI + confirmation modal + admin reviewer reassignment
- [ ] 03-05-PLAN.md — Two-phase sign-off atomicity + generating status page + SSE + immutability guard
- [ ] 03-06-PLAN.md — Phase 3 gate: all stubs to passing, full suite green, Docker e2e human verification

### Phase 4: Exports, History, and Amendments
**Goal**: Engineers can download formal deliverables, browse past runs, and correct finalized reviews without destroying the original signed record
**Depends on**: Phase 3
**Requirements**: PACKET-01, PACKET-02, PACKET-03, WORK-01, WORK-02, WORK-03, WORK-04, HISTORY-01, HISTORY-02, HISTORY-03, HISTORY-04, AMEND-01, AMEND-02, AMEND-03
**Success Criteria** (what must be TRUE):
  1. After sign-off, the audit packet is downloadable as PDF (formal cover page, characteristic summary table, per-item detail cards with inline Rev A and Rev B snippets) and as CSV (one row per characteristic with all review fields); the packet can be re-downloaded from run history at any time
  2. After finalization, engineer can generate a partial FAI work order on demand as PDF and CSV; the work order lists only changed and added characteristics, each with characteristic number, requirement text, drawing reference, and a priority flag distinguishing changed from added
  3. Engineer can view a list of all shop runs filterable by part number and date; can reopen any finalized run to view its review decisions in read-only mode; finalized runs are retained indefinitely
  4. Admin can configure an auto-cleanup period for unfinished and failed runs; admin can create, view, and deactivate engineer accounts and manage shop configuration
  5. Engineer can reopen a finalized run to create an amendment; the original signed packet is preserved unchanged; the amendment has its own sign-off producing a new versioned packet; both packets are accessible from the run record; input files cannot be changed in an amendment
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 8/8 | Complete   | 2026-03-03 |
| 2. Pipeline Bridge | 8/8 | Complete   | 2026-03-04 |
| 3. Review and Sign-Off | 1/6 | In Progress|  |
| 4. Exports, History, and Amendments | 0/TBD | Not started | - |
