# Delta Preservation

## What This Is

Delta Preservation is a web application for aerospace machine shop quality control teams that automates the most painful step in drawing revision management: determining exactly what changed between two engineering drawing revisions before the shop can ship again. Engineers upload two PDF drawings and an AS9102 Form 3 inspection spreadsheet, a computer vision pipeline detects and matches every numbered characteristic across both revisions, and a structured review queue lets the QC engineer confirm each classification with image evidence and sign off on an audit-ready change packet — without replacing the human judgment aerospace quality standards require.

## Core Value

Every characteristic classification is confirmed by a human engineer with image evidence before the change packet becomes an audit artifact — the system accelerates the review, never bypasses it.

## Requirements

### Validated

<!-- Existing pipeline capabilities shipped and working in CLI form -->

- ✓ 8-stage pipeline orchestrated in `delta_preservation/cli.py:run_pipeline()` — existing
- ✓ AS9102 Form 3 Excel parsing with auto-detected column headers (`io/xlsx.py`) — existing
- ✓ Hybrid balloon detection: PDF text extraction + OpenCV HoughCircles fallback (`vision/balloons.py`) — existing
- ✓ Rev A text extraction via PyMuPDF with spatial bounding boxes (`io/pdf.py`) — existing
- ✓ Anchor building linking Form 3 requirements to Rev A balloon locations via token/distance scoring (`reconcile/anchors.py`) — existing
- ✓ Cross-revision alignment via ORB feature detection + RANSAC homography (`vision/alignment.py`) — existing
- ✓ Candidate generation and matching: location (35%) + text/numeric overlap (50%) + context Jaccard (15%) scoring (`reconcile/match.py`) — existing
- ✓ Delta classification: unchanged / changed / removed / uncertain / added (`reconcile/classify.py`) — existing
- ✓ Confidence scoring and structured reasons per characteristic — existing
- ✓ Evidence snippet generation (PNG crops) per characteristic (`vision/snippets.py`) — existing
- ✓ `DeltaPacket` JSON output with run_id, input metadata, per-item evidence (`types.py`) — existing
- ✓ Text normalization and `MatchFingerprint` extraction (`reconcile/normalize.py`) — existing
- ✓ Tolerance parsing: inline ±, signed pairs, stacked bilateral/unilateral (`reconcile/tolerance_pdf.py`) — existing

### Active

**Input handling:**
- [ ] File upload UI: two PDFs (Rev A, Rev B) + AS9102 Form 3 Excel with part metadata (part number, revision levels, customer, job number)
- [ ] Raster PDF detection at upload time with clear error before run submission
- [ ] Multi-page PDF support: engineer selects which page to analyze when multiple pages detected
- [ ] Revision labeling: system tracks which PDF is Rev A vs Rev B with revision letter metadata

**Pipeline behavior (production decisions):**
- [ ] Confidence-gated removal: characteristic not found in Rev B at spatial confidence ≥ 0.9 → auto-classified removed; below 0.9 → unresolved in review queue
- [ ] Text extraction failure: if spatial confidence ≥ 0.9 → low-confidence unchanged; if spatial confidence also < 0.9 → unresolved
- [ ] Balloon detection failure: Rev A failure → hard fail with clear error; Rev B failure → partial result with warning, engineer decides whether usable
- [ ] Duplicate balloon numbers (e.g., "4X Ø 0.5"): treated as one entry, count token tracked; count change flagged as classification signal
- [ ] GD&T feature control frames: opaque string matching in v1 (semantic parsing deferred to v2)
- [ ] Uniform low alignment confidence: run enters warning state, engineer sees confidence distribution and decides to proceed or abort before reviewing items
- [ ] Uniformly low confidence threshold: 0.9 (configurable by admin for pilot tuning)

**Review queue:**
- [ ] Per-item card layout: Rev A snippet (top) and Rev B snippet (bottom) stacked on left panel with clear labels; char number, system classification, confidence score, requirement text (A → B diff), Approve / Override controls, required note field on right panel
- [ ] Override always allowed; override note is required (no silent overrides — every human correction must have a written reason)
- [ ] All items must reach terminal state before sign-off is available (no soft gate)
- [ ] Review session persistence: engineer can pause and resume; partial review state saved
- [ ] Run assignment: submitter defaults as reviewer; can explicitly assign to any other engineer in the shop
- [ ] Run failure: in-app alert to assigned reviewer when logged in

**Amendment model:**
- [ ] Finalized run can be reopened as an amendment: original signed packet preserved, amendment tracked separately with its own sign-off
- [ ] Amendment scope: review decisions only — input files are locked at first sign-off; wrong input files require a new run

**User accounts and roles:**
- [ ] Admin role: manages shop config, creates/manages engineer accounts, configures Excel column mapping, sets retention policy
- [ ] Engineer role: submits runs, reviews queue items, signs off; cannot change shop settings
- [ ] Reviewer name attached to approvals and overrides in audit record

**Shop onboarding:**
- [ ] Setup wizard on first admin login: name the shop, set admin password, create first engineer account, upload and map a sample Form 3; cannot be dismissed until complete
- [ ] Excel column mapping: one-time setup per shop; fatal errors (unreadable file, empty sheet) caught at upload; minor issues (wrong column guesses) surface during mapping UI
- [ ] Non-contiguous or non-standard characteristic numbers: accepted as-is, no normalization

**Run management:**
- [ ] Run history list: browse all past runs filterable by part number and date
- [ ] Reopen any past completed review to view it
- [ ] Download/re-export signed audit packet and work order from any finalized run

**Audit packet:**
- [ ] Format: PDF (formal cover page + summary table of all characteristics + detail section with per-item card including inline snippets) and CSV
- [ ] Cover page: part number, revision levels, customer, run ID, reviewer name, sign-off timestamp
- [ ] Sign-off is atomic with packet generation: if PDF generation fails, sign-off is rolled back; run stays reviewable until packet generation succeeds
- [ ] Immutable once signed; amendments produce a new versioned packet

**Partial FAI work order:**
- [ ] Generated on demand after finalization (separate action, not at sign-off)
- [ ] Format: PDF + CSV
- [ ] Contents: characteristic number, requirement text (original tolerance/dimension), drawing reference (balloon number, page, zone), priority flag (changed vs added)

**Deployment:**
- [ ] Docker deployment via `docker-compose.yml`; shop pulls image and runs `docker compose up`
- [ ] Updates via `docker compose pull` (manual for pilot; formalized update path in future milestone)

**Storage and retention:**
- [ ] Finalized runs (signed packets): retained indefinitely
- [ ] Unfinished and failed runs: admin-configurable auto-cleanup period

### Out of Scope

- GD&T semantic parsing (feature control frame symbol/datum/modifier interpretation) — deferred to v2; string matching sufficient for pilot validation
- Concurrent multi-user review of the same run — one reviewer at a time; simpler state model
- Email notifications — in-app alerts only for v1
- Auto-update mechanism — manual Docker pull for pilot; user experience concern, not a correctness concern
- Cross-revision timeline view (Rev A→B, B→C, C→D history) — run history covers individual comparisons; timeline is a v2 analytics feature
- QMS integration (ETQ, IQS, etc.) — CSV export is the integration surface for v1; direct API connectors require per-customer work
- OAuth / SSO — simple email+password accounts appropriate for shop-scale deployment

## Context

The existing CLI pipeline (8 stages, Python, uv) is the computational core. It produces `delta_packet.json` with evidence snippets and is functionally correct for single-page vector-text drawings. The gap being closed in this project is the production shell around it: the web interface, user model, review workflow, and audit artifact generation that let a real QC team use it in their actual workflow.

The target user is a QC engineer or QC manager at an aerospace machine shop — technically literate but not a software developer. The interface must make the right action obvious and the wrong action (approving without reading, skipping items) require deliberate effort.

AS9102 Rev C governs the characteristic accountability requirements this tool supports. Partial FAI re-accomplishment (re-inspect only changed characteristics) is the primary cost-saving mechanism the tool enables. See `.claude/AS9102C.md` for full standard reference.

## Constraints

- **Tech stack (pipeline)**: Python 3.10–3.13, uv, PyMuPDF, OpenCV, NumPy, Pydantic — existing pipeline is non-negotiable; web layer must integrate with it
- **Deployment target**: Docker on shop-hosted Linux server; no cloud dependency, no external services; air-gapped shop networks must be supported
- **Scale**: Single shop, O(10) engineers, O(100s) of runs per year — no distributed systems concerns for v1
- **Audit integrity**: Signed packets must be immutable; amendment trail must be preserved; reviewer identity must be unambiguous in the record
- **AS9102 compliance**: Output format and terminology must be recognizable to a QC engineer familiar with AS9102 Form 3

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Web UI over CLI for review | QC engineers need a guided, visual review experience; raw JSON delta packets are not usable for sign-off | — Pending |
| Confidence threshold = 0.9 for removal auto-classification | Conservative default; errs toward human review over auto-classification; pilot will tune | — Pending |
| Override notes required | Every human correction to a system call must be documented; audit trail demands it | — Pending |
| Atomic sign-off + packet generation | Prevents "signed but no packet" state that would create audit ambiguity | — Pending |
| Amendment creates new version, not overwrite | Original signed packet must be preserved for audit chain; amendments are additive | — Pending |
| GD&T as opaque string in v1 | Semantic GD&T parsing is a significant engineering investment; string matching catches tolerance value changes which are the most common and consequential changes | — Pending |
| Docker deployment | Fits shop IT constraints (no cloud, no installer complexity, repeatable); update path is simple | — Pending |
| Finalized runs retained indefinitely | Audit records must survive; storage management is shop responsibility via admin config for unfinished runs | — Pending |
| Work order generated on demand, not at sign-off | Engineer may need to coordinate timing of inspection handoff; decoupling generation from sign-off gives that flexibility | — Pending |

---
*Last updated: 2026-03-01 after initialization*
