# Delta Preservation

## What This Is

Delta Preservation is a web application for aerospace machine shop quality control teams that automates the most painful step in drawing revision management: determining exactly what changed between two engineering drawing revisions before the shop can ship again. Engineers upload two PDF drawings and an AS9102 Form 3 inspection spreadsheet, a computer vision pipeline detects and matches every numbered characteristic across both revisions, and a structured review queue lets the QC engineer confirm each classification with image evidence and sign off on an audit-ready change packet — without replacing the human judgment aerospace quality standards require.

## Core Value

Every characteristic classification is confirmed by a human engineer with image evidence before the change packet becomes an audit artifact — the system accelerates the review, never bypasses it.

## Requirements

### Validated

<!-- Existing pipeline capabilities shipped before v2.1 -->

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

<!-- v2.1 MVP — shipped 2026-03-08 -->

- ✓ File upload UI: two PDFs (Rev A, Rev B) + AS9102 Form 3 Excel with part metadata — v2.1
- ✓ Raster PDF detection at upload time with clear error before run submission — v2.1
- ✓ Multi-page PDF support: engineer selects which page when multiple pages detected — v2.1
- ✓ Revision labeling: tracks Rev A vs Rev B with revision letter metadata — v2.1
- ✓ Confidence-gated removal: ≥0.9 spatial confidence → auto-classified removed; <0.9 → unresolved in review queue — v2.1
- ✓ Balloon detection failure handling: Rev A hard-fail, Rev B partial warning — v2.1
- ✓ Duplicate balloon numbers ("4X Ø 0.5"): treated as one entry, count token tracked — v2.1
- ✓ GD&T feature control frames: opaque string matching (semantic parsing deferred) — v2.1
- ✓ Uniform low alignment confidence: warning state with engineer proceed/abort decision — v2.1
- ✓ Per-item review queue with Rev A/Rev B snippet cards, approve/override controls, required notes — v2.1
- ✓ Override always allowed; override note required — v2.1
- ✓ All items terminal state required before sign-off — v2.1
- ✓ Review session persistence: engineer can pause and resume — v2.1
- ✓ Run assignment: submitter defaults, admin can reassign — v2.1
- ✓ In-app failure alert to assigned reviewer — v2.1
- ✓ Amendment model: reopen finalized run, original packet preserved, own sign-off produces versioned packet — v2.1
- ✓ Amendment scope limited to review decisions; input files locked — v2.1
- ✓ Admin/engineer roles with RBAC enforcement — v2.1
- ✓ Setup wizard on first admin login: 2 steps (shop name, admin password → /login) — M002
- ✓ Per-run xlsx column mapping confirmation inline on new run form (HTMX; amber indicators for undetected columns) — M002
- ✓ Non-contiguous characteristic numbers accepted as-is — v2.1
- ✓ Run history list filterable by part number and date — v2.1
- ✓ Reopen any past completed review (read-only) — v2.1
- ✓ Audit packet PDF (cover page, summary table, per-item detail cards with inline snippets) + CSV — v2.1
- ✓ Sign-off atomic with packet generation: failed PDF generation rolls back sign-off — v2.1
- ✓ Signed packets immutable; amendments produce new versioned packet — v2.1
- ✓ Partial FAI work order on demand: PDF + CSV, changed/added only, priority flag — v2.1
- ✓ Docker deployment via `docker-compose.yml`; air-gapped network compatible — v2.1
- ✓ Admin-configurable auto-cleanup for unfinished/failed runs — v2.1
- ✓ Finalized runs retained indefinitely — v2.1

### Active

<!-- Carried forward from v2.1 tech debt and future features -->

- [ ] WORK-03 gap: `requirement_revB` is always blank in work order — Rev B requirement text not shown for changed items
- [ ] PIPE-05 dead code: `revB_balloon` warning_type path in status.html never emitted by tasks.py; only `low_confidence` is used
- [ ] `Run.signed_by` ORM relationship missing — FK column exists but no SQLAlchemy `relationship()` declared

### Out of Scope

- GD&T semantic parsing (feature control frame symbol/datum/modifier interpretation) — deferred to v3; string matching sufficient for pilot validation
- Concurrent multi-user review of the same run — one reviewer at a time; simpler state model
- Email notifications — in-app alerts only
- Auto-update mechanism — manual Docker pull; user experience concern, not a correctness concern
- Cross-revision timeline view (Rev A→B, B→C, C→D history) — run history covers individual comparisons; timeline is a future analytics feature
- QMS integration (ETQ, IQS, etc.) — CSV export is the integration surface; direct API connectors require per-customer work
- OAuth / SSO — simple email+password accounts appropriate for shop-scale deployment
- Bulk approve / select-all — destroys per-item accountability required by audit trail
- Editable requirement text in review UI — extraction errors require a new run
- Auto-sign-off when all items approved — sign-off is a formal attestation requiring deliberate intent

## Context

**Shipped v2.1** with ~11,000 LOC Python across pipeline and web app.

Tech stack: FastAPI + Jinja2 + HTMX + DaisyUI (frontend), SQLAlchemy + SQLite (storage), Huey SqliteHuey (task queue), WeasyPrint (PDF export), PyMuPDF + OpenCV + NumPy (pipeline), Docker + supervisord (deployment).

Pipeline is functionally correct for single-page vector-text drawings. Known limitation: raster/scanned PDFs are rejected at upload; multi-page support requires engineer page selection.

v2.1 closes the gap between the raw pipeline CLI and a production-ready quality workflow tool for aerospace machine shops. The full review → sign-off → audit packet → work order → amendment cycle is implemented and Docker-verified.

**Known tech debt from v2.1:**
- `requirement_revB` always blank in work order (WORK-03 partial gap)
- Dead `revB_balloon` code path in status.html
- `Run.signed_by` missing ORM relationship declaration
- Nyquist validation partial across all phases (human-approved for Docker-dependent behaviors)

## Constraints

- **Tech stack (pipeline)**: Python 3.10–3.13, uv, PyMuPDF, OpenCV, NumPy, Pydantic — existing pipeline is non-negotiable; web layer must integrate with it
- **Deployment target**: Docker on shop-hosted Linux server; no cloud dependency, no external services; air-gapped shop networks must be supported
- **Scale**: Single shop, O(10) engineers, O(100s) of runs per year — no distributed systems concerns
- **Audit integrity**: Signed packets must be immutable; amendment trail must be preserved; reviewer identity must be unambiguous in the record
- **AS9102 compliance**: Output format and terminology must be recognizable to a QC engineer familiar with AS9102 Form 3

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Web UI over CLI for review | QC engineers need a guided, visual review experience; raw JSON delta packets are not usable for sign-off | ✓ Good — review queue with image evidence is the core UX |
| Confidence threshold = 0.9 for removal auto-classification | Conservative default; errs toward human review over auto-classification; pilot will tune | ✓ Good — no false removals in testing |
| Override notes required | Every human correction to a system call must be documented; audit trail demands it | ✓ Good — clean audit record in practice |
| Atomic sign-off + packet generation | Prevents "signed but no packet" state that would create audit ambiguity | ✓ Good — two-phase write with rollback works correctly |
| Amendment creates new version, not overwrite | Original signed packet must be preserved for audit chain; amendments are additive | ✓ Good — `parent_run_id` + `packet_versions` list cleanly tracks lineage |
| GD&T as opaque string in v1 | Semantic GD&T parsing is a significant engineering investment; string matching catches tolerance value changes | ✓ Good — deferred correctly; pilot will validate scope |
| Docker deployment | Fits shop IT constraints (no cloud, no installer complexity, repeatable) | ✓ Good — single `docker compose up` verified end-to-end |
| Finalized runs retained indefinitely | Audit records must survive; storage management via admin config for unfinished runs | ✓ Good — retention cleanup task implemented |
| Work order generated on demand, not at sign-off | Engineer may need to coordinate timing of inspection handoff | ✓ Good — decoupled correctly |
| Amendment as cloned Run | Simplest model: clone ReviewItems, copy output_dir, set `parent_run_id`; no new table needed | ✓ Good — open_review_queue short-circuit handles existing items |
| `_effective_classification()` helper | Centralizes override vs pipeline classification resolution in exports | ✓ Good — used consistently across audit packet, work order |
| Phase gate as blocking milestone step | Full suite + Docker build + human E2E before closing each major phase | ✓ Good — caught several bugs before archiving |

## Milestone History

| Milestone | Description | Completed | Test Count |
|-----------|-------------|-----------|------------|
| M001: Migration | Full-stack web application wrapping pipeline CLI — upload → review → sign-off → audit packet → amendment | 2026-03-13 | 87 passed, 2 xfailed |
| M002: Setup Simplification & Per-Run Column Mapping | Reduce wizard to 2 steps (shop name + admin password); move Form 3 column mapping to per-run inline confirmation at upload time | 2026-03-13 | 93 passed, 2 xfailed |

---
*Last updated: 2026-03-13 after M002 S01 complete*
