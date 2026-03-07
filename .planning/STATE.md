---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
stopped_at: Completed 03-01-PLAN.md
last_updated: "2026-03-07T20:22:18.847Z"
last_activity: "2026-03-04 - Completed quick task 1: Review gitignore and README, push to GitHub, draft v0.2 release message"
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 22
  completed_plans: 17
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-01)

**Core value:** Every characteristic classification is confirmed by a human engineer with image evidence before the change packet becomes an audit artifact — the system accelerates the review, never bypasses it.
**Current focus:** Phase 3 — Review and Sign-Off

## Current Position

Phase: 2 of 4 (Pipeline Bridge) — COMPLETE
Plan: 8 of 8 in Phase 2 — all plans complete
Status: Phase 2 complete; ready for Phase 3
Last activity: 2026-03-04 - Completed quick task 1: Review gitignore and README, push to GitHub, draft v0.2 release message

Progress: [████████████████████░░░░░░░░░░░░░░░░░░░░] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Average duration: ~13 min (includes 01-08 human verification time)
- Total execution time: ~6h 45min (41 min automated + ~6h human Docker verification)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation | 8 | ~6h 45min | ~50 min (incl. human gate) |

**Recent Trend:**
- Last 8 plans: 01-01 (2 min), 01-02 (7 min), 01-03 (15 min), 01-04 (8 min), 01-05 (2 min), 01-06 (3 min), 01-07 (4 min), 01-08 (~6h human gate)
- Trend: Phase 1 complete

*Updated after each plan completion*
| Phase 02-pipeline-bridge P01 | 3 | 3 tasks | 5 files |
| Phase 02-pipeline-bridge P02 | 6min | 3 tasks | 7 files |
| Phase 02-pipeline-bridge P03 | 7 | 2 tasks | 4 files |
| Phase 02-pipeline-bridge P04 | 8min | 2 tasks | 8 files |
| Phase 02-pipeline-bridge P07 | 6min | 2 tasks | 3 files |
| Phase 02-pipeline-bridge P06 | 4min | 2 tasks | 4 files |
| Phase 02-pipeline-bridge P05 | 10min | 2 tasks | 4 files |
| Phase 02-pipeline-bridge P08 | human-verify | 3 tasks | 1 files |
| Phase 03-review-and-sign-off P01 | 7 | 3 tasks | 4 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Huey + SqliteStorage chosen as job queue (not FastAPI BackgroundTasks, not Redis-backed Dramatiq) — zero external service dependency for air-gapped deployment
- [Roadmap]: pwdlib 0.3.0 replaces passlib (unmaintained since 2020, breaks on Python 3.13)
- [Roadmap]: SQLAlchemy 2.0 + raw Pydantic schemas chosen over SQLModel (performance and coupling concerns)
- [Roadmap]: Sign-off atomicity — two-phase write pattern with rollback must be designed in Phase 3 schema, not added later
- [01-01]: xfail(strict=False) chosen over pytest.skip so test names appear in collection output and xfail count is visible
- [01-01]: conftest imports from shop/ intentionally scaffold-first — imports succeed only after Plan 02 creates the package
- [01-02]: secrets.token_urlsafe(32) chosen over itsdangerous for session tokens — DB is authority, signing overhead unnecessary for this threat model
- [01-02]: bcrypt pinned <5.0.0 to avoid HasherNotAvailable bug in pwdlib 0.3.0 when bcrypt 5.x removes __about__
- [01-02]: Routers NOT registered in app.py — Plans 03-05 register auth, setup, admin routers respectively
- [01-03]: Standard HTML POST (not HTMX POST) for login/logout — browser follows 302 redirect natively; HTMX intercepts redirects as partial DOM swaps causing broken navigation
- [01-03]: Router imports deferred inside create_app() body to break circular import (auth.py imports templates from shop.app; shop.app imports auth.router)
- [01-03]: HTMX 2.x and htmx-sse bundled locally via curl from unpkg — no CDN dependency at runtime (DEPLOY-03)
- [01-04]: TemplateResponse uses new Starlette 0.52 signature (request first) matching auth.py convention throughout codebase
- [01-04]: Deactivation is soft-delete only (is_active=False) — user record preserved for audit trail
- [01-04]: hx-confirm for browser-native deactivation confirm dialog — no custom JavaScript needed
- [01-05]: Wizard templates use standard HTML POST (not HTMX) — wizard is full-page flow; HTMX partial swaps conflict with browser back/forward and step validation redirects
- [01-05]: step3 GET redirects to wizard_step+1 (computed), not hardcoded step2, for correct mid-wizard resume from any incomplete state
- [Phase 01]: step4_mapping_partial.html reused for wizard and admin settings via form_action variable — eliminates template duplication
- [Phase 01]: Unmatched columns get select name=col_{idx} — save endpoint field-name loop naturally ignores them without JavaScript
- [01-07]: python:3.11-slim used (not 3.12/3.13) to respect pyproject.toml requires-python >=3.10,<3.13
- [01-07]: Node.js confined to tailwind-builder stage only — not present in runtime image (DEPLOY-03)
- [01-07]: DATABASE_URL env var in database.py enables Docker volume path override without code changes
- [01-07]: Admin seed is idempotent — checks for existing admin role before inserting
- [01-08]: Root redirect GET / -> /login added to auth router (co-located with auth entry-point logic)
- [01-08]: Admin email displayed on wizard step 2 from ShopConfig.admin_email to prevent first-run confusion
- [01-08]: Role-gated dashboard navigation via Jinja2 conditional on request.state.user.role — no JavaScript needed
- [01-08]: Human verification approved with note "revisions to the process will need to be made in the future" — UX iteration deferred to Phase 2
- [Phase 02-01]: HUEY_DB local fallback: tasks.py falls back to project-root huey.db when /app/data does not exist (dev/test environments outside Docker)
- [Phase 02-01]: Deferred shop.* imports in run_pipeline_task body to avoid circular imports between tasks.py and shop.database/models
- [Phase 02-01]: Run.reviewer_id nullable FK — submitter becomes reviewer in runs router (Plan 02); explicit reassignment deferred to Phase 3
- [Phase 02-03]: stage_callback called BEFORE each stage so UI shows current stage during execution
- [Phase 02-03]: Rev A balloon failure detected post-run via empty delta_packet.json items (pipeline never raises for empty balloons)
- [Phase 02-03]: run_pipeline_task.call_local() used in tests to execute synchronously without Huey queue; module-level SessionLocal/run_pipeline exports for test patching
- [Phase 02-02]: UPLOADS_DIR local fallback: falls back to project-root uploads/ when /app/data does not exist — same pattern as HUEY_DB from Plan 01
- [Phase 02-02]: save_upload() takes file_bytes+suffix+run_uuid (not UploadFile) — router reads bytes before calling services, keeping service functions pure and testable
- [Phase 02-02]: Raster detection: text extraction is primary signal; image area >= 95% coverage is secondary heuristic only evaluated when no text found
- [Phase 02-04]: redirect_slashes=False on APIRouter for /runs prefix — FastAPI default redirects /runs -> /runs/ (307), breaking nav links; solved by empty string route path
- [Phase 02-04]: Nav bar context helper _get_nav_context() in runs.py centralizes unread_alert_count + shop_name queries for all authenticated routes
- [Phase 02-pipeline-bridge]: [Phase 02-07]: supervisord manages uvicorn + huey_consumer in single container; pidfile=/tmp/supervisord.pid; uv binary copied to runtime for huey_consumer.py discovery
- [Phase 02-pipeline-bridge]: Dismiss route in auth.py (no prefix) keeps URL at /alerts/dismiss/{id} avoiding /runs prefix conflict
- [Phase 02-pipeline-bridge]: Dashboard queries RunAlert list (not count) — unread_alert_count = len(unread_alerts) saves a second DB query
- [Phase 02-pipeline-bridge]: acknowledge-warning is a Phase 3 placeholder redirect to /review/{run_id}; full review queue deferred
- [Phase 02-05]: FastAPI SSE route must be async generator with response_class=EventSourceResponse (not return EventSourceResponse(generator)) — routing layer detects is_gen_callable and encodes ServerSentEvent objects
- [Phase 02-05]: SSE test uses terminal-state runs to avoid TestClient hang — async generator loops with asyncio.sleep for non-terminal runs
- [Phase 02-pipeline-bridge]: Admin is created during setup wizard step 2 (not in seed_admin) because supervisord starts uvicorn directly without running run_web.py
- [Phase 02-pipeline-bridge]: Bell badge links to /dashboard instead of /alerts (no alerts route exists; dashboard shows unread banners)
- [Phase 02-pipeline-bridge]: SSE close handler injects banner immediately from event data then reloads after 500ms delay to avoid race condition
- [Phase 02-pipeline-bridge]: Lazy import fallback for run_pipeline: module-level symbol may be None; task body uses _get_run_pipeline() at execution time
- [Phase 02-pipeline-bridge]: SSE raw_data vs data: ServerSentEvent(data=str) double-encodes; use raw_data= for pre-serialised JSON payloads
- [Phase 02-pipeline-bridge]: SQLAlchemy SSE polling: db.expire(run)+db.refresh(run) required before each poll to bypass identity-map cache in async generators
- [Phase 02-pipeline-bridge]: validate-pdf single-page: return hidden input (not empty string) to keep swap target populated and page selector reliably dismissible
- [Phase 02-08]: Human verification approved after Docker e2e flow: upload form, stage progress SSE, and alert dismiss all confirmed working
- [Phase 03-review-and-sign-off]: foreign_keys=[reviewer_id] on Run.reviewer/User.runs to resolve AmbiguousForeignKeysError when signed_by_id added second FK to users table
- [Phase 03-review-and-sign-off]: Removed-item revB bbox uses apply_transform_bbox(revA_bbox_pdf, H) inside try/except in cli.py; falls back to None on failure (review card shows placeholder)

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3]: WeasyPrint behavior with real 300 DPI engineering drawing PNG crops is unverified — run a proof-of-concept with actual pipeline output before finalizing the audit packet PDF template
- [Phase 4]: Partial FAI work order field format inferred from AS9102C section 4.6; validate with a real QC engineer during pilot before finalizing the layout

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 1 | Review gitignore and README, push to GitHub, draft v0.2 release message | 2026-03-04 | 533f021 | [1-review-gitignore-and-readme-push-to-gith](./quick/1-review-gitignore-and-readme-push-to-gith/) |

## Session Continuity

Last session: 2026-03-07T20:22:18.845Z
Stopped at: Completed 03-01-PLAN.md
Resume file: None
