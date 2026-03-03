# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-01)

**Core value:** Every characteristic classification is confirmed by a human engineer with image evidence before the change packet becomes an audit artifact — the system accelerates the review, never bypasses it.
**Current focus:** Phase 1 — Foundation

## Current Position

Phase: 1 of 4 (Foundation)
Plan: 2 of 8 in current phase
Status: In progress
Last activity: 2026-03-03 — Plan 01-02 complete: core shop/ package (DB models, auth service, app factory, dependencies, middleware)

Progress: [██░░░░░░░░] 6%

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 4.5 min
- Total execution time: 9 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation | 2 | 9 min | 4.5 min |

**Recent Trend:**
- Last 5 plans: 01-01 (2 min), 01-02 (7 min)
- Trend: -

*Updated after each plan completion*

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

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3]: WeasyPrint behavior with real 300 DPI engineering drawing PNG crops is unverified — run a proof-of-concept with actual pipeline output before finalizing the audit packet PDF template
- [Phase 4]: Partial FAI work order field format inferred from AS9102C section 4.6; validate with a real QC engineer during pilot before finalizing the layout

## Session Continuity

Last session: 2026-03-03
Stopped at: Plan 01-02 complete — core shop/ package implemented (DB models, auth service, app factory, dependencies, middleware)
Resume file: None
