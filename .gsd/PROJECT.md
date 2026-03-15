# Project

## What This Is

Delta Preservation is a FastAPI + Jinja2 + HTMX web application for aerospace quality teams performing AS9102 FAIR compliance work. It compares engineering drawing revisions (Rev A vs Rev B), identifies changed/unchanged/removed/added characteristics via an 8-stage pipeline, and produces a signed audit packet after human review.

## Core Value

A quality engineer can upload two PDFs and a Form 3 Excel, run the pipeline, review characteristic deltas with evidence snippets, and sign off a traceable audit packet — all in one tool.

## Current State

Two milestones complete:
- **M001:** Core pipeline and web app — login, setup wizard, run submission, SSE pipeline progress, review queue with approve/override, sign-off, audit packet PDF/CSV export, amendments
- **M002:** Setup simplification and per-run column mapping — column mapping moved to run submission form, setup wizard trimmed to 2 steps

Active redesign: M003 is a complete UI/UX redesign of all ~20 templates. S01, S02, and S03 complete — industrial dark theme compiles, sidebar + top-bar layout shell live, login and wizard redesigned (S01); dashboard, runs list, new run form, and HTMX partials (xlsx mapping, page selector, alert banner) all redesigned with bordered industrial sections (S02); run status page redesigned with bordered sections and border-l-4 status banners, SSE JS updated to match Jinja2 banners (S03). Backend and pipeline are untouched.

## Architecture / Key Patterns

- **Backend:** FastAPI, SQLAlchemy (SQLite), Huey task queue
- **Frontend:** Tailwind CSS v4 + DaisyUI v5, HTMX, server-sent events for pipeline progress
- **Templates:** Jinja2 in `shop/templates/` — base layout, per-section subdirs, HTMX partials
- **Static assets:** `static/src/input.css` → `static/dist/output.css` via `npm run build:css`
- **Pipeline:** 8-stage reconciliation in `delta_preservation/`, orchestrated by `shop/tasks.py`

## Capability Contract

See `.gsd/REQUIREMENTS.md` for the explicit capability contract, requirement status, and coverage mapping.

## Milestone Sequence

- [x] M001: Core Pipeline & Web App — end-to-end FAIR compliance tool
- [x] M002: Setup Simplification & Per-Run Column Mapping — streamlined onboarding
- [ ] M003: UI/UX Redesign — industrial dark-mode aesthetic, sidebar layout, full-screen review queue
