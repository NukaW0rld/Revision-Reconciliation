# M002: Setup Simplification & Per-Run Column Mapping

**Vision:** First-run setup completes in 2 steps. Column mapping confirmation happens at the point of use — when the engineer uploads an xlsx on the new run form — not during a disconnected setup wizard step.

## Success Criteria

- Fresh install: setup wizard has exactly 2 steps (shop name, admin password), then lands on `/login`
- `admin@shop.local` with the chosen password logs in immediately after setup
- `/setup/step3` and `/setup/step4` return redirect to `/login` (not 404, not a wizard page)
- New run form (`/runs/new`): selecting an xlsx file triggers inline column mapping confirmation panel via HTMX
- Column mapping panel shows auto-detected fields with amber indicators for undetected columns
- Engineer can correct undetected columns before submitting; pipeline runs to completion
- Admin settings page (`/admin/settings`) loads without any column mapping section
- All existing tests pass (87 baseline) plus new tests covering wizard reduction and per-run mapping

## Key Risks / Unknowns

- Per-run xlsx HTMX partial inside an existing outer form — nested `<form>` would break submission; must output bare fragment (selects only, no form tag)

## Proof Strategy

- Nested form risk → retire in S01 by building `runs/_xlsx_mapping.html` as a bare fragment and verifying the outer form submits correctly with mapping selects included

## Verification Classes

- Contract verification: `uv run pytest --tb=short` — 87+ existing tests pass, new tests for 2-step wizard, per-run mapping, removed routes
- Integration verification: none required (all changes are within the web layer; pipeline is untouched)
- Operational verification: Docker `docker compose up --build` — fresh setup completes in 2 steps, new run with xlsx shows mapping confirmation
- UAT / human verification: none — contract tests are sufficient for this scope

## Milestone Definition of Done

This milestone is complete only when all are true:

- S01 is complete with all tasks checked off
- `uv run pytest --tb=short` passes with ≥87 tests (existing baseline preserved, new tests added)
- Wizard step 3 and step 4 routes redirect gracefully (not 404)
- New run form shows inline column mapping confirmation after xlsx file selection
- Admin settings page renders without column mapping section
- Docker build succeeds and fresh container completes 2-step setup

## Requirement Coverage

- Covers: no formal requirement IDs (operating without REQUIREMENTS.md; requirements tracked in PROJECT.md Active section)
- Active tech items unaffected: WORK-03, PIPE-05, Run.signed_by ORM gap — all remain active, out of scope for M002
- New capability: per-run column mapping confirmation (not previously tracked as a requirement)
- Orphan risks: none

## Slices

- [ ] **S01: Two-step wizard & per-run column mapping** `risk:medium` `depends:[]`
  > After this: fresh install completes setup in 2 steps; new run form shows inline xlsx column mapping confirmation after file selection; admin settings page has no column mapping section; all tests pass

## Boundary Map

### S01 (only slice)

Produces:
- 2-step wizard flow: `/setup/step1` (shop name) → `/setup/step2` (admin password) → `/login`
- `GET /runs/validate-xlsx` — HTMX endpoint returning `runs/_xlsx_mapping.html` partial (bare select elements, no form wrapper)
- Graceful redirect for removed wizard routes (`/setup/step3`, `/setup/step4/*`)
- Clean admin settings page (no column mapping section)

Consumes:
- Existing `shop/services/form3.py:parse_excel_preview()` and `detect_column_mapping()` — unchanged
- Existing validate-pdf HTMX pattern in `shop/routers/runs.py` — followed as template
- Existing `shop/templates/setup/step4_error.html` — reused for xlsx validation errors
