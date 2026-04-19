# Phase 11: web-run-to-review-e2e-automation - Context

**Gathered:** 2026-04-18 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Prove the live maintainer web workflow from `/runs/new` through background processing, persisted `delta_packet.json`, review/debug surfacing, and sign-off/export enforcement with automated coverage. This phase closes the Phase 7 algorithm-only verification gap by adding a live web integration proof. It does not change classifier behavior, ground-truth logic, export semantics, or product scope.

</domain>

<decisions>
## Implementation Decisions

### Test Harness
- **D-01:** Phase 11 uses the existing `pytest` + FastAPI `TestClient` harness with `huey_immediate` so `POST /runs/new` executes the pipeline task inline during the test. Do not add browser automation or a real Huey worker as the default proof path.

### Live Fixture Source
- **D-02:** The core live-flow proof submits real corpus files from `assets/partN/` through `/runs/new`, not a hand-built mocked packet path.
- **D-03:** The test isolates side effects by patching `shop.services.runs.UPLOADS_DIR` and `shop.tasks.OUT_DIR` to per-test temporary directories, then explicitly asserts that the run writes a real `delta_packet.json` that the review/debug layer reloads.

### Coverage Shape
- **D-04:** Phase 11 should add a dedicated web-flow integration test module that stitches submission, task execution, status/review entry, and sign-off/export assertions together. Do not treat scattered extensions to `tests/test_runs.py`, `tests/test_review.py`, and `tests/test_exports.py` as sufficient milestone proof on their own.
- **D-05:** The live integration proof must verify the actual Phase 10 service contract from the web surface: unresolved debug exceptions block sign-off/export, and cleared state allows the signed/export path to proceed.

### Advisory Coverage
- **D-06:** Advisory-flag coverage is required, but it does not need to be forced into the main live corpus run if the chosen corpus part does not reliably emit non-empty `confidence_flags`.
- **D-07:** Planning may satisfy advisory coverage in one of two approved ways: use a representative corpus run only after verifying it emits stable non-empty `confidence_flags`, or pair the live `/runs/new` proof with a narrow seeded packet/snapshot assertion that exercises the existing advisory UI/export contract.

### the agent's Discretion
- Choose the representative part fixture after confirming it gives stable live-web assertions for submission, review/debug state, and sign-off/export behavior.
- Choose the exact patching technique for temp upload/output isolation (`monkeypatch`, direct module patching, or equivalent) as long as import-time directory globals are controlled deterministically.
- Decide whether the live integration module owns one broad end-to-end test or a small cluster of tightly-related integration tests, as long as the flow remains traceable as a single Phase 11 proof artifact.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope and gap closure
- `.planning/ROADMAP.md` — Phase 11 goal, success criteria, and explicit `/runs/new` through review/debug/export boundary.
- `.planning/v1.1-MILESTONE-AUDIT.md` — `GAP-03` definition and the exact missing live-web proof this phase must close.
- `.planning/REQUIREMENTS.md` — milestone boundary showing Phase 11 as audit flow closure, not new algorithm work.
- `.planning/PROJECT.md` — maintainer-only workflow scope, no-product-expansion constraint, and immutable-ground-truth rule.

### Prior phase decisions
- `.planning/phases/07-regression-tests-and-verification/07-CONTEXT.md` — Phase 7 locked algorithm-only baseline/verification, which Phase 11 must complement rather than repeat.
- `.planning/phases/10-debug-exception-gating-and-advisory-surfacing/10-CONTEXT.md` — authoritative sign-off/export gating and `confidence_flags` surfacing contract that Phase 11 must prove through the live web path.
- `.planning/debug/resolved/debug-report-download-fails-part8-9.md` — prior resolved live-surface mismatch showing why hidden debug blockers must be exercised from the web layer, not assumed from lower-level tests.

### Existing web/test entrypoints
- `tests/conftest.py` — shared FastAPI `TestClient`, in-memory DB wiring, and `huey_immediate` fixture.
- `tests/test_runs.py` — current `/runs/new`, upload validation, synchronous enqueue, and run status coverage.
- `tests/test_pipeline_task.py` — task lifecycle, `delta_packet.json`, failure, and warning-path assertions.
- `tests/test_review.py` — review queue, sign-off, and Phase 10 debug gate coverage using seeded packet data.
- `tests/test_exports.py` — signed snapshot and export contract coverage, including advisory/export assertions.
- `tests/test_focused_debug_queue.py` — focused admin debug queue coverage and unresolved-exception visibility.

### Core implementation paths
- `shop/routers/runs.py` — `/runs/new`, status page, and run SSE entrypoints.
- `shop/services/runs.py` — upload persistence and run creation helpers.
- `shop/tasks.py` — Huey pipeline execution and `delta_packet.json` output contract.
- `shop/routers/review.py` — review queue, debug queue, and sign-off confirmation route.
- `shop/services/review.py` — debug queue assembly, advisory extraction, debug summary, and sign-off gate logic.
- `shop/routers/exports.py` — signed export routes that must remain consistent with the review/sign-off contract.

### Fixture provenance
- `assets/part1/` through `assets/part9/` — real corpus inputs available to drive the live `/runs/new` submission path.
- `tests/fixtures/phase7_algorithm_only/README.md` — explicit distinction between algorithm-only baseline fixtures and the missing live web proof that Phase 11 owns.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/conftest.py` already provides the in-memory DB, authenticated `TestClient`, and `huey_immediate` fixture needed to execute `/runs/new` inline without extra infrastructure.
- `tests/test_runs.py` already has upload helpers for valid vector PDFs and Excel files, plus a supported pattern for patching `shop.tasks.run_pipeline` and `SessionLocal`.
- `tests/test_pipeline_task.py` already proves task-side lifecycle behavior once `run_pipeline_task` is invoked, so Phase 11 can focus on the web submission-to-review wiring.
- `tests/test_review.py`, `tests/test_focused_debug_queue.py`, and `tests/test_exports.py` already encode the sign-off/export/debug contracts that the new live-flow test should reuse rather than duplicate from scratch.

### Established Patterns
- Route tests use FastAPI `TestClient` against the real app and an in-memory SQLite database.
- Long-running Huey work is tested in immediate mode rather than through a separate worker process.
- Focused tests seed packet and snapshot data directly when validating one layer in isolation; this is acceptable existing coverage, but not sufficient for Phase 11's cross-surface proof.
- Export and sign-off contracts are service-first: routes and templates consume `build_signoff_gate_state`, `build_run_debug_summary`, and signed snapshot loaders instead of re-deriving state in the UI.

### Integration Points
- `POST /runs/new` in `shop/routers/runs.py` is the live submission boundary and calls `run_pipeline_task(...)` after persisting uploads and creating the `Run`.
- `shop/tasks.py` writes `delta_packet.json`, which `shop/services/review.py` later reloads to build review/debug state.
- `GET /review/{run_id}` and `GET /review/{run_id}?debug=1` are the maintainer surfaces where Phase 10 gating and advisory rules become visible.
- `POST /review/{run_id}/sign-off/confirm` and `shop/routers/exports.py` are the terminal contract points that must behave consistently with the unresolved debug state produced from the live run.

</code_context>

<specifics>
## Specific Ideas

- The new Phase 11 proof should add the missing cross-surface integration artifact, not replace the existing slice tests that already validate upload, task, review, and export logic independently.
- The representative fixture-backed run should exercise real file upload and packet persistence in isolated temp directories so the test proves live behavior without contaminating checked-in `uploads/` or `out/`.
- Advisory coverage should stay explicit in planning: if the chosen real corpus part does not emit a stable non-empty `confidence_flags`, keep that assertion in a small companion seeded case rather than pretending the live run covered it.

</specifics>

<deferred>
## Deferred Ideas

None — analysis stayed within the Phase 11 boundary.

</deferred>

---

*Phase: 11-web-run-to-review-e2e-automation*
*Context gathered: 2026-04-18 (assumptions mode)*
