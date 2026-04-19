# Phase 11: Web Run-to-Review E2E Automation - Research

**Researched:** 2026-04-18
**Domain:** Live `/runs/new` web submission, immediate Huey execution in tests, packet persistence, review/debug gating, signed export reachability, and advisory coverage strategy for the existing maintainer workflow
**Confidence:** HIGH (grounded in Phase 11 context, the v1.1 milestone audit, current review/export code paths, existing slice tests, and current fixture/debug-report evidence)

<user_constraints>
## User Constraints

Phase 11 already has locked context in `11-CONTEXT.md`; this research translates
that context into planning-ready implementation guidance.

### Locked Decisions

- **D-01:** Use the existing `pytest` + FastAPI `TestClient` harness with
  `huey_immediate` so `/runs/new` executes inline in-process. Do not introduce
  browser automation or a real Huey worker as the default proof path.
- **D-02 / D-03:** The core proof must submit real corpus files from
  `assets/partN/` through `/runs/new`, and it must isolate side effects by
  patching `shop.services.runs.UPLOADS_DIR` plus `shop.tasks.OUT_DIR` to
  per-test temporary directories. The test must assert a real
  `delta_packet.json` exists under the isolated output tree.
- **D-04 / D-05:** Phase 11 needs a dedicated integration module that stitches
  submission, task execution, packet persistence, review/debug entry, and the
  Phase 10 sign-off/export gate together. Scattered extensions to existing
  slice-test modules are not sufficient milestone proof.
- **D-06 / D-07:** Advisory coverage is required, but it does not need to come
  from the live corpus run if no stable live fixture emits non-empty
  `confidence_flags`. A narrow seeded packet/snapshot companion assertion is an
  approved fallback.
- Preserve the maintainer-only scope. Phase 11 proves the existing workflow; it
  does not change classifier behavior, ground truth, export semantics, or the
  product surface.

### Recommended Decisions For Planning

- **Use `assets/part6/` as the primary live fixture.** Current
  `tests/fixtures/phase7_algorithm_only/part6-debug-report.json` shows zero
  `missing_added_truth_indexes`, only one `review_needed` row, and the paired
  PDFs are relatively small (`revA.pdf` ~35 KB, `revB.pdf` ~30 KB). That gives
  the smallest currently-known unresolved-debug surface among the live corpus
  candidates while still exercising a real blocker path.
- **Keep all Phase 11 proof cases in one dedicated module** named
  `tests/test_web_run_review_e2e.py`, even when one companion test uses seeded
  snapshot data. This preserves the roadmap requirement that Phase 11 own a
  traceable cross-surface artifact.
- **Use route-first assertions, then service- or DB-assisted clearance only
  after the route boundary is proven.** The test should prove the real
  submission/review/debug/sign-off/export seams; it does not need to spend most
  of its runtime clicking dozens of per-row approval forms that are already
  covered elsewhere.

### Out of Scope

- New classifier heuristics, ground-truth evaluation changes, or export-format
  redesign
- Replacing the existing slice tests in `tests/test_runs.py`,
  `tests/test_review.py`, or `tests/test_exports.py`
- Browser-driven E2E automation
- Forcing advisory coverage into the live corpus path when the current live
  fixtures do not reliably emit `confidence_flags`

</user_constraints>

<phase_requirements>
## Phase Requirements

| Source | Requirement / Gap | Research Support |
|--------|--------------------|------------------|
| GAP-03 | There must be automated proof of `/runs/new -> background task -> delta_packet.json -> review/debug/export surfaces`. | The repo currently has slice tests for upload, task lifecycle, review gating, and export contracts, but no single artifact crosses the live web boundary end to end. |
| TST-02 | Phase 7's benchmark proof must be complemented by a live-web integration proof. | Current algorithm-only fixtures prove scoring parity, not the actual maintainer submission/review/export loop. |
| VER-01 | Milestone verification must include the live web path, not only standalone artifacts. | `shop/routers/runs.py`, `shop/tasks.py`, `shop/routers/review.py`, and `shop/routers/exports.py` already form the runtime chain; Phase 11 only needs to exercise it coherently. |
| Phase 10 downstream contract | Sign-off/export blockers and advisory surfacing must be proven through the live path. | The Phase 10 routes and services are already covered in isolation. Phase 11 needs to prove a real run reaches those existing gates and export paths. |

</phase_requirements>

<architectural_responsibility_map>
## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Why it belongs there |
|------------|-------------|----------------|----------------------|
| Live submission proof | `tests/test_web_run_review_e2e.py` | `tests/conftest.py` | The missing artifact is a dedicated integration module, not a production-code feature. |
| Test DB/session wiring for inline Huey execution | `tests/conftest.py` | local `sessionmaker(bind=db_engine)` patching | `shop.tasks.SessionLocal` must point at the test engine during `/runs/new` or the live task writes to the wrong DB. |
| Upload/output isolation | `shop.services.runs.UPLOADS_DIR`, `shop.tasks.OUT_DIR` | temp dirs from pytest | The test must prove real file persistence without polluting checked-in `uploads/` or `out/`. |
| Review/debug gate proof | `shop/routers/review.py` + `shop/services/review.py` | dedicated integration assertions | Phase 11 should consume the existing sign-off gate instead of rebuilding or reinterpreting it in tests. |
| Signed export reachability | `shop/routers/exports.py` + `shop/services/exports.py` | dedicated integration assertions | The export behavior already exists; Phase 11 only needs to prove a live run can reach it after gate clearance. |
| Advisory fallback proof | dedicated seeded helper inside `tests/test_web_run_review_e2e.py` | existing Phase 10 export/review helpers as reference only | Current live corpus fixtures do not emit stable non-empty `confidence_flags`, so the advisory clause needs a controlled companion case. |

</architectural_responsibility_map>

<research_summary>
## Summary

Phase 11 is a proof-integration phase, not a new product phase. The main
runtime pieces already exist:

1. **Live web submission already works in tests.**
   `tests/conftest.py` provides the in-memory DB and `huey_immediate`, while
   `shop/routers/runs.py` calls `run_pipeline_task(...)` directly after saving
   uploads. The missing piece is a dedicated artifact that keeps the route,
   task, persistence, and review/export surfaces in one test narrative.
2. **The main hidden trap is `shop.tasks.SessionLocal`.**
   The app's dependency override points HTTP routes at the test DB, but the
   Huey task still resolves its own module-level `SessionLocal`. Any real
   `/runs/new` proof must patch that symbol to the same test session factory or
   the task will write progress into a different database.
3. **Phase 10 already closed the blocker/export logic in slices.**
   `tests/test_review.py`, `tests/test_debug_verdicts.py`, and
   `tests/test_exports.py` already prove unresolved-debug gating, signed
   snapshot persistence, and advisory/export contracts. Phase 11 should reuse
   those service boundaries, not copy their logic into a second stack.
4. **No currently checked-in corpus debug report gives stable advisory coverage.**
   Both `assets/debug_report_part*.json` and
   `tests/fixtures/phase7_algorithm_only/part*-debug-report.json` show zero
   non-empty `confidence_flags` rows. The Phase 11 advisory clause therefore
   needs the approved companion seeded packet/snapshot case.

The safest planning split is:

- **Plan 01:** create the dedicated live-flow integration harness and prove
  `/runs/new` submission, isolated `delta_packet.json` persistence, status page
  visibility, review/debug entry, and blocked sign-off on the real Phase 10
  gate using the recommended `part6` fixture.
- **Plan 02:** extend the same module to clear the live run into a signed state,
  assert signed export reachability, and add one seeded advisory companion case
  inside that same module so the milestone artifact still owns advisory
  coverage.

**Primary recommendation:** keep the Phase 11 artifact concentrated in
`tests/test_web_run_review_e2e.py`, use `assets/part6/` for the real live run,
and let the advisory clause stay seeded and companion-style because the live
corpus currently cannot prove it deterministically.

</research_summary>

<standard_stack>
## Standard Stack

### Core

| Library / Tool | Purpose | Why standard here |
|----------------|---------|-------------------|
| `pytest` via `uv run pytest` | Integration and regression execution | The repo already uses pytest for route, task, review, and export coverage. |
| FastAPI `TestClient` | Live route submission and HTML/redirect assertions | Phase 11 needs route-level proof, not service-only proof. |
| `huey_immediate` | Inline background task execution | This preserves the real `/runs/new` code path while keeping the test deterministic. |
| SQLAlchemy `sessionmaker(bind=db_engine)` | Patch target for `shop.tasks.SessionLocal` | Required so the live task and the HTTP route share the same in-memory DB. |
| `tmp_path` + `monkeypatch` | Isolated upload/output roots | Required by the context's no-side-effects constraint. |

### Supporting

| Artifact | Purpose | When to use |
|----------|---------|-------------|
| `tests/conftest.py` | Existing auth/app/db fixtures | Extend only if a shared Phase 11 isolation helper materially reduces duplication. |
| `tests/test_runs.py` | Existing multipart upload and status-page patterns | Use as the route/request baseline for the live `/runs/new` test. |
| `tests/test_pipeline_task.py` | Inline Huey task patching pattern | Use as the reference for patching `shop.tasks.SessionLocal` and `OUT_DIR`. |
| `tests/test_review.py` | Existing sign-off and review-gate assertions | Reuse the same gate expectations in the dedicated live-flow module. |
| `tests/test_exports.py` | Existing signed export and advisory expectations | Use as the reference contract for the companion seeded advisory case. |

### Alternatives Considered

| Instead of | Could use | Tradeoff |
|------------|-----------|----------|
| Dedicated `tests/test_web_run_review_e2e.py` module | Add a few assertions to `tests/test_runs.py`, `tests/test_review.py`, and `tests/test_exports.py` | Easier locally, but it fails the phase requirement for one traceable live-flow artifact. |
| Real corpus upload via `/runs/new` | Patch `run_pipeline_task` or seed `delta_packet.json` directly | Faster, but it misses the exact gap Phase 11 is supposed to close. |
| Advisory proof from live corpus fixture | Companion seeded packet/snapshot case | The live corpus currently provides no stable non-empty `confidence_flags`, so forcing it would make the test brittle or dishonest. |
| Full route-only row clearance | Directly submitting dozens of normal approval forms | Realistic, but not a good trade here because those row-level interactions already have slice coverage and would dominate runtime. |

</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Real Submission, Controlled Environment

**What:** submit real `assets/part6/*` bytes through `/runs/new`, but patch all
filesystem and DB globals into temp/test-owned resources first.

**Why it fits Phase 11:** the milestone gap is about the live route/task/persist
chain, not about unisolated side effects or production database wiring.

### Pattern 2: Dedicated Integration Artifact, Shared Slice References

**What:** keep the milestone proof in one dedicated module while reusing the
existing slice-test patterns and contracts as reference.

**Why it fits Phase 11:** the roadmap wants one traceable artifact, but the repo
already has good lower-level coverage that should not be duplicated.

### Pattern 3: Route-First, Then Assisted Clearance

**What:** prove the real route/task/review/debug boundary first, then use a
small helper or DB update to mass-clear already-covered normal-review rows so
the test can focus on the unresolved debug gate and signed export reachability.

**Why it fits Phase 11:** the phase is about cross-surface wiring, not about
retesting every approval form path that Phase 3/10 already covers.

### Pattern 4: Companion Seeded Advisory Clause

**What:** keep one narrow seeded packet/snapshot test in the same module to
exercise advisory surfacing when the live run cannot prove it.

**Why it fits Phase 11:** it satisfies the context's approved fallback without
lying about current live fixture behavior.

### Anti-Patterns to Avoid

- **Scattered proof:** touching three existing test modules and calling that the
  Phase 11 artifact
- **Unpatched task DB/session globals:** letting `shop.tasks.SessionLocal` hit a
  different DB than the HTTP route
- **Shared-dir pollution:** writing the live run into checked-in `uploads/` or
  `out/`
- **Advisory wishful thinking:** asserting live `confidence_flags` that the
  current corpus does not produce

</architecture_patterns>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Patching `UPLOADS_DIR` and `OUT_DIR` but not `SessionLocal`

**What goes wrong:** the files land in temp dirs, but the task still updates a
different DB, so `/runs/{id}` and `/review/{id}` never see the live task state.
**How to avoid:** patch `shop.tasks.SessionLocal` to the same
`sessionmaker(bind=db_engine)` used by the app fixture before calling
`/runs/new`.

### Pitfall 2: Reusing historical `assets/debug_report_part*.json` as live proof

**What goes wrong:** the test appears to prove the live path, but it never
executes `/runs/new` or writes a fresh `delta_packet.json`.
**How to avoid:** always assert the live temp `output_dir` actually contains a
new `delta_packet.json` created by the inline task.

### Pitfall 3: Letting the live run own the advisory clause

**What goes wrong:** the test becomes flaky or fails because the chosen corpus
fixture does not emit `confidence_flags`.
**How to avoid:** keep the advisory clause in a companion seeded packet/snapshot
case inside the same module.

### Pitfall 4: Re-testing every approval UI path inside the E2E artifact

**What goes wrong:** runtime and brittleness balloon, while the test adds little
new confidence beyond existing slice coverage.
**How to avoid:** use the live route for submission/review/debug/sign-off/export
seams, then assist normal-review clearance once the route boundary is proven.

</common_pitfalls>

<code_examples>
## Code Examples

### Existing live submission boundary

```bash
rg -n "POST /runs/new|run_pipeline_task\\(|save_upload\\(|create_run\\(" \
  shop/routers/runs.py shop/services/runs.py shop/tasks.py tests/test_runs.py
```

Current finding: `/runs/new` already saves uploads, creates the `Run`, and calls
`run_pipeline_task(...)` directly; no new production seam is needed for Phase 11.

### Existing sign-off/export contract

```bash
rg -n "build_signoff_gate_state|attempt_sign_off|debug_snapshot_path|_get_signed_run" \
  shop/services/review.py shop/routers/review.py shop/routers/exports.py tests/test_review.py tests/test_exports.py
```

Current finding: Phase 10 already enforced the gate and signed snapshot
contract. Phase 11 needs to reach those same seams from a real live run.

### Current advisory evidence reality check

```bash
uv run python - <<'PY'
import json
from pathlib import Path
for path in sorted(Path("tests/fixtures/phase7_algorithm_only").glob("part*-debug-report.json")):
    payload = json.loads(path.read_text())
    flagged = sum(1 for item in payload.get("items", []) if item.get("confidence_flags"))
    print(path.name, flagged)
PY
```

Current finding: all checked-in algorithm-only fixtures report `0` flagged rows,
so a seeded companion case is currently required for advisory coverage.

</code_examples>

<validation_architecture>
## Validation Architecture

### Test Framework

- Framework: `pytest`
- Quick command:
  `uv run pytest -q tests/test_web_run_review_e2e.py -x`
- Full phase command:
  `uv run pytest -q tests/test_web_run_review_e2e.py tests/test_runs.py tests/test_review.py tests/test_exports.py -x`
- Estimated runtime: ~120 seconds quick, ~240 seconds full

### Phase Requirements -> Evidence Map

| Requirement | Current automated evidence | Gap Phase 11 must close |
|-------------|----------------------------|--------------------------|
| TST-02 | `tests/test_phase7_benchmark.py` proves algorithm-only parity. | Add live route/task/review/export proof in a dedicated Phase 11 module. |
| VER-01 | Phase 7 rerun and algorithm-only baseline docs prove standalone behavior. | Prove the maintainer web path consumes that behavior correctly from `/runs/new` through sign-off/export. |
| GAP-03 | Slice tests cover uploads, tasks, review gating, and exports independently. | Stitch those seams together in one dedicated integration artifact and keep it stable under temp-dir isolation. |

### Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | TST-02 | T-11-01 | Live `/runs/new` uses temp uploads/out roots and the test DB-backed task session | integration | `uv run pytest -q tests/test_web_run_review_e2e.py -k "submission_persists_packet" -x` | ❌ W0 | ⬜ pending |
| 11-01-02 | 01 | 1 | VER-01 | T-11-02 | Real live run reaches status/review/debug surfaces and blocks sign-off while unresolved debug exceptions remain | integration | `uv run pytest -q tests/test_web_run_review_e2e.py -k "blocks_signoff_until_debug_queue_is_cleared" -x` | ❌ W0 | ⬜ pending |
| 11-02-01 | 02 | 2 | VER-01 | T-11-03 | Cleared live run can sign off and reach signed export routes using the existing snapshot contract | integration | `uv run pytest -q tests/test_web_run_review_e2e.py -k "can_be_cleared_signed_and_exported" -x` | ❌ W0 | ⬜ pending |
| 11-02-02 | 02 | 2 | TST-02 | T-11-04 | Companion seeded case surfaces advisory/export state without depending on unstable live fixture flags | integration | `uv run pytest -q tests/test_web_run_review_e2e.py -k "companion_seeded_snapshot_surfaces_advisories" -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

### Wave 0 Requirements

- [ ] `tests/test_web_run_review_e2e.py` — dedicated Phase 11 integration artifact
- [ ] `tests/conftest.py` — shared live-flow isolation helper only if local duplication becomes error-prone

### Manual-Only Verifications

All phase behaviors have automated verification. Manual HTML/PDF spot-checks are
optional follow-up, not milestone gate requirements.

### Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 240s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

</validation_architecture>

<open_questions>
## Open Questions

None. The current repo evidence is sufficient to plan Phase 11 with `part6` as
the primary live fixture and a seeded companion advisory case.

</open_questions>
