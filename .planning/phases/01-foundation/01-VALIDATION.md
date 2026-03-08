---
phase: 1
slug: foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-02
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (already in `pyproject.toml`) |
| **Config file** | none — Wave 0 adds `[tool.pytest.ini_options]` to `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/ -x -q` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 0 | AUTH-01 | integration | `uv run pytest tests/test_auth.py::test_admin_creates_engineer -x` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 0 | AUTH-02 | integration | `uv run pytest tests/test_auth.py::test_session_persists -x` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 0 | AUTH-03 | integration | `uv run pytest tests/test_auth.py::test_logout -x` | ❌ W0 | ⬜ pending |
| 1-01-04 | 01 | 0 | AUTH-04 | integration | `uv run pytest tests/test_admin.py::test_deactivate_engineer -x` | ❌ W0 | ⬜ pending |
| 1-01-05 | 01 | 0 | AUTH-05 | integration | `uv run pytest tests/test_rbac.py::test_engineer_cannot_access_admin -x` | ❌ W0 | ⬜ pending |
| 1-01-06 | 01 | 0 | AUTH-06 | unit | `uv run pytest tests/test_models.py::test_reviewer_fields -x` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 0 | SETUP-01 | integration | `uv run pytest tests/test_setup.py::test_wizard_intercepts_all_routes -x` | ❌ W0 | ⬜ pending |
| 1-02-02 | 02 | 0 | SETUP-02 | integration | `uv run pytest tests/test_setup.py::test_form3_upload_autodetect -x` | ❌ W0 | ⬜ pending |
| 1-02-03 | 02 | 0 | SETUP-03 | integration | `uv run pytest tests/test_setup.py::test_empty_file_error -x` | ❌ W0 | ⬜ pending |
| 1-02-04 | 02 | 0 | SETUP-04 | unit | `uv run pytest tests/test_setup.py::test_noncontiguous_char_no -x` | ❌ W0 | ⬜ pending |
| 1-03-01 | 03 | 1 | DEPLOY-01 | manual | `docker compose up --build` + smoke test GET / | N/A | ⬜ pending |
| 1-03-02 | 03 | 1 | DEPLOY-02 | manual | Run `uv run python -c "import shop"` inside container post-build | N/A | ⬜ pending |
| 1-03-03 | 03 | 1 | DEPLOY-03 | manual | Network-isolated container test | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/__init__.py` — makes tests/ a package
- [ ] `tests/conftest.py` — in-memory SQLite engine, TestClient fixture, admin user seed
- [ ] `tests/test_auth.py` — AUTH-01 through AUTH-05 stubs
- [ ] `tests/test_rbac.py` — AUTH-05 (RBAC enforcement) stubs
- [ ] `tests/test_models.py` — AUTH-06 (reviewer fields) stubs
- [ ] `tests/test_setup.py` — SETUP-01 through SETUP-04 stubs
- [ ] `tests/test_admin.py` — AUTH-04 (user management) stubs
- [ ] Add `[tool.pytest.ini_options]` to `pyproject.toml` with `testpaths = ["tests"]`
- [ ] `uv add --dev httpx` — required by FastAPI TestClient for async testing

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `docker compose up` succeeds, no external services | DEPLOY-01 | Requires Docker environment, network isolation | Run `docker compose up --build`; smoke test `GET /` returns 200 |
| All deps bundled in image | DEPLOY-02 | Container runtime verification | Run `uv run python -c "import shop"` inside container post-build |
| No outbound HTTP at runtime | DEPLOY-03 | Network-level verification | Run container with `--network none` or `iptables` isolation, trigger requests |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
