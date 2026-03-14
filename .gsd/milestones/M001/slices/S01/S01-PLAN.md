# S01: Foundation

**Goal:** Create the test infrastructure scaffold for Phase 1 before any production code is written.
**Demo:** Create the test infrastructure scaffold for Phase 1 before any production code is written.

## Must-Haves


## Tasks

- [x] **T01: 01-foundation 01** `est:2min`
  - Create the test infrastructure scaffold for Phase 1 before any production code is written. This establishes the pytest fixtures, in-memory SQLite override, and test stubs that all subsequent plans' verify steps depend on.

Purpose: Without this Wave 0 plan, no automated verify command can pass — every subsequent plan's `<automated>` command references test files that don't yet exist.
Output: Working pytest setup with TestClient fixture; all test files importable with stubs that fail gracefully (pytest.skip or xfail) until production code arrives.
- [x] **T02: 01-foundation 02** `est:7min`
  - Create the core `shop/` Python package: database engine, ORM models, password hashing service, and FastAPI dependency functions. This is the foundation all subsequent plans import from.

Purpose: Plans 03-07 all depend on `shop.models`, `shop.dependencies`, and `shop.services.auth`. Nothing can be tested until this package exists.
Output: Importable `shop/` package with working DB layer, auth service, and dependency injection chain.
- [x] **T03: 01-foundation 03** `est:15min`
  - Build the login/logout auth routes, base Jinja2 template, and bundle HTMX + SSE extension as local static files. After this plan, engineers and admins can log in, maintain a session, and log out.

Purpose: Auth routes are the entry point for all authenticated flows. The base template establishes the HTMX + Tailwind CSS pattern used by all subsequent templates.
Output: Working login/logout flow; base.html that all templates extend; HTMX bundled locally for air-gapped operation.
- [x] **T04: 01-foundation 04** `est:8min`
  - Build the admin user management router: create engineer accounts, list users, deactivate accounts. Implements AUTH-01 (admin-initiated account creation), AUTH-04 (view/deactivate), AUTH-05 (RBAC enforcement), and AUTH-06 (timestamp on actions).

Purpose: This plan runs parallel to Plan 03 (auth routes) because both depend only on Plan 02 (core package) and touch different files.
Output: Admin can manage engineer accounts; RBAC enforced via require_admin dependency.
- [x] **T05: 01-foundation 05** `est:2min`
  - Build the setup wizard middleware guard and steps 1-3: shop name, admin password change, and first engineer account creation. After this plan, the undismissable setup guard is in effect and the first three wizard steps are completable.

Purpose: SETUP-01 (undismissable wizard) is one of the most critical Phase 1 requirements — the system must not allow any authenticated operation until configuration is complete.
Output: Working wizard middleware + steps 1-3 with step-ordering enforcement and mid-wizard resume.
- [x] **T06: 01-foundation 06** `est:3min`
  - Build wizard step 4: Form 3 column mapping. The admin uploads a sample Excel file, sees a preview table with auto-detected column assignments, corrects any mismatches, and saves. Also adds the admin settings panel for post-setup reconfiguration.

Purpose: SETUP-02 and SETUP-03 are the most complex UI in Phase 1 — two-phase validation (fatal errors before UI, column issues in UI) with HTMX partial swap for the mapping table.
Output: Step 4 completable, setup_complete flag set, admin can reconfigure later.
- [x] **T07: 01-foundation 07** `est:4min`
  - Create the Docker multi-stage build configuration, docker-compose.yml, Tailwind/DaisyUI build setup, and a startup entrypoint script. After this plan, the entire application runs via `docker compose up --build` with no external services.

Purpose: DEPLOY-01, DEPLOY-02, and DEPLOY-03 are the deployment requirements. The multi-stage build bundles Node (Tailwind), Python (uv), and the runtime together; no internet access needed at runtime.
Output: Working docker-compose.yml; Dockerfile with Node→uv→runtime stages; admin user seeded from env vars on first start.
- [x] **T08: 01-foundation 08** `est:~6h (includes human Docker build and verification time)`
  - Human verification checkpoint: run `docker compose up --build`, confirm the container starts, walk through the full Phase 1 happy path (login -> setup wizard -> engineer creation -> RBAC), and confirm air-gapped operation.

Purpose: DEPLOY-01/02/03 cannot be verified by automated pytest — they require a real Docker runtime. This checkpoint collects that confirmation before Phase 1 is marked complete.
Output: Human confirmation that the Docker build and full auth/setup flow works end-to-end.

## Files Likely Touched

- `pyproject.toml`
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_auth.py`
- `tests/test_rbac.py`
- `tests/test_models.py`
- `tests/test_setup.py`
- `tests/test_admin.py`
- `shop/__init__.py`
- `shop/app.py`
- `shop/database.py`
- `shop/models.py`
- `shop/dependencies.py`
- `shop/services/__init__.py`
- `shop/services/auth.py`
- `shop/routers/__init__.py`
- `shop/routers/auth.py`
- `shop/templates/base.html`
- `shop/templates/auth/login.html`
- `shop/templates/auth/login_error.html`
- `shop/templates/dashboard.html`
- `static/js/htmx.min.js`
- `static/js/htmx-sse.js`
- `static/src/input.css`
- `shop/routers/admin.py`
- `shop/templates/admin/users.html`
- `shop/templates/admin/users_row.html`
- `shop/routers/setup.py`
- `shop/templates/setup/step1_shop_name.html`
- `shop/templates/setup/step2_password.html`
- `shop/templates/setup/step3_engineer.html`
- `shop/templates/setup/wizard_layout.html`
- `shop/routers/setup.py`
- `shop/services/form3.py`
- `shop/templates/setup/step4_column_mapping.html`
- `shop/templates/setup/step4_error.html`
- `shop/templates/setup/step4_mapping_partial.html`
- `shop/templates/admin/settings.html`
- `docker/Dockerfile`
- `docker/docker-compose.yml`
- `package.json`
- `tailwind.config.js`
- `run_web.py`
