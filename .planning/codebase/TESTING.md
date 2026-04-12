# Testing Patterns

**Analysis Date:** 2026-04-09

## Test Framework

**Runner:**
- pytest 8+ (specified in `pyproject.toml`)
- Config: `pyproject.toml` with `[tool.pytest.ini_options]`
- Test discovery: `testpaths = ["tests"]`
- Output: `-q` flag (quiet mode)

**Assertion Library:**
- Standard `assert` statements
- No custom assertion library (pytest built-in assertions used)

**Run Commands:**
```bash
pytest tests/                    # Run all tests
pytest tests/test_auth.py        # Run specific test file
pytest -v                        # Verbose output
pytest -k "test_name"            # Run tests matching pattern
pytest --collect-only            # List all tests without running
```

## Test File Organization

**Location:**
- Co-located in `/home/khoa2/delta-preservation/tests/` directory (separate from source)
- Not alongside source code; centralized test directory

**Naming:**
- `test_<module>.py` pattern: `test_auth.py`, `test_models.py`, `test_setup.py`, `test_grid.py`
- Test function names: `def test_<scenario>()` with descriptive names
- Examples: `test_session_persists()`, `test_setup_step1_post_advances_wizard()`, `test_grid_inference_part1_revA()`

**Structure:**
```
tests/
├── conftest.py                          # Shared fixtures and configuration
├── test_auth.py                         # Authentication tests
├── test_setup.py                        # Setup wizard tests
├── test_models.py                       # Database model tests
├── test_grid.py                         # Drawing grid inference tests
├── test_semantic_extraction.py          # Semantic callout parsing tests
├── test_reconcile_semantic_integration.py  # Reconciliation integration
├── test_amendments.py                   # Amendment workflow tests
├── test_cli_stage_callback.py           # CLI pipeline stage tests
├── test_pipeline_task.py                # Background task tests
└── ... (20+ test modules total)
```

## Test Structure

**Suite Organization:**
Standard pytest discovery. Tests are flat functions with no explicit test classes (except for rare organizational groupings like `TestInferGrid` in `test_grid.py`).

**Typical test structure:**
```python
def test_feature_description(fixture1, fixture2):
    """Human-readable test description with pass/fail criteria.
    
    Steps:
    - Step 1 with action
    - Step 2 with assertion
    - Step 3 with final check
    """
    # Arrange: Setup initial state
    response = fixture1.get("/endpoint", follow_redirects=False)
    
    # Act: Perform the test action (often implicit in first step)
    
    # Assert: Verify expected outcome
    assert response.status_code == 200
    assert "expected text" in response.text
```

**Patterns:**

1. **HTTP endpoint testing:**
   - Use `TestClient` from FastAPI: `from fastapi.testclient import TestClient`
   - Fixture provides pre-configured client: `client` fixture in `conftest.py`
   - Example from `test_auth.py`:
   ```python
   def test_session_persists(client, admin_user):
       resp = client.post("/login", data={"username": "admin", "password": "changeme"})
       assert resp.status_code == 302
       dashboard_resp = client.get("/dashboard")
       assert dashboard_resp.status_code == 200
   ```

2. **Database tests:**
   - Create in-memory SQLite: `sqlite://` with `StaticPool` in `conftest.py`
   - Session fixtures provided by `db_engine` fixture
   - Example from `test_amendments.py`:
   ```python
   def test_create_amendment(client, engineer_user):
       db_gen = client.app.dependency_overrides.get(get_db)
       db = next(db_gen())
       parent = _make_signed_run_with_items(db, engineer_user.id)
       assert parent.id is not None
   ```

3. **Core algorithm tests:**
   - Use minimal synthetic data fixtures
   - Example from `test_semantic_extraction.py`:
   ```python
   def _span(text: str, *, block_id: int = 0, x0: float = 0.0):
       return TextSpan(
           text=text,
           bbox_pdf=(x0, y0, x0 + 10.0, y0 + 5.0),
           font_size=10.0,
           block_id=block_id,
           line_id=line_id,
           span_id=span_id,
       )
   
   def test_extract_semantic_callout_gdt_parsed():
       semantic = extract_semantic_callout(pdf_spans=[_span("⌖ ...")])
       assert semantic.status.state == "parsed"
   ```

## Mocking

**Framework:** `unittest.mock` from standard library
- `MagicMock` for creating mock objects
- `patch()` for replacing functions/modules during tests
- Example from `test_cli_stage_callback.py`:
```python
with patch("delta_preservation.cli.load_form3", return_value=[]), \
     patch("delta_preservation.cli.detect_balloons", return_value={}), \
     patch("delta_preservation.cli.estimate_transform", return_value=MagicMock(...)):
    # Test code here
```

**What to Mock:**
- External I/O: file system operations, PDF processing, image rendering
- Database operations when testing business logic (use `db_engine` fixture instead)
- Long-running background tasks (set `huey.immediate = True` via `huey_immediate` fixture)
- Network calls (not present in current codebase)

**What NOT to Mock:**
- Core algorithm logic (test actual implementations)
- Database models (use real in-memory SQLite via `db_engine` fixture)
- Pydantic validation (test with real model instances)
- Router/endpoint handlers (test with real FastAPI app via `client` fixture)

## Fixtures and Factories

**Test Data Pattern:**
Inline helper functions with `_` prefix for creating test objects:

From `test_semantic_extraction.py`:
```python
def _span(text: str, *, block_id: int = 0, line_id: int = 0, span_id: int = 0, x0: float = 0.0):
    return TextSpan(
        text=text,
        bbox_pdf=(x0, y0, x0 + 10.0, y0 + 5.0),
        font_size=10.0,
        block_id=block_id,
        line_id=line_id,
        span_id=span_id,
    )
```

From `test_amendments.py`:
```python
def _make_signed_run_with_items(db, engineer_id):
    """Create signed-off run with 3 ReviewItems for amendment testing."""
    run = Run(
        part_number="PN-AMEND",
        status="signed_off",
        revA_path="/tmp/revA.pdf",
        # ... more fields
    )
    db.add(run)
    db.flush()
    # Add related items
    return run
```

**Pytest Fixtures:**
Centralized in `conftest.py` with function scope (recreated per test):

1. **Database Fixtures:**
   - `db_engine`: In-memory SQLite with all tables
   - Provides clean isolation between tests
   ```python
   @pytest.fixture(scope="function")
   def db_engine():
       engine = create_engine("sqlite://", ...)
       Base.metadata.create_all(engine)
       yield engine
       Base.metadata.drop_all(engine)
   ```

2. **Client Fixtures:**
   - `client`: FastAPI TestClient with setup_complete=True
   - `client_setup_incomplete`: TestClient for setup wizard testing (setup_complete=False)
   - Both override `get_db` dependency with test session factory

3. **User Fixtures:**
   - `admin_user`: Creates admin user with role="admin"
   - `engineer_user`: Creates engineer user with role="engineer"
   - Both seeded with hashed password "changeme"

4. **Configuration Fixtures:**
   - `shop_config`: Seeded ShopConfig with setup_complete=False
   - `huey_immediate`: Sets task queue to synchronous mode for test isolation

**Location:** `tests/conftest.py` (lines 1-141)

## Coverage

**Requirements:** Not enforced (no coverage config in `pyproject.toml`)

**Coverage Status:** 261 test functions across 20+ test modules (as of 2026-04-09)

**Test Categories:**
- Authentication & Authorization: `test_auth.py`, `test_rbac.py`
- Setup Wizard: `test_setup.py`
- Pipeline Execution: `test_cli_stage_callback.py`, `test_pipeline_task.py`
- Core Algorithms: `test_grid.py`, `test_alignment_multishift.py`, `test_semantic_extraction.py`
- Reconciliation: `test_reconcile_semantic_integration.py`, `test_classify_bugfixes.py`
- Workflow: `test_amendments.py`, `test_review.py`, `test_admin.py`
- Data Formatting: `test_output_formatting.py`
- Debugging: `test_debug_internals.py`, `test_debug_verdicts.py`

## Test Types

**Unit Tests:**
- Scope: Individual functions and classes in isolation
- Examples: `test_semantic_extraction.py` tests parser outputs with synthetic spans
- Approach: Synthetic data fixtures, no database, no HTTP

**Integration Tests:**
- Scope: Multiple components working together
- Examples: 
  - `test_reconcile_semantic_integration.py`: Tests matching + semantic extraction + classification pipeline
  - `test_auth.py`: Tests login → session persistence → logout workflow
  - `test_setup.py`: Tests full setup wizard flow (step1 → step2 → step3)
- Approach: Real FastAPI app with test client, real database

**E2E Tests:**
- Framework: Not present
- No end-to-end tests detected (would require full PDF/XLSX file processing)
- Integration tests serve as closest equivalent

## Common Patterns

**Async Testing:**
Not applicable—no async code detected in codebase. All tests are synchronous.

**Error Testing:**
Tests explicitly check error conditions and error messages:

From `test_auth.py`:
```python
def test_login_invalid_credentials(client, admin_user):
    """POST /login with wrong password returns 200 with error message."""
    resp = client.post(
        "/login",
        data={"username": "admin", "password": "wrongpass"},
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "Invalid username or password" in resp.text
```

From `test_setup.py`:
```python
def test_setup_step3_rejects_default_password(client_setup_incomplete, db_engine):
    """POST /setup/step3 with 'changeme' returns 200 with error message."""
    response = client_setup_incomplete.post(
        "/setup/step3",
        data={"admin_username": "admin", "new_password": "changeme", ...},
        follow_redirects=False,
    )
    assert response.status_code == 200
    assert "Cannot reuse the default password" in response.text
```

**Parametrized Tests:**
Not heavily used; tests are written as individual functions with descriptive names instead of `@pytest.mark.parametrize`.

**Test Markers:**
- `@pytest.mark.xfail(strict=False, reason="...")`: Tests marked for future implementation
- Example from `test_auth.py`:
  ```python
  @pytest.mark.xfail(strict=False, reason="Admin user creation requires Plan 05 (admin routes)")
  def test_admin_creates_engineer(client, admin_user):
      raise NotImplementedError("Admin routes not yet implemented — Plan 05")
  ```

**Fixtures with Cleanup:**
Fixtures use `yield` for setup/teardown:
```python
@pytest.fixture(scope="function")
def db_engine():
    engine = create_engine(...)
    Base.metadata.create_all(engine)
    yield engine  # Test runs here
    Base.metadata.drop_all(engine)  # Cleanup
```

**Database Transactions:**
Tests use session factories that auto-commit:
```python
db = Session()
try:
    db.add(user)
    db.commit()
finally:
    db.close()
```

## Dependency Injection in Tests

FastAPI dependency overrides used extensively:

From `conftest.py`:
```python
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app = create_app(session_factory=TestingSessionLocal)
app.dependency_overrides[get_db] = override_get_db
```

This allows tests to inject test database sessions without modifying source code.

## Known Testing Gaps

**Unimplemented Tests:**
- `test_models.py`: Reviewer fields test marked xfail (shop/ package pending)
- `test_auth.py::test_admin_creates_engineer`: Admin routes pending Plan 05

**Areas with Limited Coverage:**
- PDF rendering edge cases (most PDF tests use mocks)
- High-DPI rendering variations
- Large file performance
- Concurrent request handling

---

*Testing analysis: 2026-04-09*
