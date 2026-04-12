# Codebase Concerns & Technical Debt

**Analysis Date:** 2026-04-09

---

## Critical Issues

### Silent Exception Handling

**Issue:** Broad exception handlers without proper logging or recovery in core pipeline paths

**Files:**
- `shop/middleware/setup_guard.py:27` — Catches all exceptions silently in setup guard, logs but returns False
- `shop/tasks.py:281` — Nested exception handlers swallow secondary errors during failure state persistence
- `delta_preservation/reconcile/tolerance_pdf.py:1070` — Catches Exception in tolerance comparison loop, logs to array but continues
- `delta_preservation/vision/grid.py:166` — Generic exception handler in grid inference
- `shop/routers/review.py:261` — Exception in debug verdict submission caught but user gets no feedback

**Impact:** Pipeline failures may silently occur or be partially recorded. Debug information is lost. Users receive no error signals for failed operations.

**Fix approach:** Replace bare `except Exception` with specific exception types. Add user-facing error messages to endpoints. Log full tracebacks with context (run ID, characteristic number) to enable debugging. Consider alerting mechanics for critical exceptions.

---

### PDF Resource Management Issues

**Issue:** PDF documents opened in multiple places without consistent close() guarantees

**Files:**
- `shop/services/runs.py:37-59` — `validate_pdf_bytes()` opens PDF with `fitz.open()` and closes, but repeated calls in request cycle
- `delta_preservation/io/pdf.py:67-98` — `render_page()` closes doc in success path but exceptions after `load_page()` may not close
- `delta_preservation/io/xlsx.py` — Workbook loaded without explicit close in `load_form3()`

**Impact:** File handle exhaustion in long-running processes. Memory leaks when PDFs are large or validation occurs repeatedly.

**Fix approach:** Use context managers (`with fitz.open(...) as doc:`) consistently. Add try-finally blocks in functions without context managers. Profile resource usage in load tests.

---

### Missing Input Validation in JSON Parsing

**Issue:** `delta_packet.json` and review item data parsed without schema validation

**Files:**
- `shop/tasks.py:222-226` — Reads `delta_packet.json` with basic existence check but no validation of packet structure
- `shop/routers/review.py` — Review item fields like `override_classification` accepted without enum validation against valid statuses
- `shop/services/exports.py` — Reads packet JSON without validating required fields before generating PDFs

**Impact:** Malformed or corrupted packet files silently produce incomplete review items or invalid exports. Type errors in PDF generation.

**Fix approach:** Add Pydantic model validation for packet loads. Validate override classifications against enum of allowed statuses. Add schema version checks to packet files.

---

## Technical Debt

### High-Complexity Reconciliation Logic

**Issue:** Classification logic is heuristic-heavy with multiple overlapping decision paths

**Files:**
- `delta_preservation/reconcile/classify.py` (1477 lines) — Single function handles multiple classification strategies, numeric matching, tolerance comparison, semantic comparison, rescue logic
- `delta_preservation/reconcile/match.py` (1154 lines) — Candidate generation and ranking mixed with grouping logic and global fallback strategies
- `delta_preservation/reconcile/tolerance_pdf.py` (1081 lines) — Tolerance extraction, comparison, and debug output generation in one module

**Impact:** Difficult to understand which heuristics apply in edge cases. Changes to one heuristic may unexpectedly affect others. Testing requires coverage of many interaction points.

**Fix approach:** Extract discrete decision functions into smaller modules (e.g., `classify_by_location.py`, `classify_by_numeric.py`, `classify_by_semantic.py`). Each should have clear input/output contracts. Document heuristic assumptions and failure modes.

---

### Incomplete Async/Session Management Pattern

**Issue:** FastAPI routes mix synchronous DB operations with async upload handling inconsistently

**Files:**
- `shop/routers/runs.py:206-250` — `submit_run()` is async but performs synchronous DB operations directly in request handler (not in background task)
- `shop/routers/review.py:527-560` — SSE endpoint uses `asyncio.sleep()` polling instead of event-driven update mechanism
- `shop/dependencies.py` — Session dependency created per-request without explicit close patterns

**Impact:** Potential blocking on long database operations in async routes. SSE polling adds latency to progress updates. Thread-safety concerns in high-concurrency scenarios.

**Fix approach:** Move DB operations to Huey tasks for long-running operations. Use proper async DB patterns (async SQLAlchemy). Replace polling with event-based updates (e.g., WebSocket or database triggers).

---

### Missing Pagination and Resource Limits

**Issue:** List endpoints retrieve all records without pagination or limit

**Files:**
- `shop/routers/runs.py:50` — `list_runs()` calls `.all()` without pagination
- `shop/routers/review.py:111` — `get_review_items()` loads all review items for a run without limit
- `shop/routers/admin.py` — User and run management endpoints missing limits

**Impact:** Memory exhaustion and slow page loads when dataset grows. High database query cost.

**Fix approach:** Add pagination with cursor or offset/limit. Set reasonable defaults (e.g., 50 items per page). Add URL parameters for page and limit.

---

### Type Inconsistencies in Database Layer

**Issue:** Optional fields and nullable relationships not consistently modeled

**Files:**
- `shop/models.py:59-81` — Run model has many Optional fields (current_stage, failure_message, output_dir) with inconsistent initialization
- `shop/models.py:113-124` — ReviewItem has many nullable fields; no validation that matched/unmatched states have consistent field values
- `delta_preservation/types.py` — DeltaPacket schema allows partial data; no validation that evidence and classification are consistent

**Impact:** NULL checks scattered throughout business logic. Potential for uninitialized state bugs.

**Fix approach:** Use TypeScript-style tagged unions or Pydantic discriminated unions to model state machines. Make state and mandatory fields type-safe (e.g., separate `ReviewItemPending` and `ReviewItemCompleted` types).

---

### Missing Retention and Cleanup Safeguards

**Issue:** `cleanup_old_runs()` periodic task lacks safeguards against deleting in-progress work

**Files:**
- `shop/tasks.py:290-337` — Cleanup runs on DELETABLE_STATUSES including "running" and "queued"

**Impact:** Runs could be deleted while still processing. File deletions could race with active tasks. No soft-delete mechanism.

**Fix approach:** Exclude "running" and "queued" from DELETABLE_STATUSES. Add soft-delete flag instead of hard delete. Re-check status before file deletion. Add metrics/alerts for deletion events.

---

### Unvalidated File Paths in Output

**Issue:** File paths from database written directly to URLs and templates without sanitization

**Files:**
- `shop/routers/review.py:99-130` — Returns review items with `revA_snippet_path` and `revB_snippet_path` read directly from database
- `shop/routers/exports.py` — Reads packet path from database and serves directly
- `shop/app.py:71` — Custom filter `basename` extracts filename from path without validation

**Impact:** Path traversal vulnerabilities possible if database is compromised. Typos in stored paths cause broken images.

**Fix approach:** Validate paths are within expected directories before serving. Use UUID-based filenames with registry tables. Serve static files only from designated directories.

---

### Weak Password Validation

**Issue:** No password complexity requirements enforced on creation or reset

**Files:**
- `shop/routers/setup.py` — Admin password set during setup with no strength checks
- `shop/routers/admin.py` — User password creation via admin UI with no validation
- `shop/services/auth.py` — Only hash/verify, no policy enforcement

**Impact:** Weak passwords possible in user accounts.

**Fix approach:** Add password strength validator (minimum length 12 chars, mixed case, special char). Use `zxcvbn` library for entropy checking.

---

### Missing CSRF Protection

**Issue:** POST endpoints lack explicit CSRF tokens

**Files:**
- `shop/routers/*.py` — All POST endpoints use form data or JSON without CSRF token validation
- `shop/app.py` — No CSRF middleware configured

**Impact:** Form-based attacks (CSRF) possible on state-changing operations (sign-off, amendments, admin changes).

**Fix approach:** Add `fastapi-csrf-protect` or similar middleware. Include CSRF tokens in all forms. Validate on POST endpoints.

---

## Test Coverage Gaps

### Reconciliation Logic Untested at Integration Level

**What's not tested:**
- End-to-end pipeline behavior with real Form 3 data + PDFs under concurrency
- Edge cases in classification (tied scores, multiple candidate matches, absent Rev A balloons with Rev B matches)
- Tolerance comparison behavior when tolerances missing from one revision

**Files:**
- `tests/test_classify_bugfixes.py` — Tests specific bugs but not full classification logic
- `tests/test_semantic_comparison.py` — Tests semantic callouts in isolation
- No test for complete pipeline with real-world complex drawings

**Risk:** Medium. Changes to classification logic may introduce regressions in edge cases only visible with real data.

---

### Missing Error Recovery Tests

**What's not tested:**
- Behavior when PDF text extraction fails mid-pipeline
- Handling when Huey task crashes during stage update
- Database recovery when session commits fail
- File upload failures (disk full, permission denied)

**Files:**
- `tests/` — No explicit error recovery tests

**Risk:** Medium. Production failure modes not exercised.

---

### Database Migrations Untested

**What's not tested:**
- Migration rollback behavior
- Schema changes with data consistency
- Multi-version deployment scenarios

**Files:**
- `alembic/` — Migrations exist but no integration tests
- `tests/test_alembic_baseline.py` — Only tests baseline creation

**Risk:** Low in v0.1, but increases as schema evolves.

---

## Performance Concerns

### Inefficient Text Span Processing

**Issue:** Spans deduplicated and sorted multiple times in pipeline

**Files:**
- `delta_preservation/cli.py:61-70` — `_dedupe_spans()` called repeatedly with full span lists
- `delta_preservation/cli.py:87` — `sorted()` call in `_expand_grouped_annotation_spans()` for every item
- `delta_preservation/reconcile/match.py` — Candidate spans scored without spatial indexing

**Impact:** O(n²) complexity in matching phase with large drawings.

**Fix approach:** Build spatial index once at start of pipeline. Pass span IDs instead of objects. Cache sorted results.

---

### Alignment Transform Recomputation

**Issue:** Alignment transform computed multiple times or stored inefficiently

**Files:**
- `delta_preservation/cli.py` — Alignment stage runs once, but transform applied multiple times in snippet generation
- `delta_preservation/vision/alignment.py` — Candidate transform strategies all computed even if first succeeds

**Impact:** CPU overhead during alignment; transforms not persisted efficiently.

**Fix approach:** Memoize transformation functions. Early exit in `estimate_transform_candidates_from_text_spans()`.

---

### Session Cleanup on Expiry

**Issue:** Expired sessions accumulate in database

**Files:**
- `shop/services/auth.py:create_session()` — Creates sessions with expiry time but no cleanup scheduled
- Database has no index on `UserSession.expires_at`

**Impact:** Database bloat over time. Slow session lookups.

**Fix approach:** Add periodic cleanup task for expired sessions. Add database index on expires_at.

---

## Scaling Limits

### Single-Page Processing Constraint

**Issue:** Pipeline only processes page 0; multi-page drawings not fully supported

**Files:**
- `delta_preservation/cli.py` — Hardcoded `page_index=0` in all `render_page()` calls
- `delta_preservation/vision/alignment.py` — Only aligns single pages
- README states as known limitation (#1)

**Impact:** Partial FAIR scoping incorrect for multi-page drawings. Balloons on pages > 0 missed.

**Fix approach:** Extend pipeline to process all pages. Build grid on all pages. Aggregate candidates across pages.

---

### File Upload Size Limits

**Issue:** No explicit file size limit validation

**Files:**
- `shop/routers/runs.py:206-250` — Accepts file uploads without size checks

**Impact:** Large PDF uploads (>500MB) could exhaust memory or disk. No graceful handling.

**Fix approach:** Add Content-Length header validation. Check disk space before save. Return 413 Payload Too Large if exceeded.

---

### Memory Usage During Alignment

**Issue:** ORB feature detection and RANSAC on full-resolution images

**Files:**
- `delta_preservation/vision/alignment.py:estimate_transform()` — Uses full image resolution for ORB

**Impact:** Memory spike on large drawings (11"x17" at 300 DPI = 3600x2550 pixels).

**Fix approach:** Downsample images before ORB (e.g., to 50% resolution). Use memory-efficient descriptor sets.

---

## Dependencies at Risk

### `openpyxl` Potential Issues

**Risk:** `openpyxl.load_workbook()` with `read_only=True` may not handle all Excel formats

**Impact:** Some Form 3 files may fail validation even if valid

**Current mitigation:** Try-catch wraps load with descriptive error

**Recommendation:** Add test with complex .xlsx files (merged cells, formulas, etc.)

---

### `weasyprint` PDF Generation

**Risk:** External dependency on system libraries (GTK, cairo). Docker-specific failures

**Impact:** PDF generation fails in certain environments. No fallback format.

**Current mitigation:** Docker includes all deps; local setup requires manual install

**Recommendation:** Add graceful fallback to HTML export if PDF generation fails.

---

### `opencv-python-headless` Version Pinning

**Risk:** Not pinned; major version updates could break alignment code

**Files:**
- `pyproject.toml:11` — `opencv-python-headless` without version constraint

**Impact:** Deployment variability across environments

**Fix approach:** Pin to `>=4.8,<5.0`.

---

## Security Concerns

### Session Token Generation

**Issue:** Session tokens created with `secrets.token_urlsafe(32)` which is good, but cookie flags not fully hardened

**Files:**
- `shop/routers/auth.py:54` — `secure=False` hardcoded for HTTP (correct), but should auto-detect from URL scheme
- `shop/services/auth.py:create_session()` — No rate limiting on session creation

**Impact:** Login brute-force possible; HTTP deployment OK but HTTPS misconfiguration silent

**Fix approach:** Auto-detect scheme from request. Rate limit login attempts.

---

### No Input Sanitization on Review Notes

**Issue:** `override_note` field in ReviewItem accepts free text without sanitization

**Files:**
- `shop/routers/review.py:326-340` — Notes submitted to database without XSS checks
- `shop/templates/` — Templates may render notes with insufficient escaping

**Impact:** Stored XSS if notes displayed in HTML contexts

**Fix approach:** Use Jinja2 autoescaping (enabled by default). Add `| escape` filter in templates explicitly. Store notes as plain text only.

---

## Known Limitations (Not Bugs)

These are documented in README but worth noting:

1. **Single-page processing** — Pipeline only processes page 0 of PDFs (known limitation in README)
2. **Vector PDF requirement** — Scanned/raster drawings rejected by web validation (by design)
3. **Ballooning assumptions** — Relies on numbered characteristics for quality (by design)
4. **Heuristic classification** — Intentionally surfaces uncertain cases for human review (by design)
5. **No external integrations** — PLM/QMS systems not included in scope (noted in README)

---

## Recommendations by Priority

**High (Fix in next sprint):**
1. Replace bare `except Exception` handlers with specific exceptions + logging
2. Validate packet JSON schema on load
3. Use context managers for all file operations
4. Add pagination to list endpoints

**Medium (Fix in next release):**
1. Refactor classification module into discrete decision functions
2. Add password strength validation
3. Implement soft-delete for retention policy
4. Add CSRF protection middleware

**Low (Improve for scale):**
1. Build spatial indices for text spans
2. Add multi-page support to pipeline
3. Implement async DB operations with `asyncio-sqlalchemy`
4. Add memory profiling to alignment stage

---

*Concerns audit: 2026-04-09*
