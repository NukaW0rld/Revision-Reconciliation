# Phase 1: Foundation - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the web application shell around the existing `delta_preservation/` computation core: a running Docker container with FastAPI, SQLite, email/password auth, role-based access control, a first-run setup wizard, and Form 3 column mapping UI. The pipeline itself (`delta_preservation/`) is untouched. Engineers and admins can securely access a running shop instance with the correct role permissions and shop configuration established.

</domain>

<decisions>
## Implementation Decisions

### Frontend stack
- Jinja2 + HTMX: FastAPI renders HTML templates; HTMX handles partial page swaps without full reloads
- No separate JS build toolchain at runtime
- Tailwind CSS built at Docker image build time via Tailwind CLI (`npx tailwindcss`); Node.js only in the build stage, not at runtime
- DaisyUI as the Tailwind component library (`btn`, `card`, `badge`, `modal`) for pre-built semantic classes — faster to build review and form UIs
- Server-Sent Events (SSE) via HTMX `sse` extension for live pipeline stage progress updates in Phase 2 (`hx-ext="sse"` + FastAPI `EventSourceResponse`)

### Auth session model
- Server-side sessions: session token in an HttpOnly cookie; session data in a `sessions` SQLite table (`id`, `user_id`, `created_at`, `expires_at`)
- 8-hour sliding window: each request resets the expiry timer; session expires after 8 hours of inactivity
- Engineer accessing an admin-only route → silent redirect to engineer dashboard (no 403 page, no error message)
- Passwords hashed with bcrypt (cost factor 12); no complexity rules enforced — admin creates all accounts, no self-registration

### Setup wizard flow
- FastAPI middleware intercepts every request and redirects to `/setup/*` until a `setup_complete` flag is true in the DB — no route can be bypassed by direct URL navigation
- Linear step-by-step: 4 screens in order — (1) shop name, (2) admin password change, (3) first engineer account creation, (4) Form 3 column mapping
- Can go back to a completed step but cannot skip forward
- Mid-wizard resume: each completed step is persisted to DB immediately; if admin closes the browser, next visit resumes from the first incomplete step
- Default admin credentials seeded via docker-compose.yml env vars (`ADMIN_EMAIL`, `ADMIN_DEFAULT_PASSWORD`); step 2 of the wizard forces a password change and blocks the default password from being reused

### Form 3 column mapping UI
- Table preview: show first 5 rows of the uploaded Excel sheet so admin can see actual data
- Dropdowns above each column let admin assign: `char_no` / `requirement` / `reference_location` / `(ignore)`
- Auto-detect (existing `xlsx.py` keyword matching) pre-fills dropdowns where confident; columns that couldn't be matched are highlighted in amber with `Unmatched` in their dropdown — admin corrects only mismatches
- Fatal errors (unreadable file, empty sheet) are caught before the mapping UI is shown; column-level issues surface in the mapping UI itself
- Column mapping is re-configurable after setup via admin settings panel; changes apply to new runs only
- Shop-wide config (column mapping + shop name) persisted in a `shop_config` SQLite table (not a file on disk)

### Claude's Discretion
- Exact DaisyUI theme/color palette
- SQLite schema details (indexes, migration approach)
- FastAPI app structure (routers, dependency injection patterns)
- Specific Jinja2 template organization

</decisions>

<specifics>
## Specific Ideas

- Default credentials pattern: `admin@shop.local` / `changeme` in docker-compose.yml — documented in README; wizard step 2 forces change before proceeding
- The `xlsx.py` auto-detect logic is already written and should be called server-side during the mapping upload step, with results returned to the HTMX partial that renders the mapping table
- SSE for pipeline progress is chosen now (Phase 1 decision) so the template architecture accommodates it in Phase 2 without a rewrite

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `delta_preservation/io/xlsx.py:load_form3()`: The existing auto-detect column matching logic runs server-side; the web layer calls this (or extracts its header detection logic) to pre-fill the column mapping dropdowns
- `delta_preservation/config.py`: `FORM3_DEFAULT_COLUMNS` and `FORM3_HEADER_KEYWORDS` define the detection vocabulary — the mapping UI maps to these same field names
- `delta_preservation/types.py`: Pydantic models establish the data contract; new web layer Pydantic models should follow the same pattern (field types, optional fields, JSON serialization)

### Established Patterns
- No web layer exists yet — all new code; follow existing Python conventions (snake_case, verb-first function names, explicit type hints)
- Error handling pattern: fail fast with descriptive messages (`FileNotFoundError`, `ValueError` with context); apply same pattern in FastAPI exception handlers
- Print-based progress logging is CLI-only; web layer should use Python `logging` module for server logs

### Integration Points
- `delta_preservation/cli.py:run_pipeline()` is the callable the web layer will invoke from a worker process in Phase 2; Phase 1 doesn't call it but the web architecture should treat it as the async job target
- `out/<run_id>/` directory structure is the pipeline's output contract; web layer will need to serve files from this directory in Phase 3+
- `delta_preservation/config.py` will need to accept column mapping from the `shop_config` DB table rather than hardcoded defaults — plan for this integration point

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-foundation*
*Context gathered: 2026-03-02*
