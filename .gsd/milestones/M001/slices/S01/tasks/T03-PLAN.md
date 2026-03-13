# T03: 01-foundation 03

**Slice:** S01 — **Milestone:** M001

## Description

Build the login/logout auth routes, base Jinja2 template, and bundle HTMX + SSE extension as local static files. After this plan, engineers and admins can log in, maintain a session, and log out.

Purpose: Auth routes are the entry point for all authenticated flows. The base template establishes the HTMX + Tailwind CSS pattern used by all subsequent templates.
Output: Working login/logout flow; base.html that all templates extend; HTMX bundled locally for air-gapped operation.

## Must-Haves

- [ ] "POST /login with valid credentials sets an HttpOnly session_token cookie and redirects to /dashboard"
- [ ] "POST /login with invalid credentials returns the login page with an error message (no redirect)"
- [ ] "POST /logout deletes the session row and clears the session_token cookie"
- [ ] "GET /dashboard with a valid session cookie returns 200"
- [ ] "GET /dashboard without a cookie redirects to /login"
- [ ] "base.html loads HTMX from /static/js/htmx.min.js (not a CDN)"
- [ ] "base.html loads CSS from /static/dist/output.css (no CDN links anywhere in templates)"

## Files

- `shop/routers/__init__.py`
- `shop/routers/auth.py`
- `shop/templates/base.html`
- `shop/templates/auth/login.html`
- `shop/templates/auth/login_error.html`
- `shop/templates/dashboard.html`
- `static/js/htmx.min.js`
- `static/js/htmx-sse.js`
- `static/src/input.css`
