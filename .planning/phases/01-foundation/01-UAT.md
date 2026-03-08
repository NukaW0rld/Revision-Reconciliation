---
status: complete
phase: 01-foundation
source: 01-01-SUMMARY.md, 01-02-SUMMARY.md, 01-03-SUMMARY.md, 01-04-SUMMARY.md, 01-05-SUMMARY.md, 01-06-SUMMARY.md, 01-07-SUMMARY.md, 01-08-SUMMARY.md
started: 2026-03-03T00:00:00Z
updated: 2026-03-03T00:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Cold Start Smoke Test
expected: Kill any running container. From the repo root, run `docker compose -f docker/docker-compose.yml up --build`. Build completes without errors. Container starts and uvicorn logs show listening on 0.0.0.0:8000. Navigating to http://localhost:8000 redirects to /login and the login page loads (200 OK, no 500 errors in logs).
result: pass

### 2. Root URL Redirect
expected: Navigating to http://localhost:8000 (or http://localhost:8000/) redirects to /login. The login form is displayed, not a 404 error.
result: pass

### 3. Login Page Load
expected: GET /login returns a login form with email and password fields and a submit button. Page uses Tailwind/DaisyUI styling served from /static/ (no CDN links in page source).
result: pass

### 4. Login with Valid Credentials
expected: Submit the seeded admin credentials (ADMIN_EMAIL + ADMIN_DEFAULT_PASSWORD from docker-compose.yml). Browser redirects to /dashboard. Dashboard shows the logged-in user's email and a logout button.
result: pass

### 5. Login with Invalid Credentials
expected: Submit an incorrect password on the login form. The page reloads with an error message visible (e.g., "Invalid credentials"). No redirect to dashboard.
result: pass

### 6. Logout Flow
expected: Click the logout button on the dashboard. Browser redirects to /login. Navigating back to /dashboard without re-logging in redirects back to /login (session cleared).
result: pass

### 7. Setup Wizard Intercepts Unauthenticated Access
expected: With setup_complete=False (fresh container before completing wizard), navigating to any app URL (e.g., /dashboard) redirects to /setup/step1 instead of showing the page. The setup wizard is the only accessible flow.
result: pass

### 8. Setup Wizard Step 1: Shop Name
expected: GET /setup/step1 shows a form with a shop name input field. Submitting a shop name advances wizard to step 2 (redirects to /setup/step2).
result: pass

### 9. Setup Wizard Step 2: Password Change
expected: Step 2 displays the seeded admin email (e.g., "You will log in as: admin@shop.local"). Submitting "changeme" as the new password shows an error and does NOT advance. Submitting a valid new password (≥8 chars, not "changeme") advances to step 3.
result: pass

### 10. Setup Wizard Step 3: First Engineer Account
expected: Step 3 shows a form to create the first engineer account with email and password fields. Submitting creates the account and advances to step 4.
result: pass

### 11. Setup Wizard Step 4: Form 3 Upload and Column Mapping
expected: Step 4 shows an HTMX file upload form. Uploading a valid Form 3 Excel file triggers an HTMX partial swap showing a column mapping table with dropdown selects pre-populated via auto-detection. Saving the mapping sets setup_complete=True and redirects to /login (or /dashboard).
result: pass

### 12. Setup Wizard Step 4: Empty File Error
expected: Uploading an empty or unreadable file on step 4 returns an error partial (DaisyUI alert-error style) swapped inline. No column dropdowns appear. The wizard does not advance.
result: pass

### 13. Setup Wizard Cannot Skip Steps
expected: Attempting to navigate directly to /setup/step3 without completing steps 1 and 2 redirects to the first incomplete step (e.g., /setup/step1 or /setup/step2). Steps cannot be skipped by URL manipulation.
result: pass

### 14. Admin User List
expected: After setup, log in as admin and navigate to /admin/users. A table of users is displayed showing at least the admin account and any created engineers. Each row has a Deactivate button.
result: pass

### 15. Admin Creates Engineer Account
expected: On /admin/users, fill in the "Add Engineer" form with email and password and submit. The new engineer appears as a new row in the table via HTMX inline swap — no full page reload.
result: pass

### 16. Admin Deactivates Engineer
expected: Click the Deactivate button for an engineer. A browser confirm dialog appears ("are you sure?" style). Confirming updates that table row inline (HTMX swap) to show the engineer as inactive — no full page reload.
result: pass

### 17. RBAC: Engineer Redirected from Admin Routes
expected: Log in as an engineer (not admin). Navigating to /admin/users silently redirects to /dashboard. No 403 or error page — just a silent redirect.
result: pass

### 18. Admin Dashboard Navigation Cards
expected: Log in as admin. The dashboard at /dashboard shows navigation cards for "User Management" (links to /admin/users) and "Settings" (links to /admin/settings). Engineer login shows a minimal dashboard without those cards.
result: pass

### 19. Admin Settings Panel
expected: Navigate to /admin/settings. The page shows current shop name and the Form 3 column mapping. Uploading a new Excel file triggers HTMX column mapping preview. Saving updates the mapping without re-running the setup wizard.
result: pass

## Summary

total: 19
passed: 19
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
