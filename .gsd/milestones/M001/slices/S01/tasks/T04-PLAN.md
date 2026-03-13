# T04: 01-foundation 04

**Slice:** S01 — **Milestone:** M001

## Description

Build the admin user management router: create engineer accounts, list users, deactivate accounts. Implements AUTH-01 (admin-initiated account creation), AUTH-04 (view/deactivate), AUTH-05 (RBAC enforcement), and AUTH-06 (timestamp on actions).

Purpose: This plan runs parallel to Plan 03 (auth routes) because both depend only on Plan 02 (core package) and touch different files.
Output: Admin can manage engineer accounts; RBAC enforced via require_admin dependency.

## Must-Haves

- [ ] "Admin can create an engineer account via POST /admin/users"
- [ ] "Admin can view the list of all users at GET /admin/users"
- [ ] "Admin can deactivate an engineer account via POST /admin/users/{id}/deactivate"
- [ ] "Engineer accessing GET /admin/users is silently redirected to /dashboard (no 403)"
- [ ] "Every user creation records created_at timestamp; deactivation is a flag not a delete"

## Files

- `shop/routers/admin.py`
- `shop/templates/admin/users.html`
- `shop/templates/admin/users_row.html`
