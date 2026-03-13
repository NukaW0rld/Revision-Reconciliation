# T08: 01-foundation 08

**Slice:** S01 — **Milestone:** M001

## Description

Human verification checkpoint: run `docker compose up --build`, confirm the container starts, walk through the full Phase 1 happy path (login -> setup wizard -> engineer creation -> RBAC), and confirm air-gapped operation.

Purpose: DEPLOY-01/02/03 cannot be verified by automated pytest — they require a real Docker runtime. This checkpoint collects that confirmation before Phase 1 is marked complete.
Output: Human confirmation that the Docker build and full auth/setup flow works end-to-end.

## Must-Haves

- [ ] "docker compose up --build completes without error"
- [ ] "GET http://localhost:8000/login returns 200 from inside the container"
- [ ] "Admin can log in with default credentials, complete the setup wizard, and create an engineer"
- [ ] "Engineer can log in and reach the dashboard but cannot access /admin"
- [ ] "Container runs with no outbound internet access required"
