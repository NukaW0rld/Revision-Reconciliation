# T07: 01-foundation 07

**Slice:** S01 — **Milestone:** M001

## Description

Create the Docker multi-stage build configuration, docker-compose.yml, Tailwind/DaisyUI build setup, and a startup entrypoint script. After this plan, the entire application runs via `docker compose up --build` with no external services.

Purpose: DEPLOY-01, DEPLOY-02, and DEPLOY-03 are the deployment requirements. The multi-stage build bundles Node (Tailwind), Python (uv), and the runtime together; no internet access needed at runtime.
Output: Working docker-compose.yml; Dockerfile with Node→uv→runtime stages; admin user seeded from env vars on first start.

## Must-Haves

- [ ] "`docker compose up --build` from the project root builds and starts the container without error"
- [ ] "The container exposes port 8000; GET http://localhost:8000/login returns 200 or redirect"
- [ ] "No outbound internet access is required at container runtime (all assets served locally)"
- [ ] "All Python and pipeline dependencies are installed inside the image via uv"
- [ ] "Tailwind CSS is compiled inside the Docker build (Node.js not present at runtime)"
- [ ] "shop.db and out/ directory are persisted via Docker volume mounts"

## Files

- `docker/Dockerfile`
- `docker/docker-compose.yml`
- `package.json`
- `tailwind.config.js`
- `run_web.py`
