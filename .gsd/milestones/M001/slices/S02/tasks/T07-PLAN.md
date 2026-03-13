# T07: 02-pipeline-bridge 07

**Slice:** S02 — **Milestone:** M001

## Description

Update the Docker setup to run uvicorn and huey_consumer together under supervisord in the single container.

Purpose: The Huey worker must run as a separate OS process alongside uvicorn. Supervisord is the established pattern for managing multiple processes in a single Docker container (air-gapped deployment constraint).
Output: docker/supervisord.conf with uvicorn + huey_consumer programs, updated Dockerfile with supervisor install and supervisord CMD, updated docker-compose.yml with HUEY_DB env var.

## Must-Haves

- [ ] "docker compose up runs both uvicorn and huey_consumer under supervisord in a single container"
- [ ] "supervisord.conf starts uvicorn on port 8000 and huey_consumer.py shop.tasks.huey with --workers=1 --worker-type=thread"
- [ ] "HUEY_DB env var in docker-compose.yml points to /app/data/huey.db (same volume as shop.db)"
- [ ] "Dockerfile installs supervisor via apt-get and runs supervisord as CMD"
- [ ] "uv is available in the runtime image to run huey_consumer.py"

## Files

- `docker/Dockerfile`
- `docker/docker-compose.yml`
- `docker/supervisord.conf`
- `pyproject.toml`
- `shop/templates/setup`
