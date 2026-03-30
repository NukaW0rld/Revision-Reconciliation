# Stage 1: Tailwind CSS compilation (Node.js)
FROM node:22-alpine AS tailwind-builder
WORKDIR /app
COPY package.json ./
RUN npm install
COPY static/src/ ./static/src/
# Copy all templates so Tailwind can scan for class usage
COPY shop/templates/ ./shop/templates/
RUN npm run build:css

# Stage 2: Python dependency builder using uv
FROM python:3.11-slim AS python-builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Install dependencies without installing the project itself
RUN uv sync --locked --no-install-project

# Stage 3: Runtime
FROM python:3.11-slim AS runtime
WORKDIR /app
# Install supervisor (manages uvicorn + huey_consumer as separate OS processes)
RUN apt-get update && apt-get install -y --no-install-recommends supervisor libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 && rm -rf /var/lib/apt/lists/*
# Copy virtualenv from builder
COPY --from=python-builder /app/.venv /app/.venv
# Copy uv binary (needed for huey_consumer.py discovery)
COPY --from=python-builder /bin/uv /bin/uv
# Copy compiled CSS from Tailwind builder
COPY --from=tailwind-builder /app/static/dist/output.css ./static/dist/output.css
# Copy application source
COPY delta_preservation/ ./delta_preservation/
COPY shop/ ./shop/
COPY static/js/ ./static/js/
COPY static/src/ ./static/src/
COPY run.py run_web.py alembic.ini pyproject.toml uv.lock ./
COPY alembic/ ./alembic/
# Copy supervisord config
COPY docker/supervisord.conf /etc/supervisor/conf.d/delta.conf
# Activate venv
ENV PATH="/app/.venv/bin:$PATH"
# Create data directories
RUN mkdir -p /app/data /app/data/uploads /app/out
# Expose port
EXPOSE 8000
# Run Alembic migrations then start supervisord (manages uvicorn + huey_consumer)
CMD ["sh", "-c", "alembic upgrade head && exec supervisord -c /etc/supervisor/supervisord.conf"]
