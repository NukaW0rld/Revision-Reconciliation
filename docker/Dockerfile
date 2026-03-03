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
# Copy virtualenv from builder
COPY --from=python-builder /app/.venv /app/.venv
# Copy compiled CSS from Tailwind builder
COPY --from=tailwind-builder /app/static/dist/output.css ./static/dist/output.css
# Copy application source
COPY delta_preservation/ ./delta_preservation/
COPY shop/ ./shop/
COPY static/js/ ./static/js/
COPY static/src/ ./static/src/
COPY run.py run_web.py ./
# Activate venv
ENV PATH="/app/.venv/bin:$PATH"
# Create data directories
RUN mkdir -p /app/data /app/out
# Expose port
EXPOSE 8000
CMD ["python", "run_web.py"]
