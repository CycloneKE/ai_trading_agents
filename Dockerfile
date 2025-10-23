
## Multi-stage Dockerfile optimized for production

# -----------------------
# Builder: frontend
# -----------------------
FROM node:20 AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN if [ -f package.json ]; then npm ci --legacy-peer-deps || npm install --legacy-peer-deps; fi
COPY frontend/ ./
RUN if [ -f package.json ]; then npm run build || echo "frontend build failed or not present"; fi


# -----------------------
# Builder: python dependencies (cacheable)
# -----------------------
FROM python:3.11-slim AS deps-builder
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system build deps required for some wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc libpq-dev ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY requirements-runtime.txt ./requirements-runtime.txt

# Upgrade pip and install runtime deps (cacheable layer)
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-runtime.txt


# -----------------------
# Optional ML stage: installs heavy ML packages into a separate image
# Usage: docker build --target ml -t ai-trading-agent-ml:latest .
# -----------------------
FROM deps-builder AS ml-builder
COPY requirements-ml.txt ./requirements-ml.txt
RUN pip install --no-cache-dir -r requirements-ml.txt || echo "ML packages installation failed; build may require more resources"


# -----------------------
# Final runtime image
# -----------------------
FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system deps for runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from deps-builder
COPY --from=deps-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps-builder /usr/local/bin /usr/local/bin

# Copy only application files required at runtime
# Exclude tests, docs, local configs via .dockerignore
COPY . .

# Copy built frontend assets if present
COPY --from=frontend-builder /build/frontend/.next ./frontend/.next
COPY --from=frontend-builder /build/frontend/package.json ./frontend/package.json

# Create runtime dirs
RUN mkdir -p /app/data /app/logs

# Add start script and make executable if present
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh || true

ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Expose ports used by the app
EXPOSE 8080

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -fsS --max-time 5 http://localhost:8080/health || exit 1

# Default command
CMD ["/usr/local/bin/start.sh"]
