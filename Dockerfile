## Backend image: the trading agent, its REST API and its monitoring endpoint.
##
## This image deliberately does NOT contain the dashboard. It used to: a
## node:20 stage built Next.js and the result was copied into a
## python:3.11-slim final stage that has no Node runtime at all. `npx next
## start` in that image fails with "not found", so the dashboard never ran,
## silently in the single-container path and as a restart loop in Compose.
## The dashboard now lives in Dockerfile.frontend, on a base that can run it.

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
COPY scripts/requirements-runtime.txt ./requirements-runtime.txt

# Upgrade pip and install runtime deps (cacheable layer)
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-runtime.txt


# -----------------------
# Optional ML stage: installs heavy ML packages into a separate image
# Usage: docker build --target ml -t ai-trading-agent-ml:latest .
# -----------------------
FROM deps-builder AS ml-builder
COPY scripts/requirements-ml.txt ./requirements-ml.txt
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

# Application code. .dockerignore keeps .env, the journals and .git out.
COPY . .

# Fail the build rather than ship an image that cannot start.
#
# This used to name five packages explicitly, which caught the auth stack and
# missed everything else. yfinance and scipy then shipped missing: the image
# built, CI passed, and the container crash-looped on the server with
# ModuleNotFoundError, three module-scope hops from the entry point.
#
# Importing the real entry points exercises the whole startup graph, so any
# missing dependency fails here, where it costs a build, instead of on a
# server at two in the morning. No SECRET_KEY is set during a build and that
# is fine: the API logs that its auth layer is locked down and still imports.
RUN python -c "import src.agent.main, src.api.api_server; print('startup import check OK')"

# Runtime dirs. /app/data MUST be a mounted volume in any real deployment:
# the SQLite journals are the run's only durable record and an unmounted
# path is destroyed on every redeploy.
RUN mkdir -p /app/data /app/logs
VOLUME ["/app/data"]

# Install the entrypoint while still root: /usr/local/bin is not writable by
# the unprivileged user, so the chmod has to happen before the USER switch.
COPY start.sh /usr/local/bin/start.sh
RUN chmod 0755 /usr/local/bin/start.sh

# Run as a non-root user.
RUN useradd --create-home --uid 10001 trader \
    && chown -R trader:trader /app
USER trader

ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# 8080 monitoring/health, 5001 REST API (config.json api.port)
EXPOSE 8080 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -fsS --max-time 5 http://localhost:8080/health || exit 1

CMD ["/usr/local/bin/start.sh"]
