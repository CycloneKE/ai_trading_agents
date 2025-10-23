#!/usr/bin/env bash
# start.sh - Launch backend (python main.py) and frontend (Next.js) together in one container.
# Backend listens on port 8080 by default (monitoring/api), frontend uses 3001 per package.json.

set -e

# Start frontend
if [ -d "./frontend" ]; then
  echo "Starting Next.js frontend on port 3001..."
  (cd frontend && npx next start -p 3001) &
else
  echo "No frontend directory found, skipping frontend start"
fi

# Run backend in foreground
echo "Starting Python backend (main.py)..."
exec python3 main.py
