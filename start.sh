#!/usr/bin/env bash
# Entrypoint for the backend image.
#
# This script used to also launch the Next.js dashboard with `npx next start`.
# The final image is python:3.11-slim and has no Node runtime, so that call
# failed with "npx: not found". Because it ran in a backgrounded subshell the
# failure was invisible: the backend came up fine and the dashboard simply
# never existed. The dashboard now has its own image (Dockerfile.frontend).

set -euo pipefail

DATA_DIR="${DATA_DIR:-/app/data}"

# The journals are the run's only durable record. An unwritable data directory
# means the agent trades and remembers nothing, so stop here with a cause
# rather than let it discover this per-write, thousands of times, in the log.
if ! mkdir -p "$DATA_DIR" 2>/dev/null || ! touch "$DATA_DIR/.write_probe" 2>/dev/null; then
  cat >&2 <<MSG
FATAL: $DATA_DIR is not writable by this container (uid $(id -u)).

The order and decision journals live here and they are the only record of
what the agent did. Refusing to start rather than trade without a record.

Most likely cause: a host directory was bind-mounted here and is owned by
root. Either use a named Docker volume, which keeps the image's ownership,
or run: chown -R $(id -u):$(id -g) <host-path>
MSG
  exit 1
fi
rm -f "$DATA_DIR/.write_probe"

echo "Data directory $DATA_DIR is writable."
echo "Starting the trading agent (src.agent.main)..."
exec python3 -m src.agent.main "$@"
