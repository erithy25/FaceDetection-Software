#!/usr/bin/env bash
# SilentWitness Python Backend Sidecar Launcher
# This script is called by Tauri's sidecar system to start the Python backend.
# It finds the correct Python interpreter and launches main.py.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_DIR="$PROJECT_ROOT/python"

# Find Python interpreter (prefer venv, then system)
if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif [ -f "$PROJECT_ROOT/venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "ERROR: Python not found. Please install Python 3.11+." >&2
    exit 1
fi

echo "Using Python: $PYTHON"
echo "Backend directory: $PYTHON_DIR"

# Launch the backend
exec "$PYTHON" "$PYTHON_DIR/main.py" "$@"
