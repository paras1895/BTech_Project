#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "Missing Python dependencies. Run: python3 -m pip install -r backend/requirements.txt" >&2
  exit 1
fi
exec python3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port "${API_PORT:-8000}"
