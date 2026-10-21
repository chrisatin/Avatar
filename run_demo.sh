#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

GL_LIB="${GL_PREFIX:-$HOME/.local/gl-libs}/usr/lib/x86_64-linux-gnu"
if [ -d "$GL_LIB" ]; then
  export LD_LIBRARY_PATH="$GL_LIB:${LD_LIBRARY_PATH:-}"
fi

if [ ! -f assets/models/face_landmarker.task ]; then
  echo "[run_demo] face_landmarker.task missing — downloading…"
  python scripts/download_models.py || true
fi

echo "[run_demo] starting server on http://${HOST}:${PORT}"
echo "[run_demo] SSH tunnel from your client machine:"
echo "           ssh -N -L ${PORT}:localhost:${PORT} user@t4-server"
exec python -m uvicorn server.app:app --host "$HOST" --port "$PORT"
