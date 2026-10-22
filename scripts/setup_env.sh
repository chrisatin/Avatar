#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Some environments (e.g. uv-created venvs) ship without pip.
python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
bash scripts/setup_headless_gl.sh || true
python scripts/download_models.py || true

echo
echo "Environment ready. Launch with:"
echo "  source $VENV_DIR/bin/activate && ./run_demo.sh"
