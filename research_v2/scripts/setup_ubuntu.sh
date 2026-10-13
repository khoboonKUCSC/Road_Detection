#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if [ ! -d ".venv" ]; then
  PYTHON_CMD=""
  for cand in python3.12 python3.11 /home/ds/anaconda3/envs/py312/bin/python3 python3; do
    if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then
      if "$cand" -c 'import sys; sys.exit(0 if sys.version_info < (3, 13) else 1)' >/dev/null 2>&1; then
        PYTHON_CMD="$cand"
        break
      fi
    fi
  done
  "${PYTHON_CMD:-python3}" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
# Matched, pinned release pair. Use TORCH_INDEX_URL for a different supported CUDA wheel.
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip install -r requirements.txt
python -m pip install -r requirements-modern.txt
python run.py doctor --device cuda
python -m pip freeze > environment.lock.txt
python -m pytest -q
