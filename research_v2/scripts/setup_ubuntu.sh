#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Matched, pinned release pair. Use TORCH_INDEX_URL for a different supported CUDA wheel.
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip install -r requirements.txt
python -m pip install -r requirements-modern.txt
python run.py doctor --device cuda
python -m pip freeze > environment.lock.txt
python -m pytest -q
