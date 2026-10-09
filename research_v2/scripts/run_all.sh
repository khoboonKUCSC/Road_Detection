#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
source .venv/bin/activate
python run.py doctor --device cuda
# Preserve the reviewed split; never recreate it automatically on subsequent runs.
if [[ ! -f artifacts/dataset/manifest.json ]]; then
  python run.py prepare
fi
python run.py audit-similarity
python run.py develop --config configs/rtx4090.yaml --resume "$@"
printf '%s\n' 'Development complete. Review validation and leakage evidence, then freeze-study before finalize-study.'
