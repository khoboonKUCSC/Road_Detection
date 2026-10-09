from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["pothole", "crack", "manhole"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def seed_all(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def resolve_device(request):
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    if request.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install CUDA-enabled PyTorch and check nvidia-smi.")
    return torch.device(request)


def environment():
    import torchvision
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    return {
        "python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "command": sys.argv, "pip_freeze": freeze.stdout.splitlines(),
        "determinism": "Seeded; deterministic algorithms requested with warn_only=True. Review console warnings.",
    }


def source_snapshot(destination):
    import zipfile
    destination = Path(destination)
    hashes = {}
    with zipfile.ZipFile(destination / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or any(x in path.relative_to(ROOT).parts for x in
                                        ("runs", "artifacts", "__pycache__", ".venv", ".pytest_cache")):
                continue
            if path.suffix not in {".py", ".yaml", ".json", ".md", ".txt", ".sh"}:
                continue
            rel = path.relative_to(ROOT).as_posix()
            archive.write(path, rel)
            hashes[rel] = digest(path)
    save_json(destination / "source_sha256.json", hashes)
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def atomic_checkpoint(path, state):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path):
    # Training checkpoints contain optimizer and Python/NumPy RNG states.
    # Only load checkpoints created by this pipeline or another trusted source.
    return torch.load(path, map_location="cpu", weights_only=False)
