from __future__ import annotations

import csv
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms

from .data import RoadDataset, collate, verify_manifest
from .metrics import coco_evaluate, confusion_matrix, operating_metrics, select_thresholds
from .models import build_model
from .utils import (CLASSES, atomic_checkpoint, digest, environment, load_checkpoint,
                    read_json, resolve_device, save_json, seed_all, source_snapshot)


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, config, training=False):
    device = resolve_device(config["device"])
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(dataset, batch_size=config["training"]["batch_size"] if training else config["evaluation"]["batch_size"],
                      shuffle=training, num_workers=config["training"]["workers"], collate_fn=collate,
                      pin_memory=device.type == "cuda", worker_init_fn=worker_seed, generator=generator,
                      persistent_workers=False)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tile_starts(length, size, overlap):
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("tile_size must be positive; tile_overlap in [0,1)")
    if length <= size:
        return [0]
    step = max(1, round(size * (1-overlap)))
    return sorted(set([*range(0, length-size+1, step), length-size]))


def predict_tensor(model, image, config):
    ev = config["evaluation"]
    if not ev["tiling"]:
        return model([image])[0]
    _, height, width = image.shape
    size = ev["tile_size"]
    results = []
    if ev["include_full_image"]:
        results.append(model([image])[0])
    for y in tile_starts(height, size, ev["tile_overlap"]):
        for x in tile_starts(width, size, ev["tile_overlap"]):
            result = model([image[:, y:y+size, x:x+size]])[0]
            boxes = result["boxes"].clone()
            boxes += boxes.new_tensor([x, y, x, y])
            results.append({**result, "boxes": boxes})
    boxes = torch.cat([r["boxes"] for r in results])
    scores = torch.cat([r["scores"] for r in results])
    labels = torch.cat([r["labels"] for r in results])
    keep = batched_nms(boxes, scores, labels, ev["tile_nms"])[:100]
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}


@torch.inference_mode()
def predict_dataset(model, dataset, config):
    model.eval()
    device = resolve_device(config["device"])
    predictions, times = [], []
    # Warm up with validation/test input, with no parameter update or scoring decisions.
    for i in range(min(config["evaluation"]["warmup_images"], len(dataset))):
        image, _ = dataset[i]
        predict_tensor(model, image.to(device), config)
    sync(device)
    # End-to-end latency is batch=1: file decode, tensor conversion, device transfer,
    # model inference, optional tiling/NMS, and CPU materialization.
    # This intentionally does NOT use the training batch size or asynchronous timing.
    for i in range(len(dataset)):
        sync(device)
        start = time.perf_counter()
        image, target = dataset[i]
        result = predict_tensor(model, image.to(device), config)
        result = {k: v.detach().cpu() for k, v in result.items()}
        sync(device)
        times.append((time.perf_counter()-start)*1000)
        for box, score, cls in zip(result["boxes"].tolist(), result["scores"].tolist(), result["labels"].tolist()):
            if cls not in (1, 2, 3):
                continue
            x1, y1, x2, y2 = box
            predictions.append({"image_id": int(target["image_id"]), "category_id": cls,
                                "bbox": [x1, y1, x2-x1, y2-y1], "score": score})
    latency = {"batch_size": 1, "includes": "decode, preprocess, transfer, inference, tiling/NMS, CPU outputs",
               "warmup_images": min(config["evaluation"]["warmup_images"], len(dataset)),
               "images": len(times), "mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
               "p95_ms": float(np.percentile(times, 95)), "FPS_from_mean": 1000 / float(np.mean(times)),
               "per_image_ms": times, "device": str(device),
               "inference_precision": "FP32", "tiling": config["evaluation"]["tiling"]}
    return predictions, latency


def rng_state(train_loader):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": train_loader.generator.get_state()}


def restore_rng(state, train_loader):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    train_loader.generator.set_state(state["loader"])


def train(config, run_dir, resume=False):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config["manifest"])
    manifest = read_json(manifest_path)
    verify_manifest(manifest, config["data_root"])
    fingerprint = digest(manifest_path)
    checkpoint = None
    if (run_dir / "last.pt").exists():
        if not resume:
            raise FileExistsError(f"Run exists: {run_dir}. Use --resume or a NEW output_root.")
        checkpoint = load_checkpoint(run_dir / "last.pt")
        if checkpoint["manifest_sha256"] != fingerprint or checkpoint["config"] != config:
            raise ValueError("Resume requires the exact original config and manifest, including epochs and paths")
    elif resume and (run_dir / "config.json").exists():
        raise ValueError("Run metadata exists but last.pt is missing; use a NEW output directory")
    elif (run_dir / "config.json").exists():
        raise FileExistsError("Run directory already contains metadata; choose a NEW output directory")
    seed_all(config["seed"], config["training"]["deterministic"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = resolve_device(config["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config, pretrained=False if checkpoint else None,
                        model_spec=checkpoint.get("model_spec") if checkpoint else None).to(device)
    # Extra auxiliary-head initialization must not alter augmentation/RPN RNG start.
    seed_all(config["seed"], config["training"]["deterministic"])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"] and device.type == "cuda")
    train_ds = RoadDataset(manifest, config["data_root"], "train", True, config["training"]["limit_train"],
                           config["training"].get("crack_crop_probability", 0.), config["training"].get("crack_crop_size", 256))
    val_ds = RoadDataset(manifest, config["data_root"], "val", False, config["training"]["limit_eval"])
    train_loader = loader(train_ds, config, True)
    start_epoch, best, best_epoch, stale, history = 0, -1., -1, 0, []
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], train_loader)
        start_epoch, best, stale = checkpoint["epoch"]+1, checkpoint["best_AP"], checkpoint["stale"]
        best_epoch = checkpoint["best_epoch"]
        history = checkpoint["history"]
        # Existing source is retained; changed code is rejected below.
        from .utils import ROOT
        hashes = read_json(run_dir / "source_sha256.json")
        for rel, expected in hashes.items():
            if not (ROOT / rel).is_file() or digest(ROOT / rel) != expected:
                raise ValueError(f"Code changed since run started: {rel}. Create a NEW run.")
        # Recover interruption between atomic last.pt and best.pt writes.
        if best_epoch == checkpoint["epoch"]:
            if not (run_dir / "best.pt").exists() or load_checkpoint(run_dir / "best.pt")["epoch"] < best_epoch:
                atomic_checkpoint(run_dir / "best.pt", checkpoint)
        if not (run_dir / "best.pt").exists() and best_epoch >= 0:
            raise ValueError("Best checkpoint missing and cannot be recovered from last.pt")
    else:
        save_json(run_dir / "config.json", config)
        save_json(run_dir / "environment.json", environment())
        shutil.copy2(manifest_path, run_dir / "manifest.json")
        source_snapshot(run_dir)
        save_json(run_dir / "initialization.json", {
            "pretrained": config["model"]["pretrained"], "weights": "COCO_V1" if config["model"]["pretrained"] else None,
            "seed": config["seed"], "architecture": config["model"]["architecture"],
            "geometry_mode": config["model"]["geometry_mode"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })
    if checkpoint and checkpoint.get("finished"):
        finalize_training(run_dir, config, checkpoint, device)
        print(f"Already trained: {run_dir}")
        return run_dir
    previous_seconds = checkpoint.get("training_seconds", 0.) if checkpoint else 0.
    clock_start = time.perf_counter()
    for epoch in range(start_epoch, config["training"]["epochs"]):
        model.train()
        totals = {}
        steps = 0
        lr_used = optimizer.param_groups[0]["lr"]
        for images, targets in train_loader:
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=scaler.is_enabled()):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}: {losses}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            for key, value in {**losses, "loss_total": loss}.items():
                totals[key] = totals.get(key, 0.) + float(value.detach())
        scheduler.step()
        row = {"epoch": epoch+1, "lr": lr_used, **{key: value/steps for key, value in totals.items()}}
        do_val = (epoch+1) % config["training"]["val_every"] == 0 or epoch+1 == config["training"]["epochs"]
        is_best = False
        if do_val:
            predictions, _ = predict_dataset(model, val_ds, config)
            metrics, _, _ = coco_evaluate(val_ds.records, predictions)
            row.update({"val_AP": metrics["AP"], "val_AP50": metrics["AP50"]})
            if metrics["AP"] is None:
                raise ValueError("No valid validation AP")
            is_best = metrics["AP"] > best
            if is_best:
                best, stale = metrics["AP"], 0
                best_epoch = epoch
            else:
                stale += 1
        history.append(row)
        save_json(run_dir / "history.json", history)
        keys = sorted({k for item in history for k in item})
        with open(run_dir / "history.csv", "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
        finished = stale >= config["training"]["patience"] or epoch+1 == config["training"]["epochs"]
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "rng": rng_state(train_loader), "epoch": epoch,
                 "best_AP": best, "stale": stale, "history": history, "config": config,
                 "best_epoch": best_epoch,
                 "model_spec": model.network.config.to_dict() if hasattr(getattr(model, "network", None), "config") else None,
                 "manifest_sha256": fingerprint, "finished": finished,
                 "training_seconds": previous_seconds+time.perf_counter()-clock_start,
                 "peak_cuda_memory_GB": max(checkpoint.get("peak_cuda_memory_GB", 0.) or 0.,
                                            torch.cuda.max_memory_allocated(device)/1024**3) if device.type == "cuda" and checkpoint else
                                        (torch.cuda.max_memory_allocated(device)/1024**3 if device.type == "cuda" else None)}
        atomic_checkpoint(run_dir / "last.pt", state)
        if is_best:
            atomic_checkpoint(run_dir / "best.pt", state)
        if (epoch+1) % config["training"]["save_every"] == 0:
            atomic_checkpoint(run_dir / f"epoch_{epoch+1:03d}.pt", state)
        print(json.dumps({"run": run_dir.name, **row}), flush=True)
        if finished:
            break
    finalize_training(run_dir, config, state, device)
    return run_dir


def finalize_training(run_dir, config, state, device):
    save_json(run_dir / "training_summary.json", {
        "best_validation_AP": state["best_AP"], "epochs_completed": len(state["history"]), "selected_checkpoint": "best.pt",
        "selection_metric": "validation COCO bbox AP50:95", "training_seconds": state["training_seconds"],
        "peak_cuda_memory_GB": state["peak_cuda_memory_GB"],
        "smoke_only": config["model"]["architecture"] == "tiny_smoke" or config["training"]["limit_train"] is not None,
    })
    selected = load_checkpoint(run_dir / "best.pt")
    atomic_checkpoint(run_dir / "best_weights.pt", {"model": selected["model"], "config": config,
                                                   "model_spec": selected.get("model_spec"),
                                                   "epoch": selected["epoch"], "manifest_sha256": state["manifest_sha256"]})


def visualizations(dataset, predictions, thresholds, output, count):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from collections import defaultdict
    lookup = defaultdict(list)
    for p in predictions:
        lookup[p["image_id"]].append(p)
    # Sort by total FP+FN so difficult examples are retained, plus a uniform sample.
    diagnostics = []
    for record in dataset.records:
        item = operating_metrics([record], lookup[record["id"]], thresholds)
        errors = sum(v["FP"]+v["FN"] for v in item["per_class"].values())
        diagnostics.append({"image_id": record["id"], "file_name": record["file_name"], "errors": errors,
                            "metrics": item})
    save_json(output.parent / (output.name.replace("_examples", "") + "_per_image_errors.json"), diagnostics)
    hardest = sorted(diagnostics, key=lambda r: (-r["errors"], r["image_id"]))[:count//2]
    uniform = dataset.records[::max(1, len(dataset)//max(count-count//2, 1))][:count-count//2]
    ids = {r["image_id"] for r in hardest} | {r["id"] for r in uniform}
    for record in dataset.records:
        if record["id"] not in ids:
            continue
        with Image.open(dataset.root / record["image_path"]) as im:
            im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        for a in record["annotations"]:
            draw.rectangle(a["bbox_xyxy"], outline="lime", width=2)
            draw.text(tuple(a["bbox_xyxy"][:2]), "GT " + CLASSES[a["category_id"]-1], fill="lime")
        for p in lookup[record["id"]]:
            name = CLASSES[p["category_id"]-1]
            if p["score"] < thresholds[name]:
                continue
            x, y, w, h = p["bbox"]
            draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
            draw.text((x, max(0, y-12)), f"{name} {p['score']:.2f}", fill="red")
        im.save(output / f"{record['id']:05d}_{Path(record['file_name']).stem}.jpg")


def evaluate(run_dir, data_root=None, split="test", external_manifest=None, tiling=None, protocol_path=None):
    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json")
    if data_root:
        config["data_root"] = str(Path(data_root).resolve())
    manifest_path = Path(external_manifest) if external_manifest else run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    verify_manifest(manifest, config["data_root"])
    if split != "val":
        from .protocol import require_registered
        require_registered(run_dir, config, manifest_path, protocol_path)
        if config.get("study", {}).get("require_registration", False) and (tiling or config["evaluation"]["tiling"]):
            raise ValueError("Registered study supports full-image final evaluation only; do not change inference after freezing")
    checkpoint = load_checkpoint(run_dir / "best.pt")
    if external_manifest is None and digest(manifest_path) != checkpoint["manifest_sha256"]:
        raise ValueError("Checkpoint and manifest disagree")
    if tiling is not None:
        config["evaluation"]["tiling"] = tiling
    output = run_dir / ("eval_tiled" if config["evaluation"]["tiling"] else "eval")
    output.mkdir(exist_ok=True)
    if (output/f"{split}_metrics.json").exists():
        previous = read_json(output/f"{split}_metrics.json")
        if previous["manifest_sha256"] != digest(manifest_path):
            raise ValueError("Different evaluation dataset would overwrite saved results; use a separate study/run directory")
    device = resolve_device(config["device"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    model = build_model(config, pretrained=False, model_spec=checkpoint.get("model_spec")).to(device)
    model.load_state_dict(checkpoint["model"])
    threshold_path = output / "thresholds.json"
    if split == "val":
        # Recalibration after final test would change the study protocol.
        if (output / "test_metrics.json").exists():
            raise ValueError("Test already evaluated for this protocol; reuse frozen thresholds or start a NEW protocol.")
        ds = RoadDataset(manifest, config["data_root"], "val", False, config["training"]["limit_eval"])
        predictions, latency = predict_dataset(model, ds, config)
        thresholds, sweep = select_thresholds(ds.records, predictions)
        save_json(threshold_path, {"selected_on": "validation", "checkpoint_sha256": digest(run_dir / "best.pt"),
                                   "manifest_sha256": digest(manifest_path), "thresholds": thresholds,
                                   "evaluation_config": config["evaluation"], "sweep": sweep})
    else:
        if not threshold_path.exists():
            raise ValueError("Evaluate validation first to freeze thresholds for this full-image/tiled protocol")
        selection = read_json(threshold_path)
        if selection["checkpoint_sha256"] != digest(run_dir / "best.pt") or selection["evaluation_config"] != config["evaluation"]:
            raise ValueError("Thresholds were calibrated for a different checkpoint or inference protocol")
        thresholds = selection["thresholds"]
        ds = RoadDataset(manifest, config["data_root"], split, False, config["training"]["limit_eval"])
        predictions, latency = predict_dataset(model, ds, config)
    metrics, curves, summary = coco_evaluate(ds.records, predictions)
    metrics.update({"split": split, "checkpoint_epoch": checkpoint["epoch"]+1,
                    "protocol_sha256": digest(protocol_path) if protocol_path else None,
                    "checkpoint_sha256": digest(run_dir / "best.pt"), "manifest_sha256": digest(manifest_path),
                    "operating_point": operating_metrics(ds.records, predictions, thresholds),
                    "confusion_matrix": confusion_matrix(ds.records, predictions, thresholds),
                    "confusion_labels": CLASSES + ["background"], "latency": latency,
                    "evaluation_config": config["evaluation"],
                    "smoke_only": config["model"]["architecture"] == "tiny_smoke" or config["training"]["limit_eval"] is not None,
                    "evaluation_environment": environment()})
    save_json(output / f"{split}_predictions.json", predictions)
    save_json(output / f"{split}_records.json", ds.records)
    save_json(output / f"{split}_metrics.json", metrics)
    save_json(output / f"{split}_pr_curves.json", curves)
    (output / f"{split}_coco_summary.txt").write_text(summary)
    visualizations(ds, predictions, thresholds, output / f"{split}_examples", config["evaluation"]["visualizations"])
    from .diagnostics import failure_analysis
    failure_analysis(ds, predictions, thresholds, output/f"{split}_failure_analysis.json")
    print(f"{run_dir.name} {split}: AP={metrics['AP']:.4f}, AP50={metrics['AP50']:.4f}", flush=True)
    return metrics
