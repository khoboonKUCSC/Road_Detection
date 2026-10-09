from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

from .utils import ROOT, read_json, save_json


def config_load(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    for key in ("data_root", "manifest", "output_root"):
        p = Path(cfg[key])
        cfg[key] = str(p.resolve() if p.is_absolute() else (ROOT / p).resolve())
    if cfg["training"]["epochs"] <= 0 or cfg["training"]["val_every"] <= 0 or cfg["training"]["save_every"] <= 0:
        raise ValueError("epochs, val_every and save_every must be positive")
    if cfg["training"]["batch_size"] <= 0 or cfg["training"]["patience"] <= 0:
        raise ValueError("batch_size and patience must be positive")
    return cfg


def experiment_config(base, experiment, seed):
    cfg = copy.deepcopy(base)
    cfg.pop("experiments")
    cfg.pop("seeds")
    cfg["experiment"] = experiment["name"]
    cfg["seed"] = seed
    cfg["model"].update({k: v for k, v in experiment.items() if k not in {"name", "training_overrides"}})
    cfg["training"].update(experiment.get("training_overrides", {}))
    return cfg


class Tee:
    def __init__(self, original, stream):
        self.original, self.stream = original, stream

    def write(self, value):
        self.original.write(value)
        self.stream.write(value)
        self.stream.flush()

    def flush(self):
        self.original.flush()
        self.stream.flush()


def logged_train(cfg, directory, resume):
    import traceback
    from .engine import train
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "console.log", "a") as stream:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, stream), Tee(stderr, stream)
        try:
            return train(cfg, directory, resume)
        except Exception as exc:
            traceback.print_exc()
            save_json(directory / "failure.json", {"error": str(exc), "traceback": traceback.format_exc()})
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent road damage research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check environment and detection operators")
    doctor.add_argument("--device", default="cuda")
    prepare = commands.add_parser("prepare", help="Audit originals and create a frozen group split")
    prepare.add_argument("--data-root", default=str(ROOT.parent / "data"))
    prepare.add_argument("--output", default=str(ROOT / "artifacts/dataset"))
    prepare.add_argument("--groups-csv")
    prepare.add_argument("--seed", type=int, default=2026)
    prepare.add_argument("--external", action="store_true")
    prepare.add_argument("--source", help="Dataset source URL/DOI, especially for external data")
    prepare.add_argument("--reviewed-links")
    prepare.add_argument("--verified-metadata", action="store_true")
    for name in ("train", "run", "develop"):
        command = commands.add_parser(name)
        command.add_argument("--config", default=str(ROOT / "configs/rtx4090.yaml"))
        command.add_argument("--experiment", help="Only this configured experiment")
        command.add_argument("--seed", type=int, help="Only this seed")
        command.add_argument("--resume", action="store_true")
    ev = commands.add_parser("evaluate")
    ev.add_argument("--run-dir", required=True)
    ev.add_argument("--split", choices=["val", "test", "external"], default="test")
    ev.add_argument("--data-root")
    ev.add_argument("--external-manifest")
    ev.add_argument("--tiled", action="store_true", default=None)
    ev.add_argument("--protocol")
    sim = commands.add_parser("audit-similarity")
    sim.add_argument("--manifest", default=str(ROOT/"artifacts/dataset/manifest.json"))
    sim.add_argument("--data-root", default=str(ROOT.parent/"data"))
    sim.add_argument("--output", default=str(ROOT/"artifacts/leakage"))
    freeze = commands.add_parser("freeze-study")
    freeze.add_argument("--config", default=str(ROOT/"configs/rtx4090.yaml"))
    freeze.add_argument("--output", default=str(ROOT/"artifacts/study/protocol.json"))
    freeze.add_argument("--similarity-audit", default=str(ROOT/"artifacts/leakage/similarity_audit.json"))
    freeze.add_argument("--decisions-csv")
    freeze.add_argument("--external-manifest")
    freeze.add_argument("--exploratory", action="store_true")
    final = commands.add_parser("finalize-study")
    final.add_argument("--protocol", default=str(ROOT/"artifacts/study/protocol.json"))
    final.add_argument("--data-root")
    final.add_argument("--external-data-root")
    rdd = commands.add_parser("import-rdd2022")
    rdd.add_argument("--root", required=True)
    rdd.add_argument("--output", required=True)
    rdd.add_argument("--subset", default="train", choices=["train", "test"])
    tuner = commands.add_parser("tune")
    tuner.add_argument("--config", default=str(ROOT/"configs/rtx4090.yaml"))
    tuner.add_argument("--grid", default=str(ROOT/"configs/tuning.yaml"))
    tuner.add_argument("--destination", default=str(ROOT/"runs/tuning"))
    tuner.add_argument("--experiment")
    tuner.add_argument("--resume", action="store_true")
    rep = commands.add_parser("report")
    rep.add_argument("--output-root", default=str(ROOT / "runs"))
    rep.add_argument("--destination", default=str(ROOT / "artifacts/paper"))
    rep.add_argument("--split", default="test", choices=["val", "test", "external"])
    rep.add_argument("--tiled", action="store_true")
    boot = commands.add_parser("bootstrap")
    boot.add_argument("--eval-dir", required=True, help="run/eval or run/eval_tiled")
    boot.add_argument("--compare-eval-dir", help="Optional B for paired AP_B - AP_A")
    boot.add_argument("--split", default="test", choices=["test", "external"])
    boot.add_argument("--repeats", type=int, default=1000)
    boot.add_argument("--seed", type=int, default=2026)
    pred = commands.add_parser("predict")
    pred.add_argument("--run-dir", required=True)
    pred.add_argument("--image", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--device", default="cuda")
    pred.add_argument("--tiled", action="store_true", default=None)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        import torch
        from torchvision.ops import nms
        from .utils import environment, resolve_device
        device = resolve_device(args.device)
        nms(torch.tensor([[0., 0., 2., 2.]], device=device), torch.tensor([0.5], device=device), 0.5)
        import pycocotools
        print(environment())
        print(f"Detection ops and COCO evaluator OK on {device}")
    elif args.command == "prepare":
        from .data import prepare
        prepare(args.data_root, args.output, args.groups_csv, args.seed, external=args.external, source=args.source,
                reviewed_links=args.reviewed_links, verified_metadata=args.verified_metadata)
    elif args.command in ("run", "train", "develop"):
        from .engine import evaluate
        from .report import report
        from .utils import resolve_device
        base = config_load(args.config)
        if args.command == "run" and base.get("study", {}).get("require_registration", False):
            raise ValueError("Use develop → freeze-study → finalize-study for the registered paper protocol")
        resolve_device(base["device"])
        if not Path(base["manifest"]).exists():
            raise FileNotFoundError("Run prepare first and review artifacts/dataset/audit.json before training")
        experiments = [e for e in base["experiments"] if not args.experiment or e["name"] == args.experiment]
        seeds = [args.seed] if args.seed is not None else base["seeds"]
        if not experiments:
            raise ValueError("Experiment name not found in config")
        runs = []
        for experiment in experiments:
            for seed in seeds:
                cfg = experiment_config(base, experiment, seed)
                directory = Path(base["output_root"]) / f"{experiment['name']}_seed{seed}"
                logged_train(cfg, directory, args.resume)
                runs.append(directory)
        if args.command in ("run", "develop"):
            # Finish all training before executing the predefined final test comparisons.
            for directory in runs:
                if not (directory / "eval/thresholds.json").exists():
                    evaluate(directory, split="val")
            if args.command == "run":
                for directory in runs:
                    if not (directory / "eval/test_metrics.json").exists():
                        evaluate(directory, split="test")
            report(base["output_root"], Path(base["output_root"]) / ("development_report" if args.command == "develop" else "paper"),
                   split="val" if args.command == "develop" else "test")
    elif args.command == "evaluate":
        from .engine import evaluate
        if (args.split == "external") != bool(args.external_manifest):
            raise ValueError("external split requires --external-manifest; internal splits must not use one")
        evaluate(args.run_dir, args.data_root, args.split, args.external_manifest, args.tiled, args.protocol)
    elif args.command == "audit-similarity":
        from .leakage import audit_similarity
        audit_similarity(args.manifest, args.data_root, args.output)
    elif args.command == "freeze-study":
        from .protocol import freeze_study
        freeze_study(config_load(args.config), args.output, args.similarity_audit, args.decisions_csv,
                     args.external_manifest, args.exploratory)
    elif args.command == "finalize-study":
        from .protocol import finalize_study
        finalize_study(args.protocol, args.data_root, args.external_data_root)
    elif args.command == "import-rdd2022":
        from .external import import_rdd2022
        import_rdd2022(args.root, args.output, args.subset)
    elif args.command == "tune":
        from .tuning import tune
        tune(config_load(args.config), args.grid, args.destination, args.experiment, args.resume)
    elif args.command == "report":
        from .report import report
        report(args.output_root, args.destination, args.split, args.tiled)
    elif args.command == "bootstrap":
        from .metrics import group_bootstrap
        directory = Path(args.eval_dir)
        records = read_json(directory / f"{args.split}_records.json")
        predictions = read_json(directory / f"{args.split}_predictions.json")
        predictions_b = None
        if args.compare_eval_dir:
            other = Path(args.compare_eval_dir)
            if read_json(other / f"{args.split}_records.json") != records:
                raise ValueError("Paired bootstrap needs exactly the same evaluation records")
            predictions_b = read_json(other / f"{args.split}_predictions.json")
        result = group_bootstrap(records, predictions, predictions_b, args.repeats, args.seed)
        save_json(directory / (f"bootstrap_vs_{Path(args.compare_eval_dir).parent.name}.json" if predictions_b is not None
                               else "bootstrap_AP.json"), result)
        print({k: v for k, v in result.items() if k != "samples"})
    elif args.command == "predict":
        predict_image(args)


def predict_image(args):
    import torch
    from PIL import Image, ImageDraw
    from torchvision.transforms.functional import pil_to_tensor
    from .engine import predict_tensor
    from .models import build_model
    from .utils import CLASSES, digest, load_checkpoint, resolve_device
    run = Path(args.run_dir)
    cfg = read_json(run / "config.json")
    cfg["device"] = args.device
    if args.tiled is not None:
        cfg["evaluation"]["tiling"] = args.tiled
    protocol = "eval_tiled" if cfg["evaluation"]["tiling"] else "eval"
    selection = read_json(run / protocol / "thresholds.json")
    if selection["checkpoint_sha256"] != digest(run / "best.pt") or selection["evaluation_config"] != cfg["evaluation"]:
        raise ValueError("Recalibrate on validation for this checkpoint/protocol before prediction")
    device = resolve_device(cfg["device"])
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    checkpoint = load_checkpoint(run / "best.pt")
    model = build_model(cfg, pretrained=False, model_spec=checkpoint.get("model_spec")).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    with Image.open(args.image) as image:
        image = image.convert("RGB")
    with torch.inference_mode():
        result = predict_tensor(model, (pil_to_tensor(image).float()/255).to(device), cfg)
    rows = []
    draw = ImageDraw.Draw(image)
    for box, cls, score in zip(result["boxes"].tolist(), result["labels"].tolist(), result["scores"].tolist()):
        if cls not in (1, 2, 3):
            continue
        name = CLASSES[cls-1]
        rows.append({"class": name, "score": score, "bbox_xyxy": box})
        if score >= selection["thresholds"][name]:
            draw.rectangle(box, outline="red", width=2)
            draw.text(tuple(box[:2]), f"{name} {score:.2f}", fill="red")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    image.save(output / "prediction.jpg")
    save_json(output / "predictions.json", {"image": str(Path(args.image).resolve()), "predictions": rows,
                                          "thresholds": selection["thresholds"], "checkpoint_sha256": selection["checkpoint_sha256"]})


if __name__ == "__main__":
    main()
