"""Equal-budget validation-only LR search; final test is never read."""
from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np
import yaml

from .utils import ROOT, read_json, save_json


def tune(base, grid_path, destination, experiment_name=None, resume=False):
    from .cli import experiment_config, logged_train
    from .engine import evaluate
    spec = yaml.safe_load(Path(grid_path).read_text())
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    settings = list(itertools.product(spec["learning_rates"], spec.get("weight_decays", [base["training"]["weight_decay"]])))
    selected = copy.deepcopy(base)
    results = []
    for experiment in selected["experiments"]:
        if experiment_name and experiment["name"] != experiment_name:
            continue
        scores = []
        for trial, (lr, decay) in enumerate(settings):
            evaluations = []
            for seed in spec["seeds"]:
                candidate = copy.deepcopy(experiment)
                candidate["name"] += f"_tuning{trial}"
                candidate.setdefault("training_overrides", {}).update(lr=lr, weight_decay=decay,
                                                   epochs=spec["epochs"], patience=spec["patience"])
                config = experiment_config(base, candidate, seed)
                config["output_root"] = str(destination)
                run = destination/f"{candidate['name']}_seed{seed}"
                logged_train(config, run, resume)
                if not (run/"eval/val_metrics.json").exists():
                    evaluate(run, split="val")
                metrics = read_json(run/"eval/val_metrics.json")
                evaluations.append(metrics["AP"])
            score = float(np.mean(evaluations))
            row = {"experiment": experiment["name"], "trial": trial, "lr": lr, "weight_decay": decay,
                   "mean_validation_AP": score, "seeds": spec["seeds"], "scores": evaluations,
                   "epochs_budget": spec["epochs"], "test_accessed": False}
            scores.append(row)
            results.append(row)
        winner = max(scores, key=lambda r: r["mean_validation_AP"])
        experiment.setdefault("training_overrides", {}).update(lr=winner["lr"], weight_decay=winner["weight_decay"])
    # Preserve the full-duration final training budget; only chosen LR/decay change.
    selected["output_root"] = str((ROOT/"runs/tuned_paper_study").resolve())
    (destination/"selected_config.yaml").write_text(yaml.safe_dump(selected, sort_keys=False))
    save_json(destination/"tuning_results.json", results)
    save_json(destination/"tuning_protocol.json", {"grid": spec, "selection_metric": "mean validation COCO AP50:95",
                                                "test_accessed": False, "ties": "first prespecified grid entry"})
    print(f"Selected configuration: {destination/'selected_config.yaml'}")
