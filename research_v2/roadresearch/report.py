from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import CLASSES, read_json, save_json


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def report(output_root, destination, split="test", tiled=False):
    output_root, destination = Path(output_root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    protocol = "eval_tiled" if tiled else "eval"
    rows, class_rows, paths = [], [], []
    for metrics_path in sorted(output_root.glob(f"*/{protocol}/{split}_metrics.json")):
        run = metrics_path.parent.parent
        cfg, metrics = read_json(run / "config.json"), read_json(metrics_path)
        row = {"experiment": cfg["experiment"], "seed": cfg["seed"], "split": split,
               "smoke_only": metrics["smoke_only"], "AP": metrics["AP"], "AP50": metrics["AP50"],
               "AP75": metrics["AP75"], "AP_shared_damage": metrics.get("AP_shared_damage"), "AP_small": metrics["AP_small"],
               "AP_medium": metrics["AP_medium"], "AP_large": metrics["AP_large"],
               "macro_F1": metrics["operating_point"]["macro_F1"], "latency_ms": metrics["latency"]["mean_ms"],
               "parameters": read_json(run / "initialization.json")["parameters"],
               "best_epoch": metrics["checkpoint_epoch"], "manifest_sha256": metrics["manifest_sha256"],
               "run": run.name}
        rows.append(row)
        paths.append(str(metrics_path))
        for name in CLASSES:
            class_rows.append({"experiment": cfg["experiment"], "seed": cfg["seed"], "class": name,
                               **metrics["per_class"][name], **metrics["operating_point"]["per_class"][name]})
        history = read_json(run / "history.json")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].plot([h["epoch"] for h in history], [h["loss_total"] for h in history])
        axes[0].set(xlabel="Epoch", ylabel="Training loss")
        vals = [h for h in history if "val_AP" in h]
        axes[1].plot([h["epoch"] for h in vals], [h["val_AP"] for h in vals], label="AP50:95")
        axes[1].plot([h["epoch"] for h in vals], [h["val_AP50"] for h in vals], label="AP50")
        axes[1].set(xlabel="Epoch", ylabel="Validation AP", ylim=(0, 1))
        axes[1].legend()
        save_figure(fig, destination / f"{run.name}_training")
        curves = read_json(metrics_path.parent / f"{split}_pr_curves.json")
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, curve in curves.items():
            ax.plot(curve["recall"], [np.nan if p is None else p for p in curve["precision50"]], label=name)
        ax.set(xlabel="Recall", ylabel="Interpolated precision (IoU=0.5)", xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        save_figure(fig, destination / f"{run.name}_{split}_PR")
        matrix = np.asarray(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(matrix, cmap="Blues")
        labels = metrics["confusion_labels"]
        ax.set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels,
               xlabel="Predicted", ylabel="Ground truth")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        save_figure(fig, destination / f"{run.name}_{split}_confusion")
    if not rows:
        raise ValueError(f"No {split} results under {output_root}")
    if len({r["manifest_sha256"] for r in rows}) != 1:
        raise ValueError("Cannot aggregate experiments evaluated on different manifests")
    if len({r["smoke_only"] for r in rows}) != 1:
        raise ValueError("Smoke and full experiments must be reported separately")
    write_csv(destination / "per_run.csv", rows)
    write_csv(destination / "per_class.csv", class_rows)
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["experiment"]].append(r)
    aggregated = []
    for name, runs in grouped.items():
        row = {"experiment": name, "n_seeds": len(runs), "seeds": ",".join(str(r["seed"]) for r in runs)}
        for key in ["AP", "AP50", "AP75", "AP_shared_damage", "macro_F1", "latency_ms"]:
            values = [r[key] for r in runs if r[key] is not None]
            row[key+"_mean"] = float(np.mean(values)) if values else None
            row[key+"_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        aggregated.append(row)
    write_csv(destination / "summary_mean_std.csv", aggregated)
    class_groups = defaultdict(list)
    for row in class_rows:
        class_groups[(row["experiment"], row["class"])].append(row)
    class_summary = []
    for (name, cls), items in class_groups.items():
        row = {"experiment": name, "class": cls, "n_seeds": len(items)}
        for key in ("AP", "AP50", "precision", "recall", "F1", "FP_per_image"):
            vals = [r[key] for r in items if r[key] is not None]
            row[key+"_mean"] = float(np.mean(vals)) if vals else None
            row[key+"_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        class_summary.append(row)
    write_csv(destination / "per_class_mean_std.csv", class_summary)
    baselines = {r["seed"]: r for r in rows if r["experiment"] == "fasterrcnn_baseline"}
    deltas = []
    for row in rows:
        if row["seed"] in baselines and row["experiment"].startswith("fasterrcnn_") and row["experiment"] != "fasterrcnn_baseline":
            baseline = baselines[row["seed"]]
            deltas.append({"experiment": row["experiment"], "seed": row["seed"],
                           "delta_AP": row["AP"]-baseline["AP"], "delta_AP50": row["AP50"]-baseline["AP50"],
                           "delta_macro_F1": row["macro_F1"]-baseline["macro_F1"]})
    write_csv(destination / "paired_seed_deltas.csv", deltas)
    manifest = read_json(Path(paths[0]).parent.parent / "manifest.json")
    distribution = []
    for subset in sorted({r["split"] for r in manifest["records"]}):
        recs = [r for r in manifest["records"] if r["split"] == subset]
        distribution.append({"split": subset, "images": len(recs), "groups": len({r["group"] for r in recs}),
                             **{name: sum(a["category_id"] == k+1 for r in recs for a in r["annotations"])
                                for k, name in enumerate(CLASSES)}})
    write_csv(destination / "source_dataset_distribution.csv", distribution)
    save_json(destination / "report_provenance.json", {"source_metrics": paths, "split": split,
                                                       "tiled": tiled, "smoke_only": rows[0]["smoke_only"],
                                                       "manifest_sha256": rows[0]["manifest_sha256"]})
    fig, ax = plt.subplots(figsize=(8, 4))
    means = [r["AP_mean"] for r in aggregated]
    stds = [r["AP_std"] or 0 for r in aggregated]
    ax.bar(range(len(aggregated)), means, yerr=stds, capsize=4)
    ax.set(xticks=range(len(aggregated)), xticklabels=[r["experiment"] for r in aggregated],
           ylabel="COCO bbox AP50:95", ylim=(0, 1))
    ax.tick_params(axis="x", labelrotation=15)
    save_figure(fig, destination / "comparison_AP")
    fig, ax = plt.subplots(figsize=(7, 4))
    for row in aggregated:
        ax.scatter(row["latency_ms_mean"], row["AP_mean"])
        ax.annotate(row["experiment"], (row["latency_ms_mean"], row["AP_mean"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Mean end-to-end latency (ms/image, batch=1)", ylabel="COCO bbox AP50:95", ylim=(0, 1))
    save_figure(fig, destination / "accuracy_latency")
    lines = ["% Automatically generated from measured results. AP values are percentages.",
             r"\begin{tabular}{lrrr}", r"\hline", r"Method & Seeds & AP50:95 & AP50 \\", r"\hline"]
    for r in aggregated:
        def value(key):
            mean, std = r[key+"_mean"], r[key+"_std"]
            return f"{100*mean:.2f}" + (f" $\\pm$ {100*std:.2f}" if std is not None else "")
        label = r["experiment"].replace("_", r"\_")
        lines.append(f"{label} & {r['n_seeds']} & {value('AP')} & {value('AP50')} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    (destination / "paper_table.tex").write_text("\n".join(lines)+"\n")
    note = "SMOKE TEST ONLY: these scores are not research results.\n\n" if rows[0]["smoke_only"] else ""
    note += ("Measured results are in per_run.csv, per_class.csv and summary_mean_std.csv.\n"
             "Standard deviation is across seeds, not across images; one seed has no estimated standard deviation.\n"
             "AP uses official COCO bbox evaluation. P/R/F1 use class-specific thresholds selected on validation.\n"
             "Class-independent confusion matrix matching can differ from class-specific P/R/F1 matching.\n"
             "Filename-date grouping is a proxy; verify capture sessions and repeated routes before paper claims.\n"
             "Do not claim novelty, superiority or cross-device/country generalization from this table alone.\n")
    (destination / "README.txt").write_text(note)
    print(f"Report written: {destination}")
