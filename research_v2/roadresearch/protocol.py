"""Development/final-test separation with an immutable local preregistration."""
from __future__ import annotations

from pathlib import Path

from .utils import digest, read_json, save_json


def freeze_study(config, output, similarity_audit, decisions_csv=None, external_manifest=None, exploratory=False):
    import csv
    from .cli import experiment_config
    output = Path(output)
    if output.exists():
        raise FileExistsError("Study protocol exists. Use it or a NEW study directory; never revise after test.")
    manifest = read_json(config["manifest"])
    if config["evaluation"]["tiling"]:
        raise ValueError("This registered protocol supports full-image evaluation only")
    audit = read_json(similarity_audit)
    if audit["manifest_sha256"] != digest(config["manifest"]):
        raise ValueError("Similarity audit belongs to a different manifest")
    reviewed = {}
    if decisions_csv:
        with open(decisions_csv, newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("decision") in {"same_session", "different_scene"}:
                    if not row.get("reviewer") or not row.get("evidence"):
                        raise ValueError("Review needs a reviewer and documented evidence")
                    reviewed[tuple(sorted([row["file_a"], row["file_b"]]))] = row
    unresolved, same_session_cross = [], []
    for pair in audit["pairs"]:
        if not pair["cross_split"]:
            continue
        review = reviewed.get(tuple(sorted([pair["file_a"], pair["file_b"]])))
        if not review:
            unresolved.append([pair["file_a"], pair["file_b"]])
        elif review["decision"] == "same_session":
            same_session_cross.append([pair["file_a"], pair["file_b"]])
    if same_session_cross:
        raise ValueError("Confirmed same-session images cross splits. Rebuild the grouped split before freezing.")
    verified = manifest.get("grouping_metadata_verified", False)
    if not exploratory and (not verified or unresolved):
        raise ValueError("Paper protocol requires verified capture/route metadata and resolved cross-split candidates. "
                         "For an explicitly exploratory study, use --exploratory; this does not establish independence.")
    entries = []
    for experiment in config["experiments"]:
        for seed in config["seeds"]:
            run = Path(config["output_root"])/f"{experiment['name']}_seed{seed}"
            if not (run/"best.pt").exists() or not (run/"eval/val_metrics.json").exists() or not (run/"eval/thresholds.json").exists():
                raise ValueError(f"Finish development/validation first: {run}")
            if (run/"eval/test_metrics.json").exists():
                raise ValueError(f"Test already opened before this preregistration: {run}. Use a new untouched test study.")
            if read_json(run/"config.json") != experiment_config(config, experiment, seed):
                raise ValueError(f"Run config disagrees with frozen config: {run}")
            selection = read_json(run/"eval/thresholds.json")
            if selection.get("checkpoint_sha256") != digest(run/"best.pt") or selection.get("manifest_sha256") != digest(config["manifest"]):
                raise ValueError(f"Validation calibration is stale: {run}")
            entries.append({"run": run.name, "checkpoint_sha256": digest(run/"best.pt"),
                            "config_sha256": digest(run/"config.json"), "thresholds_sha256": digest(run/"eval/thresholds.json")})
    study = {"schema_version": 2, "config": config, "manifest_sha256": digest(config["manifest"]),
             "similarity_audit_sha256": digest(similarity_audit), "decisions_sha256": digest(decisions_csv) if decisions_csv else None,
             "capture_metadata_verified": verified, "unresolved_similarity_pairs": unresolved,
             "exploratory": exploratory, "entries": entries,
             "primary_endpoint": "COCO bbox AP50:95", "secondary_endpoints": ["crack AP", "AP50", "validation-frozen macro F1", "end-to-end latency"],
             "bootstrap": {"unit": "whole manifest group", "paired": True, "repeats": config.get("study", {}).get("bootstrap_repeats", 1000)},
             "external_manifest": str(Path(external_manifest).resolve()) if external_manifest else None,
             "external_manifest_sha256": digest(external_manifest) if external_manifest else None,
             "interpretation": "Training seed SD and paired group percentile CI are distinct uncertainties; no guaranteed novelty or significant improvement."}
    save_json(output, study)
    print(f"Study frozen: {output}")
    return study


def require_registered(run_dir, config, manifest_path, protocol_path):
    if not config.get("study", {}).get("require_registration", False):
        return
    if not protocol_path:
        raise ValueError("Final test/external requires --protocol. Run develop and freeze-study first.")
    study = read_json(protocol_path)
    run = Path(run_dir)
    entry = next((e for e in study["entries"] if e["run"] == run.name), None)
    if entry is None or digest(run/"best.pt") != entry["checkpoint_sha256"] or digest(run/"config.json") != entry["config_sha256"]:
        raise ValueError("Run/checkpoint changed or absent from the frozen study")
    if digest(run/"eval/thresholds.json") != entry["thresholds_sha256"]:
        raise ValueError("Validation thresholds changed after registration")
    if digest(manifest_path) not in {study["manifest_sha256"], study["external_manifest_sha256"]}:
        raise ValueError("Evaluation manifest was not preregistered")


def finalize_study(protocol_path, data_root=None, external_data_root=None):
    from .engine import evaluate
    from .metrics import group_bootstrap
    from .report import report
    study = read_json(protocol_path)
    config = study["config"]
    root = Path(config["output_root"])
    if study["external_manifest"] and digest(study["external_manifest"]) != study["external_manifest_sha256"]:
        raise ValueError("Registered external manifest changed")
    for entry in study["entries"]:
        run = root/entry["run"]
        require_registered(run, read_json(run/"config.json"), run/"manifest.json", protocol_path)
        if not (run/"eval/test_metrics.json").exists():
            evaluate(run, data_root=data_root, split="test", protocol_path=protocol_path)
        if study["external_manifest"]:
            if not external_data_root:
                raise ValueError("Registered external dataset needs --external-data-root")
            if not (run/"eval/external_metrics.json").exists():
                evaluate(run, data_root=external_data_root, split="external",
                         external_manifest=study["external_manifest"], protocol_path=protocol_path)
    report(root, root/"paper", split="test")
    if study["external_manifest"]:
        report(root, root/"paper_external", split="external")
    comparisons = config.get("study", {}).get("comparisons", [])
    repeats = study["bootstrap"]["repeats"]
    statistical = []
    for split in ["test"] + (["external"] if study["external_manifest"] else []):
        for a, b in comparisons:
            for seed in config["seeds"]:
                da, db = root/f"{a}_seed{seed}"/"eval", root/f"{b}_seed{seed}"/"eval"
                if not (da/f"{split}_records.json").exists() or not (db/f"{split}_records.json").exists():
                    raise ValueError(f"Missing preregistered comparison: {a}, {b}, seed {seed}")
                records = read_json(da/f"{split}_records.json")
                if records != read_json(db/f"{split}_records.json"):
                    raise ValueError("Paired comparisons need exactly the same records")
                path = db/f"bootstrap_vs_{a}_{split}.json"
                if path.exists():
                    result = read_json(path)
                else:
                    try:
                        result = group_bootstrap(records, read_json(da/f"{split}_predictions.json"),
                                                  read_json(db/f"{split}_predictions.json"), repeats, seed)
                    except ValueError as exc:
                        result = {"status": "not_estimable", "reason": str(exc)}
                    save_json(path, result)
                statistical.append({"baseline": a, "method": b, "seed": seed, "split": split,
                                    **{k: v for k, v in result.items() if k != "samples"}})
    save_json(root/"paper/statistical_comparisons.json", statistical)
    save_json(root/"paper/frozen_study.json", study)
