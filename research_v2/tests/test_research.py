import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from roadresearch.cli import config_load, experiment_config
from roadresearch.data import RoadDataset, geometry_target, parse_label, prepare, verify_manifest
from roadresearch.metrics import coco_evaluate, confusion_matrix, group_bootstrap, operating_metrics
from roadresearch.models import build_model
from roadresearch.utils import ROOT, load_checkpoint, read_json


def fixture_data(tmp_path):
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for day in range(1, 7):
        for frame in range(2):
            name = f"202501{day:02d}_12000{frame}"
            image = np.zeros((64, 96, 3), dtype=np.uint8)
            image[:, :, 0] = day*25
            image[frame*10:frame*10+10, :, 1] = 200
            Image.fromarray(image).save(root / "images" / (name + ".png"))
            (root / "labels" / (name + ".txt")).write_text(
                "0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n"
                "1 0.1 0.5 0.8 0.5 0.8 0.55 0.1 0.55\n"
                "2 0.7 0.1 0.9 0.1 0.9 0.3 0.7 0.3\n")
    return root


def test_parse_clip_and_geometry(tmp_path):
    path = tmp_path / "label.txt"
    path.write_text("1 -0.01 0.1 0.8 0.1 0.8 0.2 -0.01 0.2\n")
    with pytest.raises(ValueError):
        parse_label(path, 100, 100)
    warnings = []
    result = parse_label(path, 100, 100, warnings)
    assert result[0]["bbox_xyxy"] == [0., 10., 80., 20.]
    assert result[0]["quadrilateral"] is None
    assert warnings[0]["action"] == "clip_AABB_to_image_disable_geometry"
    horizontal, _ = geometry_target([[0, 0], [10, 0], [10, 1], [0, 1]])
    vertical, _ = geometry_target([[0, 0], [1, 0], [1, 10], [0, 10]])
    assert horizontal[2] == pytest.approx(1)
    assert vertical[2] == pytest.approx(-1)
    assert horizontal[0] == pytest.approx(vertical[0])


def test_prepare_whole_groups_and_conflict_quarantine(tmp_path):
    root = fixture_data(tmp_path)
    source = root / "images/20250101_120000.png"
    (root / "images/20250107_120000.png").write_bytes(source.read_bytes())
    (root / "labels/20250107_120000.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    manifest = prepare(root, tmp_path / "prepared")
    audit = read_json(tmp_path / "prepared/audit.json")
    assert len(audit["conflicting_duplicate_clusters"]) == 1
    assert "20250101_120000.png" not in [r["file_name"] for r in manifest["records"]]
    groups = {}
    for r in manifest["records"]:
        groups.setdefault(r["group"], set()).add(r["split"])
    assert all(len(v) == 1 for v in groups.values())
    verify_manifest(manifest, root)
    record = manifest["records"][0]
    (root / record["label_path"]).write_text("")
    with pytest.raises(ValueError, match="Data changed"):
        verify_manifest(manifest, root)


def perfect_records():
    records = []
    predictions = []
    for i in range(1, 4):
        annotations = [{"category_id": cls, "bbox_xyxy": [10*cls, 10, 10*cls+8, 18], "quadrilateral": None}
                       for cls in range(1, 4)]
        records.append({"id": i, "file_name": f"{i}.png", "width": 80, "height": 64,
                        "group": f"g{i}", "annotations": annotations})
        for cls in range(1, 4):
            predictions.append({"image_id": i, "category_id": cls, "bbox": [10*cls, 10, 8, 8], "score": 0.9})
    return records, predictions


def test_official_coco_perfect_empty_and_bootstrap():
    records, predictions = perfect_records()
    metrics, _, _ = coco_evaluate(records, predictions)
    assert metrics["AP"] == pytest.approx(1)
    empty, _, _ = coco_evaluate(records, [])
    assert empty["AP"] == 0
    thresholds = {name: .5 for name in ["pothole", "crack", "manhole"]}
    assert operating_metrics(records, predictions, thresholds)["macro_F1"] == 1
    confusion = confusion_matrix(records, predictions, thresholds)
    assert np.diag(confusion).tolist() == [3, 3, 3, 0]
    result = group_bootstrap(records, predictions, predictions, repeats=4)
    assert result["CI95_percentile"] == [0., 0.]


def test_geometry_gradient_and_detection_initialization(tmp_path):
    root = fixture_data(tmp_path)
    manifest = prepare(root, tmp_path / "prepared")
    base = config_load(ROOT / "configs/smoke_cpu.yaml")
    cfg = experiment_config(base, base["experiments"][1], 0)
    torch.set_num_threads(2)
    torch.manual_seed(123)
    geometry = build_model(cfg)
    baseline_cfg = copy.deepcopy(cfg)
    baseline_cfg["model"]["geometry_mode"] = "none"
    torch.manual_seed(123)
    baseline = build_model(baseline_cfg)
    for key, value in baseline.detector.state_dict().items():
        assert torch.equal(value, geometry.detector.state_dict()[key])
    ds = RoadDataset(manifest, root, "train")
    image, target = ds[0]
    geometry.train()
    losses = geometry([image], [target])
    auxiliary = losses["loss_geometry_aspect"] + losses["loss_geometry_orientation"]
    auxiliary.backward()
    assert torch.isfinite(auxiliary)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in geometry.detector.backbone.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in geometry.geometry_head.parameters())
    geometry.eval()
    with torch.inference_mode():
        output = geometry([image])[0]
    assert {"boxes", "labels", "scores"}.issubset(output)


def test_tiling_coordinates_and_nms():
    from roadresearch.engine import predict_tensor, tile_starts
    assert tile_starts(640, 384, .25) == [0, 256]
    cfg = config_load(ROOT / "configs/smoke_cpu.yaml")
    cfg["evaluation"].update(tiling=True, tile_size=32, tile_overlap=0., include_full_image=False)
    class Fake:
        def __call__(self, images):
            return [{"boxes": torch.tensor([[1., 2., 5., 6.]]), "labels": torch.tensor([1]),
                     "scores": torch.tensor([.9])}]
    result = predict_tensor(Fake(), torch.zeros(3, 32, 64), cfg)
    assert result["boxes"].tolist() == [[1., 2., 5., 6.], [33., 2., 37., 6.]]


def test_crack_crop_boxes_and_disabled_truncated_geometry(tmp_path):
    root = fixture_data(tmp_path)
    manifest = prepare(root, tmp_path / "prepared")
    torch.manual_seed(0)
    dataset = RoadDataset(manifest, root, "train", augment=True, crop_probability=1., crop_size=32)
    image, target = dataset[0]
    assert image.shape == (3, 32, 32)
    assert len(target["boxes"]) > 0
    assert (target["boxes"] >= 0).all() and (target["boxes"] <= 32).all()
    assert (target["boxes"][:, 2:] > target["boxes"][:, :2]).all()
    assert not target["geometry_valid"][target["labels"] == 2].any()


def test_epoch_boundary_resume_equivalence(tmp_path, monkeypatch):
    from roadresearch import engine
    root = fixture_data(tmp_path)
    manifest_path = tmp_path / "prepared/manifest.json"
    prepare(root, manifest_path.parent)
    base = config_load(ROOT / "configs/smoke_cpu.yaml")
    cfg = experiment_config(base, base["experiments"][1], 12)
    cfg.update(data_root=str(root), manifest=str(manifest_path))
    cfg["training"].update(limit_train=2, limit_eval=2, epochs=2, patience=5)
    # A tied score must retain epoch 1, testing recovery of a missing best.pt.
    monkeypatch.setattr(engine, "coco_evaluate", lambda records, predictions: ({"AP": 0., "AP50": 0.}, {}, ""))
    engine.train(cfg, tmp_path / "continuous")
    original = engine.atomic_checkpoint
    def stop_after_first_last(path, state):
        original(path, state)
        if Path(path).name == "last.pt" and state["epoch"] == 0:
            raise InterruptedError("simulated interruption after completed epoch")
    monkeypatch.setattr(engine, "atomic_checkpoint", stop_after_first_last)
    with pytest.raises(InterruptedError):
        engine.train(cfg, tmp_path / "interrupted")
    monkeypatch.setattr(engine, "atomic_checkpoint", original)
    # first interruption happened before best.pt: resume must recover the saved best state.
    engine.train(cfg, tmp_path / "interrupted", resume=True)
    a, b = load_checkpoint(tmp_path / "continuous/last.pt"), load_checkpoint(tmp_path / "interrupted/last.pt")
    assert a["history"] == b["history"]
    assert a["total_skipped_amp_steps"] == b["total_skipped_amp_steps"]
    assert a["overflow_events"] == b["overflow_events"]
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key


def test_finite_gradient_step_clips_and_updates(tmp_path):
    from roadresearch import engine
    root = fixture_data(tmp_path)
    manifest_path = tmp_path / "prepared/manifest.json"
    prepare(root, manifest_path.parent)
    base = config_load(ROOT / "configs/smoke_cpu.yaml")
    cfg = experiment_config(base, base["experiments"][0], 0)
    cfg.update(data_root=str(root), manifest=str(manifest_path))
    cfg["training"].update(limit_train=2, limit_eval=2, epochs=1, grad_clip=0.01)
    run_dir = tmp_path / "finite_step"
    engine.train(cfg, run_dir)
    ckpt = load_checkpoint(run_dir / "last.pt")
    assert ckpt["history"][0]["skipped_amp_steps"] == 0
    assert ckpt["total_skipped_amp_steps"] == 0
    assert len(ckpt["overflow_events"]) == 0
    assert not (run_dir / "overflow_events.jsonl").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP scaler test")
def test_amp_nonfinite_gradient_skips_step_and_backs_off(tmp_path):
    from roadresearch import engine
    root = fixture_data(tmp_path)
    manifest_path = tmp_path / "prepared/manifest.json"
    prepare(root, manifest_path.parent)
    base = config_load(ROOT / "configs/smoke_cpu.yaml")
    cfg = experiment_config(base, base["experiments"][0], 0)
    cfg.update(data_root=str(root), manifest=str(manifest_path), device="cuda")
    cfg["training"].update(limit_train=4, limit_eval=2, epochs=1, patience=5, amp=True, batch_size=2)
    run_dir = tmp_path / "amp_nan"

    orig_unscale = torch.amp.GradScaler.unscale_
    call_count = 0

    def unscale_mock(scaler_self, optimizer):
        nonlocal call_count
        call_count += 1
        orig_unscale(scaler_self, optimizer)
        if call_count == 1:
            p = next(p for p in optimizer.param_groups[0]["params"] if p.grad is not None)
            p.grad.data.fill_(float("nan"))

    torch.amp.GradScaler.unscale_ = unscale_mock
    try:
        engine.train(cfg, run_dir)
    finally:
        torch.amp.GradScaler.unscale_ = orig_unscale

    ckpt = load_checkpoint(run_dir / "last.pt")
    assert ckpt["history"][0]["skipped_amp_steps"] == 1
    assert ckpt["total_skipped_amp_steps"] == 1
    assert len(ckpt["overflow_events"]) == 1
    event = ckpt["overflow_events"][0]
    assert event["epoch"] == 1
    assert event["step"] == 1
    assert len(event["image_ids"]) > 0
    assert event["scale_before"] == 65536.0
    assert event["scale_after"] == 32768.0
    assert event["bad_parameter_count"] >= 1
    assert len(event["parameters"]) >= 1
    assert ckpt["scaler"]["scale"] == 32768.0
    jsonl_lines = (run_dir / "overflow_events.jsonl").read_text().splitlines()
    assert len(jsonl_lines) == 1


def test_nonfinite_gradient_amp_disabled_raises_floating_point_error(tmp_path):
    from roadresearch import engine
    root = fixture_data(tmp_path)
    manifest_path = tmp_path / "prepared/manifest.json"
    prepare(root, manifest_path.parent)
    base = config_load(ROOT / "configs/smoke_cpu.yaml")
    cfg = experiment_config(base, base["experiments"][0], 0)
    cfg.update(data_root=str(root), manifest=str(manifest_path), device="cpu")
    cfg["training"].update(limit_train=2, limit_eval=2, epochs=1, patience=5, amp=False)
    run_dir = tmp_path / "cpu_nan"

    orig_unscale = torch.amp.GradScaler.unscale_

    def unscale_mock(scaler_self, optimizer):
        orig_unscale(scaler_self, optimizer)
        for p in optimizer.param_groups[0]["params"]:
            if p.grad is not None:
                p.grad.data.fill_(float("nan"))
                break

    torch.amp.GradScaler.unscale_ = unscale_mock
    try:
        with pytest.raises(FloatingPointError) as exc_info:
            engine.train(cfg, run_dir)
        err = str(exc_info.value)
        assert "Nonfinite gradients with AMP disabled" in err
        assert "image_ids" in err
        assert "losses" in err
        assert "detector.backbone.net.0.weight" in err
    finally:
        torch.amp.GradScaler.unscale_ = orig_unscale



@pytest.mark.parametrize("architecture,mode", [("fasterrcnn_r50_fpn_v2", "full"), ("fcos_r50_fpn", "none")])
def test_real_architecture_forward_backward_reload(architecture, mode):
    """Exercise the production models offline, with random weights and small input."""
    torch.set_num_threads(2)
    base = config_load(ROOT / "configs/smoke_cpu.yaml")
    cfg = experiment_config(base, base["experiments"][0], 0)
    cfg["model"].update(architecture=architecture, geometry_mode=mode, pretrained=True,
                        trainable_backbone_layers=3)
    model = build_model(cfg, pretrained=False)
    target = {"boxes": torch.tensor([[10., 10., 40., 20.]]), "labels": torch.tensor([2]),
              "geometry": torch.tensor([[.5, 0., 1.]]), "geometry_valid": torch.tensor([True]),
              "anisotropy": torch.tensor([.8])}
    image = torch.rand(3, 64, 96)
    model.train()
    losses = model([image], [target])
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    signature = [(name, p.requires_grad) for name, p in model.named_parameters()]
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    del model
    reloaded = build_model(cfg, pretrained=False)
    reloaded.load_state_dict(state)
    assert signature == [(name, p.requires_grad) for name, p in reloaded.named_parameters()]
    reloaded.eval()
    with torch.inference_mode():
        prediction = reloaded([image])[0]
    assert prediction["boxes"].shape[1] == 4


def test_migrate_run_preserves_initialization_metadata(tmp_path):
    from roadresearch.engine import migrate_run
    from roadresearch.utils import read_json, save_json

    source = tmp_path / "source_run"
    source.mkdir()
    (source / "best.pt").write_bytes(b"best_data")
    (source / "last.pt").write_bytes(b"last_data")
    save_json(source / "manifest.json", {"records": []})
    save_json(source / "history.json", [{"epoch": 1}])
    (source / "history.csv").write_text("epoch\n1\n")
    save_json(source / "config.json", {"experiment": "test_exp", "seed": 0})
    save_json(source / "source_sha256.json", {"dummy.py": "abc123hash"})
    init_data = {
        "pretrained": True, "weights": "COCO_V1", "seed": 0,
        "architecture": "fasterrcnn_r50_fpn_v2", "geometry_mode": "none",
        "parameters": 43266403, "trainable_parameters": 43041059
    }
    save_json(source / "initialization.json", init_data)

    dest = tmp_path / "dest_run"
    migrate_run(source, dest, reason="test initialization preservation")

    assert (dest / "initialization.json").exists()
    assert read_json(dest / "initialization.json") == init_data
    assert (dest / "migration.json").exists()
    assert (dest / "best.pt").read_bytes() == b"best_data"
    assert (dest / "last.pt").read_bytes() == b"last_data"
    assert (dest / "history.csv").read_text() == "epoch\n1\n"
    assert (dest / "manifest.json").exists()
    assert (dest / "history.json").exists()


def test_report_missing_initialization_raises_contextual_error(tmp_path):
    from roadresearch.report import report
    from roadresearch.utils import save_json

    output_root = tmp_path / "runs"
    run_dir = output_root / "fasterrcnn_baseline_seed0"
    (run_dir / "eval").mkdir(parents=True)
    save_json(run_dir / "config.json", {"experiment": "fasterrcnn_baseline", "seed": 0})
    save_json(run_dir / "eval/val_metrics.json", {
        "smoke_only": False, "AP": 0.15, "AP50": 0.25, "AP75": 0.15,
        "AP_small": 0.05, "AP_medium": 0.15, "AP_large": 0.25,
        "checkpoint_epoch": 18, "manifest_sha256": "fake_hash",
        "latency": {"mean_ms": 10.0},
        "operating_point": {"macro_F1": 0.5, "per_class": {c: {} for c in ["pothole", "crack", "manhole"]}},
        "per_class": {c: {"AP": 0.15, "AP50": 0.25} for c in ["pothole", "crack", "manhole"]},
        "confusion_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "confusion_labels": ["pothole", "crack", "manhole", "background"],
    })
    save_json(run_dir / "history.json", [{"epoch": 1, "loss_total": 0.5, "val_AP": 0.15, "val_AP50": 0.25}])
    save_json(run_dir / "eval/val_pr_curves.json", {c: {"recall": [0, 1], "precision50": [1, 0]} for c in ["pothole", "crack", "manhole"]})
    save_json(run_dir / "migration.json", {"migrated_from": "/path/to/orig/fasterrcnn_baseline_seed0"})

    dest = tmp_path / "report_out"
    with pytest.raises(FileNotFoundError) as exc_info:
        report(output_root, dest, split="val")

    err_msg = str(exc_info.value)
    assert "fasterrcnn_baseline_seed0" in err_msg
    assert "missing initialization.json" in err_msg
    assert "migrated run" in err_msg
    assert "/path/to/orig/fasterrcnn_baseline_seed0" in err_msg


def test_configs_match_for_run_criteria(tmp_path):
    from roadresearch.protocol import configs_match_for_run
    from roadresearch.utils import save_json

    expected = {
        "output_root": "/new/runs_root",
        "experiment": "fasterrcnn_baseline",
        "seed": 0,
        "manifest": "/path/to/manifest.json",
        "model": {"architecture": "fasterrcnn_r50_fpn_v2", "geometry_mode": "none"},
        "training": {"lr": 0.0002, "epochs": 20},
    }
    run_dir = tmp_path / "fasterrcnn_baseline_seed0"
    run_dir.mkdir()

    # 1. Equal expected/actual configs pass without migration.json
    actual_equal = copy.deepcopy(expected)
    assert configs_match_for_run(expected, actual_equal, run_dir) is True

    # 2. A non-migrated run differing only in output_root fails
    actual_diff_root = copy.deepcopy(expected)
    actual_diff_root["output_root"] = "/old/runs_root"
    assert configs_match_for_run(expected, actual_diff_root, run_dir) is False

    # 3. A migrated run differing only in output_root passes
    save_json(run_dir / "migration.json", {
        "migrated_from": f"/old/runs_root/{run_dir.name}",
        "reason": "AMP fix",
    })
    assert configs_match_for_run(expected, actual_diff_root, run_dir) is True

    # 4. A migrated run with any additional changed field fails
    # 4a. Changed seed
    actual_bad_seed = copy.deepcopy(actual_diff_root)
    actual_bad_seed["seed"] = 42
    assert configs_match_for_run(expected, actual_bad_seed, run_dir) is False

    # 4b. Changed training.lr
    actual_bad_lr = copy.deepcopy(actual_diff_root)
    actual_bad_lr["training"]["lr"] = 0.05
    assert configs_match_for_run(expected, actual_bad_lr, run_dir) is False

    # 4c. Changed model.architecture
    actual_bad_arch = copy.deepcopy(actual_diff_root)
    actual_bad_arch["model"]["architecture"] = "fcos_r50_fpn"
    assert configs_match_for_run(expected, actual_bad_arch, run_dir) is False

    # 5. A malformed/mismatched migration record fails if recommended provenance validation is implemented
    # 5a. Mismatched run name
    save_json(run_dir / "migration.json", {
        "migrated_from": "/old/runs_root/different_experiment_seed0",
    })
    with pytest.raises(ValueError, match="Mismatched migration provenance"):
        configs_match_for_run(expected, actual_diff_root, run_dir)

    # 5b. Malformed migration record (missing migrated_from)
    save_json(run_dir / "migration.json", {"reason": "missing source"})
    with pytest.raises(ValueError, match="Malformed migration metadata"):
        configs_match_for_run(expected, actual_diff_root, run_dir)

    # 5c. Malformed JSON file
    (run_dir / "migration.json").write_text("{not valid json")
    with pytest.raises(ValueError, match="Malformed migration metadata"):
        configs_match_for_run(expected, actual_diff_root, run_dir)


def test_freeze_study_migration_integration(tmp_path):
    from roadresearch.protocol import freeze_study
    from roadresearch.utils import digest, save_json

    manifest_path = tmp_path / "manifest.json"
    save_json(manifest_path, {"records": [], "grouping_metadata_verified": True})
    manifest_hash = digest(manifest_path)

    audit_path = tmp_path / "similarity_audit.json"
    save_json(audit_path, {
        "manifest_sha256": manifest_hash,
        "pairs": [],
    })

    runs_root = tmp_path / "runs_recovered"
    run_dir = runs_root / "smoke_exp_seed0"
    (run_dir / "eval").mkdir(parents=True)

    base_config = {
        "output_root": str(runs_root),
        "manifest": str(manifest_path),
        "seeds": [0],
        "model": {},
        "training": {},
        "experiments": [{"name": "smoke_exp", "architecture": "fasterrcnn_r50_fpn_v2", "geometry_mode": "none"}],
        "evaluation": {"tiling": False},
        "study": {"bootstrap_repeats": 10},
    }

    from roadresearch.cli import experiment_config
    expected_cfg = experiment_config(base_config, base_config["experiments"][0], 0)

    # Stored run config has old output_root
    stored_run_cfg = copy.deepcopy(expected_cfg)
    stored_run_cfg["output_root"] = "/old/runs_root"
    save_json(run_dir / "config.json", stored_run_cfg)

    # Required development artifacts
    (run_dir / "best.pt").write_bytes(b"checkpoint_weights")
    ckpt_hash = digest(run_dir / "best.pt")
    save_json(run_dir / "eval/val_metrics.json", {"AP": 0.2})
    save_json(run_dir / "eval/thresholds.json", {
        "checkpoint_sha256": ckpt_hash,
        "manifest_sha256": manifest_hash,
        "thresholds": {"pothole": 0.5, "crack": 0.5, "manhole": 0.5},
    })

    protocol_out = tmp_path / "study/protocol.json"

    # Case 2: Non-migrated run differing only in output_root fails
    with pytest.raises(ValueError, match="Run config disagrees with frozen config"):
        freeze_study(base_config, protocol_out, audit_path, exploratory=True)
    assert not protocol_out.exists()

    # Case 4: Migrated run with additional changed field fails
    save_json(run_dir / "migration.json", {"migrated_from": f"/old/runs_root/{run_dir.name}"})
    stored_run_cfg_bad = copy.deepcopy(stored_run_cfg)
    stored_run_cfg_bad["seed"] = 999
    save_json(run_dir / "config.json", stored_run_cfg_bad)
    with pytest.raises(ValueError, match="Run config disagrees with frozen config"):
        freeze_study(base_config, protocol_out, audit_path, exploratory=True)
    assert not protocol_out.exists()

    # Case 5: Malformed/mismatched migration record fails
    save_json(run_dir / "config.json", stored_run_cfg)
    save_json(run_dir / "migration.json", {"migrated_from": "/old/runs_root/mismatched_run_seed0"})
    with pytest.raises(ValueError, match="Mismatched migration provenance"):
        freeze_study(base_config, protocol_out, audit_path, exploratory=True)
    assert not protocol_out.exists()

    # Case 3: Migrated run differing only in output_root passes and writes protocol
    save_json(run_dir / "migration.json", {"migrated_from": f"/old/runs_root/{run_dir.name}"})
    study = freeze_study(base_config, protocol_out, audit_path, exploratory=True)
    assert protocol_out.exists()
    assert study["entries"][0]["run"] == "smoke_exp_seed0"
    # Ensure config.json in run_dir was untouched
    assert read_json(run_dir / "config.json") == stored_run_cfg

    # Rejects if test already opened
    protocol_out_2 = tmp_path / "study/protocol_2.json"
    save_json(run_dir / "eval/test_metrics.json", {"AP": 0.2})
    with pytest.raises(ValueError, match="Test already opened"):
        freeze_study(base_config, protocol_out_2, audit_path, exploratory=True)
    assert not protocol_out_2.exists()

