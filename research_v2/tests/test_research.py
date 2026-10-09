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
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key


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
