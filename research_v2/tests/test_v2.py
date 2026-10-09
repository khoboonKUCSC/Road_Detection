import copy
import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from roadresearch.cli import config_load, experiment_config
from roadresearch.data import RoadDataset, prepare
from roadresearch.leakage import audit_similarity
from roadresearch.metrics import group_bootstrap
from roadresearch.models import build_model
from roadresearch.protocol import freeze_study, require_registered
from roadresearch.utils import ROOT, read_json, save_json
from test_research import fixture_data


def test_similarity_candidates_and_reviewed_whole_group_merge(tmp_path):
    root = fixture_data(tmp_path)
    image = np.asarray(Image.open(root/"images/20250101_120000.png")).copy()
    image[0, 0, 0] += 1
    Image.fromarray(image).save(root/"images/20250102_120000.png")
    manifest = prepare(root, tmp_path/"dataset")
    pairs = audit_similarity(tmp_path/"dataset/manifest.json", root, tmp_path/"audit", contact_pairs=2)
    pair = next(r for r in pairs if {r["file_a"], r["file_b"]} == {"20250101_120000.png", "20250102_120000.png"})
    assert pair["phash_distance"] <= 2
    decisions = tmp_path/"decisions.csv"
    with decisions.open("w", newline="") as stream:
        w=csv.DictWriter(stream, fieldnames=["file_a","file_b","decision","reviewer","evidence"])
        w.writeheader();w.writerow({"file_a":pair["file_a"],"file_b":pair["file_b"],"decision":"same_session",
                                   "reviewer":"test","evidence":"synthetic provenance"})
    merged = prepare(root, tmp_path/"merged", reviewed_links=decisions)
    grouped = {r["file_name"]:r["group"] for r in merged["records"]}
    assert grouped[pair["file_a"]] == grouped[pair["file_b"]]


def test_rdd_import_maps_and_coordinates(tmp_path):
    from roadresearch.external import import_rdd2022
    root=tmp_path/"rdd/India/train"
    (root/"images").mkdir(parents=True);(root/"annotations/xmls").mkdir(parents=True)
    Image.new("RGB",(100,100)).save(root/"images/India_1.jpg")
    xml='<annotation><filename>India_1.jpg</filename><object><name>D00</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>20</xmax><ymax>10</ymax></bndbox></object><object><name>D40</name><bndbox><xmin>50</xmin><ymin>50</ymin><xmax>70</xmax><ymax>70</ymax></bndbox></object></annotation>'
    (root/"annotations/xmls/India_1.xml").write_text(xml)
    output=tmp_path/"converted"
    import_rdd2022(tmp_path/"rdd",output)
    manifest=prepare(output,tmp_path/"external",groups_csv=output/"groups.csv",external=True,source="synthetic RDD test")
    annotations=manifest["records"][0]["annotations"]
    assert [a["category_id"] for a in annotations] == [2,1]
    assert annotations[0]["bbox_xyxy"] == [0.,0.,20.,10.]


def test_bootstrap_two_class_external():
    from test_research import perfect_records
    records,predictions=perfect_records()
    records=[{**r,"annotations":[a for a in r["annotations"] if a["category_id"] != 3]} for r in records]
    predictions=[p for p in predictions if p["category_id"] != 3]
    result=group_bootstrap(records,predictions,predictions,repeats=4)
    assert result["CI95_percentile"] == [0.,0.]


def test_registration_blocks_test_and_changes(tmp_path):
    base=config_load(ROOT/"configs/smoke_cpu.yaml")
    base.update(output_root=str(tmp_path/"runs"),manifest=str(tmp_path/"manifest.json"))
    base["experiments"]=base["experiments"][:1];base["study"]={"require_registration":True,"bootstrap_repeats":4}
    save_json(base["manifest"],{"grouping":"proxy","grouping_metadata_verified":False,"records":[]})
    from roadresearch.utils import digest
    audit=tmp_path/"audit.json"
    save_json(audit,{"manifest_sha256":digest(base["manifest"]),"pairs":[]})
    exp=base["experiments"][0];cfg=experiment_config(base,exp,0)
    run=tmp_path/"runs"/f"{exp['name']}_seed0";(run/"eval").mkdir(parents=True)
    save_json(run/"config.json",cfg);(run/"best.pt").write_bytes(b"synthetic checkpoint")
    save_json(run/"eval/thresholds.json",{"checkpoint_sha256":digest(run/"best.pt"),"manifest_sha256":digest(base["manifest"])})
    save_json(run/"eval/val_metrics.json",{})
    with pytest.raises(ValueError,match="requires --protocol"):
        require_registered(run,cfg,base["manifest"],None)
    with pytest.raises(ValueError,match="verified capture"):
        freeze_study(base,tmp_path/"strict.json",audit)
    protocol=tmp_path/"protocol.json"
    freeze_study(base,protocol,audit,exploratory=True)
    require_registered(run,cfg,base["manifest"],protocol)
    (run/"best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed"):
        require_registered(run,cfg,base["manifest"],protocol)


@pytest.mark.parametrize("architecture",["yolo11s","rtdetr_v2_r50vd"])
def test_modern_adapters_loss_inference_and_reload(architecture):
    if architecture == "yolo11s":
        pytest.importorskip("ultralytics")
    else:
        pytest.importorskip("transformers")
    base=config_load(ROOT/"configs/smoke_cpu.yaml")
    cfg=experiment_config(base,base["experiments"][0],0)
    cfg["model"].update(architecture=architecture,input_size=64,geometry_mode="none",yolo_yaml="yolo11n.yaml")
    if architecture == "rtdetr_v2_r50vd":
        from transformers import RTDetrV2Config
        cfg["model"]["hf_config"]=RTDetrV2Config(num_labels=3,num_queries=20,decoder_layers=1,encoder_layers=1,
                                 num_denoising=0,anchor_image_size=[64,64],disable_custom_kernels=True).to_dict()
    torch.set_num_threads(2)
    model=build_model(cfg,pretrained=False)
    image=torch.rand(3,64,96)
    target={"boxes":torch.tensor([[10.,10.,40.,20.]]),"labels":torch.tensor([2]),"image_id":torch.tensor(1)}
    model.train();loss=sum(model([image,image],[target,target]).values())
    assert torch.isfinite(loss);loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    spec=model.network.config.to_dict() if architecture == "rtdetr_v2_r50vd" else None
    del model
    model=build_model(cfg,pretrained=False,model_spec=spec)
    model.load_state_dict(state);model.eval()
    with torch.inference_mode():prediction=model([image])[0]
    assert prediction["boxes"].shape[1] == 4
    assert ((prediction["labels"] >= 1)&(prediction["labels"] <= 3)).all()
