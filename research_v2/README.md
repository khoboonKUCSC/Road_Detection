# Road damage research v2 — Ubuntu / RTX 4090

ชุดใหม่เพิ่ม 4 ส่วน: related-work/claim review, strong baselines, leakage auditing และการประเมิน
แบบตรึง protocol พร้อมสถิติและข้อมูลภายนอก โดยเก็บ `research_v1/` ไว้ครบ

## โมเดล

8 configurations × seeds `[0,42,2026]` = **24 runs**:

| Configuration | Model / intervention |
|---|---|
| fasterrcnn_baseline | Faster R-CNN ResNet-50-FPN v2, COCO pretrained |
| fasterrcnn_aspect | Baseline + enclosure elongation auxiliary loss |
| fasterrcnn_geometry | Baseline + enclosure elongation/orientation losses |
| fasterrcnn_detail | Baseline + crack-centered native crop augmentation |
| fasterrcnn_detail_geometry | Crop + enclosure auxiliary losses |
| fcos_baseline | FCOS ResNet-50-FPN, COCO pretrained |
| yolo11s_baseline | YOLO11s, COCO pretrained |
| rtdetr_v2_baseline | RT-DETRv2 R50-vd via Hugging Face, COCO pretrained |

YOLO ใช้ native loss/NMS แต่ใช้ optimizer/scheduler/checkpoint loop ร่วมกับชุดนี้
RT-DETRv2 ใช้ native Hugging Face loss/processor; exact model config เก็บใน checkpoint/model_spec
ทุกโมเดลส่ง predictions ไป pycocotools COCO bbox evaluator เดียวกัน
ไม่ใช่การนำ native metrics ต่างสูตรมาเปรียบเทียบ IDs ภายใน 1=pothole,2=crack,3=manhole
input labels ใช้ 0,1,2 ตามต้นฉบับ

Faster R-CNN/FCOS preserve aspect ratio, YOLO letterboxes, RT-DETRv2 resizes square
เก็บ input config, parameters และ end-to-end latency จึงไม่อ้างว่า pixels/FLOPs เท่ากัน
**ยังไม่รวม RF-DETR adapter**; RT-DETRv2 เป็น modern-transformer comparator ใน v2

## 1. ติดตั้ง

วาง `research_v2/` ข้าง `data/` โดยภาพต้นฉบับอยู่ใน `data/images/` และ labels ใน `data/labels/`
ไม่ใช้ train/val subfolders เก่า ต้องมี internet สำหรับ dependencies/pretrained ครั้งแรก

```bash
cd Road_Damage/research_v2
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

แนะนำ Python 3.11/3.12 และ driver รองรับ CUDA wheel ที่เลือก สคริปต์ pin Torch 2.7.1 /
Torchvision 0.22.1, cu128 และติดตั้ง requirements ทั้งสองไฟล์
override ได้ เช่น `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126`
ตรวจ https://pytorch.org/get-started/locally/ ให้ตรงเครื่อง
ค่าเริ่มต้น batch 4, AMP FP16, 80 epochs, patience 15 ยังไม่ได้วัด VRAM จริงบน 4090
ถ้า OOM ลด batch ก่อนเริ่ม study ใหม่ ต้องเผื่อพื้นที่หลายสิบ GB สำหรับ 24 runs และ tuning

## 2. ความใหม่ / แผน paper

อ่าน `docs/related_work_matrix.csv` และ `docs/research_protocol.md`
ระบุ source, overlap, access level ของหลักฐาน และ claim boundaries
**Related-work analysis ไม่ได้ทำให้วิธี novel โดยอัตโนมัติ** Crop/slicing และ auxiliary supervision
มี prior art แล้ว และ quadrilaterals ที่ใช้ได้ในข้อมูลนี้เป็น axis-aligned ทั้งหมด
geometry จึงเป็น enclosure supervision จาก bbox ไม่ใช่ทิศ crack จริง
ก่อน algorithmic claim ต้องเทียบ full methods ของงานใกล้เคียงและพิสูจน์ผลหลาย domain

## 3. Audit / grouping

มี `artifacts/dataset/manifest.json` ของข้อมูลปัจจุบัน: 1,999 ภาพ / 4,701 annotations
หลัง cleaning ที่บันทึกใน audit ต้นฉบับไม่ถูกแก้ไข มี v1 reference สำหรับ lineage
manifest นี้ยังใช้ date proxy ที่ไม่ได้ยืนยัน capture metadata

นโยบาย: deduplicate decoded RGB; quarantine conflicting-label duplicates;
clip AABB; disable geometry สำหรับ invalid/clipped quadrilaterals;
drop zero-area annotation พร้อมเหตุผล; malformed/missing/nonfinite labels หยุดพร้อม audit

```bash
python run.py audit-similarity
```

ได้ `artifacts/leakage/similarity_audit.json`, `review_pairs.csv`, contact sheets
pHash/dHash + thumbnail error หา cross-split candidates แต่ไม่รับรอง session identity
กรอก decision=`same_session`, `different_scene` หรือ `uncertain` พร้อม reviewer/evidence
เมื่อยืนยัน same-session ให้ merge ทั้งกลุ่มแล้วแบ่งใหม่:

```bash
python run.py prepare --output artifacts/dataset_reviewed \
  --reviewed-links artifacts/leakage/review_pairs.csv
python run.py audit-similarity --manifest artifacts/dataset_reviewed/manifest.json \
  --output artifacts/leakage_reviewed
```

แก้ manifest ใน YAML ให้ตรง ห้ามย้ายเฉพาะเฟรมเพื่อเพิ่มคะแนน
หากมี capture/route metadata จริง ใช้ CSV `file_name,group_id` ครอบคลุมต้นฉบับทุกภาพ:

```bash
python run.py prepare --groups-csv /path/capture_groups.csv --verified-metadata \
  --output artifacts/dataset_verified
```

`--verified-metadata` เป็นคำประกาศที่ต้องมี evidence ไม่ใช่การรับรองโดยโค้ด
CSV ที่เดาจากชื่อไฟล์ไม่ทำให้ metadata verified โค้ดไม่สร้าง route/device labels เทียม

## 4. Development — ไม่เปิด test

```bash
bash scripts/run_all.sh
```

doctor → audit → develop: ฝึก 24 runs และสร้าง validation report เท่านั้น
หรือ `python run.py develop --config configs/rtx4090.yaml --resume`
ผลอยู่ `runs/paper_study/` ทุก run เก็บ best/last/periodic models, optimizer, scheduler,
AMP scaler, RNG/loader states, source snapshots/hash, config, environment, histories และ predictions
resume จาก completed epoch ล่าสุด ต้องเป็น code/config/manifest เดิม การเปลี่ยนสูตรใช้ output_root ใหม่

Optional equal-budget validation tuning:

```bash
python run.py tune --config configs/rtx4090.yaml --grid configs/tuning.yaml --resume
python run.py develop --config runs/tuning/selected_config.yaml --resume
```

default grid LR 3 ค่า, 40 epochs, seed 0 เท่ากันทุก configuration
เลือกด้วย validation AP และเก็บ tuning_results.json
selected_config คง full-duration 80 epochs / 3 seeds และเปลี่ยนเฉพาะ LR/decay
ขั้นนี้เพิ่มเวลา/พื้นที่มาก กำหนดงบก่อนเริ่ม หากอ้าง equal tuning budget ต้องจัดงบทุก comparator เท่ากัน

## 5. External data

ให้จัด RDD2022 ที่มี Pascal VOC XMLs แล้วรัน (ไม่มี automatic download):

```bash
python run.py import-rdd2022 --root /path/RDD2022 --output /path/rdd_external --subset train
python run.py prepare --data-root /path/rdd_external --groups-csv /path/rdd_external/groups.csv \
  --output artifacts/external_rdd --external --source https://arxiv.org/abs/2209.08538
```

mapping D00/D10/D20→crack; D40→pothole ไม่มี manhole
ใช้ labeled subset เป็น independent external evaluation ของ source-only model
ไม่ใช่ official challenge test score; unlabeled challenge test ประเมิน AP ไม่ได้
importer ใช้ VOC 1-indexed inclusive → continuous zero-based bounds ตรวจ convention ของ export จริงก่อนใช้
country grouping ใช้ country-cluster bootstrap แต่ไม่ใช่ capture-session metadata
ภาพถนนไทยใช้ images/labels contract เดิมและระบุ provenance จริง

รายงาน `AP_shared_damage` ของ pothole+crack; source three-class AP ห้ามเทียบตรงกับ RDD two-class AP
threshold ตรึงจาก source validation ไม่ tune/fine-tune บน external test

## 6. Freeze แล้วเปิด final test

หลัง develop ครบ เลือก settings จาก validation และตรวจ metadata/similarity:

```bash
python run.py freeze-study --config configs/rtx4090.yaml \
  --similarity-audit artifacts/leakage/similarity_audit.json \
  --decisions-csv artifacts/leakage/review_pairs.csv \
  --external-manifest artifacts/external_rdd/manifest.json
```

ใช้ selected_config ถ้าผ่าน tuning และ audit ของ manifest ที่ตรงกัน
default paper protocol ต้องมี verified metadata และ cross-split candidates ที่ตัดสินแล้ว
หากหา metadata ไม่ได้ ให้ใช้ `--exploratory` ซึ่งบันทึกข้อจำกัดไว้และไม่รับรอง independence
confirmed same-session ที่ยังข้าม split ถูกปฏิเสธแม้ exploratory
หากยังไม่มี external data ไม่ต้องใส่ external-manifest; จะยังตอบ RQ4 ไม่ได้

freeze บันทึก immutable config/manifest/model/threshold hashes, comparisons, bootstrap budget
และ external manifest ก่อนเปิด test เป็น **local preregistration** ไม่ใช่ public registered report

```bash
bash scripts/finalize.sh --protocol artifacts/study/protocol.json \
  --external-data-root /path/rdd_external
```

ถ้าไม่ได้ register external ไม่ต้องส่ง external-data-root
finalize ตรวจ hashes → test → external → reports → paired group bootstrap
direct evaluate ของ paper study ต้องมี protocol:

```bash
python run.py evaluate --run-dir runs/paper_study/fasterrcnn_detail_seed0 \
  --split test --protocol artifacts/study/protocol.json
```

## 7. Paper outputs

- best.pt / last.pt / periodic checkpoints / best_weights.pt พร้อม config/model_spec
- histories, source/software provenance, per-image predictions และ COCO AP/AR summaries
- CSV per-run/per-class/mean-SD, shared damage AP, paired-seed deltas, source distribution, LaTeX table
- PNG 300 DPI / SVG training, PR, confusion, AP และ accuracy/latency plots
- Failure JSON: recall ตาม object area, aspect, brightness, contrast และ group
- Hard examples, missed objects, per-image FP/FN
- paper/statistical_comparisons.json พร้อม paired whole-group percentile CI
- paper/frozen_study.json บันทึก protocol และข้อจำกัด

AP ใช้ score floor .001/max 100; P/R/F1 ใช้ validation-frozen class thresholds ที่ IoU .5
latency batch 1 FP32 หลัง warmup/CUDA sync รวม decode/preprocess/transfer/model/merge/CPU output
confusion อนุญาต cross-class matching จึงอาจต่างจาก class-specific AP diagnostics
seed SD กับ conditional group CI เป็นความไม่แน่นอนคนละส่วน ไม่สร้าง p-value หรือ significance เทียม
กลุ่มน้อยทำให้ CI ไม่มั่นคง; class-incomplete bootstrap draws ถูกข้ามและนับ; absent class AP=null
brightness bins เป็น pixel statistics ไม่ใช่ weather labels

Tiling ทดลองบน validation ได้ด้วย `evaluate --split val --tiled`
frozen paper protocol ของ v2 ใช้ full-image evaluation เท่านั้น การเปลี่ยน inference protocol
ต้อง preregister แยกก่อน test ห้ามเลือก tiling จากคะแนน test

## 8. CPU checks / inference

```bash
python -m pytest -q
python run.py run --config configs/smoke_cpu.yaml --resume
python run.py predict --run-dir runs/paper_study/fasterrcnn_detail_seed0 \
  --image /path/road.jpg --output artifacts/prediction --device cuda
```

smoke ใช้ tiny backbone/ไม่กี่ภาพ ไม่ใช่ paper results
adapter tests ใช้ random weights และลด input/decoder เพื่อทำงาน offline
GPU/AMP/pretrained/full training ต้องทดสอบบน RTX 4090 ถ้า modern dependencies ไม่ครบ tests จะระบุ skip
GPU study ไม่เปลี่ยน missing detector เป็น dummy แบบเงียบ ๆ
การฝึกซ้ำ offline ต้องเก็บ Torch Hub/Hugging Face caches Ultralytics code/weights มี license ของผู้เผยแพร่

## สิ่งที่ต้องอาศัยข้อมูลจริง

โค้ดเพิ่มทั้ง 4 ส่วนแล้ว แต่ verified capture metadata, genuine external dataset,
ผลฝึกเต็มที่มีความหมาย และข้อยืนยัน novelty ไม่สามารถสร้างขึ้นแทนหลักฐานจริงได้
ตรวจ license ของข้อมูล/โมเดลก่อน redistribution และรายงานข้อจำกัดใน paper
