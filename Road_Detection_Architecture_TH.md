# สรุปสถาปัตยกรรมโปรเจกต์ Road Detection

เอกสารนี้อธิบายสถาปัตยกรรมของโปรเจกต์ **Road_Detection** (ตรวจจับความเสียหายบนถนน: หลุมบ่อ / ร้าว / ฝาท่อ) ทั้งหมด

---

## 1. ภาพรวม (Overview)

โปรเจกต์เป็นงานวิจัย **Object Detection** สำหรับจับ damage บนผิวถนน 3 คลาส:

| ID | คลาส | ความหมาย |
|----|------|-----------|
| 0 | pothole | หลุมบ่อ |
| 1 | crack | ร้าว |
| 2 | manhole | ฝาท่อ / ฝาปิดท่อ |

จุดสำคัญของสถาปัตยกรรมคือ **แบ่งเป็น 2 ชั้น (two-layer)** ที่แยกบทบาทกันชัดเจน:

```
Road_Detection/
├── (ชั้นที่ 1) โฟลเดอร์ระดับบนสุด + model/   → งานสำรวจแบบ Notebook (Exploratory / YOLO)
└── (ชั้นที่ 2) research_v2/                  → แพ็กเกจ Python แบบ reproduce ได้ (งานวิจัยจริง)
```

---

## 2. ชั้นที่ 1 — งานสำรวจ (Exploratory Layer)

อยู่ที่ root ของ repo และโฟลเดอร์ `model/` เป็นชุด **Jupyter Notebook** สำหรับลองของเร็ว ๆ:

- **`model/`** — notebook ทดลองวนหลายเวอร์ชัน: `v1-yolo11n` → `v10-yolo11s`, `EDA.ipynb` (สำรวจข้อมูล), `pic+label.ipynb` (ดูภาพ+label)
- **weights ตระกูล YOLO11** — `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt` (n=nano, s=small, m=medium)
- **`data/`** — ภาพต้นฉบับ + label รูปแบบ YOLO และสคริปต์แปลงรูปแบบ:
  - `YOLO-conversion-script.py`, `COCO-conversion-script.py` (แปลงระหว่าง YOLO ⇄ COCO)
  - `annotate_dataset.py` (ช่วย annotate)
  - `images/`, `labels-YOLO/`, `image-label/`, `crack-label/`, `data_crack_filter/`
- **`predic-pic/`, `png/`, `runs/`** — ผลลัพธ์การทำนายและภาพที่ preview

> ชั้นนี้เป็น "**research v1**" — ยังไม่ reproduce ง่าย แต่ใช้สำรวจและปั้นโมเดลเร็ว

---

## 3. ชั้นที่ 2 — `research_v2` (Reproducible Research Layer)

หัวใจของสถาปัตยกรรมจริง เป็น **แพ็กเกจ Python ชื่อ `roadresearch`** ที่ออกแบบมาให้รันซ้ำได้แน่นอน (reproducible) และเปรียบเทียบโมเดลได้อย่างเป็นธรรม

### 3.1 โครงสร้างโฟลเดอร์

```
research_v2/
├── run.py                     # entry point เดียว → roadresearch.cli.main()
├── roadresearch/              # แพ็กเกจหลัก (โมดูลแยกตามหน้าที่)
│   ├── cli.py                 # สั่งงานทุกอย่างผ่าน subcommand
│   ├── data.py                # อ่าน/ตรวจ/แปลง dataset + manifest
│   ├── models.py              # สร้างโมเดล torchvision (Faster R-CNN, FCOS) + GeometryDetector
│   ├── adapters.py            # adapter รวม YOLO11 และ RT-DETRv2 ให้ contract เดียวกัน
│   ├── engine.py              # training loop / evaluate / predict / tiling
│   ├── metrics.py             # COCO mAP (pycocotools) + confusion + bootstrap
│   ├── leakage.py             # audit ข้อมูลรั่ว (similarity / near-duplicate)
│   ├── protocol.py            # freeze/finalize study (preregistration แบบ local)
│   ├── external.py            # import ข้อมูลภายนอก (RDD2022)
│   ├── tuning.py              # grid search LR แบบ equal-budget
│   ├── report.py              # สร้างรายงาน/ตาราง/กราฟสรุป
│   ├── diagnostics.py         # เครื่องมือตรวจโรคโมเดล
│   └── utils.py               # seed, checkpoint atomic, hashing, CLASSES
├── configs/
│   ├── rtx4090.yaml           # config หลักสำหรับ GPU (งานวิจัยจริง)
│   ├── smoke_cpu.yaml         # smoke test เล็ก ๆ บน CPU
│   ├── smoke_registered.yaml
│   └── tuning.yaml            # grid สำหรับ tune LR
├── scripts/
│   ├── setup_ubuntu.sh        # ติดตั้ง env (pin Torch 2.7.1 / cu128)
│   ├── run_all.sh             # doctor → audit → develop
│   └── finalize.sh            # ขั้นตอนปิดท้าย protocol
├── tests/                     # pytest (ทำงาน offline ด้วย random weights)
├── docs/                      # related_work_matrix.csv, research_protocol.md
├── artifacts/                 # manifest, audit, leakage, protocol ที่ freeze แล้ว
└── runs/                      # ผลลัพธ์การฝึก (checkpoints, histories, predictions)
```

### 3.2 จุดออกแบบสำคัญ (Key Architectural Decisions)

#### (ก) Adapter Pattern — รวม 4 ตระกูลโมเดล ให้ใช้ contract เดียวกัน

`adapters.py` + `models.py` ห่อโมเดลต่างสายพันธุ์ให้**รับ input / คืน output แบบเดียวกัน** เพื่อให้เปรียบเทียบกันได้ยุติธรรม:

| โมเดล | ที่มา | วิธี integrate |
|-------|--------|----------------|
| Faster R-CNN R50-FPN v2 | torchvision | โดยตรง + optional geometry head |
| FCOS R50-FPN | torchvision | โดยตรง (baseline) |
| YOLO11s | Ultralytics | ผ่าน `YOLODetector` adapter |
| RT-DETRv2 R50-vd | HuggingFace | ผ่าน `RTDETRv2Detector` adapter |

ทุกโมเดล:
- **รับ** `images, targets` (boxes/labels format เดียวกัน)
- **คืน** dict losses (ตอน train) หรือ `{"boxes","scores","labels"}` (ตอน eval)
- dependency ของ library ภายนอก (ultralytics/transformers) เป็น **import แบบ lazy** เพื่อให้ CPU smoke test รัน offline ได้

> หลักการ: **ไม่ใช้ native metric ของแต่ละโมเดลมาเทียบกัน** (คนละสูตร) แต่ส่ง predictions ทุกตัวเข้า **COCO evaluator เดียวกัน (pycocotools)**

#### (ข) GeometryDetector — auxiliary supervision เฉพาะ Faster R-CNN

`GeometryDetector` (ใน `models.py`) เป็น wrapper ที่เพิ่ม **head ทำนายรูปร่าง crack** (aspect ratio / orientation) ตอน train เท่านั้น:
- `geometry_mode`: `none` / `aspect` ( elongation ) / `full` (orientation เพิ่ม)
- orientation แทนด้วย `sin(2θ), cos(2θ)` (mod π)
- ใช้ anisotropy ถ่วงน้ำหนัก orientation ที่เชื่อถือไม่ได้
- inference ไม่เปลี่ยน — บังคับเฉพาะตอนฝึก (auxiliary loss)

#### (ค) ปริมาณ experiment: 8 configs × 3 seeds = 24 runs

```
seeds: [0, 42, 2026]

experiments:
  fasterrcnn_baseline        ← baseline
  fasterrcnn_aspect          ← + elongation aux loss
  fasterrcnn_geometry        ← + elongation/orientation aux losses
  fasterrcnn_detail          ← + crack-centered crop augmentation
  fasterrcnn_detail_geometry ← crop + geometry
  fcos_baseline
  yolo11s_baseline
  rtdetr_v2_baseline         ← modern transformer comparator
```

config-driven ทั้งหมดผ่าน YAML (`rtx4090.yaml`) — เปลี่ยน experiment โดยไม่ต้องแก้โค้ด

#### (ง) Data Pipeline & Integrity (`data.py`)

- รองรับ label รูปแบบ YOLO ทั้ง **AABB (5 ค่า)** และ **quadrilateral 4 จุด (9 ค่า)**
- ตรวจ label เสีย (nonfinite, class ผิด, box นอกภาพ, quad ไม่ convex) → clip/disable geometry ตามนโยบาย พร้อม **audit log**
- สร้าง **manifest** (เช่น 1,999 ภาพ / 4,701 annotations) และ **group split** กัน leakage
- augmentation พิเศษ: `crack_crop_probability` (ครอปรอบ crack ตอน train)

#### (ค') Leakage Auditing (`leakage.py`)

- ใช้ **pHash/dHash** หา near-duplicate ข้าม split (candidate leakage)
- ให้ human review กรอก decision: `same_session` / `different_scene` / `uncertain`
- ยืนยันแล้วค่อย merge/re-split —  code ไม่เดา metadata ปลอม

#### (ง') Protocol / Preregistration (`protocol.py`)

Workflow แบบ "**freeze ก่อนดู test**" กัน data snooping:

```
develop (train + val)  →  freeze-study (lock hashes/config/threshold)  →  finalize-study (เปิด test)
```

- `freeze-study` เก็บ config/manifest/model/threshold hash แบบ immutable = **local preregistration**
- บังคับ verified metadata + cross-split decisions ที่ตัดสินแล้ว (หรือ `--exploratory` พร้อมบันทึกข้อจำกัด)
- config `require_registration: true` ทำให้ `run` ตรง ๆ ถูกปฏิเสธ ต้องผ่าน protocol

#### (ง'') Evaluation & Statistics (`metrics.py`)

- **COCO AP/AR** ด้วย pycocotools (score floor .001, max 100 detections)
- P/R/F1 ใช้ class threshold ที่ freeze จาก validation (IoU .5)
- **paired group bootstrap** ทำ CI (ไม่สร้าง p-value ปลอม)
- วัด **end-to-end latency** จริง (batch 1, หลัง warmup + CUDA sync)
- รายงาน failure ตาม object area / aspect / brightness

### 3.3 ลำดับการใช้งาน (Pipeline Flow)

```bash
# 0. ติดตั้ง
bash scripts/setup_ubuntu.sh

# 1. เตรียมข้อมูล + audit
python run.py prepare --output artifacts/dataset
python run.py audit-similarity

# 2. develop: ฝึก 24 runs (validation เท่านั้น)
bash scripts/run_all.sh          # doctor → audit → develop

# 3. (optional) tune LR แบบ equal-budget
python run.py tune --config configs/rtx4090.yaml --grid configs/tuning.yaml

# 4. freeze protocol (preregistration)
python run.py freeze-study --config configs/rtx4090.yaml

# 5. finalize: เปิด test + สร้างรายงาน
bash scripts/finalize.sh --protocol artifacts/study/protocol.json

# 6. inference ภาพเดียว
python run.py predict --run-dir runs/paper_study/<run> --image road.jpg --output out/
```

### 3.4 หลักการออกแบบที่ซ้ำได้ (Reproducibility Guarantees)

ทุก run เก็บ:
- checkpoint `best/last/periodic` + optimizer/scheduler/AMP scaler states
- RNG + DataLoader states, source snapshot + hash
- config, environment, histories, per-image predictions

**Resume** ได้จาก epoch สุดท้าย แต่ต้องเป็น code/config/manifest เดิม — ถ้าเปลี่ยนสูตรต้องใช้ `output_root` ใหม่

---

## 4. ผังข้อมูล (Data Flow)

```
data/images + labels(YOLO) 
        │
        ▼
   data.py (parse + audit + manifest + group split)
        │
        ▼
   RoadDataset / DataLoader ──► engine.train()
        │                             │
        │                             ▼
        │                    models.py / adapters.py
        │                    (FasterRCNN / FCOS / YOLO / RT-DETR)
        │                             │
        ▼                             ▼
   engine.evaluate()  ◄──── predictions (boxes/scores/labels)
        │
        ▼
   metrics.py (COCO mAP + bootstrap)
        │
        ▼
   report.py (ตาราง LaTeX, กราฟ PNG/SVG, failure JSON)
        │
        ▼
   runs/paper_study/  +  artifacts/paper/
```

---

## 5. สรุปจุดแข็งของสถาปัตยกรรม

1. **แยกชั้น explored vs reproducible** — notebook ไว้ลอง, แพ็กเกจไว้สรุปงานวิจัย
2. **Adapter pattern** ทำให้เปรียบเทียบ 4 ตระกูลโมเดลภายใต้ evaluator เดียวกันได้ยุติธรรม
3. **Config-driven** ทุก experiment ผ่าน YAML เปลี่ยนได้โดยไม่แตะโค้ด
4. **Data integrity เป็นระบบ** — audit, leakage check, protocol freeze ป้องกัน data snooping
5. **Reproducible จริง** — seed, checkpoint states, source hash, resume ที่ผูกกับ config เดิม
6. **Dependency แบบ lazy** — ให้ CPU test รัน offline ได้ ไม่บังคับมี GPU

---

*เอกสารสรุปนี้จัดทำจากโครงสร้างโค้ดจริง ณ วันที่ 13 ต.ค. 2026*
