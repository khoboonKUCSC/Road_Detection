# สรุปภาพรวมและคู่มือวิธีการตรวจสอบงานวิจัย (Research Verification & Workflow Guide)

เอกสารนี้รวบรวม **สถานะปัจจุบันของโปรเจกต์**, **วิธีการตรวจสอบผลการทดลอง**, **การทดสอบความถูกต้องของระบบและโค้ด**, ตลอดจน **ขั้นตอนการดำเนินงานต่อสู่รอบ Final Test และตีพิมพ์ผลงาน (Research Protocol)**

---

## 1. สถานะปัจจุบันของงานในระบบ (Current Project Status)

ในโฟลเดอร์ [`runs/paper_study_recovered/`](file:///home/ds/Desktop/Road_Detection/research_v2/runs/paper_study_recovered) โมเดลทั้งหมด **8 สถาปัตยกรรม × 3 seeds (0, 42, 2026) รวม 24 รัน** ได้รับการฝึก (Training), ประเมินบน Validation Set และสร้างรายงานรอบพัฒนา ([`development_report`](file:///home/ds/Desktop/Road_Detection/research_v2/runs/paper_study_recovered/development_report)) เสร็จสมบูรณ์แล้ว 100%

### ตารางสรุปผลการทดลองรอบพัฒนา (Validation Set Summary: Mean ± SD)

| อันดับ | สถาปัตยกรรม (Experiment) | จำนวน Seeds | COCO AP (50:95) | AP50 | AP75 | Shared AP (Pothole+Crack) | Macro F1 | Latency (ms/img) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **`fasterrcnn_detail_geometry`** | 3 | **0.1542 ± 0.0033** | **0.3665 ± 0.0131** | **0.1012 ± 0.0132** | 0.0870 ± 0.0068 | 0.4184 ± 0.0147 | 21.44 ± 0.11 |
| **2** | **`fasterrcnn_detail`** | 3 | 0.1516 ± 0.0046 | 0.3598 ± 0.0016 | 0.0867 ± 0.0145 | **0.0894 ± 0.0089** | 0.4215 ± 0.0136 | 21.75 ± 0.23 |
| 3 | `fasterrcnn_baseline` | 3 | 0.1495 ± 0.0043 | 0.3512 ± 0.0222 | 0.0963 ± 0.0068 | 0.0788 ± 0.0056 | 0.4141 ± 0.0214 | 21.78 ± 0.75 |
| 4 | `fasterrcnn_geometry` | 3 | 0.1486 ± 0.0017 | 0.3492 ± 0.0273 | 0.0952 ± 0.0243 | 0.0724 ± 0.0044 | **0.4216 ± 0.0071** | 21.40 ± 0.11 |
| 5 | `fasterrcnn_aspect` | 3 | 0.1432 ± 0.0036 | 0.3424 ± 0.0186 | 0.0909 ± 0.0125 | 0.0782 ± 0.0089 | 0.4110 ± 0.0168 | 21.35 ± 0.20 |
| 6 | `fcos_baseline` | 3 | 0.1378 ± 0.0083 | 0.3236 ± 0.0175 | 0.0914 ± 0.0090 | 0.0620 ± 0.0085 | 0.3899 ± 0.0171 | 15.44 ± 0.23 |
| 7 | `yolo11s_baseline` | 3 | 0.0947 ± 0.0041 | 0.2098 ± 0.0043 | 0.0657 ± 0.0156 | 0.0187 ± 0.0041 | 0.2827 ± 0.0155 | **5.58 ± 0.47** |
| 8 | `rtdetr_v2_baseline` | 3 | 0.0884 ± 0.0026 | 0.2030 ± 0.0162 | 0.0552 ± 0.0051 | 0.0333 ± 0.0122 | 0.2872 ± 0.0275 | 21.75 ± 0.10 |

---

## 2. วิธีการตรวจสอบผลการทดลองรอบพัฒนา (How to Inspect Development Results)

รายงานสรุปทั้งหมดถูกจัดเก็บอยู่ในไดเรกทอรี [`runs/paper_study_recovered/development_report/`](file:///home/ds/Desktop/Road_Detection/research_v2/runs/paper_study_recovered/development_report)

### 2.1 ตรวจสอบตารางผลลัพธ์ผ่าน CSV
- **ผลการทดลองแยกรายโมเดลและราย Seed ครบ 24 รัน**:
  [`runs/paper_study_recovered/development_report/per_run.csv`](file:///home/ds/Desktop/Road_Detection/research_v2/runs/paper_study_recovered/development_report/per_run.csv)
- **ตารางสรุปสถิติค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐาน (Mean ± SD)**:
  [`runs/paper_study_recovered/development_report/summary_mean_std.csv`](file:///home/ds/Desktop/Road_Detection/research_v2/runs/paper_study_recovered/development_report/summary_mean_std.csv)
- **ผลการทดลองแยกตามคลาสความเสียหาย (Pothole, Crack, Manhole)**:
  [`runs/paper_study_recovered/development_report/per_class.csv`](file:///home/ds/Desktop/Road_Detection/research_v2/runs/paper_study_recovered/development_report/per_class.csv)
- **ความแตกต่างแบบจับคู่ Seed (Paired Seed Deltas)**:
  [`runs/paper_study_recovered/development_report/paired_seed_deltas.csv`](file:///home/ds/Desktop/Road_Detection/research_v2/runs/paper_study_recovered/development_report/paired_seed_deltas.csv)

### 2.2 ตรวจสอบกราฟและการวินิจฉัย (Visualizations & Diagnostics)
ในโฟลเดอร์รายงานมีไฟล์ภาพ PNG/SVG สำหรับแต่ละโมเดล:
1. **กราฟการเทรน (Training Loss & Validation AP ข้าม Epoch)**:
   - ตัวอย่าง: `{experiment}_seed{seed}_training.png`
2. **กราฟ Precision-Recall Curve แต่ละคลาส (IoU=0.5)**:
   - ตัวอย่าง: `{experiment}_seed{seed}_val_PR.png`
3. **เมทริกซ์ความสับสน (Confusion Matrix)**:
   - ตัวอย่าง: `{experiment}_seed{seed}_val_confusion.png`
4. **ตัวอย่างภาพที่ตรวจจับผิดพลาดและภาพยาก (Per-image Errors)**:
   - จัดเก็บในโฟลเดอร์ย่อยของแต่ละรัน: `{run_dir}/eval/val_examples/`
   - การวิเคราะห์ข้อผิดพลาดเชิงลึก: `{run_dir}/eval/val_failure_analysis.json`

### 2.3 สคริปต์คำสั่งดึงข้อมูลสรุปอย่างรวดเร็วผ่าน Terminal
```bash
# ตรวจสอบ 5 อันดับโมเดลที่มีค่า Validation AP สูงสุด
python3 -c "
import csv
with open('runs/paper_study_recovered/development_report/per_run.csv') as f:
    rows = sorted(csv.DictReader(f), key=lambda r: float(r['AP']), reverse=True)
for r in rows[:5]:
    print(f\"{r['run']:35s} | AP: {float(r['AP']):.4f} | AP50: {float(r['AP50']):.4f} | F1: {float(r['macro_F1']):.4f}\")
"
```

---

## 3. วิธีการตรวจสอบความสมบูรณ์ของระบบและโค้ด (System Integrity Verification)

### 3.1 การรัน Automated Test Suite ครบวงจร
โปรเจกต์มีชุดทดสอบครอบคลุมทั้ง Unit Tests, Model Reload, Loss Invariance, AMP Overflow Guard, FP32 Policy และ Migration Artifacts:
```bash
# รันการทดสอบทั้งหมด (23 ข้อ)
pytest tests/
```
*ผลลัพธ์ที่ถูกต้อง*: ต้องผ่านครบ `23 passed` (100% Pass)

### 3.2 ตรวจสอบสภาพแวดล้อม GPU และ CUDA Detection Operators
```bash
python run.py doctor --device cuda
```
*ผลลัพธ์*: ตรวจสอบเวอร์ชัน Python, PyTorch, Torchvision, CUDA Runtime, GPU Hardware (NVIDIA GeForce RTX 4090) และความพร้อมของ pycocotools

### 3.3 การตรวจสอบ Checkpoint Provenance และความถูกต้องของข้อมูล
แต่ละรันมีไฟล์รับรอง provenance ป้องกันการแก้ไขหรือสับเปลี่ยนข้อมูล:
- `initialization.json`: บันทึกจำนวน Parameters และ Architecture ดั้งเดิม
- `manifest.json`: เก็บสำเนาข้อมูลที่ใช้เทรนพร้อม SHA-256 fingerprint
- `source_sha256.json`: บันทึก SHA-256 ของซอร์สโค้ดทุกไฟล์ ณ เวลาเริ่มรัน
- `migration.json`: (สำหรับรันที่ migrate มา) บันทึกเส้นทางเดิมและสาเหตุการย้าย

---

## 4. ขั้นตอนการดำเนินงานต่อสู่รอบ Final Test (Research Protocol Workflow)

ตามระเบียบวิธีวิจัยของโปรเจกต์ ([`docs/research_protocol.md`](file:///home/ds/Desktop/Road_Detection/research_v2/docs/research_protocol.md)) หลังจากโมเดลผ่านรอบพัฒนาครบแล้ว หากต้องการเปิดประเมินบนชุดทดสอบจริง (Test Set) เพื่อเขียนเปเปอร์ ต้องทำตามขั้นตอน Preregistration ดังนี้:

### ขั้นตอนที่ 4.1: ตรวจสอบความซ้ำซ้อนของข้อมูล (Data Leakage Audit)
เพื่อป้องกันการรั่วไหลของภาพจากเส้นทางเดียวกันข้าม Train/Val/Test:
```bash
python run.py audit-similarity \
  --manifest artifacts/dataset/manifest.json \
  --output artifacts/leakage
```
คำสั่งนี้จะสร้าง `similarity_audit.json` และ `review_pairs.csv` เพื่อตรวจสอบคู่ภาพที่คล้ายคลึงกัน

### ขั้นตอนที่ 4.2: ตรึงโปรโตคอลการวิจัย (Freeze Study / Preregistration)
คำสั่งนี้จะคำนวณและบันทึก Immutable Checksums ของ Config, Manifest, โมเดล และ Thresholds ที่ได้จาก Validation ไว้ใน `protocol.json` ก่อนที่จะเปิดดู Test Set:
```bash
python run.py freeze-study \
  --config configs/rtx4090_recovered.yaml \
  --similarity-audit artifacts/leakage/similarity_audit.json \
  --output artifacts/study/protocol.json \
  --exploratory
```
*(หมายเหตุ: ใส่แฟล็ก `--exploratory` หากชุดข้อมูลใช้ date proxy split)*

### ขั้นตอนที่ 4.3: ปิดงานวิจัยและประเมินผลบน Test Set (Finalize Study)
ทำการประเมินผลบน Test Set ของทุกโมเดล พร้อมคำนวณนัยสำคัญทางสถิติ (Paired Group Bootstrap 1,000 repeats สำหรับ p-values และ 95% Confidence Intervals) ตลอดจนสร้างตาราง LaTeX สำหรับเปเปอร์:
```bash
bash scripts/finalize.sh --protocol artifacts/study/protocol.json
```
ผลลัพธ์ฉบับสมบูรณ์จะถูกบันทึกไว้ในโฟลเดอร์:
`runs/paper_study_recovered/paper/`

### ขั้นตอนที่ 4.4: (ทางเลือก) ประเมิน Test Set เฉพาะโมเดลเดี่ยว
```bash
python run.py evaluate \
  --run-dir runs/paper_study_recovered/fasterrcnn_detail_geometry_seed0 \
  --split test \
  --protocol artifacts/study/protocol.json
```

---

## 5. วิธีการนำโมเดลไปทดสอบตรวจจับภาพจริง (Inference / Prediction Walkthrough)

หากต้องการนำโมเดลที่ฝึกเสร็จแล้ว (`best.pt`) ไปทดสอบตรวจจับรอยชำรุดบนภาพถนนใหม่:

### 5.1 ตรวจจับแบบภาพเดี่ยวปกติ (Standard Inference)
```bash
python run.py predict \
  --run-dir runs/paper_study_recovered/fasterrcnn_detail_geometry_seed2026 \
  --image /path/to/road_damage_image.jpg \
  --output runs/prediction_output.jpg \
  --device cuda
```

### 5.2 ตรวจจับแบบแบ่งชิ้นภาพความละเอียดสูง (Tiled Inference with NMS)
กรณีภาพถนนมีความละเอียดสูงมากและต้องการตรวจจับรอยแตกขนาดเล็ก (Crack):
```bash
python run.py predict \
  --run-dir runs/paper_study_recovered/fasterrcnn_detail_geometry_seed2026 \
  --image /path/to/high_res_road.jpg \
  --output runs/prediction_tiled.jpg \
  --tiled \
  --device cuda
```

---

## 6. สรุปประวัติการแก้ไขปัญหาและการตรวจสอบย้อนกลับ (Bugfix Provenance)

โปรเจกต์มีการบันทึกรายงานการแก้ไขปัญหาและการสืบสวนย้อนหลังอย่างเป็นระบบไว้ในโฟลเดอร์ `bugfix/`:

1. **[`bugfix/1/`](file:///home/ds/Desktop/Road_Detection/research_v2/bugfix/1/) — AMP Non-finite Gradient Handling & Migration**:
   - ปัญหา: `clip_grad_norm_` แครชเมื่อเกิด non-finite gradients ใน Faster R-CNN
   - การแก้ไข: เพิ่ม Pre-clipping finite check, บันทึก `overflow_events.jsonl`, กลไก backoff ของ GradScaler และระบบ `migrate_run`
   - รายงาน: [`fasterrcnn_baseline_seed0_amp_fix_result.md`](file:///home/ds/Desktop/Road_Detection/research_v2/bugfix/1/fasterrcnn_baseline_seed0_amp_fix_result.md)
2. **[`bugfix/2/`](file:///home/ds/Desktop/Road_Detection/research_v2/bugfix/2/) — RT-DETRv2 FP16 NaN Boxes / Matcher Crash**:
   - ปัญหา: โครงสร้าง Deformable Attention & Head ของ RT-DETRv2 เกิด Numerical Instability ภายใต้ FP16 ทำนายได้กล่อง `NaN` จน Hungarian Matcher แครช
   - การแก้ไข: กำหนด `uses_fp32_forward = True`, รัน Forward ใน FP32 และเพิ่ม Defensive Diagnostics
   - รายงาน: [`rtdetr_v2_baseline_seed0_nan_boxes_fix_result.md`](file:///home/ds/Desktop/Road_Detection/research_v2/bugfix/2/rtdetr_v2_baseline_seed0_nan_boxes_fix_result.md)
3. **[`bugfix/3/`](file:///home/ds/Desktop/Road_Detection/research_v2/bugfix/3/) — Missing `initialization.json` in Migrated Run**:
   - ปัญหา: `migrate_run` ไม่ได้คัดลอก `initialization.json` ทำให้ขั้นตอน `report()` เกิด `FileNotFoundError`
   - การแก้ไข: เพิ่ม `initialization.json` ในรายการไฟล์ migrate, ปรับปรุง Error Reporting ให้ระบุบริบทของรัน และกู้คืนไฟล์ของ `fasterrcnn_baseline_seed0`
   - รายงาน: [`report_missing_initialization_fix_result.md`](file:///home/ds/Desktop/Road_Detection/research_v2/bugfix/3/report_missing_initialization_fix_result.md)
