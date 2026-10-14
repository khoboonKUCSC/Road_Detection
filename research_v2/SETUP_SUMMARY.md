# บันทึกผลการ Setup และคู่มือการเริ่มใช้งาน Road Damage Research v2

วันที่บันทึก: 13 ตุลาคม 2026  
สภาพแวดล้อมฮาร์ดแวร์: Ubuntu Linux / NVIDIA GeForce RTX 4090

---

## 1. ภาพรวมการ Setup (Setup Summary)

ระบบและสภาพแวดล้อมการทำงานของโปรเจกต์ `research_v2` ได้รับการติดตั้ง ทดสอบ และพร้อมใช้งาน 100% เรียบร้อยแล้ว

| รายการ | รายละเอียด | สถานะ |
|---|---|:---:|
| **Python Version** | Python 3.12.15 (สร้างผ่าน Anaconda base และแยก virtual environment) | พร้อมใช้งาน |
| **Virtual Environment** | `.venv/` ตั้งอยู่ที่ `research_v2/.venv` | พร้อมใช้งาน |
| **PyTorch & CUDA** | `torch==2.7.1+cu128`, `torchvision==0.22.1+cu128` | พร้อมใช้งาน |
| **Dependencies** | ติดตั้งครบทั้ง `requirements.txt` และ `requirements-modern.txt` | พร้อมใช้งาน |
| **Data Integrity** | ตรวจสอบภาพใน `data/images/` และกู้คืน `data/labels/` ครบ 1,999 ภาพ | ผ่านการตรวจสอบ |
| **Unit Tests** | รัน `pytest` ผ่านทั้งหมด 15/15 tests | ผ่าน |
| **Hardware Doctor** | `python run.py doctor --device cuda` ตรวจพบ RTX 4090 และ NMS/COCO evaluator | ผ่าน |
| **Pipeline Smoke Test** | รัน `configs/smoke_cpu.yaml` และคำสั่ง `predict` สำเร็จ | ผ่าน |

---

## 2. รายละเอียดการดำเนินการและปัญหาที่ได้รับการแก้ไข

1. **การจัดการเวอร์ชัน Python (`.venv`):**
   * ระบบเดิมมี Python ค่าเริ่มต้นเป็น 3.14 ซึ่งยังไม่มี pre-built wheel ที่รองรับ PyTorch 2.7.1
   * ดำเนินการสร้าง `.venv` โดยใช้ Python 3.12 ตามข้อกำหนดของเอกสาร `README.md`
   * ปรับปรุงสคริปต์ `scripts/setup_ubuntu.sh` ให้ตรวจจับและเลือก Python 3.12/3.11 โดยอัตโนมัติ และป้องกันการเขียนทับ `.venv` เดิมที่มีอยู่
2. **การกู้คืนชุด Label ต้นฉบับ (`data/labels/`):**
   * ตรวจสอบพบว่าโฟลเดอร์ `data/labels/` ซึ่งบรรจุ polygon annotations 9 ค่า (quadrilateral) ถูกลบไปในคอมมิตก่อนหน้า เหลือเพียง `labels-YOLO/` (5 ค่า) ทำให้ฟังก์ชัน `verify_manifest()` ตรวจสอบ SHA256 checksum ไม่ผ่าน
   * ดำเนินการดึงไฟล์ต้นฉบับจาก git commit ย้อนหลังกลับมาไว้ที่ `data/labels/` ทำให้ SHA256 ตรงตาม `artifacts/dataset/manifest.json` ทุกประการ
3. **การล็อกเวอร์ชันแพ็กเกจ:**
   * สภาพแวดล้อมทั้งหมดถูกบันทึกไว้ใน `environment.lock.txt` เพื่อความ reproducible ของการวิจัย

---

## 3. วิธีเริ่มใช้งานโปรเจกต์ (Getting Started)

### ขั้นตอนที่ 1: เข้าสู่โฟลเดอร์และเปิดใช้งาน Virtual Environment

```bash
cd /home/ds/Desktop/Road_Detection/research_v2
source .venv/bin/activate
```

### ขั้นตอนที่ 2: ตรวจสอบความพร้อมของระบบ (Doctor Check)

```bash
# ตรวจสอบการทำงานของ PyTorch, CUDA, NMS Ops และ pycocotools บน GPU
python run.py doctor --device cuda
```

### ขั้นตอนที่ 3: ตรวจสอบความคล้ายคลึงของข้อมูล (Data Leakage & Similarity Audit)

```bash
python run.py audit-similarity
```
* คำสั่งนี้จะสร้างไฟล์ `artifacts/leakage/similarity_audit.json`, `review_pairs.csv` และ contact sheets เพื่อตรวจสอบภาพที่อาจข้าม split หรือซ้ำซ้อนกัน

### ขั้นตอนที่ 4: การรันงานวิจัยหลัก (Development Study - 24 Runs บน RTX 4090)

```bash
python run.py develop --config configs/rtx4090.yaml --resume
```
* **รายละเอียด:** จะฝึกโมเดลทั้งหมด 8 configurations × 3 seeds (`[0, 42, 2026]`) รวม 24 runs
* **การบันทึกผล:** บันทึก checkpoints (`best.pt`, `last.pt`), training history, RNG state และ validation reports ไว้ที่ `runs/paper_study/`
* รองรับการ resume อัตโนมัติจาก epoch ล่าสุดหากหยุดการทำงานกลางคัน

### ขั้นตอนที่ 5: การรัน Hyperparameter Tuning (ทางเลือกเพิ่มเติม)

```bash
python run.py tune --config configs/rtx4090.yaml --grid configs/tuning.yaml --resume
python run.py develop --config runs/tuning/selected_config.yaml --resume
```

### ขั้นตอนที่ 6: การ Freeze Study และเปิดประเมินผล Final Test

เมื่อฝึกโมเดลและตรวจสอบผล validation เรียบร้อยแล้ว ให้ทำการ freeze protocol ก่อนเปิดประเมินผลบนชุด test:

```bash
python run.py freeze-study --config configs/rtx4090.yaml \
  --similarity-audit artifacts/leakage/similarity_audit.json \
  --decisions-csv artifacts/leakage/review_pairs.csv \
  --exploratory
```

จากนั้นรันการประเมินผลสรุปผลงานวิจัย:

```bash
bash scripts/finalize.sh --protocol artifacts/study/protocol.json
```

### ขั้นตอนที่ 7: การนำโมเดลไปทดสอบทำนายภาพเดี่ยว (Inference / Predict)

```bash
python run.py predict \
  --run-dir runs/paper_study/fasterrcnn_detail_seed0 \
  --image /path/to/road.jpg \
  --output artifacts/prediction \
  --device cuda
```

---

## 4. โครงสร้างไฟล์และโมเดลที่สำคัญ

* `configs/rtx4090.yaml`: การตั้งค่าหลักสำหรับการทดลอง 24 runs บน GPU
* `configs/smoke_cpu.yaml`: การตั้งค่าสำหรับการทดสอบระบบแบบรวดเร็วบน CPU
* `artifacts/dataset/manifest.json`: รายการข้อมูล 1,999 ภาพ / 4,701 annotations พร้อม split assignments
* `roadresearch/`: ซอร์สโค้ดหลักของระบบ (Data loaders, Engine, Models, Metrics, Protocols)
* `runs/`: โฟลเดอร์เก็บผลลัพธ์ checkpoint, logs และกราฟการประเมินผล
