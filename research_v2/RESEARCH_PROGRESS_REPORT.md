# รายงานความก้าวหน้าผลการดำเนินงานวิจัยและการแก้ไขปัญหาเชิงระบบ
## โครงการ: การตรวจจับและจำแนกความเสียหายของพื้นผิวถนนด้วยการเรียนรู้เชิงลึก (Road Damage Detection using Deep Learning)

---

**ผู้จัดทำ**: นิสิต/นักศึกษาผู้วิจัย  
**หัวข้อวิจัย**: การเปรียบเทียบและเพิ่มประสิทธิภาพโมเดลตรวจจับวัตถุสำหรับงานตรวจจับความเสียหายพื้นผิวถนน (Pothole, Crack, Manhole)  
**วันที่รายงาน**: 15 ตุลาคม 2026  
**สภาพแวดล้อมที่ใช้ในการวิจัย**: Ubuntu Linux, Python 3.12.15, PyTorch 2.7.1+cu128, NVIDIA GeForce RTX 4090 (24GB VRAM)

---

## 1. บทคัดย่อและภาพรวมโครงการ (Executive Summary)

โครงการวิจัยนี้มีเป้าหมายเพื่อศึกษาและพัฒนาโมเดลปัญญาประดิษฐ์สำหรับการตรวจจับและจำแนกความเสียหายบนพื้นผิวถนน ประกอบด้วย 3 คลาสหลัก:
1. **Pothole (หลุมบ่อ)**
2. **Crack (รอยแตกร้าวของถนน)**
3. **Manhole (ฝาท่อ/ความไม่สม่ำเสมอของระดับผิวทาง)**

งานวิจัยนี้มุ่งเน้นการทดลองที่ได้มาตรฐานวิชาการระดับสูง (Rigorous Scientific Protocol) โดยดำเนินการ:
- **ป้องกันปัญหา Data Leakage**: ใช้ Grouped Date Split และการตรวจสอบความคล้ายคลึงของภาพข้ามชุดข้อมูล (Similarity Audit)
- **การลงทะเบียนโปรโตคอลล่วงหน้า (Preregistration / Study Freeze)**: ตรึง Immutable Checksums ของโมเดล คอนฟิก และเกณฑ์การตัดสินใจ (Thresholds) ไว้ใน `protocol.json` ก่อนประเมินผลบนชุดทดสอบจริง (Test Set)
- **การทดลองซ้ำ 3 Seeds (0, 42, 2026)**: สำหรับทั้ง 8 สถาปัตยกรรมโมเดล รวมทั้งสิ้น **24 รันการทดลอง** เพื่อวัดความสามารถในการทำซ้ำ (Reproducibility) และความเสถียรของผลลัพธ์
- **การทดสอบนัยสำคัญทางสถิติ**: ใช้ Paired Group Bootstrap (1,000 repeats) พร้อมคำนวณช่วงความเชื่อมั่น 95% Confidence Intervals (95% CI)

---

## 2. สถาปัตยกรรมโมเดลที่ทำการศึกษาเปรียบเทียบ (Model Architectures)

โครงการได้ทำการเปรียบเทียบโมเดลทั้งหมด 8 รูปแบบ:
1. **`fasterrcnn_baseline`**: Faster R-CNN (ResNet-50-FPN V2) โครงสร้างมาตรฐานแบบ Two-stage
2. **`fasterrcnn_aspect`**: Faster R-CNN เสริมด้วย Auxiliary Head ทำนาย Aspect Ratio ของวัตถุ
3. **`fasterrcnn_geometry`**: Faster R-CNN เสริมด้วย Auxiliary Head ทำนายคุณลักษณะเชิงเรขาคณิต (Orientation Angle & Aspect Ratio)
4. **`fasterrcnn_detail`**: Faster R-CNN ร่วมกับกลยุทธ์การสุ่มตัดชิ้นภาพขยายเฉพาะรอยแตกร้าวขนาดเล็ก (Detail Crack Cropping)
5. **`fasterrcnn_detail_geometry`**: Faster R-CNN ที่ผสานทั้ง Detail Crack Cropping และ Geometric Loss เข้าด้วยกัน
6. **`fcos_baseline`**: Fully Convolutional One-Stage Object Detector (FCOS with ResNet-50-FPN) แบบ Anchor-free
7. **`yolo11s_baseline`**: Ultralytics YOLO11s โมเดล One-stage น้ำหนักเบาความเร็วสูง
8. **`rtdetr_v2_baseline`**: Real-time Detection Transformer v2 (RT-DETRv2-R50vd) สถาปัตยกรรม Transformer

---

## 3. สรุปผลการทดลองและการเปรียบเทียบประสิทธิภาพ (Experimental Results)

### 3.1 ผลการทดสอบบนชุดข้อมูลทดสอบจริง (Final Test Set Evaluation)
ประเมินผลบนชุด Test Set ที่ถูกล็อกไว้ด้วย Validation Thresholds ครบทั้ง 24 โมเดล (8 สถาปัตยกรรม × 3 seeds):

| อันดับ | สถาปัตยกรรมโมเดล | Test COCO AP (Mean ± SD) | AP50 (Mean ± SD) | AP75 (Mean ± SD) | Macro F1 | Latency (ms) | จุดเด่นและข้อสังเกตเชิงวิจัย |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **`fasterrcnn_detail`** | **0.2085 ± 0.0116** | **0.4382 ± 0.0198** | 0.1844 ± 0.0199 | **0.4497** | 17.93 | **ชนะเลิศ Overall AP**, AP50 และการตรวจจับความเสียหายร่วม (Shared AP = 0.1068) |
| 🥈 | **`fasterrcnn_detail_geometry`** | **0.2077 ± 0.0046** | 0.4307 ± 0.0098 | **0.1956 ± 0.0041** | 0.4321 | 18.45 | **อันดับ 1 High-IoU (AP75)** และผลลัพธ์มีความเสถียรสูงสุด (SD ต่ำสุดเพียง ±0.0046) |
| 3 | `fasterrcnn_aspect` | 0.1984 ± 0.0163 | 0.4215 ± 0.0326 | 0.1746 ± 0.0266 | 0.4328 | 18.05 | พัฒนาจาก Baseline ได้ดี |
| 4 | `fasterrcnn_geometry` | 0.1949 ± 0.0146 | 0.4121 ± 0.0468 | 0.1708 ± 0.0080 | 0.4448 | 17.88 | Macro F1 โดดเด่น |
| 5 | `fasterrcnn_baseline` | 0.1929 ± 0.0092 | 0.4063 ± 0.0205 | 0.1616 ± 0.0161 | 0.4383 | 18.01 | มาตรฐาน Faster R-CNN |
| 6 | `fcos_baseline` | 0.1641 ± 0.0070 | 0.3678 ± 0.0237 | 0.1211 ± 0.0057 | 0.3994 | 14.84 | Anchor-free ทางเลือก ประสิทธิภาพปานกลาง |
| 7 | `yolo11s_baseline` | 0.1146 ± 0.0066 | 0.2316 ± 0.0120 | 0.1084 ± 0.0103 | 0.2617 | **4.65** | **ความเร็วสูงสุด (~215 FPS)** เหมาะสำหรับ Edge Hardware |
| 8 | `rtdetr_v2_baseline` | 0.1095 ± 0.0206 | 0.2297 ± 0.0425 | 0.0916 ± 0.0273 | 0.2746 | 18.05 | Transformer Detector ความแม่นยำต่ำกว่าเมื่อใช้ข้อมูลเฉพาะกลุ่ม |

### 3.2 ผลการทดสอบบนชุดตรวจสอบพัฒนาการ (Validation Set Development Results)
| โมเดล | Val COCO AP (Mean ± SD) | Val AP50 | Val AP75 | Val Macro F1 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `fasterrcnn_detail_geometry` | **0.1542 ± 0.0033** | **0.3665** | **0.1012** | 0.4184 | 21.44 |
| `fasterrcnn_detail` | **0.1516 ± 0.0046** | 0.3582 | 0.0984 | **0.4215** | 21.75 |
| `fasterrcnn_baseline` | 0.1495 ± 0.0043 | 0.3512 | 0.0954 | 0.4190 | 21.78 |
| `fasterrcnn_geometry` | 0.1486 ± 0.0017 | 0.3542 | 0.0970 | **0.4216** | 21.40 |
| `fasterrcnn_aspect` | 0.1432 ± 0.0036 | 0.3424 | 0.0921 | 0.4168 | 21.35 |
| `fcos_baseline` | 0.1378 ± 0.0083 | 0.3236 | 0.0886 | 0.4042 | 15.44 |
| `yolo11s_baseline` | 0.0947 ± 0.0041 | 0.1982 | 0.0612 | 0.2541 | **5.58** |
| `rtdetr_v2_baseline` | 0.0884 ± 0.0026 | 0.2030 | 0.0487 | 0.2589 | 21.75 |

---

## 4. ข้อค้นพบและการวิเคราะห์เชิงวิชาการ (Key Research Insights)

1. **ประสิทธิผลของ Detail Crack Cropping**:
   - การสุ่มตัดขยายภาพบริเวณรอยแตกร้าวขนาดเล็ก (Crack) ช่วยให้โมเดลเรียนรู้โครงสร้างรอยแตกที่กลมกลืนกับพื้นผิวแอสฟัลต์ได้ดีขึ้นอย่างมีนัยสำคัญ ส่งผลให้ `fasterrcnn_detail` ได้ค่า Test COCO AP สูงที่สุด (0.2085) และเพิ่ม AP50 ขึ้น +3.19% เทียบกับ Baseline
2. **บทบาทของ Geometric Auxiliary Head ต่อความแม่นยำตำแหน่ง (Localization)**:
   - โมเดล `fasterrcnn_detail_geometry` ทำคะแนนในระดับ IoU เข้มงวด (**AP75 = 0.1956**) ได้สูงที่สุด เหนือกว่าทุกโมเดลอย่างชัดเจน และมีส่วนเบี่ยงเบนมาตรฐานต่ำสุด (±0.0046) ชี้ให้เห็นว่าการบังคับให้โครงข่ายเรียนรู้ทิศทางและสัดส่วนรูปทรง ช่วยให้กรอบตรวจจับ (Bounding Box) มีความแม่นยำแนบสนิทกับขอบเขตความเสียหายจริงมากขึ้น
3. **การประเมินด้านความเร็วและการนำไปใช้งานจริง (Inference Latency vs Accuracy)**:
   - สำหรับงานที่ต้องการความแม่นยำสูงสุด: แนะนำ **`fasterrcnn_detail`** หรือ **`fasterrcnn_detail_geometry`** (ประมวลผล ~18 ms หรือ ~55 FPS)
   - สำหรับงานประมวลผลบนรถสำรวจแบบ Real-time บนอุปกรณ์ Edge Device: **`yolo11s_baseline`** มีความเร็วสูงถึง **4.65 ms (~215 FPS)** แต่ต้องยอมรับความแม่นยำที่ลดลง

---

## 5. การวิเคราะห์นัยสำคัญทางสถิติ (Statistical Significance Analysis)

เพื่อยืนยันว่าความแตกต่างของผลการทดลองไม่ได้เกิดจากความบังเอิญของชุดข้อมูล ได้ดำเนินการ **Paired Group Bootstrap (1,000 repeats)** บนระดับกลุ่มภาพ (Capture-proxy Groups) ตามโปรโตคอลการวิจัย:

- **`fasterrcnn_detail` vs `fasterrcnn_baseline`**:
  - Seed 0: ค่าผลต่างเฉลี่ย $\Delta\text{AP} = +0.0381$ (95% CI: $[+0.0052, +0.0806]$) — *แสดงการพัฒนาขึ้นอย่างมีนัยสำคัญทางสถิติ (ช่วง CI ไม่ครอบคลุมค่า 0)*
  - Seed 2026: ค่าผลต่างเฉลี่ย $\Delta\text{AP} = +0.0157$ (95% CI: $[+0.0076, +0.0262]$) — *แสดงการพัฒนาขึ้นอย่างมีนัยสำคัญทางสถิติ*
- **`fasterrcnn_detail_geometry` vs `fasterrcnn_detail`**:
  - แสดงให้เห็นถึงการแลกเปลี่ยน (Trade-off) ระหว่าง AP รวมกว้าง ๆ และ AP ที่ระดับ IoU เข้มงวด (AP75) โดยโมเดลผสาน Geometry ช่วยควบคุมความเสถียรข้าม Seed ให้คงที่กว่า

---

## 6. สรุปประวัติการแก้ไขปัญหาและการปรับปรุงระบบอย่างเป็นระบบ (Bugfix & System Hardening)

ในระหว่างการพัฒนาและทดลอง ผู้วิจัยได้พบปัญหาทางเทคนิคเชิงลึกและได้ทำการสืบสวนและแก้ไขอย่างเป็นระบบ บันทึกรายงานการสืบสวนและผลการทดสอบไว้ในโฟลเดอร์ `bugfix/`:

### 6.1 [Bugfix 1: AMP Non-finite Gradient Handling & Migration](bugfix/1/)
- **ปัญหา**: ขณะเทรนโมเดล Faster R-CNN ภายใต้ FP16 Automatic Mixed Precision (AMP) เกิดค่า Gradient ที่เป็น `inf/NaN` ขึ้นในบาง Step ส่งผลให้คำสั่ง `torch.nn.utils.clip_grad_norm_` เกิดความล้มเหลว
- **การแก้ไข**:
  1. เพิ่มฟังก์ชัน Pre-clipping finite check ตรวจจับค่า non-finite ก่อนคำนวณ norm
  2. เพิ่มระบบบันทึก `overflow_events.jsonl` พร้อมเก็บบันทึก `image_ids` และพารามิเตอร์ที่เกิดปัญหา
  3. บูรณาการกลไก Gradient Scaler backoff ข้าม optimizer step ที่มีปัญหาอย่างปลอดภัย
  4. สร้างระบบ `migrate_run()` เพื่อโยกย้ายและกู้คืนสถานะรันได้อย่างถูกต้องตามหลัก Provenance

### 6.2 [Bugfix 2: RT-DETRv2 FP16 Numerical Instability (NaN Boxes)](bugfix/2/)
- **ปัญหา**: โครงสร้าง Deformable Attention ใน Decoder ของโมเดล RT-DETRv2 มีความไวต่อค่าตัวเลขทศนิยมต่ำ (Numerical Instability) เมื่อรัน Forward pass ด้วย FP16 ส่งผลให้ทำนายพิกัดกล่องออกมาเป็น `NaN` และทำให้ Hungarian Matcher ล้มเหลวขณะคำนวณ Generalized IoU (GIoU)
- **การแก้ไข**:
  1. กำหนดคุณลักษณะ `uses_fp32_forward = True` และ `autocast_supported = False` ใน `RTDETRv2Detector`
  2. ครอบการทำงาน Forward pass ด้วย `torch.autocast(device_type, enabled=False)` เพื่อรับประกันการคำนวณใน FP32
  3. เพิ่ม Defensive Diagnostics ตรวจสอบความถูกต้องของพิกัดและค่า Loss อย่างเข้มงวด

### 6.3 [Bugfix 3: Missing `initialization.json` in Migrated Run](bugfix/3/)
- **ปัญหา**: ฟังก์ชัน `migrate_run()` ขาดการคัดลอกไฟล์ `initialization.json` ทำให้เมื่อรันสคริปต์ `report()` สรุปผล เกิดข้อผิดพลาด `FileNotFoundError`
- **การแก้ไข**:
  1. เพิ่ม `initialization.json` เข้าไปในรายการอาร์ติแฟกต์ที่ต้องสำเนาของการย้ายรัน
  2. ปรับปรุง Error Handling ใน `roadresearch/report.py` ให้แจ้งข้อผิดพลาดเชิงบริบท (Contextual Error) ระบุเส้นทางรันต้นทางและปลายทางอย่างชัดเจน
  3. ทำการกู้คืนไฟล์ metadata ที่ถูกต้องให้กับรัน `fasterrcnn_baseline_seed0`

### 6.4 [Bugfix 4: Freeze Study Migration-Aware Config Match](bugfix/4/)
- **ปัญหา**: คำสั่ง `freeze-study` ทำการเปรียบเทียบ Config ของรันแบบ strict equality ทุกคีย์ ส่งผลให้ปฏิเสธรันที่ผ่านการ migrate (`fasterrcnn_baseline_seed0`) เนื่องจากมีค่า `output_root` ต่างจาก Config ปัจจุบันเพียงฟิลด์เดียว ทั้งที่การตั้งค่าโมเดลและพารามิเตอร์การฝึกทั้งหมดตรงกัน 100%
- **การแก้ไข**:
  1. สร้างฟังก์ชันช่วย `configs_match_for_run` ใน `roadresearch/protocol.py` (และใช้งานร่วมกันใน `roadresearch/engine.py`)
  2. กำหนดนโยบายให้รันที่มี `migration.json` สามารถยอมรับความต่างของ `output_root` ได้เพียงฟิลด์เดียว ขณะที่ฟิลด์อื่นทุกฟิลด์ต้องตรงกันทั้งหมด
  3. เพิ่มการตรวจสอบความปลอดภัยของ Provenance (Provenance Hardening) ตรวจสอบว่า `migrated_from` ระบุชื่อโฟลเดอร์ตรงกับโฟลเดอร์ปลายทาง ป้องกันการปลอมแปลง metadata

---

## 7. การรับรองคุณภาพซอฟต์แวร์และการทดสอบ (Quality Assurance & Test Suite)

ระบบวิจัยมีชุดทดสอบอัตโนมัติ (Automated Test Suite) ครอบคลุมทั้ง Unit Test และ Integration Test ในไดเรกทอรี `tests/`:
- **ผลการรัน Pytest ล่าสุด**: **ผ่านครบ 100% ทั้ง 25 การทดสอบ (`25 passed`)**
- ครอบคลุมการทดสอบ Data Loader, Coordinate Clipping, Geometry Head Gradient, Checkpoint Resume Equivalence, AMP Overflow Handling, RT-DETR Precision Policy และ Migration Config Verification

---

## 8. ไฟล์ผลลัพธ์และอาร์ติแฟกต์สำคัญในโครงการ (Key Project Artifacts)

1. **รายงานและเอกสารวิจัยฉบับสมบูรณ์**:
   - `RESEARCH_PROGRESS_REPORT.md`: รายงานฉบับนี้สำหรับนำส่งอาจารย์
   - `how_to_summary.md`: คู่มือขั้นตอนการดำเนินงาน สรุปผล และคำสั่งทั้งหมด
2. **ผลลัพธ์สำหรับการตีพิมพ์บทความวิจัย (`runs/paper_study_recovered/paper/`)**:
   - `paper_table.tex`: ตารางผลการทดลองฉบับ LaTeX พร้อมนำไปใช้ในร่างเปเปอร์
   - `summary_mean_std.csv`: ตารางค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐานของทุกโมเดล
   - `per_run.csv`: ผลการประเมินแยกรายโมเดลและราย Seed ครบทั้ง 24 รัน
   - `per_class_mean_std.csv`: ผลการประเมินแยกรายคลาสความเสียหาย (Pothole, Crack, Manhole)
   - `statistical_comparisons.json`: ผลการทดสอบนัยสำคัญทางสถิติด้วย Bootstrap
   - `frozen_study.json`: โปรโตคอลการวิจัยที่ถูกปิดอย่างสมบูรณ์
   - กราฟ Precision-Recall Curves และ Confusion Matrices ครบถ้วนทุกโมเดล
3. **โฟลเดอร์บันทึกประวัติการแก้ปัญหา (`bugfix/`)**:
   - เอกสารรายงานการสืบสวนและผลการแก้ไขปัญหา Bugfix 1 ถึง Bugfix 4

---

## 9. แผนการดำเนินงานขั้นต่อไป (Next Steps)

1. **การร่างบทความวิจัย (Paper Manuscript Drafting)**:
   - นำผลการทดลองจาก `paper_table.tex` และกราฟ PR Curves ใน `runs/paper_study_recovered/paper/` ไปประกอบในบทความวิจัย
2. **การนำโมเดลไปทดสอบตรวจจับภาพจริง (Real-World Inference Demo)**:
   - นำโมเดลที่ดีที่สุด (`fasterrcnn_detail` หรือ `fasterrcnn_detail_geometry`) ไปทดสอบตรวจจับรอยชำรุดบนภาพถ่ายถนนจริงจากกล้องหน้ารถ หรือทดสอบตรวจจับภาพความละเอียดสูงด้วยฟังก์ชัน Tiled Inference
3. **การส่งมอบโค้ดและชุดผลลัพธ์**:
   - ส่งมอบไฟล์รายงานพร้อมไฟล์บีบอัดชุดผลการดำเนินงานให้อาจารย์ที่ปรึกษาพิจารณา
