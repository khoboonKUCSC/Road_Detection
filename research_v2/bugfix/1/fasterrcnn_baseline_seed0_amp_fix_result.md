# ผลลัพธ์การแก้ไขปัญหา AMP Non-finite Gradients (Faster R-CNN Baseline Seed 0)

## 1. สรุปภาพรวมของปัญหา (Problem Summary)
จากรายงานใน [fasterrcnn_baseline_seed0_amp_nonfinite_gradients.md](file:///home/ds/Desktop/Road_Detection/research_v2/bugfix/1/fasterrcnn_baseline_seed0_amp_nonfinite_gradients.md):
- การเทรน `runs/paper_study/fasterrcnn_baseline_seed0` รันผ่านไปได้ 12 epochs และเกิดข้อผิดพลาดที่ epoch 13
- Exception ที่เกิดขึ้นคือ:
  ```text
  RuntimeError: The total norm of order 2.0 for gradients from `parameters` is non-finite, so it cannot be clipped.
  ```
- **สาเหตุรากเหง้า (Root Cause)**: เกิดจากลำดับการควบคุม (control flow) ใน training loop ของ `roadresearch/engine.py` ที่เรียก `torch.nn.utils.clip_grad_norm_(..., error_if_nonfinite=True)` ทันทีหลัง `scaler.unscale_(optimizer)` ก่อนที่ `scaler.step(optimizer)` และ `scaler.update()` จะมีโอกาสตรวจจับ non-finite gradient เพื่อข้าม batch และปรับลด scale (backoff) ตามกลไกปกติของ PyTorch GradScaler

---

## 2. รายละเอียดการแก้ไขโค้ด (Code Changes)

### 2.1 ปรับปรุง Training Loop ใน [`roadresearch/engine.py`](file:///home/ds/Desktop/Road_Detection/research_v2/roadresearch/engine.py)
1. **Finite Gradient Check ก่อน Clipping**:
   - หลังจาก `scaler.scale(loss).backward()` และ `scaler.unscale_(optimizer)` ตรวจสอบ `p.grad` ของทุก parameter ผ่าน `torch.isfinite(p.grad).all()`
   - หาก gradients ทั้งหมดเป็น finite จะเรียก `clip_grad_norm_(..., error_if_nonfinite=True)` ตามปกติ
2. **AMP Overflow Handling**:
   - หากพบ gradient ที่เป็น NaN/Inf และ AMP เปิดใช้งาน (`scaler.is_enabled()` เป็น True):
     - **ไม่เรียก** `clip_grad_norm_` (ป้องกัน propagation ของ NaN เข้าสู่น้ำหนัก)
     - บันทึก metadata เหตุการณ์ overflow (`epoch`, `batch`, `step`, `image_ids`, `losses`, `scale_before`, `parameters`, `bad_parameter_count`)
     - เรียก `scaler.step(optimizer)` ซึ่ง PyTorch GradScaler จะตรวจพบ `found_inf` และข้ามการอัปเดตน้ำหนัก (`optimizer.step()` ไม่ถูกเรียก)
     - เรียก `scaler.update()` เพื่อลด scale (เช่น จาก 262,144.0 เหลือ 131,072.0)
     - บันทึก `scale_after` ลงใน structured event
     - เขียน event ลงไฟล์ `overflow_events.jsonl` ในโฟลเดอร์รัน
     - เพิ่มตัวนับ `epoch_skipped_amp_steps` และ `total_skipped_amp_steps`
     - เคลียร์ gradient ด้วย `optimizer.zero_grad(set_to_none=True)` แล้ว `continue` ไปยัง batch ถัดไป
     - มี threshold ตรวจสอบ `max_consecutive_amp_overflows` (default: 20) หากเกิด overflow ติดต่อกันเกินกำหนด จะ raise `FloatingPointError` เพื่อป้องกันการเทรนค้างแบบ silent failure
3. **AMP Disabled Behavior**:
   - หาก AMP ปิดใช้งานอยู่ (`scaler.is_enabled()` เป็น False) และพบ non-finite gradient จะ raise `FloatingPointError` พร้อมส่งข้อมูล diagnostic ทันที ป้องกันไม่ให้ optimizer เขียนค่า NaN ลงในน้ำหนักโมเดล
4. **Checkpoint State & History Tracking**:
   - เพิ่ม `skipped_amp_steps` ลงในแต่ละแถวของ `history.json` / `history.csv`
   - บันทึก `total_skipped_amp_steps` และ sample ของ `overflow_events` (สูงสุด 10 รายการล่าสุด) ลงใน checkpoint state (`last.pt`) เพื่อความสมบูรณ์ในการ resume
   - ป้องกัน zero division ในการคำนวณ loss เฉลี่ยกรณี steps ใน epoch เป็น 0
5. **Resume Strictness & Migration Support**:
   - ปรับปรุงการตรวจสอบ `checkpoint["config"] == config` เพื่อรองรับการย้ายรันผ่าน `migration.json` โดยอนุญาตให้ `output_root` ต่างกันได้ แต่ hyperparameter, seed, architecture และ dataset hash อื่นๆ ต้องตรงกันทุกประการ

### 2.2 เพิ่มระบบ Migration ใน [`roadresearch/engine.py`](file:///home/ds/Desktop/Road_Detection/research_v2/roadresearch/engine.py) และ [`roadresearch/cli.py`](file:///home/ds/Desktop/Road_Detection/research_v2/roadresearch/cli.py)
- เพิ่มฟังก์ชัน `migrate_run(source_dir, destination_dir, reason)` และคำสั่ง CLI `python3 -m roadresearch.cli migrate`
- คัดลอก checkpoints (`last.pt`, `best.pt`), `manifest.json`, `history.json`, `history.csv`, `config.json` ไปยังไดเรกทอรีใหม่
- บันทึก `source_snapshot` ใหม่สำหรับโค้ดที่แก้ไข พร้อมสร้าง `migration.json` บันทึก provenance และความไม่ต่อเนื่องของ source hash (`source_sha256_discontinuity`)
- **ไม่แก้ไขหรือเขียนทับ** ข้อมูลในโฟลเดอร์ต้นฉบับ `runs/paper_study/fasterrcnn_baseline_seed0` เพื่อรักษาหลักฐานความล้มเหลวเดิมไว้ 100%

---

## 3. การทดสอบและการตรวจสอบ (Validation & Tests)

### 3.1 เพิ่ม Unit & Regression Tests ใน [`tests/test_research.py`](file:///home/ds/Desktop/Road_Detection/research_v2/tests/test_research.py)
1. **`test_finite_gradient_step_clips_and_updates`**:
   - ทดสอบการรัน training step ปกติที่มี finite gradients ยืนยันว่า `clip_grad_norm_` ทำงานได้สำเร็จ, พารามิเตอร์ได้รับการอัปเดต, ไม่มี overflow events และ `skipped_amp_steps == 0`
2. **`test_amp_nonfinite_gradient_skips_step_and_backs_off`**:
   - จำลอง non-finite gradient บน CUDA เมื่อเปิด AMP ยืนยันว่า:
     - `clip_grad_norm_` ไม่ทำให้โปรแกรมแครช
     - `scaler.update()` ลด scale ลงครึ่งหนึ่ง (จาก 65536.0 -> 32768.0)
     - `overflow_events.jsonl` และ checkpoint ถูกบันทึกข้อมูลครบถ้วน (`epoch`, `step`, `image_ids`, `scale_before`, `scale_after`, `parameters`)
     - Training loop ทำงานต่อไปยัง batch ถัดไปและอัปเดตน้ำหนักได้ตามปกติ
3. **`test_nonfinite_gradient_amp_disabled_raises_floating_point_error`**:
   - จำลอง non-finite gradient เมื่อปิด AMP ยืนยันว่า raise `FloatingPointError` พร้อมข้อความ diagnostic ที่ระบุชื่อ parameter, image IDs และ losses
4. **`test_epoch_boundary_resume_equivalence`**:
   - ขยาย assertion ตรวจสอบ `total_skipped_amp_steps` และ `overflow_events` ระหว่าง continuous run และ resumed run เพื่อยืนยันความ deterministic ข้าม epoch boundary

### 3.2 ผลการรัน Pytest
รันชุดทดสอบทั้งหมด 18 ข้อ (รวมของเดิมและข้อใหม่):
```text
============================= test session starts ==============================
platform linux -- Python 3.12.15, pytest-8.3.5, pluggy-1.6.0
rootdir: /home/ds/Desktop/Road_Detection/research_v2
configfile: pytest.ini
testpaths: tests
collected 18 items

tests/test_research.py ............                                      [ 66%]
tests/test_v2.py ......                                                  [100%]

============================= 18 passed in 26.72s ==============================
```
**ผลลัพธ์**: ผ่านทั้งหมด 18/18 รายการ (100% Pass)

---

## 4. ผลการทดสอบ Reproduction บน NVIDIA GeForce RTX 4090

### 4.1 ขั้นตอนการทดสอบ
1. ใช้คำสั่ง migrate คัดลอก checkpoint จาก `runs/paper_study/fasterrcnn_baseline_seed0` ไปยัง `runs/paper_study_recovered/fasterrcnn_baseline_seed0` (คงโฟลเดอร์ต้นฉบับไว้สมบูรณ์)
2. สั่ง `--resume` เพื่อรันต่อจาก epoch 12 (`last.pt`) เข้าสู่ epoch 13 บน RTX 4090

### 4.2 ข้อมูล Overflow Event ที่ตรวจจับได้จริงใน Epoch 13
ระบบตรวจพบ non-finite gradient ได้อย่างถูกต้องที่ batch 177:
```json
{
  "epoch": 13,
  "batch": 177,
  "step": 177,
  "image_ids": [1549, 28, 1581, 1189],
  "losses": {
    "loss_classifier": 0.08259528875350952,
    "loss_box_reg": 0.18611246347427368,
    "loss_objectness": 0.0060702222399413586,
    "loss_rpn_box_reg": 0.010320434346795082
  },
  "scale": 262144.0,
  "scale_before": 262144.0,
  "bad_parameter_count": 1,
  "parameters": ["detector.roi_heads.box_predictor.bbox_pred.weight"],
  "scale_after": 131072.0
}
```

### 4.3 พฤติกรรมหลังการตรวจจับ
- **การข้าม Batch และลด Scale**: Batch 177 ถูกข้ามโดยไม่เกิดข้อผิดพลาด, Scale ของ AMP GradScaler ลดลงจาก `262,144.0` เหลือ `131,072.0`
- **Batch ถัดไป**: Gradients กลับมาเป็น finite และน้ำหนักโมเดลได้รับการอัปเดตตามปกติ
- **ความสำเร็จของ Epoch 13**:
  - เทรนครบทั้ง 348 batches ใน epoch 13
  - ประเมินผล Validation COCO AP ได้สำเร็จ (`val_AP: 0.1214`, `val_AP50: 0.2884`)
  - บันทึก checkpoint `last.pt` ตัวใหม่ (epoch 12 zero-indexed) พร้อม `skipped_amp_steps: 1` และประวัติการเทรน 13 แถว
  - ข้อมูลใน `last.pt`:
    - `epoch`: 12 (epoch ที่ 13)
    - `scaler`: `{'scale': 131072.0, 'growth_factor': 2.0, 'backoff_factor': 0.5, 'growth_interval': 2000, '_growth_tracker': 172}`
    - `total_skipped_amp_steps`: 1
    - `overflow_events`: มี 1 รายการบันทึกไว้ใน checkpoint

---

## 5. สรุปสถานะ (Conclusion)
- บั๊ก `RuntimeError` จาก `clip_grad_norm_` ในสภาวะ AMP non-finite gradients ได้รับการแก้ไขเรียบร้อยแล้ว
- รองรับทั้งกรณีเปิด AMP (ข้าม batch + backoff scale + log structured diagnostic) และปิด AMP (raise `FloatingPointError` พร้อม diagnostic)
- ชุดทดสอบ 18 รายการผ่านทั้งหมด
- ตรวจสอบและยืนยันผลจริงบนฮาร์ดแวร์ NVIDIA RTX 4090 โดยโมเดล Faster R-CNN สามารถเทรนผ่านจุดที่เคยแครชใน epoch 13 และบันทึก checkpoint ได้สำเร็จสมบูรณ์
