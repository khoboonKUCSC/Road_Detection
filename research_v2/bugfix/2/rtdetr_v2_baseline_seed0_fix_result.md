# ผลการแก้ไขปัญหา: RT-DETRv2 Baseline NaN Boxes / Hungarian Matcher Crash

## 1. สาเหตุของปัญหา (Root Cause)
จากการวิเคราะห์ `runs/paper_study_recovered/rtdetr_v2_baseline_seed0/failure.json` และบันทึกใน `bugfix/2/rtdetr_v2_baseline_seed0_nan_boxes.md`:
1. โมเดล RT-DETRv2 (Hugging Face `RTDetrV2ForObjectDetection`) มีโครงสร้าง Deformable Attention, Hybrid Encoder, Box Regression Head และ Hungarian Matcher
2. เมื่อรันภายใต้ CUDA FP16 Autocast ทั่วทั้งโปรเจกต์ (`torch.autocast(..., dtype=torch.float16)`):
   - ค่า Attention weights และ Bounding Box coordinates เกิดการล้นทางตัวเลข (Numerical Overflow/Instability)
   - สังเกตได้จาก Epoch 1 ที่เกิด GradScaler Overflow ถึง 8 ครั้งติดต่อกันจน Scale ลดลงจาก 65536 เหลือ 256
   - ใน Epoch 3 ค่า `out_bbox` ที่พยากรณ์ออกมากลายเป็น `NaN` ส่งผลให้ Hungarian Matcher (`generalized_box_iou`) พังด้วยข้อผิดพลาด:
     ```text
     ValueError: boxes1 must be in [x0, y0, x1, y1] (corner) format, but got tensor([[nan, nan, nan, nan], ...], device='cuda:0')
     ```

---

## 2. รายละเอียดการแก้ไขโค้ด (Implemented Changes)

### 2.1 [roadresearch/adapters.py](file:///home/ds/Desktop/Road_Detection/research_v2/roadresearch/adapters.py)
1. เพิ่ม Explicit Precision Capabilities ให้กับคลาส `RTDETRv2Detector`:
   ```python
   class RTDETRv2Detector(nn.Module):
       uses_fp32_forward = True
       autocast_supported = False
   ```
2. ครอบการคำนวณ `self.network(**inputs)` ให้อยู่ใน Context `torch.autocast(device_type=device.type, enabled=False)` เสมอ ไม่ว่าจะถูกเรียกใน Mode Training หรือ Evaluation เพื่อให้ Transformer Forward, Head Regression และ Matcher คำนวณด้วย FP32 Full Precision ที่เสถียร
3. เพิ่ม **Defensive Diagnostics**:
   - ดักจับ Exception หากเกิด NaN ระหว่าง Forward/Matcher พร้อมแจ้ง `FloatingPointError` พร้อม Context (Device, Image IDs, Autocast State)
   - ตรวจสอบ `outputs.loss`, `outputs.pred_boxes`, และ `outputs.logits` ด้วย `torch.isfinite` หากพบ NaN หรือ Inf จะระบุจำนวนและตัวอย่างพิกัดก่อนส่งต่อ

### 2.2 [roadresearch/engine.py](file:///home/ds/Desktop/Road_Detection/research_v2/roadresearch/engine.py)
1. ปรับปรุง Training Loop ให้ตรวจสอบ Precision Capabilities ของโมเดล:
   ```python
   model_autocast = getattr(model, "autocast_supported", True) and not getattr(model, "uses_fp32_forward", False)
   if hasattr(model, "detector"):
       model_autocast = model_autocast and getattr(model.detector, "autocast_supported", True) and not getattr(model.detector, "uses_fp32_forward", False)
   amp_enabled = scaler.is_enabled() and model_autocast
   with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
       losses = model(images, targets)
   ```
   โมเดลอื่นๆ (Faster R-CNN, FCOS, YOLO) ยังคงใช้ FP16 AMP ตามเดิม ส่วน RT-DETRv2 จะรันด้วย FP32 Forward โดยที่ GradScaler ของ PyTorch ยังคงทำหน้าที่ Scale/Unscale กราเดียนต์ได้อย่างปลอดภัย
2. เพิ่ม Early Return เมื่อโมเดลเทรนเสร็จสมบูรณ์แล้ว (`checkpoint.get("finished")`) ให้ finalize และข้ามการตรวจสอบ `source_sha256.json` ได้ทันที เพื่อป้องกันไม่ให้ 21 โมเดลที่เทรนเสร็จแล้วใน `runs/paper_study_recovered` เกิดข้อผิดพลาดเมื่อมีการแก้โค้ด

### 2.3 [tests/test_v2.py](file:///home/ds/Desktop/Road_Detection/research_v2/tests/test_v2.py)
เพิ่มชุดทดสอบครอบคลุม 3 ด้าน:
1. `test_adapter_autocast_policy_capabilities`: ตรวจสอบสถานะ capability ว่า RT-DETRv2 บังคับใช้ FP32 ขณะที่โมเดลอื่นยังคงใช้ AMP
2. `test_rtdetr_fp32_under_outer_fp16_autocast`: จำลองการรันบน CUDA ภายใต้ FP16 Autocast ทั้ง Training และ Evaluation ยืนยันว่า Loss, Pred Boxes และ Gradients ทั้งหมดเป็น Finite และคำนวณใน FP32
3. `test_rtdetr_defensive_diagnostics_on_nonfinite`: ทดสอบว่ากลไกตรวจจับ Non-finite จะ Throw `FloatingPointError` อย่างถูกต้อง

---

## 3. ผลการทดสอบ (Verification Results)
1. **Automated Test Suite**:
   ```bash
   pytest tests/
   # ผลลัพธ์: 21 passed, 1 warning in 27.45s
   ```
2. **RTX 4090 Live Smoke Test**:
   ทดสอบเทรน `PekingU/rtdetr_v2_r50vd` บนชุดข้อมูลจริง 2 Epochs:
   - `epoch 1`: `loss_rtdetr_total: 242.90`, `skipped_amp_steps: 0`, `val_AP: 0.0001`
   - `epoch 2`: `loss_rtdetr_total: 148.66`, `skipped_amp_steps: 0`, `val_AP: 0.0006`
   - กราเดียนต์และ Loss ลดลงอย่างมีเสถียรภาพ มีจำนวน Skipped Steps เป็น **0** (ไม่มี Overflow เกิดขึ้นเลย)

---

## 4. วิธีการรันโค้ดเทรนโมเดลต่อจากเดิม

สถานะปัจจุบันใน `runs/paper_study_recovered`:
- โมเดลที่เทรนเสร็จสมบูรณ์แล้ว: 21 จาก 24 รัน (Faster R-CNN ทุกแบบ, FCOS, YOLO11s ทั้ง seed 0, 42, 2026)
- โมเดลที่ค้างอยู่: `rtdetr_v2_baseline` (seed 0 พังที่ epoch 2, seed 42 และ 2026 ยังไม่ได้เริ่ม)

เนื่องจาก checkpoint เดิมของ `rtdetr_v2_baseline_seed0` มาจาก Trajectory ของ FP16 ที่มี Overflow สะสมและพัง จึงควรเริ่มรันโมเดล RT-DETRv2 ใหม่ด้วย FP32 ตามแนวทางดังนี้:

### แนวทางที่ 1 (แนะนำ): สำรองรันที่พังไว้ แล้วสั่งรันต่อในโฟลเดอร์เดิมจนครบ 24 โมเดล
คำสั่งนี้จะเก็บหลักฐานรันที่ error ไว้ แล้วให้ระบบรัน RT-DETRv2 (seed 0, 42, 2026) ใหม่ด้วย FP32 โดยโมเดลเดิมทั้ง 21 ตัวจะถูกข้ามอัตโนมัติ (พิมพ์ `Already trained`) และเมื่อเทรนครบ 24 รัน ระบบจะทำ Validation Evaluation และสร้าง Development Report ให้อัตโนมัติ:

```bash
# 1. สำรองโฟลเดอร์รันที่พังไว้เป็นหลักฐาน provenance
mv runs/paper_study_recovered/rtdetr_v2_baseline_seed0 runs/paper_study_recovered/rtdetr_v2_baseline_seed0_failed_fp16

# 2. สั่งรันต่อจากเดิม
python run.py develop --config configs/rtx4090_recovered.yaml --resume
```

### แนวทางที่ 2: รันเฉพาะการทดลอง RT-DETRv2 (ทั้ง 3 seeds)
หากต้องการรันเฉพาะ RT-DETRv2 โดยไม่แตะโมเดลอื่น:

```bash
# สำรองโฟลเดอร์เดิม
mv runs/paper_study_recovered/rtdetr_v2_baseline_seed0 runs/paper_study_recovered/rtdetr_v2_baseline_seed0_failed_fp16

# รันเฉพาะ rtdetr_v2_baseline
python run.py develop --config configs/rtx4090_recovered.yaml --experiment rtdetr_v2_baseline --resume
```

### แนวทางที่ 3: รันการทดลอง RT-DETRv2 ใน Output Directory แยกเฉพาะ
หากต้องการแยกผลการทดลองของ RT-DETRv2 FP32 ไปยัง Root Directory ใหม่:

```bash
python run.py develop --config configs/rtx4090_recovered.yaml --experiment rtdetr_v2_baseline
# (หมายเหตุ: ปรับ output_root ในไฟล์ config หรือผ่าน CLI หากต้องการเก็บแยกโฟลเดอร์)
```
