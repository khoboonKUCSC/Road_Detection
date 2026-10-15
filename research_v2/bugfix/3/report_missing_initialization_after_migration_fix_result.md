# ผลการแก้ไขปัญหา: Report Missing Initialization After Migration

## 1. สาเหตุของปัญหา (Root Cause)
จากการวิเคราะห์ `bugfix/3/report_missing_initialization_after_migration.md`:
- เมื่อขั้นตอนการสรุปผล `report()` ใน `roadresearch/report.py:44` ทำงาน จะอ่านค่าพารามิเตอร์โมเดลจากไฟล์ `initialization.json`:
  ```python
  read_json(run / "initialization.json")["parameters"]
  ```
- แต่ในฟังก์ชัน `migrate_run()` ของ `roadresearch/engine.py` มีรายการไฟล์ที่คัดลอกตายตัวเพียง:
  ```python
  ("best.pt", "last.pt", "manifest.json", "history.json", "history.csv")
  ```
  โดยไม่ได้ระบุ `"initialization.json"` รวมไว้ด้วย ส่งผลให้รันที่เกิดจากการ migrate (`runs/paper_study_recovered/fasterrcnn_baseline_seed0`) ขาดไฟล์ `initialization.json` จนทำให้การสร้าง Report ล้มเหลวด้วย `FileNotFoundError`

---

## 2. รายละเอียดการแก้ไขโค้ด (Implemented Changes)

### 2.1 [roadresearch/engine.py](file:///home/ds/Desktop/Road_Detection/research_v2/roadresearch/engine.py)
เพิ่ม `"initialization.json"` ใน Tuple ของ Artifacts ที่ต้องคัดลอกในฟังก์ชัน `migrate_run`:
```python
for name in ("best.pt", "last.pt", "manifest.json", "history.json", "history.csv", "initialization.json"):
    if (source_dir / name).exists():
        shutil.copy2(source_dir / name, destination_dir / name)
```
การคัดลอกไฟล์ต้นฉบับโดยตรงช่วยรักษา Provenance ดั้งเดิมของโมเดล (ทั้ง architecture, pretrained weights, parameter counts) โดยไม่มีการ Reconstruct ขึ้นมาใหม่

### 2.2 [roadresearch/report.py](file:///home/ds/Desktop/Road_Detection/research_v2/roadresearch/report.py)
เพิ่มการตรวจสอบ Prerequisite ก่อนสร้าง Figures และตารางสรุปผล:
- หากรันใดไม่มี `initialization.json` ระบบจะตรวจสอบว่ามี `migration.json` หรือไม่
- หากเป็น migrated run จะแสดงข้อความแจ้งเตือนที่ชัดเจนระบุเส้นทางต้นทาง (migrated_from source) เพื่อให้ผู้ใช้สามารถกู้คืนไฟล์ได้ถูกต้อง แทนที่จะเกิด raw `FileNotFoundError` ที่ไม่บอกบริบท

```python
init_file = run / "initialization.json"
if not init_file.exists():
    migrated_info = ""
    if (run / "migration.json").exists():
        mig = read_json(run / "migration.json")
        source_from = mig.get("migrated_from", "the recorded source")
        migrated_info = f" (migrated run; restore it from the recorded migrated_from source {source_from} or rerun migration)"
    raise FileNotFoundError(
        f"Run {run.name} cannot be reported: missing initialization.json{migrated_info}."
    )
```

### 2.3 [tests/test_research.py](file:///home/ds/Desktop/Road_Detection/research_v2/tests/test_research.py)
เพิ่ม Unit & Integration Tests ครอบคลุม 2 ประเด็น:
1. **`test_migrate_run_preserves_initialization_metadata`**: ยืนยันว่า `migrate_run` คัดลอก `initialization.json` ครบถ้วนตรงกับต้นทาง 100% ควบคู่กับ legacy artifacts อื่นๆ
2. **`test_report_missing_initialization_raises_contextual_error`**: ยืนยันว่าเมื่อขาด `initialization.json` ใน migrated run ระบบจะ raise `FileNotFoundError` ที่ระบุชื่อรันและเส้นทาง source จาก `migration.json` อย่างชัดเจน

---

## 3. การกู้คืนข้อมูลเดิม (Data Repair)
ได้ตรวจสอบความสอดคล้องของ Config, Seed, Architecture และ Manifest ระหว่างต้นทางและปลายทาง แล้วดำเนินการคัดลอกไฟล์ `initialization.json` ต้นฉบับ:
```text
runs/paper_study/fasterrcnn_baseline_seed0/initialization.json
    -> runs/paper_study_recovered/fasterrcnn_baseline_seed0/initialization.json
```
- ค่าพารามิเตอร์ที่กู้คืน: `parameters: 43266403`, `trainable_parameters: 43041059`

---

## 4. ผลการทดสอบและการตรวจสอบ (Verification Results)

1. **Automated Test Suite**:
   ```bash
   pytest tests/
   # ผลลัพธ์: 23 passed, 15 warnings in 30.68s (100% Pass)
   ```
2. **Manual Acceptance Report Generation**:
   ทดสอบรันคำสั่งสร้างรายงาน:
   ```python
   from roadresearch.report import report
   report("runs/paper_study_recovered", "runs/paper_study_recovered/development_report", split="val")
   ```
   ผลลัพธ์:
   - รายงานถูกสร้างสำเร็จที่ [`runs/paper_study_recovered/development_report`](file:///home/ds/Desktop/Road_Detection/research_v2/runs/paper_study_recovered/development_report)
   - ไฟล์ `development_report/per_run.csv` มีผลรวมครบทั้ง 24 รัน
   - แถวของ `fasterrcnn_baseline_seed0` ระบุ `parameters: 43266403`, `best_epoch: 10`, `AP: 0.1450` ถูกต้องสมบูรณ์
