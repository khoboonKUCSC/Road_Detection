# รายงานผลการแก้ไขข้อผิดพลาด (Bugfix Result): Freeze Study Migration-Aware Config Match

## 1. ข้อมูลทั่วไปของปัญหา (Issue Overview)
- **ไฟล์รายงานปัญหา**: [`bugfix/4/freeze_study_migrated_output_root_config.md`](file:///home/ds/Desktop/Road_Detection/research_v2/bugfix/4/freeze_study_migrated_output_root_config.md)
- **จุดที่เกิดข้อผิดพลาด**: ฟังก์ชัน `freeze_study()` ใน [`roadresearch/protocol.py`](file:///home/ds/Desktop/Road_Detection/research_v2/roadresearch/protocol.py) ระหว่างดำเนินการ preregistration ตามคำสั่งขั้นตอน 4.2:
  ```bash
  python run.py freeze-study \
    --config configs/rtx4090_recovered.yaml \
    --similarity-audit artifacts/leakage/similarity_audit.json \
    --output artifacts/study/protocol.json \
    --exploratory
  ```
- **ข้อความ Error ที่พบ**:
  ```text
  ValueError: Run config disagrees with frozen config: /home/ds/Desktop/Road_Detection/research_v2/runs/paper_study_recovered/fasterrcnn_baseline_seed0
  ```

---

## 2. สาเหตุของปัญหา (Root Cause Analysis)
1. โฟลเดอร์ `runs/paper_study_recovered/fasterrcnn_baseline_seed0` เป็นรันที่ถูกย้ายมาจาก `runs/paper_study/fasterrcnn_baseline_seed0` ผ่านฟังก์ชัน `migrate_run()` เพื่อแก้ไขปัญหา AMP non-finite gradient
2. `migrate_run()` ได้ทำการสำเนาไฟล์คอนฟิกเดิม (`config.json`) เก็บไว้เพื่อรักษา Provenance ดั้งเดิม โดยในคอนฟิกเดิมระบุ `output_root: ".../runs/paper_study"` ในขณะที่คอนฟิกสำหรับรันใหม่ใน `configs/rtx4090_recovered.yaml` ระบุ `output_root: ".../runs/paper_study_recovered"` โดยฟิลด์อื่น ๆ ทั้งหมด (model architecture, seed, manifest, training parameters, evaluation parameters) ตรงกันทุกประการ
3. ใน [`roadresearch/engine.py`](file:///home/ds/Desktop/Road_Detection/research_v2/roadresearch/engine.py) ฟังก์ชัน `train()` ได้รองรับข้อยกเว้นนี้แล้วเมื่อพบไฟล์ `migration.json` แต่ใน [`roadresearch/protocol.py`](file:///home/ds/Desktop/Road_Detection/research_v2/roadresearch/protocol.py) ฟังก์ชัน `freeze_study()` ยังคงใช้การเปรียบเทียบแบบ strict equality ทั้ง dictionary (`read_json(run/"config.json") != experiment_config(...)`) ทำให้การ freeze study ไม่สอดคล้องกับระบบ migration ของโปรเจกต์

---

## 3. การแก้ไขโค้ด (Implementation Details)

### 3.1 เพิ่มฟังก์ชัน `configs_match_for_run` ใน [`roadresearch/protocol.py`](file:///home/ds/Desktop/Road_Detection/research_v2/roadresearch/protocol.py)
สร้างฟังก์ชันช่วยสำหรับตรวจสอบคอนฟิกของรัน โดย:
1. หากคอนฟิกเท่ากันทุกประการ ให้ผ่านทันที
2. หากคอนฟิกไม่เท่ากัน ให้ตรวจสอบว่ามีไฟล์ `migration.json` อยู่ในโฟลเดอร์รันหรือไม่ หากไม่มีให้คืนค่า `False` (Reject)
3. ตรวจสอบความสมบูรณ์ของ `migration.json` (Validation Hardening):
   - ต้องเป็นไฟล์ JSON ที่ถูกต้อง
   - ต้องมีฟิลด์ `migrated_from` ที่เป็นสตริงและไม่ว่างเปล่า
   - ชื่อโฟลเดอร์ต้นทาง (`Path(migrated_from).name`) ต้องตรงกับชื่อโฟลเดอร์ปลายทาง (`run_dir.name`) เพื่อป้องกันการนำ metadata จากรันอื่นมาสวมสิทธิ์
4. อนุญาตเฉพาะความแตกต่างของฟิลด์ `output_root` เท่านั้น ฟิลด์อื่นทุกฟิลด์ต้องตรงกันทั้งหมด

```python
def configs_match_for_run(expected, actual, run_dir):
    run_dir = Path(run_dir)
    migration_path = run_dir / "migration.json"
    if migration_path.exists():
        try:
            migration_meta = read_json(migration_path)
        except Exception as exc:
            raise ValueError(f"Malformed migration metadata in {run_dir}: {exc}") from exc
        if not isinstance(migration_meta, dict):
            raise ValueError(f"Malformed migration metadata in {run_dir}: root must be a JSON object")
        migrated_from = migration_meta.get("migrated_from")
        if not isinstance(migrated_from, str) or not migrated_from.strip():
            raise ValueError(f"Malformed migration metadata in {run_dir}: missing or invalid 'migrated_from'")
        if Path(migrated_from).name != run_dir.name:
            raise ValueError(
                f"Mismatched migration provenance in {run_dir}: "
                f"migrated_from '{Path(migrated_from).name}' does not match run directory '{run_dir.name}'"
            )
        if actual == expected:
            return True
        return (
            {k: v for k, v in actual.items() if k != "output_root"}
            == {k: v for k, v in expected.items() if k != "output_root"}
        )
    return actual == expected
```

และปรับปรุง `freeze_study()` ให้เรียกใช้ `configs_match_for_run`:
```python
if not configs_match_for_run(experiment_config(config, experiment, seed), read_json(run/"config.json"), run):
    raise ValueError(f"Run config disagrees with frozen config: {run}")
```

### 3.2 ใช้งานร่วมกันใน [`roadresearch/engine.py`](file:///home/ds/Desktop/Road_Detection/research_v2/roadresearch/engine.py)
นำเข้า `configs_match_for_run` มาใช้ในฟังก์ชัน `train()` สำหรับตรวจสอบคอนฟิกระหว่าง resume เพื่อให้มีนโยบายการตรวจสอบที่เป็นมาตรฐานเดียวกันทั่วทั้งระบบ

---

## 4. ผลการทดสอบ (Verification & Test Results)

### 4.1 Unit & Integration Regression Tests
ได้เพิ่มชุดการทดสอบครอบคลุมเงื่อนไข 5 ข้อตามข้อกำหนดใน [`tests/test_research.py`](file:///home/ds/Desktop/Road_Detection/research_v2/tests/test_research.py):
1. **Case 1**: คอนฟิกที่ตรงกันทุกประการผ่านการตรวจสอบแม้ไม่มี `migration.json`
2. **Case 2**: รันที่ไม่ได้ migrate แต่ต่างกันที่ `output_root` ถูกปฏิเสธ (Fail)
3. **Case 3**: รันที่ migrate มาและต่างกันเฉพาะ `output_root` ผ่านการตรวจสอบ (Pass)
4. **Case 4**: รันที่ migrate มาแต่มีฟิลด์อื่นเปลี่ยน (เช่น `seed`, `training.lr`, `model.architecture`) ถูกปฏิเสธ (Fail)
5. **Case 5**: ไฟล์ `migration.json` ที่เสีย โครงสร้างไม่ถูกต้อง หรือระบุรันต้นทางไม่ตรงโฟลเดอร์ ถูกปฏิเสธพร้อมแจ้งข้อผิดพลาด (Fail)
6. **Integration Test**: ตรวจสอบการสร้าง `protocol.json` และยืนยันว่าไม่มีการแก้ไข `config.json` ดั้งเดิมของรัน

ผลการรันชุดทดสอบผ่าน 100%:
```text
tests/test_research.py::test_configs_match_for_run_criteria PASSED
tests/test_research.py::test_freeze_study_migration_integration PASSED
```

### 4.2 การทดสอบยอมรับด้วยคำสั่งจริง (Manual Acceptance)
รันคำสั่ง freeze-study:
```bash
python run.py freeze-study \
  --config configs/rtx4090_recovered.yaml \
  --similarity-audit artifacts/leakage/similarity_audit.json \
  --output artifacts/study/protocol.json \
  --exploratory
```
**ผลลัพธ์**:
```text
Study frozen: artifacts/study/protocol.json
```
- สร้างไฟล์ `artifacts/study/protocol.json` สำเร็จสมบูรณ์
- บันทึก entries ครบทั้ง 24 รัน (8 โมเดล × 3 seeds)
- คอนฟิก `config.json` ภายในทุกรันไม่ถูกดัดแปลงและคงค่า hash ดั้งเดิมไว้ครบถ้วน
