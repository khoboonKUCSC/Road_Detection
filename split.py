import os
import shutil
import random

def split_train_val(data_dir, images_folder="images", labels_folder="labels-YOLO", 
                     val_ratio=0.2, seed=0):
    """
    แบ่งข้อมูลเป็น train/val แล้วจัดเรียงเป็นโครงสร้างที่ YOLO ต้องการ:
    data_dir/
        images/train/
        images/val/
        labels/train/
        labels/val/
    """
    random.seed(seed)
    
    images_path = os.path.join(data_dir, images_folder)
    labels_path = os.path.join(data_dir, labels_folder)
    
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(images_path) if f.lower().endswith(valid_ext)]
    
    print(f"📊 จำนวนรูปภาพทั้งหมด: {len(image_files)}")
    
    # เช็คว่าทุกรูปมี label คู่กันไหม
    missing_labels = []
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        if not os.path.exists(os.path.join(labels_path, label_file)):
            missing_labels.append(img_file)
    
    if missing_labels:
        print(f"⚠️ พบรูปภาพ {len(missing_labels)} ไฟล์ที่ไม่มี label คู่กัน")
        print(f"   ตัวอย่าง: {missing_labels[:5]}")
    
    random.shuffle(image_files)
    val_size = int(len(image_files) * val_ratio)
    val_files = image_files[:val_size]
    train_files = image_files[val_size:]
    
    print(f"✅ Train: {len(train_files)} รูป | Val: {len(val_files)} รูป")
    
    # หมายเหตุ: โฟลเดอร์ปลายทางตั้งชื่อเป็น "labels" ตามมาตรฐาน YOLO
    # (YOLO auto-replace "images" -> "labels" ใน path ตอนหา label คู่กัน)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(data_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'labels', split), exist_ok=True)
    
    def move_files(file_list, split_name):
        for img_file in file_list:
            label_file = os.path.splitext(img_file)[0] + ".txt"
            
            src_img = os.path.join(images_path, img_file)
            dst_img = os.path.join(data_dir, 'images', split_name, img_file)
            
            src_label = os.path.join(labels_path, label_file)  # อ่านจาก labels-YOLO
            dst_label = os.path.join(data_dir, 'labels', split_name, label_file)  # เขียนลง labels
            
            shutil.copy2(src_img, dst_img)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
    
    move_files(train_files, 'train')
    move_files(val_files, 'val')
    
    print("🎉 แบ่งข้อมูลเสร็จสมบูรณ์!")
    print(f"   📁 {data_dir}/images/train ({len(train_files)} ไฟล์)")
    print(f"   📁 {data_dir}/images/val ({len(val_files)} ไฟล์)")
    print(f"   📁 {data_dir}/labels/train")
    print(f"   📁 {data_dir}/labels/val")

# --- ใช้งาน ---
split_train_val(data_dir="./data", images_folder="images", 
                 labels_folder="labels-YOLO", val_ratio=0.2, seed=0)