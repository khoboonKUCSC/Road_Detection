import os

def verify_split(data_dir):
    """เช็คสรุปว่า images และ labels ใน train/val ตรงกันหรือไม่"""
    
    print(f"{'='*55}")
    print(f"📋 สรุปผลการแบ่งข้อมูล Train/Val")
    print(f"{'='*55}\n")
    
    results = {}
    
    for split in ['train', 'val']:
        images_dir = os.path.join(data_dir, 'images', split)
        labels_dir = os.path.join(data_dir, 'labels', split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"❌ ไม่พบโฟลเดอร์ {split} (images หรือ labels)")
            continue
        
        # ดึงชื่อไฟล์ (ตัดนามสกุลออกเพื่อเทียบกัน)
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        image_names = {os.path.splitext(f)[0] for f in os.listdir(images_dir) 
                       if f.lower().endswith(valid_ext)}
        label_names = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) 
                       if f.endswith('.txt')}
        
        n_images = len(image_names)
        n_labels = len(label_names)
        
        # หาไฟล์ที่ไม่ตรงกัน
        images_without_label = image_names - label_names
        labels_without_image = label_names - image_names
        
        results[split] = {
            'n_images': n_images,
            'n_labels': n_labels,
            'images_without_label': images_without_label,
            'labels_without_image': labels_without_image
        }
        
        # แสดงผล
        print(f"📁 [{split.upper()}]")
        print(f"   🖼️  จำนวนรูปภาพ : {n_images}")
        print(f"   🏷️  จำนวน label : {n_labels}")
        
        if n_images == n_labels and not images_without_label and not labels_without_image:
            print(f"   ✅ ตรงกันสมบูรณ์!")
        else:
            print(f"   ⚠️  ไม่ตรงกัน!")
            if images_without_label:
                print(f"      - รูปที่ไม่มี label: {len(images_without_label)} ไฟล์ "
                      f"(ตัวอย่าง: {list(images_without_label)[:3]})")
            if labels_without_image:
                print(f"      - label ที่ไม่มีรูป: {len(labels_without_image)} ไฟล์ "
                      f"(ตัวอย่าง: {list(labels_without_image)[:3]})")
        print()
    
    # สรุปภาพรวม
    if 'train' in results and 'val' in results:
        total_images = results['train']['n_images'] + results['val']['n_images']
        train_pct = results['train']['n_images'] / total_images * 100 if total_images else 0
        val_pct = results['val']['n_images'] / total_images * 100 if total_images else 0
        
        print(f"{'='*55}")
        print(f"📊 ภาพรวม")
        print(f"{'='*55}")
        print(f"   รวมทั้งหมด : {total_images} รูป")
        print(f"   Train      : {results['train']['n_images']} รูป ({train_pct:.1f}%)")
        print(f"   Val        : {results['val']['n_images']} รูป ({val_pct:.1f}%)")
        
        # เช็คว่ามีรูปซ้ำกันระหว่าง train/val ไหม (ไม่ควรมี)
        train_imgs = {os.path.splitext(f)[0] for f in 
                      os.listdir(os.path.join(data_dir, 'images', 'train'))}
        val_imgs = {os.path.splitext(f)[0] for f in 
                    os.listdir(os.path.join(data_dir, 'images', 'val'))}
        overlap = train_imgs & val_imgs
        
        if overlap:
            print(f"   ❌ พบไฟล์ซ้ำกันระหว่าง train/val: {len(overlap)} ไฟล์!")
        else:
            print(f"   ✅ ไม่มีไฟล์ซ้ำกันระหว่าง train/val")

# --- ใช้งาน ---
verify_split(data_dir="./data")