import os
import json
import glob
from PIL import Image

# ------------------------------
# CONFIGURATION
# ------------------------------
images_dir = "images"
labels_dir = "labels"
output_json = "annotations_coco.json"

# Class mapping (ID can start from 0)
categories = [
    {"id": 0, "name": "pothole"},
    {"id": 1, "name": "crack"},
    {"id": 2, "name": "manhole"},
]

# ------------------------------
# Helper Functions
# ------------------------------
def polygon_to_bbox(coords, img_w, img_h):
    """
    coords: [x1,y1,...,x4,y4] normalized
    returns COCO bbox [x_min, y_min, width, height] in pixels
    """
    xs = coords[0::2]
    ys = coords[1::2]
    xs = [x * img_w for x in xs]
    ys = [y * img_h for y in ys]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [
        x_min,
        y_min,
        x_max - x_min,
        y_max - y_min
    ]

# ------------------------------
# COCO Structure
# ------------------------------
coco = {
    "images": [],
    "annotations": [],
    "categories": categories
}

annotation_id = 1
image_id = 1

# ------------------------------
# Main Conversion Loop
# ------------------------------
label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))

for label_path in label_files:
    base = os.path.splitext(os.path.basename(label_path))[0]
    
    # Find corresponding image
    img_path = None
    for ext in [".jpg", ".png", ".jpeg"]:
        candidate = os.path.join(images_dir, base + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break
    
    if img_path is None:
        print(f"⚠️ Image not found for {base}, skipping")
        continue
    
    # Load image
    with Image.open(img_path) as img:
        width, height = img.size
    
    # Add image entry
    coco["images"].append({
        "id": image_id,
        "file_name": os.path.basename(img_path),
        "width": width,
        "height": height
    })
    
    # Read labels
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 9:
            print(f"⚠️ Malformed line in {label_path} line {line_num+1}")
            continue
        
        category_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        
        bbox = polygon_to_bbox(coords, width, height)
        area = bbox[2] * bbox[3]
        
        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        })
        
        annotation_id += 1
    
    image_id += 1

# ------------------------------
# Save JSON
# ------------------------------
with open(output_json, "w") as f:
    json.dump(coco, f, indent=2)

print(f"✅ COCO annotations saved to '{output_json}'")
print(f"📊 Total images: {len(coco['images'])}")
print(f"📊 Total annotations: {len(coco['annotations'])}")
