import os
import glob

# ------------------------------
# CONFIGURATION
# ------------------------------
labels_dir = "labels"          # directory with txt polygons
yolo_dir = "labels-YOLO"      # directory with standard YOLO txt format
os.makedirs(yolo_dir, exist_ok=True)

# ------------------------------
# Conversion Function
# ------------------------------
def polygon_to_yolo(coords):
    """
    coords: lista di 8 float [x1,y1,x2,y2,x3,y3,x4,y4], normalizzati [0,1]
    return: x_center, y_center, width, height (normalizzati [0,1])
    """
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return x_center, y_center, width, height

# ------------------------------
# Conversion of all files
# ------------------------------
txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))

for txt_path in txt_files:
    with open(txt_path, "r") as f:
        lines = f.readlines()

    yolo_lines = []

    for line_num, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 9:
            print(f"⚠️ Skipping malformed line in {txt_path} line {line_num+1}")
            continue

        cls = parts[0]
        coords = list(map(float, parts[1:]))
        x_c, y_c, w, h = polygon_to_yolo(coords)
        yolo_lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    # Salva nel folder YOLO con stesso nome
    filename = os.path.basename(txt_path)
    yolo_path = os.path.join(yolo_dir, filename)
    with open(yolo_path, "w") as f:
        f.write("\n".join(yolo_lines))

print(f"✅ Conversion complete! YOLO labels saved in '{yolo_dir}'")

