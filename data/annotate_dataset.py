#!/usr/bin/env python3
"""
Annotate all pictures with bounding boxes and class labels from labels-YOLO
and save results to image-label directory.
"""

import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2

# Configuration
IMAGE_DIR = "images"
LABEL_DIR = "labels-YOLO"
OUTPUT_DIR = "image-label"

CLASS_NAMES = {
    0: "pothole",
    1: "crack",
    2: "manhole"
}

# Colors in BGR format
CLASS_COLORS = {
    0: (36, 36, 235),    # Pothole: Crimson Red
    1: (40, 190, 40),    # Crack: Vibrant Green
    2: (235, 140, 20),   # Manhole: Vivid Blue
}

TEXT_COLOR = (255, 255, 255)  # White text


def annotate_single_image(img_path, label_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        return False, f"Failed to read image: {img_path}"

    h, w = img.shape[:2]

    # Read YOLO annotations
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = int(round((xc - bw / 2.0) * w))
                    y1 = int(round((yc - bh / 2.0) * h))
                    x2 = int(round((xc + bw / 2.0) * w))
                    y2 = int(round((yc + bh / 2.0) * h))

                    # Clamp to image boundaries
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(0, min(w - 1, x2))
                    y2 = max(0, min(h - 1, y2))

                    boxes.append((cls_id, x1, y1, x2, y2))

    # Draw bounding boxes and label badges
    for cls_id, x1, y1, x2, y2 in boxes:
        color = CLASS_COLORS.get(cls_id, (0, 255, 255))
        class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label badge styling
        font_scale = 0.45
        font_thickness = 1
        (tw, th), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Position badge above box if possible, otherwise inside top edge
        if y1 - th - 6 >= 0:
            bx1, by1 = x1, y1 - th - 6
            bx2, by2 = min(w, x1 + tw + 6), y1
            tx, ty = x1 + 3, y1 - 4
        else:
            bx1, by1 = x1, y1
            bx2, by2 = min(w, x1 + tw + 6), y1 + th + 6
            tx, ty = x1 + 3, y1 + th + 2

        # Draw badge background and label text
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, -1)
        cv2.putText(
            img,
            class_name,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            TEXT_COLOR,
            font_thickness,
            cv2.LINE_AA
        )

    # Save output image
    success = cv2.imwrite(output_path, img)
    if not success:
        return False, f"Failed to save image: {output_path}"
    return True, None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*")))
    # Filter valid image extensions
    img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png"]]

    total_images = len(img_files)
    print(f"Found {total_images} images in '{IMAGE_DIR}'.")
    print(f"Reading labels from '{LABEL_DIR}'...")
    print(f"Saving annotated images to '{OUTPUT_DIR}'...")

    tasks = []
    for img_path in img_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        label_path = os.path.join(LABEL_DIR, f"{stem}.txt")
        output_path = os.path.join(OUTPUT_DIR, f"{stem}{ext}")
        tasks.append((img_path, label_path, output_path))

    start_time = time.time()
    num_workers = min(os.cpu_count() or 4, 16)
    completed = 0
    errors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(annotate_single_image, task[0], task[1], task[2]): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            success, err = future.result()
            if success:
                completed += 1
            else:
                errors.append(err)

            if completed % 200 == 0 or completed == total_images:
                elapsed = time.time() - start_time
                print(f"Progress: {completed}/{total_images} images processed ({elapsed:.1f}s)...")

    elapsed_total = time.time() - start_time
    print(f"\nDone! Successfully annotated {completed}/{total_images} images in {elapsed_total:.2f} seconds.")
    if errors:
        print(f"Encountered {len(errors)} errors:")
        for e in errors[:5]:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
