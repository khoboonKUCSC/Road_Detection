from __future__ import annotations

from collections import defaultdict

import numpy as np
from PIL import Image

from .metrics import box_iou, coco_evaluate, xywh_to_xyxy
from .utils import CLASSES, save_json


def failure_analysis(dataset, predictions, thresholds, output):
    by_image = defaultdict(list)
    for prediction in predictions:
        by_image[prediction["image_id"]].append(prediction)
    image_conditions, objects = [], []
    for record in dataset.records:
        with Image.open(dataset.root/record["image_path"]) as image:
            gray = np.asarray(image.convert("L").resize((64, 36)), dtype=float)/255
        conditions = {"mean_luminance": float(gray.mean()), "luminance_std": float(gray.std())}
        conditions["brightness_bin"] = "dark" if gray.mean() < .25 else "bright" if gray.mean() > .75 else "mid"
        conditions["contrast_bin"] = "low" if gray.std() < .12 else "normal"
        image_conditions.append({"image_id": record["id"], "group": record["group"], **conditions})
        annotations = record["annotations"]
        matched = set()
        for cls, name in enumerate(CLASSES, 1):
            indexes = [j for j, a in enumerate(annotations) if a["category_id"] == cls]
            boxes = [annotations[j]["bbox_xyxy"] for j in indexes]
            preds = sorted([p for p in by_image[record["id"]] if p["category_id"] == cls and p["score"] >= thresholds[name]],
                           key=lambda p: -p["score"])
            for pred in preds:
                overlaps = box_iou(xywh_to_xyxy(pred["bbox"]), boxes)
                for j, index in enumerate(indexes):
                    if index in matched:
                        overlaps[j] = -1
                best = int(overlaps.argmax()) if len(overlaps) else -1
                if best >= 0 and overlaps[best] >= .5:
                    matched.add(indexes[best])
        for index, annotation in enumerate(annotations):
            x1, y1, x2, y2 = annotation["bbox_xyxy"]
            width, height = x2-x1, y2-y1
            area, aspect = width*height, max(width/height, height/width)
            objects.append({"image_id": record["id"], "class": CLASSES[annotation["category_id"]-1],
                            "area_bin": "small" if area < 32**2 else "medium" if area < 96**2 else "large",
                            "aspect_bin": "elongated" if aspect >= 4 else "compact", "matched": index in matched,
                            "group": record["group"], **conditions})
    summary = []
    for factor in ["area_bin", "aspect_bin", "brightness_bin", "contrast_bin", "group"]:
        strata = defaultdict(list)
        for obj in objects:
            strata[(obj["class"], obj[factor])].append(obj)
        for (cls, value), items in sorted(strata.items()):
            hits = sum(o["matched"] for o in items)
            summary.append({"class": cls, "factor": factor, "stratum": value, "GT_objects": len(items),
                            "detected": hits, "missed": len(items)-hits, "recall_at_frozen_threshold": hits/len(items)})
    # Condition/group AP uses all annotations and predictions of selected images.
    ap_strata = []
    for factor in ["brightness_bin", "contrast_bin", "group"]:
        for value in sorted({r[factor] for r in image_conditions}):
            ids = {r["image_id"] for r in image_conditions if r[factor] == value}
            records = [r for r in dataset.records if r["id"] in ids]
            metrics, _, _ = coco_evaluate(records, [p for p in predictions if p["image_id"] in ids])
            ap_strata.append({"factor": factor, "stratum": value, "images": len(ids), "AP": metrics["AP"],
                              "AP_shared_damage": metrics["AP_shared_damage"], "per_class": metrics["per_class"]})
    save_json(output, {"objects": objects, "recall_strata": summary, "image_conditions": image_conditions,
                       "AP_strata": ap_strata,
                       "limitations": "Brightness/contrast bins are pixel statistics, not weather or device labels. Size bins use native-image bbox area."})
