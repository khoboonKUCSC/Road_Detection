from __future__ import annotations

import contextlib
import io
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .data import to_coco
from .utils import CLASSES


def coco_evaluate(records, predictions):
    """Official COCO bbox evaluator; absent-class/area scores represented as null."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        gt = COCO()
        gt.dataset = to_coco(records)
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {**gt.dataset, "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.imgIds = [r["id"] for r in records]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    metrics = {name: float(value) if value >= 0 else None for name, value in zip(names, evaluator.stats)}
    precision = evaluator.eval["precision"]
    per_class, curves = {}, {}
    for k, name in enumerate(CLASSES):
        values = precision[:, :, k, 0, -1]
        values50 = values[0]
        per_class[name] = {
            "AP": float(values[values >= 0].mean()) if (values >= 0).any() else None,
            "AP50": float(values50[values50 >= 0].mean()) if (values50 >= 0).any() else None,
            "instances": sum(a["category_id"] == k+1 for r in records for a in r["annotations"]),
        }
        curves[name] = {"recall": evaluator.params.recThrs.tolist(),
                        "precision50": [float(v) if v >= 0 else None for v in values50]}
    metrics["per_class"] = per_class
    shared = [per_class[name]["AP"] for name in ("pothole", "crack")]
    metrics["AP_shared_damage"] = float(np.mean(shared)) if all(v is not None for v in shared) else None
    return metrics, curves, stream.getvalue()


def box_iou(box, others):
    if not len(others):
        return np.zeros(0)
    others = np.asarray(others, dtype=float)
    box = np.asarray(box, dtype=float)
    inter = np.maximum(0, np.minimum(box[2:], others[:, 2:]) - np.maximum(box[:2], others[:, :2])).prod(1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    areas = np.maximum(0, others[:, 2:] - others[:, :2]).prod(1)
    return inter / np.maximum(area + areas - inter, 1e-12)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x+w, y+h]


def operating_metrics(records, predictions, thresholds, iou_threshold=0.5):
    """Score-ordered class-specific matching at a frozen deployment threshold."""
    by_image = defaultdict(list)
    for pred in predictions:
        by_image[pred["image_id"]].append(pred)
    per_class = {}
    for cls, name in enumerate(CLASSES, 1):
        tp = fp = fn = 0
        for record in records:
            gt = [a["bbox_xyxy"] for a in record["annotations"] if a["category_id"] == cls]
            matched = set()
            preds = sorted([p for p in by_image[record["id"]] if p["category_id"] == cls and
                            p["score"] >= thresholds[name]], key=lambda p: -p["score"])
            for pred in preds:
                overlaps = box_iou(xywh_to_xyxy(pred["bbox"]), gt)
                for j in matched:
                    overlaps[j] = -1
                best = int(overlaps.argmax()) if len(overlaps) else -1
                if best >= 0 and overlaps[best] >= iou_threshold:
                    tp += 1
                    matched.add(best)
                else:
                    fp += 1
            fn += len(gt) - len(matched)
        precision, recall = tp / max(tp+fp, 1), tp / max(tp+fn, 1)
        per_class[name] = {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall,
                           "F1": 2*tp / max(2*tp+fp+fn, 1), "FP_per_image": fp / len(records),
                           "threshold": thresholds[name]}
    return {"IoU": iou_threshold, "per_class": per_class,
            "macro_F1": float(np.mean([p["F1"] for p in per_class.values()]))}


def select_thresholds(records, predictions):
    # Tie breaking prefers the higher threshold. Never select using test data.
    grid = np.linspace(0.05, 0.95, 19)
    thresholds = {}
    rows = []
    for threshold in grid:
        result = operating_metrics(records, predictions, {c: float(threshold) for c in CLASSES})
        for name in CLASSES:
            rows.append({"class": name, **result["per_class"][name]})
    for name in CLASSES:
        eligible = [r for r in rows if r["class"] == name]
        thresholds[name] = max(eligible, key=lambda r: (r["F1"], r["threshold"]))["threshold"]
    return thresholds, rows


def confusion_matrix(records, predictions, thresholds, iou_threshold=0.5):
    """Rows=true, columns=predicted; unmatched objects/predictions use background.

    Unlike AP, this diagnostic allows cross-class matches to expose confusion.
    """
    matrix = np.zeros((4, 4), dtype=int)
    by_image = defaultdict(list)
    for pred in predictions:
        if pred["score"] >= thresholds[CLASSES[pred["category_id"]-1]]:
            by_image[pred["image_id"]].append(pred)
    for record in records:
        gt = record["annotations"]
        matched = set()
        for pred in sorted(by_image[record["id"]], key=lambda p: -p["score"]):
            overlaps = box_iou(xywh_to_xyxy(pred["bbox"]), [a["bbox_xyxy"] for a in gt])
            for j in matched:
                overlaps[j] = -1
            best = int(overlaps.argmax()) if len(overlaps) else -1
            if best >= 0 and overlaps[best] >= iou_threshold:
                matrix[gt[best]["category_id"]-1, pred["category_id"]-1] += 1
                matched.add(best)
            else:
                matrix[3, pred["category_id"]-1] += 1
        for j, annotation in enumerate(gt):
            if j not in matched:
                matrix[annotation["category_id"]-1, 3] += 1
    return matrix.tolist()


def group_bootstrap(records, predictions_a, predictions_b=None, repeats=1000, seed=2026):
    """Resample capture-proxy groups, evaluate COCO AP; paired delta if B supplied."""
    if repeats < 2:
        raise ValueError("Use at least 2 bootstrap repeats")
    rng = np.random.default_rng(seed)
    groups = sorted({r["group"] for r in records})
    if len(groups) < 2:
        raise ValueError("Need at least 2 test groups for a group bootstrap interval")
    by_group = {g: [r for r in records if r["group"] == g] for g in groups}
    required_classes = {a["category_id"] for r in records for a in r["annotations"]}
    pred_sets = []
    for predictions in [predictions_a] + ([predictions_b] if predictions_b is not None else []):
        lookup = defaultdict(list)
        for p in predictions:
            lookup[p["image_id"]].append(p)
        pred_sets.append(lookup)
    values, class_complete = [], 0
    for _ in range(repeats):
        sampled, mapped_predictions = [], [[] for _ in pred_sets]
        for group in rng.choice(groups, size=len(groups), replace=True):
            for r in by_group[group]:
                new_id = len(sampled) + 1
                sampled.append({**r, "id": new_id})
                for j, lookup in enumerate(pred_sets):
                    mapped_predictions[j].extend({**p, "image_id": new_id} for p in lookup[r["id"]])
        # Preserve the 3-class estimand rather than changing AP class coverage.
        if {a["category_id"] for r in sampled for a in r["annotations"]} != required_classes:
            continue
        class_complete += 1
        scores = [coco_evaluate(sampled, p)[0]["AP"] for p in mapped_predictions]
        values.append(scores[0] if len(scores) == 1 else scores[1]-scores[0])
    if len(values) < 2:
        raise ValueError("Too few class-complete bootstrap samples")
    return {"metric": "AP" if predictions_b is None else "AP_B_minus_A", "seed": seed,
            "requested_repeats": repeats, "valid_repeats": class_complete, "groups": len(groups),
            "mean": float(np.mean(values)), "CI95_percentile": np.percentile(values, [2.5, 97.5]).tolist(),
            "samples": values, "caution": "Few filename-date groups imply unstable intervals; this is not proof of significance."}
