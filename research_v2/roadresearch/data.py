from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .utils import CLASSES, digest, read_json, save_json


def parse_label(path, width, height, warnings=None):
    """Convert true quadrilateral / YOLO AABB labels; never invent segmentation masks."""
    annotations = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) not in (5, 9):
            raise ValueError(f"{path}:{number}: expected 5 or 9 values, got {len(parts)}")
        values = np.asarray([float(x) for x in parts], dtype=np.float64)
        cls = int(values[0])
        if values[0] != cls or cls not in range(3) or not np.isfinite(values).all():
            raise ValueError(f"{path}:{number}: invalid class or nonfinite coordinate")
        clipped = bool(np.any(values[1:] < 0) or np.any(values[1:] > 1))
        if clipped:
            if warnings is None:
                raise ValueError(f"{path}:{number}: normalized coordinates outside [0,1]")
            warnings.append({"label": str(path), "line": number, "action": "clip_AABB_to_image_disable_geometry",
                             "original_values": values.tolist()})
        polygon = None
        if len(parts) == 9:
            points = values[1:].reshape(4, 2) * [width, height]
            # Reject degenerate, concave, or self-intersecting four-point annotations.
            edges = np.roll(points, -1, axis=0) - points
            following = np.roll(edges, -1, axis=0)
            crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
            ordered_convex = np.all(crosses > 1e-7) or np.all(crosses < -1e-7)
            if not ordered_convex and warnings is None:
                raise ValueError(f"{path}:{number}: quadrilateral must be ordered, convex and nondegenerate")
            if not ordered_convex:
                warnings.append({"label": str(path), "line": number, "action": "retain_AABB_disable_geometry_invalid_quad"})
            lo, hi = points.min(0), points.max(0)
            polygon = points.tolist() if ordered_convex and not clipped else None
        else:
            cx, cy, bw, bh = values[1:] * [width, height, width, height]
            lo, hi = np.array([cx - bw / 2, cy - bh / 2]), np.array([cx + bw / 2, cy + bh / 2])
            if (np.any(lo < -1e-4) or np.any(hi > np.array([width, height]) + 1e-4)) and warnings is None:
                raise ValueError(f"{path}:{number}: YOLO box extends outside image")
            if warnings is not None and not clipped and (np.any(lo < 0) or np.any(hi > [width, height])):
                warnings.append({"label": str(path), "line": number, "action": "clip_AABB_to_image",
                                 "original_values": values.tolist()})
        lo, hi = np.maximum(lo, 0), np.minimum(hi, [width, height])
        if np.any(hi <= lo):
            if warnings is None:
                raise ValueError(f"{path}:{number}: empty box")
            warnings.append({"label": str(path), "line": number, "action": "drop_degenerate_annotation",
                             "original_values": values.tolist()})
            continue
        annotations.append({"category_id": cls + 1, "bbox_xyxy": [*lo.tolist(), *hi.tolist()],
                            "quadrilateral": polygon})
    return annotations


def inferred_group(stem):
    match = re.search(r"(20\d{2})-(\d{2})-(\d{2})", stem) or re.match(r"(20\d{2})(\d{2})(\d{2})_", stem)
    if match:
        return "filename_date:" + "-".join(match.groups())
    # All frames without trustworthy metadata remain together. No arbitrary chunks.
    return "unknown_session"


class UnionFind:
    def __init__(self):
        self.parents = {}

    def find(self, x):
        self.parents.setdefault(x, x)
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def split_records(records, seed, ratios):
    """Search deterministic whole-group assignments, balancing image and class counts."""
    groups = sorted({r["group"] for r in records})
    if len(groups) < 3:
        raise ValueError("At least 3 independent groups are needed. Supply verified groups.csv; never split a session into chunks.")
    stats = np.asarray([[sum(r["group"] == g for r in records)] +
                        [sum(a["category_id"] == c for r in records if r["group"] == g
                             for a in r["annotations"]) for c in range(1, 4)] for g in groups], dtype=float)
    rng = np.random.default_rng(seed)
    best = None
    target = np.asarray(ratios)
    for _ in range(20000):
        order = rng.permutation(len(groups))
        assignment = rng.choice(3, size=len(groups), p=target)
        assignment[order[:3]] = np.arange(3)
        totals = np.asarray([stats[assignment == i].sum(0) for i in range(3)])
        if np.any(totals[:, 1:] == 0):
            continue
        fractions = totals / np.maximum(stats.sum(0), 1)
        cost = float(np.mean((fractions - target[:, None]) ** 2))
        if best is None or cost < best[0]:
            best = cost, assignment.copy()
    if best is None:
        raise ValueError("Cannot put all 3 classes in each split using whole groups. Add data or use a different study protocol.")
    mapping = {g: ["train", "val", "test"][int(i)] for g, i in zip(groups, best[1])}
    for record in records:
        record["split"] = mapping[record["group"]]
    return mapping


def prepare(data_root, output, groups_csv=None, seed=2026, ratios=(0.70, 0.15, 0.15), external=False, source=None,
            reviewed_links=None, verified_metadata=False):
    root, output = Path(data_root).resolve(), Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError(f"{output}/manifest.json already exists. Reuse it or choose a NEW output directory.")
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError("ratios must contain 3 positive numbers summing to 1")
    metadata = {}
    if verified_metadata and not groups_csv:
        raise ValueError("--verified-metadata requires a capture/route groups CSV backed by real evidence")
    if groups_csv:
        with open(groups_csv, newline="") as stream:
            for row in csv.DictReader(stream):
                if not row.get("file_name") or not row.get("group_id"):
                    raise ValueError("groups CSV requires file_name,group_id and nonempty values")
                if row["file_name"] in metadata:
                    raise ValueError("duplicate file_name in groups CSV")
                metadata[row["file_name"]] = row["group_id"]
    files = sorted(p for p in (root / "images").iterdir()
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise ValueError("No images found directly under data_root/images. Existing train/val subfolders are intentionally ignored.")
    uf, records, seen, duplicates, errors = UnionFind(), [], {}, [], []
    warnings, conflicts, conflict_hashes = [], [], set()
    for path in files:
        try:
            label = root / "labels" / (path.stem + ".txt")
            if not label.exists():
                raise ValueError("missing label; add an empty file only for a verified background image")
            with Image.open(path) as image:
                image = image.convert("RGB")
                width, height = image.size
                pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if metadata and path.name not in metadata:
                raise ValueError("missing group in groups CSV")
            group = "verified:" + metadata[path.name] if metadata else inferred_group(path.stem)
            uf.find(group)
            annotations = parse_label(label, width, height, warnings)
            record = {"id": len(records) + 1, "file_name": path.name,
                      "image_path": path.relative_to(root).as_posix(),
                      "label_path": label.relative_to(root).as_posix(), "width": width, "height": height,
                      "sha256": digest(path), "label_sha256": digest(label), "pixel_sha256": pixel_hash,
                      "source_group": group, "group": group, "annotations": annotations}
            if pixel_hash in seen:
                old = seen[pixel_hash]
                uf.union(old["group"], group)
                if old["annotations"] != annotations:
                    conflict_hashes.add(pixel_hash)
                    conflicts.append({"file_a": old["file_name"], "file_b": path.name,
                                      "sha256_a": old["sha256"], "sha256_b": record["sha256"],
                                      "label_sha256_a": old["label_sha256"], "label_sha256_b": record["label_sha256"],
                                      "annotations_a": old["annotations"], "annotations_b": annotations,
                                      "action": "quarantine_all_images_in_conflicting_duplicate_cluster"})
                else:
                    duplicates.append({"kept": old["file_name"], "removed": path.name})
            else:
                seen[pixel_hash] = record
                records.append(record)
        except (ValueError, OSError) as exc:
            errors.append({"file": str(path), "error": str(exc)})
    output.mkdir(parents=True, exist_ok=True)
    quarantined = [r for r in records if r["pixel_sha256"] in conflict_hashes]
    records = [r for r in records if r["pixel_sha256"] not in conflict_hashes]
    for index, record in enumerate(records, 1):
        record["id"] = index
    audit = {"source": source or ("user-supplied external dataset; source unspecified" if external else
                                  "https://zenodo.org/records/17834373"), "source_files": len(files),
             "unique_images": len(records), "duplicate_images_removed": duplicates, "errors": errors,
             "annotation_adjustments": warnings, "conflicting_duplicate_clusters": conflicts,
             "quarantined_canonical_images": [r["file_name"] for r in quarantined],
             "ignored_subfolders": True,
             "orphan_labels": sorted(p.name for p in (root / "labels").glob("*.txt")
                                     if p.stem not in {x.stem for x in files}),
             "grouping": "verified CSV" if metadata else "filename date (proxy, NOT verified capture session); unknown files held together",
             "limitations": ["Exact decoded-image duplicates are removed; near duplicates and repeated roads require visual/metadata review.",
                             "Filename timestamps may be extraction times; date grouping does not prove route or device independence.",
                             "Quadrilaterals are enclosure annotations, not pixel-level damage masks."]}
    save_json(output / "audit.json", audit)
    if errors:
        raise ValueError(f"Audit failed for {len(errors)} image(s). Read {output}/audit.json. No invalid data silently dropped.")
    for record in records:
        record["group"] = uf.find(record["group"])
    if reviewed_links:
        lookup = {r["file_name"]: r for r in records}
        with open(reviewed_links, newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("decision") != "same_session":
                    continue
                if not row.get("reviewer") or not row.get("evidence"):
                    raise ValueError("Reviewed same-session links need reviewer and evidence")
                if row["file_a"] not in lookup or row["file_b"] not in lookup:
                    raise ValueError("Reviewed link refers to a quarantined/absent image")
                uf.union(lookup[row["file_a"]]["group"], lookup[row["file_b"]]["group"])
        for record in records:
            record["group"] = uf.find(record["group"])
        audit["reviewed_links_sha256"] = digest(reviewed_links)
    if external:
        for record in records:
            record["split"] = "external"
        mapping = {r["group"]: "external" for r in records}
    else:
        mapping = split_records(records, seed, ratios)
    splits = ["external"] if external else ["train", "val", "test"]
    audit["splits"] = {s: {"images": sum(r["split"] == s for r in records),
                           "groups": sorted(g for g, value in mapping.items() if value == s),
                           "instances": {CLASSES[c - 1]: sum(a["category_id"] == c for r in records
                                                           if r["split"] == s for a in r["annotations"])
                                         for c in range(1, 4)}} for s in splits}
    audit["resolution_counts"] = dict(Counter(f"{r['width']}x{r['height']}" for r in records))
    geometry_counts = Counter()
    for record in records:
        for annotation in record["annotations"]:
            points = annotation["quadrilateral"]
            if points is None:
                geometry_counts["disabled"] += 1
            else:
                points = np.asarray(points)
                edges = np.roll(points, -1, axis=0)-points
                aligned = all(abs(e[0]) < 0.002 or abs(e[1]) < 0.002 for e in edges)
                geometry_counts["axis_aligned" if aligned else "oriented"] += 1
    audit["geometry_annotation_counts"] = dict(geometry_counts)
    if geometry_counts["oriented"] == 0:
        audit["limitations"].append("All eligible quadrilaterals are axis-aligned; auxiliary orientation is box-shape supervision, not physical crack direction.")
    if not external:
        actual = [audit["splits"][s]["images"] / len(records) for s in splits]
        audit["requested_ratios"] = list(ratios)
        audit["actual_ratios"] = actual
        audit["ratio_warning"] = any(abs(a - b) > 0.10 for a, b in zip(actual, ratios))
    save_json(output / "audit.json", audit)
    manifest = {"schema_version": 1, "classes": CLASSES, "split_seed": seed,
                "source": audit["source"], "grouping": audit["grouping"],
                "grouping_metadata_verified": verified_metadata, "records": records}
    save_json(output / "manifest.json", manifest)
    with open(output / "splits.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "file_name", "group", "source_group", "split", "sha256", "label_sha256"])
        writer.writeheader()
        writer.writerows({key: r[key] for key in writer.fieldnames} for r in records)
    for split in splits:
        save_json(output / f"{split}_coco.json", to_coco([r for r in records if r["split"] == split]))
    print(f"Prepared {len(records)} unique images. Splits: {audit['splits']}")
    return manifest


def to_coco(records):
    images, annotations = [], []
    for record in records:
        images.append({"id": record["id"], "file_name": record["file_name"],
                       "width": record["width"], "height": record["height"]})
        for item in record["annotations"]:
            x1, y1, x2, y2 = item["bbox_xyxy"]
            annotations.append({"id": len(annotations) + 1, "image_id": record["id"],
                                "category_id": item["category_id"], "bbox": [x1, y1, x2-x1, y2-y1],
                                "area": (x2-x1)*(y2-y1), "iscrowd": 0})
    return {"info": {"description": "Road damage axis-aligned bbox evaluation"}, "images": images,
            "annotations": annotations, "categories": [{"id": i+1, "name": name} for i, name in enumerate(CLASSES)]}


def verify_manifest(manifest, root):
    for record in manifest["records"]:
        for path_key, hash_key in [("image_path", "sha256"), ("label_path", "label_sha256")]:
            path = Path(root) / record[path_key]
            if not path.is_file() or digest(path) != record[hash_key]:
                raise ValueError(f"Data changed or missing: {path}. Create a NEW manifest for changed data.")
    split_groups = defaultdict(set)
    hashes = defaultdict(set)
    for record in manifest["records"]:
        split_groups[record["group"]].add(record["split"])
        hashes[record["pixel_sha256"]].add(record["split"])
    if any(len(v) > 1 for v in list(split_groups.values()) + list(hashes.values())):
        raise ValueError("Manifest contains groups or exact duplicates across splits")


def geometry_target(points):
    points = np.asarray(points, dtype=float)
    centered = points - points.mean(0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(points))
    minor, major = np.maximum(eigenvalues, 1e-6)
    direction = eigenvectors[:, 1]
    theta = np.arctan2(direction[1], direction[0])
    elongation = np.clip(np.log(np.sqrt(major / minor)), 0, 4) / 4
    anisotropy = (major - minor) / (major + minor)
    return [float(elongation), float(np.sin(2*theta)), float(np.cos(2*theta))], float(anisotropy)


class RoadDataset(Dataset):
    def __init__(self, manifest, root, split, augment=False, limit=None, crop_probability=0., crop_size=256):
        self.records = [r for r in manifest["records"] if r["split"] == split]
        if limit is not None:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f"No records in split {split}")
        self.root, self.augment = Path(root), augment
        if not 0 <= crop_probability <= 1 or crop_size <= 0:
            raise ValueError("crop_probability must be in [0,1] and crop_size positive")
        self.crop_probability, self.crop_size = crop_probability, crop_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["image_path"]) as image:
            image = image.convert("RGB")
        annotations = record["annotations"]
        crack_annotations = [a for a in annotations if a["category_id"] == 2]
        if self.augment and self.crop_probability > 0 and crack_annotations and torch.rand(()).item() < self.crop_probability:
            chosen = crack_annotations[int(torch.randint(len(crack_annotations), ()).item())]
            box = chosen["bbox_xyxy"]
            width, height = image.size
            cw, ch = min(self.crop_size, width), min(self.crop_size, height)
            jitter_x, jitter_y = (torch.rand(2).numpy()-0.5) * np.array([cw, ch]) * 0.4
            x = int(np.clip((box[0]+box[2])/2-cw/2+jitter_x, 0, width-cw))
            y = int(np.clip((box[1]+box[3])/2-ch/2+jitter_y, 0, height-ch))
            image = image.crop((x, y, x+cw, y+ch))
            annotations = []
            for item in record["annotations"]:
                original = np.asarray(item["bbox_xyxy"])-[x, y, x, y]
                clipped = np.clip(original, [0, 0, 0, 0], [cw, ch, cw, ch])
                if np.any(clipped[2:] <= clipped[:2]):
                    continue
                points = item["quadrilateral"]
                if points is not None and np.allclose(original, clipped):
                    points = (np.asarray(points)-[x, y]).tolist()
                else:
                    points = None
                annotations.append({**item, "bbox_xyxy": clipped.tolist(), "quadrilateral": points})
        flip = self.augment and torch.rand(()).item() < 0.5
        if self.augment:
            image = ImageEnhance.Brightness(image).enhance(0.8 + torch.rand(()).item() * 0.4)
            image = ImageEnhance.Contrast(image).enhance(0.8 + torch.rand(()).item() * 0.4)
        if flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        boxes, labels, geometry, valid, anisotropy = [], [], [], [], []
        for annotation in annotations:
            box = list(annotation["bbox_xyxy"])
            if flip:
                box = [image.width - box[2], box[1], image.width - box[0], box[3]]
            boxes.append(box)
            labels.append(annotation["category_id"])
            points = annotation["quadrilateral"]
            if points is not None:
                points = np.asarray(points).copy()
                if flip:
                    points[:, 0] = image.width - points[:, 0]
                target, reliability = geometry_target(points)
                geometry.append(target)
                anisotropy.append(reliability)
                valid.append(True)
            else:
                geometry.append([0., 0., 0.])
                anisotropy.append(0.)
                valid.append(False)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        return pil_to_tensor(image).float() / 255, {
            "boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(record["id"]), "area": (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            "geometry": torch.tensor(geometry, dtype=torch.float32).reshape(-1, 3),
            "geometry_valid": torch.tensor(valid, dtype=torch.bool),
            "anisotropy": torch.tensor(anisotropy, dtype=torch.float32),
        }


def collate(batch):
    return tuple(zip(*batch))
