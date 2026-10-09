from __future__ import annotations

import csv
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .utils import save_json

RDD_CLASSES = {"D00": 1, "D10": 1, "D20": 1, "D40": 0}
COUNTRIES = {"Japan", "India", "Czech", "Norway", "United_States", "China", "UnitedStates", "USA"}


def import_rdd2022(root, output, subset="train"):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a NEW output directory for the external import")
    xmls = sorted(p for p in root.rglob("*.xml") if subset in p.relative_to(root).parts or
                  not ({"train", "test"} & set(p.relative_to(root).parts)))
    if not xmls:
        raise ValueError("No annotated Pascal VOC XMLs found. Unlabeled challenge test images cannot be scored.")
    lookup = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            lookup[p.stem].append(p)
    (output/"images").mkdir(parents=True)
    (output/"labels").mkdir()
    rows, converted, rejected = [], [], []
    try:
        for xml in xmls:
            tree = ET.parse(xml).getroot()
            filename = tree.findtext("filename") or (xml.stem+".jpg")
            candidates = lookup[Path(filename).stem]
            country = next((p for p in xml.relative_to(root).parts if p in COUNTRIES), Path(filename).stem.split("_")[0])
            compatible = [p for p in candidates if country in p.parts or p.stem.startswith(country)]
            candidates = compatible or candidates
            candidates = [p for p in candidates if subset in p.relative_to(root).parts or
                          not ({"train", "test"} & set(p.relative_to(root).parts))]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous/missing image for {xml}: {candidates}")
            source = candidates[0]
            with Image.open(source) as image:
                width, height = image.size
            name = country+"__"+source.stem+source.suffix.lower()
            if (output/"images"/name).exists():
                raise ValueError(f"External filename collision: {name}")
            lines = []
            for obj in tree.findall("object"):
                label = obj.findtext("name")
                if label not in RDD_CLASSES:
                    raise ValueError(f"Unknown RDD category {label} in {xml}; do not silently drop annotated objects")
                box = obj.find("bndbox")
                x1, y1, x2, y2 = [float(box.findtext(k)) for k in ["xmin", "ymin", "xmax", "ymax"]]
                # VOC uses 1-indexed inclusive bounds. Convert to continuous 0-based xyxy.
                x1, y1 = x1-1, y1-1
                x1, x2, y1, y2 = max(0, x1), min(width, x2), max(0, y1), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Degenerate RDD annotation in {xml}")
                values = [(x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height]
                lines.append(str(RDD_CLASSES[label])+" "+" ".join(f"{v:.9f}" for v in values))
            shutil.copy2(source, output/"images"/name)
            (output/"labels"/(Path(name).stem+".txt")).write_text("\n".join(lines)+("\n" if lines else ""))
            rows.append([name, "country:"+country])
            converted.append({"file": name, "source_image": str(source.relative_to(root)),
                              "source_xml": str(xml.relative_to(root)), "country": country, "objects": len(lines)})
    except Exception as exc:
        rejected.append(str(exc))
        save_json(output/"conversion_audit.json", {"converted": converted, "errors": rejected, "incomplete": True})
        raise
    with open(output/"groups.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file_name", "group_id"])
        writer.writerows(rows)
    save_json(output/"conversion_audit.json", {"source": "https://arxiv.org/abs/2209.08538", "subset": subset,
              "mapping": RDD_CLASSES, "converted": converted, "errors": [], "incomplete": False,
              "coordinate_convention": "VOC 1-indexed inclusive to continuous 0-based xyxy",
              "limitations": ["Manhole is absent; report shared pothole/crack AP separately.",
                              "Country grouping supports clustered external uncertainty, not capture-session provenance.",
                              "No download is performed; supply legally obtained, labeled RDD2022 files."]})
    print(f"Imported {len(converted)} labeled RDD2022 images into {output}")
