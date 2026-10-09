from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn

from .data import UnionFind
from .utils import read_json, save_json


def hashes(image):
    gray = image.convert("L")
    dh = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    dh = (dh[:, 1:] > dh[:, :-1]).reshape(-1)
    spectrum = dctn(np.asarray(gray.resize((32, 32)), dtype=float), norm="ortho")[:8, :8].reshape(-1)[1:]
    ph = spectrum > np.median(spectrum)
    def integer(bits):
        return sum(int(value) << i for i, value in enumerate(bits))
    thumb = np.asarray(gray.resize((64, 36)), dtype=np.float32)/255
    return integer(dh), integer(ph), thumb


def audit_similarity(manifest_path, data_root, output, phash_distance=8, dhash_distance=6, contact_pairs=60):
    """Candidate finding only. Hash matches never prove identical capture sessions."""
    manifest, output = read_json(manifest_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = manifest["records"]
    signatures = []
    for record in records:
        with Image.open(Path(data_root)/record["image_path"]) as image:
            signatures.append(hashes(image))
    candidates = []
    for i, (d1, p1, t1) in enumerate(signatures):
        for j in range(i):
            d2, p2, t2 = signatures[j]
            dd, pd = (d1^d2).bit_count(), (p1^p2).bit_count()
            if pd > phash_distance and dd > dhash_distance:
                continue
            mse = float(np.mean((t1-t2)**2))
            # Suppress weak low-detail hash matches, but retain very close hashes.
            if mse > .04 and min(pd, dd) > 2:
                continue
            a, b = records[j], records[i]
            candidates.append({"file_a": a["file_name"], "file_b": b["file_name"],
                               "id_a": a["id"], "id_b": b["id"], "split_a": a["split"], "split_b": b["split"],
                               "group_a": a["group"], "group_b": b["group"], "phash_distance": pd,
                               "dhash_distance": dd, "thumbnail_MSE": mse, "cross_split": a["split"] != b["split"],
                               "decision": "", "reviewer": "", "evidence": ""})
    candidates.sort(key=lambda r: (not r["cross_split"], min(r["phash_distance"], r["dhash_distance"]), r["thumbnail_MSE"]))
    fields = ["file_a", "file_b", "id_a", "id_b", "split_a", "split_b", "group_a", "group_b",
              "phash_distance", "dhash_distance", "thumbnail_MSE", "cross_split", "decision", "reviewer", "evidence"]
    with open(output / "review_pairs.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    save_json(output / "similarity_audit.json", {"manifest_sha256": __import__("hashlib").sha256(Path(manifest_path).read_bytes()).hexdigest(),
               "images": len(records), "candidate_pairs": len(candidates), "cross_split_candidates": sum(r["cross_split"] for r in candidates),
               "phash_distance": phash_distance, "dhash_distance": dhash_distance,
               "interpretation": "Similarity candidates require review; not an automatic determination of leakage or independent routes.",
               "grouping": manifest["grouping"], "pairs": candidates})
    lookup = {r["id"]: r for r in records}
    selected = candidates[:contact_pairs]
    for start in range(0, len(selected), 10):
        sheet = Image.new("RGB", (660, 220*min(10, len(selected)-start)), "white")
        draw = ImageDraw.Draw(sheet)
        for row, pair in enumerate(selected[start:start+10]):
            for col, key in enumerate(["id_a", "id_b"]):
                record = lookup[pair[key]]
                with Image.open(Path(data_root)/record["image_path"]) as image:
                    image = ImageOps.contain(image.convert("RGB"), (320, 180))
                sheet.paste(image, (col*330, row*220+35))
                draw.text((col*330+3, row*220+3), f"{record['id']} {record['split']} {record['group']}", fill="black")
            draw.text((3, row*220+18), f"pHash={pair['phash_distance']} dHash={pair['dhash_distance']} MSE={pair['thumbnail_MSE']:.4f}", fill="black")
        sheet.save(output/f"contact_sheet_{start//10:03d}.jpg")
    print(f"Similarity audit: {len(candidates)} candidates, {sum(r['cross_split'] for r in candidates)} cross split")
    return candidates


def resolve_groups(manifest_path, decisions_csv, output_csv):
    """Emit a full source-file mapping with reviewed same-session links merged."""
    manifest = read_json(manifest_path)
    lookup = {r["file_name"]: r for r in manifest["records"]}
    uf = UnionFind()
    for r in lookup.values():
        uf.find(r["group"])
    decisions = []
    with open(decisions_csv, newline="") as stream:
        for row in csv.DictReader(stream):
            if row["file_a"] not in lookup or row["file_b"] not in lookup:
                raise ValueError("Review contains a file absent from manifest")
            if row["decision"] not in {"same_session", "different_scene", "uncertain", ""}:
                raise ValueError("decision must be same_session, different_scene, uncertain, or blank")
            if row["decision"] in {"same_session", "different_scene"} and not (row.get("reviewer") and row.get("evidence")):
                raise ValueError("Decisions require reviewer and evidence; visual similarity is not verified capture metadata")
            if row["decision"] == "same_session":
                uf.union(lookup[row["file_a"]]["group"], lookup[row["file_b"]]["group"])
            decisions.append(row)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file_name", "group_id"])
        writer.writerows([name, uf.find(r["group"])] for name, r in sorted(lookup.items()))
    save_json(output.with_suffix(".review.json"), {"original_grouping": manifest["grouping"],
              "grouping_status": "reviewed similarity links; capture metadata remains unverified unless supplied separately",
              "decisions": decisions, "remaining_groups": len({uf.find(r['group']) for r in lookup.values()})})
