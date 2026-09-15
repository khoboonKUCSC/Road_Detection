# Road Damage Dataset: Potholes, Cracks, and Manholes

This dataset contains images of road surfaces with potholes, cracks, and manholes, along with polygon-based, YOLO-compatible, and COCO JSON annotations. It is intended for object detection research and evaluation.

## Dataset Structure

```
annotations_coco.json
COCO-conversion-script.py
images/
labels/
labels-YOLO/
README.md
YOLO-conversion-script.py
```

- **images/**: contains all the images in JPG format.
- **labels/**: original polygon-based annotations. Each `.txt` file corresponds to an image and contains one line per object: `class_id x1 y1 x2 y2 x3 y3 x4 y4`

  - Coordinates are normalized in `[0,1]`.
  - Four points define the quadrilateral around each object (ordered and convex).

- **labels-YOLO/**: axis-aligned bounding boxes in YOLO format `(class_id, x_center, y_center, width, height)`, ready for YOLO training.
- **annotations_coco.json**: dataset annotations in COCO format, with bounding boxes `[x_min, y_min, width, height]` in pixels.
- **YOLO-conversion-script.py**: converts polygon-based labels (`labels/`) to YOLO format (`labels-YOLO/`).
- **COCO-conversion-script.py**: converts polygon-based labels (`labels/`) to COCO JSON format (`annotations_coco.json`).

## Classes

| Class ID | Name      |
|----------|-----------|
| 0        | Pothole   |
| 1        | Crack     |
| 2        | Manhole   |

## Annotation Format

- **Polygon labels (`labels/`)**: 4-point quadrilaterals `(x1,y1,...,x4,y4)` normalized.  
- **YOLO labels (`labels-YOLO/`)**: axis-aligned bounding boxes derived from polygons, format `(class_id, x_center, y_center, width, height)` normalized.  
- **COCO JSON (`annotations_coco.json`)**: bounding boxes in `[x_min, y_min, width, height]` format in pixels, with optional segmentation polygons.  
- **Conversion**: Use `YOLO-conversion-script.py` or `COCO-conversion-script.py` to convert polygons. This ensures full compatibility with YOLO, COCO-based frameworks, and other object detection/segmentation tools.

## Data Acquisition

- **Devices**: GoPro and smartphone cameras mounted on vehicles at ~1.2–1.5 m height, angled 10–15° relative to road plane.  
- **Original resolution**: 1920×1080 (GoPro), 1280×720 (smartphone). Downsampled to 640×360 for dataset release.  
- **Road types**: Urban, suburban, and rural roads across Region X and Y.  
- **Conditions**: February–March 2025, varying lighting (sunny, cloudy), mild precipitation.  
- **Vehicle speed**: 20–50 km/h.  
- **Inclusion/Exclusion**: Images with severe blur, occlusion, or nighttime conditions are excluded. Stabilized horizontal FOV mode used for all images.

## Usage

- **YOLO training**: use `labels-YOLO/` folder directly.  
- **COCO-based tasks**: use `annotations_coco.json` for object detection.  
- **Advanced usage**: original polygon labels in `labels/` can be used for oriented bounding boxes or segmentation tasks.  
- **Reproducibility**: conversion scripts are provided separately for YOLO and COCO JSON formats.

## Citation

If you use this dataset, please cite our work:

https://doi.org/10.5281/zenodo.17834373
