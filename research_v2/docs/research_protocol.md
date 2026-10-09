# Research questions and claims — v2

## Prespecified questions

- RQ1: Does native crack-focused crop training improve crack AP and shared damage AP,
  relative to the same Faster R-CNN architecture and training budget?
- RQ2: Does auxiliary enclosure-shape supervision add value beyond crop training?
- RQ3: How do the accuracy/latency tradeoffs compare with FCOS, YOLO11s and RT-DETRv2?
- RQ4: Do observed gains persist on an independent, untouched external dataset?

Primary endpoint: COCO bbox AP50:95. Secondary: crack AP, AP50, shared pothole/crack AP,
F1 at validation-selected thresholds, object-stratum recall, end-to-end batch-1 latency.
Fixed seeds: 0, 42, 2026. Paired whole-group bootstrap: 1,000 samples per seed/comparison.
Do not select the best seed. Report mean and sample SD for every method.

## Novelty decision

The related-work matrix is a source-grounded starting comparison, not a systematic review
or a guarantee that no prior paper implements the same idea. Access levels are explicit.
Crop, slicing, FPN, geometry losses and structural auxiliary supervision already have prior art.
This dataset contains AABB-like quadrilaterals; enclosure geometry is derived from detection
labels, not extra physical crack-direction evidence. Do not call it oriented crack supervision.

Current claim candidate: a reproducible, leakage-audited comparison of crack-detail training
and enclosure-shape ablations on this particular dataset, with external transfer and cost
analysis. Its publication value depends on meaningful, stable findings and a literature review.
If all differences are negligible, report that result and investigate a stronger research question;
do not manufacture novelty from the number of modules or runs.

Before an algorithmic-novelty claim, compare full methods of the closest papers, verify target
definitions and architectural differences, and show isolated ablation gains on at least two
data domains. For an applied benchmark paper, emphasize independent metadata, annotation
audit, complete reproducibility and practically useful transfer findings.

## Development and final test

`develop` trains and calibrates only on training/validation. It never evaluates final test.
Choose tuning candidates and inference settings from validation; use a new output directory
for revised experiments. `freeze-study` records the exact config, manifest, models,
validation thresholds, comparisons and optional external manifest before final test.
`finalize-study` verifies these hashes, opens the preregistered tests, and generates reports
and paired bootstrap intervals. This is local preregistration, not a public registered report.

The default paper config requires this workflow; direct final evaluation is rejected without
the frozen protocol. Existing v1 results must not be repurposed as an untouched v2 final test.

## Dataset independence

Perceptual hashes detect review candidates, not proof of capture identity. Inspect contact
sheets and record `same_session` / `different_scene`, a reviewer and evidence. Confirmed
same-session links are merged before splitting. Near-duplicate audit is repeated after a new
split. Uncertain cross-split candidates remain a limitation, not an automatically cleared check.

Filename date is only an extraction/capture proxy. True route/session/device metadata must
come from the original data provider or documented capture records. `--verified-metadata`
is an explicit declaration backed by those records; do not set it merely because a CSV exists.
If evidence is unavailable, freeze an `--exploratory` study and state the unresolved limitation.
The code does not infer device or location labels from visual appearance.

## External data

RDD2022 importer maps D00/D10/D20 to crack and D40 to pothole, preserving source provenance.
Manhole is absent. Report `AP_shared_damage` for the fixed pothole+crack label intersection.
Raw source three-class AP and external two-class AP must not be treated as the same endpoint.
External threshold stays fixed from source validation; no external tuning/fine-tuning in this
zero-shot protocol. Country grouping supports country-cluster bootstrap but is not verified
capture-session metadata. Thailand data can use the same images/labels importer contract.

## Statistical limits

The group bootstrap conditions on fitted models and captures clustered test variation;
seed SD reflects training variation. They cannot be substituted for one another.
Percentile intervals from a few date groups are unstable and do not establish significance.
Bootstrap draws lacking any originally observed evaluation class are skipped and counted.
The prespecified comparisons are descriptive estimates; this version does not claim familywise
significance across many comparisons or combine seed-level intervals into a fabricated p-value.

Brightness/contrast failure strata are pixel statistics, not weather annotations. Area/aspect
strata use native-image enclosures and frozen deployment thresholds. Include difficult examples
and missed objects in the paper; never show only successful detections.
