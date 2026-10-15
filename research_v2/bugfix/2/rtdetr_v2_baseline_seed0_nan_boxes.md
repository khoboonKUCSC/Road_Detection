# Handoff: RT-DETRv2 baseline produces NaN boxes during training

## Scope and evidence

Investigated run: `runs/paper_study_recovered/rtdetr_v2_baseline_seed0`.

The terminating exception is raised inside Hugging Face RT-DETRv2's Hungarian matcher while it computes GIoU:

```text
ValueError: boxes1 must be in [x0, y0, x1, y1] (corner) format,
but got tensor([[nan, nan, nan, nan], ...], device='cuda:0')
```

`boxes1` is `center_to_corners_format(out_bbox)` in `transformers/loss/loss_rt_detr.py`. Therefore the model's *predicted* boxes (`out_bbox`) are already `NaN`; this is not an invalid-corner-order error from a ground-truth annotation.

The last usable checkpoint is `last.pt`:

- `epoch: 1` (zero based), two completed history rows, `finished: false`;
- AMP scaler scale is `256.0`;
- its stored model parameters and optimizer tensors are finite when loaded on CPU;
- it is not a clean recovery checkpoint: the next training forward pass fails before a new optimizer step/checkpoint is made.

The log establishes the numerical sequence:

1. The run uses the shared training setting `amp: true`, which enters `torch.autocast(..., dtype=torch.float16)` around `model(images, targets)` in `roadresearch/engine.py`.
2. In epoch 1, the first eight batches produced finite scalar losses but non-finite gradients. The AMP guard correctly skipped all eight and backed GradScaler off from `65536` to `256` (see `overflow_events.jsonl`). Losses were very large: about 443--723.
3. Epoch 1 completed with 8 skipped AMP steps; epoch 2 completed with none.
4. At the start of the following epoch, `RTDetrV2ForObjectDetection.forward()` produces NaN box logits. The matcher detects the NaNs and throws the reported `ValueError`.

Relevant code and configuration:

- `roadresearch/adapters.py:104-123` converts valid XYXY targets to COCO XYWH and calls `self.network(**inputs)`.
- `roadresearch/engine.py:219-224` runs that forward/loss under global CUDA FP16 autocast and only checks whether the returned scalar loss is finite.
- `configs/rtx4090_recovered.yaml` / run `config.json`: RT-DETRv2 R50-vd, batch 4, LR `1e-4`, FP16 AMP enabled.
- `roadresearch/data.py:20-67` rejects non-finite labels and degenerate boxes while preparing the manifest. This, plus the traceback identifying `out_bbox`, makes malformed labels an unsupported primary cause.

The warning about mismatched COCO classifier-head shapes is expected because the pretrained 80-class head is intentionally replaced by a 3-class head. The `grid_sampler_2d_backward_cuda` determinism warning and missing `hf_xet` package are also not the terminating error.

## Root cause

The immediate root cause is **numerical instability of RT-DETRv2 under the project-wide CUDA FP16 autocast policy**. FP16 is applied to the entire Hugging Face RT-DETRv2 forward/loss path, including deformable-attention/grid-sampling and box/matching computations. The repeated initial overflows are direct evidence that this path is outside a stable FP16 range for this model/data/hyperparameter combination. Eventually a later FP16 forward produces NaN predicted boxes, and the exception only surfaces downstream in GIoU validation.

The existing AMP-overflow fix is working for the earlier backward-gradient failures: it skipped unsafe optimizer updates and reduced the scale. It cannot repair a later **forward-pass** NaN, because no scalar finite loss, gradients, or optimizer step exists at that point. Treating this as a target-box-format bug, clipping labels again, or merely raising the consecutive-overflow limit will not fix it.

## Required implementation

Make mixed precision a per-architecture policy. For `rtdetr_v2_r50vd`, use full FP32 for the Hugging Face network forward and loss during training. Keep the outer training-loop GradScaler behavior unchanged; scaling FP32 gradients is safe and preserves the common loop contract.

Preferred minimal change in `RTDETRv2Detector.forward` (`roadresearch/adapters.py`):

1. Prepare processor inputs exactly as today, retaining FP32 `pixel_values`.
2. Detect the network device and call `self.network(**inputs)` inside a nested disabled autocast context when CUDA autocast is active, e.g. `torch.autocast(device_type=device.type, enabled=False)`.
3. Return `outputs.loss` unchanged. Do not cast it down to FP16.
4. Apply the same policy to RT-DETR inference/evaluation (or centralize it in a small adapter helper) so that validation cannot reintroduce FP16 NaN boxes.

An explicit architecture capability such as `uses_fp32_forward = True` / `autocast_enabled = False` is preferable to a fragile class-name special case in `engine.py`. The engine can then combine its global AMP setting with the model capability when entering autocast. Either design is acceptable if all RT-DETR training and prediction calls run in FP32 and the other detectors retain their present AMP behavior.

Do **not** silently replace NaNs using `nan_to_num`, discard targets, relax the Transformers GIoU check, or continue after a non-finite model output. Those would conceal corrupted predictions and invalidate training.

### Defensive diagnostics

Add an early guard around the RT-DETR output/loss, before the Transformers matcher error becomes opaque:

- Check `outputs.pred_boxes` (when available) and `outputs.loss` with `torch.isfinite`.
- On failure, raise `FloatingPointError` containing epoch/batch context supplied by the engine (or enough adapter context), AMP/autocast mode, input image IDs, and a bounded sample/count of non-finite tensors.
- Do not serialize image pixel data or unbounded tensor dumps.

This guard is diagnostic only; FP32 execution is the corrective action.

## Configuration and recovery policy

Do not resume the failed directory after editing source. Resume correctly rejects altered code through `source_sha256.json`; also, its current `last.pt` is an unstable FP16 trajectory.

Run a new study output root after the implementation, for example a new `runs/paper_study_rtdetr_fp32/` root, with the same seed/config except for recorded RT-DETR precision policy. Preserve `runs/paper_study_recovered/rtdetr_v2_baseline_seed0` unchanged as failure evidence. If a migration command is used, label the result as migrated and do not present it as a seed-comparable clean paper run.

FP32 may need more VRAM and run slower. If memory is insufficient, reduce RT-DETR batch size only in a newly declared configuration and record it; never silently alter the recovered run. A later BF16 option may be investigated separately on the RTX 4090, but must first pass the stability tests below and be recorded as a distinct precision policy. It is not part of this immediate fix.

## Tests and acceptance criteria

Add focused coverage (the current RT-DETR adapter test in `tests/test_v2.py` only tests CPU FP32):

1. Unit-test the policy without CUDA: mock/inspect the adapter helper and verify RT-DETR requests disabled autocast while YOLO/torchvision models preserve the global setting.
2. CUDA regression test, when CUDA and Transformers are installed: construct the small RT-DETR config already used by `test_modern_adapters_loss_inference_and_reload`, invoke it inside an outer FP16 autocast context, and assert the adapter's training loss and `pred_boxes` are finite. Backpropagate and assert finite gradients.
3. CUDA inference regression: call the adapter in eval mode under outer FP16 autocast and assert finite prediction boxes/scores with valid XYXY ordering.
4. Preserve existing AMP overflow tests in `tests/test_research.py`; confirm non-RT-DETR detectors still use their configured AMP path.
5. On the RTX 4090, start a fresh RT-DETR seed-0 run and complete at least several epochs including validation. Require: no `NaN`/`Inf` in loss, model parameters, gradients, predicted boxes, or optimizer state; zero forward-nonfinite events; valid checkpoints and evaluation output. Complete the intended 80-epoch run before using it in paper comparisons.

## Files for the implementing agent

- `runs/paper_study_recovered/rtdetr_v2_baseline_seed0/failure.json` — machine-readable traceback.
- `runs/paper_study_recovered/rtdetr_v2_baseline_seed0/console.log` — epoch metrics and warning context.
- `runs/paper_study_recovered/rtdetr_v2_baseline_seed0/overflow_events.jsonl` — eight initial AMP-gradient overflows and scale backoff.
- `runs/paper_study_recovered/rtdetr_v2_baseline_seed0/last.pt` — diagnostic checkpoint after epoch 2.
- `roadresearch/adapters.py` — RT-DETR input conversion and network call to change.
- `roadresearch/engine.py` — global autocast context and existing finite-loss/gradient safeguards.
- `tests/test_v2.py` and `tests/test_research.py` — relevant regression-test locations.
