# Handoff: Faster R-CNN baseline stops from non-finite gradients

## Scope and evidence

Investigated run: `runs/paper_study/fasterrcnn_baseline_seed0`.

The run completed 12 full epochs, then stopped at the first training batch of epoch 13. The terminal exception is:

```text
RuntimeError: The total norm of order 2.0 for gradients from `parameters` is non-finite, so it cannot be clipped.
```

It is raised at `roadresearch/engine.py:217` by:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"], error_if_nonfinite=True)
```

The final completed checkpoint is `last.pt`, with `epoch: 11` (zero-based, so epoch 12), `finished: false`, 12 history rows, and best validation AP `0.14500957244777493` at epoch 10. It remains usable as the diagnostic starting point.

The CUDA/NVML message at the start of `console.log` is only a warning. The model trained for 12 epochs on the RTX 4090, so it is not the terminating error.

## Cause

The immediate cause is at least one `parameter.grad` being `NaN` or `Inf` after backward and `scaler.unscale_(optimizer)`. The scalar training loss was checked before backward and was finite, so this is a **non-finite gradient with a finite forward loss**, not a logged non-finite loss.

This is exposed as a control-flow bug in the AMP loop:

1. `GradScaler` is intended to detect non-finite unscaled gradients and skip that optimizer update, then lower its scale on `scaler.update()`.
2. The code calls `clip_grad_norm_(..., error_if_nonfinite=True)` between `unscale_` and `scaler.step()`.
3. Therefore the clipping check raises before `GradScaler.step()` can safely skip the overflowing batch or `GradScaler.update()` can reduce the scale.

The checkpoint scaler state supports AMP scale overflow as a likely trigger: `scale = 262144.0` and `_growth_tracker = 188` with default growth interval 2000. This does not prove that every affected gradient is only scale-related; a numerical instability in a particular batch or operation is still possible. Current logging does not record the batch image IDs, parameter names, individual gradient values, or scaler scale at the failure, so it cannot identify the originating tensor from this run alone.

## Required code change

Modify the training step around `roadresearch/engine.py:209-219` so finite-gradient validation happens before clipping, and AMP gets to handle overflow.

Suggested behavior:

1. Keep the existing finite-loss check before backward.
2. Run `scaler.scale(loss).backward()` and `scaler.unscale_(optimizer)`.
3. Inspect every non-`None` gradient with `torch.isfinite(grad).all()`.
4. If all gradients are finite, apply `clip_grad_norm_` and retain `error_if_nonfinite=True` as a defensive invariant.
5. If any gradient is non-finite and AMP is enabled:
   - do **not** clip;
   - call `scaler.step(optimizer)` so its recorded `found_inf` skips the optimizer update;
   - call `scaler.update()` so the scale backs off;
   - record a structured overflow event, then continue to the next batch.
6. If AMP is disabled, raise `FloatingPointError` with the same diagnostics, because a normal optimizer step would otherwise write non-finite values into model weights.

Do not "fix" this by changing `error_if_nonfinite=False`. Clipping a non-finite norm can propagate `NaN` values into gradients and weights. Do not silently discard the event either: persist it so persistent model/data instability is visible.

Illustrative shape of the guarded section (adapt to project style):

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
bad_grads = [name for name, p in model.named_parameters()
             if p.grad is not None and not torch.isfinite(p.grad).all()]
if bad_grads:
    event = {"epoch": epoch + 1, "step": steps + 1,
             "image_ids": [int(t["image_id"]) for t in targets],
             "losses": {k: float(v.detach()) for k, v in losses.items()},
             "scale": float(scaler.get_scale()), "parameters": bad_grads}
    # append event to a JSONL file or checkpoint state
    if not scaler.is_enabled():
        raise FloatingPointError(f"Nonfinite gradients: {event}")
else:
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"],
                                   error_if_nonfinite=True)
scaler.step(optimizer)  # skips update when AMP found non-finite gradients
scaler.update()
```

Avoid accumulating the full parameter-name list indefinitely in checkpoint history. Store a count plus a bounded sample of names, image IDs, loss components, scale before/after update, and an epoch-level `skipped_amp_steps` count. Add a configurable or documented maximum for consecutive skipped AMP steps; raise with the saved diagnostic after the threshold, so a persistent fault does not turn into a superficially successful training run.

## Validation requested

Add focused tests near `tests/test_research.py`:

1. A finite-gradient step still clips and updates parameters.
2. Simulate a non-finite gradient while AMP is enabled. Verify: clipping is not invoked; optimizer parameters are unchanged for that batch; `scaler.update()` is invoked; an event includes epoch, step, image IDs, scaler scale, and a parameter-name sample; training continues to the following finite batch.
3. Simulate the same condition with AMP disabled and assert `FloatingPointError` plus diagnostics.
4. Extend the existing resume-equivalence coverage to include saved overflow counters/events and ensure resumed state remains deterministic at an epoch boundary.

Run the focused tests and then an RTX 4090 reproduction beginning from the existing `last.pt` if code policy permits it. Confirm that the next overflowing batch produces a diagnostic and a reduced scaler scale, and that later batches have finite weights/losses. Checkpoint after a completed epoch before treating the run as recovered.

## Run and provenance policy

The current resume logic verifies hashes in `source_sha256.json` and rejects changed source code. After changing `roadresearch/engine.py`, `runs/paper_study/fasterrcnn_baseline_seed0 --resume` will correctly fail with `Code changed since run started`.

Use a new output root for runs produced by the fix, preserving the failed directory as evidence. A fresh run is the clean paper-study option. If an engineer adds an explicit migration/recovery workflow that starts from the epoch-12 checkpoint, it must preserve the original checkpoint and clearly label the resulting run as migrated, including the source-hash discontinuity and overflow events; do not overwrite the original artifacts.

## Files relevant to the fix

- `runs/paper_study/fasterrcnn_baseline_seed0/console.log` — traceback and epoch metrics.
- `runs/paper_study/fasterrcnn_baseline_seed0/failure.json` — machine-readable copy of the same failure.
- `runs/paper_study/fasterrcnn_baseline_seed0/last.pt` — epoch-12 state and AMP scaler state.
- `roadresearch/engine.py:209-219` — failing training-step ordering.
- `tests/test_research.py:154-180` — current epoch-boundary resume test to extend.

