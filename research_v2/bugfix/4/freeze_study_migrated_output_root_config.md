# Bugfix handoff: `freeze-study` rejects a valid migrated run

## Observed failure

Step 4.2 in `how_to_summary.md` fails before a protocol is written:

```text
ValueError: Run config disagrees with frozen config:
.../runs/paper_study_recovered/fasterrcnn_baseline_seed0
```

The command itself is valid.  `--exploratory` affects only the capture-metadata
and unresolved-similarity checks; it cannot change run configuration matching.

## Evidence and root cause

`runs/paper_study_recovered/fasterrcnn_baseline_seed0/migration.json` confirms
that this is a migrated run.  Comparing its stored `config.json` with
`experiment_config(config_load("configs/rtx4090_recovered.yaml"),
fasterrcnn_baseline, 0)` shows exactly one difference:

| Field | Stored run config | Frozen recovered config |
| --- | --- | --- |
| `output_root` | `.../runs/paper_study` | `.../runs/paper_study_recovered` |

All other fields are equal.  This happens because `migrate_run()` intentionally
preserves the source run config while placing copied artifacts under a new
output root.  `train()` already explicitly accepts this one difference when
`migration.json` exists (see `roadresearch/engine.py`, resume validation), but
`freeze_study()` currently does a strict whole-dictionary equality check:

```python
if read_json(run / "config.json") != experiment_config(config, experiment, seed):
    raise ValueError(...)
```

Consequently the preregistration workflow is inconsistent with the pipeline's
own migration workflow.  This is a Python protocol-validation defect, not a
Bash syntax or shell-path defect.

## Required code change

Update `roadresearch/protocol.py` so the run-config comparison:

1. continues to require exact equality for ordinary (non-migrated) runs;
2. for a run containing `migration.json`, permits **only** the
   `output_root` difference; and
3. still rejects every other difference, including model, seed, manifest,
   training, evaluation, or study settings.

The comparison should mirror the established logic in `roadresearch/engine.py`
instead of duplicating a subtly different policy.  A small shared helper (for
example `configs_match_for_run(expected, actual, run_dir)`) is preferable if it
can be used by both `train()` and `freeze_study()` without widening the allowed
exception.

Illustrative behavior:

```python
expected = experiment_config(config, experiment, seed)
actual = read_json(run / "config.json")
matches = actual == expected
if not matches and (run / "migration.json").exists():
    matches = (
        {k: v for k, v in actual.items() if k != "output_root"}
        == {k: v for k, v in expected.items() if k != "output_root"}
    )
if not matches:
    raise ValueError(f"Run config disagrees with frozen config: {run}")
```

Do not solve this by editing the run's stored `config.json`, changing the
recovered YAML back to `runs/paper_study`, removing `migration.json`, or
disabling config checks.  Those approaches either corrupt provenance, point the
protocol at the wrong directory, or allow a different experiment to be
registered.

## Recommended validation hardening

When using the migration exception, parse `migration.json` and validate at
least that `migrated_from` has the same run-directory name as the destination.
If it is malformed or names a different run, fail with a specific error.  This
keeps the narrow `output_root` exception tied to recorded migration provenance.

Keep the stored run config unchanged.  The frozen protocol should retain the
recovered top-level config and continue recording each run's actual
`config_sha256`; `require_registered()` will therefore detect later changes to
the run metadata.

## Regression tests / acceptance criteria

Add tests for `freeze_study()` (or a unit-testable comparison helper):

1. Equal expected/actual configs pass without `migration.json`.
2. A non-migrated run differing only in `output_root` fails.
3. A migrated run differing only in `output_root` passes.
4. A migrated run with any additional changed field (for example `seed`,
   `training.lr`, or `model.architecture`) fails.
5. A malformed/mismatched migration record fails if the recommended provenance
   validation is implemented.

For the integration test, provide the otherwise required development artifacts
(`best.pt`, `eval/val_metrics.json`, `eval/thresholds.json`, a matching
manifest, and matching similarity audit) and assert a protocol is written only
in case 3.  No test metrics may be present, because preregistration correctly
rejects a study whose test set has already been opened.

Manual acceptance after the code fix:

```bash
python run.py freeze-study \
  --config configs/rtx4090_recovered.yaml \
  --similarity-audit artifacts/leakage/similarity_audit.json \
  --output artifacts/study/protocol.json \
  --exploratory
```

It should create `artifacts/study/protocol.json`, include all configured
experiment/seed entries, and leave every run's `config.json` untouched.  If an
old partial `protocol.json` exists from a prior attempt, inspect it first; do
not overwrite a successfully frozen protocol.  The observed failed attempt
raises before `save_json()`, so it should not have created one.

## Files in scope

- `roadresearch/protocol.py` — primary fix for freeze-study config comparison.
- Optionally `roadresearch/engine.py` — factor the existing migration-aware
  config-match policy into shared code.
- `tests/test_research.py` or a dedicated protocol test — regression coverage.
- `how_to_summary.md` — updated in this change to direct users to the fixed
  migration-aware behavior and prevent unsafe manual workarounds.

