# Bugfix handoff: report crashes after a migrated run omits `initialization.json`

## Symptom

Running the `run` / `develop` report stage stops with:

```text
FileNotFoundError: .../runs/paper_study_recovered/
fasterrcnn_baseline_seed0/initialization.json
```

The exception originates at `roadresearch/report.py:44`, where `report()`
unconditionally reads:

```python
read_json(run / "initialization.json")["parameters"]
```

This is a metadata failure, not a model-training or evaluation failure.  The
affected directory contains `config.json`, `last.pt`, `best.pt`,
`manifest.json`, history, and `eval/val_metrics.json`; it does not contain
`initialization.json`.

## Confirmed root cause

`fasterrcnn_baseline_seed0` is a migrated run.  Its `migration.json` records
that it came from `runs/paper_study/fasterrcnn_baseline_seed0`.  The original
directory still has a valid `initialization.json` (including `parameters:
43266403`).

However, `roadresearch.engine.migrate_run()` only copies this fixed list:

```python
("best.pt", "last.pt", "manifest.json", "history.json", "history.csv")
```

It then writes a new config/environment/source snapshot, but never copies the
source run's `initialization.json`.  That violates the report contract: every
run selected by `*/eval/{split}_metrics.json` must also provide the metadata
used to populate the `parameters` column.

The other recovered baseline seeds have `initialization.json`, so only the
run migrated with this defective implementation causes the report to fail.

## Required code change

Modify `roadresearch/engine.py` in `migrate_run()` so that
`initialization.json` is copied verbatim from the source when it exists.  Add
it to the artifact-copy tuple, for example:

```python
for name in (
    "best.pt", "last.pt", "manifest.json", "history.json", "history.csv",
    "initialization.json",
):
    if (source_dir / name).exists():
        shutil.copy2(source_dir / name, destination_dir / name)
```

Do **not** reconstruct this file from the current model inside `migrate_run`.
It is initialization provenance for the original run, and copying the original
JSON preserves its exact architecture, pretrained-weight, parameter-count, and
trainable-parameter claims.

Prefer retaining the current conditional behavior for older source runs that
truly have no initialization file.  The migration command itself should remain
usable for those runs; the report can give a clear run-specific error if a
required provenance artifact is genuinely unavailable.

## Required repair for existing data

Changing `migrate_run()` prevents future bad migrations but does not repair
the already-created recovered directory.  After validating the source and
destination refer to the same run (seed, experiment, architecture, and
manifest), restore the existing artifact by copying:

```text
runs/paper_study/fasterrcnn_baseline_seed0/initialization.json
    ->
runs/paper_study_recovered/fasterrcnn_baseline_seed0/initialization.json
```

Use the pipeline's JSON-writing/copy convention and do not edit the numeric
parameter count by hand.  This is a provenance restoration, not a retraining
step; preserve `migration.json`, checkpoints, histories, and evaluation
artifacts unchanged.

## Reporting robustness improvement (recommended)

Make `roadresearch/report.py` validate per-run prerequisites before generating
figures.  At minimum, replace the raw `FileNotFoundError` with a contextual
error containing the run name, missing path, and whether `migration.json`
exists.  Example behavior:

```text
Run fasterrcnn_baseline_seed0 cannot be reported: missing initialization.json
(migrated run; restore it from the recorded migrated_from source or rerun migration).
```

This diagnostic must not silently use a parameter count from a different run,
silently write `None`, or omit the run from a paper aggregate.  Any of those
would make the report misleading.

## Tests / acceptance criteria

Add a focused filesystem test (no CUDA or training required), preferably in
`tests/test_research.py`:

1. Create a minimal source run directory containing the five existing copied
   artifacts, `config.json`, `source_sha256.json`, and a distinctive valid
   `initialization.json`.
2. Call `migrate_run(source, destination)`.
3. Assert destination `initialization.json` exists and its parsed JSON is
   exactly equal to the source JSON.  Also assert `migration.json` exists.
4. Verify legacy artifacts (`last.pt`, `best.pt`, manifest/history) still
   migrate as before.
5. Add a report-prerequisite test (or unit-test a small helper) that confirms
   missing initialization metadata raises the new contextual error, rather
   than exposing an unqualified `Path.read_text` `FileNotFoundError`.

Manual acceptance:

1. Restore the existing seed-0 file from its recorded source as described
   above.
2. Run the same report command that failed.
3. Confirm `development_report/per_run.csv` (or the requested report output)
   contains `fasterrcnn_baseline_seed0` with `parameters=43266403` and all
   expected runs remain in the aggregate.

## Files in scope

- `roadresearch/engine.py` — primary defect: incomplete migration artifact set.
- `roadresearch/report.py` — recommended actionable validation at the report
  boundary.
- `tests/test_research.py` — regression coverage.
- Existing artifact only: `runs/paper_study_recovered/fasterrcnn_baseline_seed0/initialization.json` — restore by copying from its `migration.json` source after verification.

