# Velox module

This package contains a standalone Velox implementation that is now wired into the training pipeline.

## What this implementation does

The Velox path now:
- builds a concrete Velox model from explicit Python classes,
- uses a dedicated Velox encoder, decoder, and objective,
- runs training through the shared batching and evaluation infrastructure,
- and is selected automatically when you invoke the framework with the Velox model name.

## How to use it

### 1. Run Velox through the normal PIDSMaker entrypoint

From the repository root, run:

```bash
python -m pidsmaker.main velox CLEARSCOPE_E3
```

You can replace the dataset with any dataset supported by the framework, such as:

```bash
python -m pidsmaker.main velox THEIA_E3
python -m pidsmaker.main velox TRACE_E3
```

### 2. Use the existing Velox config

The repository already includes a Velox config at [config/velox.yml](config/velox.yml). The new module is activated automatically when the selected model is `velox`.

### 3. Run inference without retraining

Once training has produced checkpoints, you can load a saved checkpoint and evaluate it on validation or test data without training again.

#### Example: load the best checkpoint and run test-set inference

```bash
python -m pidsmaker.detection.inference \
  --config velox \
  --dataset CLEARSCOPE_E3 \
  --epoch -1 \
  --split test
```

#### Example: load a specific epoch

```bash
python -m pidsmaker.detection.inference \
  --config velox \
  --dataset CLEARSCOPE_E3 \
  --epoch 5 \
  --split val
```

#### Notes

- `--epoch -1` loads the best checkpoint recorded by the checkpoint manager.
- `--split test` evaluates on the test split.
- `--split val` evaluates on the validation split.
- The checkpoint directory is inferred from the task path created by the framework, but you can override it with `--checkpoint-dir` if needed.

### 4. Inspect the outputs

Training artifacts and logs are written under the task artifact directory created by the framework. The exact path depends on your runtime configuration, but you can inspect the generated task folders under your artifact directory.

Velox also writes model-inspection score artifacts during training:

```text
<task artifact dir>/velox_scores/model_epoch_<epoch>/
  threshold_summary.json
  val.csv
  test.csv
```

The CSV files include:
- `anomaly_score`: the per-edge edge-type prediction loss, i.e. how surprising the observed edge type was to the model,
- `threshold`: the validation-derived threshold used for the epoch,
- `distance_to_threshold`: `anomaly_score - threshold`, so positive values are above threshold,
- prediction diagnostics such as source/destination node, observed/predicted edge type, and observed/predicted probability.

## Important implementation details

The Velox module is implemented in these files:
- [pidsmaker/pids/velox/encoder.py](pidsmaker/pids/velox/encoder.py): the Velox encoder
- [pidsmaker/pids/velox/decoder.py](pidsmaker/pids/velox/decoder.py): the Velox decoder
- [pidsmaker/pids/velox/objectives.py](pidsmaker/pids/velox/objectives.py): the Velox objective/loss
- [pidsmaker/pids/velox/model.py](pidsmaker/pids/velox/model.py): the Velox model wrapper
- [pidsmaker/pids/velox/trainer.py](pidsmaker/pids/velox/trainer.py): the training entrypoint
- [pidsmaker/pids/velox/score_artifacts.py](pidsmaker/pids/velox/score_artifacts.py): Velox raw score and threshold artifact writer

## How to modify Velox

Because the implementation now lives in its own module, you can edit the code directly in these files without going through the shared factory builder.

### Example: change the encoder

Edit [pidsmaker/pids/velox/encoder.py](pidsmaker/pids/velox/encoder.py) and replace the current simple encoder with your own architecture.

### Example: change the objective

Edit [pidsmaker/pids/velox/objectives.py](pidsmaker/pids/velox/objectives.py) to change the loss or prediction target.

### Example: change the decoder

Edit [pidsmaker/pids/velox/decoder.py](pidsmaker/pids/velox/decoder.py) to change the decoder head.

## Notes for development

- The current implementation uses a lightweight, explicit model definition so it is easy to understand and extend.
- The shared framework is still used for data loading, batching, evaluation, checkpointing, and logging.
- If you want a more faithful version of the original Velox system, you can progressively replace the lightweight components in this module with more advanced ones.

## Margin-to-threshold analysis

[MARGIN_ANALYSIS_FINDINGS.md](MARGIN_ANALYSIS_FINDINGS.md) documents findings
on whether a node's distance from the evaluation threshold correlates with
prediction correctness, using [margin_analysis.py](margin_analysis.py), a
reusable script that runs directly against a run's saved
`scores_model_epoch_*.pkl` files (no retraining needed).

## Quick sanity check

To confirm that the Velox module imports and compiles correctly, run:

```bash
python -m compileall pidsmaker/pids/velox
```
