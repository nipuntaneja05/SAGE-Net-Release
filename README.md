# SAGE-Net

SAGE-Net combines kinematic and visual predictions with a scene-conditioned adaptive fusion gate for pedestrian crossing anticipation. This repository contains the model, training and evaluation code, split manifests, and original experiment records.

## Repository contents

- `models/` — model implementation.
- `datasets/` — IDD-PeD dataset loader and split manifests for IDD-PeD, JAAD, and PIE.
- `training/` — training and evaluation entry points.
- `evaluation/` — architecture checks and experiment plotting utilities.
- `ablations/` — ablation experiment driver.
- `results/source_records/` — original experiment JSON files, retained without editing for provenance.

Dataset videos, precomputed visual features, tensor caches, and model checkpoints are not included. The training scripts expect the tensor-cache files named below in a data directory supplied with `--data-root`. The caches must contain lists of sample dictionaries with `V_M`, `V_S`, `V_I`, `K_M`, `label`, and `video_id` fields; the IDD-PeD path also needs the precomputed held-out test tensor file described by the script. The release does not identify the visual-feature backbone, so recreating those features requires the original preprocessing pipeline.

## Environment

Python 3.10 or newer is recommended. Install the runtime dependencies with:

```sh
python -m pip install -r requirements.txt
```

## Run training and evaluation

Place the required feature caches under a data directory, then run:

```sh
python training/train_and_eval_canonical_unified.py --data-root /path/to/sagenet-data
```

The script reads the included split manifests and writes new run results under `results/generated/` and checkpoints under `checkpoints/generated/`.

The JAAD/PIE-only driver and the IDD-PeD five-seed driver are also available in `training/`; each accepts `--data-root` and `--output-dir`.

## Results and provenance

Original result files are under `results/source_records/`. They include runs with different protocols and conflicting IDD-PeD summaries. They are retained for provenance and should not be treated as a single reconciled benchmark result. Their values have not been independently reproduced from this repository.

Reproducing an experiment requires the corresponding feature caches, preprocessing configuration, and evaluation protocol. The missing preprocessing assets and checkpoints mean this repository alone is not a complete reproduction package.
