# SAGE-Net release

This repository contains the anonymous CVPR manuscript, its figures and supplementary material, the SAGE-Net model and evaluation code, split manifests, and machine-readable result records.

## Repository contents

- `models/` — model implementation.
- `datasets/` — IDD-PeD dataset loader and split manifests for IDD-PeD, JAAD, and PIE.
- `training/` — training and evaluation entry points.
- `results/paper_reported_results.json` — values transcribed from the manuscript tables. IDD-PeD main-table values are marked as manuscript-reported because the bundled run records do not reproduce them.
- `results/source_records/` — original experiment JSON files, retained without editing for provenance.
- `cvpr-paper/` — anonymous manuscript and supplementary source.

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

The script reads the included split manifests and writes new run results under `results/generated/` and checkpoints under `checkpoints/generated/`. This computation is separate from the manuscript-reported values in `results/paper_reported_results.json`.

The JAAD/PIE-only driver and the IDD-PeD five-seed driver are also available in `training/`; each accepts `--data-root` and `--output-dir`.

## Paper

Build the main paper and supplementary PDF from `cvpr-paper/` using the CVPR author-kit LaTeX toolchain. The source is anonymous. The local package does not include a compiled final PDF; inspect the output and venue checklist before submission.

## Results and provenance

The paper JSON is a transcription of `cvpr-paper/sections/results.tex` and `cvpr-paper/supplementary.tex`, not an independent experiment result. The JAAD and PIE benchmark summaries agree with the canonical three-seed records. IDD-PeD main-table mean/std and ensemble metrics are split across conflicting records; this repository preserves the values printed in the manuscript and records the unresolved evidence status in the JSON and `release/verification_report.md`. Do not cite the paper JSON as a reproduced run.

Original result files are under `results/source_records/`. They were not overwritten while preparing this package.
