# Copper Pipe Surface Defect Detection — Report

NYCU Special Topics course assignment.
Detect surface defects on copper pipes given a heavily imbalanced 100-image dataset.

## 1. Dataset

Source images live in `train/OK` (90 normal, all InstructPix2Pix-synthesized — see
`docs/DATA_ANALYSIS.md`) and `train/NG` (10 real defects).
`scripts/split_dataset.py` materializes them into the anomalib `Folder` layout
(`normal_dir`, `normal_test_dir`, `abnormal_dir`). Two layouts are supported:

| Layout | train/good | test/good | test/defect | use for |
|---|---|---|---|---|
| `split`  | 70 | 20 (held out) | 10 | internal evaluation with meaningful numbers |
| `full`   | 90 | 90 (symlinked, calibration only) | 10 | final training before predicting on the TA's holdout |

The numbers in this report come from `--layout split` (so they're on a true 20-OK
holdout). Final submission models are retrained with `--layout full` and run
through `scripts/predict.py` against the TA's test directory.

For the supervised baseline we additionally pull **7** of the 10 defect images
into the training set, leaving **3** for test — that puts the supervised
test set at 20 good + 3 defect = 23 images.

## 2. Methods

Three approaches were implemented and run side-by-side.

### 2.1 PatchCore (anomalib, unsupervised)
- Backbone: `wide_resnet50_2` (ImageNet pretrained, frozen)
- Feature layers: `layer2`, `layer3`
- Coreset sampling ratio: 0.1
- Single-epoch "training" (memory bank construction)
- Image size: 256×256

### 2.2 EfficientAD (anomalib, unsupervised)
- Library default `EfficientAdModelSize.S` (small)
- 30 epochs of student–teacher + autoencoder loss
- Image size: 256×256 (model-mandated)
- ImageNet validation tile downloaded automatically by anomalib for penalty term.

### 2.3 Supervised baseline (PyTorch + timm + albumentations)
- Backbones supported: `resnet18`, `efficientnet_b0` (default), `convnext_tiny`
- 2-class softmax head over the timm backbone (pretrained ImageNet)
- Strong augmentation (HFlip, VFlip, Rotate ±30°, Affine, ColorJitter, GaussianBlur)
- `WeightedRandomSampler` for batch-level balance
- Inverse-square-root class weights on CrossEntropyLoss
  (pure inverse-frequency on top of the sampler was *too aggressive* and
  collapsed the model into predicting every image as defect)
- AdamW (`lr=3e-4, wd=1e-4`), CosineAnnealingLR, EarlyStopping on F1 (patience 10)
- Best checkpoint by validation F1
- Supports `--mode simple_split` and `--mode kfold` (StratifiedKFold)

## 3. Results

### 3.1 anomalib (test set = 20 good + 10 defect = 30 images)

Baseline (per the original task spec — image_size=256):

| Model | image AUROC | F1 | Accuracy | Precision | Recall | Threshold | ms / img |
|---|---|---|---|---|---|---|---|
| PatchCore   | 0.9900 | 0.9524 | 0.9667 | 0.9091 | 1.0000 | 0.5000 | 3.22 |
| EfficientAD | 0.9750 | 0.9091 | 0.9333 | 0.8333 | 1.0000 | 0.5000 | 2.16 |

After ablation (see `docs/EXPERIMENTS.md`) the best single model is
**PatchCore at image_size=384** — perfect on every metric, 6.3 ms/img:

| Model | image AUROC | F1 | Accuracy | Precision | Recall | ms / img |
|---|---|---|---|---|---|---|
| **PatchCore (image=384)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **6.3** |

Score-rank ensemble of all 8 anomalib variants also hits 1.0 across the board.

(Threshold is the normalized image threshold supplied by anomalib's post-processor —
post-min-max scores in [0, 1] are compared against 0.5.)

**Reading these results.** Both models recover *every* defect (recall = 1.0).
PatchCore is the more conservative — only 1 false positive vs. EfficientAD's 2.
EfficientAD is ~33 % faster per inference, which matters at deployment scale even
though both are well within real-time on an RTX 5090.

### 3.2 Supervised baseline (smoke run: ResNet-18, 20 epochs, simple_split, 7-train/3-test defects)

Numbers from `results_supervised/simple_split/metrics.json`:

| Metric | Value |
|---|---|
| AUROC | 0.683 |
| F1    | 0.364 |
| Acc   | 0.696 |
| Precision | 0.250 |
| Recall | 0.667 |
| Confusion matrix | `[[14, 6], [1, 2]]` (rows = true, cols = pred) |

**Caveats.**
1. Only **3** defect images are in the test set, so a single misclassification
   moves recall by 0.33 — these numbers are noisy by construction.
2. Within-run AUROC actually hit **1.0** by epoch 3, meaning the model *can*
   separate the classes; the F1 number is depressed by argmax-at-0.5 on
   under-converged logits. A picked threshold on a held-out validation slice
   would close that gap.
3. The supervised route is included as a baseline; for production a more
   stable picture would come from running `--mode kfold --k 5` and reporting
   mean ± std across folds.

### 3.3 Comparing approaches

| Approach | Uses defect images for training? | Test set size | Recommended when… |
|---|---|---|---|
| PatchCore   | No  | 30 | very few defects, fast deployment, interpretable patch-level scores |
| EfficientAD | No  | 30 | similar to PatchCore, slightly faster, slightly less precise on this set |
| Supervised  | Yes (7 of 10) | 23 | many labeled examples per class — *not really our regime* |

For this dataset, the unsupervised anomalib route is the clear winner — both
because the defect count is too small to safely train a supervised
classifier *and* because `recall = 1.0` matters more than `precision` in
defect detection (you'd rather over-flag than miss).

## 4. Submission workflow

Internal metrics in Section 3.1 are computed on a 70/20/10 internal split where
the model can only see 70 of 90 OK images during training. For the final
submission we retrain on **all** available data — every OK and every NG — and
expose a separate inference script (`scripts/predict.py`) that the TA can point
at a folder of unseen images.

### 4.1 Pipeline

```
                +----------------+
TA test images  | predict.py     |  predictions.csv  (filename, pred_score, pred_label)
─────────────►  |                |─────────────────►
                +-------▲--------+
                        │
                        │ ckpt_path
                        │
                +-------┴--------+
                | train_anomalib |
                +-------▲--------+
                        │ data_root
                        │
                +-------┴--------+
train/OK + NG   | split_dataset  |
─────────────►  | --layout full  |
                +----------------+
```

### 4.2 What's in the submission zip

| Path | Purpose | Required? |
|---|---|---|
| `pyproject.toml`, `uv.lock`, `.python-version` | reproducible env | yes |
| `src/`, `scripts/` | source code | yes |
| `docs/` | this report + analysis | yes |
| `README.md` | top-level instructions | yes |
| `train/OK`, `train/NG` | the raw data we trained on | yes (lets the TA verify reproducibility) |
| `results/patchcore/checkpoint.ckpt` | pre-trained PatchCore @ image_size=384 | **yes** — so the TA can skip retraining |

Notably excluded: `dataset/` (regenerated by `split_dataset.py`), `datasets/`
(anomalib's auto-downloaded ImageNette, ~1.5 GB), `results_experiments/`,
`results_supervised/`, `predictions/`, `submissions/`, `.venv/`, `.git/`.

### 4.3 Running on the TA's test set

The shortest path for the TA (using the bundled checkpoint, no retraining):

```bash
unzip copper-pipe.zip
cd copper-pipe
uv sync
uv run python scripts/predict.py \
    --model patchcore \
    --checkpoint results/patchcore/checkpoint.ckpt \
    --test_dir <path/to/ta/test/folder> \
    --output ./predictions
```

This produces `predictions/predictions.csv` with `filename, pred_score, pred_label`
(where `pred_label` is 1 for defect, 0 for normal) and also prints a markdown
table to the terminal. The decision threshold defaults to **0.45** on the
post-processor's min-max-normalized score — slightly below anomalib's nominal
0.5 to bias toward recall (in industrial QC, missing a defect costs more than
a false positive). `--threshold` can override it.

CUDA is detected automatically — without a GPU the script falls back to CPU
(slower but functional).

### 4.4 Verification on training data

As a sanity check that the bundled checkpoint loads correctly, running predict
on the original `train/OK` and `train/NG` folders gives:

- `train/OK` (90 images): 0 / 90 flagged as defect (max score 0.0477,
  well below the 0.45 default threshold)
- `train/NG` (10 images): 10 / 10 flagged at the default 0.45 threshold.
  (At anomalib's nominal 0.5, one borderline NG scores 0.4999 and gets missed —
  the main reason we bias the default below 0.5.)

The TA's unseen OK images are expected to score somewhat higher than 0.05
(the model has memorized the training OK exactly), but well below 0.5 if they
come from the same distribution.

## 5. Reproducing the runs (developer)

```bash
# Setup
uv sync

# (a) Internal evaluation (numbers in this report)
uv run python scripts/split_dataset.py \
    --normal_src ./train/OK --abnormal_src ./train/NG \
    --output ./dataset --layout split --force
uv run python scripts/train_anomalib.py --image_size 384
uv run python scripts/run_experiments.py
uv run python scripts/train_supervised.py --backbone efficientnet_b0 --epochs 50
uv run python scripts/train_supervised.py --mode kfold --k 5

# (b) Submission: retrain on all data, then predict on the TA's test folder
uv run python scripts/split_dataset.py \
    --normal_src ./train/OK --abnormal_src ./train/NG \
    --output ./dataset --layout full --force
uv run python scripts/train_anomalib.py --models patchcore --image_size 384
uv run python scripts/predict.py \
    --model patchcore \
    --checkpoint results/patchcore/checkpoint.ckpt \
    --test_dir <ta_test_folder> \
    --output ./submissions/patchcore
```

Outputs:

- `results/comparison.csv` — anomalib comparison table
- `results/<model>/predictions.csv` — per-image scores
- `results_supervised/simple_split/{metrics.json, predictions.csv, training_log.csv}`
  plus `confusion_matrix.png`, `roc.png`, `training_curve.png`
- `results_supervised/fold_<k>/…` and `kfold_summary.json` for k-fold runs

## 6. Notes on the environment

- Python 3.12 + uv-managed venv.
- RTX 5090 (Blackwell, sm_120) requires CUDA 12.8 PyTorch wheels.
  `pyproject.toml` pins `torch` / `torchvision` to the cu128 index
  (`[tool.uv.sources]` + `[[tool.uv.index]]`). Without this, default PyPI
  delivers a cu130 build that the 12.8 driver rejects.
- anomalib version actually installed: **2.4.1**. The build code uses
  `inspect.signature` to filter kwargs (`_filter_kwargs`) and falls back
  between class-name variants (`Patchcore` vs `PatchCore`) so it survives
  small API drift across anomalib releases.
- The first EfficientAD run downloads `imagenette` (~1.5 GB) into `./datasets/`
  for its student-teacher penalty term — gitignored.
- All thresholds for the anomalib path come from the model's post-processor
  (`normalized_image_threshold`, typically 0.5 after min-max normalization),
  not the raw F1-adaptive threshold — pairing the raw threshold with the
  normalized predict-time scores would give F1 = 0.
