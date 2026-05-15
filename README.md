# Copper Pipe Anomaly Detection

Train PatchCore + EfficientAD on copper-pipe surface images using
[anomalib](https://github.com/openvinotoolkit/anomalib).

## Setup

```bash
uv sync
```

If `uv sync` fails for `torch` on RTX 5090 (Blackwell, sm_120), install a
PyTorch build with CUDA 12.8 support first, e.g.:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision
uv sync
```

## Prepare dataset

Raw data lives in `train/OK/` (90) and `train/NG/` (10).
Split into the layout anomalib expects:

```bash
uv run python scripts/prepare_dataset.py --src ./train --dst ./dataset --seed 42
```

Produces:

```
dataset/
├── train/good/      (70 normal)
└── test/
    ├── good/        (20 normal)
    └── defect/      (10 abnormal)
```

## Train

```bash
uv run python scripts/train_anomalib.py            # both models
uv run python scripts/train_anomalib.py --models patchcore
uv run python scripts/train_anomalib.py --models efficientad --image_size 256
```

Outputs land in `results/`:

- `results/<model>/` — Lightning checkpoint + per-image `predictions.csv`
- `results/comparison.csv` — side-by-side metrics
- A markdown table printed to stdout (copy/paste into the report)

## Static analysis

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
```
