# Copper Pipe Surface Defect Detection

NYCU special-topics course assignment.
100 銅管表面照（90 正常 + 10 瑕疵），用三條路線做異常偵測 — 兩條 anomalib 的 unsupervised
路線 (PatchCore / EfficientAD) + 一條 PyTorch 監督式 baseline (timm + albumentations)，
再用 ablation 把 PatchCore 推到 100%。

## TL;DR — 最佳成績

| Model | AUROC | F1 | Acc | P | R | ms/img |
|---|---|---|---|---|---|---|
| Baseline PatchCore (image=256) | 0.99 | 0.95 | 0.97 | 0.91 | 1.0 | 3.2 |
| **PatchCore (image=384)** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **6.3** |
| Rank-ensemble of 8 anomalib variants | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |

> **注意**：測試集中 OK 全為 InstructPix2Pix 合成的（見 `docs/DATA_ANALYSIS.md`），
> 100% 主要反映模型對該合成分佈的擬合，實際產線表現需另行驗證。

## 專案結構

```
copper-pipe/
├── pyproject.toml          # uv + ruff + pyright + torch cu128 source
├── train/                  # 助教提供的原始資料
│   ├── OK/                 # 90 張，InstructPix2Pix 合成的「正常」
│   └── NG/                 # 10 張，真實瑕疵
├── dataset/                # split_dataset.py 產出的 anomalib Folder layout（gitignored）
├── src/copper_pipe/        # 主要模組
│   ├── data.py             # build_datamodule
│   ├── models.py           # PatchCore / EfficientAD / PaDiM / Dinomaly builders
│   ├── training.py         # build_engine + train_one_model
│   ├── evaluation.py       # evaluate_one_model + inference timing + threshold resolution
│   └── reporting.py        # comparison.csv + markdown table
├── scripts/
│   ├── split_dataset.py    # OK/NG → train/good + test/good + test/defect
│   ├── train_anomalib.py   # PatchCore + EfficientAD pipeline (assignment 主腳本)
│   ├── train_supervised.py # 監督式 baseline (timm + albumentations)
│   └── run_experiments.py  # 8 個 anomalib 變體 + ensemble 的 ablation runner
├── docs/
│   ├── REPORT.md           # 整理報告
│   ├── DATA_ANALYSIS.md    # 助教資料集分析（OK 全是合成這件事）
│   └── EXPERIMENTS.md      # Ablation 結果逐一解讀
└── results*/               # 訓練輸出（全 gitignored）
```

## Setup

```bash
uv sync
```

RTX 5090 (Blackwell, sm_120) 需要 CUDA 12.8 wheel；`pyproject.toml` 已用 `[tool.uv.sources]`
鎖定 PyTorch 的 cu128 index，預設就會抓對版本。

## 0. 切分資料

```bash
uv run python scripts/split_dataset.py \
    --normal_src ./train/OK \
    --abnormal_src ./train/NG \
    --output ./dataset \
    --force
```

產出：

```
dataset/
├── train/good/      (70)
└── test/
    ├── good/        (20)
    └── defect/      (10)
└── split_manifest.csv
```

選項：`--train_ratio` (預設 0.78)、`--seed` (預設 42)、`--mode {copy,symlink}`。

## 1. anomalib baseline — PatchCore + EfficientAD

```bash
uv run python scripts/train_anomalib.py
```

輸出：
- `results/<model>/predictions.csv` — 每張圖的 filename / true_label / pred_score / pred_label
- `results/comparison.csv` — 兩模型 side-by-side
- stdout 印 markdown 表格

可用 `--models patchcore` 單跑一個、`--image_size 384` 調 resize、`--batch_size`、
`--output_dir` 等。

## 2. 監督式 baseline (對照組)

```bash
uv run python scripts/train_supervised.py                      # 預設 simple_split
uv run python scripts/train_supervised.py --mode kfold --k 5   # k-fold
uv run python scripts/train_supervised.py --backbone resnet18 --epochs 30
```

- 把 7 張 defect 加入訓練、3 張留測試
- AdamW + CosineAnnealingLR，EarlyStopping on F1
- albumentations 強 augmentation + WeightedRandomSampler

輸出（`results_supervised/simple_split/` 或 `fold_<k>/`）：
- `best_model.pth`、`training_log.csv`、`predictions.csv`、`metrics.json`
- `confusion_matrix.png`、`roc.png`、`training_curve.png`

## 3. Ablation experiments

把 PatchCore 的調參、PaDiM、Dinomaly、EfficientAD、+ ensemble 一次跑完：

```bash
uv run python scripts/run_experiments.py
```

或挑特定變體：

```bash
uv run python scripts/run_experiments.py --variants patchcore_big dinomaly efficientad
```

詳細結果見 `docs/EXPERIMENTS.md`。

## Lint / type check

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

兩個都是 0 errors。

## 文件導讀

| 想知道什麼 | 看哪份 |
|---|---|
| 結果跟整體 setup | `docs/REPORT.md` |
| 資料集到底是什麼 / OK 為什麼是合成的 | `docs/DATA_ANALYSIS.md` |
| 為什麼選 image_size=384、各 ablation 取捨 | `docs/EXPERIMENTS.md` |
