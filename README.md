# Copper Pipe Surface Defect Detection

NYCU special-topics course assignment.
100 銅管表面照（90 正常 + 10 瑕疵），用三條路線做異常偵測 — 兩條 anomalib 的 unsupervised
路線 (PatchCore / EfficientAD) + 一條 PyTorch 監督式 baseline (timm + albumentations)，
再用 ablation 把 PatchCore 推到 100%。

## 助教使用方式（最短路徑）

zip 裡已附訓練好的 `results/patchcore/checkpoint.ckpt`（PatchCore + image_size=384，
我們實驗中最強的單一模型），所以不用重訓。三步：

```bash
# 1. 解壓 + 安裝環境（任選一個）
unzip copper-pipe.zip
cd copper-pipe

# 1a. 用 uv（推薦，會用 uv.lock 鎖定的精準版本）
uv sync

# 1b. 沒有 uv 的話，用 pip + requirements.txt（裡面已含 CUDA 12.8 index）
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 對助教 test set 做推論（uv 版）
uv run python scripts/predict.py \
    --model patchcore \
    --checkpoint results/patchcore/checkpoint.ckpt \
    --test_dir <測試資料夾的路徑> \
    --output ./predictions

# 2'. 用 pip venv 的版本
python scripts/predict.py \
    --model patchcore \
    --checkpoint results/patchcore/checkpoint.ckpt \
    --test_dir <測試資料夾的路徑> \
    --output ./predictions
```

輸出：

- **`predictions/predictions.csv`** 是助教唯一需要看的檔案：欄位 `filename, pred_score, pred_label`
  （`pred_label`：1 = 瑕疵 / 0 = 正常；`pred_score` ∈ [0, 1] 是 anomalib 經 min-max
  normalize 後的異常分數）
- terminal 也會直接印一張 markdown 表格，可即時看每張的判定
- `predictions/` 內**只會有 `predictions.csv` 一個檔**，腳本會把 anomalib 內部
  的 heatmap、lightning checkpoint 等中介產物全寫到 tempdir 並在結束時清掉

注意事項：

- 測試資料夾請將圖片**平鋪**放在最上層（支援 .png / .jpg / .jpeg / .bmp / .tif）。
  子資料夾不會被遞迴讀取。
- 有 NVIDIA GPU + CUDA 12.x 會自動用 GPU；沒有也會 fallback 到 CPU（會比較慢）。
- 預設 threshold 是 normalized score 的 **0.45**（略低於 anomalib nominal 的 0.5，
  偏向多抓瑕疵；我們 `train/OK` 最高分才 0.0477，遠低於 0.45 所以不會誤判正常）。
  若 false positive 太多可以改 `--threshold 0.55`。

想要完全從頭重訓的話，請看下方「Step 0 → 1」。

## TL;DR — 最佳成績

| Model | AUROC | F1 | Acc | P | R | ms/img |
|---|---|---|---|---|---|---|
| Baseline PatchCore (image=256) | 0.99 | 0.95 | 0.97 | 0.91 | 1.0 | 3.2 |
| **PatchCore (image=384)** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **6.3** |
| Rank-ensemble of 8 anomalib variants | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |

> **注意**：上面的成績是內部測試（OK 與 NG 都來自助教給的訓練集再切分），且 OK 全為
> InstructPix2Pix 合成的（見 `docs/DATA_ANALYSIS.md`），這 100% 主要反映模型對該合成
> 分佈的擬合，助教 holdout test set 的真實成績可能略低。

## 專案結構

```
copper-pipe/
├── pyproject.toml          # uv + ruff + pyright + torch cu128 source
├── uv.lock                 # uv 鎖定版本
├── requirements.txt        # pip 備援（含 cu128 --extra-index-url）
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
│   ├── split_dataset.py    # OK/NG → train/good + test/good + test/defect (full or split)
│   ├── train_anomalib.py   # PatchCore + EfficientAD pipeline (assignment 主腳本)
│   ├── predict.py          # 載入 checkpoint，對助教 test_dir 推論 → predictions.csv
│   ├── train_supervised.py # 監督式 baseline (timm + albumentations)
│   └── run_experiments.py  # 8 個 anomalib 變體 + ensemble 的 ablation runner
├── docs/
│   ├── REPORT.md           # 整理報告
│   ├── DATA_ANALYSIS.md    # 助教資料集分析（OK 全是合成這件事）
│   └── EXPERIMENTS.md      # Ablation 結果逐一解讀
├── results/                # 訓練輸出（其中 results/patchcore/checkpoint.ckpt 有附在 zip）
├── results_supervised/     # 監督式 baseline 輸出（gitignored，未附）
├── results_experiments/    # ablation 8 變體輸出（gitignored，未附）
└── predictions/            # predict.py 對助教 test_dir 的輸出
```

## Setup（重訓 / 開發者用）

```bash
uv sync
```

RTX 5090 (Blackwell, sm_120) 需要 CUDA 12.8 wheel；`pyproject.toml` 已用 `[tool.uv.sources]`
鎖定 PyTorch 的 cu128 index，預設就會抓對版本。其他 CUDA 12.x GPU 也相容。

> 以下 Step 0 ~ Step 3 是「從頭重訓 + 內部評估」的完整流程。
> 助教如果只想用我們訓練好的 checkpoint 跑 test，看上面「助教使用方式」就好。

## Step 0. 切分資料

兩種 layout：

- **`--layout full`（預設，繳交版用）**：所有 90 張 OK 都進 train/good，
  同一批 OK 用 symlink 鏡像到 test/good 給 anomalib 的 post-processor 做
  min/max normalization 校準，10 張 NG 進 test/defect。
- **`--layout split`（內部驗證用）**：原本的 70/20/10 隨機切，會留出 20 張
  OK 當作有意義的內部 test。

```bash
# 繳交流程：用全部資料訓練
uv run python scripts/split_dataset.py \
    --normal_src ./train/OK \
    --abnormal_src ./train/NG \
    --output ./dataset \
    --layout full --force

# 想看內部 metric 的話
uv run python scripts/split_dataset.py \
    --normal_src ./train/OK --abnormal_src ./train/NG \
    --output ./dataset --layout split --force
```

其他選項：`--train_ratio` (split layout 用，預設 0.78)、`--seed` (預設 42)、
`--mode {copy,symlink}`。

## Step 1. anomalib baseline — PatchCore + EfficientAD

```bash
uv run python scripts/train_anomalib.py --image_size 384
```

輸出：
- `results/<model>/predictions.csv` — 每張圖的 filename / true_label / pred_score / pred_label
- `results/<model>/checkpoint.ckpt` — 訓練好的權重（給 `predict.py` 用）
- `results/comparison.csv` — 兩模型 side-by-side
- stdout 印 markdown 表格 + 各模型 checkpoint 路徑

可用 `--models patchcore` 單跑一個、`--image_size 384` 調 resize、`--batch_size`、
`--output_dir` 等。

> **注意**：用 `--layout full` 切的資料訓練時，內部 metric 會很漂亮（因為
> test/good = train/good），那個分數沒意義。最終分數要看 Step 1.5（或最上方「助教
> 使用方式」）對真正 test set 的輸出。

## Step 1.5. 對 test set 做推論（繳交版 / 助教評分用）

訓練完 Step 1 之後，用得到的 checkpoint 對任意 test 資料夾推論：

```bash
uv run python scripts/predict.py \
    --model patchcore \
    --checkpoint results/patchcore/checkpoint.ckpt \
    --test_dir <測試資料夾> \
    --output ./predictions
```

詳細用法（包含助教評分的最短路徑）見最上方「助教使用方式」。

## Step 2. 監督式 baseline (對照組)

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

## Step 3. Ablation experiments

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
