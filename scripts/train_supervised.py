"""Supervised binary classifier (timm + albumentations) for copper-pipe defect detection.

Two modes:
    --mode simple_split   train one model (70 good + 7 defect train, 20 good + 3 defect test)
    --mode kfold          stratified k-fold over the full 100-image pool

Example:
    uv run python scripts/train_supervised.py
    uv run python scripts/train_supervised.py --backbone resnet18 --epochs 30
    uv run python scripts/train_supervised.py --mode kfold --k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

LOGGER = logging.getLogger("train_supervised")

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------- Reproducibility -------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- Data ------------------------------------------------------------------------------


@dataclass
class Sample:
    path: Path
    label: int  # 0 = good, 1 = defect


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def gather_samples(data_root: Path) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Return (train_good, test_good, all_defects)."""
    train_good = [Sample(p, 0) for p in list_images(data_root / "train" / "good")]
    test_good = [Sample(p, 0) for p in list_images(data_root / "test" / "good")]
    defects = [Sample(p, 1) for p in list_images(data_root / "test" / "defect")]
    return train_good, test_good, defects


def split_defects_for_supervised(
    defects: list[Sample], n_train: int, seed: int
) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    shuffled = defects.copy()
    rng.shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:]


class CopperPipeDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: A.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        s = self.samples[idx]
        with Image.open(s.path) as im:
            arr = np.array(im.convert("RGB"))
        out = self.transform(image=arr)
        return out["image"], s.label, str(s.path)


def build_train_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(int(image_size * 1.15), int(image_size * 1.15)),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, border_mode=0, p=0.7),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                p=0.5,
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.7),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_eval_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def make_sampler(samples: list[Sample]) -> WeightedRandomSampler:
    """Oversample the minority class so each batch sees defects roughly as often as goods."""
    counts = np.bincount([s.label for s in samples], minlength=2).astype(np.float64)
    weights_per_class = 1.0 / np.clip(counts, 1.0, None)
    sample_weights = [weights_per_class[s.label] for s in samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(samples), replacement=True)


def build_loaders(
    train_samples: list[Sample],
    test_samples: list[Sample],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = CopperPipeDataset(train_samples, build_train_transform(image_size))
    test_ds = CopperPipeDataset(test_samples, build_eval_transform(image_size))
    sampler = make_sampler(train_samples)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


# ---------- Model / Loss / EarlyStopping ------------------------------------------------------


def build_model(backbone: str) -> nn.Module:
    """Two-class classifier on top of a timm backbone. CrossEntropyLoss eats logits directly."""
    import timm

    model = timm.create_model(backbone, pretrained=True, num_classes=2)
    return model


class FocalLoss(nn.Module):
    """Multi-class focal loss. Use when CE struggles on heavily imbalanced data."""

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        p_t = torch.exp(-ce)
        loss = (1 - p_t) ** self.gamma * ce
        return loss.mean()


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# ---------- Training loop ---------------------------------------------------------------------


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    val_f1: float
    val_auroc: float


@dataclass
class FoldOutput:
    fold: int
    best_f1: float
    best_epoch: int
    history: list[EpochStats] = field(default_factory=list)
    final_metrics: dict[str, Any] = field(default_factory=dict)
    pred_scores: list[float] = field(default_factory=list)
    pred_labels: list[int] = field(default_factory=list)
    true_labels: list[int] = field(default_factory=list)
    filenames: list[str] = field(default_factory=list)


def _safe_auroc(y_true: list[int], y_score: list[float]) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(auc(fpr, tpr))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, list[float], list[int], list[int], list[str]]:
    model.eval()
    total_loss = 0.0
    n_seen = 0
    scores: list[float] = []
    preds: list[int] = []
    trues: list[int] = []
    files: list[str] = []
    with torch.inference_mode():
        for batch_imgs, batch_labels, batch_paths in loader:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = model(batch_imgs)
            loss = loss_fn(logits, batch_labels)
            total_loss += float(loss.item()) * batch_imgs.size(0)
            n_seen += batch_imgs.size(0)
            prob = F.softmax(logits, dim=1)[:, 1]
            scores.extend(prob.detach().cpu().tolist())
            preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            trues.extend(batch_labels.detach().cpu().tolist())
            files.extend(batch_paths)
    return total_loss / max(1, n_seen), scores, preds, trues, files


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_seen = 0
    pbar = tqdm(loader, desc=f"epoch {epoch:3d} train", leave=False)
    for batch_imgs, batch_labels, _ in pbar:
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_imgs)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch_imgs.size(0)
        n_seen += batch_imgs.size(0)
        pbar.set_postfix(loss=f"{total_loss / max(1, n_seen):.4f}")
    return total_loss / max(1, n_seen)


def train_one_fold(
    fold_idx: int,
    train_samples: list[Sample],
    test_samples: list[Sample],
    *,
    backbone: str,
    image_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    num_workers: int,
    use_focal_loss: bool,
    patience: int,
    device: torch.device,
    fold_output_dir: Path,
) -> FoldOutput:
    train_loader, test_loader = build_loaders(
        train_samples, test_samples, image_size, batch_size, num_workers
    )
    LOGGER.info(
        "fold %d: train=%d (good=%d, defect=%d), test=%d (good=%d, defect=%d)",
        fold_idx,
        len(train_samples),
        sum(1 for s in train_samples if s.label == 0),
        sum(1 for s in train_samples if s.label == 1),
        len(test_samples),
        sum(1 for s in test_samples if s.label == 0),
        sum(1 for s in test_samples if s.label == 1),
    )

    model = build_model(backbone).to(device)

    counts = np.bincount([s.label for s in train_samples], minlength=2).astype(np.float32)
    # Inverse-sqrt class weights. We already use WeightedRandomSampler to balance
    # batches; pure inverse-frequency on top of that overcorrects and makes the model
    # predict the minority class for everything.
    raw_inv_sqrt = 1.0 / np.sqrt(np.clip(counts, 1.0, None))
    normalized = raw_inv_sqrt / raw_inv_sqrt.mean()
    class_weights = torch.tensor(normalized, device=device, dtype=torch.float32)
    LOGGER.info("class weights (good, defect) = %s (from counts %s)", class_weights.tolist(), counts.tolist())

    loss_fn: nn.Module
    if use_focal_loss:
        loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early = EarlyStopping(patience=patience, mode="max")

    fold_output_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_output_dir / "best_model.pth"
    out = FoldOutput(fold=fold_idx, best_f1=-1.0, best_epoch=-1)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        val_loss, scores, preds, trues, files = evaluate(model, test_loader, loss_fn, device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_f1 = float(f1_score(trues, preds))
        val_acc = sum(p == t for p, t in zip(preds, trues, strict=True)) / max(1, len(trues))
        val_auroc = _safe_auroc(trues, scores)
        scheduler.step()

        stats = EpochStats(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            val_f1=val_f1,
            val_auroc=val_auroc,
        )
        out.history.append(stats)
        LOGGER.info(
            "fold %d ep %3d  train_loss=%.4f  val_loss=%.4f  acc=%.4f  f1=%.4f  auroc=%.4f",
            fold_idx,
            epoch,
            train_loss,
            val_loss,
            val_acc,
            val_f1,
            val_auroc,
        )

        improved = early.step(val_f1)
        if improved:
            out.best_f1 = val_f1
            out.best_epoch = epoch
            out.pred_scores = scores
            out.pred_labels = preds
            out.true_labels = trues
            out.filenames = files
            torch.save(model.state_dict(), best_path)
        if early.should_stop:
            LOGGER.info("fold %d: early stop at epoch %d (best F1=%.4f at epoch %d)", fold_idx, epoch, out.best_f1, out.best_epoch)
            break

    # Re-evaluate the best snapshot to populate final_metrics.
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    out.final_metrics = compute_final_metrics(out.true_labels, out.pred_scores, out.pred_labels)
    return out


def compute_final_metrics(
    trues: list[int], scores: list[float], preds: list[int]
) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = float(f1_score(trues, preds))
        prec = float(precision_score(trues, preds))
        rec = float(recall_score(trues, preds))
    acc = sum(p == t for p, t in zip(preds, trues, strict=True)) / max(1, len(trues))
    cm = confusion_matrix(trues, preds, labels=[0, 1]).tolist()
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": _safe_auroc(trues, scores),
        "confusion_matrix": cm,
    }


# ---------- Plots / outputs -------------------------------------------------------------------


def plot_training_curve(history: list[EpochStats], path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs = [h.epoch for h in history]
    ax1.plot(epochs, [h.train_loss for h in history], label="train_loss", color="tab:blue")
    ax1.plot(epochs, [h.val_loss for h in history], label="val_loss", color="tab:orange")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h.val_f1 for h in history], label="val_f1", color="tab:green", linestyle="--")
    ax2.plot(epochs, [h.val_auroc for h in history], label="val_auroc", color="tab:red", linestyle="--")
    ax2.set_ylabel("metric")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm: list[list[int]], path: Path, title: str = "Confusion matrix") -> None:
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1], ["good", "defect"])
    ax.set_yticks([0, 1], ["good", "defect"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(trues: list[int], scores: list[float], path: Path) -> None:
    if len(set(trues)) < 2:
        LOGGER.warning("only one class present, skipping ROC plot")
        return
    fpr, tpr, _ = roc_curve(trues, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_predictions_csv(
    filenames: list[str], trues: list[int], scores: list[float], preds: list[int], path: Path
) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_label", "pred_score", "pred_label"])
        for fn, t, s, p in zip(filenames, trues, scores, preds, strict=True):
            w.writerow([Path(fn).name, t, f"{s:.6f}", p])


def write_training_log_csv(history: list[EpochStats], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_f1", "val_auroc"])
        for h in history:
            w.writerow([h.epoch, f"{h.train_loss:.6f}", f"{h.val_loss:.6f}", f"{h.val_acc:.6f}", f"{h.val_f1:.6f}", f"{h.val_auroc:.6f}"])


def save_fold_outputs(out: FoldOutput, fold_dir: Path) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_training_log_csv(out.history, fold_dir / "training_log.csv")
    write_predictions_csv(out.filenames, out.true_labels, out.pred_scores, out.pred_labels, fold_dir / "predictions.csv")
    with (fold_dir / "metrics.json").open("w") as f:
        json.dump(
            {"fold": out.fold, "best_f1": out.best_f1, "best_epoch": out.best_epoch, **out.final_metrics},
            f,
            indent=2,
        )
    plot_training_curve(out.history, fold_dir / "training_curve.png")
    plot_confusion_matrix(out.final_metrics["confusion_matrix"], fold_dir / "confusion_matrix.png")
    plot_roc(out.true_labels, out.pred_scores, fold_dir / "roc.png")


# ---------- Mode dispatch ---------------------------------------------------------------------


def run_simple_split(args: argparse.Namespace, device: torch.device) -> int:
    train_good, test_good, defects = gather_samples(args.data_root)
    train_defects, test_defects = split_defects_for_supervised(defects, n_train=7, seed=args.seed)
    train_samples = train_good + train_defects
    test_samples = test_good + test_defects

    out_dir = args.output_dir / "simple_split"
    fold_out = train_one_fold(
        fold_idx=0,
        train_samples=train_samples,
        test_samples=test_samples,
        backbone=args.backbone,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        use_focal_loss=args.use_focal_loss,
        patience=args.patience,
        device=device,
        fold_output_dir=out_dir,
    )
    save_fold_outputs(fold_out, out_dir)
    LOGGER.info(
        "DONE simple_split: best F1=%.4f at epoch %d  | final %s",
        fold_out.best_f1,
        fold_out.best_epoch,
        fold_out.final_metrics,
    )
    return 0


def run_kfold(args: argparse.Namespace, device: torch.device) -> int:
    train_good, test_good, defects = gather_samples(args.data_root)
    pool = train_good + test_good + defects
    y = np.array([s.label for s in pool])
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    fold_outputs: list[FoldOutput] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros_like(y), y), start=1):
        train_samples = [pool[i] for i in train_idx]
        test_samples = [pool[i] for i in test_idx]
        fold_dir = args.output_dir / f"fold_{fold_idx}"
        out = train_one_fold(
            fold_idx=fold_idx,
            train_samples=train_samples,
            test_samples=test_samples,
            backbone=args.backbone,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            num_workers=args.num_workers,
            use_focal_loss=args.use_focal_loss,
            patience=args.patience,
            device=device,
            fold_output_dir=fold_dir,
        )
        save_fold_outputs(out, fold_dir)
        fold_outputs.append(out)
        LOGGER.info("fold %d done: best F1=%.4f", fold_idx, out.best_f1)

    summary = summarize_kfold(fold_outputs)
    with (args.output_dir / "kfold_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("k-fold summary: %s", summary)
    return 0


def summarize_kfold(fold_outputs: list[FoldOutput]) -> dict[str, Any]:
    keys = ("accuracy", "precision", "recall", "f1", "auroc")
    per_fold = [{k: o.final_metrics.get(k, float("nan")) for k in keys} for o in fold_outputs]
    means = {k: float(np.nanmean([m[k] for m in per_fold])) for k in keys}
    stds = {k: float(np.nanstd([m[k] for m in per_fold])) for k in keys}
    return {
        "per_fold": per_fold,
        "mean": means,
        "std": stds,
    }


# ---------- CLI -------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=Path, default=Path("./dataset"))
    parser.add_argument(
        "--backbone",
        choices=("resnet18", "efficientnet_b0", "convnext_tiny"),
        default="efficientnet_b0",
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mode", choices=("simple_split", "kfold"), default="simple_split")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("./results_supervised"))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    set_seed(args.seed)

    if not torch.cuda.is_available():
        LOGGER.warning("CUDA not available — training on CPU will be slow.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("device=%s, backbone=%s, image_size=%d, batch_size=%d", device, args.backbone, args.image_size, args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "simple_split":
        return run_simple_split(args, device)
    return run_kfold(args, device)


if __name__ == "__main__":
    try:
        rc = main()
    except RuntimeError as e:
        if "CUDA" in str(e) or "cuDNN" in str(e):
            print(
                "RuntimeError likely from CUDA/cuDNN. RTX 5090 needs CUDA 12.8 wheels:\n"
                "  uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision",
                file=sys.stderr,
            )
        raise
    raise SystemExit(rc)
