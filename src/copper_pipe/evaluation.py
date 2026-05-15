"""Per-image prediction extraction, metric computation, and inference timing.

We deliberately compute metrics from raw scores/labels ourselves (rather than relying
on anomalib's metric-key naming) so the script works across anomalib versions.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class ImagePrediction:
    filename: str
    true_label: int
    pred_score: float
    pred_label: int


@dataclass
class EvalResult:
    model_name: str
    image_auroc: float
    f1: float
    accuracy: float
    precision: float
    recall: float
    threshold: float
    inference_ms_per_image: float
    predictions: list[ImagePrediction] = field(default_factory=list)


def _get_attr_or_key(obj: Any, *names: str) -> Any:
    """Try several attribute names, then dict keys, then return None."""
    for name in names:
        if hasattr(obj, name):
            v = getattr(obj, name)
            if v is not None:
                return v
    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
    return None


def _to_list(t: Any) -> list:
    if t is None:
        return []
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().flatten().tolist()
    if isinstance(t, (list, tuple)):
        return list(t)
    return [t]


def _extract_predictions(batches: Iterable[Any]) -> tuple[list[float], list[int], list[str]]:
    """Flatten engine.predict() output into (scores, labels, filenames)."""
    scores: list[float] = []
    labels: list[int] = []
    paths: list[str] = []
    for batch in batches:
        s = _get_attr_or_key(batch, "pred_score", "pred_scores", "anomaly_score")
        y = _get_attr_or_key(batch, "gt_label", "label", "labels")
        p = _get_attr_or_key(batch, "image_path", "image_paths", "path", "paths")
        s_list = [float(x) for x in _to_list(s)]
        y_list = [int(x) for x in _to_list(y)]
        p_list = [str(x) for x in _to_list(p)]
        # Some versions emit a single path string for the batch; pad if needed.
        if len(p_list) == 1 and len(s_list) > 1:
            p_list = p_list * len(s_list)
        if not (len(s_list) == len(y_list) == len(p_list)):
            LOGGER.warning(
                "batch field length mismatch: scores=%d labels=%d paths=%d — truncating",
                len(s_list),
                len(y_list),
                len(p_list),
            )
            n = min(len(s_list), len(y_list), len(p_list))
            s_list, y_list, p_list = s_list[:n], y_list[:n], p_list[:n]
        scores.extend(s_list)
        labels.extend(y_list)
        paths.extend(p_list)
    return scores, labels, paths


def _binary_auroc(scores: list[float], labels: list[int]) -> float:
    """Trapezoidal AUROC; ties handled by averaging ranks."""
    if not scores or len(set(labels)) < 2:
        return float("nan")
    # Sort by score descending; iterate and compute ROC points.
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    pos = sum(1 for v in labels if v == 1)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    prev_score = None
    prev_tp = prev_fp = 0
    auc = 0.0
    for i in order:
        s = scores[i]
        if prev_score is not None and s != prev_score:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_tp, prev_fp = tp, fp
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        prev_score = s
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (pos * neg)


def _confusion_metrics(
    scores: list[float], labels: list[int], threshold: float
) -> tuple[float, float, float, float, list[int]]:
    """Return (accuracy, precision, recall, f1, pred_labels) at the given threshold."""
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for p, y in zip(preds, labels, strict=False) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels, strict=False) if p == 1 and y == 0)
    tn = sum(1 for p, y in zip(preds, labels, strict=False) if p == 0 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels, strict=False) if p == 0 and y == 1)
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1, preds


def _read_threshold_attr(owner: Any, attr: str) -> float | None:
    """Pull a threshold-shaped value (tensor / scalar / object with .value) off owner.attr."""
    val = getattr(owner, attr, None)
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        if val.numel() == 1 and not torch.isnan(val).any():
            return float(val.item())
        return None
    if isinstance(val, (int, float)):
        return float(val)
    inner = getattr(val, "value", None)
    if isinstance(inner, torch.Tensor) and inner.numel() == 1 and not torch.isnan(inner).any():
        return float(inner.item())
    if isinstance(inner, (int, float)):
        return float(inner)
    return None


def _resolve_threshold(model: Any, scores: list[float], labels: list[int]) -> float:
    """Pull image threshold from model. Prefer normalized (matches predict() score scale)."""
    # predict() returns post-processed scores in [0, 1]; pair them with the
    # normalized threshold (typically 0.5). Fall back to raw thresholds, then
    # to a max-F1 sweep over the actual scores.
    owners: list[Any] = []
    pp = getattr(model, "post_processor", None)
    if pp is not None:
        owners.append(pp)
    owners.append(model)
    attrs = ("normalized_image_threshold", "image_threshold", "_image_threshold")
    for owner in owners:
        for attr in attrs:
            v = _read_threshold_attr(owner, attr)
            if v is not None:
                LOGGER.info("threshold from %s.%s = %.4f", type(owner).__name__, attr, v)
                return v
    # Fallback: best-F1 threshold over the test set.
    if not scores:
        return 0.5
    sorted_scores = sorted(set(scores))
    best_f1, best_thr = -1.0, sorted_scores[len(sorted_scores) // 2]
    for thr in sorted_scores:
        _, _, _, f1, _ = _confusion_metrics(scores, labels, thr)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    LOGGER.warning("threshold not found on model; chose max-F1 threshold = %.4f", best_thr)
    return float(best_thr)


def evaluate_one_model(model_name: str, model: Any, engine: Any, datamodule: Any) -> EvalResult:
    """Run engine.predict + compute all required metrics + measure inference time."""
    LOGGER.info("running engine.predict() for per-image scores")
    predictions = engine.predict(model=model, datamodule=datamodule) or []
    scores, labels, paths = _extract_predictions(predictions)
    LOGGER.info("collected %d test predictions", len(scores))

    auroc = _binary_auroc(scores, labels)
    threshold = _resolve_threshold(model, scores, labels)
    acc, prec, rec, f1, pred_labels = _confusion_metrics(scores, labels, threshold)

    inference_ms = measure_inference_time(model, datamodule)

    image_preds = [
        ImagePrediction(
            filename=Path(p).name,
            true_label=int(y),
            pred_score=float(s),
            pred_label=int(pl),
        )
        for p, y, s, pl in zip(paths, labels, scores, pred_labels, strict=False)
    ]

    return EvalResult(
        model_name=model_name,
        image_auroc=float(auroc),
        f1=float(f1),
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        threshold=float(threshold),
        inference_ms_per_image=float(inference_ms),
        predictions=image_preds,
    )


def measure_inference_time(model: Any, datamodule: Any, warmup: int = 5, n_timed: int = 20) -> float:
    """Time single-image forwards through the model on GPU. Returns ms/image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = model.to(device)
    except Exception:
        LOGGER.warning("model.to(device) failed; timing on whatever device it sits on")
    model.eval()

    loader = datamodule.test_dataloader()
    # Pull image tensors out one at a time.
    tensors: list[torch.Tensor] = []
    needed = warmup + n_timed
    for batch in loader:
        img = _get_attr_or_key(batch, "image", "images")
        if img is None or not isinstance(img, torch.Tensor):
            continue
        for i in range(img.shape[0]):
            tensors.append(img[i : i + 1].to(device))
            if len(tensors) >= needed:
                break
        if len(tensors) >= needed:
            break

    if len(tensors) < needed:
        LOGGER.warning(
            "only %d test images available, need %d for warmup+timed — recycling",
            len(tensors),
            needed,
        )
        if not tensors:
            return float("nan")
        while len(tensors) < needed:
            tensors.append(tensors[len(tensors) % max(1, len(tensors))])

    use_cuda = device.type == "cuda"
    with torch.inference_mode():
        # Warmup
        for x in tensors[:warmup]:
            _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()
        # Timed
        start = time.perf_counter()
        for x in tensors[warmup : warmup + n_timed]:
            _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    ms_per_image = (elapsed / n_timed) * 1000.0
    LOGGER.info("inference: %.3f ms/image (averaged over %d, %d warmup)", ms_per_image, n_timed, warmup)
    return ms_per_image
