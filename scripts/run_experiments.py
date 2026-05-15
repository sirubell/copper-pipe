"""Ablation runner: tries multiple anomalib model/config variants on the copper-pipe
dataset and computes a score-ensemble across them.

Each experiment trains, runs engine.predict() for per-image scores, evaluates, and
saves results under results_experiments/<name>/. Ensembles (mean of normalized scores
across the requested set of variants) are appended to the final comparison CSV.

Example:
    uv run python scripts/run_experiments.py
    uv run python scripts/run_experiments.py --variants patchcore_baseline patchcore_full
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

LOGGER = logging.getLogger("run_experiments")


@dataclass
class Variant:
    name: str
    builder: Callable[[int], Any]  # takes image_size, returns model
    image_size: int
    train_batch_size: int
    max_epochs: int


def all_variants() -> list[Variant]:
    """All known experiment configurations."""
    from copper_pipe.models import (
        build_dinomaly,
        build_efficientad,
        build_padim,
        build_patchcore,
    )

    return [
        Variant(
            "patchcore_baseline",
            lambda sz: build_patchcore(image_size=sz, layers=["layer2", "layer3"], coreset_sampling_ratio=0.1),
            image_size=256, train_batch_size=8, max_epochs=1,
        ),
        Variant(
            "patchcore_full_coreset",
            lambda sz: build_patchcore(image_size=sz, layers=["layer2", "layer3"], coreset_sampling_ratio=1.0),
            image_size=256, train_batch_size=8, max_epochs=1,
        ),
        Variant(
            "patchcore_3layers",
            lambda sz: build_patchcore(image_size=sz, layers=["layer1", "layer2", "layer3"], coreset_sampling_ratio=0.1),
            image_size=256, train_batch_size=8, max_epochs=1,
        ),
        Variant(
            "patchcore_big",
            lambda sz: build_patchcore(image_size=sz, layers=["layer2", "layer3"], coreset_sampling_ratio=0.1),
            image_size=384, train_batch_size=4, max_epochs=1,
        ),
        Variant(
            "patchcore_kitchen_sink",
            lambda sz: build_patchcore(image_size=sz, layers=["layer1", "layer2", "layer3"], coreset_sampling_ratio=0.25),
            image_size=384, train_batch_size=4, max_epochs=1,
        ),
        Variant(
            "padim",
            lambda sz: build_padim(image_size=sz),
            image_size=256, train_batch_size=8, max_epochs=1,
        ),
        Variant(
            "dinomaly",
            lambda sz: build_dinomaly(image_size=sz),
            image_size=448, train_batch_size=4, max_epochs=20,
        ),
        Variant(
            "efficientad",
            lambda sz: build_efficientad(image_size=sz),
            image_size=256, train_batch_size=1, max_epochs=30,
        ),
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, default=Path("./dataset"))
    p.add_argument("--output_dir", type=Path, default=Path("./results_experiments"))
    p.add_argument("--variants", nargs="+", default=None, help="subset of variant names to run (default: all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ensemble", nargs="+", default=None, help="variant names to score-average for the ensemble row (default: all completed)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def banner(msg: str) -> None:
    line = "=" * 72
    LOGGER.info("\n%s\n  %s\n%s", line, msg, line)


def run_variant(v: Variant, args: argparse.Namespace) -> dict[str, Any] | None:
    """Train + evaluate one variant. Returns metrics dict or None if it crashed."""
    from copper_pipe.data import build_datamodule
    from copper_pipe.evaluation import evaluate_one_model
    from copper_pipe.reporting import write_predictions_csv
    from copper_pipe.training import build_engine, run_engine_test, train_one_model

    banner(f"START: {v.name}")
    try:
        datamodule = build_datamodule(
            data_root=args.data_root,
            image_size=v.image_size,
            train_batch_size=v.train_batch_size,
            eval_batch_size=8,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        model = v.builder(v.image_size)
        variant_dir = args.output_dir / v.name
        engine = build_engine(output_dir=variant_dir, max_epochs=v.max_epochs)

        train_one_model(model=model, datamodule=datamodule, engine=engine)
        raw_test = run_engine_test(model=model, datamodule=datamodule, engine=engine)
        LOGGER.info("engine.test() raw output: %s", raw_test)

        result = evaluate_one_model(model_name=v.name, model=model, engine=engine, datamodule=datamodule)
    except Exception as e:
        LOGGER.exception("variant %s failed: %s", v.name, e)
        return None

    write_predictions_csv(result, args.output_dir / v.name / "predictions.csv")
    LOGGER.info(
        "%s: AUROC=%.4f F1=%.4f Acc=%.4f P=%.4f R=%.4f thr=%.4f ms=%.3f",
        v.name,
        result.image_auroc,
        result.f1,
        result.accuracy,
        result.precision,
        result.recall,
        result.threshold,
        result.inference_ms_per_image,
    )
    banner(f"END:   {v.name}")
    return {
        "model": v.name,
        "image_AUROC": result.image_auroc,
        "F1": result.f1,
        "Accuracy": result.accuracy,
        "Precision": result.precision,
        "Recall": result.recall,
        "threshold": result.threshold,
        "inference_ms_per_image": result.inference_ms_per_image,
        # for ensembling:
        "_predictions": result.predictions,
    }


def compute_ensembles(
    completed: list[dict[str, Any]], selected_names: list[str] | None
) -> list[dict[str, Any]]:
    """Compute score-mean ensembles. Returns 1+ ensemble rows."""
    if not completed:
        return []
    by_name = {r["model"]: r for r in completed}
    selected = selected_names or [r["model"] for r in completed]
    selected = [n for n in selected if n in by_name]
    if len(selected) < 2:
        LOGGER.info("ensemble skipped (need >=2 completed variants)")
        return []

    # Align predictions by filename. Each result has .predictions list with .filename, .pred_score, .true_label.
    name_to_scores: dict[str, dict[str, float]] = {}
    name_to_truth: dict[str, dict[str, int]] = {}
    for name in selected:
        preds = by_name[name]["_predictions"]
        name_to_scores[name] = {p.filename: p.pred_score for p in preds}
        name_to_truth[name] = {p.filename: p.true_label for p in preds}

    # Common filenames across selected variants
    common = set.intersection(*[set(name_to_scores[n].keys()) for n in selected])
    if not common:
        LOGGER.warning("ensemble: no overlapping filenames across variants")
        return []
    truth = {fn: next(iter(name_to_truth[n][fn] for n in selected)) for fn in common}

    avg_scores = {
        fn: float(np.mean([name_to_scores[n][fn] for n in selected])) for fn in common
    }
    # We also try a rank-average ensemble (more robust when score scales differ).
    rank_avg_scores = rank_average(name_to_scores, selected, list(common))

    rows = []
    for ens_name, score_dict in (
        (f"ensemble_mean({'+'.join(selected)})", avg_scores),
        (f"ensemble_rank({'+'.join(selected)})", rank_avg_scores),
    ):
        rows.append(metrics_from_scores(ens_name, truth, score_dict))
    return rows


def rank_average(
    name_to_scores: dict[str, dict[str, float]], names: list[str], filenames: list[str]
) -> dict[str, float]:
    """Convert each model's scores to ranks, then average. Score-scale-invariant."""
    rank_per_model: dict[str, dict[str, float]] = {}
    for n in names:
        items = sorted(filenames, key=lambda fn: name_to_scores[n][fn])
        # rank 0 = lowest; normalize to [0, 1]
        rank_per_model[n] = {fn: i / max(1, len(items) - 1) for i, fn in enumerate(items)}
    return {
        fn: float(np.mean([rank_per_model[n][fn] for n in names])) for fn in filenames
    }


def metrics_from_scores(
    name: str, truth: dict[str, int], scores: dict[str, float]
) -> dict[str, Any]:
    """Compute AUROC + max-F1-threshold-based confusion metrics from raw scores."""
    from copper_pipe.evaluation import _binary_auroc, _confusion_metrics

    fns = list(scores.keys())
    s = [scores[fn] for fn in fns]
    y = [truth[fn] for fn in fns]
    auroc = _binary_auroc(s, y)
    # For ensemble there's no model-side threshold; sweep for the F1-optimal one over actual scores.
    if not s:
        return {"model": name, "image_AUROC": float("nan"), "F1": 0.0, "Accuracy": 0.0,
                "Precision": 0.0, "Recall": 0.0, "threshold": float("nan"), "inference_ms_per_image": float("nan")}
    candidates = sorted(set(s))
    best_f1, best_thr = -1.0, candidates[len(candidates) // 2]
    for thr in candidates:
        _, _, _, f1, _ = _confusion_metrics(s, y, thr)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    acc, prec, rec, f1, _ = _confusion_metrics(s, y, best_thr)
    LOGGER.info("ensemble %s: AUROC=%.4f F1=%.4f thr=%.4f", name, auroc, f1, best_thr)
    return {
        "model": name,
        "image_AUROC": auroc,
        "F1": f1,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "threshold": best_thr,
        "inference_ms_per_image": float("nan"),
    }


def write_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "image_AUROC", "F1", "Accuracy", "Precision", "Recall", "threshold", "inference_ms_per_image"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["model"]] + [f"{r[c]:.4f}" if isinstance(r[c], float) and not np.isnan(r[c]) else "" for c in cols[1:]])


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print()
    print("## Experiment comparison")
    print()
    print("| Model | AUROC | F1 | Acc | Precision | Recall | Threshold | ms/img |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        def fmt(x: Any) -> str:
            if isinstance(x, float):
                if np.isnan(x):
                    return "—"
                return f"{x:.4f}"
            return str(x)
        print(
            f"| {r['model']} | {fmt(r['image_AUROC'])} | {fmt(r['F1'])} | {fmt(r['Accuracy'])} "
            f"| {fmt(r['Precision'])} | {fmt(r['Recall'])} | {fmt(r['threshold'])} | {fmt(r['inference_ms_per_image'])} |"
        )
    print()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    set_seed(args.seed)

    variants = all_variants()
    if args.variants:
        names = set(args.variants)
        unknown = names - {v.name for v in variants}
        if unknown:
            raise SystemExit(f"unknown variants: {sorted(unknown)}; known: {[v.name for v in variants]}")
        variants = [v for v in variants if v.name in names]

    if not (args.data_root / "train" / "good").is_dir():
        raise SystemExit(f"missing {args.data_root}/train/good — run scripts/split_dataset.py first")

    completed: list[dict[str, Any]] = []
    for v in variants:
        result = run_variant(v, args)
        if result is not None:
            completed.append(result)

    rows = [
        {k: v for k, v in r.items() if not k.startswith("_")} for r in completed
    ]
    ensemble_rows = compute_ensembles(completed, args.ensemble)
    rows.extend(ensemble_rows)

    write_comparison(rows, args.output_dir / "comparison.csv")
    LOGGER.info("wrote %s", args.output_dir / "comparison.csv")
    print_markdown(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
