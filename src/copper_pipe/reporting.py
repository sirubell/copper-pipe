"""CSV + markdown table writers for the comparison report."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from .evaluation import EvalResult

LOGGER = logging.getLogger(__name__)


def write_predictions_csv(result: EvalResult, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label", "pred_score", "pred_label"])
        for p in result.predictions:
            writer.writerow([p.filename, p.true_label, f"{p.pred_score:.6f}", p.pred_label])
    LOGGER.info("wrote %d rows → %s", len(result.predictions), target)


def write_comparison_csv(results: list[EvalResult], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "image_AUROC",
                "F1",
                "Accuracy",
                "Precision",
                "Recall",
                "threshold",
                "inference_ms_per_image",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.model_name,
                    f"{r.image_auroc:.4f}",
                    f"{r.f1:.4f}",
                    f"{r.accuracy:.4f}",
                    f"{r.precision:.4f}",
                    f"{r.recall:.4f}",
                    f"{r.threshold:.4f}",
                    f"{r.inference_ms_per_image:.3f}",
                ]
            )
    LOGGER.info("wrote comparison → %s", target)


def render_markdown_table(results: list[EvalResult]) -> str:
    header = (
        "| Model | image AUROC | F1 | Accuracy | Precision | Recall | Threshold | ms/img |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    rows = [
        f"| {r.model_name} | {r.image_auroc:.4f} | {r.f1:.4f} | {r.accuracy:.4f} "
        f"| {r.precision:.4f} | {r.recall:.4f} | {r.threshold:.4f} | {r.inference_ms_per_image:.3f} |"
        for r in results
    ]
    return "\n".join([header, *rows])


def compare_and_save(results: list[EvalResult], output_dir: Path) -> str:
    """Persist per-model predictions + comparison CSV, return markdown table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        write_predictions_csv(r, output_dir / r.model_name / "predictions.csv")
    write_comparison_csv(results, output_dir / "comparison.csv")
    return render_markdown_table(results)
