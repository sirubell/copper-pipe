"""Run a trained anomalib model on a folder of images (TA-provided test set).

Outputs:
    <output>/predictions.csv     filename, pred_score, pred_label
    + a terminal table

Example:
    uv run python scripts/predict.py \\
        --model patchcore \\
        --checkpoint results/patchcore/checkpoint.ckpt \\
        --test_dir ./teacher_test \\
        --output ./submissions/patchcore

The test directory should contain images flat-out (any of .png/.jpg/.bmp/.tif).
Subdirectories are NOT recursed — keep submission images at the top level.

pred_label = 1 means defect/abnormal, 0 means good/normal.
The default decision threshold is 0.45 on anomalib's min-max-normalized score —
slightly below the nominal 0.5 to bias toward recall (in industrial QC, missing
a defect costs more than a false positive). Override with --threshold if needed.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

LOGGER = logging.getLogger("predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["patchcore", "efficientad", "padim", "dinomaly"],
        help="model architecture matching the checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to the .ckpt file produced by train_anomalib.py",
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        required=True,
        help="folder of images to predict on (flat layout)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./predictions"),
        help="output directory (predictions.csv goes here)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=384,
        help="resize images to this size before inference (defaults to 384 — our best PatchCore setting)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help=(
            "decision threshold on the normalized [0,1] score (default 0.45). "
            "Slightly below anomalib's nominal 0.5 to bias toward recall — "
            "missing a defect costs more than a false positive in production."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not args.test_dir.is_dir():
        raise SystemExit(f"test_dir not found or not a directory: {args.test_dir}")

    import torch
    from anomalib.engine import Engine

    from copper_pipe.evaluation import _get_attr_or_key, _to_list
    from copper_pipe.models import MODEL_BUILDERS, MODEL_REQUIRED_IMAGE_SIZE

    image_size = MODEL_REQUIRED_IMAGE_SIZE[args.model] or args.image_size
    if image_size != args.image_size:
        LOGGER.info(
            "%s pins image_size=%d (ignoring --image_size=%d)",
            args.model,
            image_size,
            args.image_size,
        )

    LOGGER.info("building model: %s (image_size=%d)", args.model, image_size)
    model = MODEL_BUILDERS[args.model](image_size=image_size)

    args.output.mkdir(parents=True, exist_ok=True)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if accelerator == "cpu":
        LOGGER.warning("CUDA not available — running on CPU (this will be slower).")

    # anomalib's ImageVisualizer callback writes a heatmap PNG per input image
    # into <default_root_dir>/<ModelName>/latest/images/. We don't need those
    # for submission — route them into a tempdir that we delete after extracting
    # the in-memory predictions, leaving args.output clean (just predictions.csv).
    with tempfile.TemporaryDirectory(prefix="predict_workspace_") as workspace:
        engine = Engine(
            accelerator=accelerator,
            devices=1,
            default_root_dir=workspace,
            logger=False,
        )

        LOGGER.info("predicting %s with ckpt=%s", args.test_dir, args.checkpoint)
        predictions = engine.predict(
            model=model,
            ckpt_path=str(args.checkpoint),
            data_path=str(args.test_dir),
        )
    batches = predictions or []
    # Predict-mode batches don't have ground-truth labels, so we can't reuse
    # _extract_predictions (which requires len(labels) == len(scores)).
    scores: list[float] = []
    paths: list[str] = []
    for batch in batches:
        s = _get_attr_or_key(batch, "pred_score", "pred_scores", "anomaly_score")
        p = _get_attr_or_key(batch, "image_path", "image_paths", "path", "paths")
        s_list = [float(x) for x in _to_list(s)]
        p_list = [str(x) for x in _to_list(p)]
        if len(p_list) == 1 and len(s_list) > 1:
            p_list = p_list * len(s_list)
        if len(s_list) != len(p_list):
            LOGGER.warning(
                "batch field length mismatch: scores=%d paths=%d — truncating",
                len(s_list),
                len(p_list),
            )
            n = min(len(s_list), len(p_list))
            s_list, p_list = s_list[:n], p_list[:n]
        scores.extend(s_list)
        paths.extend(p_list)

    if not scores:
        raise SystemExit(
            "engine.predict() returned no predictions — check that test_dir has "
            "supported image files (.png/.jpg/.jpeg/.bmp/.tif/.tiff)"
        )

    pred_labels = [1 if s >= args.threshold else 0 for s in scores]
    rows = sorted(
        zip(paths, scores, pred_labels, strict=True),
        key=lambda r: Path(r[0]).name.lower(),
    )

    out_csv = args.output / "predictions.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "pred_score", "pred_label"])
        for p, s, lbl in rows:
            writer.writerow([Path(p).name, f"{float(s):.6f}", int(lbl)])

    # Terminal output: markdown table + summary line.
    name_w = max(8, *(len(Path(p).name) for p, _, _ in rows))
    print()
    print(f"| {'filename':<{name_w}} | pred_score | pred_label |")
    print(f"|{'-' * (name_w + 2)}|{'-' * 12}|{'-' * 12}|")
    for p, s, lbl in rows:
        print(f"| {Path(p).name:<{name_w}} | {float(s):10.4f} | {int(lbl):10d} |")

    n_total = len(rows)
    n_defect = sum(1 for _, _, lbl in rows if lbl == 1)
    n_normal = n_total - n_defect
    print()
    print(
        f"Total: {n_total}    Predicted normal: {n_normal}    "
        f"Predicted defect: {n_defect}    Threshold: {args.threshold}"
    )
    print(f"CSV:   {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
