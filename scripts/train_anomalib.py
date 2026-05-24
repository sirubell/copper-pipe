"""Train PatchCore + EfficientAD on the copper-pipe dataset and compare them.

Example:
    uv run python scripts/train_anomalib.py
    uv run python scripts/train_anomalib.py --models patchcore
    uv run python scripts/train_anomalib.py --models efficientad --image_size 256
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

# Make src/ importable without installing the package
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

LOGGER = logging.getLogger("train_anomalib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_root", type=Path, default=Path("./dataset"))
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=Path, default=Path("./results"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["patchcore", "efficientad"],
        choices=["patchcore", "efficientad"],
        help="which models to train",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def check_imports() -> None:
    missing: list[str] = []
    for name in ("anomalib", "torch", "torchvision", "timm"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        msg = (
            "missing required packages: "
            + ", ".join(missing)
            + "\nrun `uv sync` (or install them manually) before training."
        )
        raise SystemExit(msg)


def banner(msg: str) -> None:
    line = "=" * 72
    LOGGER.info("\n%s\n  %s\n%s", line, msg, line)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    check_imports()
    set_seed(args.seed)

    # Imports deferred until after check_imports() so the error message is friendly.
    import anomalib

    from copper_pipe.data import build_datamodule
    from copper_pipe.evaluation import EvalResult, evaluate_one_model
    from copper_pipe.models import (
        MODEL_BUILDERS,
        MODEL_MAX_EPOCHS,
        MODEL_REQUIRED_IMAGE_SIZE,
        MODEL_TRAIN_BATCH_SIZE,
    )
    from copper_pipe.reporting import compare_and_save
    from copper_pipe.training import build_engine, find_checkpoint, run_engine_test, train_one_model

    LOGGER.info(
        "anomalib %s, output_dir=%s, data_root=%s",
        anomalib.__version__,
        args.output_dir,
        args.data_root,
    )

    if not (args.data_root / "train" / "good").is_dir():
        raise SystemExit(
            f"expected {args.data_root}/train/good to exist — run scripts/prepare_dataset.py first"
        )

    results: list[EvalResult] = []
    for name in args.models:
        banner(f"START: {name}")
        image_size = MODEL_REQUIRED_IMAGE_SIZE[name] or args.image_size
        train_bs = MODEL_TRAIN_BATCH_SIZE[name] or args.batch_size
        max_epochs = MODEL_MAX_EPOCHS[name]
        if image_size != args.image_size:
            LOGGER.info(
                "%s requires image_size=%d (overriding --image_size=%d)",
                name,
                image_size,
                args.image_size,
            )
        if train_bs != args.batch_size:
            LOGGER.info(
                "%s requires train_batch_size=%d (overriding --batch_size=%d)",
                name,
                train_bs,
                args.batch_size,
            )

        datamodule = build_datamodule(
            data_root=args.data_root,
            image_size=image_size,
            train_batch_size=train_bs,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        model = MODEL_BUILDERS[name](image_size=image_size)
        model_output = args.output_dir / name
        engine = build_engine(output_dir=model_output, max_epochs=max_epochs)

        train_one_model(model=model, datamodule=datamodule, engine=engine)
        raw_test = run_engine_test(model=model, datamodule=datamodule, engine=engine)
        LOGGER.info("engine.test() raw output: %s", raw_test)

        # Copy the auto-saved Lightning ckpt to a stable path for predict.py.
        ckpt_src = find_checkpoint(engine)
        if ckpt_src is not None:
            ckpt_dst = model_output / "checkpoint.ckpt"
            ckpt_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ckpt_src, ckpt_dst)
            LOGGER.info("checkpoint: %s", ckpt_dst)
        else:
            LOGGER.warning(
                "could not locate a .ckpt for %s; predict.py will need a manual path", name
            )

        result = evaluate_one_model(
            model_name=name, model=model, engine=engine, datamodule=datamodule
        )
        LOGGER.info(
            "%s: AUROC=%.4f F1=%.4f Acc=%.4f P=%.4f R=%.4f thr=%.4f ms=%.3f",
            name,
            result.image_auroc,
            result.f1,
            result.accuracy,
            result.precision,
            result.recall,
            result.threshold,
            result.inference_ms_per_image,
        )
        results.append(result)
        banner(f"END:   {name}")

    table = compare_and_save(results, args.output_dir)
    # Final results are the one thing we print directly (per task spec).
    print()
    print("## Comparison")
    print()
    print(table)
    print()
    print(f"CSV: {args.output_dir / 'comparison.csv'}")
    print()
    print("Checkpoints (use with predict.py --checkpoint):")
    for name in args.models:
        ckpt = args.output_dir / name / "checkpoint.ckpt"
        if ckpt.is_file():
            print(f"  {name:<12} {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
