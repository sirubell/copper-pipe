"""Split a copper-pipe-style dataset (normal + abnormal folders) into the
anomalib ``Folder`` layout.

Two layouts are supported via ``--layout``:

``split`` (legacy, internal-evaluation mode)::

    <output>/train/good/      (train_ratio * |normal|)
    <output>/test/good/       (remainder of normal)
    <output>/test/defect/     (all abnormal)

``full`` (recommended for final submission — train on all available data,
let the TA's held-out test set be the true evaluation)::

    <output>/train/good/      all normal images
    <output>/test/good/       symlinks to the same normal images
                              (purely so anomalib's post-processor can
                              calibrate its min/max normalization range)
    <output>/test/defect/     all abnormal images

Also writes ``<output>/split_manifest.csv`` for traceability.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import shutil
import sys
from pathlib import Path

LOGGER = logging.getLogger("split_dataset")

IMG_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(directory: Path) -> list[Path]:
    """Return image files in ``directory`` (case-insensitive on extension), sorted."""
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    out: list[Path] = []
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


def confirm_overwrite(target: Path, force: bool) -> bool:
    """Return True iff target may be wiped. Asks the user if --force is not set."""
    if not target.exists():
        return True
    if force:
        return True
    if not sys.stdin.isatty():
        LOGGER.error("%s already exists and stdin is not a TTY; pass --force to overwrite", target)
        return False
    answer = input(f"{target} already exists. Overwrite? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def place_file(src: Path, dst: Path, mode: str) -> None:
    """Copy or symlink ``src`` to ``dst``. Existing dst is removed first."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        # Use absolute path for symlinks so the file resolves regardless of cwd.
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"unknown mode: {mode!r}")


def split_normal(files: list[Path], train_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"--train_ratio must be in (0, 1), got {train_ratio}")
    rng = random.Random(seed)
    shuffled = files.copy()
    rng.shuffle(shuffled)
    n_train = round(len(shuffled) * train_ratio)
    n_train = max(1, min(n_train, len(shuffled) - 1))
    return shuffled[:n_train], shuffled[n_train:]


def run_split(
    normal_src: Path,
    abnormal_src: Path,
    output: Path,
    train_ratio: float,
    seed: int,
    mode: str,
    force: bool,
    layout: str = "full",
) -> int:
    normal_files = list_images(normal_src)
    abnormal_files = list_images(abnormal_src)
    LOGGER.info("found %d normal, %d abnormal images", len(normal_files), len(abnormal_files))

    train_dir = output / "train" / "good"
    test_good_dir = output / "test" / "good"
    test_defect_dir = output / "test" / "defect"

    for d in (train_dir, test_good_dir, test_defect_dir):
        if not confirm_overwrite(d, force):
            LOGGER.error("aborted by user")
            return 2
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    if layout == "split":
        train_normal, test_normal = split_normal(normal_files, train_ratio, seed)
    elif layout == "full":
        # Use every normal image for training. Mirror them into test/good as
        # symlinks so anomalib's post-processor still sees both classes
        # during validation/test (needed to calibrate min/max normalization),
        # but no images are actually held out from training.
        train_normal = list(normal_files)
        test_normal = list(normal_files)
    else:
        raise ValueError(f"unknown layout: {layout!r}")

    # Sanity: in 'split' layout there must be no overlap; in 'full' the same files
    # appear in both train and test by design, so we skip the overlap check.
    if layout == "split":
        train_names = {p.name for p in train_normal}
        test_names = {p.name for p in test_normal}
        overlap = train_names & test_names
        if overlap:
            raise RuntimeError(
                f"BUG: overlap between train/good and test/good: {sorted(overlap)[:5]}…"
            )

    manifest_rows: list[tuple[str, str, str, str]] = []

    def _emit(files: list[Path], target_dir: Path, split: str, label: str, place_mode: str) -> None:
        for src in files:
            dst = target_dir / src.name
            place_file(src, dst, place_mode)
            manifest_rows.append((split, label, str(src), str(dst)))

    _emit(train_normal, train_dir, "train", "good", mode)
    # In 'full' layout, force symlinks for the mirrored test/good copies
    # regardless of --mode, so we don't duplicate every normal image on disk.
    test_good_mode = "symlink" if layout == "full" else mode
    _emit(test_normal, test_good_dir, "test", "good", test_good_mode)
    _emit(abnormal_files, test_defect_dir, "test", "defect", mode)

    manifest_path = output / "split_manifest.csv"
    output.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "original_path", "new_path"])
        writer.writerows(manifest_rows)

    LOGGER.info("final stats:")
    LOGGER.info("  layout       = %s", layout)
    LOGGER.info("  train/good   = %d", len(train_normal))
    LOGGER.info("  test/good    = %d", len(test_normal))
    LOGGER.info("  test/defect  = %d", len(abnormal_files))
    LOGGER.info("  manifest     → %s (%d rows)", manifest_path, len(manifest_rows))
    LOGGER.info("  mode         = %s, seed = %d, train_ratio = %.4f", mode, seed, train_ratio)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--normal_src", type=Path, required=True, help="folder of normal images")
    parser.add_argument(
        "--abnormal_src", type=Path, required=True, help="folder of abnormal images"
    )
    parser.add_argument("--output", type=Path, default=Path("./dataset"), help="output root")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.78,
        help="fraction of normal → train (split layout only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="how to place files (full layout always symlinks test/good)",
    )
    parser.add_argument(
        "--layout",
        choices=("full", "split"),
        default="full",
        help="full: train on all normal (recommended for submission). split: hold out 22%% for internal eval.",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing output without asking"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    return run_split(
        normal_src=args.normal_src,
        abnormal_src=args.abnormal_src,
        output=args.output,
        train_ratio=args.train_ratio,
        seed=args.seed,
        mode=args.mode,
        force=args.force,
        layout=args.layout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
