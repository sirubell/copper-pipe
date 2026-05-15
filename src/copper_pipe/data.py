"""DataModule construction for anomalib's Folder layout."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs not accepted by the callable (handles anomalib API drift)."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    accepted = set(sig.parameters.keys())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = set(kwargs) - set(filtered)
    if dropped:
        LOGGER.debug("dropped unsupported kwargs for %s: %s", callable_obj, sorted(dropped))
    return filtered


def build_datamodule(
    data_root: Path,
    image_size: int,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int = 4,
    seed: int = 42,
):
    """Build an anomalib Folder datamodule for the copper-pipe layout."""
    from anomalib.data import Folder

    kwargs: dict[str, Any] = {
        "name": "copper_pipe",
        "root": str(data_root),
        "normal_dir": "train/good",
        "abnormal_dir": "test/defect",
        "normal_test_dir": "test/good",
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        "image_size": (image_size, image_size),
        "seed": seed,
        # Default Folder behavior splits half the test set into a val set, which
        # leaves only 15 images for the test loop. We want all 30 evaluated, so
        # mirror val on the test set instead of carving it out.
        "val_split_mode": "same_as_test",
    }
    kwargs = _filter_kwargs(Folder.__init__, kwargs)
    LOGGER.info("building Folder datamodule with kwargs: %s", kwargs)
    dm = Folder(**kwargs)
    dm.setup()
    return dm
