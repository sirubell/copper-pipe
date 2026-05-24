"""Train + test orchestration around anomalib.engine.Engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .data import _filter_kwargs

LOGGER = logging.getLogger(__name__)


def build_engine(output_dir: Path, max_epochs: int) -> Any:
    """Build an anomalib Engine targeting a GPU."""
    from anomalib.engine import Engine

    kwargs: dict[str, Any] = {
        "accelerator": "gpu",
        "devices": 1,
        "max_epochs": max_epochs,
        "default_root_dir": str(output_dir),
        "logger": False,
    }
    kwargs = _filter_kwargs(Engine.__init__, kwargs)
    LOGGER.info("building Engine with kwargs: %s", kwargs)
    return Engine(**kwargs)


def train_one_model(model: Any, datamodule: Any, engine: Any) -> None:
    """Fit a model on the datamodule using the given engine."""
    LOGGER.info("starting fit")
    engine.fit(model=model, datamodule=datamodule)
    LOGGER.info("fit complete")


def find_checkpoint(engine: Any) -> Path | None:
    """Locate the .ckpt anomalib wrote during fit() (best, else last, else scan dir)."""
    cb = getattr(engine.trainer, "checkpoint_callback", None)
    for attr in ("best_model_path", "last_model_path"):
        path = getattr(cb, attr, None) if cb is not None else None
        if path:
            p = Path(path)
            if p.is_file():
                return p
    # Fallback: scan default_root_dir for *.ckpt and take the newest.
    root = Path(getattr(engine.trainer, "default_root_dir", "."))
    ckpts = sorted(root.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def run_engine_test(model: Any, datamodule: Any, engine: Any) -> list[dict[str, float]]:
    """Run engine.test() and return the raw metric list (one dict per test loop)."""
    LOGGER.info("running engine.test()")
    results = engine.test(model=model, datamodule=datamodule)
    # results is typically List[Dict[str, float]]; defend against None.
    return results or []
