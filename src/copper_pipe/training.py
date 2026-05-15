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


def run_engine_test(model: Any, datamodule: Any, engine: Any) -> list[dict[str, float]]:
    """Run engine.test() and return the raw metric list (one dict per test loop)."""
    LOGGER.info("running engine.test()")
    results = engine.test(model=model, datamodule=datamodule)
    # results is typically List[Dict[str, float]]; defend against None.
    return results or []
