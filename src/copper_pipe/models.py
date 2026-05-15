"""Anomalib model builders with version-tolerant API handling."""

from __future__ import annotations

import logging
from typing import Any

from .data import _filter_kwargs

LOGGER = logging.getLogger(__name__)


def _resolve_class(module_path: str, candidates: tuple[str, ...]):
    """Pick the first class name that exists in module_path (handles naming drift).

    anomalib has used both ``Patchcore`` and ``PatchCore``, ``EfficientAd`` and ``EfficientAD``
    across versions — try them all.
    """
    import importlib

    mod = importlib.import_module(module_path)
    for name in candidates:
        cls = getattr(mod, name, None)
        if cls is not None:
            LOGGER.info("resolved %s.%s", module_path, name)
            return cls
    available = sorted(n for n in dir(mod) if not n.startswith("_"))
    raise ImportError(
        f"none of {candidates} found in {module_path}; available names include: {available[:30]}"
    )


def _maybe_pre_processor(cls: Any, image_size: int) -> Any:
    """Build a model-appropriate PreProcessor for the requested image size, or None."""
    cfg = getattr(cls, "configure_pre_processor", None)
    if cfg is None:
        return None
    try:
        return cfg(image_size=(image_size, image_size))
    except TypeError:
        LOGGER.warning("%s.configure_pre_processor rejected image_size kwarg; using default", cls.__name__)
        return cfg()


def build_patchcore(
    image_size: int = 256,
    backbone: str = "wide_resnet50_2",
    layers: list[str] | None = None,
    coreset_sampling_ratio: float = 0.1,
    num_neighbors: int = 9,
) -> Any:
    """PatchCore (memory-based). Defaults match the original paper."""
    cls = _resolve_class("anomalib.models", ("Patchcore", "PatchCore"))
    kwargs: dict[str, Any] = {
        "backbone": backbone,
        "layers": layers or ["layer2", "layer3"],
        "pre_trained": True,
        "coreset_sampling_ratio": coreset_sampling_ratio,
        "num_neighbors": num_neighbors,
        "pre_processor": _maybe_pre_processor(cls, image_size),
    }
    kwargs = _filter_kwargs(cls.__init__, kwargs)
    LOGGER.info("building PatchCore: image_size=%d, layers=%s, coreset_ratio=%.2f", image_size, kwargs.get("layers"), coreset_sampling_ratio)
    return cls(**kwargs)


def build_efficientad(image_size: int = 256) -> Any:
    """EfficientAD with library defaults; image_size injected via PreProcessor."""
    cls = _resolve_class("anomalib.models", ("EfficientAd", "EfficientAD", "Efficientad"))
    kwargs: dict[str, Any] = {
        "pre_processor": _maybe_pre_processor(cls, image_size),
    }
    kwargs = _filter_kwargs(cls.__init__, kwargs)
    LOGGER.info("building EfficientAD with kwargs keys: %s, image_size=%d", list(kwargs), image_size)
    return cls(**kwargs)


def build_padim(image_size: int = 256, backbone: str = "resnet18") -> Any:
    """PaDiM — parametric memory bank using multivariate gaussian per spatial position."""
    cls = _resolve_class("anomalib.models", ("Padim", "PaDiM", "PADIM"))
    kwargs: dict[str, Any] = {
        "backbone": backbone,
        "layers": ["layer1", "layer2", "layer3"],
        "pre_trained": True,
        "pre_processor": _maybe_pre_processor(cls, image_size),
    }
    kwargs = _filter_kwargs(cls.__init__, kwargs)
    LOGGER.info("building PaDiM: image_size=%d, backbone=%s", image_size, backbone)
    return cls(**kwargs)


def build_dinomaly(image_size: int = 448) -> Any:
    """Dinomaly — DINOv2-based reverse distillation. image_size must be 14-aligned for DINOv2,
    and >= 392 because anomalib's default pre-processor center-crops to 392."""
    cls = _resolve_class("anomalib.models", ("Dinomaly",))
    if image_size % 14 != 0 or image_size < 392:
        LOGGER.warning("Dinomaly: image_size=%d invalid (need >=392, multiple of 14); falling back to 448", image_size)
        image_size = 448
    kwargs: dict[str, Any] = {
        "pre_processor": _maybe_pre_processor(cls, image_size),
    }
    kwargs = _filter_kwargs(cls.__init__, kwargs)
    LOGGER.info("building Dinomaly: image_size=%d", image_size)
    return cls(**kwargs)


MODEL_BUILDERS: dict[str, Any] = {
    "patchcore": build_patchcore,
    "efficientad": build_efficientad,
    "padim": build_padim,
    "dinomaly": build_dinomaly,
}


# EfficientAD requires train_batch_size == 1 in anomalib (it asserts on it).
MODEL_TRAIN_BATCH_SIZE = {
    "patchcore": None,
    "efficientad": 1,
    "padim": None,
    "dinomaly": None,
}


# Memory-based / parametric models converge in 1 epoch.
# EfficientAD and Dinomaly are end-to-end trained.
MODEL_MAX_EPOCHS = {
    "patchcore": 1,
    "efficientad": 30,
    "padim": 1,
    "dinomaly": 20,
}


# Some models pin the input size.
MODEL_REQUIRED_IMAGE_SIZE = {
    "patchcore": None,
    "efficientad": 256,
    "padim": None,
    "dinomaly": 224,
}
