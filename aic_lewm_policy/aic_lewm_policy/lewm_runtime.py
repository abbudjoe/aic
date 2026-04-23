"""Lazy LEWM runtime loading.

This module is intentionally optional at import time. The AIC lifecycle node has
strict configure/activate deadlines, so heavyweight ML dependencies are loaded
only when a checkpoint is explicitly requested.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LewmRuntime:
    model: Any | None
    checkpoint: str | None
    device: str | None
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.model is not None


def load_lewm_runtime(logger) -> LewmRuntime:
    checkpoint = os.getenv("AIC_LEWM_CHECKPOINT")
    if not checkpoint:
        return LewmRuntime(
            model=None,
            checkpoint=None,
            device=None,
            unavailable_reason="AIC_LEWM_CHECKPOINT is not set",
        )

    _add_vendor_path()

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on runtime image
        return LewmRuntime(
            model=None,
            checkpoint=checkpoint,
            device=None,
            unavailable_reason=f"torch import failed: {exc}",
        )

    device = os.getenv("AIC_LEWM_DEVICE")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = _load_checkpoint(checkpoint=checkpoint, device=device)
        model = model.to(device) if hasattr(model, "to") else model
        model = model.eval() if hasattr(model, "eval") else model
        if hasattr(model, "requires_grad_"):
            model.requires_grad_(False)
        _log(logger, "info", f"Loaded LEWM runtime on {device}: {checkpoint}")
        return LewmRuntime(model=model, checkpoint=checkpoint, device=device)
    except Exception as exc:  # pragma: no cover - depends on external deps/checkpoints
        return LewmRuntime(
            model=None,
            checkpoint=checkpoint,
            device=device,
            unavailable_reason=f"LEWM checkpoint load failed: {exc}",
        )


def _load_checkpoint(checkpoint: str, device: str):
    path = Path(checkpoint).expanduser()
    if path.exists() and path.suffix == ".ckpt":
        import torch

        return torch.load(path, map_location=device, weights_only=False)

    import stable_worldmodel as swm

    cache_dir = os.getenv("AIC_LEWM_CACHE_DIR") or None
    if cache_dir:
        return swm.policy.AutoCostModel(checkpoint, cache_dir=cache_dir)
    return swm.policy.AutoCostModel(checkpoint)


def _add_vendor_path() -> None:
    vendor_dir = Path(__file__).resolve().parent / "lewm_vendor"
    vendor_path = str(vendor_dir)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)


def _log(logger, level: str, message: str) -> None:
    log_fn = getattr(logger, level, None) or getattr(logger, "info", None)
    if log_fn is not None:
        log_fn(message)
