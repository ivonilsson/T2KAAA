"""Transformer2DModel compatibility shim.

IDM-VTON expects ``diffusers.models.transformer_2d`` but some modern wheels
omit that module. This shim first tries to import the upstream implementation
and otherwise falls back to the patched variant that already lives in the
IDM-VTON source tree.
"""

from __future__ import annotations

try:  # pragma: no cover
    from diffusers.models.transformer_2d import Transformer2DModel, Transformer2DModelOutput  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback path
    from src.transformerhacked_tryon import (  # type: ignore
        Transformer2DModel,
        Transformer2DModelOutput,
    )

__all__ = ["Transformer2DModel", "Transformer2DModelOutput"]
