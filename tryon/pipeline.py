"""Compatibility helpers for IDM-VTON try-on."""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

from PIL import Image

from .runner import IDMVTONTryOn


@lru_cache(maxsize=1)
def _default_runner() -> IDMVTONTryOn:
    return IDMVTONTryOn()


def run_idm_vton(
    person_image_path: str,
    garment_image_path: str,
    garment_description: str,
    denoise_steps: int = 30,
    seed: int = 42,
    use_auto_mask: bool = True,
    use_auto_crop: bool = False,
    manual_mask_path: str | None = None,
) -> Tuple[Image.Image, Image.Image]:
    """Run IDM-VTON on a single pair using the lightweight runner."""

    runner = _default_runner()
    return runner.run_pair(
        person_image_path=person_image_path,
        garment_image_path=garment_image_path,
        garment_description=garment_description,
        denoise_steps=denoise_steps,
        seed=seed,
        auto_mask=use_auto_mask,
        auto_crop=use_auto_crop,
        manual_mask_path=manual_mask_path,
    )
