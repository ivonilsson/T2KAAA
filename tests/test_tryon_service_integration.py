"""Integration test that exercises IDM-VTON with real weights on GPU.

This test is intentionally slow and requires GPU + access to the pretrained
weights (yisol/IDM-VTON). It uses the sample person and garment images from the
repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (config.GRADIO_DEMO_DIR, config.IDM_VTON_ROOT, config.PROJECT_ROOT):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

from services import tryon_service


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for IDM-VTON integration test")
def test_tryon_with_weights():
    person_img = Image.open(PROJECT_ROOT / "tests" / "test_images" / "person.jpeg").convert("RGB")
    garment_path = PROJECT_ROOT / "tests" / "test_images" / "garment.jpg"

    selected_choices = ["#1: garment.jpg"]
    catalog_metadata = [
        {"choice": selected_choices[0], "path": str(garment_path), "label": "Integration Garment"}
    ]
    selection_details = {
        selected_choices[0]: {
            "description": "integration garment",
            "path": str(garment_path),
            "label": "Integration Garment",
        }
    }

    gallery, summary = tryon_service.start_tryon(
        person_img,
        selected_choices,
        catalog_metadata,
        selection_details,
        True,   # run VLM eval to cover full path
        True,   # auto mask
        True,  # auto crop to reduce randomness
        8,      # few steps to keep runtime manageable
        1234,
    )

    assert gallery, "Expected at least one generated look"
    image, caption = gallery[0]
    assert isinstance(image, Image.Image)
    assert image.size[0] > 0 and image.size[1] > 0
    assert isinstance(summary, str)
    assert caption and "Integration Garment" in caption
