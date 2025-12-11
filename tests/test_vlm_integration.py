"""Integration test for VLM helpers with real model weights on GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import services.vlm as vlm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for VLM integration test")
def test_vlm_real_model_end_to_end():
    model, processor = vlm.load_vlm()
    garment_img = Image.open(PROJECT_ROOT / "tests" / "test_images" / "garment.jpg").convert("RGB")
    person_img = Image.open(PROJECT_ROOT / "tests" / "test_images" / "person.jpeg").convert("RGB")

    description = vlm.generate_garment_description(garment_img)
    assert isinstance(description, str)
    assert description.strip()

    eval_result = vlm.generate_tryon_evaluation(person_img, model, processor, garment_description=description)
    assert isinstance(eval_result, dict)
    assert eval_result.get("text")
