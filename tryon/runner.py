"""Lightweight IDM-VTON runner without starting the Gradio demo server."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal, Optional, Tuple

import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

IDM_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "IDM-VTON"
GRADIO_DEMO_ROOT = IDM_ROOT / "gradio_demo"

# Ensure the third_party repo is importable
for extra_path in (IDM_ROOT, GRADIO_DEMO_ROOT):
    path_str = str(extra_path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from diffusers import DDPMScheduler, AutoencoderKL  # type: ignore  # pylint: disable=wrong-import-position
from transformers import (  # type: ignore  # pylint: disable=wrong-import-position
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModelRef  # type: ignore  # noqa: E402
from src.unet_hacked_tryon import UNet2DConditionModel  # type: ignore  # noqa: E402
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline  # type: ignore  # noqa: E402
from utils_mask import get_mask_location  # type: ignore  # noqa: E402
from preprocess.humanparsing.run_parsing import Parsing  # type: ignore  # noqa: E402
from preprocess.openpose.run_openpose import OpenPose  # type: ignore  # noqa: E402
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation  # type: ignore  # noqa: E402
import apply_net  # type: ignore  # noqa: E402


def _pil_to_binary_mask(pil_image: Image.Image, threshold: int = 0) -> Image.Image:
    np_image = np.array(pil_image.convert("L"))
    mask = (np_image > threshold).astype(np.uint8) * 255
    return Image.fromarray(mask)


class IDMVTONTryOn:
    """Utility that mirrors the original Gradio demo pipeline without launching Gradio."""

    def __init__(
        self,
        base_repo: str = "yisol/IDM-VTON",
        device: Optional[str] = None,
        precision: Literal["auto", "fp16", "fp32"] = "auto",
        guidance_scale: float = 2.0,
        enable_sequential_cpu_offload: bool = False,
        enable_vae_slicing: bool = True,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.guidance_scale = guidance_scale
        self.base_repo = base_repo
        self.precision = precision
        self.enable_sequential_cpu_offload = enable_sequential_cpu_offload
        self.enable_vae_slicing = enable_vae_slicing

        if self.device.type != "cuda" and enable_sequential_cpu_offload:
            raise ValueError("Sequential CPU offload only applies to CUDA devices.")

        if self.device.type != "cuda":
            self.dtype = torch.float32
        elif precision == "fp32":
            self.dtype = torch.float32
        else:
            # default / auto => fp16 on GPU
            self.dtype = torch.float16

        self.tensor_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

        self._hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        self._pipe = None
        self._parsing: Optional[Parsing] = None
        self._openpose: Optional[OpenPose] = None

    @property
    def pipe(self) -> TryonPipeline:
        if self._pipe is None:
            self._pipe = self._load_pipeline()
        return self._pipe

    @property
    def parsing_model(self) -> Parsing:
        if self._parsing is None:
            if self.device.type != "cuda":
                raise RuntimeError("Auto-masking currently requires CUDA for parsing model.")
            self._parsing = Parsing(self.device.index or 0)
        return self._parsing

    @property
    def openpose_model(self) -> OpenPose:
        if self._openpose is None:
            if self.device.type != "cuda":
                raise RuntimeError("OpenPose currently requires CUDA.")
            self._openpose = OpenPose(self.device.index or 0)
        return self._openpose

    def _resolve_path(self, *relative_paths: str | Path) -> Path:
        for rel in relative_paths:
            rel_path = Path(rel)
            candidate = (GRADIO_DEMO_ROOT / rel_path).resolve()
            if candidate.exists():
                return candidate
        for rel in relative_paths:
            rel_path = Path(rel)
            candidate = (IDM_ROOT / rel_path).resolve()
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not locate any of the following paths: "
            + ", ".join(str((GRADIO_DEMO_ROOT / Path(rel)).resolve()) for rel in relative_paths)
        )

    def _load_pipeline(self) -> TryonPipeline:
        model_kwargs = {"torch_dtype": self.dtype}
        if self._hf_token:
            model_kwargs["token"] = self._hf_token

        unet = UNet2DConditionModel.from_pretrained(
            self.base_repo,
            subfolder="unet",
            **model_kwargs,
        )
        noise_scheduler = DDPMScheduler.from_pretrained(self.base_repo, subfolder="scheduler")
        tokenizer_one = AutoTokenizer.from_pretrained(self.base_repo, subfolder="tokenizer", use_fast=False)
        tokenizer_two = AutoTokenizer.from_pretrained(self.base_repo, subfolder="tokenizer_2", use_fast=False)
        text_encoder_one = CLIPTextModel.from_pretrained(
            self.base_repo,
            subfolder="text_encoder",
            **model_kwargs,
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            self.base_repo,
            subfolder="text_encoder_2",
            **model_kwargs,
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.base_repo,
            subfolder="image_encoder",
            **model_kwargs,
        )
        vae = AutoencoderKL.from_pretrained(
            self.base_repo,
            subfolder="vae",
            **model_kwargs,
        )
        unet_encoder = UNet2DConditionModelRef.from_pretrained(
            self.base_repo,
            subfolder="unet_encoder",
            **model_kwargs,
        )

        pipe = TryonPipeline.from_pretrained(
            self.base_repo,
            unet=unet,
            vae=vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            scheduler=noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=self.dtype,
        )
        pipe.unet_encoder = unet_encoder
        pipe.set_progress_bar_config(disable=True)

        if self.enable_sequential_cpu_offload and self.device.type == "cuda":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(self.device)

        if self.enable_vae_slicing:
            pipe.enable_vae_slicing()

        return pipe

    def _build_prompt_embeds(self, pipe: TryonPipeline, description: str) -> Tuple[torch.Tensor, ...]:
        prompt = f"model is wearing {description}"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        cloth_prompt = f"a photo of {description}"
        (
            cloth_prompt_embeds,
            _,
            _,
            _,
        ) = pipe.encode_prompt(
            cloth_prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=negative_prompt,
        )

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            cloth_prompt_embeds,
        )

    def _compute_mask(
        self,
        human_img: Image.Image,
        mask_image: Optional[Image.Image],
        auto_mask: bool,
        category: Literal["upper_body", "lower_body", "dresses"] = "upper_body",
    ) -> Tuple[Image.Image, Image.Image]:
        if auto_mask:
            resized = human_img.resize((384, 512))
            keypoints = self.openpose_model(resized)
            model_parse, _ = self.parsing_model(resized)
            mask, mask_gray = get_mask_location("hd", category, model_parse, keypoints)
            mask = mask.resize((768, 1024))
        else:
            if mask_image is None:
                raise ValueError("Manual mask requested but no mask image provided.")
            mask = _pil_to_binary_mask(mask_image).resize((768, 1024))
            mask_gray = mask.copy()
        mask_gray_tensor = (1 - transforms.ToTensor()(mask)) * self.tensor_transform(human_img.resize((768, 1024)))
        mask_gray = to_pil_image((mask_gray_tensor + 1.0) / 2.0)
        return mask, mask_gray

    def _densepose(self, human_img: Image.Image) -> Image.Image:
        resized = human_img.resize((384, 512))
        human_img_arg = _apply_exif_orientation(resized)
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

        detectron_cfg = self._resolve_path(
            Path("configs/densepose_rcnn_R_50_FPN_s1x.yaml"),
            Path("preprocess/humanparsing/mhp_extension/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"),
        )
        detectron_ckpt = self._resolve_path(Path("ckpt/densepose/model_final_162be9.pkl"))
        args = apply_net.create_argument_parser().parse_args(
            (
                "show",
                str(detectron_cfg),
                str(detectron_ckpt),
                "dp_segm",
                "-v",
                "--opts",
                "MODEL.DEVICE",
                    "cuda" if self.device.type == "cuda" else "cpu",
            )
        )
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        return Image.fromarray(pose_img).resize((768, 1024))

    def run_pair(
        self,
        person_image_path: str,
        garment_image_path: str,
        garment_description: str,
        denoise_steps: int = 30,
        seed: int = 42,
        auto_mask: bool = True,
        auto_crop: bool = False,
        manual_mask_path: Optional[str] = None,
    ) -> Tuple[Image.Image, Image.Image]:
        person_img_original = Image.open(person_image_path).convert("RGB")
        garment_img = Image.open(garment_image_path).convert("RGB").resize((768, 1024))

        if auto_crop:
            width, height = person_img_original.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = int((width - target_width) / 2)
            top = int((height - target_height) / 2)
            right = left + target_width
            bottom = top + target_height
            cropped = person_img_original.crop((left, top, right, bottom))
            crop_size = cropped.size
            human_img = cropped.resize((768, 1024))
        else:
            human_img = person_img_original.resize((768, 1024))
            crop_size = None
            left = top = 0

        manual_mask_img = None
        if manual_mask_path:
            manual_mask_img = Image.open(manual_mask_path)

        mask, mask_gray = self._compute_mask(human_img, manual_mask_img, auto_mask)
        pose_img = self._densepose(human_img)

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            cloth_prompt_embeds,
        ) = self._build_prompt_embeds(self.pipe, garment_description)

        pose_tensor = self.tensor_transform(pose_img).unsqueeze(0).to(self.device, self.dtype)
        garment_tensor = self.tensor_transform(garment_img).unsqueeze(0).to(self.device, self.dtype)

        generator = None
        if seed is not None and seed >= 0:
            generator = torch.Generator(self.device).manual_seed(seed)

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda", dtype=self.dtype):
                images = self.pipe(
                    prompt_embeds=prompt_embeds.to(self.device, self.dtype),
                    negative_prompt_embeds=negative_prompt_embeds.to(self.device, self.dtype),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, self.dtype),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, self.dtype),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_tensor,
                    text_embeds_cloth=cloth_prompt_embeds.to(self.device, self.dtype),
                    cloth=garment_tensor,
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garment_img,
                    guidance_scale=self.guidance_scale,
                )[0]

        result_img = images[0]
        if auto_crop and crop_size is not None:
            pasted = result_img.resize(crop_size)
            person_img_original = person_img_original.copy()
            person_img_original.paste(pasted, (left, top))
            result_img = person_img_original

        return result_img, mask_gray


@lru_cache(maxsize=1)
def get_default_runner() -> IDMVTONTryOn:
    return IDMVTONTryOn()
