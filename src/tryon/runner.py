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

from PIL import ImageDraw

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),     # right arm
    (1, 5), (5, 6), (6, 7),             # left arm
    (1, 8), (8, 9), (9, 10),            # right leg
    (1, 11), (11, 12), (12, 13),        # left leg
    (0, 14), (14, 16),                  # right face
    (0, 15), (15, 17),                  # left face
]

def draw_openpose_on_human(human_768x1024: Image.Image, keypoints: dict) -> Image.Image:
    pts = keypoints["pose_keypoints_2d"]  # list of 18 [x, y, ...] or [x, y]
    scale_x = 768.0 / 384.0
    scale_y = 1024.0 / 512.0

    coords = []
    for p in pts:
        x, y = float(p[0]), float(p[1])
        x2 = x * scale_x
        y2 = y * scale_y
        coords.append((x2, y2))

    out = human_768x1024.copy()
    draw = ImageDraw.Draw(out)

    # joints
    r = 4
    for x, y in coords:
        if x == 0 and y == 0:
            continue
        draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=2)

    # bones
    for i, j in SKELETON_EDGES:
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
            continue
        draw.line((x1, y1, x2, y2), fill="red", width=2)

    return out


# add these helpers near the top of the file
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_pil(img) -> Image.Image:
    """
    Accepts:
      - PIL.Image.Image
      - numpy.ndarray (H x W or H x W x C, uint8 or float in [0,1])
      - torch.Tensor (C x H x W or 1 x C x H x W, float in [0,1] or [-1,1])
    Returns a PIL.Image.
    """
    if isinstance(img, Image.Image):
        return img

    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            x = np.clip(img, 0.0, 1.0)
            x = (x * 255).round().astype("uint8")
        else:
            x = img
        if x.ndim == 2:
            return Image.fromarray(x)
        if x.ndim == 3:
            return Image.fromarray(x)
        raise ValueError(f"Unsupported numpy shape {x.shape}")

    if torch.is_tensor(img):
        x = img.detach().cpu()
        # collapse batch if present
        if x.ndim == 4:
            x = x[0]
        if x.ndim != 3:
            raise ValueError(f"Unsupported tensor shape {x.shape}")
        # C x H x W, assume [-1,1] or [0,1]
        if x.min() < 0:
            x = (x + 1) / 2
        x = x.clamp(0, 1)
        x = x.permute(1, 2, 0).numpy()
        x = (x * 255).round().astype("uint8")
        return Image.fromarray(x)

    raise TypeError(f"Unsupported image type {type(img)}")


def _save_debug_image(img, path: Path) -> None:
    pil = _to_pil(img)
    _ensure_dir(path.parent)
    pil.save(path)


# Discover repository roots relative to src layout
SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
IDM_ROOT = PROJECT_ROOT / "third_party" / "IDM-VTON"
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
        debug_dir: Optional[str] = None,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Run IDM-VTON on a single person + garment pair.

        Adds:
        - debug_dir: optional directory where intermediate images are saved
        - snapshots of diffusion steps via diffusers callback
        """

        debug_root = Path(debug_dir) if debug_dir is not None else None
        if debug_root is not None:
            _ensure_dir(debug_root)

        # ---------- load and (optionally) crop images ----------

        person_img_original = Image.open(person_image_path).convert("RGB")
        garment_img = Image.open(garment_image_path).convert("RGB")

        # IDM-VTON works at 768x1024 (W x H)
        target_w, target_h = 768, 1024

        if auto_crop:
            # placeholder: keep full image but track crop info
            cropped = person_img_original
            human_img = cropped.resize((target_w, target_h))
            crop_size: Optional[Tuple[int, int]] = cropped.size
            left, top = 0, 0
        else:
            human_img = person_img_original.resize((target_w, target_h))
            crop_size = None
            left, top = 0, 0

        if debug_root is not None:
            _save_debug_image(person_img_original, debug_root / "01_person_input.png")
            _save_debug_image(garment_img, debug_root / "02_garment_input.png")
            _save_debug_image(human_img, debug_root / "03_human_img_768x1024.png")

        # ---------- mask and DensePose ----------

        manual_mask_img = None
        if manual_mask_path is not None:
            manual_mask_img = Image.open(manual_mask_path).convert("L")

        mask, mask_gray = self._compute_mask(human_img, manual_mask_img, auto_mask)
        pose_img = self._densepose(human_img)

        if debug_root is not None:
            _save_debug_image(mask, debug_root / "04_mask_binary.png")
            _save_debug_image(mask_gray, debug_root / "05_mask_gray.png")
            _save_debug_image(pose_img, debug_root / "06_densepose.png")
            keypoints = self.openpose_model(human_img.resize((384, 512)))
            kp_vis = draw_openpose_on_human(human_img, keypoints)
            _save_debug_image(kp_vis, debug_root / "07_openpose_keypoints.png")
            overlay = kp_vis.copy()
            mask_rgba = mask.convert("L").resize((768, 1024))
            overlay.putalpha(255)
            # simple red mask overlay
            red = Image.new("RGBA", overlay.size, (255, 0, 0, 120))
            red.putalpha(mask_rgba.point(lambda v: int(v > 0) * 120))
            overlay = Image.alpha_composite(overlay.convert("RGBA"), red)
            _save_debug_image(overlay, debug_root / "08_openpose_keypoints_and_mask.png")

        # ---------- tensors and text embeddings ----------

        pipe = self.pipe  # lazy-loaded TryonPipeline

        human_tensor = self.tensor_transform(human_img).unsqueeze(0).to(self.device, self.dtype)
        cloth_tensor = self.tensor_transform(garment_img.resize((target_w, target_h))).unsqueeze(0).to(
            self.device, self.dtype
        )
        pose_tensor = self.tensor_transform(pose_img).unsqueeze(0).to(self.device, self.dtype)

        # mask stays as PIL for the pipeline; mask_gray is just for visualization

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            cloth_prompt_embeds,
        ) = self._build_prompt_embeds(pipe, garment_description)

        # ---------- diffusion callback for intermediate steps ----------

        debug_steps = {
            0,
            denoise_steps // 4,
            denoise_steps // 2,
            3 * denoise_steps // 4,
            denoise_steps - 1,
        }

        def _decode_latents(latents: torch.Tensor) -> Iterable[Image.Image]:
            lat = latents.detach().to(self.device)
            scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
            lat = lat / scaling_factor
            with torch.no_grad():
                imgs = pipe.vae.decode(lat).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
            imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
            imgs = (imgs * 255).round().astype("uint8")
            return [Image.fromarray(arr) for arr in imgs]

        def _callback(step: int, timestep: int, latents: torch.FloatTensor) -> None:
            if debug_root is None:
                return
            if step not in debug_steps:
                return
            for i, pil in enumerate(_decode_latents(latents)):
                fname = f"20_denoise_step_{step:03d}_sample_{i}.png"
                _save_debug_image(pil, debug_root / fname)

        # ---------- run pipeline ----------

        generator = torch.Generator(device=self.device).manual_seed(seed)

        pipe_kwargs = dict(
            prompt_embeds=prompt_embeds.to(self.device, self.dtype),
            negative_prompt_embeds=negative_prompt_embeds.to(self.device, self.dtype),
            pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, self.dtype),
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, self.dtype),
            num_inference_steps=denoise_steps,
            generator=generator,
            strength=1.0,
            pose_img=pose_tensor,
            text_embeds_cloth=cloth_prompt_embeds.to(self.device, self.dtype),
            cloth=cloth_tensor,
            mask_image=mask,          # PIL mask
            image=human_img,          # PIL human image
            height=target_h,
            width=target_w,
            ip_adapter_image=garment_img,
            guidance_scale=self.guidance_scale,
        )

        with torch.no_grad():
            autocast_on = self.device.type == "cuda"
            with torch.autocast(
                device_type=self.device.type, enabled=autocast_on, dtype=self.dtype
            ):
                if debug_root is not None:
                    try:
                        result = pipe(callback=_callback, callback_steps=1, **pipe_kwargs)
                    except TypeError:
                        # pipeline does not support callback (fallback)
                        result = pipe(**pipe_kwargs)
                else:
                    result = pipe(**pipe_kwargs)

        # diffusers-style output
        if hasattr(result, "images"):
            images = result.images
        else:
            images = result[0]

        result_img = images[0]

        # ---------- undo crop (if any) and save final images ----------

        if auto_crop and crop_size is not None:
            pasted = result_img.resize(crop_size)
            person_img_original = person_img_original.copy()
            person_img_original.paste(pasted, (left, top))
            result_img = person_img_original

        if debug_root is not None:
            _save_debug_image(result_img, debug_root / "30_final_output.png")
            _save_debug_image(mask_gray, debug_root / "31_final_mask_gray.png")

        return result_img, mask_gray



@lru_cache(maxsize=1)
def get_default_runner() -> IDMVTONTryOn:
    return IDMVTONTryOn()
