"""Try-on runtime service that orchestrates IDM-VTON pipeline and evaluations."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms

import apply_net
import numpy as np
from diffusers import AutoencoderKL, DDPMScheduler
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModelRef
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
from utils_mask import get_mask_location
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

import config
from services.catalog import open_image, prepare_selected_garments
from services.vlm import (
    build_gallery_annotations,
    generate_tryon_evaluation,
    load_vlm,
    resize_for_vlm,
    summarize_vlm_recommendations,
    unload_vlm,
)
import tracking

device = "cuda:0" if torch.cuda.is_available() else "cpu"

base_path = "yisol/IDM-VTON"

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)

tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(
    base_path,
    subfolder="vae",
    torch_dtype=torch.float16,
)

UNet_Encoder = UNet2DConditionModelRef.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)

tensor_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder


def _extract_human_image(editor_state):
    if isinstance(editor_state, Image.Image):
        return editor_state.convert("RGB")
    if isinstance(editor_state, dict):
        for key in ("image", "background", "composite"):
            candidate = editor_state.get(key)
            if isinstance(candidate, Image.Image):
                return candidate.convert("RGB")
        layers = editor_state.get("layers")
        if isinstance(layers, list):
            for layer in layers:
                if isinstance(layer, Image.Image):
                    return layer.convert("RGB")
    raise ValueError("No person image supplied. Please upload a photo.")


def _extract_manual_mask(editor_state):
    if not isinstance(editor_state, dict):
        return None
    layers = editor_state.get("layers")
    if isinstance(layers, list):
        for layer in layers:
            if isinstance(layer, Image.Image):
                return layer.convert("L")
    mask = editor_state.get("mask")
    if isinstance(mask, Image.Image):
        return mask.convert("L")
    return None


def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j] is True:
                mask[i, j] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


def _prepare_human_context(editor_state, is_checked, is_checked_crop):
    human_img_orig = _extract_human_image(editor_state)
    crop_meta = None
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_meta = {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "size": cropped_img.size,
        }
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    manual_mask_img = _extract_manual_mask(editor_state)
    use_auto_mask = is_checked or manual_mask_img is None

    if use_auto_mask:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, _ = get_mask_location("hd", "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(manual_mask_img.resize((768, 1024)))
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(
        (
            "show",
            str(config.DENSEPOSE_CFG),
            str(config.DENSEPOSE_CKPT),
            "dp_segm",
            "-v",
            "--opts",
            "MODEL.DEVICE",
            "cuda",
        )
    )
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))
    pose_tensor = tensor_transfrom(pose_img).unsqueeze(0)

    return {
        "human_img_orig": human_img_orig,
        "human_img": human_img,
        "mask": mask,
        "pose_img": pose_img,
        "pose_tensor": pose_tensor,
        "crop_meta": crop_meta,
    }


def _compose_output(human_ctx, generated_image: Image.Image) -> Image.Image:
    crop_meta = human_ctx.get("crop_meta")
    if not crop_meta:
        return generated_image
    out_img = generated_image.resize(crop_meta["size"])
    canvas = human_ctx["human_img_orig"].copy()
    canvas.paste(out_img, (int(crop_meta["left"]), int(crop_meta["top"])))
    return canvas


def _shrink_for_logging(image: Image.Image | None, max_side: int = 512) -> Image.Image | None:
    if image is None:
        return None
    preview = image.copy()
    preview = preview.convert("RGB")
    preview.thumbnail((max_side, max_side))
    return preview


def _add_image_artifact(store: dict, key: str, image: Image.Image | None) -> None:
    preview = _shrink_for_logging(image)
    if preview is not None:
        store[key] = preview


def _run_tryon_for_garment(human_ctx, garment_image: Image.Image, garment_des: str, denoise_steps: int, seed_value: int | None):
    garm_img = garment_image.convert("RGB").resize((768, 1024))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prompt = "model is wearing " + garment_des
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

            prompt = "a photo of " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            if not isinstance(prompt, List):
                prompt = [prompt]
            if not isinstance(negative_prompt, List):
                negative_prompt = [negative_prompt]

            (
                prompt_embeds_c,
                _,
                _,
                _,
            ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt,
            )

            pose_tensor = human_ctx["pose_tensor"].to(device, torch.float16)
            garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, torch.float16)
            generator = None
            if seed_value is not None:
                generator = torch.Generator(device).manual_seed(seed_value)
            images = pipe(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                cloth=garm_tensor,
                mask_image=human_ctx["mask"],
                image=human_ctx["human_img"],
                height=1024,
                width=768,
                ip_adapter_image=garm_img,
                guidance_scale=2.0,
            )[0]

    final_image = _compose_output(human_ctx, images[0])
    return final_image


def _offload_tryon_models_from_gpu():
    if not torch.cuda.is_available():
        return
    try:
        pipe.to("cpu", torch.float32)
        pipe.unet_encoder.to("cpu", torch.float32)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to offload try-on pipe: {exc}")
    try:
        openpose_model.preprocessor.body_estimation.model.to("cpu")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to offload OpenPose: {exc}")
    torch.cuda.empty_cache()


def start_tryon(
    editor_state,
    selected_catalog_choices,
    catalog_metadata,
    selection_details,
    run_vlm_eval,
    is_checked,
    is_checked_crop,
    denoise_steps,
    seed,
):
    run_id = f"tryon_{uuid.uuid4().hex[:8]}"
    run_start = time.perf_counter()
    stage_metrics: dict[str, float] = {}
    generation_latencies: list[float] = []
    garments_info: list[dict] = []
    image_artifacts: dict[str, Image.Image] = {}
    text_artifacts: dict[str, str] = {}
    vlm_evaluations: list[dict] = []
    status = "success"
    error_message = ""
    final_comment = "Enable VLM evaluation to get automated notes."
    annotated_gallery: list = []
    generated_images: list[dict] = []
    result_payload = (annotated_gallery, final_comment)

    if torch.cuda.is_available() and hasattr(torch.cuda, "reset_peak_memory_stats"):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # pylint: disable=broad-except
            pass

    try:
        t0 = time.perf_counter()
        garments = prepare_selected_garments(selected_catalog_choices, catalog_metadata, selection_details)
        stage_metrics["stage_catalog_selection_ms"] = (time.perf_counter() - t0) * 1000

        unload_vlm()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        openpose_model.preprocessor.body_estimation.model.to(device)

        t0 = time.perf_counter()
        human_ctx = _prepare_human_context(editor_state, is_checked, is_checked_crop)
        stage_metrics["stage_human_context_ms"] = (time.perf_counter() - t0) * 1000
        _add_image_artifact(image_artifacts, "inputs/person.png", human_ctx.get("human_img_orig"))
        _add_image_artifact(image_artifacts, "inputs/mask.png", human_ctx.get("mask"))
        _add_image_artifact(image_artifacts, "inputs/pose.png", human_ctx.get("pose_img"))

        target_dtype = torch.float16 if device != "cpu" else torch.float32
        pipe.to(device, target_dtype)
        pipe.unet_encoder.to(device, target_dtype)

        gallery_entries = []
        seed_value = None if seed is None else int(seed)

        for idx, garment in enumerate(garments):
            garment_desc = garment.get("description") or "a garment"
            garment_image = garment.get("image")
            if garment_image is None:
                garment_image = open_image(Path(garment["path"]))
            if garment_image is None:
                print(f"[WARN] Skipping garment {garment['path']} (failed to load)")
                continue
            _add_image_artifact(image_artifacts, f"inputs/garment_{idx}.png", garment_image)
            current_seed = seed_value
            if seed_value is not None:
                current_seed = seed_value + idx
            gen_start = time.perf_counter()
            result_image = _run_tryon_for_garment(
                human_ctx,
                garment_image,
                garment_desc,
                int(denoise_steps),
                current_seed,
            )
            latency_ms = (time.perf_counter() - gen_start) * 1000
            generation_latencies.append(latency_ms)
            stage_metrics[f"stage_generation_{idx}_ms"] = latency_ms

            generated_images.append({"image": result_image, "description": garment_desc})
            caption = f"{garment.get('label', Path(garment['path']).name)}: {garment_desc}"
            gallery_entries.append((result_image, caption))
            _add_image_artifact(image_artifacts, f"outputs/look_{idx}.png", result_image)
            garments_info.append(
                {
                    "index": idx,
                    "label": garment.get("label", Path(garment["path"]).name),
                    "path": garment.get("path"),
                    "description": garment_desc,
                    "seed": current_seed,
                    "latency_ms": latency_ms,
                }
            )

        final_comment = "Enable VLM evaluation to get automated notes."
        annotated_gallery = gallery_entries
        try:
            if run_vlm_eval and generated_images:
                t0 = time.perf_counter()
                _offload_tryon_models_from_gpu()
                model = None
                processor = None
                try:
                    model, processor = load_vlm()
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"[WARN] Failed to load VLM: {exc}")
                evaluations = []
                for generated in generated_images:
                    small_img = resize_for_vlm(generated["image"])
                    evaluations.append(
                        generate_tryon_evaluation(
                            small_img,
                            model,
                            processor,
                            generated.get("description"),
                        )
                    )
                if evaluations:
                    vlm_evaluations = evaluations
                    annotated_gallery = build_gallery_annotations(gallery_entries, evaluations)
                    final_comment = summarize_vlm_recommendations(gallery_entries, evaluations)
                    text_artifacts["vlm_summary.txt"] = final_comment
                    text_artifacts["vlm_evaluations.json"] = json.dumps(evaluations, indent=2)
                    stage_metrics["stage_vlm_eval_ms"] = (time.perf_counter() - t0) * 1000
        finally:
            unload_vlm()

        result_payload = (annotated_gallery, final_comment)
    except Exception as exc:  # pylint: disable=broad-except
        status = "error"
        error_message = str(exc)
        raise
    finally:
        total_latency_ms = (time.perf_counter() - run_start) * 1000
        metrics: dict[str, float] = {k: float(v) for k, v in stage_metrics.items()}
        metrics["total_latency_ms"] = total_latency_ms
        metrics["num_garments_generated"] = float(len(generated_images))
        metrics["num_garments_requested"] = float(len(selected_catalog_choices or []))
        if generation_latencies:
            metrics["generation_avg_ms"] = sum(generation_latencies) / len(generation_latencies)
            metrics["generation_max_ms"] = max(generation_latencies)
        if vlm_evaluations:
            scores = [eval.get("score") for eval in vlm_evaluations if isinstance(eval.get("score"), (int, float))]
            if scores:
                metrics["vlm_score_avg"] = sum(scores) / len(scores)
                metrics["vlm_score_best"] = max(scores)
            metrics["vlm_eval_count"] = float(len(vlm_evaluations))
            metrics["vlm_eval_failures"] = float(sum(1 for eval in vlm_evaluations if eval.get("score") is None))
        gpu_name = None
        if torch.cuda.is_available():
            try:
                metrics["gpu_max_mem_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
            except Exception:  # pylint: disable=broad-except
                pass
            try:
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                metrics["gpu_total_mem_gb"] = props.total_memory / (1024 ** 3)
                gpu_name = props.name
            except Exception:  # pylint: disable=broad-except
                pass
        try:
            gpu_cost_per_hour = float(os.environ.get("T2KAAA_GPU_COST_PER_HOUR", "2.0"))
            metrics["estimated_cost_usd"] = (total_latency_ms / 1000 / 3600) * gpu_cost_per_hour
        except Exception:  # pylint: disable=broad-except
            pass
        text_artifacts["garments.json"] = json.dumps(garments_info, indent=2)
        if not text_artifacts.get("vlm_summary.txt") and final_comment:
            text_artifacts["vlm_summary.txt"] = final_comment
        if error_message and status != "success":
            text_artifacts["error.txt"] = error_message
        params: dict[str, str] = {
            "auto_mask": str(bool(is_checked)),
            "auto_crop": str(bool(is_checked_crop)),
            "denoise_steps": str(int(denoise_steps)),
            "seed": str(seed) if seed is not None else "random",
            "run_vlm_eval": str(bool(run_vlm_eval)),
            "device": device,
            "guidance_scale": "2.0",
            "num_garments_requested": str(len(selected_catalog_choices or [])),
            "num_garments_generated": str(len(generated_images)),
        }
        if gpu_name:
            params["gpu_name"] = gpu_name
        if garments_info:
            params["garment_labels"] = ",".join(info.get("label", "unknown") for info in garments_info)
        clean_images = {k: v for k, v in image_artifacts.items() if v is not None}
        tags = {
            "component": "tryon",
            "stage": "run",
            "status": status,
            "vlm_enabled": str(bool(run_vlm_eval)).lower(),
            "run_id": run_id,
        }
        tracking.log_event(
            run_name=run_id,
            params=params,
            metrics=metrics,
            tags=tags,
            text_artifacts=text_artifacts,
            image_artifacts=clean_images,
        )

    return result_payload

    return annotated_gallery, final_comment
