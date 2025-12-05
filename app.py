import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR
_IDM_VTON_ROOT = _PROJECT_ROOT / "third_party" / "IDM-VTON"
_GRADIO_DEMO_DIR = _IDM_VTON_ROOT / "gradio_demo"

_DENSEPOSE_CFG = _IDM_VTON_ROOT / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml"
_DENSEPOSE_CKPT = _IDM_VTON_ROOT / "ckpt" / "densepose" / "model_final_162be9.pkl"

for path in (_GRADIO_DEMO_DIR, _IDM_VTON_ROOT, _PROJECT_ROOT):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List

import torch
import os
import gc
import random
import re
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
try:  # transformers>=4.47 adds native Qwen2.5-VL class
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover - fallback for older wheels
    Qwen2_5_VLForConditionalGeneration = None
try:  # transformers>=4.47 exposes Llava class
    from transformers import LlavaForConditionalGeneration
except ImportError:  # pragma: no cover
    LlavaForConditionalGeneration = None
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

_DEFAULT_VLM_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
VLM_MODEL_ID = os.environ.get("T2KAAA_VLM_MODEL_ID", _DEFAULT_VLM_MODEL_ID)
_vlm_model = None
_vlm_processor = None

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
_VLM_MAX_NEW_TOKENS = 48
_VLM_SCORE_PATTERN = re.compile(r"score\s*[:|-]?\s*(\d{1,2})", re.IGNORECASE)
_CATALOG_ROOT = _PROJECT_ROOT / "data"
_MAX_SELECTED_GARMENTS = 3
_DEFAULT_SAMPLE_COUNT = 10


def _list_catalog_categories() -> List[str]:
    if not _CATALOG_ROOT.exists():
        return []
    return sorted(path.name for path in _CATALOG_ROOT.iterdir() if path.is_dir())


def _find_category_images(category: str) -> List[Path]:
    if not category:
        return []
    category_dir = _CATALOG_ROOT / category
    if not category_dir.exists():
        return []
    return [
        path
        for path in category_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    ]


def _sample_catalog_images(category: str, sample_count: int) -> List[Path]:
    paths = _find_category_images(category)
    if not paths:
        return []
    sample_count = max(1, min(sample_count, len(paths)))
    if len(paths) <= sample_count:
        return random.sample(paths, len(paths))
    return random.sample(paths, sample_count)


def _vlm_dtype():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _load_vlm():
    global _vlm_model, _vlm_processor
    if _vlm_model is None or _vlm_processor is None:
        dtype = _vlm_dtype()
        if "llava" in VLM_MODEL_ID.lower():  # llava kernels expect fp16 on many GPUs
            if torch.cuda.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32
        processor_kwargs = {"trust_remote_code": True}
        processor_cls = AutoProcessor
        if "llava" in VLM_MODEL_ID.lower():
            processor_kwargs["use_fast"] = False
        _vlm_processor = processor_cls.from_pretrained(
            VLM_MODEL_ID,
            **processor_kwargs,
        )
        model_cls = AutoModelForCausalLM
        vlm_id_lower = VLM_MODEL_ID.lower()
        if "llava" in vlm_id_lower and LlavaForConditionalGeneration is not None:
            model_cls = LlavaForConditionalGeneration
        elif "qwen2.5" in vlm_id_lower and Qwen2_5_VLForConditionalGeneration is not None:
            model_cls = Qwen2_5_VLForConditionalGeneration
        _vlm_model = model_cls.from_pretrained(
            VLM_MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            _vlm_model = _vlm_model.to(dtype=dtype)
        _vlm_model.eval()
    return _vlm_model, _vlm_processor


def _unload_vlm():
    global _vlm_model, _vlm_processor
    if _vlm_model is not None:
        del _vlm_model
        _vlm_model = None
    if _vlm_processor is not None:
        del _vlm_processor
        _vlm_processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _clean_vlm_response(text: str) -> str:
    if not text:
        return "a garment"
    cleaned = text.strip()
    if "Assistant:" in cleaned:
        cleaned = cleaned.split("Assistant:")[-1].strip()
    if cleaned.lower().startswith("assistant"):
        cleaned = cleaned.split(":", 1)[-1].strip()
    if not cleaned:
        return "a garment"
    return cleaned.replace("\n", " ").strip()


def generate_garment_description(image: Image.Image | None) -> str:
    if image is None:
        return "a garment"
    try:
        model, processor = _load_vlm()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to load VLM: {exc}")
        return "a garment"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Describe the garment in this image for a virtual try-on system. "
                        "Mention color, garment type, sleeve/fit, and distinguishing details in ~15 words."
                    ),
                },
            ],
        }
    ]

    try:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )
        processed_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if value.dtype in (torch.float16, torch.float32, torch.bfloat16):
                    processed_inputs[key] = value.to(model.device, dtype=model.dtype)
                else:
                    processed_inputs[key] = value.to(model.device)
            else:
                processed_inputs[key] = value
        inputs = processed_inputs
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=_VLM_MAX_NEW_TOKENS,
                temperature=0.2,
                top_p=0.85,
                do_sample=True,
            )
        prompt_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        new_tokens = generated_ids[:, prompt_length:]
        decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)
        return _clean_vlm_response(decoded[0] if decoded else "")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to generate garment description: {exc}")
        return "a garment"

def resize_for_vlm(image: Image.Image, size=(224, 224)) -> Image.Image:
    return image.resize(size).convert("RGB")


def generate_tryon_evaluation(output_image, model, processor) -> dict:
    if output_image is None:
        return {"text": "No output image to evaluate.", "score": None}
    if model is None or processor is None:
        return {"text": "Evaluation unavailable.", "score": None}

    instruction = (
        "You are a professional fashion stylist and visual QA. "
        "Look at the try-on result image and give a short, constructive evaluation "
        "about: 1) fit; 2) color harmony; 3) realism/blending; 4) one suggestion. "
        "Respond in the format: 'Score: <1-10>/10 | Comment: <30-50 word critique>'."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": output_image},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    try:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = processor(
            text=[prompt],
            images=[output_image],
            return_tensors="pt",
        )
        processed_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if value.dtype in (torch.float16, torch.float32, torch.bfloat16):
                    processed_inputs[key] = value.to(model.device, dtype=model.dtype)
                else:
                    processed_inputs[key] = value.to(model.device)
            else:
                processed_inputs[key] = value
        inputs = processed_inputs

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=_VLM_MAX_NEW_TOKENS,
                temperature=0.2,
                top_p=0.85,
                do_sample=True,
            )

        prompt_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        new_tokens = out[:, prompt_len:]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True)

        cleaned = _clean_vlm_response(text[0])
        score, comment = _extract_vlm_score(cleaned)
        if not comment:
            comment = cleaned
        return {"text": comment, "score": score}

    except Exception as exc:
        print(f"[WARN] Try-on eval failed: {exc}")
        return {"text": "Evaluation failed.", "score": None}


def _extract_vlm_score(text: str) -> tuple[int | None, str]:
    match = _VLM_SCORE_PATTERN.search(text)
    if not match:
        return None, text
    try:
        score = int(match.group(1))
    except ValueError:
        return None, text
    score = max(1, min(score, 10))
    cleaned = (text[:match.start()] + text[match.end():]).strip(" |-:")
    return score, cleaned


def _build_gallery_annotations(gallery_entries, evaluations):
    annotated = []
    for (image, caption), evaluation in zip(gallery_entries, evaluations):
        parts = [caption]
        score = evaluation.get("score")
        text = evaluation.get("text")
        if score is not None:
            parts.append(f"Score: {score}/10")
        if text:
            parts.append(f"VLM: {text}")
        annotated.append((image, "\n\n".join(parts)))
    return annotated


def _summarize_vlm_recommendations(gallery_entries, evaluations):
    best_score = None
    best_indices: List[int] = []
    for idx, evaluation in enumerate(evaluations):
        score = evaluation.get("score")
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_indices = [idx]
        elif score == best_score:
            best_indices.append(idx)

    summary_lines = []
    for idx, ((_, caption), evaluation) in enumerate(zip(gallery_entries, evaluations), start=1):
        score_text = f"Score {evaluation['score']}/10" if evaluation.get("score") is not None else "Score N/A"
        summary_lines.append(
            f"{idx}. {caption} — {score_text}. {evaluation.get('text', 'No comment')}"
        )

    if best_score is None:
        heading = "VLM feedback (no numeric scores provided)."
    else:
        winners = ", ".join(str(i + 1) for i in best_indices)
        heading = f"Top choice(s): look {winners} with score {best_score}/10."

    return heading + "\n\n" + "\n".join(summary_lines)


def _offload_tryon_models_from_gpu():
    if not torch.cuda.is_available():
        return
    try:
        pipe.to("cpu")
        pipe.unet_encoder.to("cpu")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to offload try-on pipe: {exc}")
    try:
        openpose_model.preprocessor.body_estimation.model.to("cpu")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to offload OpenPose: {exc}")
    torch.cuda.empty_cache()

def _open_image(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to load garment image {path}: {exc}")
    return None


def _describe_garments(garments: List[dict]) -> List[dict]:
    described: List[dict] = []
    for garment in garments:
        desc = generate_garment_description(garment["image"])
        described.append({
            "path": str(garment["path"]),
            "label": garment.get("label") or Path(garment["path"]).name,
            "description": desc,
        })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    _unload_vlm()
    return described


def _summarize_descriptions(items: List[dict]) -> str:
    if not items:
        return "No garments processed."
    lines = []
    for idx, item in enumerate(items, start=1):
        label = item.get("label", f"Garment {idx}")
        desc = item.get("description", "a garment")
        lines.append(f"{idx}. {label}: {desc}")
    return "\n".join(lines)

def _format_choice_label(path: Path, idx: int) -> str:
    return f"#{idx}: {path.name}"


def load_catalog_samples(category: str, sample_count):
    if not category:
        raise gr.Error("Select a garment category to load samples.")
    try:
        sample_count_int = int(sample_count)
    except (TypeError, ValueError):
        sample_count_int = _DEFAULT_SAMPLE_COUNT
    sample_paths = _sample_catalog_images(category, sample_count_int)
    if not sample_paths:
        raise gr.Error(f"No garments found under '{category}'.")
    gallery_entries = []
    metadata = []
    checkbox_choices = []
    for idx, path in enumerate(sample_paths, start=1):
        image = _open_image(path)
        if image is None:
            continue
        label = _format_choice_label(path, idx)
        gallery_entries.append((image, label))
        metadata.append({"choice": label, "path": str(path), "label": label})
        checkbox_choices.append(label)
    if not gallery_entries:
        raise gr.Error("Failed to load catalog images. Please try again.")
    summary = (
        f"Loaded {len(gallery_entries)} looks from '{category}'. "
        f"Select up to {_MAX_SELECTED_GARMENTS} garments below."
    )
    selection_reset = gr.update(choices=checkbox_choices, value=[])
    return gallery_entries, metadata, selection_reset, summary


def update_catalog_selection(selected_choices, sample_metadata):
    sample_metadata = sample_metadata or []
    selected_choices = selected_choices or []
    limited = selected_choices[:_MAX_SELECTED_GARMENTS]
    meta_map = {item["choice"]: item for item in sample_metadata}
    summary_lines = []
    for choice in limited:
        meta = meta_map.get(choice)
        if meta:
            summary_lines.append(f"- {meta['label']}")
    if not limited:
        summary = "No garments selected."
    else:
        summary = "Selected garments:\n" + "\n".join(summary_lines)
        if len(selected_choices) > _MAX_SELECTED_GARMENTS:
            summary += f"\n(Only the first {_MAX_SELECTED_GARMENTS} selections are used.)"
    return gr.update(value=limited), summary


def _prepare_selected_garments(selected_choices, catalog_metadata):
    selected_choices = selected_choices or []
    if not selected_choices:
        raise gr.Error("Select up to three catalog garments before running try-on.")
    catalog_metadata = catalog_metadata or []
    meta_map = {item["choice"]: item for item in catalog_metadata}
    garments = []
    for choice in selected_choices:
        meta = meta_map.get(choice)
        if meta is None:
            continue
        image = _open_image(Path(meta["path"]))
        if image is None:
            print(f"[WARN] Catalog garment missing: {meta['path']}")
            continue
        garments.append({"path": meta["path"], "label": meta["label"], "image": image})
    if not garments:
        raise gr.Error("Unable to load the selected catalog garments.")
    return garments


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
        mask, _ = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(manual_mask_img.resize((768, 1024)))
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(
        (
            'show',
            str(_DENSEPOSE_CFG),
            str(_DENSEPOSE_CKPT),
            'dp_segm',
            '-v',
            '--opts',
            'MODEL.DEVICE',
            'cuda',
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
        "mask_gray": mask_gray,
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


base_path = 'yisol/IDM-VTON'
example_path = _GRADIO_DEMO_DIR / 'example'

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

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
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


def start_tryon(editor_state, selected_catalog_choices, catalog_metadata, run_vlm_eval, is_checked, is_checked_crop, denoise_steps, seed):
    garments = _prepare_selected_garments(selected_catalog_choices, catalog_metadata)
    described_garments = _describe_garments(garments)

    openpose_model.preprocessor.body_estimation.model.to(device)
    human_ctx = _prepare_human_context(editor_state, is_checked, is_checked_crop)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    gallery_entries = []
    seed_value = None if seed is None else int(seed)
    generated_images = []

    for idx, garment in enumerate(described_garments):
        garment_desc = garment.get("description") or "a garment"
        garment_image = _open_image(Path(garment["path"]))
        if garment_image is None:
            print(f"[WARN] Skipping garment {garment['path']} (failed to load)")
            continue
        current_seed = seed_value
        if seed_value is not None:
            current_seed = seed_value + idx
        result_image = _run_tryon_for_garment(
            human_ctx,
            garment_image,
            garment_desc,
            int(denoise_steps),
            current_seed,
        )

        generated_images.append(result_image)
        caption = f"{garment.get('label', Path(garment['path']).name)}: {garment_desc}"
        gallery_entries.append((result_image, caption))

    final_comment = "Enable VLM evaluation to get automated notes."
    annotated_gallery = gallery_entries
    if run_vlm_eval and generated_images:
        _offload_tryon_models_from_gpu()
        model = None
        processor = None
        try:
            model, processor = _load_vlm()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Failed to load VLM: {exc}")
        evaluations = []
        for img in generated_images:
            small_img = resize_for_vlm(img)
            evaluations.append(generate_tryon_evaluation(small_img, model, processor))
        if evaluations:
            annotated_gallery = _build_gallery_annotations(gallery_entries, evaluations)
            final_comment = _summarize_vlm_recommendations(gallery_entries, evaluations)
        _unload_vlm()

    return annotated_gallery, human_ctx["mask_gray"], final_comment


def _list_example_paths(folder: Path):
    if not folder.exists():
        return []
    return sorted(str(path) for path in folder.iterdir() if path.is_file())


human_list_path = _list_example_paths(example_path / "human")

human_ex_list = []
for ex_human in human_list_path:
    ex_dict = {}
    ex_dict['background'] = ex_human
    ex_dict['layers'] = None
    ex_dict['composite'] = None
    human_ex_list.append(ex_dict)

image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    #gr.Markdown("## IDM-VTON 👕👔👚")
    #gr.Markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(
                sources='upload',
                type="pil",
                label='Human (auto mask enabled by default). Draw only if you want a custom mask.',
                interactive=True,
            )
            with gr.Row():
                is_checked = gr.Checkbox(label="Use auto-generated mask", info="Recommended", value=True)
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing", value=False)

            #gr.Examples(
            #    inputs=imgs,
            #    examples_per_page=10,
            #    examples=human_ex_list
            #)

        with gr.Column():
            catalog_categories = _list_catalog_categories()
            catalog_dropdown = gr.Dropdown(
                label="Catalog category",
                choices=catalog_categories,
                value=catalog_categories[0] if catalog_categories else None,
                interactive=True,
            )
            sample_count = gr.Slider(
                label="Sample size",
                minimum=4,
                maximum=12,
                step=1,
                value=_DEFAULT_SAMPLE_COUNT,
            )
            load_samples_btn = gr.Button("Load catalog samples")
            catalog_gallery = gr.Gallery(
                label="Catalog preview",
                columns=5,
                height=400,
                allow_preview=True,
            )
            selection_box = gr.CheckboxGroup(
                label="Select up to three garments",
                choices=[],
                value=[],
            )
            selection_summary = gr.Markdown("No garments selected.")
            catalog_samples_state = gr.State([])
        with gr.Column():
            masked_img = gr.Image(label="Masked image output", elem_id="masked-img")
        with gr.Column():
            run_vlm_eval = gr.Checkbox(
                label="Run AI stylist evaluation (slower)",
                info="Disable to skip the VLM judging step.",
                value=True,
            )
            vlm_comment_box = gr.Textbox(
                label="VLM Try-on Evaluation",
                interactive=False,
                lines=12,
                max_lines=20,
            )
            image_out = gr.Gallery(label="Try-on results", elem_id="output-img", columns=2, height=600)

    with gr.Column():
        try_button = gr.Button(value="Try-on")
        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=30, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)

    try_button.click(
        fn=start_tryon,
        inputs=[imgs, selection_box, catalog_samples_state, run_vlm_eval, is_checked, is_checked_crop, denoise_steps, seed],
        outputs=[image_out, masked_img, vlm_comment_box],
        api_name='tryon'
    )

    load_samples_btn.click(
        fn=load_catalog_samples,
        inputs=[catalog_dropdown, sample_count],
        outputs=[catalog_gallery, catalog_samples_state, selection_box, selection_summary],
        queue=True,
    )

    selection_box.change(
        fn=update_catalog_selection,
        inputs=[selection_box, catalog_samples_state],
        outputs=[selection_box, selection_summary],
        queue=False,
    )

image_blocks.launch(server_name="0.0.0.0", server_port=9090, share=True)
