"""Visual-language model helpers for garment descriptions and stylist evaluations."""

from __future__ import annotations

import gc
import os
import re
import time
from typing import List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

import tracking

try:  # transformers>=4.47 adds native Qwen2.5-VL class
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover - fallback for older wheels
    Qwen2_5_VLForConditionalGeneration = None

try:  # transformers>=4.47 exposes Llava class
    from transformers import LlavaForConditionalGeneration
except ImportError:  # pragma: no cover
    LlavaForConditionalGeneration = None

_DEFAULT_VLM_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
VLM_MODEL_ID = os.environ.get("T2KAAA_VLM_MODEL_ID", _DEFAULT_VLM_MODEL_ID)

_vlm_model = None
_vlm_processor = None

_VLM_MAX_NEW_TOKENS = 128
_VLM_SCORE_PATTERN = re.compile(r"score\s*[:|-]?\s*(\d{1,2})", re.IGNORECASE)


def _ensure_complete_sentence(text: str) -> str:
    if not text:
        return text
    matches = list(re.finditer(r"[.!?]", text))
    if matches:
        return text[: matches[-1].end()].strip()
    return text.rstrip() + "."


def _vlm_dtype():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_vlm():
    """Load (or return cached) VLM model + processor tuple."""
    global _vlm_model, _vlm_processor  # noqa: PLW0603
    if _vlm_model is None or _vlm_processor is None:
        dtype = _vlm_dtype()
        if "llava" in VLM_MODEL_ID.lower():  # llava kernels expect fp16 on many GPUs
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        processor_kwargs = {"trust_remote_code": True}
        processor_cls = AutoProcessor
        if "llava" in VLM_MODEL_ID.lower():
            processor_kwargs["use_fast"] = False
        _vlm_processor = processor_cls.from_pretrained(VLM_MODEL_ID, **processor_kwargs)
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


def unload_vlm():
    global _vlm_model, _vlm_processor  # noqa: PLW0603
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
    instruction = (
        "Describe the garment in this image for a virtual try-on system. "
        "Mention color, garment type, sleeve/fit, and distinguishing details in ~15 words."
    )
    start_time = time.perf_counter()
    status = "success"
    error_message = ""
    response_text = "a garment"
    image_dims = f"{image.width}x{image.height}" if image is not None else "none"

    if image is None:
        status = "missing-image"
        _log_garment_description_metrics(
            instruction, response_text, status, image_dims, 0.0, error_message
        )
        return response_text

    try:
        model, processor = load_vlm()
    except Exception as exc:  # pylint: disable=broad-except
        status = "load-error"
        error_message = str(exc)
        print(f"[WARN] Failed to load VLM: {exc}")
        _log_garment_description_metrics(
            instruction, response_text, status, image_dims, 0.0, error_message
        )
        return response_text

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
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
        response_text = _clean_vlm_response(decoded[0] if decoded else "")
        return response_text
    except Exception as exc:  # pylint: disable=broad-except
        status = "generation-error"
        error_message = str(exc)
        print(f"[WARN] Failed to generate garment description: {exc}")
        return response_text
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        _log_garment_description_metrics(
            instruction,
            response_text,
            status,
            image_dims,
            duration_ms,
            error_message,
        )


def resize_for_vlm(image: Image.Image, size=(224, 224)) -> Image.Image:
    return image.resize(size).convert("RGB")


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


def generate_tryon_evaluation(output_image, model, processor, garment_description: str | None = None) -> dict:
    start_time = time.perf_counter()
    status = "success"
    error_message = ""
    response_text = "Evaluation unavailable."
    score_value: int | None = None

    if output_image is None:
        status = "missing-output"
        response = {"text": "No output image to evaluate.", "score": None}
        _log_tryon_eval_metrics(
            garment_description,
            response,
            status,
            time.perf_counter() - start_time,
            error_message,
            output_dims="none",
        )
        return response
    if model is None or processor is None:
        status = "missing-model"
        response = {"text": "Evaluation unavailable.", "score": None}
        _log_tryon_eval_metrics(
            garment_description,
            response,
            status,
            time.perf_counter() - start_time,
            error_message,
            output_dims=f"{output_image.width}x{output_image.height}",
        )
        return response

    garment_focus = garment_description or "the top garment"
    instruction = (
        "You are a professional fashion stylist and visual QA. "
        "Evaluate ONLY the try-on garment described as "
        f"'{garment_focus}'. Ignore jeans, pants, shoes, or other unrelated pieces. "
        "Comment on: 1) how the top fits the model's torso and sleeves; "
        "2) how the top's colors harmonize with the rest of the outfit; "
        "3) how realistic the top's blending looks; 4) one improvement suggestion for the top. "
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
        score_value, comment = _extract_vlm_score(cleaned)
        if not comment:
            comment = cleaned
        response_text = _ensure_complete_sentence(comment)
        return {"text": response_text, "score": score_value}

    except Exception as exc:  # pylint: disable=broad-except
        status = "generation-error"
        error_message = str(exc)
        print(f"[WARN] Try-on eval failed: {exc}")
        response_text = "Evaluation failed."
        score_value = None
        return {"text": response_text, "score": score_value}
    finally:
        duration_s = time.perf_counter() - start_time
        _log_tryon_eval_metrics(
            garment_description,
            {"text": response_text, "score": score_value},
            status,
            duration_s,
            error_message,
            output_dims=f"{output_image.width}x{output_image.height}" if output_image else "none",
            instruction=instruction,
        )


def _log_garment_description_metrics(
    instruction: str,
    response_text: str,
    status: str,
    image_dims: str,
    latency_ms: float,
    error_message: str,
):
    text_artifacts = {
        "instruction.txt": instruction,
        "response.txt": response_text,
    }
    if error_message:
        text_artifacts["error.txt"] = error_message
    tracking.log_event(
        "vlm_garment_description",
        params={
            "vlm_model_id": VLM_MODEL_ID,
            "image_dims": image_dims,
        },
        metrics={"latency_ms": latency_ms},
        tags={
            "component": "vlm",
            "stage": "garment_description",
            "status": status,
        },
        text_artifacts=text_artifacts,
    )


def _log_tryon_eval_metrics(
    garment_description: str | None,
    response: dict,
    status: str,
    duration_s: float,
    error_message: str,
    output_dims: str,
    instruction: str | None = None,
):
    response_text = response.get("text", "")
    text_artifacts = {
        "response.txt": response_text,
    }
    if instruction:
        text_artifacts["instruction.txt"] = instruction
    if error_message:
        text_artifacts["error.txt"] = error_message
    metrics = {"latency_ms": duration_s * 1000}
    score_val = response.get("score")
    if score_val is not None:
        try:
            metrics["score"] = float(score_val)
        except (TypeError, ValueError):
            pass
    tracking.log_event(
        "vlm_tryon_evaluation",
        params={
            "vlm_model_id": VLM_MODEL_ID,
            "garment_focus": garment_description or "the top garment",
            "output_dims": output_dims,
        },
        metrics=metrics,
        tags={
            "component": "vlm",
            "stage": "tryon_evaluation",
            "status": status,
        },
        text_artifacts=text_artifacts,
    )


def build_gallery_annotations(gallery_entries, evaluations):
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


def summarize_vlm_recommendations(gallery_entries, evaluations):
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

    def _friendly_caption_parts(idx_zero_based: int, caption: str):
        default_name = f"Look {idx_zero_based + 1}"
        if not caption:
            return default_name, ""
        parts = caption.split(":", 2)
        look_tag = parts[0].strip()
        number_match = re.search(r"\d+", look_tag)
        look_name = f"Look {number_match.group()}" if number_match else (look_tag or default_name)
        garment_desc = ""
        if len(parts) == 3:
            garment_desc = parts[2].strip()
        elif len(parts) == 2:
            garment_desc = parts[1].strip()
        if garment_desc:
            garment_desc = garment_desc[0].upper() + garment_desc[1:]
        return look_name, garment_desc

    def _trim_comment(text: str) -> str:
        if not text:
            return "No comment"
        cleaned = text.strip()
        cleaned = re.sub(r"^/?10\s*\|\s*", "", cleaned)
        cleaned = re.sub(r"^Comment\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*\|\s*$", "", cleaned)
        cleaned = cleaned.strip()
        return cleaned or "No comment"

    summary_lines = []
    for idx, ((_, caption), evaluation) in enumerate(zip(gallery_entries, evaluations), start=1):
        look_name, garment_desc = _friendly_caption_parts(idx - 1, caption)
        score_val = evaluation.get("score")
        score_text = f"{score_val}/10" if score_val is not None else "N/A"
        comment = _trim_comment(evaluation.get("text"))
        desc_line = garment_desc or "Garment description unavailable."
        summary_lines.append(
            f"{idx}) {look_name}\n"
            f"   Garment: {desc_line}\n"
            f"   Score: {score_text}\n"
            f"   Stylist note: {comment}"
        )

    if best_score is None:
        heading = "VLM feedback (no numeric scores provided)."
    else:
        winners = ", ".join(f"Look {i + 1}" for i in best_indices)
        heading = f"Top choice(s): {winners} — {best_score}/10."

    return heading + "\n\n" + "\n\n".join(summary_lines)
