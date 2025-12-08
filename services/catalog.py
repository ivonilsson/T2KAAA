"""Catalog helpers for browsing, sampling, and garment selection."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

import gradio as gr
from PIL import Image

import config
from services.vlm import generate_garment_description

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def list_catalog_categories(catalog_root: Path | None = None) -> List[str]:
    root = catalog_root or config.CATALOG_ROOT
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _find_category_images(category: str, catalog_root: Path) -> List[Path]:
    if not category:
        return []
    category_dir = catalog_root / category
    if not category_dir.exists():
        return []
    return [
        path
        for path in category_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    ]


def _sample_catalog_images(category: str, sample_count: int, catalog_root: Path) -> List[Path]:
    paths = _find_category_images(category, catalog_root)
    if not paths:
        return []
    sample_count = max(1, min(sample_count, len(paths)))
    if len(paths) <= sample_count:
        return random.sample(paths, len(paths))
    return random.sample(paths, sample_count)


def open_image(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to load garment image {path}: {exc}")
    return None


def _format_choice_label(path: Path, idx: int) -> str:
    return f"#{idx}: {path.name}"


def load_catalog_samples(
    category: str,
    sample_count,
    catalog_root: Path | None = None,
    default_sample_count: int | None = None,
    max_selected: int | None = None,
):
    if not category:
        raise gr.Error("Select a garment category to load samples.")
    root = catalog_root or config.CATALOG_ROOT
    default_count = default_sample_count or config.DEFAULT_SAMPLE_COUNT
    max_items = max_selected or config.MAX_SELECTED_GARMENTS
    try:
        sample_count_int = int(sample_count)
    except (TypeError, ValueError):
        sample_count_int = default_count
    sample_paths = _sample_catalog_images(category, sample_count_int, root)
    if not sample_paths:
        raise gr.Error(f"No garments found under '{category}'.")
    gallery_entries = []
    metadata = []
    checkbox_choices = []
    for idx, path in enumerate(sample_paths, start=1):
        image = open_image(path)
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
        f"Select up to {max_items} garments below."
    )
    selection_reset = gr.update(choices=checkbox_choices, value=[])
    proceed_reset = gr.update(interactive=False)
    details_reset = {}
    return gallery_entries, metadata, selection_reset, summary, proceed_reset, details_reset


def update_catalog_selection(
    selected_choices,
    sample_metadata,
    selection_details,
    max_selected: int | None = None,
):
    sample_metadata = sample_metadata or []
    selected_choices = selected_choices or []
    max_items = max_selected or config.MAX_SELECTED_GARMENTS
    limited = selected_choices[:max_items]
    meta_map = {item["choice"]: item for item in sample_metadata}
    previous_details = selection_details or {}
    selection_details = {}

    for choice in limited:
        meta = meta_map.get(choice)
        if meta is None:
            continue
        entry = previous_details.get(choice, {}).copy()
        entry["choice"] = choice
        entry["path"] = meta["path"]
        entry["label"] = meta["label"]
        if not entry.get("description"):
            image = open_image(Path(meta["path"]))
            if image is None:
                print(f"[WARN] Catalog garment missing for selection: {meta['path']}")
                entry["description"] = "a garment"
            else:
                entry["description"] = generate_garment_description(image) or "a garment"
        selection_details[choice] = entry

    summary_lines = []
    for choice in limited:
        meta = meta_map.get(choice)
        if not meta:
            continue
        desc = selection_details.get(choice, {}).get("description") or "a garment"
        summary_lines.append(f"- {meta['label']}: {desc}")

    if not limited:
        summary = "No garments selected."
    else:
        summary = "Selected garments:\n" + "\n".join(summary_lines)
        if len(selected_choices) > max_items:
            summary += f"\n(Only the first {max_items} selections are used.)"

    proceed_state = gr.update(interactive=bool(limited))
    return gr.update(value=limited), summary, proceed_state, selection_details


def prepare_selected_garments(selected_choices, catalog_metadata, selection_details):
    selected_choices = selected_choices or []
    if not selected_choices:
        raise gr.Error("Select up to three catalog garments before running try-on.")
    catalog_metadata = catalog_metadata or []
    selection_details = selection_details or {}
    meta_map = {item["choice"]: item for item in catalog_metadata}
    garments = []
    for choice in selected_choices:
        meta = meta_map.get(choice)
        if meta is None:
            continue
        image = open_image(Path(meta["path"]))
        if image is None:
            print(f"[WARN] Catalog garment missing: {meta['path']}")
            continue
        desc_entry = selection_details.get(choice) or {}
        description = desc_entry.get("description") or "a garment"
        garments.append(
            {
                "path": meta["path"],
                "label": meta["label"],
                "image": image,
                "description": description,
            }
        )
    if not garments:
        raise gr.Error("Unable to load the selected catalog garments.")
    return garments
