import sys
from pathlib import Path

import config

for path in (config.SRC_ROOT, config.GRADIO_DEMO_DIR, config.IDM_VTON_ROOT, config.PROJECT_ROOT):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

import gradio as gr

import tracking

from services.catalog import (
    ensure_garment_descriptions,
    list_catalog_categories,
    load_catalog_samples,
    update_catalog_selection,
)
from services.tryon_service import start_tryon


def _log_app_startup():
    tracking.log_event(
        "app_startup",
        params={
            "share": True,
            "server_port": 9090,
        },
        tags={
            "component": "app",
            "stage": "boot",
            "status": "starting",
        },
        text_artifacts={
            "message.txt": "T2KAAA Gradio app launched via app.py",
        },
    )


_log_app_startup()


def _read_custom_css() -> str:
    try:
        return config.CUSTOM_CSS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _panel_visibility(target: str):
    target = (target or "").lower()
    return (
        gr.update(visible=target == "welcome"),
        gr.update(visible=target == "catalog"),
        gr.update(visible=target == "tryon"),
    )


def _build_scroll_js(target_id: str) -> str:
    return (
        "function(){const el=document.getElementById('"
        + target_id
        + "'); if(el){el.scrollIntoView({behavior:'smooth',block:'start'});}}"
    )


_CUSTOM_CSS = _read_custom_css()
_SCROLL_TO_CATALOG = _build_scroll_js("catalog-panel")
_SCROLL_TO_TRYON = _build_scroll_js("tryon-panel")

image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    if _CUSTOM_CSS:
        gr.HTML(f"<style>{_CUSTOM_CSS}</style>")
    catalog_categories = list_catalog_categories()
    catalog_samples_state = gr.State([])
    selection_details_state = gr.State({})

    with gr.Column(elem_id="welcome-panel", visible=True) as welcome_panel:
        gr.Markdown("## Welcome to the T2KAAA showroom")
        gr.Markdown(
            "Explore curated garments, pick up to three looks, then head into our virtual fitting room to see them on you."
        )
        start_catalog_btn = gr.Button("Browse catalog looks", elem_id="cta-browse", variant="primary")

    with gr.Column(elem_id="catalog-panel", visible=False) as catalog_panel:
        gr.Markdown("### Step 1 · Browse the rack")
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
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
                    value=config.DEFAULT_SAMPLE_COUNT,
                )
                load_samples_btn = gr.Button("Load catalog samples")
                proceed_to_tryon_btn = gr.Button(
                    "Proceed to virtual fitting room",
                    elem_id="cta-to-tryon",
                    interactive=False,
                    variant="primary",
                )
            with gr.Column(scale=2):
                catalog_gallery = gr.Gallery(
                    label="Catalog preview",
                    columns=5,
                    height=420,
                    allow_preview=True,
                )
        selection_box = gr.CheckboxGroup(
            label=f"Select up to {config.MAX_SELECTED_GARMENTS} garments",
            choices=[],
            value=[],
        )

    with gr.Column(elem_id="tryon-panel", visible=False) as tryon_panel:
        gr.Markdown("### Step 2 · Virtual fitting room")
        back_to_catalog_btn = gr.Button("Back to catalog", variant="secondary")
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                imgs = gr.ImageEditor(
                    sources="upload",
                    type="pil",
                    label="Upload your photo (auto mask enabled by default).",
                    interactive=True,
                )
                is_checked = gr.Checkbox(label="Use auto-generated mask", info="Recommended", value=True)
                is_checked_crop = gr.Checkbox(label="Use smart crop", info="Centers the subject", value=False)
                run_vlm_eval = gr.Checkbox(
                    label="Run AI stylist evaluation",
                    info="Disable to skip the VLM judging step.",
                    value=True,
                )
                with gr.Accordion(label="Advanced Settings", open=False):
                    with gr.Row():
                        denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=30, step=1)
                        seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)
                try_button = gr.Button("Generate try-on", elem_id="run-tryon-btn")
                garment_desc_box = gr.Textbox(
                    label="Garment auto-description",
                    interactive=False,
                    lines=6,
                    max_lines=10,
                )
            with gr.Column(scale=2):
                image_out = gr.Gallery(
                    label="Try-on results",
                    elem_id="output-img",
                    columns=2,
                    height=620,
                    allow_preview=True,
                )
                vlm_comment_box = gr.Textbox(
                    label="AI Stylist summary",
                    elem_id="stylist-summary-box",
                    interactive=False,
                    lines=14,
                    max_lines=24,
                )

    start_catalog_btn.click(
        fn=lambda: _panel_visibility("catalog"),
        outputs=[welcome_panel, catalog_panel, tryon_panel],
        queue=False,
    )

    proceed_to_tryon_btn.click(
        fn=lambda: _panel_visibility("tryon"),
        outputs=[welcome_panel, catalog_panel, tryon_panel],
        js=_SCROLL_TO_TRYON,
        queue=False,
    )

    back_to_catalog_btn.click(
        fn=lambda: _panel_visibility("catalog"),
        outputs=[welcome_panel, catalog_panel, tryon_panel],
        js=_SCROLL_TO_CATALOG,
        queue=False,
    )

    def _prepare_tryon_descriptions(selected_choices, catalog_metadata, selection_details):
        updated_details, desc_summary = ensure_garment_descriptions(
            selected_choices,
            selection_details,
            catalog_metadata,
        )
        return desc_summary, updated_details

    def _run_tryon_job(
        editor_state,
        selected_choices,
        catalog_metadata,
        selection_details,
        run_vlm_eval_flag,
        auto_mask_flag,
        auto_crop_flag,
        denoise_steps_val,
        seed_val,
    ):
        gallery, stylist_summary = start_tryon(
            editor_state,
            selected_choices,
            catalog_metadata,
            selection_details,
            run_vlm_eval_flag,
            auto_mask_flag,
            auto_crop_flag,
            denoise_steps_val,
            seed_val,
        )
        return gallery, stylist_summary

    tryon_event = try_button.click(
        fn=_prepare_tryon_descriptions,
        inputs=[selection_box, catalog_samples_state, selection_details_state],
        outputs=[garment_desc_box, selection_details_state],
    )

    tryon_event.then(
        fn=_run_tryon_job,
        inputs=[
            imgs,
            selection_box,
            catalog_samples_state,
            selection_details_state,
            run_vlm_eval,
            is_checked,
            is_checked_crop,
            denoise_steps,
            seed,
        ],
        outputs=[image_out, vlm_comment_box],
        api_name="tryon",
    )

    load_samples_btn.click(
        fn=load_catalog_samples,
        inputs=[catalog_dropdown, sample_count],
        outputs=[
            catalog_gallery,
            catalog_samples_state,
            selection_box,
            proceed_to_tryon_btn,
            selection_details_state,
        ],
        queue=True,
    )

    selection_box.change(
        fn=update_catalog_selection,
        inputs=[selection_box, catalog_samples_state, selection_details_state],
        outputs=[selection_box, proceed_to_tryon_btn, selection_details_state],
        queue=True,
    )

image_blocks.launch(server_name="0.0.0.0", server_port=9090, share=True)
