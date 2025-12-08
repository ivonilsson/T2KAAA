from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
IDM_VTON_ROOT = PROJECT_ROOT / "third_party" / "IDM-VTON"
GRADIO_DEMO_DIR = IDM_VTON_ROOT / "gradio_demo"

DENSEPOSE_CFG = IDM_VTON_ROOT / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml"
DENSEPOSE_CKPT = IDM_VTON_ROOT / "ckpt" / "densepose" / "model_final_162be9.pkl"

CATALOG_ROOT = PROJECT_ROOT / "data"
MAX_SELECTED_GARMENTS = 3
DEFAULT_SAMPLE_COUNT = 10

CUSTOM_CSS_PATH = PROJECT_ROOT / "assets" / "gradio_theme.css"
EXAMPLE_PATH = GRADIO_DEMO_DIR / "example"
