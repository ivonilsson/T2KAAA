"""Simple IDM-VTON inference helper for a single person/garment pair."""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tryon import IDMVTONTryOn

def _auto_description(garment_path: str) -> str:
    stem = Path(garment_path).stem.replace("_", " ").replace("-", " ")
    stem = stem.strip() or "garment"
    return f"a {stem}" if "shirt" in stem.lower() else f"a {stem} garment"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IDM-VTON on a single person/garment pair")
    parser.add_argument("--person", required=True, help="Path to the person image")
    parser.add_argument("--garment", required=True, help="Path to the garment image")
    parser.add_argument("--output", "--out", dest="output", default="outputs/tryon.png", help="Result image path")
    parser.add_argument("--mask-output", "--out-mask", dest="mask_output", default=None, help="Optional path to save mask visualization")
    parser.add_argument("--manual-mask", default=None, help="Binary mask image used when --no-auto-mask is set")
    parser.add_argument("--desc", default=None, help="Text description of the garment (auto derived if omitted)")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-auto-mask", action="store_true", help="Disable automatic mask generation")
    parser.add_argument("--auto-crop", action="store_true", help="Enable automatic cropping")
    parser.add_argument("--device", default=None, help="Torch device to use (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto", help="Computation precision")
    parser.add_argument("--guidance-scale", type=float, default=2.0, help="Guidance scale passed to pipeline")
    parser.add_argument("--offload", action="store_true", help="Enable sequential CPU offload to cut VRAM usage")
    parser.add_argument("--no-vae-slicing", action="store_true", help="Disable VAE slicing (enabled by default)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path_label, path_value in [("person", args.person), ("garment", args.garment)]:
        if not Path(path_value).exists():
            raise SystemExit(f"{path_label} image not found: {path_value}")

    description = args.desc or _auto_description(args.garment)

    runner = IDMVTONTryOn(
        device=args.device,
        precision=args.precision,
        guidance_scale=args.guidance_scale,
        enable_sequential_cpu_offload=args.offload,
        enable_vae_slicing=not args.no_vae_slicing,
    )

    result_img, mask_img = runner.run_pair(
        person_image_path=args.person,
        garment_image_path=args.garment,
        garment_description=description,
        denoise_steps=args.steps,
        seed=args.seed,
        auto_mask=not args.no_auto_mask,
        auto_crop=args.auto_crop,
        manual_mask_path=args.manual_mask,
    )

    result_path = Path(args.output)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_img.save(result_path)
    print(f"Saved try-on result to {result_path}")

    if args.mask_output and mask_img is not None:
        mask_path = Path(args.mask_output)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(mask_path)
        print(f"Saved mask visualization to {mask_path}")
if __name__ == "__main__":
    main()
