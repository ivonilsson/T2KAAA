# T2KAAA Course AI System NeuroFit

## Sprint Documentation
- [Sprint 1](docs/SPRINT1.md)
- [Sprint 2](docs/SPRINT2.md)
- [Sprint 3](docs/SPRINT3.md)

## Authors
- [Ivo Österberg Nilsson](mailto:osiv20jw@student.ju.se) · [@ivonilsson](https://github.com/ivonilsson)
- [Simon De Reuver](mailto:desi20kt@student.ju.se) · [@simondereuver](https://github.com/simondereuver)
- [Samuel Wallander Leyonberg](mailto:wasa20zy@student.ju.se) · [@SamuelWLeyonberg](https://github.com/SamuelWLeyonberg)

## Requirements
This project currently **requires Python 3.10** for the IDM-VTON integration to work.

On Windows it is recommended to use a dedicated virtual environment:

```bash
# From project root
py -3.10 -m venv venv
venv\Scripts\activate
```

If you are on another OS or CPU-only setup, see the official install guide:
https://pytorch.org/get-started/locally/

On Windows, install GPU-enabled PyTorch first:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining dependencies:

```bash
pip install -r requirements_vton.txt
```

### IDM-VTON submodule setup
IDM-VTON is included as a git submodule under `third_party/IDM-VTON`. After pulling the submodule, expect ~30–35 GB of disk usage once all checkpoints finish downloading:

- `diffusion_pytorch_model.bin` – ~12 GB
- SDXL / IP-Adapter / VAE weights – ~16 GB combined
- DensePose, OpenPose, human parsing checkpoints – ~1 GB total
- Virtual environment and auxiliary packages – ~3–4 GB

Place the required checkpoints as follows:

```
third_party/IDM-VTON/ckpt/
  densepose/model_final_162be9.pkl
  humanparsing/parsing_atr.onnx
  humanparsing/parsing_lip.onnx
  openpose/ckpts/body_pose_model.pth
```

Download the real files (the repo only includes placeholders) from Hugging Face:

For Windows:
```bash
curl -L "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx" -o ckpt\humanparsing\parsing_atr.onnx
curl -L "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_lip.onnx" -o ckpt\humanparsing\parsing_lip.onnx
curl -L "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl" -o ckpt\densepose\model_final_162be9.pkl
curl -L "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth" -o ckpt\openpose\ckpts\body_pose_model.pth
```

For Linux:
```bash
curl -L "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx" -o ckpt/humanparsing/parsing_atr.onnx
curl -L "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_lip.onnx" -o ckpt/humanparsing/parsing_lip.onnx
curl -L "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl" -o ckpt/densepose/model_final_162be9.pkl
curl -L "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth" -o ckpt/openpose/ckpts/body_pose_model.pth
```

## Quick inference test
Run the thin wrapper around IDM-VTON to bypass the original Gradio app and directly generate a try-on result:

```bash
python inference_pair.py --person assets\test\person.jpeg --garment assets\test\shirt.jpg --desc "long sleeve blue shirt" --out outputs\tryon.png --out-mask outputs\tryon_mask.png
```

The script loads both images, executes OpenPose, human parsing, DensePose, then IDM-VTON, and saves the synthesized render plus mask under `outputs/`.

## Build and run
1. **Clone and update submodules**
  ```bash
  git clone --recurse-submodules git@github.com:ivonilsson/T2KAAA.git
  ```
2. **Create a Python 3.10 virtual environment** (see Requirements section).
3. **Install dependencies** with `pip install -r requirements_vton.txt` after installing the correct PyTorch build.
4. **Download checkpoints** into `third_party/IDM-VTON/ckpt/` using the commands above.
5. **Run inference (experiments to be added)** using `python inference_pair.py ...` 

## Attribution
This project integrates IDM-VTON for virtual try-on:

Yisol et al., “IDM-VTON: Improving Diffusion Models for Authentic Virtual Try-on in the Wild” (ECCV 2024).
Code: https://github.com/yisol/IDM-VTON

The IDM-VTON code and checkpoints (under `third_party/IDM-VTON`) are licensed under CC BY-NC-SA 4.0 and may be used only for non-commercial purposes. See `third_party/IDM-VTON/LICENSE.txt` for full terms.