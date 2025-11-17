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

On windows, recommendation is to first run ```pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118``` for GPU supported Torch, followed by ```pip install -r requirements_vton.txt``` for the remaining dependencies.

### IDM-VTON
Added IDM-VTON as git submodule, after installing all required components, we are looking at this kind of disc space required:

After installing Python dependencies and pulling the submodule, the first run will download large model weights from Hugging Face:

- diffusion_pytorch_model.bin – ~12 GB

- Additional SDXL / IP-Adapter / VAE weights – ~0.5 GB + 2.78 GB + 2.53 GB + 0.33 GB + 10.3 GB

- Human parsing / DensePose / OpenPose checkpoints – ~0.8–1 GB total

- Plus a few GB for the virtual environment and packages

- In total, expect 30–35 GB disk usage for the full try-on pipeline.

#### IDM-VTON expects the following files in third_party/IDM-VTON/ckpt:
ckpt/
  densepose/
    model_final_162be9.pkl
  humanparsing/
    parsing_atr.onnx
    parsing_lip.onnx
  openpose/
    ckpts/
      body_pose_model.pth

The repo ships small placeholder text files with these names; they need to be overwritten with the real checkpoints. From third_party/IDM-VTON, run:

```curl -L "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx" -o ckpt\humanparsing\parsing_atr.onnx```

```curl -L "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_lip.onnx" -o ckpt\humanparsing\parsing_lip.onnx```

```curl -L "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl" -o ckpt\densepose\model_final_162be9.pkl```

```curl -L "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth" -o ckpt\openpose\ckpts\body_pose_model.pth```

### Test usage
Small wrapper around IDM-VTON to bypass the Gradio app and directly run inference on a person + clothing image. Run from root directory with activated venv: ```python inference_pair.py --person assets\test\person.jpeg --garment assets\test\shirt.jpg --desc "long sleeve blue shirt" --out outputs\tryon.png --out-mask outputs\tryon_mask.png```

This will:

- Load the person and garment image

- Run OpenPose, human parsing, DensePose and IDM-VTON

- Save the try-on result to outputs\tryon.png

## Data

### NOT NEEDED FOR NOW MAYBE DELETE LATER
Dataset can be aquired here:
https://drive.google.com/file/d/1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w/view

After download VITON-HD dataset, move vitonhd_test_tagged.json into the test folder, and move vitonhd_train_tagged.json into the train folder.

Structure of the Dataset directory should be as follows. (according to https://github.com/yisol/IDM-VTON?tab=readme-ov-file)

Structure of dataset:
train
|-- image
|-- image-densepose
|-- agnostic-mask
|-- cloth
|-- vitonhd_train_tagged.json

test
|-- image
|-- image-densepose
|-- agnostic-mask
|-- cloth
|-- vitonhd_test_tagged.json

## Attribution
This project integrates IDM-VTON for virtual try-on:

Yisol et al., “IDM-VTON: Improving Diffusion Models for Authentic Virtual Try-on in the Wild” (ECCV 2024).
Code: https://github.com/yisol/IDM-VTON

The IDM-VTON code and checkpoints (under third_party/IDM-VTON) are licensed under
CC BY-NC-SA 4.0 and may be used only for non-commercial purposes.
See third_party/IDM-VTON/LICENSE.txt for full terms.