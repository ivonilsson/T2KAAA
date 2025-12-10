# T2KAAA Course AI System NeuroFit

## Authors
- [Ivo Österberg Nilsson](mailto:osiv20jw@student.ju.se) · [@ivonilsson](https://github.com/ivonilsson)
- [Simon De Reuver](mailto:desi20kt@student.ju.se) · [@simondereuver](https://github.com/simondereuver)
- [Samuel Wallander Leyonberg](mailto:wasa20zy@student.ju.se) · [@SamuelWLeyonberg](https://github.com/SamuelWLeyonberg)

## Background
Online retail generates high volumes of returns. The National Retail Federation estimates 19.3% of online sales will be returned this year (2025) [[NRF][nrf]]. Each return triggers extra transport, repackaging, and handling. Economically, high return rates provide little to no utility, incurring costs without creating value. Retailers lose profits margins and customers experience inconvenience and frustration. Furthermore, repeated shipping and repackaging contribute to higher co2 emotions and waste products, increasing the industry's environmental footprint. 

One key factor in returns is misleading or inaccurate visualisation of the item being sold. Product images are often displayed by professional models that may not represent the customer's body type or proportions, leading to a mismatch in expectations and a higher chance of return. 

To address this, we can utilize the fast evolving technology in image generation and processing. By using pretrained models for virtual fitting of clothes powered by AI, a customer can “try on” clothes on a picture of themselves. Getting a more fair visualisation and aligning expectations before he/she completes the purchas. thereby reducing returns and addressing the problem. This also allows the customer to get a fair view of the product, without any misleading models. This technology allows for new ways of online shipping that are more fun and fair. Customers get the chance to “try on” multiple clothes and to be more creative than what traditional online shopping allows for. Encouraging experimentation, transparency, more sustainable and enjoyable shopping for consumers and retailers.

This is not a silver bullet for the challenge at hand, but it is a step in the right direction to minimize returns and reduce the industries environmental footprint.
[nrf]: https://nrf.com/research/2025-retail-returns-landscape

## Objective
Given:
- An image of a person (jpg/png) `P`
- An image of a garment (jpg/png) `G`

Generate:
- A text description of the garment `T`
- A try-on image `Y = f(P, G, T)` that preserves:
  - Person identity/pose
  - Garment attributes (shape, texture, logo/pattern)
  - Realistic occlusion and lighting
  - Textual evaluation and stylistic feedback

- MLflow:
  - Input images
  - Output images
  - Metrics
  - Params
  - Tags


## System walkthrough
To fulfill the objectives stated above, this system uses pre-trained AI models as well as software structures such as Docker and Gradio. From the user perspective, a gradio instance serves as the GUI reachable through the browser while the rest of the system runs in a docker container. 
- Workflow
  - 1. The user visits the URL and is presented by a landing page before directed to the actual system interface.
  - 2. The user is displayed by filters to apply upon a catalog containing images clothings that are pre-defined in the container `G`.
  - 3. The user may select up to three garments before proceeding to virtual try-on.
  - 4. A Vision-Language Model (VLM) is loaded in the container to describes the garment `T`.
  - 5. The user is asked to upload an image containing a person `P` that will be targeted for the try-on.
  - 6. As the user presses to continue, the VLM is unloaded from the container to save memory as the image-generation model is loaded in.
  - 7. The image-generation model is given the inputs `P, G & T` to starts its inference. 
  - 8. The try-on images are generated.
  - 9. The image-generating model is unloaded as the VLM is once more loaded in.
  - 10. The VLM evaluates the generated images and score them from a stylistic point of view.
  - 11. Results are presented to the user in the gradio interface.

![Alt text](assets/arch.drawio1.webp)

---
## Pre-trained model/method and the AI component
This project integrates two pre-trained models to reach our objective, IDM-VTON for the virtual try-on image generation and LLaVA as the vision-language model (VLM) for the garment description. IDM-VTON requires a textual prompt describing the garment to work as intended. To automate this step, LLaVa performs image-to-text inference on the garment image and produces the descriptive prompt. LLaVa is further used in a post-generation stage. After IDM-VTON produces the try-on images, LLaVa evaluation and generates stylistic feedback, providing the user with suggestions for potential improvements or alternative styling options. Thus generating the textual evaluation and stylistic feedback.

- IDM-VTON (ECCV 2024)
A state-of-the-art image-generation, diffusion-based virtual try-on model. It is a recent research model specifically targeting garment fidelity and realistic try-on “in the wild”.
- Why IDM-VTON?:
  - Official code/checkpoints are available for reproducible experimentation.
  - Free to use.
  - State-Of-The-Art model for try-on generation.
  - Suits our case without the need for fine-tuning.

- LLaVA
A VLM that generated descriptive texts from a given image. It combines a vision encoder with a large language model, enabling it to describe images, answer questions about visual content, and perform reasoning that mixes image and text.
- Why LLaVA?:
  - Open-source
  - Lightweight
  - IDM-VTON outputs can look “good” but still fail on:
    - Wrong garment texture/pattern
    - Garment warping
    - Missing limbs / artifacts
    - Unrealistic occlusions
   Prompting IDM-VTON with a descriptive text of the garment produces better and more consistent inference. However, we do not want the user to have to write this prompt for each garment.
- Caveats
  - VLM scores can be biased/inconsistent → calibrate with a small human-labeled subset.
  - Avoid optimizing only for VLM score (risk of “gaming” the evaluator).

## Model Deployment and Inference Serving
We deploy the IDM-VTON try-on pipeline as a Gradio-driven inference service (app.py) that hosts catalog browsing, photo upload, and generation. Each request logs end-to-end stage timings, prompts, scores, and preview images to MLflow via the new tracking.py helper, giving us latency tracing plus visual QA artifacts. MLflow’s experiment UI now shows these metrics and screenshots per run, so we can monitor reliability and quality.

## Limitations
As this system where hosted on our university's internal server as it's free for us as students to use. However, this comes with the caveat that we needed to adapt to the servers hardware limitation and administrative limitations/rules. This is why our system onload and offloads models in its workflow, in order to save memory. It is also one of the reason for the system to run in a docker container as that setup is required on the server.

---

## MLFlow
Because paired ground-truth may not always exist for our own images, we focus to log these practical evaluation:
- Inputs
  - Garment image
  - Garment mask
  - Person image
  - dens-pose mask
- Outputs
  - VLM garment description
  - VLM try-on image evaluation
  - Generated try-on image
- Metrics
  - Generation times
  - Total latency
  - GPU memory usage
- Params
  - Seed
  - Setup settings
  - GPU name
- Tags
  - IDs
  - Status

## Infrastructure requirement: MLflow provision via Docker Compose
We run experiments on a GPU server. To make tracking reproducible for the whole group, we will add:

- `infra/mlflow/docker-compose.yml`
  - MLflow tracking server

This allows:
- multiple machines to log to the same MLflow server
- persistent metrics + artifacts

---

## Conclusion
Our modular try-on service plus MLflow proved the core idea, trying on garments virtually through IDM-VTON while logging every prompt, latency, and stylist score, works reliably. We would have liked for this to become a web page similar to a real deployment, using IaC and other deployment techniques for full deployment checklist and scalability, instead of a Gradio interface. But we were limited by the compute infrastructure available to us.

## Usage
This project currently **requires Python 3.10** for the IDM-VTON integration to work.

Use virtual environment:
Linux
```bash
python -3.10 -m venv venv
source venv/bin/activate
```
Windows
```bash
python -3.10 -m venv venv
venv\Scripts\activate
```

Install pytorch
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

```bash
git submodule update --init --recursive
```

Download the IDM-VTON preprocessing checkpoints (the repo only includes placeholders) from Hugging Face
From repo root:
```bash
curl -L "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx" -o third_party/IDM-VTON/ckpt/humanparsing/parsing_atr.onnx
curl -L "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_lip.onnx" -o third_party/IDM-VTON/ckpt/humanparsing/parsing_lip.onnx
curl -L "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl" -o third_party/IDM-VTON/ckpt/densepose/model_final_162be9.pkl
curl -L "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth" -o third_party/IDM-VTON/ckpt/openpose/ckpts/body_pose_model.pth
```

## Quick inference test
Run the thin wrapper around IDM-VTON to bypass the original Gradio app and directly generate a try-on result:

```bash
python inference_pair.py --person assets/test/person.jpeg --garment assets/test/shirt.jpg --desc "long sleeve blue shirt" --out outputs/tryon.png --out-mask outputs/tryon_mask.png
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