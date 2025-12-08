# Sprint 2
This document details the achievables in the second sprint

## Updated Objective
Given:
- a person image `P`
- a garment image `G`
- optional text description `T`

Generate:
- a try-on image `Y = f(P, G, T)` that preserves:
  - person identity/pose
  - garment attributes (shape, texture, logo/pattern)
  - realistic occlusion and lighting

This has also been updated in - [Sprint 1](SPRINT1.md).

---

## AI Component (system perspective)
We treat IDM-VTON as the core AI component and place it inside a small “AI try-on service” boundary.

### Intended users
- **End-user** (future): uploads a photo + chooses a garment, gets try-on result.

### Inputs
- Person image (jpg/png)
- Garment image (jpg/png)
- Optional text prompt describing garment (string)
- Optional run config (seed, steps, guidance, resolution)

### Outputs
- Generated try-on image (png)
- Optional mask / intermediate outputs (for debugging)
- MLflow artifacts:
  - input images
  - output images
  - metrics
  - runtime/latency

### Interaction flow (current)
1) Run inference through `inference_pair.py`
2) Save output images to `outputs/`
3) Log run info and artifacts to MLflow (experiment tracking) (YET TO BE ADDED)

---

## Idea: add a VLM evaluator
We propose adding a Vision-Language Model (VLM) evaluator to score try-on results. The VLM is treated as a **noisy judge**, not ground truth.

### Why it helps
IDM-VTON outputs can look “good” but still fail on:
- wrong garment texture/pattern
- garment warping
- missing limbs / artifacts
- unrealistic occlusions

### How we would implement it
- Define a rubric with 4–6 criteria (0–5 each), e.g.:
  1) garment present & recognizable
  2) logo/pattern preservation
  3) body/pose plausibility
  4) occlusion realism (hair/arms/straps)
  5) artifact severity (optional)
- Ask the VLM to score **and briefly justify** each run.
- Log:
  - numeric rubric scores as **MLflow metrics**
  - the VLM justification text as an **MLflow artifact**

### Caveats
- VLM scores can be biased/inconsistent → calibrate with a small human-labeled subset.
- Avoid optimizing only for VLM score (risk of “gaming” the evaluator).

### Metrics we plan to log
Because paired ground-truth may not always exist for our own images, we focus on practical evaluation:

- **Runtime/latency** (seconds)
- **CLIP-like similarity proxies**
  - output vs garment (garment fidelity proxy)
  - output vs text prompt (text alignment proxy)
- **VLM rubric score** (optional evaluator: treated as a noisy judge)
  - garment present & recognizable?
  - logos/patterns preserved?
  - body/pose plausible (no missing limbs)?
  - occlusion realistic (hair/arms/straps)?

---

## Infrastructure requirement: MLflow provision via Docker Compose
We run experiments on a GPU server. To make tracking reproducible for the whole group, we will add:

- `infra/mlflow/docker-compose.yml`
  - MLflow tracking server

This allows:
- multiple machines to log to the same MLflow server
- persistent metrics + artifacts

---

## Sprint 2 checklist
- [ ] GitHub: repository includes code for model building (training and evaluation code)
- [x] GitHub: repository includes code for infrastructure provision/deploy/installation MLFlow e.g., Docker compose scripts
- [x] GitHub: repository includes test scripts and saved trained model
- [x] GitHub: README.md improve previous project proposal description where necessary
- [x] GitHub: README.md Pre-trained model/method and the AI component: In addition to description of the approach used by the selected pre-trained AI/ML models or algortithms, add details from system perspective of the AI component e.g., how users (end-user/other system) are envisioned to interact with the AI component, its inputs and outputs etc
- [ ] README.md: Modify the previous dataset title to Experiment and Dataset. Then add details of experiments conducted with supporting screenshots from MLFlow of model evaluation metrics. Improve the previous dataset description with insights gained from initial experiments.
- [x] README.md: Add at the end of the description project build/running instructions
- [x] README.md: Add initial ideas for model deployment and inference serving.