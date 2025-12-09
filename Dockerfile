FROM python:3.10.12-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        openssh-server \
        sudo \
        wget \
        libnuma1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd

RUN python3 -m pip install --upgrade pip==23.3.1

RUN python3 -m pip install --no-cache-dir -f https://download.pytorch.org/whl/cu118/torch_stable.html \
        torch==2.0.1+cu118 \
        torchaudio==2.0.2+cu118 \
        torchvision==0.15.2+cu118

RUN python3 -m pip install --no-cache-dir \
        accelerate==0.25.0 \
        torchmetrics==1.2.1 \
        tqdm==4.66.1 \
        transformers==4.36.2 \
        diffusers==0.25.0 \
        einops==0.7.0 \
        bitsandbytes==0.39.0 \
        scipy==1.11.1 \
        opencv-python \
        gradio==4.24.0 \
        fvcore \
        cloudpickle \
        omegaconf \
        pycocotools \
        basicsr \
        av \
        onnxruntime==1.16.2

COPY entrypoint_demo.sh /entrypoint_demo.sh
RUN chmod +x /entrypoint_demo.sh
