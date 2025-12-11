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
		libgl1 \
		libglib2.0-0 \
		libsm6 \
		libxext6 \
		libxrender1 \
	&& rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd

RUN python3 -m pip install --upgrade pip==23.3.1

RUN python3 -m pip install --no-cache-dir -f https://download.pytorch.org/whl/cu118/torch_stable.html \
		torch==2.0.1+cu118 \
		torchaudio==2.0.2+cu118 \
		torchvision==0.15.2+cu118

RUN python3 -m pip install --no-cache-dir \
	absl-py==2.3.1 \
	accelerate==0.27.2 \
	addict==2.4.0 \
	aiofiles==23.2.1 \
	altair==5.5.0 \
	annotated-doc==0.0.4 \
	annotated-types==0.7.0 \
	antlr4-python3-runtime==4.9.3 \
	anyio==4.11.0 \
	attrs==25.4.0 \
	av==16.0.1 \
	basicsr==1.4.2 \
	bitsandbytes==0.39.0 \
	certifi==2022.12.7 \
	charset-normalizer==2.1.1 \
	click==8.3.0 \
	cloudpickle==3.1.2 \
	colorama==0.4.6 \
	coloredlogs==15.0.1 \
	contourpy==1.3.2 \
	cycler==0.12.1 \
	diffusers==0.30.2 \
	einops==0.7.0 \
	exceptiongroup==1.3.0 \
	fastapi==0.121.2 \
	ffmpy==1.0.0 \
	filelock==3.19.1 \
	flatbuffers==25.9.23 \
	fonttools==4.60.1 \
	fsspec==2025.10.0 \
	future==1.0.0 \
	fvcore==0.1.5.post20221221 \
	gradio==6.0.1 \
	gradio_client==2.0.0 \
	grpcio==1.76.0 \
	h11==0.16.0 \
	httpcore==1.0.9 \
	httpx==0.28.1 \
	huggingface-hub==0.36.0 \
	humanfriendly==10.0 \
	idna==3.4 \
	ImageIO==2.37.2 \
	importlib_metadata==8.7.0 \
	importlib_resources==6.5.2 \
	iopath==0.1.10 \
	Jinja2==3.1.6 \
	jsonschema==4.25.1 \
	jsonschema-specifications==2025.9.1 \
	kiwisolver==1.4.9 \
	lazy_loader==0.4 \
	lightning-utilities==0.15.2 \
	lmdb==1.7.5 \
	Markdown==3.10 \
	markdown-it-py==4.0.0 \
	MarkupSafe==2.1.5 \
	matplotlib==3.10.7 \
	mdurl==0.1.2 \
	mlflow==2.16.2 \
	mpmath==1.3.0 \
	narwhals==2.11.0 \
	networkx==3.3 \
	numpy==1.26.4 \
	omegaconf==2.3.0 \
	onnxruntime==1.16.2 \
	opencv-python==4.11.0.86 \
	orjson==3.11.4 \
	packaging==24.2 \
	pandas==2.3.3 \
	pillow==10.4.0 \
	platformdirs==4.5.0 \
	portalocker==3.2.0 \
	protobuf==5.26.0 \
	psutil==7.1.3 \
	pycocotools==2.0.10 \
	pydantic==2.12.4 \
	pydantic_core==2.41.5 \
	pydub==0.25.1 \
	Pygments==2.19.2 \
	pyparsing==3.2.5 \
	pytest==9.0.2 \
	python-dateutil==2.9.0.post0 \
	python-multipart==0.0.20 \
	pytz==2025.2 \
	PyYAML==6.0.3 \
	referencing==0.37.0 \
	regex==2025.11.3 \
	requests==2.28.1 \
	rich==14.2.0 \
	rpds-py==0.28.0 \
	ruff==0.14.5 \
	safetensors==0.6.2 \
	scikit-image==0.24.0 \
	scipy==1.11.1 \
	semantic-version==2.10.0 \
	shellingham==1.5.4 \
	six==1.17.0 \
	sniffio==1.3.1 \
	starlette==0.49.3 \
	sympy==1.14.0 \
	tabulate==0.9.0 \
	tb-nightly==2.21.0a20251023 \
	tensorboard-data-server==0.7.2 \
	termcolor==3.2.0 \
	tifffile==2025.5.10 \
	tokenizers==0.21.0 \
	tomli==2.3.0 \
	tomlkit==0.12.0 \
	torch==2.0.1+cu118 \
	torchaudio==2.0.2+cu118 \
	torchmetrics==1.2.1 \
	torchvision==0.15.2+cu118 \
	tqdm==4.66.1 \
	sentencepiece==0.2.0 \
	transformers==4.49.0 \
	typer==0.20.0 \
	typing-inspection==0.4.2 \
	typing_extensions==4.15.0 \
	tzdata==2025.2 \
	urllib3==1.26.13 \
	uvicorn==0.38.0 \
	websockets==11.0.3 \
	Werkzeug==3.1.3 \
	yacs==0.1.8 \
	yapf==0.43.0 \
	zipp==3.23.0


COPY entrypoint_demo.sh /entrypoint_demo.sh
RUN chmod +x /entrypoint_demo.sh
