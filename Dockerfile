FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/efficientsam3

COPY requirements.txt /tmp/requirements.txt

# Torch binaries aligned with CUDA 12.6
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && \
    python3 -m pip install -r /tmp/requirements.txt

COPY . /workspace/efficientsam3

RUN python3 -m pip install -e .

# Conservative defaults for RTX 8GB / RAM 16GB setups.
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.8,expandable_segments:True \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

CMD ["bash"]
