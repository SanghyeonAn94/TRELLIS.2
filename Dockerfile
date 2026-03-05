FROM docker.io/nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="10.0"
ENV FORCE_CUDA=1
ENV MAX_JOBS=32

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git ninja-build \
    libjpeg-dev libpng-dev libwebp-dev libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    libopenexr-dev \
    wget curl ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && python -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.10.0 + CUDA 13.0 (B200 sm_100 지원)
RUN pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130

# 기본 의존성 (setup.sh --basic)
RUN pip install \
    imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless ninja trimesh transformers \
    tensorboard pandas lpips zstandard \
    kornia timm

# utils3d (특정 커밋)
RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# pillow-simd (성능 최적화)
RUN pip uninstall -y pillow \
    && pip install pillow-simd

# flash-attn (소스 빌드)
RUN pip install flash-attn --no-build-isolation

# nvdiffrast v0.4.0 (소스 빌드)
RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast \
    && pip install /tmp/nvdiffrast --no-build-isolation \
    && rm -rf /tmp/nvdiffrast

# nvdiffrec renderutils 브랜치 (소스 빌드, v2 신규)
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/nvdiffrec \
    && pip install /tmp/nvdiffrec --no-build-isolation \
    && rm -rf /tmp/nvdiffrec

# CuMesh (소스 빌드, v2 신규)
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh --recursive \
    && pip install /tmp/CuMesh --no-build-isolation \
    && rm -rf /tmp/CuMesh

# FlexGEMM (소스 빌드, v2 신규)
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM --recursive \
    && pip install /tmp/FlexGEMM --no-build-isolation \
    && rm -rf /tmp/FlexGEMM

# o-voxel (워크스페이스에서 빌드 — 소스는 볼륨 마운트 시 사용)
COPY o-voxel /tmp/o-voxel
RUN pip install /tmp/o-voxel --no-build-isolation \
    && rm -rf /tmp/o-voxel

# HuggingFace CLI
RUN pip install huggingface_hub[cli]

WORKDIR /workspace
