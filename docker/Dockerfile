# Dockerfile for Single-Pass UAV 3D Reconstruction System
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# System dependencies: FFmpeg, OpenGL, GLX for Open3D and OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy requirements and install
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

EXPOSE 8000 3000

CMD ["python3.11", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
