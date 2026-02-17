# Dockerfile for Qwen3-TTS Streaming FastAPI Server
# Uses Miniconda for environment management

# CRITICAL: Ensure you are using the 'devel' tag, or the next error will be 'nvcc not found'
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev git wget build-essential ffmpeg sox ninja-build nginx && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /etc/nginx/sites-enabled/default

# FIX FOR TRITON/INDUCTOR: Create symlink for libcuda.so
RUN ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so || true

# Set up working directory
WORKDIR /app

# Install torch/torchaudio/torchvision with CUDA 12.6 wheels
RUN pip install --upgrade pip
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# CHANGE 1: Set CUDA_HOME explicitly to help torch find the compiler
ENV CUDA_HOME=/usr/local/cuda

# CHANGE 2: Added 'psutil' to the build dependencies
RUN pip install packaging ninja wheel setuptools psutil

# Install flash-attn as user does
RUN pip uninstall flash-attn -y || true
RUN pip install flash-attn --no-build-isolation --no-cache-dir -v

# Install python requirements (from pyproject.toml)
RUN pip install fastapi uvicorn pydantic soundfile gradio librosa sox onnxruntime einops transformers==4.57.3 accelerate==1.12.0

# Copy source code
COPY . /app

# Install local package in editable mode
RUN pip install -e .

# Use tini for signal handling and zombie reaping
ENV TINI_VERSION=v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Use supervisor to manage nginx + uvicorn workers
RUN pip install --no-cache supervisor
COPY supervisord.conf /etc/supervisord.conf

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create voices directory for cross-worker voice metadata
RUN mkdir -p /app/voices

# Default env vars
ENV TTS_NUM_WORKERS=2
ENV TTS_VOICE_META_DIR=/app/voices
ENV TTS_API_KEY=""

# Expose port
EXPOSE 8000

CMD ["/app/entrypoint.sh"]