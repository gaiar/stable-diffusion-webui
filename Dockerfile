# Intel OpenVINO Stable Diffusion WebUI
# Optimized for CPU-only operation
# Based on: https://github.com/openvinotoolkit/stable-diffusion-webui

FROM python:3.10-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/gaiar/stable-diffusion-webui"
LABEL org.opencontainers.image.description="Stable Diffusion WebUI with Intel OpenVINO acceleration for CPU"
LABEL org.opencontainers.image.licenses="AGPL-3.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash webui
WORKDIR /app

# Clone the WebUI from forked repo (contains our fixes)
RUN git clone --depth 1 https://github.com/gaiar/stable-diffusion-webui.git . \
    && chown -R webui:webui /app

# Clone required repositories from working mirrors
# Note: Stability-AI/stablediffusion is no longer available, using w-e-w fork
RUN mkdir -p repositories && \
    git clone --depth 1 https://github.com/w-e-w/stablediffusion.git \
        repositories/stable-diffusion-stability-ai && \
    git clone --depth 1 https://github.com/Stability-AI/generative-models.git \
        repositories/generative-models && \
    git clone --depth 1 https://github.com/crowsonkb/k-diffusion.git \
        repositories/k-diffusion && \
    git clone --depth 1 https://github.com/sczhou/CodeFormer.git \
        repositories/CodeFormer && \
    git clone --depth 1 https://github.com/salesforce/BLIP.git \
        repositories/BLIP

# Install PyTorch CPU-only (version 2.1.2 for torchvision compatibility)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install fixed versions of problematic packages
RUN pip install --no-cache-dir \
    huggingface_hub==0.21.4 \
    numpy==1.23.5 \
    Pillow==10.0.1

# Install requirements (torchsde already fixed to 0.2.6 in requirements_versions.txt)
RUN pip install --no-cache-dir -r requirements_versions.txt

# Install additional dependencies for repositories
RUN pip install --no-cache-dir \
    -r repositories/CodeFormer/requirements.txt || true

# Set ownership
RUN chown -R webui:webui /app

# Switch to non-root user
USER webui

# Environment variables for Intel CPU optimization
ENV PYTORCH_TRACING_MODE=TORCHFX
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV OPENBLAS_NUM_THREADS=8
ENV NUMEXPR_MAX_THREADS=8
ENV DNNL_PRIMITIVE_CACHE_CAPACITY=1024

# Disable git operations (repos already cloned)
ENV GIT_PULL=false

# Create directories for models and outputs
RUN mkdir -p models/Stable-diffusion outputs cache

# Expose WebUI port
EXPOSE 7860

# Volume mounts for persistence
VOLUME ["/app/models", "/app/outputs", "/app/cache"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:7860/ || exit 1

# Start WebUI with CPU-optimized settings
CMD ["python", "launch.py", \
    "--skip-torch-cuda-test", \
    "--precision", "full", \
    "--no-half", \
    "--listen", \
    "--api", \
    "--port", "7860"]
