#!/bin/bash
#########################################################
# Intel OpenVINO configuration for CPU                  #
#########################################################

# Use the venv we created
venv_dir="venv"

# Disable git operations (repos already cloned)
export GIT_PULL=false

# Fixed git commits to prevent fetching from non-existent repos
export STABLE_DIFFUSION_COMMIT_HASH="cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf"
export CODEFORMER_COMMIT_HASH="c5b4593074ba6214284d6acd5f1719b6c5d739af"
export BLIP_COMMIT_HASH="48211a1594f1321b00f14c9f7a5b4813144b2fb9"
export K_DIFFUSION_COMMIT_HASH="ab527a9a6d347f364e3d185ba6d714e22d80cb3c"

# Commandline arguments for OpenVINO CPU mode
export COMMANDLINE_ARGS="--skip-torch-cuda-test --precision full --no-half --listen --api"

# PyTorch tracing mode for OpenVINO
export PYTORCH_TRACING_MODE="TORCHFX"

# Intel CPU threading optimizations
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# Intel oneDNN optimizations
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

# Install CPU-only PyTorch
export TORCH_COMMAND="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
