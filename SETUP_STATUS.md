# Intel OpenVINO Stable Diffusion WebUI - Setup Status

**Setup Date:** December 2024
**Status:** Working

## System Configuration

- **CPU:** Intel i5-8279U (4 cores, 8 threads)
- **RAM:** 64GB
- **GPU:** None (CPU-only mode)
- **OS:** Linux (Debian-based)
- **Python:** 3.10.13 (conda-forge)

## Installation Source

Based on Intel's official OpenVINO fork:
- Repository: https://github.com/openvinotoolkit/stable-diffusion-webui
- Instructions: https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon
- WebUI Version: 1.6.0
- Commit: e5a634da06c62d72dbdc764b16c65ef3408aa588

## Performance Results

| Mode | Speed | Notes |
|------|-------|-------|
| PyTorch CPU (baseline) | ~12.13 s/step | Standard InvokeAI attention |
| OpenVINO (first run) | ~50s first step, then ~8.75 s/step | Model compilation overhead |
| OpenVINO (cached) | ~8.17 s/step | **~33% faster than baseline** |

Test settings: 512x512, 20 steps, Euler sampler, v1-5-pruned-emaonly model

## Setup Steps Performed

### 1. Clone Repository
```bash
git clone https://github.com/openvinotoolkit/stable-diffusion-webui.git sd-openvino
cd sd-openvino
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Copy Pre-existing Repositories
The Stability-AI/stablediffusion repository is no longer available (404). Repositories were copied from an existing Docker image:
```bash
# Copied from sd-auto:72 container:
# - repositories/stable-diffusion-stability-ai
# - repositories/generative-models
# - repositories/k-diffusion
# - repositories/CodeFormer
# - repositories/BLIP
```

### 4. Install Dependencies
```bash
# CPU-only PyTorch (version 2.1.2 required for compatibility)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Fix huggingface_hub compatibility
pip install huggingface_hub==0.21.4 diffusers==0.23.0 --force-reinstall

# Fix numpy/pillow versions
pip install numpy==1.23.5 Pillow==10.0.1 --force-reinstall
```

### 5. Run WebUI
```bash
./webui.sh
```

## Files Modified

### `webui-user.sh` - Intel CPU Configuration
```bash
#!/bin/bash
venv_dir="venv"
export GIT_PULL=false
export STABLE_DIFFUSION_COMMIT_HASH="cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf"
export CODEFORMER_COMMIT_HASH="c5b4593074ba6214284d6acd5f1719b6c5d739af"
export BLIP_COMMIT_HASH="48211a1594f1321b00f14c9f7a5b4813144b2fb9"
export K_DIFFUSION_COMMIT_HASH="ab527a9a6d347f364e3d185ba6d714e22d80cb3c"
export COMMANDLINE_ARGS="--skip-torch-cuda-test --precision full --no-half --listen --api"
export PYTORCH_TRACING_MODE="TORCHFX"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export TORCH_COMMAND="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
```

### `modules/launch_utils.py` - Skip Git Operations
Modified `git_clone()` function to skip git operations when repository directory already exists:
```python
def git_clone(url, dir, name, commithash=None):
    if os.path.exists(dir):
        print(f"{name} already exists at {dir}, skipping git operations")
        return
    # ... rest of function
```

### `requirements_versions.txt` - Version Fix
Changed `torchsde==0.2.5` to `torchsde==0.2.6` (pip 24.1+ compatibility)

## Errors Encountered and Fixes

### Error 1: Stability-AI Repository Not Found
```
fatal: repository 'https://github.com/Stability-AI/stablediffusion.git/' not found
```
**Fix:** Copied repositories from existing Docker image (sd-auto:72)

### Error 2: Git Fetch Still Failing
```
RuntimeError: Couldn't fetch Stable Diffusion
```
**Fix:** Patched `modules/launch_utils.py` to skip git operations for existing repos

### Error 3: torchsde Invalid Metadata
```
Please use pip<24.1 if you need to use this version (torchsde==0.2.5)
```
**Fix:** Updated `requirements_versions.txt` to use `torchsde==0.2.6`

### Error 4: torchvision.transforms.functional_tensor Not Found
```
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```
**Cause:** PyTorch 2.9.1 torchvision removed this module
**Fix:** Downgraded to PyTorch 2.1.2 with torchvision 0.16.2

### Error 5: huggingface_hub cached_download Not Found
```
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```
**Fix:** Installed `huggingface_hub==0.21.4`

## How to Use OpenVINO Acceleration

1. Start the WebUI: `./webui.sh`
2. Access at: http://localhost:7861
3. In txt2img tab, scroll to **Scripts** dropdown
4. Select **"accelerate with OpenVINO"**
5. Configure:
   - **Device:** CPU
   - **Model caching:** Enable for faster subsequent runs
6. Generate images as normal

## Key Package Versions

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.1.2+cpu | CPU-only build |
| torchvision | 0.16.2+cpu | Must match torch version |
| openvino | 2023.2.0 | Intel optimization |
| diffusers | 0.23.0 | OpenVINO compatible |
| huggingface_hub | 0.21.4 | Has cached_download |
| numpy | 1.23.5 | Required by basicsr |
| Pillow | 10.0.1 | Image processing |

## Model Location

Models stored in: `/home/gaiar/developer/sd-openvino/models/Stable-diffusion/`
- v1-5-pruned-emaonly.safetensors (tested, working)

## Comparison with Docker Setup

| Setup | Speed | Overhead |
|-------|-------|----------|
| Previous Docker (sd-auto:72) | ~9.4 s/step | Docker overhead |
| This setup (OpenVINO cached) | ~8.17 s/step | Native, no Docker |

**Result:** ~13% faster than Docker setup, ~33% faster than PyTorch baseline

## Docker Usage

### Quick Start with Docker

```bash
# Pull the pre-built image
docker pull ghcr.io/gaiar/sd-openvino:latest

# Create directories for models and outputs
mkdir -p models outputs cache

# Download a model (e.g., SD 1.5)
# Place your .safetensors or .ckpt files in ./models/Stable-diffusion/

# Run with docker-compose
docker-compose up -d

# Or run directly
docker run -d \
  -p 7860:7860 \
  -v ./models:/app/models \
  -v ./outputs:/app/outputs \
  -v ./cache:/app/cache \
  ghcr.io/gaiar/sd-openvino:latest
```

### Build Locally

```bash
# Clone the repository
git clone https://github.com/gaiar/stable-diffusion-webui.git
cd stable-diffusion-webui

# Build the image
docker build -t sd-openvino .

# Run
docker-compose up -d
```

### Docker Image Details

- **Base:** Python 3.10 slim (Debian Bookworm)
- **Size:** ~8GB (includes PyTorch, OpenVINO, all dependencies)
- **Repositories:** Pre-cloned from working mirrors
  - `w-e-w/stablediffusion` (mirror of unavailable Stability-AI/stablediffusion)
  - `Stability-AI/generative-models`
  - `crowsonkb/k-diffusion`
  - `sczhou/CodeFormer`
  - `salesforce/BLIP`

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| OMP_NUM_THREADS | 8 | OpenMP threads |
| MKL_NUM_THREADS | 8 | Intel MKL threads |
| PYTORCH_TRACING_MODE | TORCHFX | Required for OpenVINO |
| DNNL_PRIMITIVE_CACHE_CAPACITY | 1024 | oneDNN cache size |

## GitHub Repository

- **Fork:** https://github.com/gaiar/stable-diffusion-webui
- **Upstream:** https://github.com/openvinotoolkit/stable-diffusion-webui
- **Docker Image:** ghcr.io/gaiar/sd-openvino:latest

## Notes

- First generation with OpenVINO has ~50s compilation overhead
- Model cache persists across sessions for instant subsequent runs
- Threading optimizations tuned for 8-thread CPU
- No CUDA/GPU required - runs entirely on CPU
