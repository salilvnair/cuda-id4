#!/usr/bin/env bash
# Ideogram 4 — CUDA/RunPod setup script.
# Clone the repo on your RunPod instance, then run:
#   chmod +x setup_and_run.sh && ./setup_and_run.sh
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== Ideogram 4 via diffusers — CUDA/RunPod Setup ==="
echo "Folder: $DIR"
echo ""

# ── 1. Verify NVIDIA GPU ───────────────────────────────────────────────────────
echo "[1/6] Checking GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "  ERROR: nvidia-smi not found. This script requires an NVIDIA GPU."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d '\r')
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' \r')
VRAM_GB=$(( VRAM_MB / 1024 ))
echo "  GPU: $GPU_NAME ($VRAM_GB GB VRAM)"

LOW_VRAM_FLAG=""
if [ "$VRAM_GB" -lt 20 ]; then
    echo "  ⚠  Less than 20 GB VRAM detected — enabling --low-vram (CPU offload) for smoke test."
    LOW_VRAM_FLAG="--low-vram"
fi

# ── 2. Find Python 3.9+ ───────────────────────────────────────────────────────
echo ""
echo "[2/6] Checking Python..."
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        major=$("$candidate" -c "import sys; print(sys.version_info.major)" 2>/dev/null || true)
        minor=$("$candidate" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || true)
        if [ -n "$major" ] && [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "  ERROR: Python 3.9+ is required but not found."
    echo "  On Ubuntu/Debian: sudo apt-get install python3.11"
    exit 1
fi
echo "  Found: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

# ── 3. Create virtual environment ─────────────────────────────────────────────
echo ""
echo "[3/6] Setting up virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    $PYTHON_BIN -m venv .venv
    echo "  Created .venv"
else
    echo "  .venv already exists — skipping creation"
fi

PYTHON="$DIR/.venv/bin/python"
PIP="$DIR/.venv/bin/pip"

# ── 4. Install PyTorch (CUDA-version-aware) ───────────────────────────────────
echo ""
echo "[4/6] Installing PyTorch + dependencies..."

# Detect CUDA version from nvcc or nvidia-smi driver info
CUDA_VERSION=""
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
fi
if [ -z "$CUDA_VERSION" ] && command -v nvidia-smi &>/dev/null; then
    # Driver version → infer max supported CUDA
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' \r')
    DRIVER_MAJOR=$(echo "$DRIVER_VER" | cut -d'.' -f1)
    if [ "$DRIVER_MAJOR" -ge 545 ]; then
        CUDA_VERSION="12.4"
    elif [ "$DRIVER_MAJOR" -ge 530 ]; then
        CUDA_VERSION="12.1"
    elif [ "$DRIVER_MAJOR" -ge 520 ]; then
        CUDA_VERSION="11.8"
    else
        CUDA_VERSION="11.8"
    fi
fi

# Pick the PyTorch wheel index URL
CUDA_SHORT=$(echo "$CUDA_VERSION" | tr -d '.')
case "$CUDA_SHORT" in
    124|125|126) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
    121|122|123) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    118|119|120) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
    *)           TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;  # default to latest stable
esac

echo "  CUDA version: ${CUDA_VERSION:-unknown} → using index $TORCH_INDEX"

# Check if torch is already installed and has CUDA
TORCH_OK=false
if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__)")
    echo "  PyTorch $TORCH_VER with CUDA already installed — skipping"
    TORCH_OK=true
fi

if [ "$TORCH_OK" = false ]; then
    echo "  Installing PyTorch from $TORCH_INDEX ..."
    $PIP install --upgrade pip -q
    $PIP install torch torchvision torchaudio --index-url "$TORCH_INDEX" -q
    # Verify
    if ! $PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
        echo ""
        echo "  ERROR: PyTorch installed but torch.cuda.is_available() is False."
        echo "  This usually means the CUDA toolkit version mismatch."
        echo "  Try running:  $PIP install torch --index-url https://download.pytorch.org/whl/cu118"
        exit 1
    fi
    TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__)")
    echo "  PyTorch $TORCH_VER installed with CUDA ✓"
fi

# Install remaining dependencies
$PIP install -r requirements.txt -q

# ideogram-ai/ideogram-4-fp8 uses Ideogram4Transformer2DModel which is not yet
# in a released diffusers pip package. Install from git main to get it.
echo "  Installing diffusers from git main (required for Ideogram4Transformer2DModel)..."
$PIP install --upgrade "git+https://github.com/huggingface/diffusers.git" -q
echo "  All dependencies installed ✓"

# ── 5. Check .env ─────────────────────────────────────────────────────────────
echo ""
echo "[5/6] Checking .env configuration..."

if [ ! -f ".env" ]; then
    cp example.env .env
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────────────────┐"
    echo "  │  .env created from example.env                                      │"
    echo "  │                                                                     │"
    echo "  │  To use --magic (recommended), add an LLM key:                      │"
    echo "  │    MAGIC_PROVIDER=deepseek  +  DEEPSEEK_API_KEY=your-key            │"
    echo "  │    MAGIC_PROVIDER=anthropic +  ANTHROPIC_API_KEY=your-key           │"
    echo "  │    MAGIC_PROVIDER=ollama    (no key — runs locally)                 │"
    echo "  │                                                                     │"
    echo "  │  To use RunPod persistent storage for model weights:                │"
    echo "  │    MODEL_PATH=/workspace/models/ideogram4                           │"
    echo "  │                                                                     │"
    echo "  │  Then re-run this script.                                           │"
    echo "  └─────────────────────────────────────────────────────────────────────┘"
    echo ""
    exit 0
fi

set -o allexport
# shellcheck disable=SC1091
source .env
set +o allexport

# If MODEL_PATH is set in .env, symlink it so the generator finds it
if [ -n "${MODEL_PATH:-}" ] && [ -d "$MODEL_PATH" ] && [ ! -e "./models/ideogram4" ]; then
    mkdir -p ./models
    ln -s "$MODEL_PATH" ./models/ideogram4
    echo "  Linked $MODEL_PATH → ./models/ideogram4"
fi

# Report optional config
EFFECTIVE_PROVIDER="${MAGIC_PROVIDER:-auto-detect}"
echo "  MAGIC_PROVIDER: $EFFECTIVE_PROVIDER"
if [ -n "${HF_TOKEN:-}" ]; then
    echo "  HF_TOKEN:       ${HF_TOKEN:0:8}... ✓"
fi
if [ -d "./models/ideogram4" ]; then
    SIZE_GB=$(du -sh ./models/ideogram4 2>/dev/null | cut -f1 || echo "?")
    echo "  Local model:    ./models/ideogram4 ($SIZE_GB)"
else
    echo "  Local model:    not cached — will download from HuggingFace on first run"
fi

# ── 6. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Running smoke test (TURBO preset — 12 steps, fastest)..."
echo "  Prompt: 'a storefront sign that says CUDA WORKS'"
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

$PYTHON ideogram4_generate.py \
    'a storefront sign that says CUDA WORKS' \
    --preset TURBO \
    --output outputs/test.png \
    $LOW_VRAM_FLAG

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                        Setup Complete  ✓                            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "GPU:    $GPU_NAME ($VRAM_GB GB VRAM)"
TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__)")
echo "Torch:  $TORCH_VER  |  CUDA: $($PYTHON -c "import torch; print(torch.version.cuda)")"
echo ""
echo "Generate images (plain text):"
echo "  $PYTHON ideogram4_generate.py 'your prompt'"
echo "  $PYTHON ideogram4_generate.py 'your prompt' --preset QUALITY"
echo ""
echo "Generate with magic prompt (structured JSON caption — best quality):"
echo "  $PYTHON ideogram4_generate.py --magic 'your prompt'"
echo "  $PYTHON ideogram4_generate.py --magic 'your prompt' --preset QUALITY"
echo "  $PYTHON ideogram4_generate.py --magic 'your prompt' --save-magic prompt.json"
echo ""
echo "Use a pre-built JSON caption:"
echo "  $PYTHON ideogram4_generate.py --json prompt.json"
echo ""
echo "Change aspect ratio:"
echo "  $PYTHON ideogram4_generate.py --magic 'your prompt' --aspect-ratio 16:9"
echo "  $PYTHON ideogram4_generate.py --magic 'your prompt' --aspect-ratio 9:16"
echo ""
echo "Low VRAM mode (CPU offload, for < 20 GB VRAM):"
echo "  $PYTHON ideogram4_generate.py 'your prompt' --low-vram"
echo ""
echo "Download model to persistent storage (do once, avoids re-download):"
echo "  $PYTHON save_model.py --save-dir /workspace/models/ideogram4"
echo "  # then set MODEL_PATH=/workspace/models/ideogram4 in .env"
echo ""
echo "Outputs → $DIR/outputs/"
