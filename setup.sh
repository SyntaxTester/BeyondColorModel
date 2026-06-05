#!/bin/bash
# BeyondColor  One-shot setup script
# Run this ONCE before first use. Downloads all weights.
# Usage: bash setup.sh [--cpu-only]

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${GREEN}BeyondColor Setup${NC}"
echo "================================"

CPU_ONLY=false
[[ "$1" == "--cpu-only" ]] && CPU_ONLY=true

# ── 1. Create weights dir ─────────────────────────────────────────────────────
mkdir -p weights

# ── 2. Python deps ────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[1/5] Installing Python dependencies...${NC}"
pip install Pillow numpy opencv-python-headless PyMuPDF \
    fastapi uvicorn python-multipart pydantic -q

# ── 3. PyTorch ────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[2/5] Installing PyTorch...${NC}"

if [ "$CPU_ONLY" = true ]; then
    echo "   CPU-only mode"
    pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu -q
else
    # Detect CUDA
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' | cut -d. -f1,2)
        echo "   CUDA $CUDA_VER detected"
        if [[ "$CUDA_VER" == "12."* ]]; then
            pip install torch torchvision \
                --index-url https://download.pytorch.org/whl/cu121 -q
        else
            pip install torch torchvision \
                --index-url https://download.pytorch.org/whl/cu118 -q
        fi
    else
        echo "   No GPU found, installing CPU PyTorch"
        pip install torch torchvision \
            --index-url https://download.pytorch.org/whl/cpu -q
        CPU_ONLY=true
    fi
fi

# ── 4. SAM ────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[3/5] Installing SAM...${NC}"

if [ "$CPU_ONLY" = true ]; then
    echo "   MobileSAM (CPU-optimized, ~40 MB)"
    pip install mobile-sam -q

    if [ ! -f "weights/mobile_sam.pt" ]; then
        echo "   Downloading MobileSAM weights..."
        wget -q --show-progress \
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt" \
            -O weights/mobile_sam.pt
    else
        echo "   weights/mobile_sam.pt already exists, skipping"
    fi
else
    echo "   SAM vit_b (GPU, ~375 MB)"
    pip install git+https://github.com/facebookresearch/segment-anything.git -q

    if [ ! -f "weights/sam_vit_b_01ec64.pth" ]; then
        echo "   Downloading SAM vit_b weights..."
        wget -q --show-progress \
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
            -O weights/sam_vit_b_01ec64.pth
    else
        echo "   weights/sam_vit_b_01ec64.pth already exists, skipping"
    fi
fi

# ── 5. PubLayNet (Detectron2 + layoutparser) ──────────────────────────────────
echo -e "\n${YELLOW}[4/5] Installing PubLayNet (layoutparser + Detectron2)...${NC}"

DETECTRON_INSTALLED=false
if [ "$CPU_ONLY" = true ]; then
    pip install 'git+https://github.com/facebookresearch/detectron2.git' -q \
        && DETECTRON_INSTALLED=true || true
else
    if [[ "$CUDA_VER" == "12."* ]]; then
        pip install detectron2 \
            -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.1/index.html -q \
            && DETECTRON_INSTALLED=true || true
    else
        pip install detectron2 \
            -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html -q \
            && DETECTRON_INSTALLED=true || true
    fi
fi

if [ "$DETECTRON_INSTALLED" = true ]; then
    pip install "layoutparser[layoutmodels]" -q
    echo "   PubLayNet ready (weights auto-download on first use)"
else
    echo -e "  ${YELLOW}⚠ Detectron2 install failed  will use OpenCV fallback${NC}"
    echo "    This is OK for MVP. Layout detection works via contours."
    pip install layoutparser -q 2>/dev/null || true
fi

# ── 6. Verify ─────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[5/5] Verifying installation...${NC}"

python - <<'PYEOF'
import sys, importlib

checks = {
    "torch":                  "PyTorch",
    "PIL":                    "Pillow",
    "numpy":                  "NumPy",
    "cv2":                    "OpenCV",
    "fitz":                   "PyMuPDF",
    "fastapi":                "FastAPI",
}
optional = {
    "mobile_sam":             "MobileSAM",
    "segment_anything":       "SAM (segment-anything)",
    "layoutparser":           "layoutparser",
    "detectron2":             "Detectron2",
}

all_ok = True
for mod, name in checks.items():
    try:
        importlib.import_module(mod)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name}  ← MISSING")
        all_ok = False

print()
for mod, name in optional.items():
    try:
        importlib.import_module(mod)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ~ {name}  (optional, fallback available)")

import torch
print(f"\n  Device: {'CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

from pathlib import Path
weights = list(Path("weights").glob("*.pt")) + list(Path("weights").glob("*.pth"))
print(f"  Weights found: {[w.name for w in weights]}")

if not all_ok:
    sys.exit(1)
PYEOF

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Run API server:"
echo "  uvicorn api_integration:app --host 0.0.0.0 --port 8001"
echo ""
echo "Test single image:"
echo "  python pipeline.py input.png output.png"
echo ""
echo "Test single PDF:"
echo "  python pipeline.py input.pdf output.pdf"
echo ""
echo "Docker (recommended for production):"
echo "  docker-compose up --build"
