#!/usr/bin/env bash
# One-command setup for zero-shot-insect-detection.
#
# Creates a Python virtual environment (.venv), installs the pinned
# dependencies from requirements.txt, and prepares the .env file from the
# .env.example template. Safe to re-run: an existing .venv and .env are
# kept as they are.
set -euo pipefail

# Work from the repo root, wherever the script is invoked from
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Python version check (3.11-3.13 required)
# ---------------------------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY=python
fi

if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11) or sys.version_info >= (3, 14))'; then
    echo "Error: Python 3.11-3.13 is required, found: $("$PY" --version 2>&1)"
    exit 1
fi
echo "Using $PY $("$PY" --version 2>&1)"

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------
if [ ! -d .venv ]; then
    echo "Creating virtual environment .venv ..."
    "$PY" -m venv .venv
else
    echo ".venv already exists, keeping it"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------------------
# Dependencies (pinned in requirements.txt)
# ---------------------------------------------------------------------------
echo "Installing dependencies, this can take a while ..."
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Jupyter kernel (so the notebooks run with this venv)
# ---------------------------------------------------------------------------
echo "Registering Jupyter kernel 'zero-shot-insect-detection' ..."
python -m ipykernel install --user --name zero-shot-insect-detection \
    --display-name "Zero-shot Insect Detection"

# ---------------------------------------------------------------------------
# Environment file
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
    echo "Creating .env from .env.example ..."
    cp .env.example .env
else
    echo ".env already exists, keeping it"
fi

# ---------------------------------------------------------------------------
# HF_TOKEN reminder (placeholder value is still in place)
# ---------------------------------------------------------------------------
if grep -q "HF_TOKEN=hf_token" .env 2>/dev/null; then
    echo
    echo "Note: HF_TOKEN in .env still has the placeholder value."
    echo "Edit .env and set HF_TOKEN=hf_... (get one at"
    echo "https://huggingface.co/settings/tokens). Required for SAM3 and DINOv3,"
    echo "not for GroundingDINO and OWL-ViT."
fi

# ---------------------------------------------------------------------------
# Environment doctor: print a readable summary of what is ready and what is
# missing, so problems are a checklist instead of an error traceback.
# ---------------------------------------------------------------------------
echo
echo "--- Environment check ---"

if python -c 'import torch' 2>/dev/null; then
    if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
        echo "[OK] torch $(python -c 'import torch; print(torch.__version__)') with CUDA"
    else
        echo "[OK] torch $(python -c 'import torch; print(torch.__version__)')"
        echo "[i]  No GPU detected: YOLO baselines are fine on CPU; the"
        echo "     zero-shot detectors (SAM3, GroundingDINO, OWL-ViT) will be"
        echo "     slow (see 'Hardware expectations' in README.md)"
    fi
else
    echo "[!!] torch failed to import, check the venv installation"
fi

if python -c 'from utils import models' 2>/dev/null; then
    echo "[OK] project modules import"
else
    echo "[!!] project modules failed to import"
fi

echo "Model cache (models-cache/):"
for f in \
    "yolov5_s_best_stark2023.pt      Stark2023" \
    "insect1201-bestF1-1280v5m6_bjerge2023.pt Bjerge2023" \
    "marin2025.pt                    Marin2025" \
    "flat_bug_M.pt                   FlatBug" \
    "insectsDCT_insects6ColorExtend.pt InsectsDCT"; do
    file="${f%% *}"
    what="${f#* }"
    if [ -f "models-cache/$file" ]; then
        echo "  [OK] $file ($what)"
    else
        echo "  [--] $file ($what) missing, see MODEL_WEIGHTS.md"
    fi
done
if ls models-cache/binary_classificator/*.joblib models-cache/binary_classificator/*.pth >/dev/null 2>&1; then
    echo "  [OK] binary classifier weights"
else
    echo "  [--] binary classifier weights missing, see MODEL_WEIGHTS.md"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo
echo "Setup complete. Next steps:"
echo "  - Run an example:  python examples/grounding_dino.py"
echo "  - Inspect results: open examples/eval.ipynb in VSCode and pick the"
echo "                     'zero-shot-insect-detection' kernel, or start"
echo "                     JupyterLab with: python -m jupyter lab"
echo "  - Weight downloads: YOLO baselines and the binary classifier are"
echo "                     manual, see MODEL_WEIGHTS.md"
