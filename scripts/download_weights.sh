#!/usr/bin/env bash
# Download the weights that are not fetched automatically.
#
# Usage:
#   bash scripts/download_weights.sh            # everything below
#   bash scripts/download_weights.sh --yolo     # only the YOLO baselines
#   bash scripts/download_weights.sh --binary   # only the binary classifiers
#
# Existing files are skipped, so the script is safe to re-run. The links are
# documented in MODEL_WEIGHTS.md (gdown and wget must be on PATH; gdown is a
# project dependency, installed by scripts/setup.sh).
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root, wherever the script is invoked from

DOWNLOAD_YOLO=0
DOWNLOAD_BINARY=0
if [ "$#" -eq 0 ]; then
    DOWNLOAD_YOLO=1
    DOWNLOAD_BINARY=1
fi
while [ "$#" -gt 0 ]; do
    case "$1" in
        --yolo)   DOWNLOAD_YOLO=1 ;;
        --binary) DOWNLOAD_BINARY=1 ;;
        *) echo "Unknown option: $1 (use --yolo, --binary, or nothing for all)"; exit 1 ;;
    esac
    shift
done

mkdir -p models-cache
mkdir -p models-cache/binary_classificator

skip() {
    [ -f "$1" ] && echo "already present, skipping: $1" && return 0
    return 1
}

if [ "$DOWNLOAD_YOLO" -eq 1 ]; then
    echo "== YOLO baselines =="

    if skip models-cache/yolov5_s_best_stark2023.pt; then :; else
        echo "Downloading Stark2023 ..."
        wget -O models-cache/yolov5_s_best_stark2023.pt \
          https://raw.githubusercontent.com/stark-t/PAI/main/detectors/trained_weights/yolov5_s_best.pt
    fi

    if skip models-cache/insect1201-bestF1-1280v5m6_bjerge2023.pt; then :; else
        echo "Downloading Bjerge2023 (zip) ..."
        wget -O /tmp/YOLOv5models.zip \
          "https://zenodo.org/records/7395752/files/YOLOv5models.zip?download=1"
        unzip -o /tmp/YOLOv5models.zip -d /tmp/YOLOv5models_extract
        mv /tmp/YOLOv5models_extract/YOLOv5models/insect1201-bestF1-1280v5m6.pt \
          models-cache/insect1201-bestF1-1280v5m6_bjerge2023.pt
        rm -rf /tmp/YOLOv5models.zip /tmp/YOLOv5models_extract
    fi

    if skip models-cache/marin2025.pt; then :; else
        command -v gdown >/dev/null 2>&1 || {
            echo "gdown not found: activate the venv (bash setup.sh) or pip install gdown"
            exit 1
        }
        echo "Downloading Marin2025 (Google Drive) ..."
        gdown --folder -O /tmp/marin2025_dl \
          "https://drive.google.com/drive/folders/1X3buOu6bhYmx0hesPwmcU_FB693SCfHn"
        marin_pt=$(find /tmp/marin2025_dl -name "*.pt" -print -quit)
        [ -n "$marin_pt" ] || { echo "no .pt file found in the download"; exit 1; }
        mv "$marin_pt" models-cache/marin2025.pt
        rm -rf /tmp/marin2025_dl
    fi

    if skip models-cache/flat_bug_M.pt; then :; else
        echo "Downloading FlatBug ..."
        wget -O models-cache/flat_bug_M.pt \
          https://anon.erda.au.dk/share_redirect/Bb0CR1FHG6/models/flat_bug_M.pt
    fi

    if skip models-cache/insectsDCT_insects6ColorExtend.pt; then :; else
        echo "Downloading InsectsDCT ..."
        wget -O models-cache/insectsDCT_insects6ColorExtend.pt \
          https://raw.githubusercontent.com/kimbjerge/insectDCT/main/runs/detect/insects6ColorExtend/weights/best.pt
    fi

    # ArthropodDetector (edgaremy/arthropod-detector on the HF Hub). Two
    # training-data variants of the same YOLO11-large checkpoint: the default
    # "+flatbug" one, and the "mosaic33" alternative.
    if skip models-cache/yolo11l_ArthroNat+flatbug.pt; then :; else
        echo "Downloading Arthropod (ArthroNat+flatbug) ..."
        wget -O models-cache/yolo11l_ArthroNat+flatbug.pt \
          https://huggingface.co/edgaremy/arthropod-detector/resolve/main/yolo11l_ArthroNat+flatbug.pt
    fi

    if skip models-cache/yolo11l_ArthroNat_mosaic33.pt; then :; else
        echo "Downloading Arthropod (ArthroNat_mosaic33) ..."
        wget -O models-cache/yolo11l_ArthroNat_mosaic33.pt \
          https://huggingface.co/edgaremy/arthropod-detector/resolve/main/yolo11l_ArthroNat_mosaic33.pt
    fi
fi

if [ "$DOWNLOAD_BINARY" -eq 1 ]; then
    echo "== Binary classifier weights =="
    if ls models-cache/binary_classificator/*.joblib models-cache/binary_classificator/*.pth >/dev/null 2>&1; then
        echo "already present, skipping"
    else
        command -v gdown >/dev/null 2>&1 || {
            echo "gdown not found: activate the venv (bash setup.sh) or pip install gdown"
            exit 1
        }
        echo "Downloading binary classifiers (Google Drive) ..."
        gdown --folder -O models-cache/binary_classificator \
          "https://drive.google.com/drive/folders/1Q_SnMAfdvSJaJ1MSYEjf8f0bLwHWLLFd"
    fi
fi

echo
echo "Done. Missing files are reported by the environment check in scripts/setup.sh."
