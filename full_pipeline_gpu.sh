#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
DATASET_NAME="inaturalist"          # must match your train.py arg
ROOT="./data"                       # your train.py uses root=./data
OUTDIR="./outputs/inat_gpu_run2"

# Training knobs (tweak as needed)
NUM_CLIENTS=8
ROUNDS=30
LOCAL_EPOCHS=3
CENTRAL_EPOCHS=10
BATCH_SIZE=64
LR=1e-4
ALPHA=0.5
AGG="fedavg"
MODEL="mobilenet_v3_small"
CLASSES=20

# If the box can't download torchvision weights, set this to 1 to flip pretrained=False
DISABLE_PRETRAINED=${DISABLE_PRETRAINED:-0}

# -----------------------------------------------
# 1) Check CUDA
# -----------------------------------------------
echo "[1/7] Checking CUDA..."
python3 - <<'PY'
import torch
print("   CUDA available:", torch.cuda.is_available())
PY

# -----------------------------------------------
# 2) Ensure iNat 2021 mini is present (via torchvision)
#    This creates:
#      ./data/train_mini/...
#      ./data/val/...
# -----------------------------------------------
echo "[2/7] Ensuring iNaturalist 2021_mini is on disk..."
python3 - <<PY
import os
from torchvision.datasets import INaturalist
from torchvision import transforms

root = "${ROOT}"
os.makedirs(root, exist_ok=True)

train_dir = os.path.join(root, "train_mini")
val_dir   = os.path.join(root, "val")

if os.path.isdir(train_dir) and os.path.isdir(val_dir):
    print(f"   Found train_mini and val in {root}")
else:
    print(f"   Downloading iNat 2021_mini into {root} (torchvision)...")
    tfm = transforms.Compose([transforms.ToTensor()])  # not used on disk, but required
    _ = INaturalist(root=root, version="2021_train_mini", transform=tfm, download=True)
    _ = INaturalist(root=root, version="2021_valid",      transform=tfm, download=True)
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        raise SystemExit("   Download finished but expected folders not found.")
    print(f"   Download complete at: {train_dir} and {val_dir}")

# Optionally mirror into ./data/inaturalist/* if you prefer that wrapper.
# Your datasets.py already falls back to ROOT if ./data/inaturalist doesn't exist, so this is NOT required.
PY

# -----------------------------------------------
# 3) Verify Plantae folders exist (naming contains 'plantae')
#    Your datasets.py filters by "plantae" in folder names and uses the penultimate token as class.
# -----------------------------------------------
echo "[3/7] Checking Plantae folders..."
SEARCH_TRAIN=""
if [ -d "${ROOT}/inaturalist/train_mini" ]; then
  SEARCH_TRAIN="${ROOT}/inaturalist/train_mini"
elif [ -d "${ROOT}/train_mini" ]; then
  SEARCH_TRAIN="${ROOT}/train_mini"
else
  echo "   train_mini not found after download."; exit 1
fi

n_plantae=$(find "${SEARCH_TRAIN}" -maxdepth 1 -type d -iname '*plantae*' | wc -l | tr -d ' ')
if [ "${n_plantae}" = "0" ]; then
  echo "   No folders containing 'plantae' found in: ${SEARCH_TRAIN}"
  echo "      Your dataset naming must contain 'plantae' for the current datasets.py filter."
  exit 1
fi
echo "   Found ${n_plantae} Plantae folders in ${SEARCH_TRAIN}"

# -----------------------------------------------
# 4) (Optional) Temporarily disable pretrained weights
# -----------------------------------------------
BACKUP_MADE=0
if [ "${DISABLE_PRETRAINED}" -eq 1 ]; then
  echo "[4/7] Patching train.py to set pretrained=False (temporary)..."
  cp train.py train.py.bak
  BACKUP_MADE=1
  sed -i.bak 's/pretrained=True/pretrained=False/g' train.py
else
  echo "[4/7] Using pretrained=True (may download torchvision weights)."
fi

# -----------------------------------------------
# 5) Train (federated + centralized inside train.py)
# -----------------------------------------------
echo "[5/7] Starting training..."
mkdir -p "${OUTDIR}"
python3 train.py \
  --dataset "${DATASET_NAME}" \
  --rounds "${ROUNDS}" \
  --local_epochs "${LOCAL_EPOCHS}" \
  --central_epochs "${CENTRAL_EPOCHS}" \
  --num_clients "${NUM_CLIENTS}" \
  --num_classes "${CLASSES}" \
  --batch_size "${BATCH_SIZE}" \
  --alpha "${ALPHA}" \
  --lr "${LR}" \
  --agg "${AGG}" \
  --model "${MODEL}" \
  --output_dir "${OUTDIR}"

echo "   Training complete."

# -----------------------------------------------
# 6) Evaluate
# -----------------------------------------------
echo "[6/7] Evaluating..."
python3 evaluate.py --experiment_dir "${OUTDIR}" --plot
echo "   Evaluation complete."

# -----------------------------------------------
# 7) Summary / cleanup
# -----------------------------------------------
echo "[7/7] Summary:"
if command -v jq >/dev/null 2>&1; then
  jq . < "${OUTDIR}/results.json" || cat "${OUTDIR}/results.json"
else
  cat "${OUTDIR}/results.json"
fi

if [ "${BACKUP_MADE}" -eq 1 ]; then
  mv -f train.py.bak train.py
fi

echo "Pipeline complete. Results in ${OUTDIR}"
