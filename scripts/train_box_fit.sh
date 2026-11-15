#!/usr/bin/env bash
set -euo pipefail

# The review page's Fit box (key C, nnslr_tools.box_fit) trained again on the boxes people drew so far, as a new
# version (#94): the candidates extracted on the CPU (OpenCV, a few minutes with 12 processes), the colour trees and the
# edge MLPs trained on the GPU (nnslr_tools.box_fit.gpu); then the box refiner (a small CNN,
# nnslr_tools.box_fit.refiner_train) on the GPU. Both with the PyTorch of .venv-gpu, on the card the panel gives
# (CUDA_VISIBLE_DEVICES): without a card the script stops, it never trains on the CPU.
#
# usage: scripts/train_box_fit.sh [--dry-run] [--no-refiner] [any other option of nnslr train-box-fit but --out]
#
# 1. `nnslr train-box-fit --dry-run`: how many boxes, cases and route groups; the script stops there when there are
#    fewer boxes than the training needs ("min_boxes").
# 2. `nnslr train-box-fit --out <version>`: the measure first (each group of routes fitted by models trained on the
#    others, --folds 4), then the two rankers trained on every box, written into the new version's folder, and a second
#    pair trained without the routes step 3 measures on, in its measure/ folder. The same boxes and seed give the same
#    files.
# 3. The box refiner, trained on the boxes of the frames people confirmed in the review (review.sqlite), the routes of
#    a quarter left out, written into the same folder. Then, on the full frames of those routes, the same rough boxes
#    fitted by this version and by the version in use (the one read at the start), each as the review page runs it,
#    each from its measure/ folder when it has one (rankers that never saw those routes). --no-refiner trains no
#    refiner: the version carries the refiner of the version in use (when it has one) and is measured the same way.
#
# The version is the folder $DATA_ROOT/runs/<YYYY-MM-DD_HHMM>-box-fit/ (colour.npz, edges.npz, refiner.onnx, the two
# summaries rankers.json and refiner.json, and box-fit.json, its record with the measure); it is made in a hidden
# folder beside it and appears whole, at the end, or not at all. Nothing in use is replaced: the version in use is
# chosen in Models › Review helpers › Fit box (models/box-fit.json), and the review page fits with it from its next
# fit, without a restart. Nothing is imported, committed or published.
#
# Environment: NNSLR_DATA_ROOT, NNSLR_NUM_WORKERS (processes extracting the candidates; default half the cores, 4 to
# 12), CUDA_VISIBLE_DEVICES (the card; the panel sets it).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
CORES="$(nproc 2>/dev/null || echo 8)"
WORKERS_DEFAULT=$(( CORES / 2 < 4 ? 4 : (CORES / 2 > 12 ? 12 : CORES / 2) ))
WORKERS="${NNSLR_NUM_WORKERS:-$WORKERS_DEFAULT}"
NNSLR="$ROOT_DIR/.venv/bin/nnslr"
PYTHON="$ROOT_DIR/.venv/bin/python"

GPU_PYTHON="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}/bin/python"
REFINER=1
args=()
for arg in "$@"; do
  if [ "$arg" = "--no-refiner" ]; then REFINER=0; else args+=("$arg"); fi
done
set -- "${args[@]+"${args[@]}"}"
STEPS=3

echo "1/$STEPS The boxes drawn by people (dry run)"
dry="$("$NNSLR" train-box-fit --data-root "$DATA_ROOT" --dry-run "$@")"
echo "$dry"
for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then
    if [ "$REFINER" = 1 ]; then
      echo "The boxes of the refiner (dry run)"
      if "$GPU_PYTHON" -c 'import torch, onnx, cv2' 2>/dev/null; then
        "$GPU_PYTHON" -m nnslr_tools.box_fit.refiner_train --data-root "$DATA_ROOT" --out "$DATA_ROOT/runs" --dry-run
      else
        echo "The PyTorch environment (.venv-gpu with torch, onnx and opencv) is missing: a real run needs it (scripts/setup_v100_training_env.sh installs it)."
      fi
    fi
    exit 0
  fi
done
if ! echo "$dry" | "$PYTHON" -c 'import json, sys; s = json.loads(sys.stdin.read().strip().splitlines()[-1]); sys.exit(s["boxes"] < s["min_boxes"])'; then
  echo "Too few boxes drawn by people to train the box fit: nothing changed." >&2
  exit 1
fi

if ! "$GPU_PYTHON" -c 'import sys, torch, onnx, cv2, onnxruntime; sys.exit(not torch.cuda.is_available())' 2>/dev/null; then
  echo "No V100 visible to the PyTorch of .venv-gpu (CUDA_VISIBLE_DEVICES, nvidia-smi; scripts/setup_v100_training_env.sh): Fit box trains on the GPU only. Nothing changed." >&2
  exit 1
fi

RUN="$(date +%Y-%m-%d_%H%M)-box-fit"
RUNS="$DATA_ROOT/runs"
PART="$RUNS/.$RUN.partial"
if [ -e "$RUNS/$RUN" ] || [ -e "$PART" ]; then
  echo "$RUNS/$RUN is already there: start the training again in a minute. Nothing changed." >&2
  exit 1
fi
mkdir -p "$PART"
# The hidden folder goes with the script when it stops before the end: a version appears whole or not at all.
trap 'rm -rf "$PART"' EXIT
IN_USE_LINE="$("$PYTHON" -m nnslr_tools.box_fit.versions in-use --data-root "$DATA_ROOT")"
IFS=$'\t' read -r IN_USE IN_USE_FOLDER <<< "$IN_USE_LINE"
echo "The new version: box_fit:$RUN; the version in use: $IN_USE"

echo "2/$STEPS The two rankers: measure on routes left out, then train on every box (candidates: $WORKERS processes; trees and MLPs: the GPU)"
"$GPU_PYTHON" -m nnslr_tools.cli train-box-fit --data-root "$DATA_ROOT" --workers "$WORKERS" --device cuda "$@" \
  --out "$PART" > "$PART/rankers.json"
cat "$PART/rankers.json"

REFINER_FROM=()
if [ "$REFINER" = 1 ]; then
  echo "3/$STEPS The box refiner: train on the GPU, then this version and the version in use on the same boxes of routes left out ($WORKERS processes)"
  "$GPU_PYTHON" -m nnslr_tools.box_fit.refiner_train --data-root "$DATA_ROOT" --out "$PART" \
    --workers "$WORKERS" --device cuda --in-use "$IN_USE" > "$PART/refiner.json"
else
  if [ -f "$IN_USE_FOLDER/refiner.onnx" ]; then
    cp -p "$IN_USE_FOLDER/refiner.onnx" "$PART/refiner.onnx"
    REFINER_FROM=(--refiner-from "$IN_USE")
    echo "3/$STEPS No refiner trained: this version carries the refiner of $IN_USE; this version and the version in use on the same boxes of routes left out"
  else
    echo "3/$STEPS No refiner trained (the version in use has none): this version and the version in use on the same boxes of routes left out"
  fi
  "$GPU_PYTHON" -m nnslr_tools.box_fit.refiner_train --data-root "$DATA_ROOT" --out "$PART" \
    --workers "$WORKERS" --measure-only --in-use "$IN_USE" > "$PART/refiner.json"
fi
cat "$PART/refiner.json"
"$PYTHON" -m nnslr_tools.box_fit.versions record --folder "$PART" --run "$RUN" --rankers "$PART/rankers.json" \
  --refiner "$PART/refiner.json" "${REFINER_FROM[@]+"${REFINER_FROM[@]}"}" > /dev/null
mv -T "$PART" "$RUNS/$RUN"
trap - EXIT
# The version in use now: someone may have chosen another one while this training ran.
IN_USE_NOW="$("$PYTHON" -m nnslr_tools.box_fit.versions in-use --data-root "$DATA_ROOT" | cut -f1)"
echo "Version box_fit:$RUN saved in $RUNS/$RUN. Nothing replaced: the review page keeps $IN_USE_NOW until you choose this version in Models › Review helpers › Fit box."
