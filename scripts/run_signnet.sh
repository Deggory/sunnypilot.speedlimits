#!/usr/bin/env bash
set -euo pipefail

# The portal's "Train the cutter": one SignNet, the cutter the comma runs (nnslr_tools.signnet), trained with the
# car's setup so the car can load it as it is: the top band of the frame at 896 x 288, a grid of 8 pixels, the
# fuse convolution 1 x 1, channels 16,24,32,48 (the car's vision.py only runs that input and that grid; the run on
# the car, 2026-10-28_1028-signnet-twothirds-s8-f1, was trained so). The best epoch is the one with the best
# validation recall in the 5 best boxes of a frame (the car keeps 5); that epoch is then measured on validation
# and test (result.json).
#
# usage: scripts/run_signnet.sh NAME
#
# Writes only into $DATA_ROOT/runs/NAME/: signnet-best.pt, result.json (nnslr train-signnet --run-dir) and
# signnet.log (everything this script prints). It never imports reviews, builds splits, kills GPU processes,
# exports, deploys or puts the SignNet in use (Models > Use this one). A NAME holding another kind of run or a
# finished SignNet is refused; a NAME with signnet-best.pt but no result.json (a training stopped half way: SignNet
# has no resume) is trained again from scratch. The log of a NAME tried before is moved to signnet.log.1 (replacing
# an older one), so the new run's epochs never mix with the old ones.
#
# Environment: NNSLR_GPU_INDEX (default 0), NNSLR_DATASET_KIND (gold), NNSLR_GPU_VENV (.venv-gpu),
# NNSLR_CUTTER_FAMILIES (all: the speed signs and the other traffic signs; speed: the speed signs only),
# NNSLR_SIGNNET_EPOCHS (40), NNSLR_SIGNNET_BATCH (16), NNSLR_SIGNNET_LEARNING_RATE (0.003), NNSLR_NUM_WORKERS (half the
# cores, 4 to 16) (the portal sets those changed in Train › Cutter › Settings, nnslr_tools.train_settings),
# NNSLR_CUTTER_INIT_FROM (a finished SignNet's signnet-best.pt to go on from, instead of random weights: Train › Cutter
# › Start from), NNSLR_ALLOW_BUSY_GPU=1 to share a GPU that already has compute processes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
RUN_ID="${1:-}"
DATASET_KIND="${NNSLR_DATASET_KIND:-gold}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
FAMILIES="${NNSLR_CUTTER_FAMILIES:-all}"
EPOCHS="${NNSLR_SIGNNET_EPOCHS:-40}"
BATCH="${NNSLR_SIGNNET_BATCH:-16}"
# The processes reading the frames: half the cores, 4 to 16 (nnslr_tools.training.DEFAULT_WORKERS).
CORES="$(nproc 2>/dev/null || echo 8)"
WORKERS_DEFAULT=$(( CORES / 2 < 4 ? 4 : (CORES / 2 > 16 ? 16 : CORES / 2) ))
NUM_WORKERS="${NNSLR_NUM_WORKERS:-$WORKERS_DEFAULT}"
# The car's setup (nnslr_tools.signnet CAR_BAND, CAR_STRIDE, CAR_FUSE_KERNEL, CAR_WIDTHS; the comma's vision.py).
BAND_WIDTH=896
BAND_HEIGHT=288
STRIDE=8
FUSE_KERNEL=1
WIDTHS=16,24,32,48
LEARNING_RATE="${NNSLR_SIGNNET_LEARNING_RATE:-0.003}"
INIT_FROM="${NNSLR_CUTTER_INIT_FROM:-}"

if [[ -z "$RUN_ID" || $# -gt 1 ]]; then
  echo "usage: $0 NAME" >&2
  echo "example: NNSLR_GPU_INDEX=0 $0 2026-10-30_1200-cutter" >&2
  exit 2
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: NAME contains unsupported characters" >&2
  exit 2
fi

if [[ "$DATASET_KIND" != "gold" && "$DATASET_KIND" != "training-candidate" ]]; then
  echo "error: NNSLR_DATASET_KIND must be gold or training-candidate" >&2
  exit 2
fi

if [[ "$FAMILIES" != "all" && "$FAMILIES" != "speed" ]]; then
  echo "error: NNSLR_CUTTER_FAMILIES must be all or speed" >&2
  exit 2
fi

for setting in EPOCHS BATCH NUM_WORKERS; do
  if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: $setting must be a whole number above 0: ${!setting}" >&2
    exit 2
  fi
done

# A positive number: 0.003, 3e-3, 1e-05.
if [[ ! "$LEARNING_RATE" =~ ^([0-9]*\.)?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
  echo "error: NNSLR_SIGNNET_LEARNING_RATE must be a number: $LEARNING_RATE" >&2
  exit 2
fi

if [[ -n "$INIT_FROM" && ! -f "$INIT_FROM" ]]; then
  echo "error: NNSLR_CUTTER_INIT_FROM is not a file: $INIT_FROM" >&2
  exit 2
fi

if [[ ! -x "$VENV/bin/nnslr" || ! -x "$VENV/bin/python" ]]; then
  echo "error: GPU environment missing: $VENV/bin/nnslr" >&2
  echo "run: bash scripts/setup_v100_training_env.sh" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found" >&2
  exit 2
fi

if ! nvidia-smi -i "$GPU_INDEX" >/dev/null 2>&1; then
  echo "error: invalid/unavailable GPU index: $GPU_INDEX" >&2
  exit 2
fi

BUSY="$(
  nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "$BUSY" && "${NNSLR_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "error: selected GPU already has compute processes; refusing to train SignNet" >&2
  echo "$BUSY" >&2
  echo "Set NNSLR_ALLOW_BUSY_GPU=1 only after intentionally accepting contention." >&2
  exit 3
fi

RUN_ROOT="$DATA_ROOT/runs/$RUN_ID"
# A folder of another kind of run (the training pipeline, experiment B, the detector sizes, a classifier, the road
# type) is never written into.
for part in detector reader export bundle evaluation experiment-b detector-sizes classifier pretrain negatives.jsonl \
            classifier.log road-type road-type.log; do
  if [[ -e "$RUN_ROOT/$part" ]]; then
    echo "error: $RUN_ROOT is another kind of run (it has $part): choose a new NAME" >&2
    exit 3
  fi
done
if [[ -f "$RUN_ROOT/result.json" ]]; then
  echo "error: $RUN_ROOT already holds a finished SignNet (result.json): choose a new NAME" >&2
  exit 3
fi
RESTART=0
if [[ -e "$RUN_ROOT/signnet-best.pt" ]]; then
  RESTART=1
fi

mkdir -p "$RUN_ROOT"
# A log of this NAME tried before: kept aside as signnet.log.1, the new one starts empty (the panel and Results
# read the last line of each epoch, so old epochs would show as the new run's).
OLD_LOG=0
if [[ -e "$RUN_ROOT/signnet.log" ]]; then
  mv -f "$RUN_ROOT/signnet.log" "$RUN_ROOT/signnet.log.1"
  OLD_LOG=1
fi
# Everything printed from here on also goes to the run's own log (the Evaluate tab reads its epochs).
exec > >(tee -a "$RUN_ROOT/signnet.log") 2>&1

export NNSLR_DATA_ROOT="$DATA_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
# Python writes each line as it comes, not when its 8 KB buffer fills: the epochs reach signnet.log through tee
# while it trains (nnslr train-signnet flushes them too).
export PYTHONUNBUFFERED=1

SETTINGS=(--families "$FAMILIES" --band-width "$BAND_WIDTH" --stride "$STRIDE" --fuse-kernel "$FUSE_KERNEL"
          --widths "$WIDTHS" --epochs "$EPOCHS" --batch "$BATCH" --learning-rate "$LEARNING_RATE")
# Where it starts: random weights, or the weights of a finished SignNet it goes on from.
if [[ -n "$INIT_FROM" ]]; then
  SETTINGS+=(--init-from "$INIT_FROM")
fi
NN="$VENV/bin/nnslr"

echo "== scripts/run_signnet.sh $RUN_ID"
echo "run:          $RUN_ID"
echo "band:         ${BAND_WIDTH}x${BAND_HEIGHT}"
echo "settings:     ${SETTINGS[*]}"
echo "workers:      $NUM_WORKERS"
echo "output:       $RUN_ROOT"
echo "data root:    $DATA_ROOT"
echo "dataset kind: $DATASET_KIND"
echo "physical GPU: $GPU_INDEX"
echo "boxes:        $FAMILIES (all: the speed signs and the other traffic signs; speed: the speed signs only)"
echo "best epoch:   validation recall in the 5 best boxes of a frame (the car keeps 5)"
echo
if [[ "$RESTART" == "1" ]]; then
  echo "restart from scratch: $RUN_ROOT/signnet-best.pt has no result.json (a training stopped half way; SignNet has no resume)"
fi
if [[ "$OLD_LOG" == "1" ]]; then
  echo "the log of the earlier try: $RUN_ROOT/signnet.log.1"
fi

echo "== 1/3 Check the training population (CPU-only) =="
CUDA_VISIBLE_DEVICES= "$NN" train-signnet --dry-run --run-dir "$RUN_ROOT" --dataset-kind "$DATASET_KIND" "${SETTINGS[@]}"

echo "== 2/3 Explicit CUDA smoke test =="
# CUDA_VISIBLE_DEVICES maps the selected physical GPU to logical cuda:0.
"$NN" env --gpu-smoke --gpu-device 0

echo "== 3/3 Train SignNet =="
# The epoch lines ("signnet epoch N/M train_loss ... val_recall ... val_false_per_frame ... 87s"), the measure of the
# best epoch and "SignNet complete: <run>" go to the terminal and to signnet.log.
"$NN" train-signnet --run-dir "$RUN_ROOT" --dataset-kind "$DATASET_KIND" "${SETTINGS[@]}" --workers "$NUM_WORKERS"

echo
echo "Nothing was imported, exported, deployed or put in use (Models > Use this one)."
