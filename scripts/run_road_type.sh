#!/usr/bin/env bash
set -euo pipefail

# The road-type network with memory (docs/NEURAL_NETWORK.md, "The road type with memory"): the kind of road from the
# camera, one frame a second (town, rural, expressway, motorway, ramp, parking), with a GRU that remembers the seconds
# before, so that the end of a ramp on the motorway (where no sign ends the ramp's 60) and the entry into and exit
# from a town are seen. On one GPU, one step after the other:
#
#   1. checks the road types, the split and the whole pipeline on the CPU (nnslr train-road-type --dry-run);
#   2. a CUDA smoke test;
#   3. resizes every frame of the drives with road types to 256 x 144 once (data/derived/road_type_frames);
#   4. trains the network with memory (nnslr train-road-type --memory gru);
#   5. trains the same network without memory, to compare (--memory none);
#   6. compares them on test (comparison.txt: F1 per class, the delay of each kind of change, false switches per
#      hour, the accuracy in the first minute after a merge) and exports the network with memory as one ONNX step
#      with its contract (nnslr export-road-type).
#
# usage: scripts/run_road_type.sh NAME
#
# Writes only into $DATA_ROOT/runs/NAME/ (road-type.log, road-type/{dry-run,gru,frame,export}/, road-type/comparison.txt)
# and the frame cache. It never imports reviews, changes the road types, the model in use or the car, never publishes.
# Run it again with the same NAME to finish: a finished training is not repeated, an export older than its weights is
# redone.
#
# Environment: NNSLR_GPU_INDEX (default 0), NNSLR_GPU_VENV (.venv-gpu), NNSLR_ROAD_TYPE_EPOCHS (60),
# NNSLR_ROAD_TYPE_PATIENCE (15), NNSLR_ROAD_TYPE_BATCH (12 sequences), NNSLR_NUM_WORKERS (half the cores, 4 to 16),
# NNSLR_ALLOW_BUSY_GPU=1 to share a GPU that already has compute processes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
RUN_ID="${1:-}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
EPOCHS="${NNSLR_ROAD_TYPE_EPOCHS:-60}"
PATIENCE="${NNSLR_ROAD_TYPE_PATIENCE:-15}"
BATCH="${NNSLR_ROAD_TYPE_BATCH:-12}"
CORES="$(nproc 2>/dev/null || echo 8)"
WORKERS_DEFAULT=$(( CORES / 2 < 4 ? 4 : (CORES / 2 > 16 ? 16 : CORES / 2) ))
NUM_WORKERS="${NNSLR_NUM_WORKERS:-$WORKERS_DEFAULT}"

if [[ -z "$RUN_ID" || $# -ne 1 ]]; then
  echo "usage: $0 NAME" >&2
  echo "example: NNSLR_GPU_INDEX=0 $0 2026-10-29_1200-road-type" >&2
  exit 2
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: NAME contains unsupported characters" >&2
  exit 2
fi

for setting in EPOCHS PATIENCE BATCH NUM_WORKERS; do
  if [[ ! "${!setting}" =~ ^[0-9]+$ ]]; then
    echo "error: $setting must be a whole number: ${!setting}" >&2
    exit 2
  fi
done

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
  echo "error: selected GPU already has compute processes; refusing to train the road type" >&2
  echo "$BUSY" >&2
  echo "Set NNSLR_ALLOW_BUSY_GPU=1 only after intentionally accepting contention." >&2
  exit 3
fi

RUN_ROOT="$DATA_ROOT/runs/$RUN_ID"
OUT="$RUN_ROOT/road-type"
# A folder of another kind of run is never written into.
for part in detector reader export bundle experiment-b detector-sizes classifier signnet; do
  if [[ -e "$RUN_ROOT/$part" ]]; then
    echo "error: $RUN_ROOT is another kind of run (it has $part/): choose a new NAME" >&2
    exit 3
  fi
done

mkdir -p "$OUT"
# Everything printed from here on also goes to the run's own log (the panel reads its steps and epochs).
exec > >(tee -a "$RUN_ROOT/road-type.log") 2>&1

export NNSLR_DATA_ROOT="$DATA_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
NN="$VENV/bin/nnslr"
SETTINGS=(--epochs "$EPOCHS" --patience "$PATIENCE" --batch-size "$BATCH" --num-workers "$NUM_WORKERS")

echo "== Road type with memory =="
echo "run:          $RUN_ID"
echo "output:       $OUT"
echo "data root:    $DATA_ROOT"
echo "physical GPU: $GPU_INDEX"
echo "settings:     ${SETTINGS[*]}"
echo

echo "== 1/6 Check the road types, the split and the pipeline (CPU-only) =="
CUDA_VISIBLE_DEVICES="" "$NN" train-road-type --dry-run --run-dir "$OUT/dry-run" --num-workers "$NUM_WORKERS"

echo "== 2/6 Explicit CUDA smoke test =="
"$NN" env --gpu-smoke --gpu-device 0

echo "== 3/6 Resize the frames to 256x144 =="
"$VENV/bin/python" -c '
import sys
from nnslr_tools import road_type_data as data
dataset = data.load_dataset(sys.argv[1])
paths = data.build_frame_cache(sys.argv[1], dataset["frames"], workers=int(sys.argv[2]))
print(f"frame cache: {len(paths)} drives ready")
' "$DATA_ROOT" "$NUM_WORKERS"

train() {
  local memory="$1" folder="$2"
  if [[ -f "$OUT/$folder/result.json" ]]; then
    echo "skip trained: $OUT/$folder/result.json"
    return
  fi
  "$NN" train-road-type --memory "$memory" --run-dir "$OUT/$folder" "${SETTINGS[@]}"
}

echo "== 4/6 Train with memory (GRU) =="
train gru gru

echo "== 5/6 Train without memory =="
train none frame

echo "== 6/6 Compare and export the ONNX step =="
"$NN" compare-road-type --run-root "$RUN_ROOT"
ONNX="$OUT/export/road-type.onnx"
if [[ -f "$ONNX" && -f "$ONNX.contract.json" && "$ONNX" -nt "$OUT/gru/road-type-best.pt" ]]; then
  echo "skip exported: $ONNX"
else
  "$NN" export-road-type --checkpoint "$OUT/gru/road-type-best.pt" --output "$ONNX"
fi
echo
echo "Road type complete: $OUT"
echo "Nothing was imported, deployed or published, and nothing on the car changed."
