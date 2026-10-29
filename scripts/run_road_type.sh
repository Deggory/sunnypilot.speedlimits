#!/usr/bin/env bash
set -euo pipefail

# The road-type network with memory (docs/NEURAL_NETWORK.md, "The road type with memory"): the kind of road from the
# camera, one frame a second (town, rural, expressway, motorway, ramp, parking), with a GRU that remembers the seconds
# before and how long ago a motorway sign was seen, so that the 60 of a ramp ends where the ramp merges into the
# motorway (no sign ends it there: the comma's rule, vision.SpeedLimitMemory) and the entry into and exit from a town
# are seen. Its junction head says when the car is inside an intersection and when it has just left one: in Poland
# (and in Switzerland by OSStr art. 16 cpv. 2) a speed sign holds at most to the end of the next intersection, and
# Poland has no end sign.
# On one GPU, one step after the other:
#
#   1. checks the road types, the split and the whole pipeline on the CPU (nnslr train-road-type --dry-run);
#   2. a CUDA smoke test;
#   3. finds the intersections every drive passed from its GPS and the map (nnslr junctions --all,
#      data/derived/junctions: the labels of the junction head, a person's corrections over them) and resizes every
#      frame of the drives with road types to 256 x 144 once (data/derived/road_type_frames);
#   4. reads the motorway signs of those drives as the comma does (nnslr road-type-signs: the car's vision.py,
#      SignNet and classifier from the sunnypilot clone, read only, 5 frames a second of fcamera.hevc, on the CPU;
#      data/derived/road_type_signs): the comma's rule measured with the signs the car confirms today;
#   5. trains the network with memory and the sign input (nnslr train-road-type --memory gru; the sign input is the
#      seconds since a motorway sign, placed at the start of every entry ramp of the map: --sign-source map);
#   6. the same without the sign input, to compare (--no-sign);
#   7. the same with the sign input but without memory, to compare (--memory none);
#   8. compares them (comparison.txt: the comma's rule on every merge of validation and test, when the ramp's limit
#      ends; F1 per class and the ramp's confusion; the changes of road type; false switches per hour; the first
#      minute after a merge; the junctions: recall of "passed" within 2 and 4 s, false "passed" per hour and per
#      10 km in town and outside, the intersections the car needed), with an earlier version too when
#      NNSLR_ROAD_TYPE_BASELINE names its checkpoint, and exports the network with memory and the sign as one ONNX
#      step with its contract (nnslr export-road-type).
# Every training has the junction head when step 3 found the intersections (NNSLR_ROAD_TYPE_JUNCTIONS=0: without it,
# as the second version).
#
# usage: scripts/run_road_type.sh NAME
#
# Writes only into $DATA_ROOT/runs/NAME/ (road-type.log, road-type/{dry-run,gru,gru-nosign,frame,baseline,export}/,
# road-type/comparison.txt), the junction labels, the frame cache and the sign readings. It never imports reviews, changes the road types,
# the model in use or the car, never publishes. Run it again with the same NAME to finish: a finished training is not
# repeated, the signs already read are not read again, an export older than its weights is redone.
#
# Environment: NNSLR_GPU_INDEX (default 0), NNSLR_GPU_VENV (.venv-gpu), NNSLR_ROAD_TYPE_EPOCHS (60),
# NNSLR_ROAD_TYPE_PATIENCE (15), NNSLR_ROAD_TYPE_BATCH (12 sequences), NNSLR_NUM_WORKERS (half the cores, 4 to 16),
# NNSLR_SIGN_WORKERS (processes reading the signs: the cores but two), NNSLR_SUNNYPILOT_CLONE (/root/sunnypilot),
# NNSLR_ROAD_TYPE_BASELINE (an earlier road-type-best.pt to compare with), NNSLR_ALLOW_BUSY_GPU=1 to share a GPU
# that already has compute processes.

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
SIGN_WORKERS="${NNSLR_SIGN_WORKERS:-$(( CORES > 3 ? CORES - 2 : 1 ))}"
BASELINE="${NNSLR_ROAD_TYPE_BASELINE:-}"
JUNCTIONS="${NNSLR_ROAD_TYPE_JUNCTIONS:-1}"

if [[ -z "$RUN_ID" || $# -ne 1 ]]; then
  echo "usage: $0 NAME" >&2
  echo "example: NNSLR_GPU_INDEX=0 $0 2026-10-29_1200-road-type" >&2
  exit 2
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: NAME contains unsupported characters" >&2
  exit 2
fi

if [[ -n "$BASELINE" && ! -f "$BASELINE" ]]; then
  echo "error: NNSLR_ROAD_TYPE_BASELINE is not a file: $BASELINE" >&2
  exit 2
fi

for setting in EPOCHS PATIENCE BATCH NUM_WORKERS SIGN_WORKERS; do
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
if [[ "$JUNCTIONS" == "0" ]]; then
  SETTINGS+=(--no-junctions)
fi

echo "== Road type with memory =="
echo "run:          $RUN_ID"
echo "output:       $OUT"
echo "data root:    $DATA_ROOT"
echo "physical GPU: $GPU_INDEX"
echo "settings:     ${SETTINGS[*]}"
echo "baseline:     ${BASELINE:-none}"
echo "junctions:    $([[ "$JUNCTIONS" == "0" ]] && echo "no head" || echo "with the junction head")"
echo

echo "== 1/8 Check the road types, the split and the pipeline (CPU-only) =="
CUDA_VISIBLE_DEVICES="" "$NN" train-road-type --dry-run --run-dir "$OUT/dry-run" --num-workers "$NUM_WORKERS"

echo "== 2/8 Explicit CUDA smoke test =="
"$NN" env --gpu-smoke --gpu-device 0

echo "== 3/8 Find the junctions, resize the frames to 256x144 =="
if [[ "$JUNCTIONS" != "0" ]]; then
  "$NN" junctions --all --data-root "$DATA_ROOT"
fi
"$VENV/bin/python" -c '
import sys
from nnslr_tools import road_type_data as data
dataset = data.load_dataset(sys.argv[1])
paths = data.build_frame_cache(sys.argv[1], dataset["frames"], workers=int(sys.argv[2]))
print(f"frame cache: {len(paths)} drives ready")
' "$DATA_ROOT" "$NUM_WORKERS"

echo "== 4/8 Read the motorway signs as the car does (CPU) =="
"$NN" road-type-signs --num-workers "$SIGN_WORKERS"

train() {
  local memory="$1" folder="$2"
  shift 2
  if [[ -f "$OUT/$folder/result.json" ]]; then
    echo "skip trained: $OUT/$folder/result.json"
    return
  fi
  "$NN" train-road-type --memory "$memory" --run-dir "$OUT/$folder" "${SETTINGS[@]}" "$@"
}

echo "== 5/8 Train with memory and the sign (GRU) =="
train gru gru

echo "== 6/8 Train with memory, without the sign =="
train gru gru-nosign --no-sign

echo "== 7/8 Train without memory =="
train none frame

echo "== 8/8 Compare and export the ONNX step =="
if [[ -n "$BASELINE" ]]; then
  "$NN" release-road-type --checkpoint "$BASELINE" --output "$OUT/baseline/release.json"
fi
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
