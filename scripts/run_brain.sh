#!/usr/bin/env bash
set -euo pipefail

# The brain (docs/NEURAL_NETWORK.md): the network that says the speed to hold, four times a second, from the picture and the
# car's own signals, learnt from every dashcam drive with its logs read, each whole: the driver's own speed, corrected
# by the blocks of Speed review, with the roadworks and the kinds of road marked there (a person or a vehicle that
# crosses our path is one of those kinds since 17 Nov 2026, #112). On one
# GPU (the reading of step 3 also on the free ones), one step after the other:
#
#   1. checks the labels, the split and the whole training on the CPU (nnslr train-brain --dry-run; its log says the
#      dashcam drives and hours, and it stops while there is none); the drives it takes are the run's, kept in
#      runs/NAME/routes.json and passed with --route to every step after it and to every seed: a drive read meanwhile
#      waits for the next run. With NNSLR_BRAIN_ROUTES_FROM=<an earlier run's NAME> (Train › Brain's Drives, "Same
#      drives as"), it takes the drives that run kept (runs/<NAME>/routes.json, or one of its seeds'), those of them
#      still dashcam drives;
#   2. a CUDA smoke test;
#   3. the networks its signs are read with (nnslr brain-networks): the cutter and the classifier chosen in Train ›
#      Brain (NNSLR_BRAIN_CUTTER, _CLASSIFIER; those in use by default), exported as Publish exports them and written
#      into runs/NAME/networks.json, kept when the run is started again (#31, 10 Nov 2026); then reads the signs of every
#      drive it learns from (every dashcam drive) as the car does: its vision.py with those networks, every sign read,
#      of another road too (no applicability network since 15 Nov 2026, #90: the brain learns whose road a sign is of)
#      (nnslr road-type-signs --labelled --sign-networks; a drive already read by them is not read again: a new
#      vision.py or a new choice reads them all again, and the training refuses anything else), the signs it is given;
#      then says what the labels give (nnslr brain-targets). The
#      reading runs on the brain's card and on every other card nothing computes on and no other job of the panel
#      holds at that moment, which the panel then counts as the brain's until the step ends (#36: the NVDEC of one
#      V100 held it back). There is no second reading of the signs since 12 Nov 2026: it made the answers of the
#      next limit, which no brain is taught any more;
#   4-6. trains seeds 0, 1 and 2, each into a run of its own, runs/NAME-s0, -s1, -s2. The best epoch by validation error
#      by place is kept by default. Train › Brain draws the seeds on one chart. With
#      NNSLR_BRAIN_SEEDS=0 seed 0 alone (step 4), a third of the time.
#
# Six steps, as before 12 Nov 2026. From 12 to 17 Nov 2026 they were seven: the fourth read the road users with an
# off-the-shelf detector (nnslr road-users) and the training learnt them in a head of its own, users_head. Both are gone
# since 17 Nov 2026 (#112): the brain learns who crosses our path from the marks of Speed review, as a kind of road
# (job_progress still reads the logs of seven steps).
#
# usage: scripts/run_brain.sh NAME
#
# Writes only into $DATA_ROOT/runs/NAME/ (brain.log, routes.json, dry-run/), $DATA_ROOT/runs/NAME-s{0,1,2}/
# (brain-last.pt, brain-best.pt, result.json),
# the sign readings (derived/road_type_signs) and the frame cache. It never imports reviews, changes the labels, the model in use or the car, and
# exports nothing: Publish exports the brain chosen in Models and checks it along a real drive. Run it again with the same
# NAME to finish: a finished seed is not repeated, the signs already read are not read again, the drives are the same.
#
# Environment: NNSLR_GPU_INDEX (default 0), NNSLR_GPU_VENV (.venv-gpu), NNSLR_BRAIN_EPOCHS (40), NNSLR_SIGN_WORKERS
# (processes reading the signs as the car does: half the cores, each with its ffmpeg),
# NNSLR_SIGN_GPUS (the cards of step 3, nvidia-smi's numbers: 0,1;
# default the brain's and every other card free then), NNSLR_BRAIN_WORKS (1: it also learns the
# roadworks; 0: not), NNSLR_BRAIN_ROAD (1: it also learns the kinds of road of Speed review; 0: not),
# NNSLR_BRAIN_TYPES (1: it also learns the road type of the map and of Speed review, and only from the seconds that have
# one; 0: not, every second),
# NNSLR_BRAIN_MAP (1: the map's limit the comma had, liveMapDataSP, is one more input of the brain, #85; 0, the default:
# not, the brain never reads the map), NNSLR_BRAIN_APPLICABILITY (1, the default: it also learns whose road a speed sign
# is of from the reviewed boxes, a head only the training has, #89, #90; 0: not),
# NNSLR_BRAIN_ROUTES_FROM (an earlier brain run's NAME: its drives instead of every dashcam drive; a run started again
# keeps its own),
# NNSLR_BRAIN_PICTURE (1, default: with image features; 0: without, for comparison only),
# NNSLR_BRAIN_KEEP (best, default: lowest validation error by place; last: the last epoch),
# NNSLR_BRAIN_BALANCE (0, default: off; 1: balance the speed loss by situation),
# NNSLR_BRAIN_POOL (grid: the picture's features keep where they are; mean: not), NNSLR_BRAIN_NUMBER_PATH (bell, the
# default since 16 Nov 2026, #104: the number in force reaches the answer as a bell around it, the shape of the answer;
# class: on its class alone, as every brain before, the comparison; Train › Brain's Number), NNSLR_BRAIN_INIT_FROM (an earlier
# brain's brain-last.pt every seed goes on from, instead of ImageNet and random weights: Train › Brain's "Start from";
# the dry run of step 1 refuses one stepped otherwise or learnt on another split), NNSLR_BRAIN_CUTTER,
# NNSLR_BRAIN_CLASSIFIER (the networks its signs are read with, ids of the Models tab: a SignNet, a classifier; empty:
# the one in use), NNSLR_BRAIN_SEEDS (the seeds it trains: "0 1 2", or
# "0" alone; Train › Brain's Seeds), NNSLR_SUNNYPILOT_CLONE
# (/root/sunnypilot, on the speed-limits branch), NNSLR_ALLOW_BUSY_GPU=1 to share a GPU that already has compute processes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
RUN_ID="${1:-}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
EPOCHS="${NNSLR_BRAIN_EPOCHS:-40}"
CORES="$(nproc 2>/dev/null || echo 8)"
SIGN_WORKERS_DEFAULT=$(( CORES / 2 > 1 ? CORES / 2 : 1 ))
SIGN_WORKERS="${NNSLR_SIGN_WORKERS:-$SIGN_WORKERS_DEFAULT}"
SIGN_GPUS="${NNSLR_SIGN_GPUS:-}"
WORKS="${NNSLR_BRAIN_WORKS:-1}"
ROAD="${NNSLR_BRAIN_ROAD:-1}"
TYPES="${NNSLR_BRAIN_TYPES:-1}"
MAP="${NNSLR_BRAIN_MAP:-0}"
APPLICABILITY="${NNSLR_BRAIN_APPLICABILITY:-1}"
ROUTES_FROM="${NNSLR_BRAIN_ROUTES_FROM:-}"
POOL="${NNSLR_BRAIN_POOL:-grid}"
# A NAME started again takes the path of that moment, as it takes the pool: give the same NNSLR_BRAIN_NUMBER_PATH, or
# its remaining seeds are of the other path (each result.json says its own).
NUMBER_PATH="${NNSLR_BRAIN_NUMBER_PATH:-bell}"
PICTURE="${NNSLR_BRAIN_PICTURE-1}"
KEEP="${NNSLR_BRAIN_KEEP-best}"
BALANCE="${NNSLR_BRAIN_BALANCE-0}"
INIT_FROM="${NNSLR_BRAIN_INIT_FROM:-}"
SIGN_CUTTER="${NNSLR_BRAIN_CUTTER:-}"
SIGN_CLASSIFIER="${NNSLR_BRAIN_CLASSIFIER:-}"
read -r -a SEEDS <<< "${NNSLR_BRAIN_SEEDS:-0 1 2}"

if [[ -z "$RUN_ID" || $# -ne 1 ]]; then
  echo "usage: $0 NAME" >&2
  echo "example: NNSLR_GPU_INDEX=0 $0 2026-11-05_1500-brain" >&2
  exit 2
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: NAME contains unsupported characters" >&2
  exit 2
fi

for setting in EPOCHS SIGN_WORKERS; do
  if [[ ! "${!setting}" =~ ^[0-9]+$ ]]; then
    echo "error: $setting must be a whole number: ${!setting}" >&2
    exit 2
  fi
done

if [[ -n "$SIGN_GPUS" && ! "$SIGN_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "error: NNSLR_SIGN_GPUS must be cards by nvidia-smi's numbers, e.g. 0,1: $SIGN_GPUS" >&2
  exit 2
fi

if [[ "$WORKS" != "0" && "$WORKS" != "1" ]]; then
  echo "error: NNSLR_BRAIN_WORKS must be 0 or 1: $WORKS" >&2
  exit 2
fi

if [[ "$ROAD" != "0" && "$ROAD" != "1" ]]; then
  echo "error: NNSLR_BRAIN_ROAD must be 0 or 1: $ROAD" >&2
  exit 2
fi

if [[ "$TYPES" != "0" && "$TYPES" != "1" ]]; then
  echo "error: NNSLR_BRAIN_TYPES must be 0 or 1: $TYPES" >&2
  exit 2
fi

if [[ "$MAP" != "0" && "$MAP" != "1" ]]; then
  echo "error: NNSLR_BRAIN_MAP must be 0 or 1: $MAP" >&2
  exit 2
fi

if [[ "$APPLICABILITY" != "0" && "$APPLICABILITY" != "1" ]]; then
  echo "error: NNSLR_BRAIN_APPLICABILITY must be 0 or 1: $APPLICABILITY" >&2
  exit 2
fi

if [[ -n "$ROUTES_FROM" && ! "$ROUTES_FROM" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: NNSLR_BRAIN_ROUTES_FROM must be the NAME of an earlier brain run: $ROUTES_FROM" >&2
  exit 2
fi

if [[ "${SEEDS[*]}" != "0 1 2" && "${SEEDS[*]}" != "0" ]]; then
  echo "error: NNSLR_BRAIN_SEEDS must be \"0 1 2\" or \"0\": ${NNSLR_BRAIN_SEEDS:-}" >&2
  exit 2
fi

# The driver's inputs are gone since 10 Nov 2026 (brain_targets.LEFT_OUT): with the car's speed in, the brain followed
# his speed on the road. A caller of before that still asks for them stops here.
if [[ "${NNSLR_BRAIN_DRIVER:-0}" != "0" ]]; then
  echo "error: NNSLR_BRAIN_DRIVER is gone: the brain never takes the gas, the brake or the car's speed" >&2
  exit 2
fi

if [[ "$POOL" != "mean" && "$POOL" != "grid" ]]; then
  echo "error: NNSLR_BRAIN_POOL must be mean or grid: $POOL" >&2
  exit 2
fi

if [[ "$NUMBER_PATH" != "bell" && "$NUMBER_PATH" != "class" ]]; then
  echo "error: NNSLR_BRAIN_NUMBER_PATH must be bell or class: $NUMBER_PATH" >&2
  exit 2
fi

for setting in PICTURE BALANCE; do
  if [[ "${!setting}" != "0" && "${!setting}" != "1" ]]; then
    echo "error: NNSLR_BRAIN_$setting must be 0 or 1: ${!setting}" >&2
    exit 2
  fi
done

if [[ "$KEEP" != "best" && "$KEEP" != "last" ]]; then
  echo "error: NNSLR_BRAIN_KEEP must be best or last: $KEEP" >&2
  exit 2
fi

for setting in SIGN_CUTTER SIGN_CLASSIFIER; do
  if [[ -n "${!setting}" && ! "${!setting}" =~ ^(cutter|classifier):[A-Za-z0-9][A-Za-z0-9._/-]*$ ]]; then
    echo "error: NNSLR_BRAIN_${setting#SIGN_} must be an id of the Models tab: ${!setting}" >&2
    exit 2
  fi
done

if [[ -n "$INIT_FROM" && ! -f "$INIT_FROM" ]]; then
  echo "error: NNSLR_BRAIN_INIT_FROM is not a file: $INIT_FROM" >&2
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
  echo "error: selected GPU already has compute processes; refusing to train the brain" >&2
  echo "$BUSY" >&2
  echo "Set NNSLR_ALLOW_BUSY_GPU=1 only after intentionally accepting contention." >&2
  exit 3
fi

RUN_ROOT="$DATA_ROOT/runs/$RUN_ID"
# A folder of another kind of run is never written into.
for part in detector reader export bundle experiment-b detector-sizes classifier road-type; do
  if [[ -e "$RUN_ROOT/$part" ]]; then
    echo "error: $RUN_ROOT is another kind of run (it has $part/): choose a new NAME" >&2
    exit 3
  fi
done
# SignNet (the cutter) writes its weights at the root of its run, not in a folder.
if [[ -e "$RUN_ROOT/signnet-best.pt" ]]; then
  echo "error: $RUN_ROOT is another kind of run (it has signnet-best.pt, a SignNet): choose a new NAME" >&2
  exit 3
fi

mkdir -p "$RUN_ROOT"
# Everything printed from here on also goes to the run's own log (the panel reads its steps and epochs).
exec > >(tee -a "$RUN_ROOT/brain.log") 2>&1

export NNSLR_DATA_ROOT="$DATA_ROOT"
# The index is nvidia-smi's, the card checked above: CUDA counts the cards in the same order, by PCI bus (its
# default order, fastest first, leaves two identical V100s in no stated order).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
NN="$VENV/bin/nnslr"
SETTINGS=(--epochs "$EPOCHS" --pool "$POOL")
if [[ "$WORKS" == "0" ]]; then
  SETTINGS+=(--no-works)
fi
if [[ "$ROAD" == "0" ]]; then
  SETTINGS+=(--no-road)
fi
if [[ "$TYPES" == "0" ]]; then
  SETTINGS+=(--no-road-types)
fi
if [[ "$MAP" == "1" ]]; then
  SETTINGS+=(--map-limit)
fi
if [[ "$APPLICABILITY" == "0" ]]; then
  SETTINGS+=(--no-applicability)
fi
# Always said, so that the log of a run names its number path (the panel's chip reads it there).
[[ "$PICTURE" == "0" ]] && SETTINGS+=(--no-picture)
[[ "$BALANCE" == "1" ]] && SETTINGS+=(--balance)
SETTINGS+=(--keep "$KEEP")
SETTINGS+=(--number-path "$NUMBER_PATH")
if [[ -n "$INIT_FROM" ]]; then
  SETTINGS+=(--init-from "$INIT_FROM")
fi

echo "== Brain =="
echo "run:          $RUN_ID"
echo "seeds:        ${SEEDS[*]}"
echo "map's limit:  $([[ "$MAP" == "1" ]] && echo "on (an input of the brain)" || echo off)"
echo "applicability: $([[ "$APPLICABILITY" == "1" ]] && echo "on (a head of the brain, every sign's box)" || echo off)"
echo "number path:  $([[ "$NUMBER_PATH" == "bell" ]] && echo "bell (a bell around the number in force)" || echo "class (the number's class alone, as before 16 Nov 2026)")"
echo "picture:      $([[ "$PICTURE" == "1" ]] && echo with || echo without)"
echo "keep:         $KEEP"
echo "balance:      $([[ "$BALANCE" == "1" ]] && echo "by situation" || echo off)"
if [[ -n "$ROUTES_FROM" ]]; then
  echo "drives:       the same as $ROUTES_FROM"
else
  echo "drives:       every dashcam drive"
fi
echo "output:       $(printf "$RUN_ROOT-s%s " "${SEEDS[@]}")"
echo "data root:    $DATA_ROOT"
echo "physical GPU: $GPU_INDEX"
echo "settings:     ${SETTINGS[*]}"
echo

echo "== 1/6 Check the labels, the split and the training (CPU-only) =="
# The drives of the run: those its first dry run took (brain_train.ROUTES), kept for every step and every seed whatever
# drive is read meanwhile; a NAME run again keeps them too.
ROUTES_FILE="$RUN_ROOT/routes.json"
ROUTE_ARGS=()
read_routes() {
  local found route
  found="$(grep -soE '[0-9a-f]{8}--[0-9a-f]{10}' "$1" || true)"
  if [[ -z "$found" ]]; then
    echo "error: no drive in $1" >&2
    exit 3
  fi
  ROUTE_ARGS=()
  while IFS= read -r route; do
    ROUTE_ARGS+=(--route "$route")
  done <<< "$found"
}
if [[ -f "$ROUTES_FILE" ]]; then
  read_routes "$ROUTES_FILE"
elif [[ -n "$ROUTES_FROM" ]]; then
  # The drives of an earlier run: its own routes.json, else the first of its seeds' (the dry run keeps those of them
  # still dashcam drives, and they become this run's).
  FROM_FILE=""
  for candidate in "$DATA_ROOT/runs/$ROUTES_FROM/routes.json" "$DATA_ROOT/runs/$ROUTES_FROM"-s{0,1,2}/routes.json; do
    if [[ -f "$candidate" ]]; then
      FROM_FILE="$candidate"
      break
    fi
  done
  if [[ -z "$FROM_FILE" ]]; then
    echo "error: the run $ROUTES_FROM kept no drives (no routes.json in runs/$ROUTES_FROM nor in its seeds)" >&2
    exit 3
  fi
  read_routes "$FROM_FILE"
  echo "drives of:    $FROM_FILE"
fi
CUDA_VISIBLE_DEVICES="" "$NN" train-brain --dry-run --run-dir "$RUN_ROOT/dry-run" "${SETTINGS[@]}" "${ROUTE_ARGS[@]}"
if [[ ! -f "$ROUTES_FILE" ]]; then
  read_routes "$RUN_ROOT/dry-run/routes.json"
  cp "$RUN_ROOT/dry-run/routes.json" "$ROUTES_FILE"
fi
echo "drives:       $(( ${#ROUTE_ARGS[@]} / 2 )) (routes.json)"

echo "== 2/6 Explicit CUDA smoke test =="
"$NN" env --gpu-smoke --gpu-device 0

echo "== 3/6 Read the signs of the drives as the car does today =="
# The networks they are read with, chosen in Train › Brain (those in use by default), exported on the brain's card into
# the run (networks.json): a run started again keeps its own.
NETWORKS_FILE="$RUN_ROOT/networks.json"
NETWORK_ARGS=()
if [[ -n "$SIGN_CUTTER" ]]; then NETWORK_ARGS+=(--cutter "$SIGN_CUTTER"); fi
if [[ -n "$SIGN_CLASSIFIER" ]]; then NETWORK_ARGS+=(--classifier "$SIGN_CLASSIFIER"); fi
"$NN" brain-networks --run-dir "$RUN_ROOT" --data-root "$DATA_ROOT" "${NETWORK_ARGS[@]}"
SIGN_ARGS=(--sign-networks "$NETWORKS_FILE")
# The signs on the GPU: .venv has onnxruntime-gpu, .venv-gpu only the CPU's (torch brings its own CUDA).
SIGNS_NN="$ROOT_DIR/.venv/bin/nnslr"
[ -x "$SIGNS_NN" ] || SIGNS_NN="$NN"
# The cards that read them (nnslr borrow-cards): the brain's, and every other card no compute process is on and no
# other job of the panel holds; or those of NNSLR_SIGN_GPUS. In a job of the panel the others are written to its record
# (NNSLR_JOB_RECORD): the panel names them as the brain's and starts nothing else on them until this step ends. Chosen
# here, not at the start (the checks above take a minute). Without an answer, the brain's card alone.
sign_cards() {
  local cards
  if [[ -n "$SIGN_GPUS" ]]; then
    cards="$("$SIGNS_NN" borrow-cards --own "$GPU_INDEX" --cards "$SIGN_GPUS" --data-root "$DATA_ROOT")" ||
      cards="$SIGN_GPUS"
  else
    cards="$("$SIGNS_NN" borrow-cards --own "$GPU_INDEX" --data-root "$DATA_ROOT")" || cards="$GPU_INDEX"
  fi
  [[ "$cards" =~ ^[0-9]+(,[0-9]+)*$ ]] || cards="$GPU_INDEX"
  echo "$cards"
}
CARDS="$(sign_cards)"
echo "sign cards:   $CARDS"
"$SIGNS_NN" road-type-signs --labelled --num-workers "$SIGN_WORKERS" --gpus "$CARDS" --data-root "$DATA_ROOT" \
  "${SIGN_ARGS[@]}" "${ROUTE_ARGS[@]}"
"$SIGNS_NN" borrow-cards --release --data-root "$DATA_ROOT" || true
if [[ "$TYPES" == "0" ]]; then
  "$NN" brain-targets --data-root "$DATA_ROOT" "${SIGN_ARGS[@]}" "${ROUTE_ARGS[@]}" --no-road-types
else
  "$NN" brain-targets --data-root "$DATA_ROOT" "${SIGN_ARGS[@]}" "${ROUTE_ARGS[@]}"
fi

# Each seed its own step, 4 to 6 (job_progress: seed 0 alone is step 4 alone).
for seed in "${SEEDS[@]}"; do
  echo "== $(( 4 + seed ))/6 Train seed $seed =="
  if [[ -f "$RUN_ROOT-s$seed/result.json" ]]; then
    echo "skip seed $seed: already trained"
    continue
  fi
  "$NN" train-brain --run-dir "$RUN_ROOT-s$seed" --seed "$seed" "${SETTINGS[@]}" "${SIGN_ARGS[@]}" "${ROUTE_ARGS[@]}"
done

if [[ "$PICTURE" == "1" ]]; then
  echo "Brain complete: choose a seed in Models › Brain, then Publish."
else
  echo "Brain complete: compare its seeds in Models › Brain. Without the picture, they cannot be published to the car."
fi
