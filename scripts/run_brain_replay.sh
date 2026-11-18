#!/usr/bin/env bash
set -euo pipefail

# Models › brain › Replay (#116): a brain already trained, replayed on its validation and test drives without training
# (nnslr_tools.brain_replay). It measures how it does by situation (a sign in force or none, after an end, road type, a
# car ahead) against the driver's median by number and by number and road type, the entries of ramps and merges, the
# time after an end, the decoders, and how much the number in force moves the answer (the paired replay with the sign
# swapped). The same situation measure a training writes since fb0ada5, for the brains trained before it.
#
# usage: NNSLR_GPU_INDEX=<card> scripts/run_brain_replay.sh <brain run>
#
# Writes only <data root>/runs/<run>/replay-validation.json and replay-test.json (Models › brain reads them), and the log
# <data root>/runs/<run>/replay.log. It trains nothing and never changes a label, a model in use or the car.
# ALWAYS ON A V100: there is no CPU run. The pictures of one partition sit on the card (several GB), so it takes a card
# of its own, as a training does.
#
# Environment: NNSLR_GPU_INDEX (default 0), NNSLR_GPU_VENV (.venv-gpu), NNSLR_ALLOW_BUSY_GPU=1 to share a GPU that already
# has compute processes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
RUN_ID="${1:-}"

if [[ -z "$RUN_ID" || $# -ne 1 ]]; then
  echo "usage: NNSLR_GPU_INDEX=<card> $0 <brain run>" >&2
  exit 2
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: the run's name contains unsupported characters" >&2
  exit 2
fi
RUN_ROOT="$DATA_ROOT/runs/$RUN_ID"
if [[ ! -f "$RUN_ROOT/result.json" || ! -f "$RUN_ROOT/brain-last.pt" ]]; then
  echo "error: $RUN_ID is not a trained brain (no result.json and brain-last.pt in $RUN_ROOT)" >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "error: GPU environment missing: $VENV/bin/python" >&2
  echo "run: bash scripts/setup_v100_training_env.sh" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found: the replay never runs on the CPU" >&2
  exit 3
fi
if ! nvidia-smi -i "$GPU_INDEX" >/dev/null 2>&1; then
  echo "error: invalid/unavailable GPU index: $GPU_INDEX" >&2
  exit 3
fi
BUSY="$(
  nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "$BUSY" && "${NNSLR_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "error: selected GPU already has compute processes; refusing the replay" >&2
  echo "$BUSY" >&2
  exit 3
fi

# Everything printed from here on also goes to the run's own replay log.
exec > >(tee -a "$RUN_ROOT/replay.log") 2>&1

export NNSLR_DATA_ROOT="$DATA_ROOT"
# The index is nvidia-smi's: CUDA counts the cards in the same order, by PCI bus.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
echo "== Replay of the brain $RUN_ID on GPU $GPU_INDEX: validation and test =="
cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$VENV/bin/python" -m nnslr_tools.brain_replay --run "$RUN_ID" --data-root "$DATA_ROOT" --partition both
echo "== Replay of the brain $RUN_ID done =="
