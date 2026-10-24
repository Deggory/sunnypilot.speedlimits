#!/usr/bin/env bash
set -euo pipefail

# Qwen3-VL-8B-Instruct (Alibaba, Apache-2.0), a model that sees images, on this server: to measure it as a reader of
# the boxes our cutter cuts out, and as a proposer on whole frames (docs/NEURAL_NETWORK.md). It does:
#
#   1. installs transformers, accelerate and qwen-vl-utils (pinned) into the GPU environment (.venv-gpu); nothing
#      already installed there changes version (checked with pip --dry-run on 25 Oct 2026);
#   2. downloads the model at a pinned revision (about 17.5 GB) into $DATA_ROOT/models/vlm/Qwen3-VL-8B-Instruct;
#   3. loads it on the GPU in float16 (the V100 has no bfloat16) and asks it about one reviewed frame.
#
# usage: scripts/setup_vlm.sh            (again: only what is missing is done)
#        scripts/setup_vlm.sh --no-test  (skip step 3, e.g. while the GPU is busy)
#
# Environment: NNSLR_DATA_ROOT (data/), NNSLR_GPU_VENV (.venv-gpu), NNSLR_GPU_INDEX (0).
# It never touches the reviews, the dataset, the models in use or the comma.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-$ROOT_DIR/data}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
REVISION="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TARGET="$DATA_ROOT/models/vlm/Qwen3-VL-8B-Instruct"
TEST=1
[[ "${1:-}" == "--no-test" ]] && TEST=0

[[ -x "$VENV/bin/python" ]] || { echo "missing $VENV: run scripts/setup_v100_training_env.sh first" >&2; exit 1; }

echo "== 1/3 Python packages in $VENV"
"$VENV/bin/python" -m pip install --quiet 'transformers==5.20.2' 'accelerate==1.15.0' 'qwen-vl-utils==0.0.14'
"$VENV/bin/python" -c "import transformers, accelerate; print('transformers', transformers.__version__, '· accelerate', accelerate.__version__)"

echo "== 2/3 $MODEL_ID at $REVISION into $TARGET"
mkdir -p "$TARGET"
"$VENV/bin/python" - "$MODEL_ID" "$REVISION" "$TARGET" <<'PY'
import sys
from huggingface_hub import snapshot_download
model_id, revision, target = sys.argv[1:]
path = snapshot_download(model_id, revision=revision, local_dir=target)
print("model files in", path)
PY
echo "$REVISION" > "$TARGET/REVISION"

if [[ "$TEST" -eq 0 ]]; then
  echo "Installed (not tried on the GPU: --no-test)."
  exit 0
fi

echo "== 3/3 Try it on the GPU"
frame="$(ls "$DATA_ROOT"/derived/frames/*/*/frame_*.png 2>/dev/null | head -1 || true)"
[[ -n "$frame" ]] || { echo "no extracted frame to try it on: installed, not tried"; exit 0; }
CUDA_VISIBLE_DEVICES="${NNSLR_GPU_INDEX:-0}" "$VENV/bin/python" - "$TARGET" "$frame" <<'PY'
import sys, time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
target, frame = sys.argv[1:]
started = time.time()
model = AutoModelForImageTextToText.from_pretrained(target, dtype=torch.float16, device_map="cuda", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained(target)
print(f"loaded in {time.time() - started:.0f} s, {torch.cuda.memory_allocated() / 1e9:.1f} GB on the GPU")
messages = [{"role": "user", "content": [{"type": "image", "image": frame},
             {"type": "text", "text": "List every traffic sign you see in this dashcam frame, with its speed value if it has one."}]}]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt").to(model.device)
started = time.time()
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
answer = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(f"answer in {time.time() - started:.1f} s on {frame}:\n{answer}")
PY
echo "Installed and tried: $TARGET"
