# Installation

Everything a fresh machine needs to use this repository, from the CPU tools to GPU training and publishing. Each part is optional: install only what you use.

Tested on Debian 13 (trixie) and macOS. Ubuntu 24.04 works the same way with `apt`.

## 1. System packages

```sh
apt update
apt install -y git curl rsync openssh-client python3 python3-venv ffmpeg zstd jq
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv: fast venvs and pip (optional, recommended)
```

| Package | Used by |
| --- | --- |
| `python3` (3.11 or newer), `python3-venv` | everything |
| `ffmpeg` (`ffmpeg` + `ffprobe`) | `nnslr extract-frames`, `nnslr video-probe`, `nnslr make-clips` |
| `zstd` | reading `rlog.zst` / `qlog.zst` from the shell |
| `rsync`, `openssh-client` | `nnslr sync-routes` (copies routes from the comma device over SSH) |
| `curl` | `scripts/publish_model.sh` (downloads the model catalog) |
| `jq` | inspecting the JSON/JSONL reports by hand |
| `git` | cloning and updating |

On macOS: `brew install ffmpeg zstd jq uv`.

GPU training also needs a working NVIDIA driver: `nvidia-smi` must work where you train (inside the container, for an LXC). This repository never installs or changes the driver; see [proxmox-gpu-container.md](proxmox-gpu-container.md) for a Proxmox LXC.

## 2. Python environments

The core CLI has no dependencies on purpose. Heavier tools live in optional extras or in a separate GPU environment, so the CPU tools stay small.

### Core CLI and tests (`.venv`)

```sh
cd sunnypilot.speedlimits
uv venv .venv                     # or: python3 -m venv .venv
uv pip install --python .venv/bin/python -e . pytest
.venv/bin/nnslr selftest
.venv/bin/python -m pytest -q
```

### Pre-annotation (same `.venv`, extra `preannotate`)

```sh
uv pip install --python .venv/bin/python -e ".[preannotate]"
```

Adds `onnxruntime`, `Pillow`, `numpy` and `rapidocr==3.4.0` (CPU). On a machine with an NVIDIA GPU (the V100 LXC) use the extra `preannotate-gpu` instead: `onnxruntime-gpu[cuda,cudnn]` below 1.27 with cuDNN 9.10.2.21 (CUDA 12: the CUDA 13 builds and the newer cuDNN no longer support the V100) instead of `onnxruntime`, and the detector, the road-surface model and the OCR run on the GPU (CPU fallback; `NNSLR_PREANNOTATE_DEVICE=cpu` forces the CPU). The two extras install the same `onnxruntime` module, so remove the other one first (`uv pip uninstall --python .venv/bin/python onnxruntime`); `helpers/update.sh` does this by itself when it finds an NVIDIA GPU. Each model uses `NNSLR_PREANNOTATE_THREADS` CPU threads (`helpers/1_prepare.sh`: cores / routes in parallel; default half of the cores). `nnslr preannotate` downloads its models on the first run and checks their SHA-256; they are cached in `<data root>/models/preannotation/<sha256>/model.onnx`:

| Model | Source | License |
| --- | --- | --- |
| Traffic-sign detector (YOLO, 82 classes, Vietnam signs) | `star092304/traffic-sign-detection-vietnam-yolo` on Hugging Face | model card Apache-2.0, embedded model AGPL-3.0 |
| Road segmentation (SegFormer-b0 Cityscapes, ONNX) | Hugging Face, pinned revision in `src/nnslr_tools/road_text.py` | NVIDIA SegFormer: non-commercial research and evaluation only |
| OCR for numbers painted on the road (PP-OCRv4) | shipped with `rapidocr` | Apache-2.0 |

These models only propose boxes for human review; they are not part of the trained speed-limit models.

### Reading comma logs (extra `log-reader`)

```sh
uv pip install --python .venv/bin/python -e ".[log-reader]"
```

Adds `pycapnp` and `zstandard` for `nnslr log-metadata` and `nnslr align-route`.

### Training on a GPU (`.venv-gpu`)

```sh
bash scripts/setup_v100_training_env.sh
.venv-gpu/bin/wandb login          # once; without it training logs to wandb offline
.venv-gpu/bin/nnslr env --gpu-smoke --gpu-device 0
```

Installs `torch==2.14.0` and `torchvision==0.29.0` from the CUDA 12.6 index (the last line that still supports the Tesla V100, sm_70), `onnx`, `onnxscript`, `onnxruntime` and the `training` extra (`wandb`). On a newer GPU you can install the default PyPI `torch` instead.

### Publishing a model on Hugging Face

Nothing to install: `scripts/publish_model.sh` runs the Hugging Face CLI through `uvx`. Log in once with a token that can write the target repository:

```sh
uvx --from "huggingface_hub>=1,<3" hf auth login
```

## 3. Data folders

| Folder | Default | Override | Content |
| --- | --- | --- | --- |
| Data root | `<clone>/data` | `NNSLR_DATA_ROOT` or `--data-root` | frames, annotations, datasets, runs, cached models |
| Comma videos | `<clone>/../comma-videos` | `COMMA_VIDEOS_DIR` | `fcamera.hevc`, `rlog.zst`, ... per `<route>--<segment>/`, shared with other projects |

Both stay out of Git. Videos are stored once in the comma-videos folder and linked into the data root, so another project on the same machine (for example openpilot.distill) can use the same files.

### Automated comma route filtering (shared fetch helper)

To run `bash helpers/fetch_comma_segments.sh` (automatic route selection and download), install the optional log reader and set schema paths:

```sh
uv pip install --python .venv/bin/python -e '.[log-reader]'
export NNSLR_OPENPILOT_ROOT=/path/to/compatible/openpilot
export NNSLR_OPENDBC_ROOT=/path/to/compatible/opendbc
bash helpers/fetch_comma_segments.sh
```

The script connects via SSH/rsync to `COMMA_HOST` (default `comma`), uses `COMMA_VIDEOS_DIR` (shared with Distill), and links accepted files into `NNSLR_DATA_ROOT/raw/routes` (default `data/raw/routes`). In automatic mode it takes every route still on the comma (dashcam or not, no date cutoff) with at least 300 seconds from first to last qlog, and excludes the first/last two available segments plus any segment with reverse gear in the full rlog. A missing/unreadable qlog/rlog causes conservative exclusion. Manual route/`--segments` selections are exempt so explicit transfers remain possible. This is distinct from `nnslr sync-routes`, which does not auto-filter. For synthetic offline tests: `python -m unittest discover -s helpers -p 'test_*.py'`.

## 4. Check the installation

```sh
ffmpeg -version | head -n 1
.venv/bin/nnslr env
.venv/bin/nnslr selftest
.venv/bin/python -m pytest -q
```

## 5. Always-on review site and daily use

Install the review site (with its control panel) as the systemd service `speedlimits-web`, then use the helpers; both are described step by step in [GUIDA.md](GUIDA.md):

```sh
scripts/install_review_service.sh --dry-run
scripts/install_review_service.sh
```

The access token ends up in the `.env` of the clone (`NNSLR_REVIEW_TOKEN=...`, mode 600, ignored by git). Later updates: `helpers/update.sh`.

## Reference machine

The maintainer's setup, for comparison: Debian 13 LXC on Proxmox with one Tesla V100 32 GB (NVIDIA driver 580.178.04), `ffmpeg` 7.1.5, `zstd` 1.5.7, `jq` 1.7, `uv`, and a `.venv` on Python 3.13 with the `preannotate` extra and `pytest`.
