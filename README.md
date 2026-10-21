# sunnypilot.speedlimits

Standalone speed-limit vision tooling: prepare a dataset, train a detector and a numeric reader, evaluate them, and export reproducible model bundles. The intended integration target is sunnypilot's existing Speed Limit Assist.

**Current status:** source import and research tooling. No trained weights are included. The archived sunnypilot integration has no working inference backend and does not feed Speed Limit Assist.

## What was recovered

- Dataset, route/video/log alignment, annotation review and split tooling from [openpilot-nnslr-tools](https://github.com/cristianku/openpilot-nnslr-tools).
- Detector training based on SSDLite320/MobileNetV3 Large and a MobileNetV3 Small crop reader.
- Offline evaluation, hard-example mining, replay, ONNX export/parity and model-bundle verification.
- A dependency-light `speed_vision_core`: typed observations, validation, serialization, temporal consensus and advisory comparison.
- A frozen copy of the `nn-speed-limit-vision` runtime changes: frame adapters, messages, lifecycle, diagnostics, tests and the separate VISION badge.

See [the import audit](docs/IMPORT_AUDIT.md) for exact source commits and limitations.

## Quickstart

Python 3.11 or newer:

```sh
git clone https://github.com/cristianku/sunnypilot.speedlimits.git
cd sunnypilot.speedlimits
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e . pytest
nnslr --help
nnslr selftest
pytest -q
```

The existing `nnslr` command and Python package names are retained for compatibility. The core/CLI do not implicitly start training or import the GPU stack.

- [Installation](docs/INSTALL.md) — system packages, Python environments, downloaded models and data folders.
- [Training and data guide](docs/upstream-training-guide.md) — recovered upstream commands and workflow.
- [GPU container setup](docs/proxmox-gpu-container.md) — historical host setup instructions.
- [Runtime snapshot](integrations/sunnypilot/README.md) — what is present and what must be implemented.
- [Development plan](docs/development-plan.md) — remaining work for this repository.

Private videos, logs, datasets, checkpoints and model weights stay outside Git. Use the configured data root and publish reviewed model artifacts separately when available.

## Helpers, commands and scripts

Run these commands **from the repository root**. [`docs/INSTALL.md`](docs/INSTALL.md) covers required system packages and virtual environments; [`docs/upstream-training-guide.md`](docs/upstream-training-guide.md) covers data preparation, training/evaluation and complete CLI options.

| Entry point | Purpose | Requirements / safe first command |
| --- | --- | --- |
| `.venv/bin/nnslr` | Main command-line interface: verify the environment, synchronize or link videos, extract frames, propose/review/import annotations, validate dataset, train, evaluate and export. | Core `.venv`; start with `.venv/bin/nnslr --help` and `.venv/bin/nnslr selftest`. GPU and pre-annotation commands need separate dependencies. |
| `scripts/setup_v100_training_env.sh` | Create `.venv-gpu` with CUDA 12.6-compatible PyTorch for **NVIDIA Tesla V100**. Does not install the NVIDIA driver or start a training job. | NVIDIA driver visible via `nvidia-smi`; run `bash scripts/setup_v100_training_env.sh` only when the GPU environment setup is authorized. |
| `scripts/run_v100_baseline.sh` | Validate dataset/splits, then train detector and reader, evaluate, mine errors, export ONNX and package a verified bundle. **Uses GPU and creates output files.** | Prepared, validated dataset and `.venv-gpu`; invoke `NNSLR_GPU_INDEX=0 bash scripts/run_v100_baseline.sh baseline-001` only when training is authorized. The run ID must be new. |
| `scripts/publish_model.sh` | Verify/stage an existing bundle, create a catalog and optionally upload it to Hugging Face. **The normal invocation publishes externally.** | Have a bundle; first run `bash scripts/publish_model.sh --no-upload`. Upload requires `uvx`, Hugging Face authorization and explicit approval. |
| `scripts/install_review_service.sh` | Install/update or remove `nnslr-review` as a systemd service for the web review UI. **Installation changes system configuration.** | Linux with systemd, `.venv` installed; inspect with `bash scripts/install_review_service.sh --dry-run` before an authorized `sudo bash scripts/install_review_service.sh`. |

Common local workflows (use `--help` on each `nnslr` subcommand for exact arguments):

```sh
# Choose a private data folder outside Git when convenient:
export NNSLR_DATA_ROOT=/path/to/speed-vision-data

# Option A: inspect an SSH route download without transferring files.
.venv/bin/nnslr sync-routes --route ROUTE_ID --dry-run

# Option B: inspect symlinks from COMMA_VIDEOS_DIR / the shared videos folder.
.venv/bin/nnslr link-videos --dry-run

# Once recordings are present in the data root:
.venv/bin/nnslr extract-frames --route ROUTE_ID
# Preannotation requires the optional [preannotate] dependencies and downloads model weights.
.venv/bin/nnslr preannotate --route ROUTE_ID
.venv/bin/nnslr review --route ROUTE_ID
```

`sync-routes` and `link-videos` are **alternative ways to make local recordings available**, not mandatory consecutive steps. Review/annotation decisions must be confirmed by a human before importing as ground truth. Training is a separate, explicitly initiated action. See [`instructions.md`](instructions.md) for the authorized reviewer/Playwright workflow and [`docs/proxmox-gpu-container.md`](docs/proxmox-gpu-container.md) for the shared `COMMA_VIDEOS_DIR` layout.

### Relationship to the other Openpilot repositories

- [`openpilot.distill`](https://github.com/cristianku/openpilot.distill) trains a **driving model** and provides its own [video/calibration helpers](https://github.com/cristianku/openpilot.distill/blob/master/helpers/README.md). It is **not** the speed-limit detector or `nnslr` CLI.
- [`openpilot_scripts`](https://github.com/cristianku/openpilot_scripts) manages fork/branch/device update workflows. Its [`OP-update-distilled`](https://github.com/cristianku/openpilot_scripts/blob/main/OP-update-distilled/SKILL.md) skill synchronizes **Distill's Sunnypilot source and installation mirror**; it does **not** train, publish, integrate or install speed-limit model weights.
- This repository currently produces **research tools and model bundles**. Its archived SunnyPilot runtime snapshot has no functioning inference backend or operational Speed Limit Assist adapter. Do not advertise an installable speed-limit feature until that integration and branch are implemented and verified.

## Publishing on Hugging Face

`scripts/publish_model.sh` publishes one model bundle (the `bundle/` folder that `scripts/run_v100_baseline.sh` writes in `data/runs/<run id>/`) to [cristianku/sunnypilot-speedlimits-models](https://huggingface.co/cristianku/sunnypilot-speedlimits-models). Only the model artifacts are uploaded: the detector and reader ONNX files with their contracts, `config.json`, `classes.json`, `model_card.json`, `capability.json` and `sha256sums.json`, plus the catalog `speed_models.json` with size, sha256 and download URL of every file. Videos, logs, datasets and checkpoints are never uploaded. The two contracts do contain the absolute path of each checkpoint on the training machine (for example `/home/<user>/.../data/runs/<run id>/...`), so that path and user name become public with the model.

Once per machine, with a Hugging Face token that can write the repository (`huggingface_hub` is not a dependency of this repo; `uvx` from [uv](https://docs.astral.sh/uv/) runs it on demand, pinned to 1.x/2.x like the script):

```sh
uvx --from 'huggingface_hub>=1,<3' hf auth login
uvx --from 'huggingface_hub>=1,<3' hf repos create cristianku/sunnypilot-speedlimits-models
```

Then try a dry run first, and publish:

```sh
scripts/publish_model.sh --no-upload        # newest data/runs/*/bundle: verify, stage, catalog; nothing is uploaded
scripts/publish_model.sh                    # the same, then one Hugging Face commit
scripts/publish_model.sh data/runs/baseline-001/bundle --name=baseline-001
```

The bundle is checked with `nnslr verify-bundle` first, and a bundle that fails is not published. Files are staged in `<data root>/publish/<name>/`; publishing the same name again replaces its catalog entry. `HF_REPO` selects another repository and `scripts/publish_model.sh --help` lists every option.

## Speed Limit Assist integration

The planned operational adapter will supply a validated vision limit to sunnypilot's existing speed-limit selection flow. Recognition alone does not establish that a sign belongs to our road or is already applicable.

The recovered runtime is intentionally preserved as an **advisory-only historical snapshot**. Its control-isolation tests describe that old implementation; copying it does not implement the planned SLA adapter. This import changes neither sunnypilot nor PSA/opendbc.

## Provenance and licenses

See [NOTICE.md](NOTICE.md) and [provenance/import.json](provenance/import.json). Training source files retain the upstream MIT license metadata. The runtime snapshot retains the original comma MIT and sunnypilot custom license notices under `LICENSES/`.

This project uses software from Haibin Wen and SUNNYPILOT LLC and is licensed under a custom license requiring permission for use.


## Consistent training modes with openpilot.distill

Both projects now expose `fresh`, `continue` and `resume`:

- **Fresh** (default): a new run in a new output directory, with the existing random/default or optional ImageNet initialization.
- **Continue**: a new run initialized from the latest **explicitly reviewed** PyTorch weights listed in Hugging Face `training/stable.json`. The manifest pins each model to a SHA256, and the detector/reader are independently checked for matching architecture and class mapping. No previous optimizer is reused.
- **Resume**: reuse the **original output directory** after interruption; restore model, optimizer, AMP scaler, validation history and epoch counters from `training-state.pt`. A completed run is rejected. Changes to the dataset, split or training config are rejected. Checkpoints are atomic and saved at every epoch boundary; an interrupted partial epoch is repeated rather than replayed exactly.

```bash
# Train the whole baseline pipeline, selecting a fresh new run ID:
bash scripts/run_v100_baseline.sh baseline-002 fresh

# New run using the published stable detector and reader weights:
bash scripts/run_v100_baseline.sh baseline-003 continue

# Recover interrupted detector/reader stages under their existing run ID:
bash scripts/run_v100_baseline.sh baseline-003 resume

# Alternatively use the same --mode flag as Distill with nnslr directly:
.venv-gpu/bin/nnslr train --task detector --mode continue --output data/runs/detector-v2
.venv-gpu/bin/nnslr train --task reader --mode resume --output data/runs/reader-v2
```

`--hf-repo` defaults to `cristianku/sunnypilot-speedlimits-models`; set `NNSLR_HF_REPO` to override. `continue` fails clearly until approved PyTorch weights are actually published. `resume` cannot recover old baseline checkpoints that only contain weights rather than optimizer and progress. Neither command touches a comma device.

### Publishing fine-tunable PyTorch weights

`scripts/publish_model.sh` continues to handle **verified inference ONNX bundles**, unchanged. To make detector and reader **PyTorch weights** available for community fine-tuning, use the separate explicit publication command. Unlike `training-state.pt`, the existing `detector-best.pt` and `reader-best.pt` contain model weights without optimizer state.

```bash
python scripts/publish_training_weights.py \
  --repo cristianku/sunnypilot-speedlimits-models --release baseline-002 \
  --model detector=data/runs/baseline-002/detector/detector-best.pt \
  --model reader=data/runs/baseline-002/reader/reader-best.pt

# Only after separate validation and approval of this release:
python scripts/publish_training_weights.py \
  --repo cristianku/sunnypilot-speedlimits-models --release baseline-002 \
  --model detector=data/runs/baseline-002/detector/detector-best.pt \
  --model reader=data/runs/baseline-002/reader/reader-best.pt --upload --stable
```

The command is **dry-run by default** and rejects full optimizer checkpoints. It uploads immutable files under `training/releases/<release>/`; `--stable` explicitly updates `training/stable.json` with verified hashes. It never uploads private videos/logs. The model licence, source weights and training-data rights must be assessed before a public release. The Sunnypilot runtime snapshot retains separate upstream licence obligations and is not published as training weights.
