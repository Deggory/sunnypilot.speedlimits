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

- [Training and data guide](docs/upstream-training-guide.md) — recovered upstream commands and workflow.
- [GPU container setup](docs/proxmox-gpu-container.md) — historical host setup instructions.
- [Runtime snapshot](integrations/sunnypilot/README.md) — what is present and what must be implemented.
- [Development plan](docs/development-plan.md) — remaining work for this repository.

Private videos, logs, datasets, checkpoints and model weights stay outside Git. Use the configured data root and publish reviewed model artifacts separately when available.

## Publishing on Hugging Face

`scripts/publish_model.sh` publishes one model bundle (the `bundle/` folder that `scripts/run_v100_baseline.sh` writes in `data/runs/<run id>/`) to [cristianku/sunnypilot-speedlimits-models](https://huggingface.co/cristianku/sunnypilot-speedlimits-models). Only the model artifacts are uploaded: the detector and reader ONNX files with their contracts, `config.json`, `classes.json`, `model_card.json`, `capability.json` and `sha256sums.json`, plus the catalog `speed_models.json` with size, sha256 and download URL of every file. Videos, logs, datasets and checkpoints are never uploaded.

Once per machine, with a Hugging Face token that can write the repository (`huggingface_hub` is not a dependency of this repo; `uvx` from [uv](https://docs.astral.sh/uv/) runs it on demand):

```sh
uvx --from huggingface_hub hf auth login
uvx --from huggingface_hub hf repo create cristianku/sunnypilot-speedlimits-models
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
