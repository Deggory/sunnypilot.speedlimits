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

## Speed Limit Assist integration

The planned operational adapter will supply a validated vision limit to sunnypilot's existing speed-limit selection flow. Recognition alone does not establish that a sign belongs to our road or is already applicable.

The recovered runtime is intentionally preserved as an **advisory-only historical snapshot**. Its control-isolation tests describe that old implementation; copying it does not implement the planned SLA adapter. This import changes neither sunnypilot nor PSA/opendbc.

## Provenance and licenses

See [NOTICE.md](NOTICE.md) and [provenance/import.json](provenance/import.json). Training source files retain the upstream MIT license metadata. The runtime snapshot retains the original comma MIT and sunnypilot custom license notices under `LICENSES/`.

This project uses software from Haibin Wen and SUNNYPILOT LLC and is licensed under a custom license requiring permission for use.
