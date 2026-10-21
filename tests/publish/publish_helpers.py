"""Shared helpers for the Hugging Face publishing tests.

make_bundle builds a small but complete bundle with nnslr_tools.bundle.package_model, so
verify_bundle accepts it: the ONNX files are a few fake bytes (verification checks hashes and
contracts, it never loads the model). The fake executables stand in for curl and uvx, so the tests
never touch the network nor need huggingface_hub.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nnslr_tools.bundle import package_model


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_contract(path: Path, task: str, onnx: Path, extra: dict) -> None:
    payload = {
        "schema_version": 1,
        "task": task,
        "onnx_sha256": _sha(onnx),
        "checkpoint_sha256": "c" * 64,
        "dataset_sha256": "a" * 64,
        "split_sha256": "b" * 64,
        "verification": {"passed": True},
    }
    for key in extra:
        payload[key] = extra[key]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def make_bundle(bundle_dir: Path, detector_bytes: bytes = b"detector-onnx") -> Path:
    inputs = bundle_dir.parent / (bundle_dir.name + "-inputs")
    inputs.mkdir(parents=True)
    detector = inputs / "detector.onnx"
    reader = inputs / "reader.onnx"
    detector.write_bytes(detector_bytes)
    reader.write_bytes(b"reader-onnx")

    detector_contract = inputs / "detector.onnx.contract.json"
    _write_contract(
        detector_contract,
        "detector",
        detector,
        {
            "input": {"shape": [1, 3, 320, 320]},
            "outputs": {
                "boxes": {"shape": ["detections", 4]},
                "scores": {"shape": ["detections"]},
                "labels": {"shape": ["detections"]},
            },
        },
    )
    reader_contract = inputs / "reader.onnx.contract.json"
    _write_contract(
        reader_contract,
        "reader",
        reader,
        {
            "input": {"shape": [1, 3, 160, 160]},
            "output": {"shape": [1, 2], "classes": ["maximum_speed:30", "maximum_speed:50"]},
        },
    )

    package_model(detector, detector_contract, reader, reader_contract, bundle_dir)
    return bundle_dir


FAKE_CURL = """#!/usr/bin/env bash
# Fake curl: records its arguments, writes the -o file and prints FAKE_CURL_CODE like -w '%{http_code}'.
echo "$@" >> "$FAKE_CURL_LOG"
output=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-o" ]; then
    shift
    output="$1"
  fi
  shift
done
if [ "$FAKE_CURL_CODE" = "200" ]; then
  cp "$FAKE_CURL_BODY" "$output"
else
  echo "error page" > "$output"
fi
printf '%s' "$FAKE_CURL_CODE"
"""

FAKE_UVX = """#!/usr/bin/env bash
# Fake uvx: records one argument per line, uploads nothing.
for arg in "$@"; do
  echo "$arg" >> "$FAKE_UVX_LOG"
done
"""


def write_fake_tools(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    curl = bin_dir / "curl"
    curl.write_text(FAKE_CURL, encoding="utf-8")
    curl.chmod(0o755)
    uvx = bin_dir / "uvx"
    uvx.write_text(FAKE_UVX, encoding="utf-8")
    uvx.chmod(0o755)
