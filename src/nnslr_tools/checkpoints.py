"""Shared training checkpoint and Hugging Face stable-weight helpers.

Importing this module never imports PyTorch or contacts a network service.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    """Prevent half-written checkpoints when a process is interrupted."""
    import torch

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=".checkpoint-", suffix=".pt", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def safe_torch_load(path: Path) -> Any:
    """torch.load with weights_only=True (tensors and plain data, never arbitrary pickled objects).

    Checkpoints written before 24 Oct 2026 hold torch.__version__ as a TorchVersion, which the safe loader
    refuses: that one class, a plain str subclass, is allowed.
    """
    import torch

    allowed = []
    version_class = getattr(getattr(torch, "torch_version", None), "TorchVersion", None)
    if version_class is not None:
        allowed.append(version_class)
    with torch.serialization.safe_globals(allowed):
        return torch.load(Path(path), map_location="cpu", weights_only=True)


# A weight of the SSDlite head whose size tells the detector's tail: 960 channels when the MobileNetV3 backbone
# started from ImageNet, 480 otherwise (torchvision reduces the tail without pretrained backbone weights).
DETECTOR_TAIL_KEY = "head.regression_head.module_list.1.0.0.weight"


def detector_is_wide(state_dict: dict[str, Any]) -> bool:
    """True when detector weights were trained with the full tail (backbone started from ImageNet)."""
    weight = state_dict.get(DETECTOR_TAIL_KEY)
    return weight is not None and int(weight.shape[0]) == 960


# The image the detector sees, (width, height): the frame is resized to it inside the model. torchvision's
# SSDlite320 uses 320 x 320; checkpoints written before the input size was recorded were trained at it.
DEFAULT_DETECTOR_INPUT_SIZE = (320, 320)
MAX_DETECTOR_INPUT_SIDE = 4096
# The smallest default box of the detector, as a fraction of the image side (torchvision's SSDlite320 value).
# The default boxes are fractions of the image: a larger input does not make them smaller in frame pixels,
# it only places them on a finer grid. 0.2 of a 1344 x 760 frame is a box of 155 x 88 pixels at the least.
DEFAULT_DETECTOR_ANCHOR_MIN_RATIO = 0.2
DETECTOR_ANCHOR_MAX_RATIO = 0.95


def parse_input_size(text: str) -> tuple[int, int]:
    """'640x360' -> (640, 360), '320' -> (320, 320): the (width, height) of the detector's input."""
    value = str(text).strip().lower()
    parts = value.split("x") if "x" in value else [value, value]
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"invalid input size {text!r}: expected WIDTHxHEIGHT, e.g. 640x360")
    width, height = int(parts[0]), int(parts[1])
    if not (32 <= width <= MAX_DETECTOR_INPUT_SIDE and 32 <= height <= MAX_DETECTOR_INPUT_SIDE):
        raise ValueError(f"invalid input size {text!r}: each side must be 32 to {MAX_DETECTOR_INPUT_SIDE} pixels")
    return width, height


def format_input_size(size: tuple[int, int] | list[int]) -> str:
    return f"{int(size[0])}x{int(size[1])}"


def detector_input_size(checkpoint: dict[str, Any]) -> tuple[int, int]:
    """The (width, height) a detector checkpoint was trained at; 320 x 320 when it does not say."""
    value = checkpoint.get("input_size")
    if value is None:
        return DEFAULT_DETECTOR_INPUT_SIZE
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(type(side) is int and 32 <= side <= MAX_DETECTOR_INPUT_SIDE for side in value)
    ):
        raise ValueError("checkpoint_invalid_input_size")
    return int(value[0]), int(value[1])


def detector_anchor_min_ratio(checkpoint: dict[str, Any]) -> float:
    """The smallest default box of a detector checkpoint; torchvision's 0.2 when it does not say."""
    value = checkpoint.get("anchor_min_ratio")
    if value is None:
        return DEFAULT_DETECTOR_ANCHOR_MIN_RATIO
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < DETECTOR_ANCHOR_MAX_RATIO:
        raise ValueError("checkpoint_invalid_anchor_min_ratio")
    return float(value)


def build_detector(*, wide: bool, input_size: tuple[int, int] = DEFAULT_DETECTOR_INPUT_SIZE,
                   anchor_min_ratio: float = DEFAULT_DETECTOR_ANCHOR_MIN_RATIO):
    """The two-class SSDlite MobileNetV3 detector, with the tail of the weights it will load, resizing the frame
    to input_size (width, height) inside the model.

    torchvision builds the full tail only when it is given pretrained backbone weights: asking for them builds
    the same shape as a training started from ImageNet (they are overwritten by the weights loaded next; the
    file comes from the torch cache, downloaded once by the training).

    The input size and the default boxes do not change any weight: the backbone and the head are convolutions,
    and the default boxes follow the size of each feature map (DefaultBoxGenerator without steps). A detector
    must still run at the size it was trained at, which is why the checkpoint records it.
    """
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
    from torchvision.models.detection.transform import GeneralizedRCNNTransform

    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if wide else None,
        num_classes=2,
    )
    width, height = (int(side) for side in input_size)
    if (width, height) != DEFAULT_DETECTOR_INPUT_SIZE:
        # The same transform SSD builds for its fixed size: bilinear resize to (width, height), then the mean and
        # std of SSDlite; the boxes come back in the pixels of the frame.
        transform = model.transform
        model.transform = GeneralizedRCNNTransform(
            min(width, height), max(width, height), transform.image_mean, transform.image_std,
            size_divisible=1, fixed_size=(width, height))
    if anchor_min_ratio != DEFAULT_DETECTOR_ANCHOR_MIN_RATIO:
        # Same aspect ratios, so the same number of boxes per cell and the same head.
        model.anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=float(anchor_min_ratio),
                                                     max_ratio=DETECTOR_ANCHOR_MAX_RATIO)
    return model


def load_detector(checkpoint: dict[str, Any]):
    """A detector rebuilt from a checkpoint as it was trained (tail, input size, default boxes), with its weights."""
    weights = checkpoint["state_dict"]
    model = build_detector(wide=detector_is_wide(weights), input_size=detector_input_size(checkpoint),
                           anchor_min_ratio=detector_anchor_min_ratio(checkpoint))
    model.load_state_dict(weights, strict=True)
    return model


def torch_load(path: Path) -> dict[str, Any]:
    """Only load tensor/state-dict checkpoints (never arbitrary pickled objects)."""
    checkpoint = safe_torch_load(path)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return checkpoint


def stable_weights(repo: str, task: str, *, revision: str = "main") -> tuple[dict[str, Any], str]:
    """Download a SHA256-verified stable PyTorch checkpoint from Hugging Face.

    training/stable.json schema:
      {"schema_version": 1, "models": {"reader": {
          "path": "training/releases/v1/reader.pt", "sha256": "<hex>"}}}
    Updating stable.json is an explicit post-validation publishing operation.
    """
    if not repo:
        raise ValueError("continue requires a Hugging Face repository (--hf-repo)")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ValueError("continue requires huggingface-hub (install training dependencies)") from exc

    index_path = hf_hub_download(repo_id=repo, filename="training/stable.json", revision=revision)
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if index.get("schema_version") != 1 or not isinstance(index.get("models"), dict):
        raise ValueError("invalid Hugging Face training/stable.json")
    entry = index["models"].get(task)
    if not isinstance(entry, dict):
        raise ValueError(f"no stable training weights for {task!r} in {repo}")
    file_name, sha = entry.get("path"), entry.get("sha256")
    if not isinstance(file_name, str) or not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("stable manifest entry needs path and sha256")
    file_path = PurePosixPath(file_name)
    if (
        file_path.is_absolute()
        or ".." in file_path.parts
        or not file_path.parts
        or file_path.parts[0] != "training"
        or not all(part not in ("", ".") for part in file_path.parts)
    ):
        raise ValueError("invalid stable weight path")
    sha = sha.lower()
    if any(ch not in "0123456789abcdef" for ch in sha):
        raise ValueError("invalid stable sha256")
    downloaded = hf_hub_download(repo_id=repo, filename=file_name, revision=revision)
    digest = hashlib.sha256()
    with open(downloaded, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != sha:
        raise ValueError(f"stable weight hash mismatch: {file_name}")
    return torch_load(Path(downloaded)), f"{repo}@{revision}/{file_name}"


def model_weights(checkpoint: dict[str, Any]) -> dict[str, Any]:
    weights = checkpoint.get("state_dict", checkpoint)
    if not isinstance(weights, dict) or not weights:
        raise ValueError("no state_dict found in training weights")
    return weights
