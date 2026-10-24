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
