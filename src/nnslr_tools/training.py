"""Reader-baseline training for NNSLR.

This module is isolated from the core/data CLI import surface. PyTorch,
torchvision and Pillow are imported only after an explicit training request,
and so is the optional wandb tracking (nnslr_tools.tracking).
prepare_reader_training is stdlib-only and can validate the exact dataset/split
that would be consumed by a GPU run.

The first baseline trains only the sign reader on reviewed bounding-box crops.
It does not train a detector and it does not produce an on-vehicle bundle.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from nnslr_tools import tracking
from nnslr_tools.checkpoints import (DEFAULT_DETECTOR_ANCHOR_MIN_RATIO, DEFAULT_DETECTOR_INPUT_SIZE,
                                     DETECTOR_ANCHOR_MAX_RATIO, MAX_DETECTOR_INPUT_SIDE, atomic_torch_save,
                                     build_detector, detector_is_wide, model_weights, stable_weights, torch_load)
from nnslr_tools.progress import report as report_progress
from nnslr_tools.annotations import load_jsonl, validate_dataset
from nnslr_tools.splits import validate_splits

TRAINING_SCHEMA_VERSION = 1
ELIGIBLE_REVIEW_STATES = frozenset({"accepted", "corrected"})
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
# ImageNet normalization of the reader crops (also undone for the wandb crops).
READER_NORMALIZE_MEAN = (.485, .456, .406)
READER_NORMALIZE_STD = (.229, .224, .225)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reader_label(row: dict[str, Any]) -> str | None:
    """Return the classification label for a reviewed sign crop."""
    if row.get("review_state") not in ELIGIBLE_REVIEW_STATES:
        return None
    bbox = row.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    family = row.get("sign_family")
    if family == "not_a_sign":
        return None

    value_state = row.get("value_state")
    value = row.get("value_kph")
    if value_state == "read" and isinstance(value, int) and not isinstance(value, bool):
        return f"{family}:{value}"
    if family in {"cancellation", "unreadable", "other_sign"}:
        return str(family)
    if value_state in {"unreadable", "unknown", "not_applicable"}:
        return f"{family}:{value_state}"
    return None


def _label_basis(row: dict[str, Any]) -> str | None:
    """Who confirmed a row: human_review, ai_review or model_review (provenance.label_basis)."""
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        return None
    return provenance.get("label_basis")


def _safe_image_path(data_root: Path, image_path: str) -> Path:
    root = data_root.resolve()
    candidate = (root / image_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"unsafe_image_path: {image_path}")
    return candidate


def _load_split_document(split_path: Path, data_root: Path) -> tuple[Path, dict[str, Any]]:
    """Load a frozen split, following the build-splits latest.json pointer.

    The pointer itself is hash-bound to the immutable split file. A changed
    or escaped target is rejected before any training plan is produced.
    """
    requested = Path(split_path)
    document = json.loads(requested.read_text(encoding="utf-8"))
    if isinstance(document, dict) and isinstance(document.get("assignments"), dict):
        return requested, document

    if not isinstance(document, dict) or not isinstance(document.get("path"), str):
        raise ValueError("split_invalid: expected assignments or latest pointer")

    root = Path(data_root).resolve()
    target = Path(document["path"])
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    if target != root and root not in target.parents:
        raise ValueError("unsafe_split_path")
    if not target.is_file():
        raise ValueError(f"split_target_missing: {target}")

    expected_sha = document.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("split_pointer_invalid_hash")
    actual_sha = _sha256(target)
    if actual_sha != expected_sha:
        raise ValueError("split_pointer_hash_mismatch")

    resolved = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(resolved, dict) or not isinstance(resolved.get("assignments"), dict):
        raise ValueError("split_invalid: target has no assignments")
    return target, resolved


def prepare_reader_training(
    dataset_path: Path,
    split_path: Path,
    data_root: Path,
    *,
    dataset_kind: str = "gold",
) -> dict[str, Any]:
    """Validate inputs and build the immutable reader-training plan.

    No torch/Pillow import and no GPU access happens here.
    """
    dataset_path = Path(dataset_path)
    split_path = Path(split_path)
    data_root = Path(data_root)

    rows = load_jsonl(dataset_path)
    validation = validate_dataset(rows, data_root, dataset_kind=dataset_kind)
    if not validation.get("valid"):
        reasons = sorted({str(e.get("reason", "dataset_invalid")) for e in validation.get("errors", [])})
        raise ValueError("dataset_invalid: " + ",".join(reasons))

    resolved_split_path, split_doc = _load_split_document(split_path, data_root)
    split_errors = validate_splits(
        rows,
        split_doc,
        validation["dataset_sha256"],
        dataset_kind=dataset_kind,
    )
    if split_errors:
        reasons = sorted({str(e.get("reason", "split_invalid")) for e in split_errors})
        raise ValueError("split_invalid: " + ",".join(reasons))

    assignments = split_doc.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("split_invalid: assignments")

    records: list[dict[str, Any]] = []
    skipped = Counter()
    for row in rows:
        label = _reader_label(row)
        if label is None:
            skipped["not_reader_eligible"] += 1
            continue
        annotation_id = row.get("annotation_id")
        split_name = assignments.get(annotation_id)
        if not isinstance(split_name, str):
            skipped["missing_split"] += 1
            continue
        image_path = row.get("image_path")
        if not isinstance(image_path, str):
            skipped["missing_image_path"] += 1
            continue
        image = _safe_image_path(data_root, image_path)
        if not image.is_file():
            raise ValueError(f"missing_image: {image_path}")
        records.append({
            "annotation_id": annotation_id,
            "split": split_name,
            "image_path": image_path,
            "image_sha256": row.get("image_sha256"),
            "bbox_xyxy": row["bbox_xyxy"],
            "label": label,
            "sign_family": row.get("sign_family"),
            "value_state": row.get("value_state"),
            "value_kph": row.get("value_kph"),
            "route_id": row.get("route_id"),
            "segment_index": row.get("segment_index"),
            "decoded_frame_index": row.get("decoded_frame_index"),
            "label_basis": _label_basis(row),
        })

    train_labels = sorted({r["label"] for r in records if r["split"] == TRAIN_SPLIT})
    val_labels = sorted({r["label"] for r in records if r["split"] == VALIDATION_SPLIT})
    unseen_validation = sorted(set(val_labels) - set(train_labels))
    counts_by_split = Counter(r["split"] for r in records)
    counts_by_class = Counter(r["label"] for r in records if r["split"] == TRAIN_SPLIT)

    trainable = (
        counts_by_split[TRAIN_SPLIT] > 0
        and counts_by_split[VALIDATION_SPLIT] > 0
        and len(train_labels) >= 2
        and not unseen_validation
    )

    limitations = [
        "reader_only_no_detector",
        "bbox_crops_require_reviewed_localization",
        "no_runtime_bundle",
    ]
    if dataset_kind == "training-candidate":
        limitations.append("model_review_candidates_are_not_human_gold")

    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "task": "reader",
        "dataset_kind": dataset_kind,
        "dataset_path": str(dataset_path),
        "dataset_sha256": validation["dataset_sha256"],
        "split_path": str(resolved_split_path),
        "requested_split_path": str(split_path),
        "split_sha256": _sha256(resolved_split_path),
        "classes": train_labels,
        "records": records,
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "train_counts_by_class": dict(sorted(counts_by_class.items())),
        "skipped": dict(sorted(skipped.items())),
        "unseen_validation_labels": unseen_validation,
        "trainable": trainable,
        "limitations": limitations,
    }


DETECTOR_FAMILIES = frozenset({
    "maximum_speed",
    "cancellation",
    "zone",
    "variable_display",
    "unreadable",
})


def prepare_detector_training(
    dataset_path: Path,
    split_path: Path,
    data_root: Path,
    *,
    dataset_kind: str = "gold",
) -> dict[str, Any]:
    """Build a full-frame detector plan from reviewed annotations.

    Only vertical speed-sign families are positive detector boxes. A frame is a
    detector negative only when it has an explicit whole-frame not_a_sign
    review. other_sign/road_marking rows alone are not silently promoted to
    whole-frame negatives.
    """
    dataset_path = Path(dataset_path)
    split_path = Path(split_path)
    data_root = Path(data_root)

    rows = load_jsonl(dataset_path)
    validation = validate_dataset(rows, data_root, dataset_kind=dataset_kind)
    if not validation.get("valid"):
        reasons = sorted({str(e.get("reason", "dataset_invalid")) for e in validation.get("errors", [])})
        raise ValueError("dataset_invalid: " + ",".join(reasons))

    resolved_split_path, split_doc = _load_split_document(split_path, data_root)
    split_errors = validate_splits(
        rows,
        split_doc,
        validation["dataset_sha256"],
        dataset_kind=dataset_kind,
    )
    if split_errors:
        reasons = sorted({str(e.get("reason", "split_invalid")) for e in split_errors})
        raise ValueError("split_invalid: " + ",".join(reasons))
    assignments = split_doc["assignments"]

    grouped: dict[str, dict[str, Any]] = {}
    ignored = Counter()
    for row in rows:
        if row.get("review_state") not in ELIGIBLE_REVIEW_STATES:
            ignored["rejected_or_pending"] += 1
            continue
        annotation_id = row["annotation_id"]
        split_name = assignments.get(annotation_id)
        if not isinstance(split_name, str):
            ignored["missing_split"] += 1
            continue
        key = row["image_sha256"]
        frame = grouped.setdefault(key, {
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
            "width": row["width"],
            "height": row["height"],
            "split": split_name,
            "route_id": row["route_id"],
            "segment_index": row["segment_index"],
            "decoded_frame_index": row["decoded_frame_index"],
            "boxes": [],
            "annotation_ids": [],
            "explicit_negative": False,
            "label_basis": _label_basis(row),
        })
        if (
            frame["image_path"] != row["image_path"]
            or frame["split"] != split_name
            or frame["width"] != row["width"]
            or frame["height"] != row["height"]
        ):
            raise ValueError("inconsistent_frame_group")
        if frame["label_basis"] != _label_basis(row):
            # The same image confirmed by different kinds of reviewer: it counts as neither.
            frame["label_basis"] = "mixed"

        family = row["sign_family"]
        if row["bbox_xyxy"] is None and family == "not_a_sign":
            frame["explicit_negative"] = True
            frame["annotation_ids"].append(annotation_id)
        elif row["bbox_xyxy"] is not None and family in DETECTOR_FAMILIES:
            frame["boxes"].append(list(row["bbox_xyxy"]))
            frame["annotation_ids"].append(annotation_id)
        else:
            ignored["non_detector_label"] += 1

    frames = []
    for frame in grouped.values():
        if not frame["boxes"] and not frame["explicit_negative"]:
            continue
        image = _safe_image_path(data_root, frame["image_path"])
        if not image.is_file():
            raise ValueError(f"missing_image: {frame['image_path']}")
        frames.append(frame)
    frames.sort(key=lambda r: (r["route_id"], r["segment_index"], r["image_path"]))

    counts_by_split = Counter(r["split"] for r in frames)
    boxes_by_split = Counter()
    negatives_by_split = Counter()
    for frame in frames:
        boxes_by_split[frame["split"]] += len(frame["boxes"])
        if frame["explicit_negative"] and not frame["boxes"]:
            negatives_by_split[frame["split"]] += 1

    trainable = (
        counts_by_split[TRAIN_SPLIT] > 0
        and counts_by_split[VALIDATION_SPLIT] > 0
        and boxes_by_split[TRAIN_SPLIT] > 0
        and boxes_by_split[VALIDATION_SPLIT] > 0
    )

    limitations = [
        "single_class_vertical_speed_sign_detector",
        "road_markings_excluded",
        "other_sign_not_assumed_whole_frame_negative",
        "no_runtime_bundle",
    ]
    if dataset_kind == "training-candidate":
        limitations.append("model_review_candidates_are_not_human_gold")

    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "task": "detector",
        "dataset_kind": dataset_kind,
        "dataset_path": str(dataset_path),
        "dataset_sha256": validation["dataset_sha256"],
        "split_path": str(resolved_split_path),
        "requested_split_path": str(split_path),
        "split_sha256": _sha256(resolved_split_path),
        "classes": ["speed_sign"],
        "frames": frames,
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "boxes_by_split": dict(sorted(boxes_by_split.items())),
        "negative_frames_by_split": dict(sorted(negatives_by_split.items())),
        "ignored": dict(sorted(ignored.items())),
        "trainable": trainable,
        "limitations": limitations,
    }


def _tracking_config(plan: dict[str, Any], config: Any, architecture: str) -> dict[str, Any]:
    """Hyperparameters plus dataset identity for the wandb run config."""
    values = dict(config.__dict__)
    values["task"] = plan["task"]
    values["architecture"] = architecture
    values["dataset_kind"] = plan["dataset_kind"]
    values["dataset_sha256"] = plan["dataset_sha256"]
    values["split_sha256"] = plan["split_sha256"]
    values["classes"] = list(plan["classes"])
    values["counts_by_split"] = dict(plan["counts_by_split"])
    if "boxes_by_split" in plan:
        values["boxes_by_split"] = dict(plan["boxes_by_split"])
    return values


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    # Stop when the validation has not improved for this many epochs (0: never stop early).
    patience: int = 5
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    input_size: int = 160
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda"
    pretrained: bool = False
    amp: bool = True


def _require_training_stack():
    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError as exc:
        raise ValueError(
            "training_dependencies_missing: install a V100-compatible PyTorch/torchvision "
            "environment plus Pillow; training dependencies are intentionally not part of "
            "the core CPU profile"
        ) from exc
    return torch, Image, DataLoader, Dataset, transforms, MobileNet_V3_Small_Weights, mobilenet_v3_small


def gpu_smoke(device_index: int = 0) -> dict[str, Any]:
    """Run an explicit CUDA forward/backward smoke test.

    This function is never called implicitly. It verifies that the installed
    PyTorch binary can actually execute kernels for the selected GPU.
    """
    torch, *_ = _require_training_stack()
    if not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError(f"cuda_device_out_of_range: {device_index}")

    device = torch.device(f"cuda:{device_index}")
    capability = tuple(torch.cuda.get_device_capability(device))
    compiled_arches = list(torch.cuda.get_arch_list())
    expected_arch = f"sm_{capability[0]}{capability[1]}"
    if expected_arch not in compiled_arches:
        raise ValueError(
            f"cuda_arch_not_compiled: device={expected_arch} compiled={compiled_arches}"
        )

    # Small enough to be a smoke test, large enough to execute real FP16 GEMM
    # and autograd kernels. This intentionally allocates GPU memory.
    x = torch.randn((1024, 1024), device=device, dtype=torch.float16, requires_grad=True)
    y = (x @ x.T).mean()
    y.backward()
    torch.cuda.synchronize(device)

    props = torch.cuda.get_device_properties(device)
    return {
        "status": "available",
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "device_index": device_index,
        "device_name": props.name,
        "capability": list(capability),
        "compiled_arches": compiled_arches,
        "total_memory_bytes": int(props.total_memory),
        "fp16_forward_backward": True,
        "finite_result": bool(torch.isfinite(y.detach()).item()),
    }


@dataclass(frozen=True)
class DetectorTrainConfig:
    epochs: int = 20
    # Stop when the validation loss has not improved for this many epochs (0: never stop early).
    patience: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda"
    pretrained_backbone: bool = False
    amp: bool = True
    # The image the detector sees, (width, height): the frame is resized to it inside the model.
    input_size: tuple[int, int] = DEFAULT_DETECTOR_INPUT_SIZE
    # The smallest default box, as a fraction of the image side (nnslr_tools.checkpoints.build_detector).
    anchor_min_ratio: float = DEFAULT_DETECTOR_ANCHOR_MIN_RATIO
    # Photometric augmentation of the training frames (DETECTOR_AUGMENTATION).
    augment: bool = True
    # The share of the frames without signs that an epoch uses: torchvision's SSD takes its negative boxes only
    # inside frames with signs, so a frame without any gives no loss and only costs time. Each epoch draws its
    # own share; validation keeps one fixed draw, so its epochs compare. 1.0: every frame, as before.
    negative_frame_fraction: float = 0.1
    # What chooses the best epoch, and what early stopping waits for: "recall" (validation recall at score
    # SELECT_SCORE_THRESHOLD, ties to fewer false boxes per frame: the detector is the cutter, what it misses
    # is lost) or "loss" (validation loss, as before).
    select_by: str = "recall"


# The settings a detector run saved before they existed was trained with: resuming such a run compares
# against these.
DETECTOR_CONFIG_BEFORE_SETTINGS = {"input_size": DEFAULT_DETECTOR_INPUT_SIZE,
                                   "anchor_min_ratio": DEFAULT_DETECTOR_ANCHOR_MIN_RATIO, "augment": False,
                                   "negative_frame_fraction": 1.0, "select_by": "loss"}
DETECTOR_SELECT_BY = ("recall", "loss")
# The validation measure of select_by "recall": a reviewed box is found by a box of score >= this with IoU >= 0.5.
SELECT_SCORE_THRESHOLD = 0.1
SELECT_IOU_THRESHOLD = 0.5


def epoch_frame_indices(frames: list[dict[str, Any]], fraction: float, seed: Any) -> list[int]:
    """The frames an epoch (or the validation) uses, in their order: every frame with signs, and
    round(fraction x the frames without) of those without, drawn with seed."""
    with_signs = [index for index, frame in enumerate(frames) if frame["boxes"]]
    sign_free = [index for index, frame in enumerate(frames) if not frame["boxes"]]
    count = min(len(sign_free), int(fraction * len(sign_free) + 0.5))
    chosen = sign_free if count == len(sign_free) else random.Random(seed).sample(sign_free, count)
    return sorted(with_signs + chosen)

# Photometric augmentation of the detector's training frames, never of validation, evaluation or export. The
# test route is mostly at dusk (35% of its frames are dark, 7% in train), so the brightness range leans to
# darkening. No flip and no geometric change: the side of the road a sign stands on matters, and the boxes stay
# where they are. Each frame is changed with this probability, every factor drawn uniformly in its range.
DETECTOR_AUGMENTATION = {
    "probability": 0.8,
    "brightness": (0.35, 1.25),
    "contrast": (0.7, 1.3),
    "gamma": (0.8, 1.6),
    "saturation": (0.7, 1.3),
    # A light colour cast: each of R, G and B multiplied by its own factor.
    "channel_gain": (0.92, 1.08),
}


def detector_photometric(torch, TF, image, generator):
    """A training frame (3 x H x W, values 0..1, on any device) with random gamma, brightness, contrast,
    saturation and colour cast (DETECTOR_AUGMENTATION); the factors come from the CPU generator."""
    def draw(name, count=1):
        low, high = DETECTOR_AUGMENTATION[name]
        return (torch.rand(count, generator=generator, dtype=torch.float64) * (high - low) + low).tolist()

    apply = float(torch.rand(1, generator=generator)) < DETECTOR_AUGMENTATION["probability"]
    # Every factor is drawn either way, so each frame uses the same amount of the generator.
    gamma, brightness, contrast, saturation = (draw(name)[0] for name in ("gamma", "brightness", "contrast",
                                                                           "saturation"))
    gains = draw("channel_gain", 3)
    if not apply:
        return image
    image = TF.adjust_gamma(image, gamma)
    image = TF.adjust_brightness(image, brightness)
    image = TF.adjust_contrast(image, contrast)
    image = TF.adjust_saturation(image, saturation)
    gains = torch.tensor(gains, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image * gains).clamp_(0.0, 1.0)


def _check_detector_config(config: "DetectorTrainConfig") -> None:
    size = config.input_size
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or not all(type(side) is int and 32 <= side <= MAX_DETECTOR_INPUT_SIDE for side in size)
    ):
        raise ValueError(f"invalid detector input_size {size!r}: (width, height), 32 to {MAX_DETECTOR_INPUT_SIDE}")
    ratio = config.anchor_min_ratio
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 < ratio < DETECTOR_ANCHOR_MAX_RATIO:
        raise ValueError(f"invalid detector anchor_min_ratio {ratio!r}: between 0 and {DETECTOR_ANCHOR_MAX_RATIO}")
    fraction = config.negative_frame_fraction
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
        raise ValueError(f"invalid detector negative_frame_fraction {fraction!r}: from 0 to 1")
    if config.select_by not in DETECTOR_SELECT_BY:
        raise ValueError(f"invalid detector select_by {config.select_by!r}: one of {', '.join(DETECTOR_SELECT_BY)}")


def _detector_config_differences(saved: Any, config: "DetectorTrainConfig") -> list[str]:
    """The settings of a saved detector run that differ from config (a run saved before a setting existed
    was trained with DETECTOR_CONFIG_BEFORE_SETTINGS)."""
    before = {**DETECTOR_CONFIG_BEFORE_SETTINGS, **(saved if isinstance(saved, dict) else {})}
    if isinstance(before.get("input_size"), list):
        before["input_size"] = tuple(before["input_size"])
    current = config.__dict__
    return [f"{key} {before.get(key)!r} -> {current.get(key)!r}" for key in sorted(set(before) | set(current))
            if before.get(key) != current.get(key)]


# The flags of nnslr train for each setting of a saved configuration (bool settings are switches).
_RESUME_FLAGS = (("epochs", "--epochs"), ("patience", "--patience"), ("batch_size", "--batch-size"),
                 ("learning_rate", "--learning-rate"), ("weight_decay", "--weight-decay"),
                 ("num_workers", "--num-workers"), ("seed", "--seed"), ("device", "--device"),
                 ("anchor_min_ratio", "--anchor-min-ratio"), ("negative_frame_fraction", "--negative-frame-fraction"),
                 ("select_by", "--select-by"))


def resume_arguments(state: dict[str, Any]) -> list[str]:
    """The nnslr train flags that give a saved run (its training-state.pt) back its configuration: a resumed
    run must train with exactly the settings it started with, ImageNet start (--pretrained) included."""
    task = state.get("task") if isinstance(state, dict) else None
    config = state.get("config") if isinstance(state, dict) else None
    if task not in ("detector", "reader") or not isinstance(config, dict):
        raise ValueError("invalid training state: no task or configuration")
    if task == "detector":
        config = {**DETECTOR_CONFIG_BEFORE_SETTINGS, **config}
    flags: list[str] = []
    for key, flag in _RESUME_FLAGS:
        if key in config:
            flags += [flag, str(config[key])]
    size = config.get("input_size")
    if task == "detector" and size is not None:
        flags += ["--input-size", f"{int(size[0])}x{int(size[1])}"]
    elif size is not None:
        flags += ["--input-size", str(int(size))]
    if config.get("pretrained_backbone" if task == "detector" else "pretrained"):
        flags.append("--pretrained")
    if config.get("amp") is False:
        flags.append("--no-amp")
    if task == "detector" and config.get("augment") is False:
        flags.append("--no-augment")
    return flags


def resume_arguments_file(path: Path) -> list[str]:
    """resume_arguments of a training-state.pt (scripts/run_v100_baseline.sh, scripts/run_detector_sizes.sh)."""
    return resume_arguments(torch_load(Path(path)))


def _require_detector_stack():
    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision.transforms import functional as TF
        from torchvision.models import MobileNet_V3_Large_Weights
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    except ImportError as exc:
        raise ValueError(
            "training_dependencies_missing: detector training needs a V100-compatible "
            "PyTorch/torchvision environment plus Pillow"
        ) from exc
    return (
        torch,
        Image,
        DataLoader,
        Dataset,
        TF,
        MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )


def train_detector(
    plan: dict[str, Any],
    data_root: Path,
    output_dir: Path,
    config: DetectorTrainConfig,
    *,
    mode: str = "fresh",
    hf_repo: str | None = None,
    hf_revision: str = "main",
    init_from: Path | None = None,
) -> dict[str, Any]:
    """Train the first single-class full-frame speed-sign detector baseline."""
    _check_detector_config(config)
    if not plan.get("trainable"):
        raise ValueError(
            "training_plan_not_trainable: detector needs positive train and validation frames"
        )
    # The measure of the validation (evaluation imports this module: imported here, not at the top).
    from nnslr_tools.evaluation import detector_metrics, score_frame

    (
        torch,
        Image,
        DataLoader,
        Dataset,
        TF,
        BackboneWeights,
        ssdlite320_mobilenet_v3_large,
    ) = _require_detector_stack()
    import torchvision

    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    device = torch.device(config.device)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    root = Path(data_root).resolve()

    class DetectionDataset(Dataset):
        def __init__(self, frames):
            self.frames = frames

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, index):
            row = self.frames[index]
            path = _safe_image_path(root, row["image_path"])
            with Image.open(path) as source:
                image = source.convert("RGB")
                tensor = TF.pil_to_tensor(image).float().div_(255.0)

            boxes = torch.tensor(row["boxes"], dtype=torch.float32)
            if boxes.numel() == 0:
                boxes = boxes.reshape(0, 4)
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([index], dtype=torch.int64),
            }
            return tensor, target

    def collate(batch):
        return tuple(zip(*batch))

    train_frames = [r for r in plan["frames"] if r["split"] == TRAIN_SPLIT]
    all_val_frames = [r for r in plan["frames"] if r["split"] == VALIDATION_SPLIT]
    # Validation: every frame with signs and one fixed draw of those without, the same frames every epoch.
    val_frames = [all_val_frames[index] for index in
                  epoch_frame_indices(all_val_frames, config.negative_frame_fraction, f"validation {config.seed}")]
    val_dataset = DetectionDataset(val_frames)
    generator = torch.Generator().manual_seed(config.seed)

    class EpochSampler(torch.utils.data.Sampler):
        """The frames of the current epoch (set_epoch) in random order. With every frame (fraction 1.0) the
        order is the one of shuffle=True with the same generator."""

        def __init__(self):
            self.indices = list(range(len(train_frames)))

        def set_epoch(self, epoch: int) -> None:
            self.indices = epoch_frame_indices(train_frames, config.negative_frame_fraction,
                                               f"train {config.seed} {epoch}")

        def __len__(self):
            # A last batch of one frame fails in BatchNorm (the last feature map is 1 x 1 at 320 x 320): that
            # frame, the last of the shuffle, waits for another epoch.
            count = len(self.indices)
            return count - 1 if count > 1 and count % config.batch_size == 1 else count

        def __iter__(self):
            order = torch.randperm(len(self.indices), generator=generator).tolist()
            return iter([self.indices[position] for position in order][:len(self)])

    train_sampler = EpochSampler()
    train_sampler.set_epoch(0)
    train_loader = DataLoader(
        DetectionDataset(train_frames),
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        generator=generator,
    )
    sign_free_train = sum(1 for frame in train_frames if not frame["boxes"])
    frame_counts = {
        "train_with_signs": len(train_frames) - sign_free_train,
        "train_sign_free": sign_free_train,
        "train_sign_free_per_epoch": len(train_sampler.indices) - (len(train_frames) - sign_free_train),
        "train_frames_per_epoch": len(train_sampler),
        "validation_with_signs": sum(1 for frame in all_val_frames if frame["boxes"]),
        "validation_sign_free": sum(1 for frame in all_val_frames if not frame["boxes"]),
        "validation_sign_free_used": sum(1 for frame in val_frames if not frame["boxes"]),
    }
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    # ImageNet backbone weights for a fresh run; continue and resume load their own weights below.
    geometry = {"input_size": config.input_size, "anchor_min_ratio": config.anchor_min_ratio}
    model = build_detector(wide=config.pretrained_backbone and mode == "fresh", **geometry)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = bool(config.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "training-state.pt"
    if mode not in ("fresh", "continue", "resume"):
        raise ValueError(f"invalid training mode: {mode}")
    if mode == "resume" and not checkpoint_path.is_file():
        raise ValueError(f"no resumable training checkpoint: {checkpoint_path}")
    plan_path = output_dir / "training-plan.json"

    def move_targets(targets):
        return [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]

    def validate() -> dict[str, Any]:
        """The validation loss, and the boxes the cutter passes on scored as nnslr evaluate scores them:
        recall at SELECT_SCORE_THRESHOLD, false boxes and boxes per frame. One pass over the frames."""
        total = 0.0
        seen = 0
        scored = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device, non_blocking=True) for img in images]
                # torchvision detection models expose training losses only in train mode.
                # Keep BatchNorm frozen so validation does not mutate running statistics.
                model.train()
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    losses = model(images, move_targets(targets))
                    loss = sum(losses.values())
                total += float(loss.detach()) * len(images)
                seen += len(images)
                # The boxes, in float32 like the evaluation.
                model.eval()
                for detection, target in zip(model(images), targets):
                    keep = detection["labels"] == 1
                    predictions = [{"box": box, "score": score} for box, score in
                                   zip(detection["boxes"][keep].float().cpu().tolist(),
                                       detection["scores"][keep].float().cpu().tolist())]
                    scored.append(score_frame(target["boxes"].tolist(), predictions,
                                              score_threshold=SELECT_SCORE_THRESHOLD,
                                              iou_threshold=SELECT_IOU_THRESHOLD))
        model.train()
        measured = detector_metrics(scored)
        return {
            "loss": total / max(seen, 1),
            "recall": measured["recall_iou"],
            "true_positives": measured["true_positives"],
            "false_positives": measured["false_positives"],
            "false_positives_per_image": measured["false_positives_per_image"],
            "predictions_per_image": measured["predictions_per_image"],
        }

    def selection_key(measured: dict[str, Any]) -> tuple | None:
        """Higher is better; None never counts as better (a validation loss that is not a number)."""
        if config.select_by == "recall":
            # The same validation frames every epoch: counts compare as recall and false boxes per frame do.
            return (measured["true_positives"], -measured["false_positives"])
        return (-measured["loss"],) if math.isfinite(measured["loss"]) else None

    def detector_samples() -> list[Any]:
        # The first validation frames with reviewed boxes (fixed order, so the
        # same frames every time), with predicted and reviewed boxes.
        chosen = []
        for index in range(len(val_frames)):
            if len(val_frames[index]["boxes"]) > 0:
                chosen.append(index)
            if len(chosen) >= tracking.DETECTOR_IMAGE_SAMPLES:
                break
        samples = []
        model.eval()
        with torch.no_grad():
            for index in chosen:
                frame = val_frames[index]
                tensor, _ = val_dataset[index]
                prediction = model([tensor.to(device)])[0]
                height = int(tensor.shape[1])
                width = int(tensor.shape[2])
                predicted = tracking.detector_box_data(
                    prediction["boxes"].detach().float().cpu().tolist(),
                    width,
                    height,
                    scores=prediction["scores"].detach().float().cpu().tolist(),
                )
                reviewed = tracking.detector_box_data(frame["boxes"], width, height)
                boxes = {
                    "predictions": {"box_data": predicted, "class_labels": tracking.DETECTOR_CLASS_LABELS},
                    "ground_truth": {"box_data": reviewed, "class_labels": tracking.DETECTOR_CLASS_LABELS},
                }
                image = TF.to_pil_image(tensor)
                image.thumbnail((tracking.IMAGE_MAX_SIDE, tracking.IMAGE_MAX_SIDE))
                caption = (
                    f"{frame['image_path']}: {len(reviewed)} reviewed, {len(predicted)} predicted "
                    f"(score >= {tracking.DETECTOR_IMAGE_MIN_SCORE})"
                )
                samples.append(tracker.image(image, caption, boxes=boxes))
        model.train()
        return samples

    history = []
    best_validation_loss = float("inf")
    # The epoch whose weights are detector-best.pt: its selection key and its validation measures.
    best_selection: dict[str, Any] | None = None
    stale = 0
    best_path = output_dir / "detector-best.pt"
    global_step = 0

    def match_detector_tail(model, optimizer, weights):
        # Weights trained from ImageNet have the full tail: a run continued or resumed from them needs that shape.
        if detector_is_wide(weights) == detector_is_wide(model.state_dict()):
            return model, optimizer
        model = build_detector(wide=detector_is_wide(weights), **geometry).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        return model, optimizer

    start_epoch = 0
    if mode == "continue":
        if init_from is not None:
            # The best weights of an earlier run of this server, checked as the stable ones are.
            saved, origin = torch_load(Path(init_from)), str(init_from)
        elif not hf_repo:
            raise ValueError("continue requires --hf-repo or --init-from")
        else:
            saved, origin = stable_weights(hf_repo, "detector", revision=hf_revision)
        if saved.get("task") != "detector" or saved.get("architecture") != "ssdlite320_mobilenet_v3_large":
            raise ValueError("stable weight task/architecture mismatch")
        if saved.get("classes") != ["background", "speed_sign"]:
            raise ValueError("stable detector classes mismatch")
        model, optimizer = match_detector_tail(model, optimizer, model_weights(saved))
        model.load_state_dict(model_weights(saved), strict=True)
        print(f"continue detector from weights: {origin}")
        # No weight depends on the input size or on the default boxes: the new run uses its own.
        saved_size = tuple(saved.get("input_size") or DEFAULT_DETECTOR_INPUT_SIZE)
        if saved_size != config.input_size:
            print(f"the weights were trained at {saved_size[0]}x{saved_size[1]}; this run trains at "
                  f"{config.input_size[0]}x{config.input_size[1]}")
    elif mode == "resume":
        saved = torch_load(checkpoint_path)
        if saved.get("format_version") != 1 or saved.get("task") != "detector":
            raise ValueError("invalid detector training-state checkpoint")
        if saved.get("status") == "completed":
            raise ValueError("training already completed; use --mode continue for a new run")
        if saved.get("status") != "in_progress":
            raise ValueError("invalid checkpoint status")
        differences = _detector_config_differences(saved.get("config"), config)
        if differences:
            raise ValueError("resume requires exactly the original training configuration: " + "; ".join(differences))
        if saved.get("dataset_sha256") != plan["dataset_sha256"] or saved.get("split_sha256") != plan["split_sha256"]:
            raise ValueError("resume dataset/split mismatch")
        start_epoch = saved.get("epoch")
        if not isinstance(start_epoch, int) or not 0 <= start_epoch < config.epochs:
            raise ValueError("invalid or already completed resume epoch")
        model, optimizer = match_detector_tail(model, optimizer, model_weights(saved))
        model.load_state_dict(model_weights(saved), strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        global_step = saved["global_step"]
        history = saved["history"]
        best_validation_loss = saved["best_validation_loss"]
        best_selection = saved.get("best_selection")
        if best_selection is None and math.isfinite(best_validation_loss):
            # A run saved before best_selection existed chose its best epoch by validation loss.
            best_selection = {"key": [-best_validation_loss], "validation_loss": best_validation_loss}
        stale = int(saved.get("stale", 0))
        print(f"resume detector at epoch {start_epoch}/{config.epochs}")

    def save_training_state(epoch: int, status: str) -> None:
        atomic_torch_save({
            "format_version": 1,
            "task": "detector",
            "architecture": "ssdlite320_mobilenet_v3_large",
            "status": status,
            "epoch": epoch,
            "global_step": global_step,
            "config": config.__dict__,
            "dataset_sha256": plan["dataset_sha256"],
            "split_sha256": plan["split_sha256"],
            "classes": ["background", "speed_sign"],
            "input_size": list(config.input_size),
            "anchor_min_ratio": config.anchor_min_ratio,
            "best_validation_loss": best_validation_loss,
            "best_selection": best_selection,
            "stale": stale,
            "history": history,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }, checkpoint_path)

    if mode != "resume":
        output_dir.mkdir(parents=True, exist_ok=False)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        save_training_state(0, "in_progress")

    tracker = tracking.start_training_run(
        "detector",
        output_dir,
        _tracking_config(plan, config, "ssdlite320_mobilenet_v3_large"),
    )
    print(f"detector input {config.input_size[0]}x{config.input_size[1]}, smallest default box "
          f"{config.anchor_min_ratio:g} of the image, photometric augmentation {'on' if config.augment else 'off'}",
          flush=True)
    # The factors of the photometric augmentation, drawn here (not in the loader workers): the same for any
    # number of workers, and reproduced after a resume.
    augment_generator = torch.Generator().manual_seed(config.seed)
    print(f"detector frames per epoch: {frame_counts['train_with_signs']} with signs and "
          f"{frame_counts['train_sign_free_per_epoch']} of {frame_counts['train_sign_free']} without; validation "
          f"{frame_counts['validation_with_signs']} with signs and {frame_counts['validation_sign_free_used']} of "
          f"{frame_counts['validation_sign_free']} without; best epoch by "
          + (f"validation recall at score {SELECT_SCORE_THRESHOLD:g}" if config.select_by == "recall"
             else "validation loss"), flush=True)
    try:
        stopped_early = None
        for epoch in range(start_epoch, config.epochs):
            epoch_number = epoch + 1
            epoch_start = time.perf_counter()
            # Reproduce each epoch's frames and DataLoader shuffle after an epoch-boundary resume.
            train_sampler.set_epoch(epoch)
            generator.manual_seed(config.seed + epoch)
            augment_generator.manual_seed(config.seed + 1_000_003 * (epoch + 1))
            model.train()
            train_total = 0.0
            train_seen = 0
            # The step time includes the wait for the batch, like distill's step_time.
            step_start = time.perf_counter()
            batch_count = len(train_loader)
            batch_index = 0
            for images, targets in train_loader:
                # Where the training is, batch by batch, for the panel of the review site.
                batch_index += 1
                report_progress("detector", epoch * batch_count + batch_index, config.epochs * batch_count,
                                f"epoch {epoch_number} of {config.epochs}, batch {batch_index} of {batch_count}")
                images = [img.to(device, non_blocking=True) for img in images]
                if config.augment:
                    images = [detector_photometric(torch, TF, image, augment_generator) for image in images]
                targets = move_targets(targets)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    losses = model(images, targets)
                    loss = sum(losses.values())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step_loss = float(loss.detach())
                train_total += step_loss * len(images)
                train_seen += len(images)
                global_step += 1
                step_end = time.perf_counter()

                if tracker.enabled:
                    step_metrics = {"train/loss": step_loss}
                    for name in losses:
                        step_metrics["train/loss_" + name] = float(losses[name].detach())
                    step_metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                    step_metrics["train/step_time"] = step_end - step_start
                    tracker.log(step_metrics, step=global_step)
                step_start = step_end

            measured = validate()
            val_loss = measured["loss"]
            epoch_result = {
                "epoch": epoch_number,
                "train_loss": train_total / max(train_seen, 1),
                "validation_loss": val_loss,
                # The cutter on the validation frames, at score SELECT_SCORE_THRESHOLD.
                "validation_recall": measured["recall"],
                "validation_false_positives_per_image": measured["false_positives_per_image"],
                "validation_predictions_per_image": measured["predictions_per_image"],
                # Training and validation of the epoch, for the cost of an input size.
                "seconds": time.perf_counter() - epoch_start,
            }
            history.append(epoch_result)

            key = selection_key(measured)
            improved = key is not None and (best_selection is None or list(key) > best_selection["key"])
            if improved:
                best_validation_loss = val_loss
                best_selection = {"key": list(key), "epoch": epoch_number, "validation_loss": val_loss,
                                  "validation_recall": measured["recall"],
                                  "validation_false_positives_per_image": measured["false_positives_per_image"]}
                torch.save({
                    "format_version": 1,
                    "architecture": "ssdlite320_mobilenet_v3_large",
                    "task": "detector",
                    "classes": ["background", "speed_sign"],
                    # What load_detector rebuilds the detector with (nnslr_tools.checkpoints).
                    "input_size": list(config.input_size),
                    "anchor_min_ratio": config.anchor_min_ratio,
                    "state_dict": model.state_dict(),
                    "dataset_sha256": plan["dataset_sha256"],
                    "split_sha256": plan["split_sha256"],
                    "epoch": epoch_number,
                    "validation_loss": best_validation_loss,
                    "validation_recall": measured["recall"],
                    "validation_false_positives_per_image": measured["false_positives_per_image"],
                    "selected_by": config.select_by,
                    "initialization": "imagenet1k_v1" if config.pretrained_backbone else "random",
                    "initialization_source": (
                        "torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1"
                        if config.pretrained_backbone else "random"
                    ),
                    "torch_version": str(torch.__version__),
                    "torchvision_version": torchvision.__version__,
                }, best_path)

            if tracker.enabled:
                epoch_metrics = {
                    "epoch": epoch_number,
                    "train/epoch_loss": epoch_result["train_loss"],
                    "validation/loss": val_loss,
                    "validation/best_loss": best_validation_loss,
                    "validation/recall": measured["recall"],
                    "validation/false_per_image": measured["false_positives_per_image"],
                }
                if tracker.should_log_images(epoch_number, config.epochs):
                    epoch_metrics["validation/detections"] = detector_samples()
                epoch_metrics["epoch_time"] = time.perf_counter() - epoch_start
                tracker.log(epoch_metrics, step=global_step)
            # Early stop: the best weights are already saved; more epochs would only memorise the training images.
            stale = 0 if improved else stale + 1
            save_training_state(epoch_number, "in_progress")
            # One line per epoch: the panel of the review site reads the progress from the log (the seconds last).
            print(f"detector epoch {epoch_number}/{config.epochs} train_loss {epoch_result['train_loss']:.4f} "
                  f"val_loss {val_loss:.4f} val_recall {measured['recall']:.4f} "
                  f"{time.perf_counter() - epoch_start:.0f}s", flush=True)
            if config.patience and stale >= config.patience and epoch_number < config.epochs:
                stopped_early = epoch_number
                measure = (f"validation recall at score {SELECT_SCORE_THRESHOLD:g}" if config.select_by == "recall"
                           else "validation loss")
                print(f"detector stops early after epoch {epoch_number}: {measure} not better for {stale} epochs "
                      f"(best epoch {(best_selection or {}).get('epoch', '-')}, "
                      f"val_loss {best_validation_loss:.4f}, "
                      f"val_recall {(best_selection or {}).get('validation_recall', float('nan')):.4f})", flush=True)
                break

        gpu = None
        if device.type == "cuda":
            gpu = {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
                # The most memory the tensors of this process held (since the start or the resume).
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }

        result = {
            "task": "detector",
            "architecture": "ssdlite320_mobilenet_v3_large",
            "classes": ["background", "speed_sign"],
            "input_size": list(config.input_size),
            "anchor_min_ratio": config.anchor_min_ratio,
            "augmentation": ({key: list(value) if isinstance(value, tuple) else value
                              for key, value in DETECTOR_AUGMENTATION.items()} if config.augment else None),
            # The validation measures of the epoch whose weights are detector-best.pt (chosen by selection.by).
            "best_validation_loss": best_validation_loss,
            "best_epoch": (best_selection or {}).get("epoch") or next(
                (item["epoch"] for item in history if item["validation_loss"] == best_validation_loss), None),
            "best_validation_recall": (best_selection or {}).get("validation_recall"),
            "best_validation_false_positives_per_image": (best_selection or {}).get(
                "validation_false_positives_per_image"),
            "selection": {"by": config.select_by, "score_threshold": SELECT_SCORE_THRESHOLD,
                          "iou_threshold": SELECT_IOU_THRESHOLD,
                          "ties": "fewer false boxes per frame" if config.select_by == "recall" else None},
            "frames": frame_counts,
            "stopped_early_after_epoch": stopped_early,
            "checkpoint": str(best_path),
            "training_plan": str(plan_path),
            "device": str(device),
            "gpu": gpu,
            "torch_version": str(torch.__version__),
            "torchvision_version": torchvision.__version__,
            "initialization": "imagenet1k_v1" if config.pretrained_backbone else "random",
            "initialization_source": (
                "torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1"
                if config.pretrained_backbone else "random"
            ),
            "config": config.__dict__,
            "history": history,
            "limitations": plan["limitations"],
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tracker.set_summary({
            "best_validation_loss": best_validation_loss,
            "best_validation_recall": (best_selection or {}).get("validation_recall"),
            "checkpoint": str(best_path),
        })
        save_training_state(config.epochs, "completed")
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    tracker.finish(exit_code=0)
    return result


def train_reader(
    plan: dict[str, Any],
    data_root: Path,
    output_dir: Path,
    config: TrainConfig,
    *,
    mode: str = "fresh",
    hf_repo: str | None = None,
    hf_revision: str = "main",
    init_from: Path | None = None,
) -> dict[str, Any]:
    """Execute the first reader baseline after explicit compute opt-in."""
    if not plan.get("trainable"):
        raise ValueError(
            "training_plan_not_trainable: need >=2 train classes, non-empty train/validation, "
            "and no validation-only classes"
        )

    torch, Image, DataLoader, Dataset, transforms, Weights, mobilenet_v3_small = _require_training_stack()
    import torchvision

    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    device = torch.device(config.device)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    classes = list(plan["classes"])
    class_to_index = {name: idx for idx, name in enumerate(classes)}
    root = Path(data_root).resolve()

    class CropDataset(Dataset):
        def __init__(self, records, *, training: bool):
            self.records = records
            pipeline = [transforms.Resize((config.input_size, config.input_size))]
            if training:
                pipeline.extend([
                    transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
                    transforms.RandomAffine(degrees=4, translate=(.03, .03), scale=(.95, 1.05)),
                ])
            pipeline.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=READER_NORMALIZE_MEAN, std=READER_NORMALIZE_STD),
            ])
            self.transform = transforms.Compose(pipeline)

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            row = self.records[index]
            image_path = _safe_image_path(root, row["image_path"])
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                x1, y1, x2, y2 = [int(v) for v in row["bbox_xyxy"]]
                crop = image.crop((x1, y1, x2, y2))
            return self.transform(crop), class_to_index[row["label"]]

    train_records = [r for r in plan["records"] if r["split"] == TRAIN_SPLIT]
    val_records = [r for r in plan["records"] if r["split"] == VALIDATION_SPLIT]
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        CropDataset(train_records, training=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        CropDataset(val_records, training=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    weights = Weights.IMAGENET1K_V1 if config.pretrained and mode == "fresh" else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    use_amp = bool(config.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "training-state.pt"
    if mode not in ("fresh", "continue", "resume"):
        raise ValueError(f"invalid training mode: {mode}")
    if mode == "resume" and not checkpoint_path.is_file():
        raise ValueError(f"no resumable training checkpoint: {checkpoint_path}")
    plan_path = output_dir / "training-plan.json"

    def reader_samples(inputs, targets, logits) -> list[Any]:
        # The first crops of the first validation batch (no shuffling, so the
        # same crops every time) with the reviewed and the predicted label.
        to_image = transforms.ToPILImage()
        mean = torch.tensor(READER_NORMALIZE_MEAN).view(3, 1, 1)
        std = torch.tensor(READER_NORMALIZE_STD).view(3, 1, 1)
        expected = targets.detach().cpu().tolist()
        predicted = logits.detach().argmax(1).cpu().tolist()
        count = min(tracking.READER_IMAGE_SAMPLES, len(expected))
        samples = []
        for index in range(count):
            pixels = inputs[index].detach().float().cpu() * std + mean
            image = to_image(pixels.clamp(0.0, 1.0))
            caption = f"true {classes[expected[index]]} / pred {classes[predicted[index]]}"
            samples.append(tracker.image(image, caption))
        return samples

    history = []
    best_accuracy = -1.0
    best_path = output_dir / "reader-best.pt"
    global_step = 0

    start_epoch = 0
    if mode == "continue":
        if init_from is not None:
            # The best weights of an earlier run of this server, checked as the stable ones are.
            saved, origin = torch_load(Path(init_from)), str(init_from)
        elif not hf_repo:
            raise ValueError("continue requires --hf-repo or --init-from")
        else:
            saved, origin = stable_weights(hf_repo, "reader", revision=hf_revision)
        if saved.get("task") != "reader" or saved.get("architecture") != "mobilenet_v3_small":
            raise ValueError("stable weight task/architecture mismatch")
        if saved.get("classes") != classes or saved.get("input_size") != config.input_size:
            raise ValueError("stable reader classes/input size mismatch")
        model.load_state_dict(model_weights(saved), strict=True)
        print(f"continue reader from weights: {origin}")
    elif mode == "resume":
        saved = torch_load(checkpoint_path)
        if saved.get("format_version") != 1 or saved.get("task") != "reader":
            raise ValueError("invalid reader training-state checkpoint")
        if saved.get("status") == "completed":
            raise ValueError("training already completed; use --mode continue for a new run")
        if saved.get("status") != "in_progress":
            raise ValueError("invalid checkpoint status")
        if saved.get("config") != config.__dict__:
            raise ValueError("resume requires exactly the original training configuration")
        if saved.get("dataset_sha256") != plan["dataset_sha256"] or saved.get("split_sha256") != plan["split_sha256"]:
            raise ValueError("resume dataset/split mismatch")
        if saved.get("classes") != classes:
            raise ValueError("resume reader classes mismatch")
        start_epoch = saved.get("epoch")
        if not isinstance(start_epoch, int) or not 0 <= start_epoch < config.epochs:
            raise ValueError("invalid or already completed resume epoch")
        model.load_state_dict(model_weights(saved), strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        global_step = saved["global_step"]
        history = saved["history"]
        best_accuracy = saved["best_accuracy"]
        print(f"resume reader at epoch {start_epoch}/{config.epochs}")

    def save_training_state(epoch: int, status: str) -> None:
        atomic_torch_save({
            "format_version": 1,
            "task": "reader",
            "architecture": "mobilenet_v3_small",
            "status": status,
            "epoch": epoch,
            "global_step": global_step,
            "config": config.__dict__,
            "dataset_sha256": plan["dataset_sha256"],
            "split_sha256": plan["split_sha256"],
            "classes": classes,
            "input_size": config.input_size,
            "best_accuracy": best_accuracy,
            "history": history,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }, checkpoint_path)

    if mode != "resume":
        output_dir.mkdir(parents=True, exist_ok=False)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        save_training_state(0, "in_progress")

    tracker = tracking.start_training_run(
        "reader",
        output_dir,
        _tracking_config(plan, config, "mobilenet_v3_small"),
    )
    try:
        stale = 0
        stopped_early = None
        for epoch in range(start_epoch, config.epochs):
            epoch_number = epoch + 1
            epoch_start = time.perf_counter()
            # Reproduce each epoch's DataLoader shuffle after an epoch-boundary resume.
            generator.manual_seed(config.seed + epoch)
            model.train()
            train_loss = 0.0
            train_seen = 0
            train_correct = 0
            # The step time includes the wait for the batch, like distill's step_time.
            step_start = time.perf_counter()
            batch_count = len(train_loader)
            batch_index = 0
            for inputs, targets in train_loader:
                # Where the training is, batch by batch, for the panel of the review site.
                batch_index += 1
                report_progress("reader", epoch * batch_count + batch_index, config.epochs * batch_count,
                                f"epoch {epoch_number} of {config.epochs}, batch {batch_index} of {batch_count}")
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step_loss = float(loss.detach())
                step_seen = targets.numel()
                step_correct = int((logits.argmax(1) == targets).sum())
                train_loss += step_loss * step_seen
                train_seen += step_seen
                train_correct += step_correct
                global_step += 1
                step_end = time.perf_counter()

                if tracker.enabled:
                    tracker.log({
                        "train/loss": step_loss,
                        "train/accuracy": step_correct / max(step_seen, 1),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/step_time": step_end - step_start,
                    }, step=global_step)
                step_start = step_end

            want_images = tracker.should_log_images(epoch_number, config.epochs)
            sample_crops = []
            model.eval()
            val_loss = 0.0
            val_seen = 0
            val_correct = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = model(inputs)
                        loss = criterion(logits, targets)
                    val_loss += float(loss.detach()) * targets.numel()
                    val_seen += targets.numel()
                    val_correct += int((logits.argmax(1) == targets).sum())
                    if want_images and len(sample_crops) == 0:
                        sample_crops = reader_samples(inputs, targets, logits)

            epoch_result = {
                "epoch": epoch_number,
                "train_loss": train_loss / max(train_seen, 1),
                "train_accuracy": train_correct / max(train_seen, 1),
                "validation_loss": val_loss / max(val_seen, 1),
                "validation_accuracy": val_correct / max(val_seen, 1),
            }
            history.append(epoch_result)

            improved = epoch_result["validation_accuracy"] > best_accuracy
            if improved:
                best_accuracy = epoch_result["validation_accuracy"]
                torch.save({
                    "format_version": 1,
                    "architecture": "mobilenet_v3_small",
                    "task": "reader",
                    "classes": classes,
                    "input_size": config.input_size,
                    "state_dict": model.state_dict(),
                    "dataset_sha256": plan["dataset_sha256"],
                    "split_sha256": plan["split_sha256"],
                    "epoch": epoch_number,
                    "validation_accuracy": best_accuracy,
                    "initialization": "imagenet1k_v1" if config.pretrained else "random",
                    "initialization_source": (
                        "torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1"
                        if config.pretrained else "random"
                    ),
                    "torch_version": str(torch.__version__),
                    "torchvision_version": torchvision.__version__,
                }, best_path)

            if tracker.enabled:
                epoch_metrics = {
                    "epoch": epoch_number,
                    "train/epoch_loss": epoch_result["train_loss"],
                    "train/epoch_accuracy": epoch_result["train_accuracy"],
                    "validation/loss": epoch_result["validation_loss"],
                    "validation/accuracy": epoch_result["validation_accuracy"],
                    "validation/best_accuracy": best_accuracy,
                    "epoch_time": time.perf_counter() - epoch_start,
                }
                if len(sample_crops) > 0:
                    epoch_metrics["validation/crops"] = sample_crops
                tracker.log(epoch_metrics, step=global_step)
            save_training_state(epoch_number, "in_progress")
            # One line per epoch: the panel of the review site reads the progress from the log.
            print(f"reader epoch {epoch_number}/{config.epochs} train_loss {epoch_result['train_loss']:.4f} "
                  f"val_loss {epoch_result['validation_loss']:.4f} "
                  f"train_accuracy {epoch_result['train_accuracy']:.4f} "
                  f"val_accuracy {epoch_result['validation_accuracy']:.4f} "
                  f"{time.perf_counter() - epoch_start:.0f}s", flush=True)
            stale = 0 if improved else stale + 1
            if config.patience and stale >= config.patience and epoch_number < config.epochs:
                stopped_early = epoch_number
                print(f"reader stops early after epoch {epoch_number}: validation accuracy not better for "
                      f"{stale} epochs (best {best_accuracy:.4f})", flush=True)
                break

        gpu = None
        if device.type == "cuda":
            gpu = {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            }

        result = {
            "task": "reader",
            "architecture": "mobilenet_v3_small",
            "classes": classes,
            "best_validation_accuracy": best_accuracy,
            "stopped_early_after_epoch": stopped_early,
            "checkpoint": str(best_path),
            "training_plan": str(plan_path),
            "device": str(device),
            "gpu": gpu,
            "torch_version": str(torch.__version__),
            "torchvision_version": torchvision.__version__,
            "initialization": "imagenet1k_v1" if config.pretrained else "random",
            "initialization_source": (
                "torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1"
                if config.pretrained else "random"
            ),
            "config": config.__dict__,
            "history": history,
            "limitations": plan["limitations"],
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tracker.set_summary({
            "best_validation_accuracy": best_accuracy,
            "checkpoint": str(best_path),
        })
        save_training_state(config.epochs, "completed")
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    tracker.finish(exit_code=0)
    return result
