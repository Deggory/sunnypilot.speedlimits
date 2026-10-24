"""The detector's input size (scripts/run_detector_sizes.sh): parsing, the checkpoint keys every rebuild reads,
the width buckets, thresholds and comparison of the evaluation, the frames without signs per epoch, the flags
that resume a run, all without torch; then a tiny CPU training at a non-square size with frames without signs,
its best epoch by validation recall, evaluation, ONNX export and replay rebuild, when a Python with torch is
available (this interpreter or the GPU environment of this checkout). The torch work runs in a subprocess:
torch never enters this interpreter, whose other tests check that it was not imported."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from nnslr_tools import cli
from nnslr_tools.annotations import load_jsonl, validate_dataset
from nnslr_tools.checkpoints import (detector_anchor_min_ratio, detector_input_size, format_input_size,
                                     parse_input_size)
from nnslr_tools.evaluation import (PER_FRAME_MAX_BOXES, PER_FRAME_MIN_SCORE, PER_FRAME_STEPS, WIDTH_BUCKETS,
                                    detector_sizes_table, match_detections, recall_by_width,
                                    score_frame, threshold_metrics, width_bucket)
from nnslr_tools.exporting import detector_io_contract
from nnslr_tools.job_progress import progress
from nnslr_tools.training import (DetectorTrainConfig, TrainConfig, _check_detector_config,
                                  _detector_config_differences, epoch_frame_indices, resume_arguments)
from test_annotations import invoke, reviewed, write
from test_training import _training_fixture

SRC = Path(__file__).resolve().parents[2] / "src"
REPO = SRC.parent


@pytest.mark.parametrize("text,size", [("640x360", (640, 360)), ("960X544", (960, 544)), ("320", (320, 320)),
                                       (" 1344x760 ", (1344, 760))])
def test_input_size_parsing(text, size):
    assert parse_input_size(text) == size
    assert format_input_size(size) == f"{size[0]}x{size[1]}"


@pytest.mark.parametrize("text", ["640x", "x360", "abc", "16x16", "640x360x3", "-640x360", "5000x100", "", "6.4x3"])
def test_input_size_parsing_rejects(text):
    with pytest.raises(ValueError):
        parse_input_size(text)


def test_checkpoint_geometry_defaults_for_old_checkpoints():
    assert detector_input_size({}) == (320, 320)
    assert detector_input_size({"input_size": [640, 360]}) == (640, 360)
    assert detector_input_size({"input_size": (960, 544)}) == (960, 544)
    assert detector_anchor_min_ratio({}) == 0.2
    assert detector_anchor_min_ratio({"anchor_min_ratio": 0.05}) == 0.05
    for bad in ([640], [640, "360"], [640, 8], "640x360", [True, 360]):
        with pytest.raises(ValueError):
            detector_input_size({"input_size": bad})
    for bad in (0, 0.95, -1, "0.1", True):
        with pytest.raises(ValueError):
            detector_anchor_min_ratio({"anchor_min_ratio": bad})


def test_detector_contract_carries_the_trained_size():
    base = {"task": "detector", "classes": ["background", "speed_sign"], "dataset_sha256": "a" * 64,
            "split_sha256": "b" * 64}
    legacy = detector_io_contract(base)
    assert legacy["input"]["model_internal_resize"] == [320, 320]
    assert legacy["default_boxes"]["min_ratio"] == 0.2
    wide = detector_io_contract({**base, "input_size": [640, 360], "anchor_min_ratio": 0.03})
    assert wide["input"]["model_internal_resize"] == [640, 360]
    assert wide["input"]["model_internal_resize_order"] == "width_height"
    # The ONNX input stays the native frame: the model resizes it.
    assert wide["input"]["shape"] == [1, 3, 760, 1344]
    assert wide["default_boxes"] == {"min_ratio": 0.03, "max_ratio": 0.95, "aspect_ratios": [2, 3]}


def test_width_buckets_match_the_evaluate_tab():
    from nnslr_tools.runs import SIZE_BUCKETS

    assert [(key, low, high) for key, _, low, high in SIZE_BUCKETS] == list(WIDTH_BUCKETS)
    assert [width_bucket(width) for width in (0, 11.9, 12, 19.99, 20, 39, 40, 500)] == [
        "lt12", "lt12", "12-20", "12-20", "20-40", "20-40", "ge40", "ge40"]


def test_match_detections_says_which_predictions_are_false():
    predicted = [{"box": [50, 50, 60, 60], "score": .4}, {"box": [0, 0, 10, 10], "score": .9},
                 {"box": [1, 1, 10, 10], "score": .8}]
    result = match_detections(predicted, [[0, 0, 10, 10]], iou_threshold=.5)
    # Score order: the 0.9 box matches, then 0.8 (duplicate) and 0.4 are false.
    assert result["false_positive_indexes"] == [2, 0]
    assert [item["score"] for item in result["false_positive_details"]] == [.8, .4]


def test_recall_by_width_and_thresholds():
    truth = [[0, 0, 8, 8], [100, 0, 115, 15], [200, 0, 230, 30], [300, 0, 360, 60]]
    predictions = [
        {"box": [0, 0, 8, 8], "score": .35},          # the < 12 px sign, found only below 0.5
        {"box": [300, 0, 360, 60], "score": .9},      # the >= 40 px sign
        {"box": [500, 0, 525, 25], "score": .6},      # a false 25 px box
        {"box": [600, 0, 605, 5], "score": .26},      # a false 5 px box, only at 0.25
    ]
    low = score_frame(truth, predictions, score_threshold=.25, iou_threshold=.5)
    high = score_frame(truth, predictions, score_threshold=.5, iou_threshold=.5)
    assert low["found_truth_indexes"] == [0, 3] and high["found_truth_indexes"] == [3]
    widths = recall_by_width([low])
    assert widths["lt12"] == {"truth": 1, "found": 1, "false_positives": 1, "missed": 0, "recall": 1.0}
    assert widths["12-20"]["recall"] == 0.0 and widths["20-40"]["false_positives"] == 1
    assert widths["ge40"]["found"] == 1
    assert recall_by_width([])["lt12"]["recall"] is None

    human = {**high, "label_basis": "human_review"}
    machine = {**score_frame([[0, 0, 50, 50]], [], score_threshold=.5, iou_threshold=.5), "label_basis": "ai_review"}
    measured = threshold_metrics([human, machine])
    assert (measured["truth_box_count"], measured["true_positives"], measured["false_positives"]) == (5, 1, 1)
    assert measured["false_positives_per_image"] == .5
    assert measured["recall_by_width"]["ge40"] == {"truth": 2, "found": 1, "false_positives": 0, "missed": 1,
                                                   "recall": .5}
    assert measured["human_reviewed"]["image_count"] == 1
    assert measured["human_reviewed"]["recall_by_width"]["ge40"]["recall"] == 1.0


OLD_DETECTOR_CONFIG = {"epochs": 20, "patience": 5, "batch_size": 8, "learning_rate": 1e-4, "weight_decay": 1e-4,
                       "num_workers": 4, "seed": 0, "device": "cuda", "pretrained_backbone": True, "amp": True}


def test_resume_compares_old_runs_with_the_settings_they_had():
    old = OLD_DETECTOR_CONFIG
    # A run saved before these settings existed was trained at 320 x 320, without augmentation, on every frame,
    # on the speed signs only, and kept its lowest validation loss.
    before = DetectorTrainConfig(pretrained_backbone=True, augment=False, negative_frame_fraction=1.0, select_by="loss",
                                 families="speed")
    assert _detector_config_differences(old, before) == []
    assert _detector_config_differences(old, DetectorTrainConfig(pretrained_backbone=True)) == [
        "augment False -> True", "families 'speed' -> 'all'", "negative_frame_fraction 1.0 -> 0.1",
        "select_by 'loss' -> 'recall'"]
    saved = dict(DetectorTrainConfig(input_size=(640, 360)).__dict__, input_size=[640, 360])
    assert _detector_config_differences(saved, DetectorTrainConfig(input_size=(640, 360))) == []
    assert _detector_config_differences(saved, DetectorTrainConfig()) == ["input_size (640, 360) -> (320, 320)"]


def _config_of(flags: list[str], task: str):
    """The configuration nnslr train builds from these flags (as scripts/run_v100_baseline.sh passes them)."""
    args = cli.build_parser().parse_args(["train", "--task", task, "--mode", "resume", "--output", "x", *flags])
    return cli._train_config(args)


def test_resume_flags_give_a_run_back_its_settings():
    # A detector run of today, from ImageNet, non-default everywhere.
    saved = dict(DetectorTrainConfig(pretrained_backbone=True, amp=False, input_size=(640, 360), anchor_min_ratio=0.02,
                                     augment=False, negative_frame_fraction=0.25, select_by="loss", epochs=7,
                                     patience=0, batch_size=4, learning_rate=3e-5, weight_decay=0.0, num_workers=2,
                                     seed=3).__dict__, input_size=[640, 360])
    flags = resume_arguments({"task": "detector", "config": saved})
    assert "--pretrained" in flags and "--no-amp" in flags and "--no-augment" in flags
    assert flags[flags.index("--input-size") + 1] == "640x360"
    assert _detector_config_differences(saved, _config_of(flags, "detector")) == []
    # A run saved before the new settings: it resumes as it was trained.
    flags = resume_arguments({"task": "detector", "config": OLD_DETECTOR_CONFIG})
    assert flags[flags.index("--select-by") + 1] == "loss" and "--no-augment" in flags
    assert flags[flags.index("--negative-frame-fraction") + 1] == "1.0"
    assert _detector_config_differences(OLD_DETECTOR_CONFIG, _config_of(flags, "detector")) == []
    # The reader started from ImageNet: the bug of scripts/run_v100_baseline.sh resumed it without --pretrained.
    reader = TrainConfig(pretrained=True, input_size=128, device="cuda").__dict__
    flags = resume_arguments({"task": "reader", "config": reader})
    assert "--pretrained" in flags and flags[flags.index("--input-size") + 1] == "128"
    assert _config_of(flags, "reader").__dict__ == reader
    with pytest.raises(ValueError):
        resume_arguments({"task": "detector"})


def test_frames_without_signs_per_epoch():
    frames = [{"boxes": [[0, 0, 5, 5]] if index % 3 == 0 else []} for index in range(30)]
    with_signs = [index for index in range(30) if index % 3 == 0]
    assert epoch_frame_indices(frames, 1.0, "train 0 0") == list(range(30))
    assert epoch_frame_indices(frames, 0.0, "train 0 0") == with_signs
    first = epoch_frame_indices(frames, 0.1, "train 0 0")
    # Every frame with signs, and 2 of the 20 without, in frame order; the same draw for the same seed.
    assert set(with_signs) <= set(first) and len(first) == 10 + 2 and first == sorted(first)
    assert epoch_frame_indices(frames, 0.1, "train 0 0") == first
    assert len({tuple(epoch_frame_indices(frames, 0.1, f"train 0 {epoch}")) for epoch in range(10)}) > 1
    assert len(epoch_frame_indices(frames, 0.5, "validation 0")) == 10 + 10


def test_detector_settings_are_checked():
    for bad in ({"negative_frame_fraction": 1.5}, {"negative_frame_fraction": -0.1}, {"select_by": "f1"},
                {"anchor_min_ratio": 0.95}, {"input_size": (640, 8)}):
        with pytest.raises(ValueError):
            _check_detector_config(DetectorTrainConfig(**bad))
    _check_detector_config(DetectorTrainConfig(negative_frame_fraction=0, select_by="loss"))


def test_the_progress_parser_reads_the_validation_recall():
    log = "== 4/10 Train detector ==\ndetector epoch 2/20 train_loss 1.2000 val_loss 1.5000 val_recall 0.4125 87s\n"
    [point] = progress("train", "running", log)["curves"]["detector"]["points"]
    assert point == {"epoch": 2, "train_loss": 1.2, "val_loss": 1.5, "val_recall": 0.4125, "seconds": 87}


def test_train_cli_checks_the_size_before_anything(tmp_path, capsys):
    dataset, _, _ = _training_fixture(tmp_path, capsys)
    code, out = invoke(capsys, "train", str(dataset), "--task", "detector", "--data-root", str(tmp_path),
                       "--input-size", "640x360", "--dry-run")
    assert code == 0, out.err
    code, out = invoke(capsys, "train", str(dataset), "--task", "detector", "--data-root", str(tmp_path),
                       "--input-size", "640by360", "--dry-run")
    assert code == 2 and "invalid input size" in out.err
    code, out = invoke(capsys, "train", str(dataset), "--data-root", str(tmp_path), "--no-augment", "--dry-run")
    assert code == 2 and "detector options" in out.err
    code, out = invoke(capsys, "train", str(dataset), "--data-root", str(tmp_path), "--input-size", "96", "--dry-run")
    assert code == 0, out.err
    code, out = invoke(capsys, "train", str(dataset), "--task", "detector", "--data-root", str(tmp_path),
                       "--negative-frame-fraction", "2", "--dry-run")
    assert code == 2 and "negative_frame_fraction" in out.err
    code, out = invoke(capsys, "train", str(dataset), "--data-root", str(tmp_path), "--select-by", "loss", "--dry-run")
    assert code == 2 and "detector options" in out.err
    assert "torch" not in sys.modules


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(found: int, truth: int, false: int, frames: int, *, small: int = 0, human=None) -> dict:
    """Detector measures with every reviewed box >= 40 px wide except `small` boxes under 12 px, all found."""
    widths = {key: {"truth": 0, "found": 0, "false_positives": 0, "missed": 0, "recall": None} for key, _, _ in WIDTH_BUCKETS}
    widths["lt12"] = {"truth": small, "found": small, "false_positives": 0, "missed": 0, "recall": 1.0 if small else None}
    widths["ge40"] = {"truth": truth - small, "found": found - small, "false_positives": false,
                      "missed": truth - found, "recall": (found - small) / (truth - small)}
    return {"image_count": frames, "truth_box_count": truth, "true_positives": found, "false_positives": false,
            "false_negatives": truth - found, "precision_iou": found / (found + false), "recall_iou": found / truth,
            "f1_iou": 0.5, "false_positives_per_image": false / frames, "predictions_per_image": (found + false) / frames,
            "recall_by_width": widths, "human_reviewed": human}


def _report(ms: float) -> dict:
    human = {"label_basis": "human_review", **_metrics(4, 10, 6, 50)}
    loose = _metrics(20, 80, 6980, 559, small=4, human=human)
    usual = _metrics(5, 80, 529, 559)
    return {**usual, "score_threshold": .25, "iou_threshold": .5,
            "by_score_threshold": {"0.05": loose, "0.25": usual, "0.5": {**usual, "human_reviewed": None}},
            "inference": {"device_name": "Tesla V100-PCIE-32GB", "mean_ms_per_frame": ms}}


def test_sizes_comparison_table_puts_the_cutter_first(tmp_path):
    root = tmp_path / "detector-sizes"
    _write(root / "320x320" / "result.json", {
        "best_validation_loss": 1.5, "anchor_min_ratio": 0.2, "augmentation": {"probability": .8},
        "history": [{"epoch": 1, "validation_loss": 2.0, "seconds": 90.0},
                    {"epoch": 2, "validation_loss": 1.5, "seconds": 100.0}, {"epoch": 3, "validation_loss": 1.7}],
        "gpu": {"peak_memory_allocated_bytes": 3 * 2**30}})
    _write(root / "320x320" / "evaluation" / "detector-test.json", _report(4.31))
    table = detector_sizes_table(root, ["320x320", "640x360"])
    lines = table.splitlines()
    assert lines[0].startswith("== test: the cutter, recall first (559 frames, 0 confirmed by a person")
    assert lines[1].split() == ["score", "rows", "size", "recall", "speed", "<12", "12-20", "20-40", ">=40",
                                "boxes/frame", "false/frame", "precision", "F1"]
    # The lowest threshold first: recall overall, of the speed signs (a report without families: the same) and by
    # width, boxes passed on, false ones, then precision.
    assert lines[2].split() == ["0.05", "all", "320x320", "0.25", "20/80", "0.25", "20/80", "1.00", "4/4", "-", "-",
                                "0.21", "16/76", "12.52", "12.49", "0.003", "0.500"]
    assert lines[3].split()[:5] == ["0.05", "human", "320x320", "0.40", "4/10"]
    assert lines[4] == ""
    assert lines[5].split()[:5] == ["0.25", "all", "320x320", "0.06", "5/80"]
    assert lines[5].split()[-4:] == ["0.96", "0.95", "0.009", "0.500"]
    assert lines[6].split() == ["0.25", "human", "320x320"] + ["-"] * 10
    assert "not evaluated: 640x360" in table and "== validation: not evaluated ==" in table
    cost = lines[lines.index(next(line for line in lines if line.startswith("== cost"))) + 1:]
    assert "Tesla V100-PCIE-32GB" in table
    assert cost[0].split() == ["size", "ms/frame", "s/epoch", "gpu_peak_GB", "best_epoch", "chosen_by", "val_recall",
                               "sign_free/epoch", "augmentation", "default_box_min", "families"]
    # A result.json from before the selection, the share of frames without signs and the families: loss, every
    # frame, the speed signs.
    assert cost[1].split() == ["320x320", "4.3", "95", "3.0", "2/3", "loss", "-", "1.00", "on", "0.20", "speed"]
    assert cost[2].split() == ["640x360", "-", "no", "result.json"]
    _write(root / "640x360" / "result.json", {
        "best_validation_loss": 1.4, "best_epoch": 3, "best_validation_recall": 0.4125, "anchor_min_ratio": 0.02,
        "selection": {"by": "recall"}, "config": {"negative_frame_fraction": 0.1}, "augmentation": None,
        "history": [{"epoch": epoch, "validation_loss": 1.5, "seconds": 120.0} for epoch in range(1, 9)]})
    table = detector_sizes_table(root, ["320x320", "640x360"]).splitlines()
    assert table[-1].split() == ["640x360", "-", "120", "-", "3/8", "recall", "0.412", "0.10", "off", "0.02", "speed"]


# ---------------------------------------------------------------- with torch, in a subprocess, CPU only

def _torch_python() -> str | None:
    """A Python with torch, torchvision and Pillow: this one, else the GPU environment of this checkout."""
    if all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "PIL")):
        return sys.executable
    candidate = Path(os.environ.get("NNSLR_GPU_VENV", REPO / ".venv-gpu")) / "bin" / "python"
    if not candidate.is_file():
        return None
    check = subprocess.run([str(candidate), "-c", "import torch, torchvision, PIL"], capture_output=True, timeout=300)
    return str(candidate) if check.returncode == 0 else None


@pytest.fixture(scope="module")
def torch_python():
    python = _torch_python()
    if python is None:
        pytest.skip("no Python with torch/torchvision/Pillow (GPU environment only)")
    return python


def _run(python: str, *args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    # Random weights (no download), CPU only, no wandb; few threads, so a busy server does not slow it down.
    environment = dict(os.environ, WANDB_MODE="disabled", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
                       PYTHONPATH=os.pathsep.join(filter(None, (str(SRC), os.environ.get("PYTHONPATH")))))
    return subprocess.run([python, *args], capture_output=True, text=True, env=environment, timeout=timeout)


REBUILD_CHECK = r"""
import sys
from pathlib import Path
import torch
from torchvision.models import mobilenet_v3_small
from nnslr_tools.checkpoints import load_detector, safe_torch_load
from nnslr_tools.replay import _load_models

best, expected_w, expected_h, ratio, scratch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), Path(sys.argv[5])
saved = safe_torch_load(best)
# detector-best.pt holds the epoch the validation recall chose.
assert saved["epoch"] == int(sys.argv[6]) and saved["selected_by"] == "recall", (saved["epoch"], saved.get("selected_by"))
state = safe_torch_load(Path(best).with_name("training-state.pt"))
assert state["best_selection"]["epoch"] == saved["epoch"] and state["stale"] in (0, 1)
model = load_detector(saved).eval()
assert tuple(model.transform.fixed_size) == (expected_w, expected_h), model.transform.fixed_size
assert abs(model.anchor_generator.scales[0] - ratio) < 1e-9, model.anchor_generator.scales
frame = torch.rand(3, 24, 32)
images, _ = model.transform([frame])
assert tuple(images.tensors.shape[-2:]) == (expected_h, expected_w), images.tensors.shape
with torch.no_grad():
    boxes = model([frame])[0]["boxes"]
# The boxes come back in the pixels of the frame, not of the model's input.
assert boxes.numel() == 0 or float(boxes[:, 2].max()) <= 32.0 + 1e-3

# A checkpoint from before the input size was recorded: 320 x 320 and torchvision's default boxes.
legacy = {key: value for key, value in saved.items() if key not in ("input_size", "anchor_min_ratio")}
old = load_detector(legacy)
assert tuple(old.transform.fixed_size) == (320, 320) and abs(old.anchor_generator.scales[0] - 0.2) < 1e-9

# Replay rebuilds the detector from its checkpoint too.
reader = mobilenet_v3_small(weights=None)
reader.classifier[-1] = torch.nn.Linear(reader.classifier[-1].in_features, 2)
reader_path = scratch / "reader-best.pt"
torch.save({"task": "reader", "classes": ["maximum_speed:30", "maximum_speed:50"], "input_size": 64,
            "state_dict": reader.state_dict(), "dataset_sha256": saved["dataset_sha256"],
            "split_sha256": saved["split_sha256"]}, reader_path)
loaded = _load_models(Path(best), reader_path, "cpu")
assert tuple(loaded[4].transform.fixed_size) == (expected_w, expected_h)

# The photometric augmentation: same size, values 0..1, reproducible from its generator, darker on average.
from torchvision.transforms import functional as TF
from nnslr_tools.training import detector_photometric
frame = torch.rand(3, 24, 32)
first = [detector_photometric(torch, TF, frame, torch.Generator().manual_seed(seed)) for seed in range(40)]
again = [detector_photometric(torch, TF, frame, torch.Generator().manual_seed(seed)) for seed in range(40)]
assert all(torch.equal(a, b) for a, b in zip(first, again))
assert all(image.shape == frame.shape and float(image.min()) >= 0 and float(image.max()) <= 1 for image in first)
changed = [image for image in first if not torch.equal(image, frame)]
assert 20 <= len(changed) < 40, len(changed)
assert sum(float(image.mean()) for image in changed) / len(changed) < float(frame.mean())
print("rebuild ok")
"""


def _kept_found(image: dict, threshold: float) -> set[int]:
    """The reviewed boxes found at a score by the boxes a frame keeps for the Watch panel ("x1 y1 x2 y2 score truth")."""
    parts = [line.split(" ") for line in image["cutter_boxes"]]
    return {int(part[5]) for part in parts if int(part[5]) >= 0 and float(part[4]) >= threshold}


def _check_per_frame_boxes(report: dict) -> None:
    """Every frame keeps the cutter's boxes from 0.05 for the Watch panel of the Evaluate tab: at every score the report
    measured, the signs they find and the boxes counted are the report's own."""
    assert report["per_frame_boxes"]["min_score"] == PER_FRAME_MIN_SCORE
    assert report["per_frame_boxes"]["steps"] == list(PER_FRAME_STEPS) and report["per_frame_boxes"]["box"] == "x1 y1 x2 y2 score truth"
    for image in report["images"]:
        assert len(image["size"]) == 2 and all(side > 0 for side in image["size"])
        assert image["cutter_box_counts"] == sorted(image["cutter_box_counts"], reverse=True)
        assert len(image["cutter_boxes"]) <= min(PER_FRAME_MAX_BOXES, image["cutter_box_counts"][0])
        assert all(len(line.split(" ")) == 6 and float(line.split(" ")[4]) >= PER_FRAME_MIN_SCORE for line in image["cutter_boxes"])
        assert len(_kept_found(image, report["score_threshold"])) == image["true_positives"]
    for key, metrics in report["by_score_threshold"].items():
        step = PER_FRAME_STEPS.index(float(key))
        assert sum(len(_kept_found(image, float(key))) for image in report["images"]) == metrics["true_positives"], key
        assert sum(image["cutter_box_counts"][step] for image in report["images"]) == round(
            metrics["predictions_per_image"] * metrics["image_count"]), key


def _fixture_with_sign_free_frames(root, capsys) -> Path:
    """12 frames on 12 routes, every other one without signs: train 0-5, validation 6-9, test 10-11."""
    rows, partition = [], {}
    for index in range(12):
        route = f"{index + 1:08x}--abcdef1234"
        rows.append(reviewed(root, route, index=index, negative=index % 2 == 1))
        partition[route] = "train" if index < 6 else "validation" if index < 10 else "test"
    code, out = invoke(capsys, "import-annotations", str(write(root, rows)), "--data-root", str(root))
    assert code == 0, out.err + out.out
    dataset = root / "annotations" / "objects.jsonl"
    canonical = load_jsonl(dataset)
    report = validate_dataset(canonical, root)
    assert report["valid"], report["errors"]
    payload = {"schema_version": 1, "dataset_kind": "gold", "dataset_sha256": report["dataset_sha256"],
               "assignments": {row["annotation_id"]: partition[row["route_id"]] for row in canonical}}
    frozen = root / "splits" / "fixture.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    (root / "splits" / "latest.json").write_text(
        json.dumps({"path": "splits/fixture.json", "sha256": digest}) + "\n", encoding="utf-8")
    return dataset


def test_tiny_cpu_detector_at_a_non_square_size(tmp_path, capsys, torch_python):
    dataset = _fixture_with_sign_free_frames(tmp_path, capsys)
    output = tmp_path / "runs" / "tiny" / "detector-sizes" / "96x64"
    process = _run(torch_python, "-m", "nnslr_tools.cli", "train", str(dataset), "--task", "detector",
                   "--data-root", str(tmp_path), "--output", str(output), "--device", "cpu", "--epochs", "2",
                   "--batch-size", "2", "--num-workers", "0", "--no-amp", "--input-size", "96x64",
                   "--anchor-min-ratio", "0.05", "--negative-frame-fraction", "0.5")
    assert process.returncode == 0, process.stderr[-3000:]
    # The epoch line keeps its format, val_recall before the seconds: the site's parser reads it.
    for epoch in (1, 2):
        assert re.search(rf"^detector epoch {epoch}/2 train_loss \S+ val_loss \S+ val_recall [0-9.]+ \d+s$",
                         process.stdout, re.M), process.stdout
    assert "detector input 96x64, smallest default box 0.05 of the image, photometric augmentation on" in process.stdout
    assert ("detector frames per epoch: 3 with signs and 2 of 3 without; validation 2 with signs and 1 of 2 "
            "without; best epoch by validation recall at score 0.1") in process.stdout
    points = progress("train", "running", "== 4/10 Train detector ==\n" + process.stdout)["curves"]["detector"]["points"]
    assert [sorted(point) for point in points] == [["epoch", "seconds", "train_loss", "val_loss", "val_recall"]] * 2

    result = json.loads((output / "result.json").read_text())
    assert result["input_size"] == [96, 64] and result["anchor_min_ratio"] == 0.05
    assert result["config"]["input_size"] == [96, 64] and result["config"]["augment"] is True
    assert result["config"]["negative_frame_fraction"] == 0.5 and result["config"]["select_by"] == "recall"
    # 5 frames per epoch in batches of 2: the last frame of the shuffle waits (a batch of one fails in BatchNorm).
    assert result["frames"] == {"train_with_signs": 3, "train_sign_free": 3, "train_sign_free_per_epoch": 2,
                                "train_frames_per_epoch": 4, "validation_with_signs": 2, "validation_sign_free": 2,
                                "validation_sign_free_used": 1}
    assert result["selection"]["by"] == "recall" and result["selection"]["score_threshold"] == 0.1
    assert result["augmentation"]["brightness"] == [0.35, 1.25]
    history = result["history"]
    assert len(history) == 2 and all(item["seconds"] > 0 for item in history)
    # The best epoch: the highest validation recall, then the fewest false boxes per frame; the first on a tie.
    best_key = max((item["validation_recall"], -item["validation_false_positives_per_image"]) for item in history)
    expected = next(item for item in history
                    if (item["validation_recall"], -item["validation_false_positives_per_image"]) == best_key)
    assert result["best_epoch"] == expected["epoch"]
    assert result["best_validation_recall"] == expected["validation_recall"]
    assert result["best_validation_loss"] == expected["validation_loss"]

    # The flags that would resume this run give back its configuration.
    flags = _run(torch_python, "-c", "import json, sys; from nnslr_tools.training import resume_arguments_file; "
                                     "print(json.dumps(resume_arguments_file(sys.argv[1])))",
                 str(output / "training-state.pt"))
    assert flags.returncode == 0, flags.stderr[-3000:]
    assert _detector_config_differences(result["config"], _config_of(json.loads(flags.stdout), "detector")) == []

    best = output / "detector-best.pt"
    process = _run(torch_python, "-c", REBUILD_CHECK, str(best), "96", "64", "0.05", str(tmp_path),
                   str(expected["epoch"]))
    assert process.returncode == 0, process.stderr[-3000:]
    assert "rebuild ok" in process.stdout

    report_path = output / "evaluation" / "detector-test.json"
    process = _run(torch_python, "-m", "nnslr_tools.cli", "evaluate", str(dataset), "--task", "detector",
                   "--data-root", str(tmp_path), "--partition", "test", "--device", "cpu",
                   "--checkpoint", str(best), "--output", str(report_path))
    assert process.returncode == 0, process.stderr[-3000:]
    report = json.loads(report_path.read_text())
    assert report["input_size"] == [96, 64] and report["input_size_label"] == "96x64"
    assert report["anchor_min_ratio"] == 0.05 and report["checkpoint_epoch"] == expected["epoch"]
    # The evaluation keeps every frame of the partition, with signs or without.
    assert report["image_count"] == 2
    # The keys the Evaluate tab and the hard example mining read are still there.
    for key in ("image_count", "truth_box_count", "true_positives", "false_positives", "false_negatives",
                "precision_iou", "recall_iou", "f1_iou", "false_positives_per_image", "counts_by_label_basis",
                "human_reviewed", "score_threshold", "iou_threshold", "images"):
        assert key in report, key
    # The cutter's low thresholds, this report's own and the usual ones.
    assert set(report["by_score_threshold"]) == {"0.05", "0.1", "0.25", "0.3", "0.5"}
    assert report["by_score_threshold"]["0.25"]["true_positives"] == report["true_positives"]
    low, high = report["by_score_threshold"]["0.05"], report["by_score_threshold"]["0.5"]
    assert low["true_positives"] >= high["true_positives"]
    assert low["predictions_per_image"] >= high["predictions_per_image"]
    assert low["predictions_per_image"] * low["image_count"] == pytest.approx(
        low["true_positives"] + low["false_positives"])
    assert set(low["recall_by_width"]) == {"lt12", "12-20", "20-40", "ge40"}
    assert report["model_postprocessing"] == {"score_floor": 0.001, "nms_iou": 0.55, "max_boxes": 300}
    assert set(report["recall_by_width"]) == {"lt12", "12-20", "20-40", "ge40"}
    # Every reviewed box of the fixture is 14 pixels wide.
    assert report["recall_by_width"]["12-20"]["truth"] == report["truth_box_count"] > 0
    assert report["human_reviewed"]["recall_by_width"]["12-20"]["truth"] == report["truth_box_count"]
    inference = report["inference"]
    assert inference["device"] == "cpu" and inference["frames"] == report["image_count"]
    assert inference["mean_ms_per_frame"] > 0 and inference["batch_size"] == 1

    table = _run(torch_python, "-c", "import sys; from nnslr_tools.evaluation import detector_sizes_table; "
                                     "print(detector_sizes_table(sys.argv[1], ['96x64']), end='')",
                 str(output.parent))
    assert table.returncode == 0, table.stderr[-3000:]
    assert re.search(r"^0\.05\s+all\s+96x64\s", table.stdout, re.M)
    assert re.search(rf"^96x64\s+[0-9.]+\s+[0-9]+\s+-\s+{expected['epoch']}/2\s+recall\s+[0-9.]+\s+0\.50\s+on\s+0\.05"
                     rf"\s+all$", table.stdout, re.M), table.stdout
    # A new cutter cuts the speed and the other signs; its report gives the speed signs alone too.
    assert result["families"] == "all" and result["config"]["families"] == "all"
    assert report["families"] == "all" and report["recall_by_family"]["speed"]["truth"] == report["truth_box_count"]
    assert report["by_score_threshold"]["0.1"]["recall_by_family"]["speed"]["truth"] == report["truth_box_count"]
    assert report["images"][0]["truth_families"] == ["maximum_speed"] * report["images"][0]["truth_count"]
    _check_per_frame_boxes(report)

    if subprocess.run([torch_python, "-c", "import onnx, onnxruntime, onnxscript"], capture_output=True).returncode:
        pytest.skip("onnx/onnxruntime/onnxscript missing: export not checked")
    onnx_path = tmp_path / "export" / "detector.onnx"
    process = _run(torch_python, "-m", "nnslr_tools.cli", "export-onnx", "--task", "detector",
                   "--checkpoint", str(best), "--output", str(onnx_path),
                   "--detector-input-width", "32", "--detector-input-height", "24")
    assert process.returncode == 0, process.stderr[-3000:]
    contract = json.loads(onnx_path.with_suffix(".onnx.contract.json").read_text())
    assert contract["input"]["shape"] == [1, 3, 24, 32]
    assert contract["input"]["model_internal_resize"] == [96, 64]
    assert contract["default_boxes"]["min_ratio"] == 0.05
    assert contract["verification"]["passed"] is True
    assert "torch" not in sys.modules
