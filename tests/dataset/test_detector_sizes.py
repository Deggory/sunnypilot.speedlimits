"""The detector's input size (scripts/run_detector_sizes.sh): parsing, the checkpoint keys every rebuild reads,
the width buckets, thresholds and comparison of the evaluation, all without torch; then a tiny CPU training at a
non-square size, its evaluation, ONNX export and replay rebuild, when a Python with torch is available (this
interpreter or the GPU environment of this checkout). The torch work runs in a subprocess: torch never enters
this interpreter, whose other tests check that it was not imported."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from nnslr_tools.checkpoints import (detector_anchor_min_ratio, detector_input_size, format_input_size,
                                     parse_input_size)
from nnslr_tools.evaluation import (WIDTH_BUCKETS, detector_sizes_table, match_detections, recall_by_width,
                                    score_frame, threshold_metrics, width_bucket)
from nnslr_tools.exporting import detector_io_contract
from nnslr_tools.training import DetectorTrainConfig, _detector_config_differences
from test_annotations import invoke
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


def test_resume_compares_old_runs_with_the_settings_they_had():
    old = {"epochs": 20, "patience": 5, "batch_size": 8, "learning_rate": 1e-4, "weight_decay": 1e-4,
           "num_workers": 4, "seed": 0, "device": "cuda", "pretrained_backbone": True, "amp": True}
    # A run saved before these settings existed was trained at 320 x 320, without augmentation.
    assert _detector_config_differences(old, DetectorTrainConfig(pretrained_backbone=True, augment=False)) == []
    assert _detector_config_differences(old, DetectorTrainConfig(pretrained_backbone=True)) == [
        "augment False -> True"]
    saved = dict(DetectorTrainConfig(input_size=(640, 360)).__dict__, input_size=[640, 360])
    assert _detector_config_differences(saved, DetectorTrainConfig(input_size=(640, 360))) == []
    assert _detector_config_differences(saved, DetectorTrainConfig()) == ["input_size (640, 360) -> (320, 320)"]


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
    assert lines[1].split() == ["score", "rows", "size", "recall", "<12", "12-20", "20-40", ">=40", "boxes/frame",
                                "false/frame", "precision", "F1"]
    # The lowest threshold first: recall overall and by width, boxes passed on, false ones, then precision.
    assert lines[2].split() == ["0.05", "all", "320x320", "0.25", "20/80", "1.00", "4/4", "-", "-", "0.21", "16/76",
                                "12.52", "12.49", "0.003", "0.500"]
    assert lines[3].split()[:5] == ["0.05", "human", "320x320", "0.40", "4/10"]
    assert lines[4] == ""
    assert lines[5].split()[:5] == ["0.25", "all", "320x320", "0.06", "5/80"]
    assert lines[5].split()[-4:] == ["0.96", "0.95", "0.009", "0.500"]
    assert lines[6].split() == ["0.25", "human", "320x320"] + ["-"] * 9
    assert "not evaluated: 640x360" in table and "== validation: not evaluated ==" in table
    cost = lines[lines.index(next(line for line in lines if line.startswith("== cost"))) + 1:]
    assert "Tesla V100-PCIE-32GB" in table
    assert cost[0].split() == ["size", "ms/frame", "s/epoch", "gpu_peak_GB", "best_epoch", "augmentation",
                               "default_box_min"]
    assert cost[1].split() == ["320x320", "4.3", "95", "3.0", "2/3", "on", "0.20"]
    assert cost[2].split() == ["640x360", "-", "no", "result.json"]


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


def test_tiny_cpu_detector_at_a_non_square_size(tmp_path, capsys, torch_python):
    dataset, _, _ = _training_fixture(tmp_path, capsys)
    output = tmp_path / "runs" / "tiny" / "detector-sizes" / "96x64"
    process = _run(torch_python, "-m", "nnslr_tools.cli", "train", str(dataset), "--task", "detector",
                   "--data-root", str(tmp_path), "--output", str(output), "--device", "cpu", "--epochs", "1",
                   "--batch-size", "2", "--num-workers", "0", "--no-amp", "--input-size", "96x64",
                   "--anchor-min-ratio", "0.05")
    assert process.returncode == 0, process.stderr[-3000:]
    assert re.search(r"^detector epoch 1/1 train_loss \S+ val_loss \S+ \d+s$", process.stdout, re.M)
    assert "detector input 96x64, smallest default box 0.05 of the image, photometric augmentation on" in process.stdout

    result = json.loads((output / "result.json").read_text())
    assert result["input_size"] == [96, 64] and result["anchor_min_ratio"] == 0.05
    assert result["config"]["input_size"] == [96, 64] and result["config"]["augment"] is True
    assert result["augmentation"]["brightness"] == [0.35, 1.25] and result["best_epoch"] == 1
    assert result["history"][0]["seconds"] > 0

    best = output / "detector-best.pt"
    process = _run(torch_python, "-c", REBUILD_CHECK, str(best), "96", "64", "0.05", str(tmp_path))
    assert process.returncode == 0, process.stderr[-3000:]
    assert "rebuild ok" in process.stdout

    report_path = output / "evaluation" / "detector-test.json"
    process = _run(torch_python, "-m", "nnslr_tools.cli", "evaluate", str(dataset), "--task", "detector",
                   "--data-root", str(tmp_path), "--partition", "test", "--device", "cpu",
                   "--checkpoint", str(best), "--output", str(report_path))
    assert process.returncode == 0, process.stderr[-3000:]
    report = json.loads(report_path.read_text())
    assert report["input_size"] == [96, 64] and report["input_size_label"] == "96x64"
    assert report["anchor_min_ratio"] == 0.05 and report["checkpoint_epoch"] == 1
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
    assert re.search(r"^96x64\s+[0-9.]+\s+[0-9]+\s+-\s+1/1\s+on\s+0\.05$", table.stdout, re.M), table.stdout

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
