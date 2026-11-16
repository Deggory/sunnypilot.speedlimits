"""Experiment B (nnslr_tools.context_experiment): the CPU-only plan and metrics, and a tiny CPU run of each
variant when torch is importable (the GPU environment); without torch those runs are skipped."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from nnslr_tools.annotations import load_jsonl, validate_dataset
from nnslr_tools.context_experiment import (
    OWNERSHIP_CLASSES,
    VARIANTS,
    comparison_table,
    decider_label,
    expected_calibration_error,
    geometry_features,
    ownership_class_weights,
    ownership_metrics,
    ownership_target,
    selection_score,
)
from test_annotations import invoke, reviewed, write

TRAIN = ("00000001--abcdef1234", "00000002--abcdef1234")
VALIDATION = "00000003--abcdef1234"
TEST = "00000004--abcdef1234"
BUFFER = "00000005--abcdef1234"
SRC = Path(__file__).resolve().parents[2] / "src"


def _frame(root, route, index, detections, *, speed=None, ai=False):
    """A reviewed frame with boxes side by side: (family, value_kph, applicability) each."""
    row = reviewed(root, route, index=index)
    boxes = []
    for number, (family, value, applicability) in enumerate(detections):
        box = copy.deepcopy(row["detections"][0])
        box.update(detection_id=f"box-{number}", bbox_xyxy=[1 + 16 * number, 2, 15 + 16 * number, 20],
                   family=family, value_kph=value, applicability=applicability)
        boxes.append(box)
    row["detections"] = boxes
    row["source_proposals"] = copy.deepcopy(boxes)
    if speed is not None:
        row["gps"] = {"latitude": 47.0, "longitude": 8.0, "speed_mps": speed, "bearing_deg": 0.0}
        row["country"] = "CH"
    if ai:
        row.update(reviewer="ai/Claude", label_basis="ai_review", reviewer_kind="ai", review_channel="mcp")
    return row


def _experiment_fixture(root, capsys):
    """Two train routes, one validation, one test and one buffer route, split by hand."""
    rows = [
        _frame(root, TRAIN[0], 0, [("maximum_speed", 30, "own_road"), ("maximum_speed", 50, "other_road")], speed=20.0),
        _frame(root, TRAIN[0], 1, [("maximum_speed", 50, "own_road")], ai=True),
        _frame(root, TRAIN[0], 2, [("other_sign", None, "unresolved"), ("cancellation", None, "unresolved")],
               speed=10.0, ai=True),
        _frame(root, TRAIN[1], 0, [("maximum_speed", 30, "unresolved")]),
        _frame(root, TRAIN[1], 1, [("maximum_speed", 30, "own_road"), ("maximum_speed", 50, "own_road")], ai=True),
        _frame(root, VALIDATION, 0, [("maximum_speed", 30, "own_road"), ("maximum_speed", 50, "other_road")],
               speed=25.0),
        _frame(root, VALIDATION, 1, [("other_sign", None, "unresolved")], ai=True),
        _frame(root, TEST, 0, [("maximum_speed", 50, "own_road")]),
        _frame(root, TEST, 1, [("maximum_speed", 30, "other_road"), ("cancellation", None, "own_road")], ai=True),
        _frame(root, BUFFER, 0, [("maximum_speed", 30, "own_road")], ai=True),
    ]
    code, out = invoke(capsys, "import-annotations", str(write(root, rows)), "--data-root", str(root))
    assert code == 0, out.err + out.out
    dataset = root / "annotations" / "objects.jsonl"
    canonical = load_jsonl(dataset)
    report = validate_dataset(canonical, root)
    assert report["valid"], report["errors"]

    partition = {route: "train" for route in TRAIN}
    partition.update({VALIDATION: "validation", TEST: "test", BUFFER: "buffer"})
    payload = {
        "schema_version": 1,
        "dataset_kind": "gold",
        "dataset_sha256": report["dataset_sha256"],
        "assignments": {r["annotation_id"]: partition[r["route_id"]] for r in canonical},
    }
    frozen = root / "splits" / "fixture.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    (root / "splits" / "latest.json").write_text(
        json.dumps({"path": "splits/fixture.json", "sha256": digest}) + "\n", encoding="utf-8")
    return dataset, frozen, report["dataset_sha256"]


def test_dry_run_counts_without_torch(tmp_path, capsys):
    _, frozen, dataset_sha = _experiment_fixture(tmp_path, capsys)
    code, out = invoke(capsys, "experiment-b", "--variant", "full", "--data-root", str(tmp_path), "--dry-run")
    assert code == 0, out.err + out.out
    summary = json.loads(out.out)

    assert summary["variant"] == "full" and summary["inputs"] == ["crop", "box_geometry", "car_speed", "scene"]
    assert summary["dataset_sha256"] == dataset_sha
    assert summary["split_sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert "records" not in summary and summary["record_count"] == 14
    counts = summary["counts"]
    assert counts["boxes_by_partition"] == {"train": 8, "validation": 3, "test": 3}
    assert counts["frames_by_partition"] == {"train": 5, "validation": 2, "test": 2}
    assert counts["human_review_boxes_by_partition"] == {"train": 3, "validation": 2, "test": 1}
    # The buffer (and any partition other than train/validation/test) is never read.
    assert summary["ignored_partitions"] == {"buffer": 1}
    assert summary["classes"] == ["cancellation", "maximum_speed:30", "maximum_speed:50", "other_sign"]
    assert counts["reading_by_partition"]["test"] == {"cancellation": 1, "maximum_speed:30": 1, "maximum_speed:50": 1}
    # Ownership only on the speed families: other_sign is masked.
    assert counts["ownership_by_partition"]["train"] == {"own_road": 4, "other_road": 1, "unresolved": 2, "masked": 1}
    assert counts["ownership_by_partition"]["validation"] == {"own_road": 1, "other_road": 1, "unresolved": 0, "masked": 1}
    assert counts["ownership_human_review_by_partition"]["train"] == {"own_road": 1, "other_road": 1, "unresolved": 1,
                                                                      "masked": 0}
    assert counts["car_speed_by_partition"]["train"] == {"present": 4, "missing": 4}
    weights = summary["ownership_class_weights"]
    assert weights["other_road"] == pytest.approx(7 / 3) and weights["own_road"] == pytest.approx(7 / 12)
    assert summary["images"] == {"frames": 9, "missing": 0, "missing_examples": []}
    assert summary["unseen_labels"] == {"validation": [], "test": []}
    assert summary["trainable"] is True
    assert "torch" not in sys.modules

    code, out = invoke(capsys, "experiment-b", "--variant", "crop", "--data-root", str(tmp_path),
                       "--dry-run", "--limit", "1")
    assert code == 0, out.err + out.out
    limited = json.loads(out.out)
    assert limited["counts"]["frames_by_partition"] == {"train": 1, "validation": 1, "test": 1}
    assert "limited_smoke_run_not_a_measure" in limited["limitations"]
    assert "torch" not in sys.modules


def test_a_real_run_needs_an_output(tmp_path, capsys):
    _experiment_fixture(tmp_path, capsys)
    code, out = invoke(capsys, "experiment-b", "--variant", "crop", "--data-root", str(tmp_path))
    assert code == 2 and "--output is required" in out.err


def test_ownership_metrics_precision_recall_abstain_and_macro_f1():
    def sample(truth, predicted, confidence=.9):
        return {"ownership_truth": truth, "ownership_predicted": predicted, "ownership_confidence": confidence}

    samples = [sample("own_road", "own_road"), sample("own_road", "own_road"), sample("own_road", "unresolved"),
               sample("other_road", "other_road"), sample("other_road", "own_road"),
               sample("unresolved", "unresolved")]
    metrics = ownership_metrics(samples)
    assert metrics["confusion"]["own_road"] == {"own_road": 2, "other_road": 0, "unresolved": 1}
    assert metrics["other_road"]["precision"] == 1.0 and metrics["other_road"]["recall"] == .5
    assert metrics["per_class"]["own_road"]["precision"] == pytest.approx(2 / 3)
    assert metrics["abstain_rate"] == pytest.approx(2 / 6)
    assert metrics["decided_accuracy"] == pytest.approx(3 / 4)
    f1 = [2 * 2 / (3 + 3), 2 * 1 / (2 + 1), 2 * 1 / (1 + 2)]
    assert metrics["macro_f1"] == pytest.approx(sum(f1) / 3)
    empty = ownership_metrics([])
    assert empty["sample_count"] == 0 and empty["macro_f1"] is None and empty["other_road"]["recall"] is None


def test_the_epoch_is_chosen_on_the_reading_and_the_other_road_f1_not_on_unresolved():
    """#72: one unresolved box on validation must not choose the epoch over the signs of another road found."""
    def sample(truth, predicted):
        return {"ownership_truth": truth, "ownership_predicted": predicted, "ownership_confidence": .9}

    ours = [sample("own_road", "own_road")] * 10
    # A gets the one unresolved box right and finds 1 of 4 signs of another road; B misses it and finds 3 of 4.
    a = ownership_metrics(ours + [sample("other_road", "other_road")] + [sample("other_road", "own_road")] * 3
                          + [sample("unresolved", "unresolved")])
    b = ownership_metrics(ours + [sample("other_road", "other_road")] * 3 + [sample("other_road", "own_road")]
                          + [sample("unresolved", "own_road")])
    assert a["macro_f1"] > b["macro_f1"]
    score_a = selection_score({"reading": {"accuracy": .9}, "ownership": a})
    score_b = selection_score({"reading": {"accuracy": .9}, "ownership": b})
    assert score_a == pytest.approx((.9 + 2 * 1 / (4 + 1)) / 2) and score_b == pytest.approx((.9 + 2 * 3 / (4 + 3)) / 2)
    assert score_b > score_a
    # No sign of another road, true or predicted: its F1 counts 0.
    none = ownership_metrics(ours)
    assert none["other_road"]["f1"] is None and selection_score({"reading": {"accuracy": .9}, "ownership": none}) == .45


def test_with_a_start_the_epoch_is_chosen_on_the_rows_no_ancestor_trained_on():
    """#104: validation rows the start's lineage trained on are read right in the first epochs and flatter them."""
    from nnslr_tools.context_experiment import epoch_scores

    def sample(row, right, truth="own_road", predicted="own_road"):
        return {"annotation_id": row, "truth": "maximum_speed:50", "predicted": "maximum_speed:50" if right else "x",
                "correct": right, "confidence": .9, "ownership_truth": truth, "ownership_predicted": predicted,
                "ownership_confidence": .9}

    classes = ["maximum_speed:50"]
    remembered = [sample(f"old{index}", True) for index in range(10)]
    clean_rows = {f"new{index}" for index in range(4)}
    # Epoch 1, close to the start: the eleven remembered rows all right, one of the four clean ones. Epoch 2: three of
    # the four clean ones, and half of the remembered ones forgotten, the sign of another road among them.
    first = remembered + [sample("old10", True, "other_road", "other_road"),
                          sample("new0", True, "other_road", "other_road"), sample("new1", False, "other_road", "own_road"),
                          sample("new2", False), sample("new3", False)]
    second = remembered[5:] + [sample(f"old{index}", False) for index in range(5)] + [
        sample("old10", True, "other_road", "own_road"), sample("new0", True, "other_road", "other_road"),
        sample("new1", True, "other_road", "other_road"), sample("new2", True), sample("new3", False)]
    whole = [epoch_scores(samples, classes, None, False) for samples in (first, second)]
    assert all(cleaned is None for _, cleaned, _ in whole) and whole[0][2] > whole[1][2]  # the first epoch, as before
    chosen = [epoch_scores(samples, classes, clean_rows, True) for samples in (first, second)]
    assert chosen[1][2] > chosen[0][2]  # the second, on the rows no ancestor trained on
    validation, cleaned, score = chosen[1]
    assert (validation["box_count"], cleaned["box_count"]) == (15, 4) and validation == whole[1][0]
    assert cleaned["reading"]["accuracy"] == .75 and cleaned["ownership"]["other_road"]["recall"] == 1.0
    assert score == selection_score(cleaned) and score != selection_score(validation)
    # Both measured while the whole validation still chooses (too few clean rows, start_lineage.choose_on).
    validation, cleaned, score = epoch_scores(second, classes, clean_rows, False)
    assert cleaned["box_count"] == 4 and score == selection_score(validation)


def test_the_clean_rows_choose_only_with_rows_enough_and_half_of_the_signs_of_another_road():
    """#104: the score is the mean of the reading (every row) and the F1 of the signs of another road (those signs
    alone). The clean rows choose the epoch when they are the floor or more and hold at least half of the
    validation's signs of another road; else the whole validation, and the record says how many of how many."""
    from nnslr_tools import start_lineage
    from nnslr_tools.context_experiment import chooses_on_clean

    def choice(another_clean: int, another_trained: int, clean: int = 400) -> tuple[bool, dict]:
        """A validation of ``clean`` rows no ancestor trained on and 300 it did, with so many signs of another road
        in each part: what chooses, and the record."""
        checked = ([{"ownership": "other_road"}] * another_trained + [{"ownership": "own_road"}] * (300 - another_trained)
                   + [{"ownership": "other_road"}] * another_clean + [{"ownership": None}] * 10
                   + [{"ownership": "own_road"}] * (clean - another_clean - 10))
        record = {"complete": True, "counted": "rows", "best_epoch_on": "validation", "why": None, "clean_cases": None,
                  "validation": {"all": len(checked), "trained_on": 300, "clean": clean}}
        return chooses_on_clean(record, checked, list(range(300, 300 + clean))), record

    # The network in use on 16 Nov 2026: 40 of the 58 signs of another road are on its clean rows.
    chosen, record = choice(40, 18, clean=1007)
    assert chosen is True
    assert (record["best_epoch_on"], record["why"], record["clean_cases"]) == ("clean_validation", None, 1007)
    assert choice(29, 29)[0] is True  # half of them: enough
    # Two of 58 among 400 clean rows: rows enough for the reading, and one sign would choose the other half.
    chosen, record = choice(2, 56)
    assert chosen is False and (record["best_epoch_on"], record["clean_cases"]) == ("validation", 400)
    assert record["why"] == "only 2 of the 58 signs of another road are on the clean rows"
    chosen, record = choice(0, 5)
    assert chosen is False and record["why"] == "only 0 of the 5 signs of another road are on the clean rows"
    # A validation with no sign of another road at all: the reading alone is the score, the floor alone decides.
    assert choice(0, 0)[0] is True
    # Fewer clean rows than the floor, with every sign of another road among them.
    chosen, record = choice(20, 0, clean=start_lineage.CLEAN_FLOOR - 1)
    assert chosen is False
    assert record["why"] == "the 199 clean rows hold 199 boxes to read, fewer than the 200 a choice needs"


def test_calibration_class_weights_and_geometry():
    assert expected_calibration_error([], []) is None
    assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert expected_calibration_error([.95, .95], [True, False]) == pytest.approx(.45)
    weights = ownership_class_weights({"own_road": 1000, "other_road": 10, "unresolved": 90})
    assert weights["own_road"] == pytest.approx(1100 / 3000) and weights["other_road"] == 10.0
    assert ownership_class_weights({}) == {"own_road": 1.0, "other_road": 1.0, "unresolved": 1.0}
    features = geometry_features([134.4, 76, 268.8, 152], 1344, 760, None)
    assert features == pytest.approx([.1, .1, .2, .2, .1, .1, .15, .15, 0.0, 1.0])
    assert geometry_features([0, 0, 1344, 760], 1344, 760, 20.0)[-2:] == [.5, 0.0]


def test_a_box_on_a_vehicle_teaches_the_decider_not_ours():
    sticker = {"sign_family": "not_a_sign", "sign_type": "on_vehicle", "review_state": "corrected",
               "bbox_xyxy": [10, 10, 30, 30]}
    assert decider_label(sticker) == "not_a_sign" and ownership_target(sticker) == "other_road"
    # A plain "not a sign" stays out of the decider; a speed sign keeps its own road.
    assert decider_label({**sticker, "sign_type": None}) is None
    assert ownership_target({**sticker, "sign_type": None}) is None
    assert ownership_target({**sticker, "review_state": "rejected"}) is None
    speed = {"sign_family": "maximum_speed", "value_state": "read", "value_kph": 80, "applicability": "own_road",
             "review_state": "accepted", "bbox_xyxy": [10, 10, 30, 30]}
    assert decider_label(speed) == "maximum_speed:80" and ownership_target(speed) == "own_road"


def test_comparison_table_reads_the_variants(tmp_path):
    ownership = {"other_road": {"precision": .5, "recall": .25}, "macro_f1": .6, "abstain_rate": .1}
    report = {"box_count": 10, "reading": {"accuracy": .8}, "ownership": ownership}
    (tmp_path / "crop").mkdir()
    (tmp_path / "crop" / "result.json").write_text(json.dumps(
        {"metrics": {"validation": {**report, "human_reviewed": None}, "test": {**report, "human_reviewed": report}}}))
    table = comparison_table(tmp_path)
    lines = table.splitlines()
    assert lines[0].split() == ["variant", "scope", "boxes", "read_acc", "other_P", "other_R", "own_mF1", "abstain"]
    assert lines[1].split() == ["crop", "test", "10", "0.800", "0.500", "0.250", "0.600", "0.100"]
    assert lines[3].split() == ["crop", "val", "(select)", "10", "0.800", "0.500", "0.250", "0.600", "0.100"]
    assert lines[4].split() == ["crop", "val", "human", "0", "-", "-", "-", "-", "-"]
    assert "crop_geom  no result.json" in table and "full" in table


# The tiny runs need torch, torchvision and Pillow (the GPU environment). They run in a subprocess, through
# the CLI: torch never enters this interpreter, whose other tests check that it was not imported.
TRAINING_STACK = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "PIL"))


@pytest.mark.skipif(not TRAINING_STACK, reason="torch/torchvision/Pillow not installed (GPU environment only)")
@pytest.mark.parametrize("variant", VARIANTS)
def test_tiny_cpu_run_of_each_variant(tmp_path, capsys, variant):
    dataset, frozen, dataset_sha = _experiment_fixture(tmp_path, capsys)
    output = tmp_path / "runs" / "tiny" / "experiment-b" / variant
    # Random weights (no download), CPU only, no wandb; few threads, so a busy server does not slow it down.
    environment = dict(os.environ, WANDB_MODE="disabled", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
                       PYTHONPATH=os.pathsep.join(filter(None, (str(SRC), os.environ.get("PYTHONPATH")))))
    process = subprocess.run(
        [sys.executable, "-m", "nnslr_tools.cli", "experiment-b", "--variant", variant,
         "--data-root", str(tmp_path), "--output", str(output), "--device", "cpu", "--epochs", "1",
         "--batch-size", "2", "--crop-size", "64", "--num-workers", "0", "--no-amp"],
        capture_output=True, text=True, env=environment, timeout=600,
    )
    assert process.returncode == 0, process.stderr[-3000:]

    assert re.search(rf"^experiment_b_{variant} epoch 1/1 train_loss \S+ val_loss \S+ .* \d+s$", process.stdout, re.M)
    assert (output / f"{variant}-best.pt").is_file()
    saved = json.loads((output / "result.json").read_text())
    assert saved["variant"] == variant and saved["best_epoch"] == 1
    assert saved["dataset_sha256"] == dataset_sha
    assert saved["split_sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert saved["counts"]["boxes_by_partition"] == {"train": 8, "validation": 3, "test": 3}
    assert saved["torch_version"] and saved["initialization"] == "random"
    for partition in ("validation", "test"):
        metrics = saved["metrics"][partition]
        assert metrics["box_count"] == 3
        assert {"accuracy", "per_class", "confusion", "expected_calibration_error"} <= set(metrics["reading"])
        assert {"confusion", "per_class", "other_road", "abstain_rate", "macro_f1",
                "expected_calibration_error"} <= set(metrics["ownership"])
        assert {"precision", "recall"} <= set(metrics["ownership"]["other_road"])
        human = metrics["human_reviewed"]
        assert human["label_basis"] == "human_review" and {"reading", "ownership"} <= set(human)
    assert saved["metrics"]["validation"]["human_reviewed"]["box_count"] == 2
    assert saved["metrics"]["test"]["ownership"]["sample_count"] == 3
    assert len((output / "samples.jsonl").read_text().splitlines()) == 6
    # Prepare's "ours" proposer loads these weights as the decider: one answer for each box of a frame.
    script = ("import sys, torch; from PIL import Image; from nnslr_tools.context_experiment import Decider; "
              "decider = Decider(sys.argv[1], torch.device('cpu')); "
              "print(json.dumps(decider.decide(Image.new('RGB', (320, 180)), [[10, 10, 40, 40], [200, 50, 260, 120]], 25.0)))")
    decided = subprocess.run([sys.executable, "-c", "import json; " + script, str(output / f"{variant}-best.pt")],
                             capture_output=True, text=True, env=environment, timeout=300)
    assert decided.returncode == 0, decided.stderr[-3000:]
    answers = json.loads(decided.stdout)
    assert len(answers) == 2 and all(road in OWNERSHIP_CLASSES and 0 < share <= 1 for road, share in answers)
    # And exports to ONNX for the package of the networks, the same answers as PyTorch for one box and for several.
    if importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"):
        script = ("import json, sys; from pathlib import Path; from nnslr_tools.exporting import export_decider_onnx; "
                  "print(json.dumps(export_decider_onnx(Path(sys.argv[1]), Path(sys.argv[2]))['verification']))")
        exported = subprocess.run([sys.executable, "-c", script, str(output / f"{variant}-best.pt"),
                                   str(tmp_path / f"decider-{variant}.onnx")],
                                  capture_output=True, text=True, env=environment, timeout=300)
        assert exported.returncode == 0, exported.stderr[-3000:]
        assert json.loads(exported.stdout.strip().splitlines()[-1])["passed"] is True
        contract = json.loads((tmp_path / f"decider-{variant}.onnx.contract.json").read_text())
        assert contract["task"] == "decider" and contract["inputs"]["crops"]["shape"][0] == "boxes"
        assert ("scene" in contract["inputs"]) == (variant == "full")
    assert "torch" not in sys.modules


def _moved_split(root: Path, dataset: Path, frozen: Path, moved: dict[str, str]) -> Path:
    """The fixture's split with the routes of ``moved`` in another partition, kept under its sha256 as build-splits
    does."""
    payload = json.loads(frozen.read_text())
    by_route = {row["annotation_id"]: row["route_id"] for row in load_jsonl(dataset)}
    text = json.dumps({**payload, "assignments": {row: moved.get(by_route[row], name)
                                                  for row, name in payload["assignments"].items()}},
                      sort_keys=True, indent=2) + "\n"
    path = root / "splits" / (hashlib.sha256(text.encode()).hexdigest() + ".json")
    path.write_text(text, encoding="utf-8")
    return path


def _crop_run(root: Path, name: str, splits: Path, *more: str, epochs: int = 1, before: str = "") -> tuple[dict, str]:
    """One tiny CPU training of the crop variant into runs/<name>, through the CLI in a process of its own:
    result.json and what it printed. ``before``: Python run in that process before the CLI, with start_lineage and
    context_experiment imported (the floor of the clean rows is the module's: a test with four validation rows
    lowers it there, or scripts the scores of its epochs)."""
    output = root / "runs" / name / "experiment-b" / "crop"
    environment = dict(os.environ, WANDB_MODE="disabled", OMP_NUM_THREADS="2",
                       PYTHONPATH=os.pathsep.join(filter(None, (str(SRC), os.environ.get("PYTHONPATH")))))
    command = ["-m", "nnslr_tools.cli"] if not before else [
        "-c", "import sys\nfrom nnslr_tools import cli, context_experiment, start_lineage\n" + before
              + "\nsys.exit(cli.main(sys.argv[1:]))"]
    process = subprocess.run(
        [sys.executable, *command, "experiment-b", "--variant", "crop", "--data-root", str(root),
         "--splits", str(splits), "--output", str(output), "--device", "cpu", "--epochs", str(epochs), "--batch-size", "2",
         "--crop-size", "64", "--num-workers", "0", "--no-amp", *more],
        capture_output=True, text=True, env=environment, timeout=600)
    assert process.returncode == 0, process.stderr[-3000:]
    return json.loads((output / "result.json").read_text()), process.stdout


@pytest.mark.skipif(not TRAINING_STACK, reason="torch/torchvision/Pillow not installed (GPU environment only)")
def test_tiny_cpu_run_from_a_start_measures_the_clean_validation_rows(tmp_path, capsys):
    """#104: a variant that goes on from a run of another split, in which a validation row of today was a train row."""
    dataset, frozen, _ = _experiment_fixture(tmp_path, capsys)
    earlier = _moved_split(tmp_path, dataset, frozen, {BUFFER: "train"})
    today = _moved_split(tmp_path, dataset, frozen, {BUFFER: "validation"})

    def run(name: str, splits: Path, *more: str, floor: int | None = None) -> tuple[dict, str]:
        return _crop_run(tmp_path, name, splits, *more,
                         before="" if floor is None else f"start_lineage.CLEAN_FLOOR = {floor}")

    first, _ = run("first", earlier)
    assert "start_lineage" not in first and "validation_clean" not in first
    start = ("--init-from", str(tmp_path / "runs" / "first" / "experiment-b" / "crop" / "crop-best.pt"))
    second, printed = run("second", today, *start)
    record = second["start_lineage"]
    assert [(entry["run"], entry["stage"], entry["split_sha256"]) for entry in record["ancestors"]] == [
        ("first", "experiment-b/crop", earlier.stem)]
    assert (record["complete"], record["another_split"], record["counted"]) == (True, True, "rows")
    # The buffer route's box was a train row of the first run: one of today's four validation rows.
    assert record["validation"] == {"all": 4, "trained_on": 1, "clean": 3} and record["test"] == {"all": 3, "trained_on": 0}
    assert (record["best_epoch_on"], record["clean_cases"]) == ("validation", 3)
    assert record["why"] == "the 3 clean rows hold 3 boxes to read, fewer than the 200 a choice needs"
    assert ("experiment_b_crop 1 of the 4 validation rows were train rows of a network it descends from (1 before it): "
            "the 3 clean rows hold 3 boxes to read, fewer than the 200 a choice needs, so the best epoch is chosen on "
            "the whole validation; both are measured") in printed
    line = re.search(r"^experiment_b_crop epoch 1/1 .*$", printed, re.M).group(0)
    assert " clean_val_accuracy " in line and " clean_val_selection_score " in line and " val_chosen " not in line
    assert re.search(r" val_selection_score \S+ clean_val_accuracy \S+ .* \d+s$", line)
    assert second["metrics"]["validation"]["box_count"] == 4 and second["validation_clean"]["box_count"] == 3
    assert set(second["validation_clean"]) == set(second["metrics"]["validation"]) - {"loss"}
    assert {"clean_validation_accuracy", "clean_validation_other_road_recall", "clean_validation_other_road_f1",
            "clean_validation_selection_score"} <= set(second["history"][0])
    assert second["best_validation_selection_score"] == second["history"][0]["validation_selection_score"]
    assert set(second) == set(first) | {"start_lineage", "validation_clean"}
    # With enough clean rows (the floor lowered) they choose the epoch, and the line says by what.
    third, printed = run("third", today, *start, floor=3)
    assert third["start_lineage"]["best_epoch_on"] == "clean_validation" and third["start_lineage"]["why"] is None
    line = re.search(r"^experiment_b_crop epoch 1/1 .*$", printed, re.M).group(0)
    chosen = float(re.search(r" val_chosen (\S+) ", line).group(1))
    assert chosen == pytest.approx(third["history"][0]["clean_validation_selection_score"], abs=1e-4)
    # best_validation_selection_score stays the best epoch's score on the whole validation.
    assert third["best_validation_selection_score"] == third["history"][0]["validation_selection_score"]
    # On the split of its start nothing is left out: result.json gains start_lineage alone.
    same, printed = run("same", earlier, *start)
    assert set(same) == set(first) | {"start_lineage"} and set(same["history"][0]) == set(first["history"][0])
    assert same["start_lineage"]["validation"] == {"all": 3, "trained_on": 0, "clean": 3}
    assert "clean_val" not in printed and "descends from" not in printed
    assert "torch" not in sys.modules


# What _crop_run runs before the CLI: epoch_scores as it is, with the two halves of the score replaced epoch by epoch
# on the whole validation and on its clean rows (the tiny network scores the same at every epoch on four rows).
SCRIPTED_SCORES = """
start_lineage.CLEAN_FLOOR = %(floor)d
real, epochs = context_experiment.epoch_scores, []
def scripted(samples, classes, clean, on_clean):
    validation, cleaned, _ = real(samples, classes, clean, on_clean)
    epochs.append(1)
    for metrics, scores in ((validation, %(whole)r), (cleaned, %(clean)r)):
        if metrics is not None:
            metrics["reading"]["accuracy"] = metrics["ownership"]["other_road"]["f1"] = scores[len(epochs) - 1]
    return validation, cleaned, context_experiment.selection_score(cleaned if on_clean and cleaned is not None else validation)
context_experiment.epoch_scores = scripted
"""


@pytest.mark.skipif(not TRAINING_STACK, reason="torch/torchvision/Pillow not installed (GPU environment only)")
def test_tiny_cpu_run_keeps_the_epoch_its_clean_validation_rows_choose(tmp_path, capsys):
    """#104, the loop of train_experiment_b itself: the whole validation is best at epoch 1 (the rows an ancestor
    trained on flatter it), the clean rows at epoch 2. The weights kept and best_epoch are the clean rows', and
    best_validation_selection_score stays that epoch's score on the whole validation."""
    dataset, frozen, _ = _experiment_fixture(tmp_path, capsys)
    earlier = _moved_split(tmp_path, dataset, frozen, {BUFFER: "train"})
    today = _moved_split(tmp_path, dataset, frozen, {BUFFER: "validation"})
    scores = {"whole": [0.9, 0.5], "clean": [0.2, 0.7]}
    saved_epoch = ("import sys, torch; saved = torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
                   "print(saved['epoch'], saved['validation_selection_score'])")

    def kept(name: str) -> str:
        weights = tmp_path / "runs" / name / "experiment-b" / "crop" / "crop-best.pt"
        return subprocess.run([sys.executable, "-c", saved_epoch, str(weights)], capture_output=True, text=True,
                              check=True, timeout=120).stdout.strip()

    _crop_run(tmp_path, "first", earlier)
    start = ("--init-from", str(tmp_path / "runs" / "first" / "experiment-b" / "crop" / "crop-best.pt"))
    # One of today's four validation rows was a train row of the first run; the other three hold the validation's
    # only sign of another road. From the floor on they choose: epoch 2.
    second, printed = _crop_run(tmp_path, "second", today, *start, epochs=2,
                                before=SCRIPTED_SCORES % {**scores, "floor": 3})
    assert second["start_lineage"]["best_epoch_on"] == "clean_validation" and second["start_lineage"]["why"] is None
    assert second["best_epoch"] == 2 and kept("second") == "2 0.5"
    assert second["best_validation_selection_score"] == 0.5  # the kept epoch's score on the whole validation
    lines = re.findall(r"^experiment_b_crop epoch \d/2 .*$", printed, re.M)
    assert " val_selection_score 0.9000 " in lines[0] and " clean_val_selection_score 0.2000 val_chosen 0.2000 " in lines[0]
    assert " val_selection_score 0.5000 " in lines[1] and " clean_val_selection_score 0.7000 val_chosen 0.7000 " in lines[1]
    assert [(epoch["validation_selection_score"], epoch["clean_validation_selection_score"])
            for epoch in second["history"]] == [(0.9, 0.2), (0.5, 0.7)]
    # A start that trained on the validation's sign of another road: today's validation route was its train, and
    # the one clean row (the buffer route's) has no such sign. Rows enough at a floor of 1, but none of the one sign
    # the other half of the score counts: the whole validation chooses, epoch 1.
    swapped = _moved_split(tmp_path, dataset, frozen, {VALIDATION: "train", BUFFER: "validation"})
    _crop_run(tmp_path, "swapped", swapped)
    start = ("--init-from", str(tmp_path / "runs" / "swapped" / "experiment-b" / "crop" / "crop-best.pt"))
    guarded, printed = _crop_run(tmp_path, "guarded", today, *start, epochs=2,
                                 before=SCRIPTED_SCORES % {**scores, "floor": 1})
    record = guarded["start_lineage"]
    assert record["validation"] == {"all": 4, "trained_on": 3, "clean": 1} and record["clean_cases"] == 1
    assert record["best_epoch_on"] == "validation"
    assert record["why"] == "only 0 of the 1 signs of another road are on the clean rows"
    assert ("only 0 of the 1 signs of another road are on the clean rows, so the best epoch is chosen on the whole "
            "validation") in printed
    assert guarded["best_epoch"] == 1 and kept("guarded") == "1 0.9" and guarded["best_validation_selection_score"] == 0.9
    assert " clean_val_selection_score " in printed and " val_chosen " not in printed
    assert guarded["validation_clean"]["box_count"] == 1
    assert "torch" not in sys.modules
