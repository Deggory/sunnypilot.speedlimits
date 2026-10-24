from __future__ import annotations

import json
import sys

import pytest

from nnslr_tools.annotations import load_jsonl
from nnslr_tools.evaluation import box_iou, match_detections
from test_training import _training_fixture
from test_annotations import invoke


def test_box_iou_exact_disjoint_and_partial() -> None:
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert box_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)


def test_match_detections_is_one_to_one_and_score_ordered() -> None:
    truth = [[0, 0, 10, 10], [20, 20, 30, 30]]
    predicted = [
        {"box": [0, 0, 10, 10], "score": .95},
        {"box": [1, 1, 9, 9], "score": .8},
        {"box": [20, 20, 30, 30], "score": .7},
    ]

    result = match_detections(predicted, truth, iou_threshold=.5)

    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 0


@pytest.mark.parametrize("task", ["reader", "detector"])
def test_evaluate_dry_run_is_cpu_only(tmp_path, capsys, task) -> None:
    dataset, _, _ = _training_fixture(tmp_path, capsys)
    assert "torch" not in sys.modules

    code, out = invoke(
        capsys,
        "evaluate",
        str(dataset),
        "--task",
        task,
        "--partition",
        "test",
        "--data-root",
        str(tmp_path),
        "--dry-run",
    )

    assert code == 0, out.err + out.out
    report = json.loads(out.out)
    assert report["task"] == task
    assert report["partition"] == "test"
    assert report["sample_count"] > 0
    assert report["checkpoint_required_for_execution"] is True
    assert "torch" not in sys.modules


def test_evaluate_without_checkpoint_rejected_after_plan(tmp_path, capsys) -> None:
    dataset, _, _ = _training_fixture(tmp_path, capsys)

    code, out = invoke(
        capsys,
        "evaluate",
        str(dataset),
        "--data-root",
        str(tmp_path),
    )

    assert code == 2
    assert "--checkpoint is required" in out.err


def test_reader_metrics_on_human_labels_are_reported_apart() -> None:
    from nnslr_tools.evaluation import human_reviewed_metrics, label_basis_counts, reader_metrics

    samples = [
        {"truth": "maximum_speed:30", "predicted": "maximum_speed:30", "correct": True, "label_basis": "human_review"},
        {"truth": "maximum_speed:50", "predicted": "maximum_speed:30", "correct": False, "label_basis": "human_review"},
        {"truth": "maximum_speed:50", "predicted": "maximum_speed:50", "correct": True, "label_basis": "ai_review"},
        {"truth": "maximum_speed:50", "predicted": "maximum_speed:50", "correct": True, "label_basis": "ai_review"},
    ]
    classes = ["maximum_speed:30", "maximum_speed:50"]
    everything = reader_metrics(samples, classes)
    assert everything["sample_count"] == 4 and everything["accuracy"] == .75
    assert everything["per_class"]["maximum_speed:50"] == {"support": 3, "correct": 2, "accuracy": 2 / 3}
    assert everything["confusion"]["maximum_speed:50"] == {"maximum_speed:30": 1, "maximum_speed:50": 2}
    human = human_reviewed_metrics(samples, lambda items: reader_metrics(items, classes))
    assert human["label_basis"] == "human_review" and human["sample_count"] == 2 and human["accuracy"] == .5
    assert label_basis_counts(samples) == {"ai_review": 2, "human_review": 2}
    assert human_reviewed_metrics(samples[2:], lambda items: reader_metrics(items, classes)) is None


def test_detector_metrics_on_human_labels_are_reported_apart() -> None:
    from nnslr_tools.evaluation import detector_metrics, human_reviewed_metrics

    images = [
        {"true_positives": 1, "false_positives": 0, "false_negatives": 1, "label_basis": "human_review"},
        {"true_positives": 2, "false_positives": 2, "false_negatives": 0, "label_basis": "ai_review"},
        {"true_positives": 0, "false_positives": 1, "false_negatives": 0, "label_basis": None},
    ]
    everything = detector_metrics(images)
    assert (everything["image_count"], everything["truth_box_count"], everything["false_positives"]) == (3, 4, 3)
    assert everything["precision_iou"] == pytest.approx(.5) and everything["recall_iou"] == pytest.approx(.75)
    assert everything["false_positives_per_image"] == 1.0
    human = human_reviewed_metrics(images, detector_metrics)
    assert (human["image_count"], human["precision_iou"], human["recall_iou"]) == (1, 1.0, .5)


@pytest.mark.parametrize("task", ["reader", "detector"])
def test_evaluate_dry_run_counts_human_reviewed_samples(tmp_path, capsys, task) -> None:
    # Every frame is an AI review except frame 1 (route 00000002), a person's review.
    dataset, frozen, _ = _training_fixture(tmp_path, capsys, ai=True)
    [human] = [r for r in load_jsonl(dataset) if r["provenance"]["label_basis"] == "human_review"]
    partition = json.loads(frozen.read_text())["assignments"][human["annotation_id"]]
    code, out = invoke(capsys, "evaluate", str(dataset), "--task", task, "--partition", partition,
                       "--data-root", str(tmp_path), "--dry-run")
    assert code == 0, out.err + out.out
    report = json.loads(out.out)
    assert report["human_reviewed_sample_count"] == report["counts_by_label_basis"]["human_review"] == 1
    assert sum(report["counts_by_label_basis"].values()) == report["sample_count"]
