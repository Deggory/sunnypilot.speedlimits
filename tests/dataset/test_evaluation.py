from __future__ import annotations

import json
import sys

import pytest

from nnslr_tools.annotations import load_jsonl
from nnslr_tools.evaluation import box_iou, centre_match, greedy_matching, match_detections, score_frame
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


# -- The lenient match ("centre"): fair to the far, small signs ------------------------------------------------
SMALL_SIGN = [100.0, 50.0, 110.0, 60.0]  # a far sign, 10 px
TIGHT_BOX = {"box": [102.0, 52.0, 108.0, 58.0], "score": 0.9}  # 2 px tighter on each side: IoU 0.36


def test_a_box_2_px_tighter_on_a_10_px_sign_is_a_strict_miss_and_a_lenient_hit() -> None:
    assert box_iou(TIGHT_BOX["box"], SMALL_SIGN) == pytest.approx(0.36)
    strict = score_frame([SMALL_SIGN], [TIGHT_BOX], score_threshold=0.0, iou_threshold=0.5)
    assert (strict["true_positives"], strict["false_positives"], strict["false_negatives"]) == (0, 1, 1)
    lenient = score_frame([SMALL_SIGN], [TIGHT_BOX], score_threshold=0.0, iou_threshold=0.5, match="centre")
    assert (lenient["true_positives"], lenient["false_positives"], lenient["false_negatives"]) == (1, 0, 0)
    assert lenient["matches"][0]["iou"] == pytest.approx(0.36) and lenient["found_truth_indexes"] == [0]


def test_a_huge_box_around_a_small_sign_is_not_a_lenient_hit() -> None:
    huge = {"box": [80.0, 30.0, 130.0, 80.0], "score": 0.9}  # centred on the sign, 5 times its size
    assert centre_match(huge["box"], SMALL_SIGN) is False
    lenient = match_detections([huge], [SMALL_SIGN], match="centre")
    assert (lenient["true_positives"], lenient["false_positives"], lenient["false_negatives"]) == (0, 1, 1)
    # Off the sign's centre, or too thin, neither; twice its size, on the edge of the rule, yes.
    assert centre_match([111.0, 52.0, 117.0, 58.0], SMALL_SIGN) is False
    assert centre_match([104.0, 52.0, 106.0, 58.0], SMALL_SIGN) is False
    assert centre_match([95.0, 45.0, 115.0, 65.0], SMALL_SIGN) is True


def test_the_lenient_match_stays_one_to_one_and_score_ordered() -> None:
    second = {"box": [103.0, 53.0, 109.0, 59.0], "score": 0.5}
    result = match_detections([second, TIGHT_BOX], [SMALL_SIGN], match="centre")
    assert (result["true_positives"], result["false_positives"]) == (1, 1)
    assert result["matches"][0]["score"] == 0.9 and result["false_positive_indexes"] == [0]


def test_the_defaults_stay_the_strict_iou_match() -> None:
    truth = [SMALL_SIGN, [0.0, 0.0, 40.0, 40.0]]
    predicted = [TIGHT_BOX, {"box": [2.0, 2.0, 38.0, 38.0], "score": 0.8},
                 {"box": [80.0, 30.0, 130.0, 80.0], "score": 0.7}]
    assert greedy_matching(predicted, truth) == greedy_matching(predicted, truth, iou_threshold=0.5, match="iou")
    assert match_detections(predicted, truth) == match_detections(predicted, truth, match="iou")
    assert [index for _, index, _ in greedy_matching(predicted, truth)] == [-1, 1, -1]
    assert [index for _, index, _ in greedy_matching(predicted, truth, match="centre")] == [0, 1, -1]
    with pytest.raises(ValueError, match="invalid_match"):
        match_detections(predicted, truth, match="near")
