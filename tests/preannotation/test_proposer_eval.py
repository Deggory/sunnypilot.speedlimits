"""The proposers of Prepare measured against the kept boxes (nnslr_tools.proposer_eval.tally)."""
from nnslr_tools.proposer_eval import tally


def _kept(x, family, value=None, basis="human_review"):
    return {"bbox_xyxy": [x, 100, x + 16, 116], "sign_family": family, "value_kph": value, "label_basis": basis}


def _proposal(x, family, value=None, score=0.9):
    return {"bbox_xyxy": [x + 1, 100, x + 17, 116], "family": family, "value_kph": value, "score": score}


def test_found_read_right_false_and_skipped_families():
    frames = [{"label_basis": "human_review", "kept": [_kept(0, "maximum_speed", 80), _kept(100, "maximum_speed", 60),
                                                       _kept(200, "other_sign"), _kept(300, "road_marking", 30),
                                                       _kept(400, "cancellation")]},
              {"label_basis": "ai_review", "kept": [_kept(0, "maximum_speed", 50, "ai_review")]}]
    rows = [{"detections": [_proposal(0, "maximum_speed", 80), _proposal(100, "maximum_speed", 80),
                            _proposal(200, "other_sign", score=None), _proposal(300, "road_marking_candidate", 30),
                            _proposal(600, "maximum_speed", 50)]},
            {"detections": []}]
    report = tally(frames, rows)
    # Speed signs: 80 found and right, 60 found as 80 (wrong), the end of limit missed, the 50 of frame 2 missed.
    assert {key: report["speed_signs"][key] for key in ("truth", "found", "read_right")} == {"truth": 4, "found": 2, "read_right": 1}
    assert report["speed_signs_person"]["truth"] == 3 and report["speed_signs_person"]["read_right"] == 1
    # Every sign: the other sign counts too; the painted 30 is left out on both sides.
    assert report["every_sign"] == {"truth": 5, "found": 3, "found_share": 0.6}
    assert report["proposals"] == 4 and report["false_proposals"] == 1 and report["false_per_frame"] == 0.5
    assert [item["truth"] for item in report["by_width"]] == [0, 4, 0, 0]
