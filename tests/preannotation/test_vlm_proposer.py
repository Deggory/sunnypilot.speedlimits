"""The answers of Qwen3-VL as proposals of the review page (nnslr preannotate --proposer qwen)."""
from nnslr_tools._vlm_proposer import parse_answer


def test_answer_in_fences_becomes_proposals_in_frame_pixels():
    answer = ('```json\n[{"bbox_2d": [500, 400, 520, 440], "kind": "speed_limit", "speed": 80},\n'
              ' {"bbox_2d": [100, 100, 110, 120], "kind": "give_way", "speed": null},\n'
              ' {"bbox_2d": [10, 10, 10, 10], "kind": "zone", "speed": 30},\n'
              ' {"bbox_2d": [0, 0, 1200, 50], "kind": "end_of_limit", "speed": 999}]\n```')
    found = parse_answer(answer, 1344, 760)
    assert [item["family"] for item in found] == ["maximum_speed", "other_sign", "cancellation"]
    assert found[0]["bbox_xyxy"] == [672.0, 304.0, 698.9, 334.4] and found[0]["value_kph"] == 80
    assert found[0]["value_state"] == "proposed" and found[0]["score"] is None
    assert found[1]["value_kph"] is None and found[1]["model_label"] == "qwen3-vl:other_sign"
    # Clipped to the frame; a number on an end of limit is not a value.
    assert found[2]["bbox_xyxy"][2] == 1344.0 and found[2]["value_kph"] is None


def test_no_list_or_bad_json_gives_no_proposals():
    assert parse_answer("I see no traffic sign.", 1344, 760) == []
    assert parse_answer("[{bbox_2d: oops}]", 1344, 760) == []
    assert parse_answer("[]", 1344, 760) == []
