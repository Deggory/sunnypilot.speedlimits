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


def test_one_line_per_sign_becomes_proposals():
    answer = "512 431 530 462 limit 80\n100 200 110 220 other -\n300 300 320 330 end -\nnoise line"
    found = parse_answer(answer, 1000, 1000)
    assert [(item['family'], item['value_kph']) for item in found] == [
        ('maximum_speed', 80), ('other_sign', None), ('cancellation', None)]
    assert found[0]['bbox_xyxy'] == [512.0, 431.0, 530.0, 462.0]


def test_none_gives_no_proposals_and_an_unknown_kind_is_another_sign():
    assert parse_answer("none", 1344, 760) == []
    assert parse_answer("10 10 20 20 triangle 30", 1000, 1000)[0]['family'] == 'other_sign'
    # A speed on another sign is not proposed: only limits, zones and displays carry one.
    assert parse_answer("10 10 20 20 other 30", 1000, 1000)[0]['value_kph'] is None


def test_an_answer_in_a_loop_keeps_only_its_speed_signs():
    # Qwen3-VL sometimes writes a box, then the one just under it, and so on past the bottom of the picture.
    column = '\n'.join(f'943 {y} 999 {y + 22} other' for y in range(170, 700, 22))
    found = parse_answer('118 257 162 437 limit 30\n' + column, 1000, 1000)
    assert [(item['family'], item['value_kph']) for item in found] == [('maximum_speed', 30)]
    # The same beside each other, along a row.
    row = '\n'.join(f'{x} 462 {x + 14} 500 other' for x in range(234, 600, 14))
    assert parse_answer(row, 1000, 1000) == []
    # The same line over and over.
    assert parse_answer('520 594 567 635 other\n' * 20, 1000, 1000) == []


def test_a_line_written_twice_counts_once_and_signs_side_by_side_stay():
    twice = '412 653 452 723 other\n523 487 618 557 other\n' * 2
    assert len(parse_answer(twice, 1000, 1000)) == 2
    # Two plates one under the other (a limit and its panel) are not a loop.
    assert len(parse_answer('500 300 540 360 limit 50\n500 360 540 380 other', 1000, 1000)) == 2
